"""Evaluate an external retriever or reranker on BEIR (nDCG@10), off the shelf or fine-tuned (--ft).

  python -m competitors.evaluate_external_beir --model bge-rerank --datasets scifact,nfcorpus,arguana --max-doc-words 10000 --window-words 350"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, torch
# the repository root, so `config` and the packages import from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from excise.sources import load_source, id2idx
from excise.metrics import ndcg_at_k_graded, recall_at_k
from competitors.evaluate_external import (MODELS, dense_index, dense_order, DEV, load_dense_encoder,
                              chunk_doc, build_passages, dense_index_windowed, dense_order_windowed)
from excise.evaluate_beir import load_qrels, load_queries_with_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--datasets", default="scifact,nfcorpus,arguana")
    ap.add_argument("--split", default="test")
    ap.add_argument("--first-k", type=int, default=100)
    ap.add_argument("--ft", default="", help='fine-tuned checkpoint (dense: replaces encoder -> re-indexed here')
    ap.add_argument("--window-words", type=int, default=0, help='0=truncate at 512 (published)')
    ap.add_argument("--max-doc-words", type=int, default=10000)
    a = ap.parse_args()
    m = MODELS[a.model]
    tag = "ft" if a.ft else "off"

    dense_src = a.ft if (a.ft and m["mode"] == "dense") else (m["repo"] if m["mode"] == "dense" else m["first"])
    print(f"[strong_noharm] model={a.model} tag={tag} ft={'yes:'+a.ft if a.ft else 'no (off-the-shelf)'} "
          f"encoder={dense_src}", flush=True)
    enc = load_dense_encoder(dense_src, m)
    ce_tok = ce_model = None
    if m["mode"] == "rerank":
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        ce_src = a.ft if a.ft else m["repo"]
        ce_tok = AutoTokenizer.from_pretrained(ce_src)
        ce_model = AutoModelForSequenceClassification.from_pretrained(ce_src).to(DEV).eval()

    @torch.no_grad()
    def ce_scores(q, docs, idxs):
        out = []
        for i in range(0, len(idxs), 64):
            b = idxs[i:i + 64]
            enc_in = ce_tok([q] * len(b), [docs[j] or "" for j in b], padding=True, truncation=True,
                            max_length=512, return_tensors="pt").to(DEV)
            out.extend(ce_model(**enc_in).logits.squeeze(-1).float().cpu().tolist())
        return out

    @torch.no_grad()
    def ce_scores_windowed(q, docs, idxs):
        """Every passage of each candidate scored; the document takes its MAX. Flattened so batches stay full."""
        flat, own = [], []
        for k, j in enumerate(idxs):
            for p in chunk_doc(docs[j], a.window_words, a.max_doc_words):
                flat.append(p); own.append(k)
        sc = []
        for i in range(0, len(flat), 64):
            enc_in = ce_tok([q] * len(flat[i:i + 64]), flat[i:i + 64], padding=True, truncation=True,
                            max_length=512, return_tensors="pt").to(DEV)
            sc.extend(ce_model(**enc_in).logits.squeeze(-1).float().cpu().tolist())
        best = [float("-inf")] * len(idxs)
        for k, s in zip(own, sc):
            if s > best[k]: best[k] = s
        return best

    try:
        import pytrec_eval
    except ImportError:
        pytrec_eval = None

    for ds in a.datasets.split(","):
        qrels = load_qrels(ds, a.split)
        ids, docs = load_source(ds); i2 = id2idx(ids)
        qs = [(qid, qt) for qid, qt in load_queries_with_ids(ds) if qid in qrels]
        if not qs:
            print(f"NOHARM model={a.model} tag={tag} ds={ds} SKIP (no queries with qrels)", flush=True); continue
        if a.window_words > 0:
            ptexts, owner = build_passages(docs, a.window_words, a.max_doc_words)
            print(f"[strong_noharm] {ds}: windowed {len(docs)} docs -> {len(ptexts)} passages "
                  f"({len(ptexts)/max(len(docs),1):.2f} per doc)", flush=True)
            dmat = dense_index_windowed(enc, ptexts, m["dp"], bs=m.get("ibs", 256))
        else:
            dmat = dense_index(enc, docs, m["dp"], bs=m.get("ibs", 256))
        first = ((lambda qt: dense_order_windowed(enc, dmat, owner, len(docs), m["qp"], qt))
                 if a.window_words > 0 else (lambda qt: dense_order(enc, dmat, m["qp"], qt)))
        ce_fn = ce_scores_windowed if a.window_words > 0 else ce_scores

        def rank(qt):
            full = first(qt)
            if m["mode"] == "dense":
                return full
            top = full[:a.first_k]
            s = ce_fn(qt, docs, top)
            return [top[i] for i in np.argsort(s)[::-1]] + full[a.first_k:]

        nd = 0.0; run = {}; skipped = 0
        for k, (qid, qt) in enumerate(qs):
            rel = {i2[d]: g for d, g in qrels[qid].items() if d in i2}
            if not rel:
                skipped += 1; continue
            order = rank(qt)
            nd += ndcg_at_k_graded(order, rel, 10)
            run[qid] = {ids[d]: float(len(order) - r) for r, d in enumerate(order)}
            if k % 200 == 0:
                print(f"  {ds} {k}/{len(qs)}", flush=True)
        n = len(run)
        ref = ""
        if pytrec_eval and n:
            ev = pytrec_eval.RelevanceEvaluator(
                {q: {d: int(g) for d, g in qrels[q].items()} for q in run}, {"ndcg_cut_10"})
            res = ev.evaluate(run)
            p_nd = sum(v["ndcg_cut_10"] for v in res.values()) / max(len(res), 1)
            ref = f"  [pytrec] nDCG@10={p_nd:.4f}"
        print(f"NOHARM model={a.model} tag={tag} ds={ds} n={n} nDCG@10={nd/max(n,1):.4f}{ref}"
              + (f"  skipped={skipped}" if skipped else ""), flush=True)
        del dmat; torch.cuda.empty_cache()
    print(f"STRONG_NOHARM_DONE model={a.model} tag={tag}", flush=True)


if __name__ == "__main__":
    main()
