"""Evaluate an external retriever or reranker on X-BENCH with the metrics and output of excise.evaluate.
Rerankers rescore the top 100 of a BGE-large first stage.

  python -m competitors.evaluate_external --model bge-rerank --records 'data/benchmark/test/esgenius/*.jsonl' --max-doc-words 10000 --window-words 350
  python -m competitors.evaluate_external --model bge-rerank --ft model/checkpoints/bge-rerank ...   # the fine-tuned model"""
from __future__ import annotations
import os
import argparse, json, os, sys
from collections import defaultdict
import numpy as np, torch
# the repository root, so `config` and the packages import from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from excise import config as C
from excise.sources import canonical, load_source, id2idx
import excise.evaluate as EB

MODELS = {
    "bge-large": dict(mode="dense", repo="BAAI/bge-large-en-v1.5",
                      qp="Represent this sentence for searching relevant passages: ", dp=""),
    "e5-large": dict(mode="dense", repo="intfloat/e5-large-v2", qp="query: ", dp="passage: "),
    "bge-rerank": dict(mode="rerank", repo="BAAI/bge-reranker-v2-m3", first="BAAI/bge-large-en-v1.5",
                       qp="Represent this sentence for searching relevant passages: ", dp=""),
    "bge-rerank-large": dict(mode="rerank", repo="BAAI/bge-reranker-large", first="BAAI/bge-large-en-v1.5",
                             qp="Represent this sentence for searching relevant passages: ", dp=""),
    "bge-rerank-base": dict(mode="rerank", repo="BAAI/bge-reranker-base", first="BAAI/bge-large-en-v1.5",
                            qp="Represent this sentence for searching relevant passages: ", dp=""),
    "msmarco-minilm": dict(mode="rerank", repo="cross-encoder/ms-marco-MiniLM-L-6-v2", first="BAAI/bge-large-en-v1.5",
                           qp="Represent this sentence for searching relevant passages: ", dp=""),
    "promptriever": dict(mode="dense", repo="samaya-ai/promptriever-mistral-v0.1-7b-v1",
                         peft_base="mistralai/Mistral-7B-v0.1",
                         qp="query: ", dp="passage: ", ibs=32, max_seq_length=512),
    "gte-qwen2": dict(mode="dense", grad_ckpt=True, repo="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
                      qp="Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
                         "Query: ", dp="", trust_remote_code=True, max_seq_length=512, ibs=64),
}
DEV = "cuda:0"


class PromptrieverEncoder:
    """Promptriever / RepLLaMA-style bi-encoder: a causal LM read as a retriever."""

    def __init__(self, repo, base, max_seq_length=512, dev=None):
        from transformers import AutoTokenizer, AutoModel
        from peft import PeftModel
        self.tok = AutoTokenizer.from_pretrained(repo)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "right"                     # RoPE: pads must not shift real-token positions
        base_m = AutoModel.from_pretrained(base, torch_dtype=torch.float16)
        self.dev = dev or DEV
        self.model = PeftModel.from_pretrained(base_m, repo).to(self.dev).eval()
        self.max_seq_length = max_seq_length

    @torch.no_grad()
    def encode(self, texts, batch_size=32, convert_to_tensor=True, normalize_embeddings=True,
               show_progress_bar=False, device=None, **kw):
        out = []
        for i in range(0, len(texts), batch_size):
            b = [t + "</s>" for t in texts[i:i + batch_size]]
            enc = self.tok(b, padding=True, truncation=True, max_length=self.max_seq_length,
                           return_tensors="pt").to(self.dev)
            hs = self.model(**enc).last_hidden_state
            last = enc["attention_mask"].sum(dim=1) - 1                # index of each row's final REAL token
            h = hs[torch.arange(hs.size(0), device=hs.device), last]
            if normalize_embeddings:
                h = torch.nn.functional.normalize(h.float(), p=2, dim=-1)
            out.append(h.float().cpu())
        return torch.cat(out).to(self.dev) if convert_to_tensor else torch.cat(out).numpy()


def load_dense_encoder(src, m, dev=None):
    """Load a dense encoder honoring the optional trust_remote_code / max_seq_length keys (needed for gte-qwen2;"""
    dev = dev or DEV
    if m.get("peft_base"):
        return PromptrieverEncoder(src, m["peft_base"], m.get("max_seq_length", 512), dev=dev)
    from sentence_transformers import SentenceTransformer
    kw = {"trust_remote_code": True} if m.get("trust_remote_code") else {}
    enc = SentenceTransformer(src, device=dev, **kw)
    if m.get("max_seq_length"):
        enc.max_seq_length = m["max_seq_length"]
    return enc


def dense_index(model, docs, dp, bs=256):
    return model.encode([dp + (d if (d and d.strip()) else "empty") for d in docs], batch_size=bs,
                        convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False, device=DEV)


def dense_order(model, dmat, qp, q):
    qe = model.encode([qp + q], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False, device=DEV)
    return torch.argsort(dmat @ qe.squeeze(0), descending=True).tolist()


def chunk_doc(text, words, max_words):
    w = (text or "").split()
    if max_words:
        w = w[:max_words]
    if not w:
        return ["empty"]                                   # same placeholder contract as dense_index
    if words <= 0 or len(w) <= words:
        return [" ".join(w)]
    out = [w[i:i + words] for i in range(0, len(w), words)]
    tmin = int(os.environ.get("EXCISE_WINDOW_MIN_WORDS", "0"))
    if tmin and len(out) > 1 and len(out[-1]) < tmin:
        _tail = out.pop()          # pop FIRST: out[-2] is evaluated against the shortened list
        out[-1] = out[-1] + _tail
    return [" ".join(x) for x in out]


def build_passages(docs, words, max_words):
    """-> (passage_texts, owner) where owner[i] is the index of the document passage i came from."""
    texts, owner = [], []
    for di, d in enumerate(docs):
        for p in chunk_doc(d, words, max_words):
            texts.append(p); owner.append(di)
    return texts, torch.tensor(owner, dtype=torch.long, device=DEV)


def dense_index_windowed(model, texts, dp, bs=256):
    return model.encode([dp + t for t in texts], batch_size=bs, convert_to_tensor=True,
                        normalize_embeddings=True, show_progress_bar=False, device=DEV)


def resolve_encode_devices(spec):
    """'auto' -> every visible CUDA device; 'none'/'' -> single-GPU path; else an explicit id list."""
    spec = (spec or "").strip()
    if spec in ("none", "0-only", "single"):
        return []
    if spec == "auto":
        n = torch.cuda.device_count()
        return [f"cuda:{i}" for i in range(n)] if n > 1 else []
    return [f"cuda:{d.strip()}" for d in spec.split(",") if d.strip() != ""]


def parallel_index_windowed(load_replica, texts, dp, bs, devices):
    """Encode the passage index across several GPUs and return ONE matrix on DEV, identical to the"""
    import threading
    n = len(texts)
    bounds = [(n * i // len(devices), n * (i + 1) // len(devices)) for i in range(len(devices))]
    parts = [None] * len(devices)
    errs = [None] * len(devices)

    replicas = []
    for k, dv in enumerate(devices):
        lo, hi = bounds[k]
        replicas.append(load_replica(dv) if hi > lo else None)
    print(f"[external] parallel index: {len(replicas)} replicas loaded on {', '.join(devices)}", flush=True)

    def work(k):
        try:
            lo, hi = bounds[k]
            if lo >= hi:
                parts[k] = torch.empty(0)
                return
            out = replicas[k].encode([dp + t for t in texts[lo:hi]], batch_size=bs, convert_to_tensor=True,
                                     normalize_embeddings=True, show_progress_bar=False, device=devices[k])
            parts[k] = out.to(DEV)
        except Exception as e:                      # a dead replica must not silently shrink the index
            errs[k] = e

    ths = [threading.Thread(target=work, args=(k,)) for k in range(len(devices))]
    for t in ths: t.start()
    for t in ths: t.join()
    for k, e in enumerate(errs):
        if e is not None:
            raise RuntimeError(f"encode replica {k} on {devices[k]} failed: {e}") from e
    mat = torch.cat([p for p in parts if p.numel()], dim=0)
    assert mat.shape[0] == n, f"parallel index lost rows: {mat.shape[0]} != {n}"
    print(f"[external] parallel index: {n} passages across {len(devices)} devices "
          f"({', '.join(devices)}) -> {tuple(mat.shape)}", flush=True)
    return mat


_POOL = {"max": "amax", "min": "amin", "mean": "mean"}


def dense_order_windowed(model, pmat, owner, ndocs, qp, q, pool="max"):
    qe = model.encode([qp + q], convert_to_tensor=True, normalize_embeddings=True, show_progress_bar=False, device=DEV)
    s = (pmat @ qe.squeeze(0)).float()
    init = float("-inf") if pool == "max" else (float("inf") if pool == "min" else 0.0)
    best = torch.full((ndocs,), init, device=s.device, dtype=s.dtype)
    best.scatter_reduce_(0, owner, s, reduce=_POOL[pool], include_self=False)
    return torch.argsort(best, descending=True).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--first-k", type=int, default=100)
    ap.add_argument("--shortlist-orders", default="",
                    help='glob of order dumps (id -> order)')
    ap.add_argument("--ft", default="", help='path to a fine-tuned checkpoint (dense: replaces the encoder')
    ap.add_argument("--dump-orders", default="", help="write per-query {pos,negs,order[:100]} jsonl so the baselines' per-tier / per-metric rows are recomputable offline, exactly like ours")
    ap.add_argument("--window-words", type=int, default=0, help='0=truncate at 512 (published)')
    ap.add_argument("--window-pool", choices=["max", "min", "mean"], default="max",
                    help="how a document's passage scores combine")
    ap.add_argument("--max-doc-words", type=int, default=10000,
                    help='content cap per document when windowing')
    ap.add_argument("--encode-devices", default="auto",
                    help="'auto' (default) uses EVERY CUDA device visible to the process")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()
    m = MODELS[a.model]
    import glob as _glob                                   # a quoted pattern is expanded here, a file list passes through
    a.records = [p for r in a.records for p in (sorted(_glob.glob(r)) or [r])]
    recs = [json.loads(l) for f in a.records for l in open(f)]
    by_src = defaultdict(list)
    for r in recs:
        by_src[canonical(r.get("source", "excluir"))].append(r)

    # dense: --ft is the fine-tuned encoder (re-indexed automatically, since docs are re-encoded here).
    # rerank: first stage stays the BASE dense retriever; --ft only swaps the cross-encoder.
    dense_src = a.ft if (a.ft and m["mode"] == "dense") else (m["repo"] if m["mode"] == "dense" else m["first"])
    print(f"[external] model={a.model} ft={'yes:'+a.ft if a.ft else 'no (off-the-shelf)'} encoder={dense_src}")
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
        """Same cross-encoder, but every passage of each candidate is scored and the document takes its MAX."""
        flat, own = [], []
        for k, j in enumerate(idxs):
            for p in chunk_doc(docs[j], a.window_words, a.max_doc_words):
                flat.append(p); own.append(k)
        sc = []
        for i in range(0, len(flat), 64):
            enc_in = ce_tok([q] * len(flat[i:i + 64]), flat[i:i + 64], padding=True, truncation=True,
                            max_length=512, return_tensors="pt").to(DEV)
            sc.extend(ce_model(**enc_in).logits.squeeze(-1).float().cpu().tolist())
        if a.window_pool == "mean":
            tot = [0.0] * len(idxs); cnt = [0] * len(idxs)
            for k, s in zip(own, sc):
                tot[k] += s; cnt[k] += 1
            return [t / c if c else float("-inf") for t, c in zip(tot, cnt)]
        keep = (lambda s, b: s > b) if a.window_pool == "max" else (lambda s, b: s < b)
        init = float("-inf") if a.window_pool == "max" else float("inf")
        best = [init] * len(idxs)
        for k, s in zip(own, sc):
            if keep(s, best[k]): best[k] = s
        return best

    od = [] if a.dump_orders else None
    for src, rs in by_src.items():
        rs = rs[a.shard::a.nshards]
        if not rs:
            continue                    # skip the index build entirely -- nothing in this source for this shard
        ids, docs = load_source(src)
        i2i = id2idx(ids)
        if a.window_words > 0:
            ptexts, owner = build_passages(docs, a.window_words, a.max_doc_words)
            print(f"[external] {src}: windowed {len(docs)} docs -> {len(ptexts)} passages "
                  f"({len(ptexts)/max(len(docs),1):.2f} per doc, {a.window_words} words, cap {a.max_doc_words})",
                  flush=True)
            _devs = resolve_encode_devices(a.encode_devices)
            if len(_devs) > 1:
                dmat = parallel_index_windowed(lambda dv: load_dense_encoder(dense_src, m, dev=dv),
                                               ptexts, m["dp"], m.get("ibs", 256), _devs)
            else:
                dmat = dense_index_windowed(enc, ptexts, m["dp"], bs=m.get("ibs", 256))
        else:
            _devs = resolve_encode_devices(a.encode_devices)
            if len(_devs) > 1:
                _docs = [(d if (d and d.strip()) else "empty") for d in docs]
                dmat = parallel_index_windowed(lambda dv: load_dense_encoder(dense_src, m, dev=dv),
                                               _docs, m["dp"], m.get("ibs", 256), _devs)
            else:
                dmat = dense_index(enc, docs, m["dp"], bs=m.get("ibs", 256))
        first = ((lambda q: dense_order_windowed(enc, dmat, owner, len(docs), m["qp"], q, a.window_pool))
                 if a.window_words > 0 else (lambda q: dense_order(enc, dmat, m["qp"], q)))
        # Join dump (id -> order) with records (id -> query); rank_fn only ever sees the query text.
        q2short = {}
        if a.shortlist_orders:
            import glob as _g
            id2o = {}
            for _f in sorted(_g.glob(a.shortlist_orders)):
                for _l in open(_f):
                    _r = json.loads(_l)
                    if isinstance(_r, dict) and "order" in _r and "id" in _r:
                        id2o[_r["id"]] = _r["order"]
            for _f in a.records:
                for _p in sorted(_g.glob(_f)) or [_f]:
                    if not os.path.exists(_p):
                        continue
                    for _l in open(_p):
                        _r = json.loads(_l)
                        if _r.get("id") in id2o and _r.get("query"):
                            q2short[_r["query"].strip()] = id2o[_r["id"]]
            print(f"[shortlist] {len(q2short)} queries carry an external shortlist", flush=True)
        ce_fn = ce_scores_windowed if a.window_words > 0 else ce_scores
        if m["mode"] == "dense" and not q2short:
            rank_fn = first
        elif m["mode"] == "dense":
            def rank_fn(q):
                full = first(q)
                sh = q2short.get(q.strip())
                if not sh:
                    return full
                top = list(sh[:a.first_k]); ts = set(top)
                return [d for d in full if d in ts] + [d for d in full if d not in ts]
        else:
            def rank_fn(q):
                sh = q2short.get(q.strip()) if q2short else None
                full = list(sh) + [d for d in first(q) if d not in set(sh)] if sh else first(q)
                top = full[:a.first_k]
                s = ce_fn(q, docs, top)
                order = [top[i] for i in np.argsort(s)[::-1]]
                return order + full[a.first_k:]
        agg, skipped = EB.evaluate(rs, i2i, rank_fn, orders=od)
        EB.report(agg, skipped, src)
        del dmat
        torch.cuda.empty_cache()
    if a.dump_orders:
        with open(a.dump_orders, "w") as f:
            f.write(json.dumps({"__provenance__": {
                "reembed_lora": f"EXTERNAL:{a.model}", "finder": a.ft or "off-the-shelf", "thr": -1.0,
                "thr_source": "n/a-external", "finder_source": "n/a-external",
                "reindex": bool(a.ft) and m["mode"] == "dense",
                "first_stage": dense_src, "scorer": (ce_src if m["mode"] == "rerank" else dense_src),
                "shortlist_source": a.shortlist_orders or "own first stage",
                "faithful": False,
                "chunk_words": a.window_words, "query_len": "",
                "doc_truncation": ("512-token, no windowing" if a.window_words <= 0 else
                                   f"windowed {a.window_words}w max-pool, cap {a.max_doc_words}w")},
                "records": a.records, "n": len(od),
                "sources": sorted({r.get("source") for r in od})}) + "\n")
            for row in od: f.write(json.dumps(row) + "\n")
        print(f"wrote {len(od)} per-query orders to {a.dump_orders}")
    print(f"STRONG_DONE model={a.model}")


if __name__ == "__main__":
    main()
