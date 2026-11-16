"""Evaluate an arm on BEIR (nDCG@10): the ordinary-retrieval check that EXCISE leaves queries without an
exclusion where the frozen index puts them.

  python -m excise.evaluate_beir --datasets scifact,nfcorpus,arguana <arm flags>"""
from __future__ import annotations
import argparse, os, sys
from collections import defaultdict
# the repository root, so `config` and the packages import from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from excise import config as C
from excise.metrics import ndcg_at_k_graded, recall_at_k
from excise.sources import load_source, id2idx


def load_qrels(ds, split="test"):
    """BEIR qrels: {query_id: {doc_id: gain}}. Graded — do NOT binarise (TREC-COVID also ships gain -1)."""
    from excise.sources import local_part
    rows = local_part(ds, "qrels")                              # the self-contained export, one place, offline
    if rows is None:
        from datasets import load_dataset
        rows = [dict(r, split=split) for r in load_dataset(f"BeIR/{ds}-qrels")[split]]
    out = defaultdict(dict)
    for r in rows:
        if r.get("split", split) != split: continue
        out[str(r["query-id"])][str(r["corpus-id"])] = int(r["score"])
    return out


def load_queries_with_ids(ds, limit=None):
    from excise.sources import local_part
    rows = local_part(ds, "queries")
    if rows is None:
        from datasets import load_dataset
        rows = list(load_dataset(f"BeIR/{ds}", "queries")["queries"])
    rows = [(str(r["_id"]), r["text"]) for r in rows if r.get("text")]
    return rows[:limit] if limit else rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="scifact,nfcorpus,arguana")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-q", type=int, default=0, help='0 = all queries with qrels')
    ap.add_argument("--dump-perq", default="", help='write per-query nDCG@10 / fired flag as JSONL')
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    from excise.arms import add_arm_args, build_method, rank_fn, index_kwargs
    a = add_arm_args(ap).parse_args()

    M = build_method(a)                                   # models loaded once; index rebuilt per dataset
    for ds in a.datasets.split(","):
        qrels = load_qrels(ds, a.split)
        ids, docs = load_source(ds); i2 = id2idx(ids)
        qs = [(qid, qt) for qid, qt in load_queries_with_ids(ds) if qid in qrels]
        if a.max_q: qs = qs[:a.max_q]
        qs = qs[a.shard::a.nshards]
        if not qs: print(f"{ds}: no queries with qrels in split '{a.split}'"); continue
        M.index(docs, **index_kwargs(a))
        rf = rank_fn(M, a)
        nd = rc = 0.0; fired = reembedded = 0; unresolved = skipped = 0; run = {}
        perq = [] if a.dump_perq else None
        for k, (qid, qt) in enumerate(qs):
            rel = {}
            for did, g in qrels[qid].items():
                if did in i2: rel[i2[did]] = g
                else: unresolved += 1
            if not rel: skipped += 1; continue             # every judged doc missing => scoring it is meaningless
            order, info = rf(qt)
            order = order.tolist()
            q_nd = ndcg_at_k_graded(order, rel, 10)
            nd += q_nd; rc += recall_at_k(order, rel, 100)
            fired += bool(info["fired"]); reembedded += bool(info.get("reembedded", True))
            if perq is not None:
                perq.append({"ds": ds, "qid": qid, "ndcg10": float(q_nd), "fired": bool(info["fired"])})
            run[qid] = {ids[d]: float(len(order) - r) for r, d in enumerate(order)}
            if k % 200 == 0: print(f"  {ds} {k}/{len(qs)}", flush=True)
        n = len(run)                                       # SCORED queries, not attempted (skipped ones would
        if not n: print(f"{ds}: 0 scorable queries"); M.free_index(); continue   # silently deflate every mean)
        try:
            import pytrec_eval
            ev = pytrec_eval.RelevanceEvaluator({q: {d: int(g) for d, g in qrels[q].items()} for q in run},
                                                {"ndcg_cut_10", "recall_100"})
            res = ev.evaluate(run)
            p_nd = sum(v["ndcg_cut_10"] for v in res.values()) / max(len(res), 1)
            p_rc = sum(v["recall_100"] for v in res.values()) / max(len(res), 1)
            ref = f"  [pytrec] nDCG@10={p_nd*100:.2f} recall@100={p_rc*100:.2f}"
        except ImportError:
            ref = "  [pytrec] unavailable"
        print(f"{ds:12s} n={n:5d}  nDCG@10={nd/n:.4f}  recall@100={rc/n:.4f}  "
              f"switch_fired={fired}/{n} ({fired/n:.3f})  reembedded={reembedded}/{n}"
              + (f"  !! {skipped} queries had no qrel doc in corpus" if skipped else "")
              + (f"  !! {unresolved} qrel doc-ids not in corpus" if unresolved else ""), flush=True)
        print(f"{'':12s} {ref}", flush=True)
        if perq is not None:
            import json as _json
            with open(a.dump_perq, "a") as fh:
                for r in perq: fh.write(_json.dumps(r) + "\n")
            print(f"{'':12s}  [perq] +{len(perq)} rows -> {a.dump_perq}", flush=True)
        M.free_index()
    print("BEIR_DONE")


if __name__ == "__main__":
    main()
