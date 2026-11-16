"""Evaluate an arm on X-BENCH records: success@10, right-rank, nDCG@10, xnDCG, hit@10 and leak, overall and
per tier. Each record's corpus is resolved from its `source`.

  python -m excise.evaluate --records 'data/benchmark/test/excluir/*.jsonl' <arm flags>"""
from __future__ import annotations
import argparse, json, os, sys
from collections import defaultdict
# the repository root, so `config` and the packages import from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from excise import config as C
from excise.metrics import success10, right_rank, ndcg10, ndcg10_excl


def is_excluir_source(source):
    """Any excluir* tag resolves to the ExcluIR corpus."""
    from excise.sources import canonical
    return canonical(source) == "excluir"


def load_corpus(source):
    """Return (docs:list[str], id2idx:dict[str,int]) for the record's source corpus."""
    from excise.sources import load_source, id2idx                       # single registry: never key the wrong corpus
    ids, docs = load_source(source)
    return docs, id2idx(ids)


def evaluate(records, id2idx, rank_fn, perquery=None, orders=None):
    """rank_fn(query) -> ranked list of corpus indices. Injectable so this is CPU-testable without a GPU."""
    agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0])        # succ, right_rank, ndcg, recall, excl, xndcg, n
    skipped = []
    for r in records:
        pid = str(r["pos_id"])
        if pid not in id2idx: skipped.append((r.get("id"), "pos_id unresolved")); continue
        p = id2idx[pid]
        negs = [id2idx[str(n)] for n in r["neg_ids"] if str(n) in id2idx]
        if len(negs) != len(r["neg_ids"]):                              # a neg didn't resolve => wrong corpus
            skipped.append((r.get("id"), "neg_id unresolved (wrong corpus?)")); continue
        order = rank_fn(r["query"])
        rec, exc, suc = success10(order, {p}, set(negs))
        rr, nd = right_rank(order, p, negs), ndcg10(order, {p})
        xnd = ndcg10_excl(order, {p}, set(negs))                        # exclusion-aware nDCG@10 (negatives penalised)
        if perquery is not None:
            perquery.append({"id": r.get("id"), "source": r.get("source"), "tier": r.get("tier"),
                             "success10": int(suc), "ndcg10": nd, "xndcg10": xnd, "excl": exc})
        if orders is not None:
            orders.append({"id": r.get("id"), "source": r.get("source"), "tier": r.get("tier"),
                           "pos": int(p), "negs": [int(n) for n in negs], "order": [int(d) for d in order[:100]]})
        for key in ("ALL", f"tier={r.get('tier','?')}", f"len={r.get('length','?')}"):
            a = agg[key]; a[0] += suc; a[1] += rr; a[2] += nd; a[3] += rec; a[4] += exc; a[5] += xnd; a[6] += 1
    return agg, skipped


def report(agg, skipped, source):
    print(f"\n===== source={source} =====")
    print(f"{'slice':22s} {'n':>6s} {'success10':>10s} {'rightRank':>10s} {'nDCG@10':>9s} {'xnDCG@10':>9s} {'hit@10':>8s} {'excl↓':>7s}")
    for k in sorted(agg, key=lambda x: (x != "ALL", x)):
        s, rr, nd, rc, ex, xn, n = agg[k]
        if n: print(f"{k:22s} {n:6d} {s/n:10.4f} {rr/n:10.4f} {nd/n:9.4f} {xn/n:9.4f} {rc/n:8.3f} {ex/n:7.3f}")
    if skipped:
        print(f"  !! SKIPPED {len(skipped)} records (id resolution failed) — first 3: {skipped[:3]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--dump-perquery", default="", help='write per-query success10 jsonl here (bootstrap CIs)')
    ap.add_argument("--dump-orders", default="", help='write per-query {pos,negs,order[:100]} jsonl here so any future metric can be recomputed offline WITHOUT re-running the ranker')
    from excise.arms import add_arm_args, build_method, rank_fn, index_kwargs
    a = add_arm_args(ap).parse_args()
    from excise.sources import canonical
    import glob as _glob                                   # a quoted pattern is expanded here, a file list passes through
    a.records = [p for r in a.records for p in (sorted(_glob.glob(r)) or [r])]
    recs = [json.loads(l) for f in a.records for l in open(f)]
    by_src = defaultdict(list)
    for r in recs: by_src[canonical(r.get("source", "excluir"))].append(r)
    M = build_method(a)                                                 # models loaded ONCE, index rebuilt per source
    pq = [] if a.dump_perquery else None
    od = [] if a.dump_orders else None
    for src, rs in by_src.items():
        rs = rs[a.shard::a.nshards]
        if not rs: continue
        docs, id2idx = load_corpus(src)
        M.index(docs, **index_kwargs(a))
        agg, skipped = evaluate(rs, id2idx, lambda q: rank_fn(M, a)(q)[0].tolist(), perquery=pq, orders=od)
        report(agg, skipped, src)
        M.free_index()                                                  # free before indexing the next source corpus
    if a.dump_perquery:
        with open(a.dump_perquery, "w") as f:
            for row in pq: f.write(json.dumps(row) + "\n")
        print(f"wrote {len(pq)} per-query rows to {a.dump_perquery}")
    if a.dump_orders:
        from excise.arms import resolved_system
        with open(a.dump_orders, "w") as f:
            f.write(json.dumps({"__provenance__": resolved_system(a),
                                "records": a.records, "n": len(od),
                                "sources": sorted({r.get("source") for r in od})}) + "\n")
            for row in od: f.write(json.dumps(row) + "\n")
        print(f"wrote {len(od)} per-query orders to {a.dump_orders}")
    print("BENCHMARK_DONE")


if __name__ == "__main__":
    main()
