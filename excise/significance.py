#!/usr/bin/env python3
"""Paired bootstrap on success@10 between two systems, from their ranking dumps (--dump-orders), aligned by
query id. Give one dump per collection for each system to pool several collections into one test.

  python -m excise.significance --a A.orders.jsonl --b B.orders.jsonl
  python -m excise.significance --a A_excluir.orders.jsonl A_fiqa.orders.jsonl --b B_excluir.orders.jsonl B_fiqa.orders.jsonl"""
import argparse, json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from excise.metrics import success10


def load(path):
    """{query id -> success@10} from one --dump-orders file."""
    out = {}
    for ln in open(path):
        r = json.loads(ln)
        if "order" not in r:
            continue
        _, _, suc = success10(r["order"], {int(r["pos"])}, {int(x) for x in r["negs"]})
        out[r["id"]] = suc
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True, help="system A's dumps, one per collection")
    ap.add_argument("--b", nargs="+", required=True, help="system B's dumps, in the same order")
    ap.add_argument("--labelA", default="A"); ap.add_argument("--labelB", default="B")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=12345)
    g = ap.parse_args()
    if len(g.a) != len(g.b):
        ap.error("--a and --b need the same number of files")

    a, b = [], []
    for fa, fb in zip(g.a, g.b):
        A, B = load(fa), load(fb)
        ids = sorted(set(A) & set(B))
        if not ids:
            sys.exit(f"no shared query ids between {fa} and {fb}")
        if len(ids) != len(A) or len(ids) != len(B):
            print(f"note: {os.path.basename(fa)}: aligning on {len(ids)} shared ids ({len(A)} in A, {len(B)} in B)")
        a += [A[i] for i in ids]; b += [B[i] for i in ids]
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    ma, mb, md = sum(a) / n, sum(b) / n, sum(d) / n

    rng = random.Random(g.seed)
    diffs = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(g.iters))
    lo, hi = diffs[int(0.025 * g.iters)], diffs[int(0.975 * g.iters)]
    p = min(1.0, 2 * min(sum(1 for x in diffs if x <= 0), sum(1 for x in diffs if x >= 0)) / g.iters)

    print(f"n = {n} paired queries, {g.iters} resamples, seed {g.seed}")
    print(f"  {g.labelA:<12} success@10 = {ma:.4f}")
    print(f"  {g.labelB:<12} success@10 = {mb:.4f}")
    print(f"  paired difference   = {md:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   p = {p:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
