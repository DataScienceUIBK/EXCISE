"""Add the detector's Z span to each training triple, so the adapter's Z margin can be supervised.

  python -m excise.label_z --triples data/benchmark/train/adapter_triples_unlabelled.jsonl --finder <detector> --thr 0.70 --out <file>"""
from __future__ import annotations
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from excise import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", required=True)
    ap.add_argument("--finder", required=True)
    ap.add_argument("--thr", type=float, default=0.76)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    from excise.detector import DeepFinder
    d = DeepFinder(a.finder, a.device, thr=a.thr)
    rows = [json.loads(l) for l in open(a.triples)]
    out, silent = [], 0
    for r in rows:
        Z = d.extract_Z(r["query"])
        ok = 1.0
        if not Z:
            Z = r["query"]; ok = 0.0; silent += 1
        out.append({"query": r["query"], "positive": r["positive"],
                    "negative": r["negative"], "z_query": Z, "label": ok})
    with open(a.out, "w") as f:
        for r in out: f.write(json.dumps(r) + "\n")
    print(f"[labelz] {len(out)} rows -> {a.out}")
    print(f"[labelz] detector silent on {silent} ({silent/len(out):.3f}); those fall back to the full query")
    print("LABELZ_DONE")


if __name__ == "__main__":
    main()
