#!/usr/bin/env python3
"""Does a rule still work when the excluded document does not name the excluded topic verbatim? (Appendix B.4)

Test queries are split by whether every content word of the excluded topic appears in the excluded document.
Success@10 is then reported for each group and each system. No model is run; the documents are read from the corpus by id, and the per-query outcomes come from
  --perquery  files written by  scripts/nine_interventions.py --dump-perquery   (its arms, e.g. maxsimz, bm25z)
  --orders    LABEL=FILE  ranking dumps written by  excise.run ... --dump-orders FILE   (e.g. EXCISE)

  python scripts/nine_interventions.py --source excluir --detector-z --dump-perquery nine_excluir.jsonl
  python scripts/nine_interventions.py --source fiqa    --detector-z --dump-perquery nine_fiqa.jsonl
  python -m excise.run --backbone reason --dataset excluir --system excise --dump-orders excise_excluir.jsonl
  python -m excise.run --backbone reason --dataset fiqa    --system excise --dump-orders excise_fiqa.jsonl
  python scripts/verbatim_overlap.py --sources excluir fiqa --perquery nine_excluir.jsonl nine_fiqa.jsonl \
      --arms maxsimz bm25z --orders EXCISE=excise_excluir.jsonl EXCISE=excise_fiqa.jsonl
"""
from __future__ import annotations
import argparse, glob, json, os, re, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
STOP = set("the a an of and or not but is are was were to in on for with by at from as that this these those it its "
           "be been being have has had do does did can could will would should may might into than then so such no nor "
           "about over under between within without other which who whom whose what when where why how all any each".split())


def tok(s):
    return [w for w in re.findall(r"[a-z0-9]+", (s or "").lower()) if w not in STOP and len(w) > 1]


def overlap(rec, text):
    """Share of the excluded topic's content words found in its excluded document (best document per topic,
    averaged over the topics of a compound query). None when the record has nothing to compare."""
    zs = rec["Z"] if isinstance(rec["Z"], list) else [rec["Z"]]
    negs = [set(tok(text[str(n)])) for n in rec.get("neg_ids", [])]
    vals = []
    for z in zs:
        zt = set(tok(z))
        if zt and negs:
            vals.append(max(len(zt & n) / len(zt) for n in negs))
    return sum(vals) / len(vals) if vals else None


def success_from_orders(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if "order" not in r:
            continue
        top = r["order"][:10]
        out[r["id"]] = int(r["pos"] in top and not any(n in top for n in r["negs"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", default=["excluir", "fiqa"])
    ap.add_argument("--perquery", nargs="*", default=[], help="files from nine_interventions.py --dump-perquery")
    ap.add_argument("--arms", nargs="*", default=["maxsimz", "bm25z"], help="arms to read from the --perquery files")
    ap.add_argument("--orders", nargs="*", default=[], help="LABEL=FILE ranking dumps from excise.run --dump-orders")
    a = ap.parse_args()

    from excise.sources import load_source
    recs, ov = {}, {}
    for src in a.sources:
        ids, texts = load_source(src); text = dict(zip(ids, texts))        # documents are linked by id
        for f in sorted(glob.glob(os.path.join(ROOT, "data", "benchmark", "test", src, "*.jsonl"))):
            for line in open(f):
                r = json.loads(line); recs[r["id"]] = r
                o = overlap(r, text)
                if o is not None: ov[r["id"]] = o
    verb = {i: (o >= 0.999) for i, o in ov.items()}

    systems = defaultdict(dict)
    for f in a.perquery:
        for line in open(f):
            r = json.loads(line)
            for arm in a.arms:
                if arm in r["s"]:
                    systems[arm][r["id"]] = int(r["s"][arm])
    for spec in a.orders:
        label, path = spec.split("=", 1)
        systems[label].update(success_from_orders(path))

    n_v = sum(verb.values()); n_n = len(verb) - n_v
    print(f"{len(recs)} test queries of {', '.join(a.sources)}; excluded topic named verbatim in the excluded document: "
          f"{n_v}; not verbatim: {n_n}")
    print(f"\n{'system':16s} {'verbatim':>16s} {'not verbatim':>16s}")
    for name, s in systems.items():
        cell = []
        for flag in (True, False):
            x = [s[i] for i in verb if verb[i] == flag and i in s]
            cell.append(f"{sum(x) / len(x):.4f} (n={len(x)})" if x else "n/a")
        print(f"{name:16s} {cell[0]:>16s} {cell[1]:>16s}")
    print("\nOVERLAP_DONE")


if __name__ == "__main__":
    main()
