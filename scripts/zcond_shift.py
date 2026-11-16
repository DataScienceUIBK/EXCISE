#!/usr/bin/env python3
"""Why Z-conditional re-embedding fails (Appendix A.3): prepending the excluded topic to a document raises the
gold's match to that topic more than the excluded document's, so the two become harder to separate.

For every test record of the chosen collections (the record's own Z, frozen Reason-ModernColBERT), the gold and its
excluded document(s) are encoded with and without "<Z>. " prepended. Reported per document:
  topic match  = length-normalised MaxSim(Z, d)
  rule signal  = length-normalised MaxSim(q_X, "<Z>. d") - MaxSim(q_X, d), the penalty the rule subtracts

  python scripts/zcond_shift.py --sources excluir fiqa
"""
from __future__ import annotations
import argparse, glob, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("EXCISE_BACKBONE", "lightonai/Reason-ModernColBERT")
os.environ.setdefault("EXCISE_CHUNK_WORDS", "512")
os.environ.setdefault("EXCISE_MAX_DOC_WORDS", "10000")

import torch
from excise.encoder import load_model, emb
from excise.maxsim import weighted_maxsim_matrix as maxsim
from excise.exclusion import strip_Z
from excise.sources import load_source


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="+", default=["excluir", "fiqa"])
    ap.add_argument("--dump", default="", help="write the per-query values to this JSON-lines file")
    a = ap.parse_args()
    dev = "cuda:0"
    M = load_model("", dev)
    pt = lambda qv, qm, V, Mk: (maxsim(qv.float(), qm.float(), V.float(), Mk)[0] / float(qm.sum().clamp(min=1.0))).tolist()
    cut = lambda t: " ".join((t or "").split()[:512])
    rows = []
    for src in a.sources:
        ids, texts = load_source(src); text = dict(zip(ids, texts))        # documents are linked by id
        for f in sorted(glob.glob(f"data/benchmark/test/{src}/*.jsonl")):
            for line in open(f):
                r = json.loads(line)
                zs = r["Z"] if isinstance(r["Z"], list) else [r["Z"]]
                Zj = "; ".join(z for z in zs if z)
                if not Zj or not r.get("neg_ids"):
                    continue
                docs = [cut(text[str(r["pos_id"])])] + [cut(text[str(n)]) for n in r["neg_ids"]]
                qxv, qxm = emb(M, [strip_Z(r["query"], Zj)], True, dev)
                ezv, ezm = emb(M, [Zj], True, dev)
                V0, M0 = emb(M, docs, False, dev)
                Vz, Mz = emb(M, [f"{Zj}. {t}" for t in docs], False, dev)
                rows.append(dict(id=r["id"], src=src, tier=r["tier"], rel0=pt(qxv, qxm, V0, M0), relZ=pt(qxv, qxm, Vz, Mz),
                                 z0=pt(ezv, ezm, V0, M0), zZ=pt(ezv, ezm, Vz, Mz)))
    if a.dump:
        with open(a.dump, "w") as f:
            for x in rows: f.write(json.dumps(x) + "\n")
    mean = lambda x: sum(x) / max(1, len(x))
    gold = lambda k: [r[k][0] for r in rows]
    neg = lambda k: [mean(r[k][1:]) for r in rows]                       # mean over the query's excluded documents
    diff = lambda x, y: [v - u for u, v in zip(x, y)]
    frac = lambda x, y: mean([float(v > u) for u, v in zip(x, y)])       # share of queries where excluded > gold
    print(f"{len(rows)} queries of {', '.join(a.sources)}")
    print(f"topic match before prepending   gold {mean(gold('z0')):.4f}   excluded {mean(neg('z0')):.4f}   "
          f"excluded higher on {frac(gold('z0'), neg('z0')):.3f} of queries")
    print(f"topic match after prepending    gold {mean(gold('zZ')):.4f}   excluded {mean(neg('zZ')):.4f}   "
          f"excluded higher on {frac(gold('zZ'), neg('zZ')):.3f} of queries")
    print(f"gain in topic match             gold {mean(diff(gold('z0'), gold('zZ'))):+.4f}   "
          f"excluded {mean(diff(neg('z0'), neg('zZ'))):+.4f}")
    dg, dn = diff(gold('rel0'), gold('relZ')), diff(neg('rel0'), neg('relZ'))
    print(f"rule signal                     gold {mean(dg):+.4f}   excluded {mean(dn):+.4f}   "
          f"excluded higher on {frac(dg, dn):.3f} of queries (0.5 = no separation)")
    print("ZCOND_DONE")


if __name__ == "__main__":
    main()
