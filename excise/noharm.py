"""How often the detector fires on ordinary queries, which carry no exclusion."""
from __future__ import annotations
import argparse, os, sys
# the repository root, so `config` and the packages import from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from excise import config as C
from excise.schema import has_neg_cue
from excise.sources import load_queries

NOHARM_DEFAULT = "scifact,nfcorpus,arguana,instructir,followir"

MINED = {"fiqa", "trec-covid", "negconstraint", "excluir"}


def anchor_queries(ds, limit):
    """Query text for a no-harm set. Anchors are not BEIR corpora, so they do not come through load_queries()."""
    if ds == "instructir":
        from excise.anchors import load_instructir
        _, q, _ = load_instructir()
        return [r["text"] for r in q if r.get("text")][:limit]
    if ds.startswith("followir"):
        from excise.anchors import load_followir
        name = ds.split("-", 1)[1] if "-" in ds else "robust04"
        d = load_followir(name)
        # FollowIR-POSITIVE: the query plus its ORIGINAL instruction. Positive instructions, no exclusion.
        return [((r.get("text") or "") + " " + (r.get("instruction_og") or "")).strip() for r in d["queries"]][:limit]
    return load_queries(ds, limit=limit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=NOHARM_DEFAULT)
    ap.add_argument("--allow-mined", action="store_true",
                    help='permit a MINED (train-side) source in the no-harm report — it is not a no-harm result')
    ap.add_argument("--finder", default=C.FINDER_CKPT)
    ap.add_argument("--thr", type=float, default=None)
    ap.add_argument("--max-q", type=int, default=2000); ap.add_argument("--device", default="cuda:0"); a = ap.parse_args()
    from excise.detector import DeepFinder
    F = DeepFinder(a.finder, a.device, thr=a.thr)
    print(f"[noharm] finder={a.finder} tap={F.tap} thr={F.thr}\n"
          f"{'dataset':12s} {'cue-free':>9s} {'false-fire':>11s} {'cue-bearing':>12s} {'fire(correct)':>14s}", flush=True)
    T = [0, 0, 0, 0]                                    # cue-free n, false fires, cue-bearing n, correct fires
    for ds in a.datasets.split(","):
        if ds in MINED and not a.allow_mined:
            print(f"{ds:12s} SKIPPED — a MINING corpus The Finder's no-fire rows are drawn "
                  f"from its queries; its fire-rate is not a no-harm result. Use --allow-mined to override.")
            continue
        qs = anchor_queries(ds, a.max_q)
        if not qs: print(f"{ds:12s} no queries available offline"); continue
        plain = [q for q in qs if not has_neg_cue(q)]
        cued = [q for q in qs if has_neg_cue(q)]
        ff = sum(1 for q in plain if F.extract_Z(q))     # FALSE fires: the operator would touch a normal query
        cf = sum(1 for q in cued if F.extract_Z(q))      # CORRECT fires: the query really does exclude something
        T = [T[0] + len(plain), T[1] + ff, T[2] + len(cued), T[3] + cf]
        print(f"{ds:12s} {len(plain):9d} {ff/max(len(plain),1):11.3f} {len(cued):12d} "
              f"{(cf/len(cued)) if cued else float('nan'):14.3f}", flush=True)
    print(f"\nOVERALL false-fire on cue-free queries: {T[1]}/{T[0]} = {T[1]/max(T[0],1):.4f}")
    print(f"        recall on cue-bearing queries : {T[3]}/{T[2]} = {T[3]/max(T[2],1):.4f}  (should be HIGH)")
    print("NOTE: a low false-fire rate is necessary but NOT sufficient for no-harm — the candidate re-embed runs")
    print("      on every query. Pair this with evaluate_beir.py (arm-frozen vs arm-ours nDCG@10).")
    print("NOHARM_DONE")


if __name__ == "__main__":
    main()
