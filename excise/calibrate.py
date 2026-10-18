"""The detector's false-firing rate on cue-free SciFact and NFCorpus queries, and its fire rate on training-split
exclusion queries, at each threshold.

  python -m excise.calibrate --finder model/checkpoints/<detector>/deepfinder.pt --budget 0.01"""
from __future__ import annotations
import argparse, json, os, sys
# the repository root, so `config` and the packages import from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from excise import config as C
from excise.schema import has_neg_cue

BUDGET_SETS = "scifact,nfcorpus"
DIAGNOSTIC_SETS = "arguana"
TRAIN_POS = ["data/benchmark/train/excluir.jsonl", "data/benchmark/train/fiqa.jsonl"]
GRID = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.76, 0.80, 0.85, 0.90, 0.94, 0.95, 0.98]
# The released threshold, kept when the sweep shows no clear optimum (see the selection rule below).
DEFAULT_TAU = 0.70


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--finder", required=True, help='detector checkpoint (explicit')
    ap.add_argument("--datasets", default=BUDGET_SETS, help='sets the BUDGET is enforced on')
    ap.add_argument("--budget-source", choices=["reported", "heldout"], default="reported",
                    help="'reported' (default) bounds false-firing on SciFact+NFCorpus")
    ap.add_argument("--diagnostic", default=DIAGNOSTIC_SETS,
                    help='sets reported but not part of the false-firing budget')
    ap.add_argument("--flat-eps", type=float, default=0.05,
                    help="if Youden's J varies by at most this across the admissible range, the default τ is kept")
    ap.add_argument("--budget", type=float, default=0.01, help='max acceptable false-fire rate on cue-free queries')
    ap.add_argument("--max-q", type=int, default=2000)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--write", action="store_true", help='write the selected τ to <ckpt-dir>/thr.txt')
    ap.add_argument("--out", default="", help='also write the full sweep as json')
    a = ap.parse_args()

    from excise.detector import DeepFinder
    from excise.noharm import anchor_queries
    F = DeepFinder(a.finder, a.device, thr=0.5)
    print(f"[calib] backbone={C.BACKBONE}")
    print(f"[calib] finder={a.finder} tap={F.tap}")
    if a.budget_source == "heldout" and a.datasets == BUDGET_SETS:
        a.datasets = "heldout-fiqa"
        a.diagnostic = (a.diagnostic + ",scifact,nfcorpus").strip(",")   # still REPORTED, never binding
    print(f"[calib] budget={a.budget} on [{a.datasets}]   diagnostic-only: [{a.diagnostic}]", flush=True)
    print(f"[calib] budget-source={a.budget_source}"
          f"{'  (SciFact/NFCorpus demoted to diagnostic -- they are reported by no-harm)' if a.budget_source=='heldout' else ''}",
          flush=True)

    def heldout_pool(max_q):
        """Cue-free queries that are NEITHER reported NOR trained on."""
        import json as _j, glob as _g
        seen = set()
        for _p in ("data/benchmark/train/detector.jsonl",):
            if os.path.exists(_p):
                for l in open(_p, encoding="utf-8"):
                    seen.add(" ".join(_j.loads(l).get("instruction", "").lower().split()))
        for _p in _g.glob("data/benchmark/test/*/*.jsonl"):
            for l in open(_p, encoding="utf-8"):
                q = _j.loads(l).get("query")
                if q: seen.add(" ".join(q.lower().split()))
        try:
            from datasets import load_dataset
            qs = [r["text"] for r in load_dataset("BeIR/fiqa", "queries")["queries"]]
        except Exception as e:
            print(f"  heldout-fiqa unavailable ({str(e)[:60]}) -- SKIPPED"); return []
        out = [q for q in qs
               if not has_neg_cue(q) and " ".join(q.lower().split()) not in seen]
        return out[:max_q] if max_q else out

    def score_sets(names):
        plain, cued, per = [], [], {}
        for ds in [d for d in names.split(",") if d]:
            qs = heldout_pool(a.max_q) if ds == "heldout-fiqa" else anchor_queries(ds, a.max_q)
            if not qs:
                print(f"  {ds:12s} no queries available offline -- SKIPPED"); continue
            p = [F.fire_score(q) for q in qs if not has_neg_cue(q)]  # ONE forward pass per query, all thresholds
            c = [F.fire_score(q) for q in qs if has_neg_cue(q)]
            plain += p; cued += c
            per[ds] = {"cue_free": len(p), "cue_bearing": len(c)}
            print(f"  {ds:12s} cue-free={len(p):5d}  cue-bearing={len(c):5d}", flush=True)
        return plain, cued, per

    print("[calib] budget sets:")
    plain_scores, cued_scores, per_ds = score_sets(a.datasets)
    print("[calib] diagnostic sets (reported, NOT in the budget):")
    dplain, dcued, per_diag = score_sets(a.diagnostic)
    per_ds = {**per_ds, **{k: {**v, "diagnostic_only": True} for k, v in per_diag.items()}}

    if not plain_scores:
        raise SystemExit("[calib] no cue-free queries scored -- cannot calibrate")

    def rate(scores, t): return (sum(1 for s in scores if s > t) / len(scores)) if scores else float("nan")
    def ff(t):  return rate(plain_scores, t)
    def rec(t): return rate(cued_scores, t)

    # POSITIVE side: TRAIN-split exclusion queries, every one of which should fire -> TPR.
    import json as _json
    tq = [_json.loads(l)["query"] for p in TRAIN_POS for l in open(p)]
    print(f"[calib] train-split exclusion queries (should ALL fire): {len(tq)}", flush=True)
    tp_scores = [F.fire_score(q) for q in tq]
    def tpr(t): return rate(tp_scores, t)

    print(f"\n{'τ':>6s} {'TPR[train]':>11s} {'false-fire':>11s} {'J':>8s} {'rho[diag]':>11s}"
          f"   (train_pos={len(tp_scores)}, budget n_cue_free={len(plain_scores)}; diag n={len(dplain)+len(dcued)})")
    sweep = []
    for t in GRID:
        f, tp = ff(t), tpr(t)
        rho = rate(dplain + dcued, t)
        sweep.append({"tau": t, "tpr_train": tp, "false_fire": f, "J": tp - f,
                      "cue_recall_noharm": rec(t), "rho_diagnostic": rho})
        print(f"{t:6.2f} {tp:11.4f} {f:11.4f} {tp-f:8.4f} {rho:11.4f}"
              f"{'   <= under budget' if f <= a.budget else ''}")

    ok = [s for s in sweep if s["false_fire"] <= a.budget]
    if not ok:
        raise SystemExit(f"\n[calib] NO threshold in the grid meets false-fire <= {a.budget}. "
                         f"Best is τ={min(sweep, key=lambda s: s['false_fire'])['tau']} at "
                         f"{min(s['false_fire'] for s in sweep):.4f}. Widen GRID or revisit the budget -- do NOT "
                         f"silently pick a different rule for this backbone.")
    Js = [s["J"] for s in ok]
    spread = max(Js) - min(Js)
    taus = [s["tau"] for s in ok]
    arg = max(ok, key=lambda s: (s["J"], s["tau"]))
    interior = min(taus) < arg["tau"] < max(taus)
    if spread <= a.flat_eps or not interior:
        best = min(ok, key=lambda s: abs(s["tau"] - DEFAULT_TAU))
        why = (f"τ IS INERT over the admissible range: J spread = {spread:.4f} <= {a.flat_eps}"
               if spread <= a.flat_eps else
               f"J HAS NO INTERIOR OPTIMUM: argmax sits at τ={arg['tau']}, the "
               f"{'lower' if arg['tau'] == min(taus) else 'upper'} edge of the admissible range "
               f"[{min(taus)}, {max(taus)}]")
        print(f"\n[calib] {why}.")
        print(f"[calib] keeping the default τ = {best['tau']} rather than reading a selection off the "
              f"grid edge.")
        rule = "inert-retain-default" if spread <= a.flat_eps else "no-interior-optimum-retain-default"
    else:
        best = arg
        rule = "argmax-J"
        print(f"\n[calib] J spread = {spread:.4f} > {a.flat_eps} and argmax τ={best['tau']} is interior to "
              f"[{min(taus)}, {max(taus)}] -- the curve bends, selecting argmax J.")
    best = {**best, "rule": rule, "J_spread": spread}
    print(f"[calib] SELECTED τ = {best['tau']}  (J={best['J']:.4f} = TPR {best['tpr_train']:.4f} "
          f"- FPR {best['false_fire']:.4f};  ArguAna ρ={best['rho_diagnostic']:.4f};  rule={best['rule']})")

    if a.write:
        p = os.path.join(os.path.dirname(a.finder), "thr.txt")
        open(p, "w").write(f"{best['tau']}\n")
        print(f"[calib] wrote {p}")
    if a.out:
        json.dump({"backbone": C.BACKBONE, "finder": a.finder, "budget": a.budget,
                   "datasets": per_ds, "sweep": sweep, "selected": best}, open(a.out, "w"), indent=2)
        print(f"[calib] wrote {a.out}")
    print("CALIBRATE_DONE")


if __name__ == "__main__":
    main()
