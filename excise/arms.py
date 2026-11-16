"""Build a retrieval arm from its command-line flags: the frozen index, a fine-tune, a query-only adapter,
or EXCISE with its operator constants. Shared by evaluate.py and evaluate_beir.py."""
from __future__ import annotations
import sys, os
# the repository root, so `config` and the packages import from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from excise import config as C


def add_arm_args(ap):
    ap.add_argument("--reembed-lora", default=C.REEMBED_LORA,
                    help='re-embed encoder ckpt')
    ap.add_argument("--finder", default=C.FINDER_CKPT, help='finder ckpt')
    ap.add_argument("--thr", type=float, default=None, help='finder fire threshold (default config FINDER_THR)')
    ap.add_argument("--lam", type=float, default=None, help='demote strength (default config DEMOTE_LAM)')
    ap.add_argument("--soft-demote", action="store_true",
                    help='CONTINUOUS evidence-scaled penalty on EVERY candidate (score -= lam*scale*relu(perTokenMaxSim(Z)-tau)) instead of hard demote-1: no-harm by construction on neutral queries + penalises ALL Z-docs')
    ap.add_argument("--signed-op", action="store_true",
                    help='TRAINING-FREE signed late-interaction on frozen candidates: score = rel - lam*perTokenMaxSim(Z) (continuous, no relu floor)')
    ap.add_argument("--lexical-neg", action="store_true",
                    help='TRAINING-FREE domain-invariant lexical backstop: hard-demote top-K docs whose text literally contains an excluded (Z) word')
    ap.add_argument("--z-veto", type=int, default=0,
                    help='TRAINING-FREE false-fire veto: if the finder flags MORE than N distinct content tokens, treat it as no-fire (a real exclusion flags ~3')
    ap.add_argument("--z-hard", type=int, default=0,
                    help='TRAINING-FREE false-fire fallback: if the finder flags MORE than N content tokens, use HARD demote-1 (not soft)')
    ap.add_argument("--blend", action="store_true",
                    help='BEST-OF-BOTH re-embed: fuse frozen relevance (strong in-domain) with LoRA relevance (strong OOD), z-normalised')
    ap.add_argument("--blend-beta", type=float, default=0.5, help='blend weight on LoRA (0=frozen, 1=LoRA)')
    ap.add_argument("--hard-m", type=int, default=1, help='hard-demote the top-m candidates by Z-evidence')
    ap.add_argument("--hard-guard", action="store_true", help='never hard-demote the top-relevant candidate (protects the answer on AND/OR false-fires)')
    ap.add_argument("--no-hard-guard", action="store_true",
                    help='ablation only: run hard demotion WITHOUT the guard')
    ap.add_argument("--no-hard-demote", action="store_true",
                    help='ablation only: force hard demotion OFF even when --hard1/--hard-auto are present')
    ap.add_argument("--hard-guard-rule", default="top_rel", choices=["top_rel", "rel_not_am"],
                    help='WHICH candidate --hard-guard protects')
    ap.add_argument("--hard1", action="store_true",
                    help='with --soft-demote: ALSO hard-demote the single strongest Z doc (recovers T1 without losing T3)')
    ap.add_argument("--relative", action="store_true",
                    help='with --soft-demote: RELATIVE penalty — demote a doc only where per-token MaxSim(Z) EXCEEDS per-token MaxSim(query)')
    ap.add_argument("--hard-auto", action="store_true",
                    help='ADAPTIVE hard-demote count: demote every candidate whose Z-evidence is >= --hard-frac of the strongest (one on a single-exclusion query, one per topic on compound T3), instead of a fixed --hard-m')
    ap.add_argument("--hard-frac", type=float, default=0.6, help='evidence fraction for --hard-auto (adopted: 0.6)')
    ap.add_argument("--two-stage", action="store_true",
                    help="TWO-STAGE penalty: an absolute floor decides WHETHER to act (safety, keeps the no-harm property and no-harm), then the penalty is re-centred on the shortlist's own Z-distribution (effectiveness, restores spread on corpora that are homogeneous in Z)")
    ap.add_argument("--ts-floor", type=float, default=0.35,
                    help='stage-1 absolute floor: act only if max_i S(Z,d_i) >= this')
    ap.add_argument("--ts-q", type=float, default=0.75,
                    help='stage-2 percentile of the shortlist Z-distribution used as the effective threshold')
    ap.add_argument("--gap-gate", action="store_true",
                    help='NON-PARAMETRIC Stage-2 cut: split the Z-evidence at its LARGEST GAP instead of at mu+sigma*std')
    ap.add_argument("--anomaly-gate", action="store_true")
    ap.add_argument("--anomaly-sigma", type=float, default=0.5)
    ap.add_argument("--xgate", action="store_true",
                    help='2-D AND-gate: hard-drop only Z-docs that are NOT positively-relevant (qpt low tail)')
    ap.add_argument("--xgate-q", type=float, default=0.5, help='qpt quantile for --xgate (drop docs with qpt <= this)')
    ap.add_argument("--zanchor", action="store_true",
                    help="Exclusion PRF: augment demotion evidence with doc-doc similarity to the shortlist's confident Z-docs (relational signal for confusable/dense corpora)")
    ap.add_argument("--zanchor-beta", type=float, default=1.0, help='weight of the Z-anchor signal in the combined evidence')
    ap.add_argument("--zanchor-m", type=int, default=3, help='number of Z-anchor docs')
    ap.add_argument("--gap-evidence", action="store_true",
                    help='combine MaxSim(Z,doc) with the FROZEN-minus-ADAPTED score gap as a second, detector-independent exclusion signal')
    ap.add_argument("--zmask-q", action="store_true",
                    help="zero the query-token weights at the detector's Z span in the relevance term")
    ap.add_argument("--zmask-soft", action="store_true",
                    help='with --zmask-q, weight each query token by 1-P(Z) instead of zeroing the decoded span')
    ap.add_argument("--gap-beta", type=float, default=0.25,
                    help='weight on the score-gap term')
    ap.add_argument("--score-gap", action="store_true",
                    help='SCORE-GAP ANOMALY DEMOTION: when the finder detects exclusion intent (has_intent=True) but cannot extract an explicit Z string, use the z-normalised gap between FROZEN and LoRA candidate scores as the demotion signal')
    ap.add_argument("--score-gap-sigma", type=float, default=1.0,
                    help='anomaly sigma for score-gap demotion (default 1.0)')
    ap.add_argument("--score-gap-intent-thr", type=float, default=0.3,
                    help='fire_score threshold for intent detection when Z is empty (default 0.3)')
    ap.add_argument("--hard-cap", type=int, default=3, help='max hard demotions under --hard-auto')
    ap.add_argument("--reembed-cache-gb", type=float, default=None,
                    help='memory budget (GB) for the query-time re-embed cache')
    ap.add_argument("--min-hard-n", type=int, default=None,
                    help="minimum shortlist size for the HARD drop's fraction-of-max rule (default 4)")
    ap.add_argument("--xgate-strict", action="store_true",
                    help='apply the positive-intent gate even when ONE candidate is selected, and allow the hard-drop set to be empty')
    ap.add_argument("--adapt-query", action="store_true",
                    help='with --no-reembed: put the adapter on the QUERY and on Z, score against the FROZEN index')
    ap.add_argument("--query-only", action="store_true",
                    help='QUERY-ONLY FINE-TUNE baseline: apply the adapter to the query only and score against the FROZEN document index (no re-encode, no re-index)')
    ap.add_argument("--learned-x", action="store_true",
                    help='take the POSITIVE span from a 5-class detector (extract_XZ) instead of deriving it as query-minus-Z')
    ap.add_argument("--adaptive-lam", action="store_true",
                    help="per-query demotion strength with no fitted constant: lam_eff = std(rel) / (scale*std(evidence)), so the penalty spread matches this shortlist's own relevance spread")
    ap.add_argument("--mask-z", action="store_true",
                    help='score relevance on the positive query q_X, with the excluded tokens stripped')
    ap.add_argument("--z-file", default="", help='JSONL of {query, Z}')
    ap.add_argument("--z-pool", default="mean", choices=["mean", "topm", "max"],
                    help='how to pool the per-Z-token evidence')
    ap.add_argument("--z-pool-m", type=int, default=2, help='m for --z-pool topm')
    ap.add_argument("--adaptive-demote", action="store_true",
                    help='BEST-OF-BOTH: auto-pick hard demote-1 (single strong Z-target -> keeps T1) vs soft global penalty (>=2 strong Z-targets -> handles T3), by counting comparably-strong Z-docs')
    ap.add_argument("--demote-tau", type=float, default=0.0,
                    help='GUARD A (no-harm): demote only if MaxSim(Z,doc).max() >= tau * top-relevance')
    ap.add_argument("--topk", type=int, default=None, help='candidates re-embedded (default config REEMBED_TOPK)')
    ap.add_argument("--no-two-stage", action="store_true",
                    help='ablation: disable the two-stage penalty even when --two-stage is in $BASE')
    ap.add_argument("--no-soft-penalty", action="store_true",
                    help='ablation: turn the CONTINUOUS soft penalty off and leave the hard drop as the only demotion')
    ap.add_argument("--no-hard-auto", action="store_true",
                    help='ablation: turn the ADAPTIVE hard-demotion count off so --hard-m applies, giving a FIXED count')
    ap.add_argument("--no-anomaly-gate", action="store_true",
                    help='ablation: disable the anomaly gate even when --anomaly-gate is in $BASE')
    ap.add_argument("--no-demote", action="store_true", help='ablation: re-embed only, never demote')
    ap.add_argument("--no-reembed", action="store_true",
                    help='E1 arm ours-frozen-demote: NO per-query re-embed — demote on the frozen candidate embeddings (only query+Z encoded)')
    ap.add_argument("--no-switch-gate", dest="switch_gates_reembed", action="store_false", default=True,
                    help='DISABLE the default Switch-gated re-embed, i.e')
    ap.add_argument("--first-stage", choices=["index", "frozen"], default="index",
                    help="with --reindex: encode the first-stage query with the encoder that built the index (index, default) or with the frozen encoder (frozen)")
    ap.add_argument("--reindex", action="store_true",
                    help='build the first-stage index with the tuned encoder (FINE-TUNE baselines only)')
    ap.add_argument("--faithful", action="store_true",
                    help='re-embed the top-K candidate texts at query time (shipped behaviour, ~half the memory')
    ap.add_argument("--device", default="cuda:0")
    return ap


def resolved_system(a):
    """The (reembed, finder, thr) triple this run will ACTUALLY use, after config + env resolution."""
    thr = a.thr if a.thr is not None else C.FINDER_THR
    return {"reembed_lora": a.reembed_lora or "", "finder": a.finder or "", "thr": float(thr),
            "thr_source": "--thr" if a.thr is not None else
                          ("env:EXCISE_FINDER_THR" if os.environ.get("EXCISE_FINDER_THR") else "config-default"),
            "finder_source": "--finder" if "--finder" in sys.argv else
                             ("env:EXCISE_FINDER_CKPT" if os.environ.get("EXCISE_FINDER_CKPT") else "config-default"),
            "reindex": bool(a.reindex), "first_stage": str(getattr(a, "first_stage", "index")), "faithful": bool(a.faithful),
            "chunk_words": int(os.environ.get("EXCISE_CHUNK_WORDS", "0")),
            # The DOCUMENT BUDGET: chunk_words alone cannot identify it, and equal-budget comparisons are
            # exactly the case where a cell must prove which budget produced it.
            "max_doc_words": int(os.environ.get("EXCISE_MAX_DOC_WORDS", "0")),
            "query_len": os.environ.get("EXCISE_QUERY_LEN", ""), "doc_len": os.environ.get("EXCISE_DOC_LEN", ""),
            "sigma": float(a.anomaly_sigma), "lam": (float(a.lam) if a.lam is not None else float(C.DEMOTE_LAM)),
            "ts_floor": float(a.ts_floor), "hard_frac": float(a.hard_frac), "hard_cap": int(a.hard_cap),
            "two_stage": bool(getattr(a, "two_stage", False)),
            "anomaly_gate": bool(getattr(a, "anomaly_gate", False)),
            "soft_demote": bool(getattr(a, "soft_demote", False)),
            "no_demote": bool(getattr(a, "no_demote", False)),
            "no_reembed": bool(getattr(a, "no_reembed", False)),
            "switch_gate": bool(getattr(a, "switch_gates_reembed", True)),
            "hard1": bool(getattr(a, "hard1", False)),
            "hard_auto": bool(getattr(a, "hard_auto", False)),
            "hard_guard": bool(getattr(a, "hard_guard", False)),
            "guard_rule": str(getattr(a, "hard_guard_rule", "top_rel")),
            "min_hard_n": (None if getattr(a, "min_hard_n", None) is None else int(a.min_hard_n)),
            "adapt_query": bool(getattr(a, "adapt_query", False)),
            "xgate": bool(getattr(a, "xgate", False)),
            "xgate_strict": bool(getattr(a, "xgate_strict", False)),
            "reembed_cache_gb": (None if getattr(a, "reembed_cache_gb", None) is None else float(a.reembed_cache_gb)),
            "demote_tau": float(a.demote_tau), "gap_gate": bool(getattr(a, "gap_gate", False)),
            "gap_evidence": bool(getattr(a, "gap_evidence", False)),
            "gap_beta": float(getattr(a, "gap_beta", 0.25)),
            "zmask_q": bool(getattr(a, "zmask_q", False)),
            "hard_m": int(getattr(a, "hard_m", 1)),
            "no_hard_auto": bool(getattr(a, "no_hard_auto", False)),
            "topk": (None if getattr(a, "topk", None) is None else int(a.topk)),
            "relative": bool(getattr(a, "relative", False)),
            "signed_op": bool(getattr(a, "signed_op", False)),
            "lexical_neg": bool(getattr(a, "lexical_neg", False)),
            "adaptive_demote": bool(getattr(a, "adaptive_demote", False)),
            "zanchor": bool(getattr(a, "zanchor", False)),
            "ts_q": float(getattr(a, "ts_q", 0.75)),
            "z_pool": str(getattr(a, "z_pool", "mean")),
            "z_pool_m": int(getattr(a, "z_pool_m", 2)),
            "mask_z": bool(getattr(a, "mask_z", False)),
            "adaptive_lam": bool(getattr(a, "adaptive_lam", False)),
            "learned_x": bool(getattr(a, "learned_x", False)),
            "query_only": bool(getattr(a, "query_only", False))}


def _guard_explicit_system(a):
    """FAIL CLOSED if the system under test was left to config defaults."""
    if os.environ.get("EXCISE_ALLOW_DEFAULT_CKPT") == "1":
        return
    missing = [f for f in ("--reembed-lora", "--finder") if f not in sys.argv]
    if missing:
        raise SystemExit(
            f"[arm] REFUSING TO RUN: {' and '.join(missing)} not passed explicitly, so the run would fall back to\n"
            f"      config defaults (reembed={C.REEMBED_LORA}\n"
            f"                       finder={C.FINDER_CKPT}\n"
            f"                       thr={C.FINDER_THR}) rather than naming the system in argv.\n"
            f"      Pass them explicitly (use \"\" for the frozen/fine-tune arms). See excise/config.py.\n"
            f"      Set EXCISE_ALLOW_DEFAULT_CKPT=1 to run on the config defaults.")
    if a.finder and a.thr is None:
        raise SystemExit(
            f"[arm] REFUSING TO RUN: --finder given without --thr, so the fire threshold would resolve to "
            f"{C.FINDER_THR} (config/env), not the calibrated operating point. The shipped detector is "
            f"the one excise/adopted.json names, at the tau in its thr.txt. Pass --thr explicitly.")


def _print_effective_config(a):
    """Print the operative knobs on every run."""
    import sys as _sys
    try:
        print("[excise] guard_rule={} hard_frac={} hard_cap={} sigma={} ts_floor={} lam={} thr={}".format(
            getattr(a, "hard_guard_rule", "?"), getattr(a, "hard_frac", "?"), getattr(a, "hard_cap", "?"),
            getattr(a, "anomaly_sigma", "?"), getattr(a, "ts_floor", "?"),
            a.lam if getattr(a, "lam", None) is not None else C.DEMOTE_LAM,
            a.thr if getattr(a, "thr", None) is not None else C.FINDER_THR),
            file=_sys.stderr, flush=True)
    except Exception:
        pass          # a banner must never be able to fail a run


def _guard_hard_demotion(a):
    """Turn the guard ON whenever hard demotion is on, unless it is explicitly ablated."""
    if getattr(a, "no_hard_demote", False):
        a.hard1 = a.hard_auto = a.hard_guard = False
        print("[excise] hard demotion OFF (--no-hard-demote) -- the ablation's control arm: soft penalty "
              "only, no hard rule, no guard.", flush=True)
        return a
    hard = getattr(a, "hard1", False) or getattr(a, "hard_auto", False)
    if getattr(a, "no_hard_guard", False):
        a.hard_guard = False
        if hard:
            print("[excise] hard demotion WITHOUT the guard -- an ablation, measured worse than no hard "
                  "demotion at all (892 against 896 on the selection split)", flush=True)
    elif hard and not getattr(a, "hard_guard", False):
        a.hard_guard = True
        print("[excise] hard demotion is on, so the guard is on: without it the rule scores below not "
              "using it. Pass --no-hard-guard to ablate.", flush=True)
    return a


def _apply_ablation_negations(a):
    """Subtract a stage that $BASE forced on, so an ablation arm can actually differ from the shipped one."""
    if getattr(a, "no_two_stage", False):
        a.two_stage = False
        print("[excise] two-stage penalty OFF (--no-two-stage) -- ablation arm", flush=True)
    if getattr(a, "no_anomaly_gate", False):
        a.anomaly_gate = False
        print("[excise] anomaly gate OFF (--no-anomaly-gate) -- ablation arm", flush=True)
    if getattr(a, "no_soft_penalty", False):
        a.soft_demote = False
        print("[excise] soft penalty OFF (--no-soft-penalty) -- hard drop is the only demotion, ablation arm",
              flush=True)
    if getattr(a, "no_hard_auto", False):
        a.hard_auto = False
        print(f"[excise] adaptive hard-demotion count OFF (--no-hard-auto) -- FIXED count of "
              f"{getattr(a, 'hard_m', 1)} (--hard-m), ablation arm", flush=True)
    return a


def build_method(a):
    a = _apply_ablation_negations(a)
    a = _guard_hard_demotion(a)
    _print_effective_config(a)
    """Construct the ExciseMethod for the selected arm, and print the arm so no run is ambiguous in a log."""
    from excise.exclusion import ExciseMethod
    _guard_explicit_system(a)
    if a.reindex and not a.reembed_lora:
        raise SystemExit("--reindex with no --reembed-lora is a no-op (nothing to re-index with)")
    if a.no_demote and a.finder:
        print("[arm] --no-demote: finder consulted for the Switch, but the demotion is never applied", flush=True)
    gated = a.switch_gates_reembed and bool(a.finder)     # inert without a Finder -- see ExciseMethod.rank()
    if a.switch_gates_reembed and not a.finder:
        print("[arm] no --finder: the Switch gate is INERT (a finderless gate would silently return the frozen "
              "ranking for every query, turning a fine-tune baseline into the frozen arm)", flush=True)
    name = ("frozen" if not a.reembed_lora else ("reindexed-finetune" if a.reindex else "candidate-reembed")) + \
           ("+finder-demote" if (a.finder and not a.no_demote) else "") + \
           ("+switch-gated" if gated else ("+ungated-reembed" if a.finder else ""))
    print(f"[arm] {name}  reembed={a.reembed_lora or '<vanilla>'}  finder={a.finder or '<none>'}  "
          f"thr={a.thr if a.thr is not None else C.FINDER_THR} "          # `a.thr or ...` would misprint thr=0.0
          f"lam={a.lam if a.lam is not None else C.DEMOTE_LAM} "
          f"topk={a.topk or C.REEMBED_TOPK}", flush=True)
    print(f"[provenance] {resolved_system(a)}", flush=True)
    M = ExciseMethod(reembed_lora=a.reembed_lora, finder_ckpt=a.finder, device=a.device, thr=a.thr)
    M.first_stage = getattr(a, "first_stage", "index")
    if getattr(a, "reembed_cache_gb", None) is not None:
        M._rc_budget = int(float(a.reembed_cache_gb) * 1e9)
    print(f"[arm] re-embed cache budget {M._rc_budget/1e9:.2f} GB"
          + (" (disabled)" if M._rc_budget <= 0 else ""), flush=True)
    return M


def rank_fn(M, a):
    zmap = None
    if getattr(a, "z_file", ""):
        import json as _json
        zmap = {}
        for _l in open(a.z_file):
            _r = _json.loads(_l)
            zmap[_r["query"].strip()] = (_r.get("Z") or "").strip()
        print(f"[z-file] {len(zmap)} injected Z spans from {a.z_file}", flush=True)
    # override_Z=None means "use the finder", which is every existing arm.
    def _oz(q):
        return None if zmap is None else zmap.get(q.strip(), "")
    return lambda q: M.rank(q, override_Z=_oz(q), topk=a.topk, lam=a.lam, demote=not a.no_demote,
                            switch_gate=a.switch_gates_reembed, reembed=not a.no_reembed,
                            demote_tau=getattr(a, "demote_tau", 0.0),
                            soft_demote=getattr(a, "soft_demote", False),
                            adaptive_demote=getattr(a, "adaptive_demote", False),
                            signed_op=getattr(a, "signed_op", False),
                            lexical_neg=getattr(a, "lexical_neg", False),
                            hard1=getattr(a, "hard1", False),
                            hard_m=getattr(a, "hard_m", 1),
                            hard_guard=getattr(a, "hard_guard", False),
                            guard_rule=getattr(a, "hard_guard_rule", "top_rel"),
                            relative=getattr(a, "relative", False),
                            query_only=getattr(a, "query_only", False),
                            adapt_query=getattr(a, "adapt_query", False),
                            hard_auto=getattr(a, "hard_auto", False),
                            hard_frac=getattr(a, "hard_frac", 0.6),
                            hard_cap=getattr(a, "hard_cap", 3),
                            **({} if getattr(a, "min_hard_n", None) is None
                               else {"min_hard_n": int(a.min_hard_n)}),
                            two_stage=getattr(a, "two_stage", False),
                            ts_floor=getattr(a, "ts_floor", 0.35),
                            ts_q=getattr(a, "ts_q", 0.75),
                            anomaly_gate=getattr(a, "anomaly_gate", False),
                            anomaly_sigma=getattr(a, "anomaly_sigma", 0.5),
                            gap_gate=getattr(a, "gap_gate", False),
                            xgate=getattr(a, "xgate", False),
                            xgate_strict=getattr(a, "xgate_strict", False),
                            xgate_q=getattr(a, "xgate_q", 0.5),
                            zanchor=getattr(a, "zanchor", False),
                            zanchor_beta=getattr(a, "zanchor_beta", 1.0),
                            zanchor_m=getattr(a, "zanchor_m", 3),
                            gap_evidence=getattr(a, "gap_evidence", False),
                            gap_beta=getattr(a, "gap_beta", 0.25),
                            zmask_q=getattr(a, "zmask_q", False),
                            zmask_soft=getattr(a, "zmask_soft", False),
                            score_gap=getattr(a, "score_gap", False),
                            score_gap_sigma=getattr(a, "score_gap_sigma", 1.0),
                            score_gap_intent_thr=getattr(a, "score_gap_intent_thr", 0.3),
                            mask_z=getattr(a, "mask_z", False),
                            adaptive_lam=getattr(a, "adaptive_lam", False),
                            learned_x=getattr(a, "learned_x", False),
                            z_pool=getattr(a, "z_pool", "mean"),
                            z_pool_m=getattr(a, "z_pool_m", 2),
                            blend=getattr(a, "blend", False),
                            blend_beta=getattr(a, "blend_beta", 0.5),
                            z_veto=getattr(a, "z_veto", 0),
                            z_hard=getattr(a, "z_hard", 0))


def index_kwargs(a):
    return dict(precompute_lora=not a.faithful, index_with_lora=a.reindex)
