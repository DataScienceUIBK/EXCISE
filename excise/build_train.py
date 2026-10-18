"""Build training triples from X-BENCH records, and write and verify the provenance sidecar of each training file."""
from __future__ import annotations
import argparse, json, os, random, sys
from collections import Counter, defaultdict
# the repository root, so `config` and the packages import from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from excise import config as C
from excise.schema import fire_words, has_neg_cue
from excise.sources import load_source, canonical, load_queries



BUILDER = "excise.build_train"

NOHARM_ANCHORS = {"scifact", "nfcorpus", "arguana", "instructir",
                  "followir-robust04", "followir-core17", "followir-news21", "nevir", "boolquestions"}


def sha256_file(path, buf=1 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while (b := f.read(buf)): h.update(b)
    return h.hexdigest()


def write_provenance(path, rows, args, stats):
    """A sidecar `<path>.meta.json` naming the builder, the record files, the split files, and the sha256 of the"""
    meta = {"builder": BUILDER, "rows": len(rows), "sha256": sha256_file(path),
            "records": sorted(args.records), "splits": sorted(args.splits),
            "seed": C.SEED, "twin_frac": C.TWIN_FRAC, "plainneg_frac": C.FINDER_PLAINNEG_FRAC,
            "stats": dict(stats)}
    with open(path + ".meta.json", "w") as f: json.dump(meta, f, indent=2)
    print(f"  provenance -> {path}.meta.json  (sha256 {meta['sha256'][:12]}…)")


def read_provenance(path, require=True):
    """Load and VERIFY `<path>.meta.json`. Returns the meta dict, or raises if the data is unverifiable."""
    mp = path + ".meta.json"
    if not os.path.exists(mp):
        if not require: return None
        raise SystemExit(f"{path}: no provenance sidecar ({mp}).\n"
                         f"  This file was not produced by {BUILDER}. Refusing to train on data of unknown origin.\n"
                         f"  Rebuild:  python -m excise.build_train --records <...> --splits <...>\n"
                         f"  Override: --allow-unverified   (you are asserting you know what this file is)")
    meta = json.load(open(mp))
    got = sha256_file(path)
    if got != meta.get("sha256"):
        raise SystemExit(f"{path}: sha256 does not match its provenance sidecar — the data changed after it was "
                         f"built. Rebuild rather than train on it.")
    return meta


def _tok(s):
    return {w for w in "".join(c if c.isalnum() else " " for c in (s or "").lower()).split() if len(w) > 1}


def is_trivial(query, doc, thresh=0.80):
    """A query that is a lexical copy of the document it must retrieve teaches nothing about ranking."""
    q, d = _tok(query), _tok(doc)
    if not q: return True
    return len(q & d) / len(q) >= thresh


def real_twins(src, train_pool, rng, cap_per_query=2, max_rows=None):
    """POSITIVE TWINS from real qrels: (real user query, a judged-relevant TRAIN doc, a different TRAIN doc)."""
    from excise.sources import local_part, load_source
    qrels, qrows = local_part(src, "qrels"), local_part(src, "queries")
    if not qrels or not qrows: return []
    ids, docs = load_source(src)
    idx = {c: i for i, c in enumerate(ids)}
    qtext = {str(r["_id"]): r["text"] for r in qrows if r.get("text")}
    pos, judged = {}, {}
    for r in qrels:
        q, d, s = str(r["query-id"]), str(r["corpus-id"]), int(r.get("score", 0))
        judged.setdefault(q, set()).add(d)
        if s > 0 and d in train_pool and d in idx: pos.setdefault(q, []).append(d)

    pool = sorted(train_pool & set(idx))                       # sorted: PYTHONHASHSEED-independent
    if len(pool) < 2: return []
    out, trivial = [], 0
    for q in sorted(pos):                                       # deterministic
        if q not in qtext: continue
        for d in pos[q][:cap_per_query]:                        # cap: one query must not dominate the pool
            p_doc = docs[idx[d]]
            if is_trivial(qtext[q], p_doc): trivial += 1; continue
            for _ in range(8):                                  # bounded search for a valid negative
                n = rng.choice(pool)
                if n not in judged.get(q, ()) and n != d:
                    out.append({"query": qtext[q], "positive": p_doc, "negative": docs[idx[n]],
                                "kind": "twin_real", "source": src})
                    break
            if max_rows and len(out) >= max_rows: break
        if max_rows and len(out) >= max_rows: break
    if trivial: print(f"  [twins] {src}: dropped {trivial} trivial pairs (query is a lexical copy of its doc)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True, help='TRAIN-split, QC-passed record files')
    ap.add_argument("--splits", nargs="+", required=True, help='split_<src>.json files (train pools)')
    ap.add_argument("--out-finder", default=os.path.join(C.OUT, "detector.jsonl"))
    ap.add_argument("--out-reembed", default=os.path.join(C.OUT, "retrieval_triples.jsonl"))
    ap.add_argument("--attribute", default="",
                    help='attribute rows from the build_attribute stage (merged into the finder set')
    a = ap.parse_args()
    rng = random.Random(C.SEED)

    # ---- train pools, keyed by canonical source ----
    pools = {}
    for f in a.splits:
        sp = json.load(open(f)); pools[canonical(sp["source"])] = set(sp["train"])
    recs = [json.loads(l) for f in a.records for l in open(f)]
    by_src = defaultdict(list)
    for r in recs: by_src[canonical(r.get("source", "excluir"))].append(r)

    finder, reembed = [], []
    stats = Counter(); violations = []

    NOFIRE_SOURCES = ["fiqa", "scidocs", "quora", "fever", "dbpedia-entity", "nq", "pubmedqa"]
    PER_SRC_CAP = 3000
    nofire_pool, per_src = [], {}
    for s in NOFIRE_SOURCES:
        if s in NOHARM_ANCHORS: continue                  # safety guard (none are test/no-harm anchors)
        qs = [q for q in load_queries(s) if q.strip() and not has_neg_cue(q)]
        rng.shuffle(qs); qs = qs[:PER_SRC_CAP]
        per_src[s] = len(qs); nofire_pool += qs
    nofire_pool = sorted(set(nofire_pool))
    longfrac = float(os.environ.get("NOFIRE_LONGFRAC", "0"))
    if longfrac > 0:
        lng = [q for q in nofire_pool if len(q.split()) >= 15]
        sht = [q for q in nofire_pool if len(q.split()) < 15]
        rng.shuffle(lng); rng.shuffle(sht)
        mix, li, si = [], 0, 0
        while li < len(lng) or si < len(sht):             # greedy interleave to hold running long-fraction ~ longfrac
            tot = li + si
            if si >= len(sht) or (li < len(lng) and (tot == 0 or li / tot < longfrac)):
                mix.append(lng[li]); li += 1
            else:
                mix.append(sht[si]); si += 1
        nofire_pool = mix
        print(f"[nofire] LENGTH-DEBIASED longfrac={longfrac}: {len(lng)} long(>=15w) + {len(sht)} short")
    else:
        rng.shuffle(nofire_pool)
    print(f"[nofire] pool: {len(nofire_pool)} cue-free queries "
          + " ".join(f"{k}={v}" for k, v in per_src.items()))
    if not nofire_pool:
        print("  !! NO real queries anywhere — every no-fire row will be a TEMPLATE, which is the known failure")

    for src, rs in by_src.items():
        if src not in pools: raise SystemExit(f"no split file for source '{src}' — pass its split_{src}.json")
        pool = pools[src]
        ids, docs = load_source(src)
        idx = {c: i for i, c in enumerate(ids)}
        src_finder, src_reembed = [], []                                # per-source, so balance is order-independent

        for r in rs:
            all_ids = [str(r["pos_id"])] + [str(n) for n in r["neg_ids"]]
            if any(i not in pool for i in all_ids):                     # HARD GUARANTEE: train docs only
                violations.append((r.get("id"), "doc outside TRAIN pool")); continue
            if any(i not in idx for i in all_ids):
                violations.append((r.get("id"), "id not in corpus")); continue

            # ---- FINDER: token labels; needs Z to be locatable in the query ----
            fw = fire_words(r["query"], r["X"], r["Z"])
            if fw:
                src_finder.append({"instruction": r["query"], "X": r["X"], "Z": r["Z"], "fire_words": fw,
                                   "neg_type": r.get("neg_type", "entity"), "source": src})
                stats[f"finder:{r.get('neg_type','entity')}"] += 1
            else:
                stats["finder:dropped_unlabelable"] += 1                # paraphrased beyond lexical recovery

            # ---- RE-EMBED: one contrastive triple per (Zi, n_i) ----
            for n in r["neg_ids"]:
                src_reembed.append({"query": r["query"], "positive": docs[idx[str(r["pos_id"])]],
                                    "negative": docs[idx[str(n)]]})
                stats["reembed:exclusion"] += 1

        if src in NOHARM_ANCHORS:
            raise SystemExit(f"refusing to draw Finder no-fire rows from '{src}': it is a no-harm TEST anchor "
                             f"Training on it would invalidate the no-harm result.")
        n_plain = round(len(src_finder) * C.FINDER_PLAINNEG_FRAC / max(1 - C.FINDER_PLAINNEG_FRAC, 1e-6))
        real_q = nofire_pool[:n_plain]
        nofire_pool = nofire_pool[len(real_q):]                      # consume: never reuse a query across sources
        for q in real_q:
            src_finder.append({"instruction": q, "X": "", "Z": [], "fire_words": [],
                               "neg_type": "plain_real", "source": src})
            stats["finder:plain_real(switch)"] += 1
        if len(real_q) < n_plain:
            # NO TEMPLATE FALLBACK. A templated no-fire row is the known failure mode; emitting one silently is
            # worse than failing. Supply more real cue-free mining-corpus queries, or generate neutral ones.
            raise SystemExit(
                f"source '{src}': need {n_plain} no-fire rows, only {len(real_q)} real queries left in the global "
                f"pool.\n  Refusing to emit templated no-fire rows -- that is exactly what taught the shipped "
                f"Switch 'stay silent on templates'.\n  Options: lower C.FINDER_PLAINNEG_FRAC, add a mining "
                f"corpus with queries, or run the (gated) neutral-query generation.")

        # ---- RE-EMBED: positive twins from REAL qrels (never templated, never trivial) ----------------
        n_twin = round(len(src_reembed) * C.TWIN_FRAC / max(1 - C.TWIN_FRAC, 1e-6))
        tw = real_twins(src, pool, rng, cap_per_query=2, max_rows=n_twin) if n_twin else []
        for t in tw:
            src_reembed.append({"query": t["query"], "positive": t["positive"], "negative": t["negative"]})
            stats["reembed:twin_real"] += 1
        if len(tw) < n_twin:
            stats["reembed:twin_SHORTFALL"] += n_twin - len(tw)
            print(f"  !! source '{src}': {len(tw)}/{n_twin} REAL twins (qrels-backed). No templates will be "
                  f"emitted. TWIN_FRAC is a testable knob -- let the real yield drive it (DATA_CONTRACT the specification).")

        finder += src_finder; reembed += src_reembed

    if a.attribute and os.path.exists(a.attribute):
        attr = [json.loads(l) for l in open(a.attribute, encoding="utf-8")]
        bad = sorted({canonical(r.get("source", "")) for r in attr} & NOHARM_ANCHORS)
        if bad:
            raise SystemExit(f"attribute file {a.attribute} draws from no-harm anchor(s) {bad} "
                             f"refusing to merge; it would leak into finder training.")
        for r in attr: r.setdefault("neg_type", "attribute")
        finder += attr
        stats["finder:attribute"] += len(attr)
        stats["attribute_file_sha"] = sha256_file(a.attribute)[:12]        # provenance: prove WHICH attr file
        print(f"  merged {len(attr)} attribute rows from {a.attribute} (sha {stats['attribute_file_sha']})")
    elif a.attribute:
        print(f"  !! --attribute {a.attribute} not found — finder will have NO attribute rows (the specification wants ~7%)")

    if violations:
        print(f"  !! {len(violations)} records rejected (not in TRAIN pool) — first 3: {violations[:3]}")
    rng.shuffle(finder); rng.shuffle(reembed)
    for path, rows in ((a.out_finder, finder), (a.out_reembed, reembed)):
        with open(path, "w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
        write_provenance(path, rows, a, stats)
    tw = stats["reembed:twin"]; tot = max(len(reembed), 1)
    print(f"[train] finder={len(finder)}  reembed={len(reembed)}  (twins {tw} = {100*tw//tot}%)")
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(stats.items())))
    print("  SWITCH = the finder's plain no-fire rows; GATE = parameter-free (nothing to train).")

    # ---- ZERO-TEMPLATE ASSERTION. Never let a template re-enter silently. -------------------------
    real, tmpl = stats["finder:plain_real(switch)"], stats["finder:plain_tmpl(switch)"]
    tw_real, tw_short = stats["reembed:twin_real"], stats["reembed:twin_SHORTFALL"]
    print(f"\n  [no templates] Finder no-fire : {real} real, {tmpl} templated  (templated MUST be 0)")
    print(f"  [no templates] Re-embed twins : {tw_real} real qrels-backed, shortfall {tw_short}")
    if tmpl: raise SystemExit("a templated row was emitted — this is the known Switch failure mode")
    if tw_real: print(f"  [no trivial]   twins whose query is a lexical copy of its own doc: dropped at source")

    kinds = Counter(r.get("neg_type", "?") for r in finder); nfin = max(len(finder), 1)
    print("\n  [balance] finder neg_type: " + "  ".join(f"{k}={v} ({v/nfin:.0%})" for k, v in sorted(kinds.items())))
    for want, frac in (("attribute", 0.07), ("compound", 0.07)):
        got = kinds.get(want, 0) / nfin
        if got < frac / 2:
            print(f"  !! the specification wants {want} ~{frac:.0%} of finder rows; got {got:.1%}. "
                  + ("run gen.build_attribute --splits ... then pass --attribute" if want == "attribute"
                     else "no T3_complex records were generated"))


if __name__ == "__main__":
    main()
