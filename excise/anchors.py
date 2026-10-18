"""Loaders for the external query suites with human labels (the ordinary-query and Boolean suites)."""
from __future__ import annotations
import glob, json, os, sys
# the repository root, so `config` and the packages import from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from excise import config as C

NEGC_DIR = os.path.join(C.ROOT, "data", "sources", "negconstraint", "NegConstraint")
FOLLOWIR = ("robust04", "core17", "news21")
LOCAL = os.path.join(C.ROOT, "data", "datasets")
# The control suites ship INSIDE the benchmark (queries + human qrels), because a run is scored on them.
# _local() searches here too, so a clean checkout finds them without materialising anything first.
CONTROL = [os.path.join(C.ROOT, "data", "benchmark", "control", "noharm"),
           os.path.join(C.ROOT, "data", "benchmark", "control", "boolean")]


def _local(name, *parts):
    """Read the self-contained export, or None if it has not been built yet. One place, plain jsonl, no"""
    for root in [LOCAL] + CONTROL:
        ps = [os.path.join(root, name, f"{p}.jsonl") for p in parts]
        if all(os.path.exists(x) for x in ps):
            return tuple(_jsonl(x) for x in ps)
    return None


def _hub_snapshot(repo):
    """Newest materialised snapshot dir for a hub repo, or None."""
    d = os.path.join(C.HF_HOME, "hub", "datasets--" + repo.replace("/", "--"), "snapshots")
    snaps = sorted(glob.glob(os.path.join(d, "*")), key=os.path.getmtime, reverse=True)
    return snaps[0] if snaps else None


def _jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_boolquestions():
    """The boolean-IR anchor: (corpus, queries). Each query has pos_ids/neg_ids and a question_type in"""
    got = _local("boolquestions", "corpus", "queries")
    if got: return got
    raise FileNotFoundError("boolquestions is not in data/datasets/")


def load_boolq(split="validation"):
    """BoolQ: boolean questions; the `not`-bearing subset is the explicit-NOT anchor."""
    got = _local("boolq", "rows")
    if got: return [r for r in got[0] if r.get("split") == split]
    from datasets import load_dataset
    return list(load_dataset("google/boolq")[split])


def load_nevir():
    """NevIR: contrastive PAIRS where the negation flips relevance. Doc-side; a footnote, not a headline —"""
    got = _local("nevir", "rows")
    if got:
        out = {}
        for r in got[0]: out.setdefault(r.get("split", "test"), []).append(r)
        return out
    from datasets import load_dataset
    d = load_dataset("orionweller/NevIR")
    return {k: list(v) for k, v in d.items()}


def load_instructir():
    """InstructIR: instruction-following retrieval. NO-HARM anchor (the instructions are positive, not"""
    got = _local("instructir", "corpus", "queries", "qrels")
    if got: return got
    from datasets import load_dataset
    corpus = list(load_dataset("kaist-ai/InstructIR", "corpus")["corpus"])
    queries = list(load_dataset("kaist-ai/InstructIR", "queries")["queries"])
    qrels = list(load_dataset("kaist-ai/InstructIR", "qrels")["test"])
    return corpus, queries, qrels


def load_followir(name="robust04"):
    """FollowIR (`{name}-instructions`): long instructions, p-MRR over an ORIGINAL vs a CHANGED qrel set."""
    if name not in FOLLOWIR: raise ValueError(f"followir set must be one of {FOLLOWIR}")
    d = os.path.join(LOCAL, f"followir-{name}")
    if os.path.isdir(d):
        return {os.path.basename(f)[:-6]: _jsonl(os.path.join(d, f))
                for f in sorted(os.listdir(d)) if f.endswith(".jsonl")}
    snap = _hub_snapshot(f"jhu-clsp/{name}-instructions")
    if not snap: raise FileNotFoundError(f"jhu-clsp/{name}-instructions not in the hub cache")
    out = {"corpus": _jsonl(os.path.join(snap, "corpus.jsonl")),
           "queries": _jsonl(os.path.join(snap, "queries.jsonl")),
           "top_ranked": _jsonl(os.path.join(snap, "top_ranked.jsonl")),
           "qrels_changed": _jsonl(os.path.join(snap, "qrels_changed", "test.jsonl"))}
    og = os.path.join(snap, "qrels_og", "test.jsonl")
    if os.path.exists(og): out["qrels_og"] = _jsonl(og)
    return out


def pmrr_ready(name):
    """Can p-MRR be computed offline for this FollowIR set? (Needs the ORIGINAL qrels, not just the changed ones.)"""
    if os.path.exists(os.path.join(LOCAL, f"followir-{name}", "qrels_og.jsonl")): return True
    snap = _hub_snapshot(f"jhu-clsp/{name}-instructions")
    return bool(snap) and os.path.exists(os.path.join(snap, "qrels_og", "test.jsonl"))


def load_negconstraint():
    """NegConstraint: 366 human explicit-negation queries. NOTE: ships 198 labelled POSITIVES and ZERO labelled"""
    got = _local("negconstraint", "corpus", "queries", "qrels")
    if got: return got
    return (_jsonl(os.path.join(NEGC_DIR, "corpus.jsonl")),
            _jsonl(os.path.join(NEGC_DIR, "queries.jsonl")), [])



AVAILABILITY = {
    "NegConstraint":    (load_negconstraint,    "explicit, human (Right-Rank)"),
    "BoolQuestions":    (load_boolquestions,   "boolean IR: explicit NOT (pairwise acc by and/or/not)"),
    "BoolQ":            (load_boolq,            "yes/no QA — context only, NOT the boolean-IR anchor"),
    "NevIR":            (load_nevir,            "doc-side boundary footnote (pair-acc)"),
    "InstructIR":       (load_instructir,       "NO-HARM (delta nDCG ~ 0)"),
    "FollowIR":         (load_followir,         "explicit-long (p-MRR)"),
}


def probe():
    """(name, ok, detail) for each anchor — used by gen/preflight.py so 'missing' is never a wrong call site."""
    rows = []
    for nm, (fn, role) in AVAILABILITY.items():
        try:
            r = fn()
            n = sum(len(x) for x in r.values()) if isinstance(r, dict) else \
                sum(len(x) for x in r) if isinstance(r, tuple) else len(r)
            rows.append((nm, True, f"{n} rows — {role}"))
        except Exception as e:
            rows.append((nm, False, f"{type(e).__name__}: {str(e)[:60]} — {role}"))
    pm = [n for n in FOLLOWIR if pmrr_ready(n)]
    rows.append(("FollowIR p-MRR", len(pm) == len(FOLLOWIR),
                 f"qrels_og present for {pm or 'none'}; missing for {[n for n in FOLLOWIR if n not in pm] or 'none'} "
                 f"(p-MRR needs original AND changed qrels)"))
    return rows


if __name__ == "__main__":
    for nm, ok, detail in probe():
        print(f"  [{'OK  ' if ok else 'MISS'}] {nm}: {detail}")
