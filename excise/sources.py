"""Read the corpora (data/datasets/, checked against data/benchmark/CORPUS_REQUIRED.json) and the BEIR suites."""
from __future__ import annotations
import json, os, sys
# the repository root, so `config` and the packages import from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from excise import config as C

NEGC_DIR = os.path.join(C.ROOT, "data", "sources", "negconstraint", "NegConstraint")
BEIR = {"fiqa", "trec-covid", "nfcorpus", "scifact", "arguana", "quora", "scidocs"}
ALL_SOURCES = ["excluir", "negconstraint", "sustainableqa", "edgar", "eurlex", "esgenius"] + sorted(BEIR)


def canonical(source):
    """Map a record's source tag to the corpus it lives in."""
    s = str(source)
    return "excluir" if s.startswith("excluir") else s


LOCAL = os.path.join(C.ROOT, "data", "datasets")




def _local_corpus(s):
    """Read the self-contained export if present. Preserves ROW ORDER, which is the contract for ExcluIR"""
    p = os.path.join(LOCAL, s, "corpus.jsonl")
    if not os.path.exists(p): return None
    ids, docs = [], []
    with open(p, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            if "doc" not in r:
                print(f"[sources] {p}: stale export (no `doc` field) — ignoring it; "
                      f"download it again with `python scripts/download.py --what corpora`", flush=True)
                return None
            ids.append(str(r["_id"])); docs.append(r["doc"])
    return ids, docs


def load_source(source, prefer_local=True):
    """Return (ids: list[str], docs: list[str]) — index-aligned."""
    s = canonical(source)
    if prefer_local:
        got = _local_corpus(s)
        if got: return got
    if s == "excluir":
        raise FileNotFoundError(
            "excluir corpus not found under data/datasets/excluir/. Materialise it; there is no fallback, "
            "because the only other copy is another project's and evaluating against it would be wrong "
            "in a way no number would reveal.")
    if s == "negconstraint":
        rows = [json.loads(l) for l in open(os.path.join(NEGC_DIR, "corpus.jsonl"))]
        return [str(r["_id"]) for r in rows], [((r.get("title") or "") + "\n" + r["text"]).strip() for r in rows]
    if s in BEIR:
        from datasets import load_dataset
        Cd = load_dataset(f"BeIR/{s}", "corpus")["corpus"]
        return [str(r["_id"]) for r in Cd], [((r.get("title") or "") + "\n" + (r.get("text") or "")).strip() for r in Cd]
    raise ValueError(f"unknown source '{source}' (known: {ALL_SOURCES})")


def id2idx(ids):
    return {c: i for i, c in enumerate(ids)}


def local_part(source, part):
    """Rows of `data/datasets/<source>/<part>.jsonl`, or None. The export is the single place every stage reads."""
    p = os.path.join(LOCAL, canonical(source), f"{part}.jsonl")
    if not os.path.exists(p): return None
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def load_queries(source, limit=20000):
    """REAL user queries for a source, or [] if it has none available offline."""
    s = canonical(source)
    rows = local_part(s, "queries")
    if rows is not None:
        return [r["text"] for r in rows if r.get("text")][:limit]
    try:
        if s == "negconstraint":
            rows = [json.loads(l) for l in open(os.path.join(NEGC_DIR, "queries.jsonl"))]
            return [r["text"] for r in rows if r.get("text")][:limit]
        if s in BEIR:
            from datasets import load_dataset
            Q = load_dataset(f"BeIR/{s}", "queries")["queries"]
            return [r["text"] for r in Q if r.get("text")][:limit]
    except Exception as e:                          # offline / not cached -> caller falls back to templates
        print(f"  [sources] no real queries for '{s}' ({type(e).__name__}); falling back to templates", flush=True)
    return []                                        # excluir queries ARE exclusion queries — never no-fire rows


def excluir_reserved():
    """Documents reserved from mining. Empty here: the file that defined the reservation belonged to a"""
    return set()
