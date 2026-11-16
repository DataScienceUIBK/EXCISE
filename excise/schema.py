"""The X-BENCH record format, the negation cues and the token helpers shared by the detector and the operator."""
from __future__ import annotations
import re
from dataclasses import dataclass, asdict

TIERS = ("T1_explicit", "T2_implicit", "T3_complex")
NEG_TYPES = ("entity", "attribute", "compound")


@dataclass
class Record:
    id: str
    tier: str                       # one of TIERS
    query: str
    X: str                          # wanted topic
    Z: list                         # excluded topic(s); >1 => compound/multi-Z
    pos_id: object                   # corpus id of p (X∈p ∧ Z∉p) — VERBATIM (str or int; never int-coerced)
    neg_ids: list                    # corpus ids of confusable n's, ONE PER Zi: neg_ids[k] holds Z[k]
    length: str = "short"           # short | long
    source: str = ""                # excluir | trec-covid | fiqa | ...

    def dict(self):
        return asdict(self)


def content_tokens(s, stop=frozenset("the a an of and or not but is are was were to in on for with about from as at by this that".split())):
    return [w for w in re.split(r"[^a-z0-9]+", (s or "").lower()) if len(w) > 1 and w not in stop]


def fire_words(query, X, Z_list):
    """Finder label = Z-content tokens that are NOT X-content, restricted to whole words present in the query."""
    xc = set(content_tokens(X)); zc = set()
    for z in Z_list: zc |= set(content_tokens(z))
    fire = zc - xc                                        # Z-content that is NOT also X-content
    qtoks = set(t.lower() for t in re.split(r"[^A-Za-z0-9]+", query) if t)
    return sorted(fire & qtoks)


NEG_CUES = (r"\bnot\b", r"\bwithout\b", r"\bexcept\b", r"\bexcluding\b", r"\bexclude[sd]?\b", r"\bother than\b",
            r"\bapart from\b", r"\baside from\b", r"\bbesides\b", r"\bneither\b", r"\bnor\b", r"\bnon-",
            r"\bomit(?:s|ted|ting)?\b", r"\brather than\b", r"\bminus\b", r"[a-z]-free\b",
            r"\bnicht\b", r"\bkein(?:e|er|es|en)?\b", r"\bau(?:ss|\u00df)er\b", r"\babgesehen von\b",
            r"\bohne\b", r"\bsauf\b", r"\bhormis\b", r"\bexcepto\b", r"\bsalvo\b", r"\btranne\b")
_CUE_RE = re.compile("|".join(NEG_CUES), re.I)


def has_neg_cue(query) -> bool:
    """True iff the query states an exclusion with an explicit lexical cue."""
    return bool(_CUE_RE.search(query or ""))


def structural_ok(r: Record) -> tuple[bool, str]:
    """Cheap structural contract (NOT semantic — that's the judge). Returns (ok, reason)."""
    if r.tier not in TIERS: return False, "bad_tier"
    if not r.query or not r.X or not r.Z: return False, "empty_field"
    if len(r.neg_ids) < len(r.Z): return False, "missing_neg_per_Z"
    if len(set(map(str, r.neg_ids))) != len(r.neg_ids): return False, "duplicate_neg_ids"
    if str(r.pos_id) in set(map(str, r.neg_ids)): return False, "pos_in_neg"   # compare as str: 7 vs "7" is one doc
    if r.tier == "T3_complex" and len(r.Z) < 2: return False, "compound_needs_2Z"
    if len(r.Z) > 1 and len({str(z).strip().lower() for z in r.Z}) != len(r.Z): return False, "duplicate_Z"
    if r.tier == "T1_explicit" and not has_neg_cue(r.query): return False, "explicit_tier_has_no_cue"
    if r.tier == "T2_implicit" and has_neg_cue(r.query): return False, "implicit_tier_has_neg_cue"
    if r.tier == "T3_complex" and not has_neg_cue(r.query): return False, "complex_tier_has_no_explicit_cue"
    # Lexical Z-in-query is valid ONLY for the verbatim tier (T1 explicit). T2 (implicit) and T3 (mixed
    # explicit+implicit) paraphrase Z by design -> they are verified SEMANTICALLY by the judge, never lexically.
    if r.tier == "T1_explicit" and not fire_words(r.query, r.X, r.Z):
        return False, "Z_not_in_query"
    if r.length not in ("short", "long"): return False, "bad_length"
    return True, "ok"
