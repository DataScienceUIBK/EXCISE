#!/usr/bin/env python3
"""Nine test-time interventions on the frozen index, each replacing the plain MaxSim(Z) demotion (Appendix A.3).

Every arm re-scores the frozen top-k shortlist of Reason-ModernColBERT and is scored by exclusion success@10.
The excluded topic Z comes from the record (oracle) or, with --detector-z, from the released detector.

  frozen       the frozen ranking, for reference
  maxsimz      the reference rule: z(base) - z(MaxSim(Z, d)), both normalised over the shortlist
  bm25z        sparse BM25-Z demotion: z(base) - z(BM25 of the Z terms against the candidate text)
  poslex_a     positional-lexical, earliest occurrence: sum over Z terms of idf(t) / log(2 + first_pos(t))
  poslex_b     positional-lexical, frequency: sum over Z terms of idf(t) * log(1 + count(t))
  contrastive  contrastive late interaction: MaxSim(q_X, d) - MaxSim(Z, d), each normalised by query length
  prf          negative-centroid feedback: subtract the mean token vector of the 3 candidates closest to Z
  orthog       subspace orthogonalisation: remove Z's principal direction from query and document vectors
  maskz        token dropping: score with q_X, the query without the Z words
  readout      learned readout gate: a logistic probe on pooled document vectors, fitted on the collection's
               training records (skipped on a collection that has none)
  zcond        Z-conditional re-embedding: re-encode each candidate as "<Z>. <document>"

  python scripts/nine_interventions.py --source excluir                # oracle Z
  python scripts/nine_interventions.py --source fiqa --detector-z      # the detector's Z
"""
from __future__ import annotations
import argparse, glob, json, math, os, re, sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
# the table is measured on Reason-ModernColBERT at its own document settings (excise/run.py BUDGET)
os.environ.setdefault("EXCISE_BACKBONE", "lightonai/Reason-ModernColBERT")
os.environ.setdefault("EXCISE_CHUNK_WORDS", "512")
os.environ.setdefault("EXCISE_MAX_DOC_WORDS", "10000")

import numpy as np, torch
from excise import config as C
from excise.encoder import emb
from excise.maxsim import weighted_maxsim_matrix as maxsim
from excise.exclusion import ExciseMethod, strip_Z
from excise.metrics import success10
from excise.sources import load_source, id2idx

FIXES = ["frozen", "maxsimz", "contrastive", "orthog", "bm25z", "poslex_a", "poslex_b", "prf", "readout", "zcond", "maskz"]


def znorm(x):
    """Shortlist z-normalisation: the scale bridge whenever a penalty of unrelated units is combined with relevance."""
    return (x - x.mean()) / (x.std() + 1e-6)


def zterms(Zs):
    """Content terms of the excluded topic(s), >= 3 characters. A compound query carries two topics; both are used."""
    out = set()
    for Z in Zs:
        out |= {w for w in re.findall(r"[a-z0-9]+", (Z or "").lower()) if len(w) >= 3}
    return out


def per_token(qv, qm, Vk, Mk):
    """MaxSim normalised by query length, so queries and topics of different length are on one scale."""
    return maxsim(qv.float(), qm.float(), Vk, Mk)[0] / float(qm.sum().clamp(min=1.0))


class Sparse:
    """Corpus IDF and BM25. Lengths and IDF are accumulated in one streaming pass; candidates are re-tokenised on demand."""

    def __init__(self, corpus, k1=1.2, b=0.75):
        self.k1, self.b, self.corpus = k1, b, corpus
        df = Counter(); tot = 0
        for d in corpus:
            toks = re.findall(r"[a-z0-9]+", d.lower())
            df.update(set(toks)); tot += len(toks)
        n = len(corpus)
        self.avgdl = (tot / n) if n else 1.0
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def _toks(self, i):
        return re.findall(r"[a-z0-9]+", self.corpus[i].lower())

    def bm25(self, doc_idx, terms):
        toks = self._toks(doc_idx); tf = Counter(toks); dl = len(toks) or 1
        s = 0.0
        for t in terms:
            f = tf.get(t, 0)
            if not f:
                continue
            s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def poslex(self, doc_idx, terms, variant):
        """(a) earliest occurrence: a document about Z names it early. (b) frequency: it names it often."""
        toks = self._toks(doc_idx)
        s = 0.0
        for t in terms:
            if t not in toks:
                continue
            idf = self.idf.get(t, 0.0)
            s += idf / math.log(2 + toks.index(t)) if variant == "a" else idf * math.log(1 + toks.count(t))
        return s


def train_readout(M, records, i2, cap=400):
    """Logistic probe on frozen mean-pooled document vectors, gold against confusable negative, fitted on the
    collection's training records. It is the one arm that is trained; it reads the frozen vectors only."""
    from sklearn.linear_model import LogisticRegression
    pos, neg = [], []
    for r in records[:cap]:
        p, ns = str(r["pos_id"]), [str(x) for x in r["neg_ids"]]
        if p in i2 and ns and ns[0] in i2:
            pos.append(i2[p]); neg.append(i2[ns[0]])
    if not pos or not neg:
        print(f"[readout] no training document of this collection is in the index -- readout arm skipped", flush=True)
        return None

    def pooled(idx):
        V, Mk = M._gather(M.van, idx)
        return ((V * Mk[..., None]).sum(1) / Mk.sum(1, keepdim=True).clamp(min=1)).cpu().numpy()
    X = np.concatenate([pooled(pos), pooled(neg)])
    y = np.concatenate([np.zeros(len(pos)), np.ones(len(neg))])       # 1 = "should be excluded"
    clf = LogisticRegression(max_iter=3000).fit(X, y)
    print(f"[readout] trained on {len(pos)} gold + {len(neg)} negatives; train acc={clf.score(X, y):.3f}", flush=True)
    return clf


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="excluir", help="excluir | fiqa | trec-covid | esgenius | edgar | eurlex")
    ap.add_argument("--fixes", default="all", help="comma-separated arm names, or 'all'")
    ap.add_argument("--topk", type=int, default=C.REEMBED_TOPK)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--detector-z", action="store_true",
                    help="take Z from the released detector, which is what a deployed system has, not from the record")
    ap.add_argument("--finder", default="model/checkpoints/detector_reason/deepfinder.pt")
    ap.add_argument("--thr", type=float, default=0.70)
    ap.add_argument("--dump-perquery", default="",
                    help="write one JSON line per query: its id, tier and the 0/1 success of every arm")
    a = ap.parse_args()
    want = FIXES if a.fixes == "all" else a.fixes.split(",")
    bridge = lambda base_s, pen: znorm(base_s) - znorm(pen)
    dev = "cuda:0"

    files = sorted(glob.glob(f"data/benchmark/test/{a.source}/*.jsonl"))
    recs = [json.loads(l) for f in files for l in open(f)]
    if a.limit:
        recs = recs[:a.limit]
    ids_, docs = load_source(a.source); i2 = id2idx(ids_)
    print(f"[nine] source={a.source}, {len(recs)} records, corpus {len(docs)} docs, topk={a.topk}, backbone={C.BACKBONE}", flush=True)
    fck = a.finder if a.detector_z else ""
    print("[provenance] " + json.dumps({"regime": "detector-Z" if a.detector_z else "oracle-Z", "finder": fck or None,
                                        "thr": (a.thr if a.detector_z else None), "frozen_index": True,
                                        "backbone": C.BACKBONE, "topk": a.topk}), flush=True)
    M = ExciseMethod(reembed_lora="", finder_ckpt=fck, device=dev, thr=a.thr)
    M.index(docs, precompute_lora=False, verbose=False)

    sp = Sparse(docs) if any(f in want for f in ("bm25z", "poslex_a", "poslex_b")) else None
    clf = None
    if "readout" in want:
        tf = f"data/benchmark/train/{a.source}.jsonl"                   # the collection's own training records
        tr = [json.loads(l) for l in open(tf)] if os.path.exists(tf) else []
        print(f"[readout] {len(tr)} training records of {a.source}", flush=True)
        clf = train_readout(M, tr, i2) if tr else None

    need_z = any(f in want for f in ("contrastive", "orthog", "prf", "zcond", "maskz", "maxsimz"))
    emitted = [f for f in want if f != "readout"] + (["readout"] if clf is not None else [])
    agg = defaultdict(lambda: [0.0, 0])
    perq = []
    skipped = n_silent = 0
    for qi, r in enumerate(recs):
        pid = str(r["pos_id"]); negs = [str(n) for n in r["neg_ids"]]
        if pid not in i2 or any(n not in i2 for n in negs):
            skipped += 1; continue
        p, ns = i2[pid], {i2[n] for n in negs}
        query = r["query"]
        Zs = r["Z"] if isinstance(r["Z"], list) else [r["Z"]]
        if a.detector_z:
            dz_str = M.finder.extract_Z(query) if M.finder is not None else ""
            Zs = [dz_str] if dz_str else []
        Zj = "; ".join(z for z in Zs if z)
        qX = strip_Z(query, Zj)

        qvv, qmv = emb(M.mv, [query], True, dev)
        base = torch.cat([maxsim(qvv.float().to(Dv.device), qmv.float().to(Dv.device), Dv, Dm)[0].to(dev) for Dv, Dm in M.van])
        cand, cids = M._select(base, a.topk)
        if not Zj:
            # The detector is silent: no arm has a topic to act on, so each returns the frozen order. It is counted
            # for every arm, so that all arms share one denominator.
            order = cids[torch.argsort(base[cand], descending=True)].tolist()
            _, _, suc = success10(order, {p}, ns)
            for nm in emitted:
                agg[nm][0] += suc; agg[nm][1] += 1
                agg[f"{nm}|{r.get('tier', '?')}"][0] += suc; agg[f"{nm}|{r.get('tier', '?')}"][1] += 1
            perq.append({"id": r.get("id"), "tier": r.get("tier"), "silent": True, "s": {nm: int(suc) for nm in emitted}})
            n_silent += 1
            continue
        topl = cand.tolist(); dids = cids.tolist()
        Vk, Mk = M._gather(M.van, topl)
        b = base[cand].clone()
        scores = {}

        if "frozen" in want:
            scores["frozen"] = b
        if need_z:
            qxv, qxm = emb(M.mv, [qX], True, dev)
            ezv, ezm = emb(M.mv, [Zj], True, dev)
        if "maskz" in want:
            scores["maskz"] = maxsim(qxv.float(), qxm.float(), Vk, Mk)[0]
        if "contrastive" in want:
            scores["contrastive"] = per_token(qxv, qxm, Vk, Mk) - per_token(ezv, ezm, Vk, Mk)
        if "orthog" in want:
            Zt = ezv[0][ezm[0].bool()].float()
            u = torch.linalg.svd(Zt, full_matrices=False)[2][0]

            def proj(V):
                Vp = V - (V @ u)[..., None] * u
                return Vp / Vp.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            scores["orthog"] = maxsim(proj(qxv.float()), qxm.float(), proj(Vk), Mk)[0]
        if "maxsimz" in want:
            scores["maxsimz"] = bridge(b, per_token(ezv, ezm, Vk, Mk))
        if "prf" in want:
            zsim = per_token(ezv, ezm, Vk, Mk)
            top_m = torch.topk(zsim, min(3, zsim.numel())).indices
            c = torch.cat([Vk[i][Mk[i]] for i in top_m.tolist()]).mean(0)
            qp = qxv.float() - c
            qp = qp / qp.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            scores["prf"] = maxsim(qp, qxm.float(), Vk, Mk)[0]
        if sp is not None:
            terms = zterms(Zs)
            if "bm25z" in want:
                pen = torch.tensor([sp.bm25(d, terms) for d in dids], device=dev, dtype=torch.float32)
                scores["bm25z"] = bridge(b, pen)
            for v, key in (("a", "poslex_a"), ("b", "poslex_b")):
                if key in want:
                    pen = torch.tensor([sp.poslex(d, terms, v) for d in dids], device=dev, dtype=torch.float32)
                    scores[key] = bridge(b, pen)
        if clf is not None:
            pooled = ((Vk * Mk[..., None]).sum(1) / Mk.sum(1, keepdim=True).clamp(min=1)).cpu().numpy()
            pe = torch.tensor(clf.predict_proba(pooled)[:, 1], device=dev, dtype=torch.float32)
            scores["readout"] = bridge(b, pe)
        if "zcond" in want:
            Vz, Mz = emb(M.mv, [f"{Zj}. {docs[d]}" for d in dids], False, dev)
            d0 = per_token(qxv, qxm, Vk, Mk)
            dz = per_token(qxv, qxm, Vz.float(), Mz)
            scores["zcond"] = bridge(b, dz - d0)

        row = {}
        for name, s in scores.items():
            order = cids[torch.argsort(s, descending=True)].tolist()
            _, _, suc = success10(order, {p}, ns)
            agg[name][0] += suc; agg[name][1] += 1
            agg[f"{name}|{r.get('tier', '?')}"][0] += suc; agg[f"{name}|{r.get('tier', '?')}"][1] += 1
            row[name] = int(suc)
        perq.append({"id": r.get("id"), "tier": r.get("tier"), "silent": False, "s": row})
        if (qi + 1) % 100 == 0:
            print(f"  ... {qi + 1}/{len(recs)}", flush=True)

    print(f"\n{'intervention':24s} {'n':>6s} {'success@10':>11s} {'vs frozen':>10s}")
    fz = agg["frozen"][0] / max(agg["frozen"][1], 1) if "frozen" in agg else float("nan")
    for k in sorted(agg):
        if "|" in k: continue
        s, n = agg[k]
        print(f"{k:24s} {n:6d} {s / n:11.4f} {s / n - fz:+10.4f}")
    print("\nper-tier:")
    for k in sorted(agg):
        if "|" not in k: continue
        s, n = agg[k]
        print(f"  {k:34s} {n:5d} {s / n:8.4f}")
    if skipped: print(f"\n!! skipped {skipped} records (id resolution)")
    if n_silent: print(f"[detector-Z] silent on {n_silent} queries -> frozen order credited to every arm")
    if a.dump_perquery:
        with open(a.dump_perquery, "w") as f:
            for x in perq: f.write(json.dumps(x) + "\n")
        print(f"wrote {len(perq)} per-query rows to {a.dump_perquery}")
    print("\nNINE_DONE")


if __name__ == "__main__":
    main()
