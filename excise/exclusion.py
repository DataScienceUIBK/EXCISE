"""The EXCISE operator: re-embed the shortlist with the exclusion adapter, score each candidate's evidence
for the excluded topic Z, then apply the soft penalty and the hard demotion."""
from __future__ import annotations
import sys, os, json, hashlib
# the repository root, so `config` and the packages import from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from excise import config as C
import torch
from excise.encoder import load_model, emb
from excise.maxsim import weighted_maxsim_matrix as maxsim
from excise.detector import DeepFinder
from excise.schema import content_tokens


def strip_Z(query, Z):
    """The POSITIVE query q_X: the query with the excluded-topic words removed (word-level, case-insensitive)."""
    import re as _re
    zw = {w for w in _re.findall(r"[a-z0-9]+", (Z or "").lower())}
    return " ".join(w for w in query.split()
                    if _re.sub(r"[^a-z0-9]+", "", w.lower()) not in zw) or query


MIN_ANOM_N = 4

MIN_HARD_N = 4


def per_token_maxsim(q_vec, q_w, d_vec, d_mask):
    """Per-QUERY-TOKEN MaxSim: (Bq,Bd,Lq) -- exactly the quantity weighted_maxsim_matrix sums over Lq. Exposing it"""
    sims = torch.einsum("ilp,jtp->ijlt", q_vec, d_vec)
    sims = sims.masked_fill(~d_mask[None, :, None, :], float("-inf"))
    return sims.max(dim=-1).values * q_w[:, None, :]


def pool_z(per_q, q_w, pool="mean", m=2):
    """Pool per-Z-token evidence (1,Bd,Lz) -> (Bd,)."""
    ev = per_q[0]                                                  # (Bd,Lz)
    w = q_w[0]                                                     # (Lz,)
    n = int(w.sum().clamp(min=1).item())
    if pool == "mean":
        return ev.sum(dim=-1) / max(n, 1)
    valid = ev.masked_fill((w <= 0)[None, :], float("-inf"))       # never pool padding tokens
    if pool == "max":
        return valid.max(dim=-1).values
    return valid.topk(max(1, min(m, n)), dim=-1).values.mean(dim=-1)



def gap_cut(score, prominence=3.0):
    """NON-PARAMETRIC cut: split the evidence at its LARGEST GAP, instead of at mu + sigma*std."""
    v, _ = torch.sort(score.float(), descending=True)
    n = v.numel()
    if n < 3:
        return v.max()
    gaps = v[:-1] - v[1:]
    upper = gaps[:max(1, n // 2)]                  # a valley must be in the upper half to be a Z-cluster edge
    k = int(torch.argmax(upper))
    med = float(torch.median(gaps).clamp(min=1e-9))
    if float(upper[k]) < 3.0 * med:
        return v.max()                             # unimodal => no cut => the no-harm property no-harm, structurally
    return (v[k] + v[k + 1]) * 0.5                 # midpoint of the valley


class ExciseMethod:
    """reembed_lora="" -> the re-embed encoder IS the frozen vanilla one (the frozen-ColBERT baseline arm)."""

    def __init__(self, reembed_lora=C.REEMBED_LORA, finder_ckpt=C.FINDER_CKPT, device="cuda:0", thr=None):
        self.dev = device
        self.mv = load_model("", device)                        # vanilla (frozen index encoder)
        self.ml = self.mv if not reembed_lora else load_model(reembed_lora, device)
        if finder_ckpt:
            self.finder = DeepFinder(finder_ckpt, device, thr=thr)
            self.finder_type = "deep"
        else:
            self.finder = None
            self.finder_type = "none"
        self.van = self.lor = self.corpus = None
        from collections import OrderedDict as _OD
        self._rc = _OD(); self._rc_bytes = 0
        self._rc_budget = int(float(os.environ.get("EXCISE_REEMBED_CACHE_GB", "4.0")) * 1e9)  # ON by default
        self.chunk2doc = None; self.n_docs = 0                  # long-doc chunking (None => 1 chunk == 1 doc)
        self.q_enc = self.mv                                    # first-stage query encoder; index() sets it
        self.first_stage = "index"                              # "index": query with the encoder that built the index; "frozen": always the frozen encoder

    @torch.no_grad()
    def index(self, corpus, doc_chunk=2048, precompute_lora=True, index_with_lora=False, verbose=True):
        """Encode the corpus. `van` = the frozen index used for first-stage retrieval."""
        self._rc.clear(); self._rc_bytes = 0  # cache keys are corpus positions; a new corpus voids them
        cw = int(os.environ.get("EXCISE_CHUNK_WORDS", "0"))
        mx = int(os.environ.get("EXCISE_MAX_DOC_WORDS", "0"))   # cap words/doc BEFORE windowing so a corpus of huge
        if cw > 0:                                             # docs (edgar avg ~9.9k words) can't explode the index
            chunked, c2d = [], []
            for di, d in enumerate(corpus):
                w = (d or "").split()
                if mx > 0 and len(w) > mx: w = w[:mx]           # short-context backbones: index the first mx words
                if len(w) <= cw:
                    chunked.append(d); c2d.append(di)
                else:
                    for s in range(0, len(w), cw): chunked.append(" ".join(w[s:s + cw])); c2d.append(di)
            self.n_docs = len(corpus); self.chunk2doc = torch.tensor(c2d, device=self.dev)
            print(f"[chunk] {len(corpus)} docs -> {len(chunked)} chunks (cap {cw} words)", flush=True)
            corpus = chunked
        else:
            self.n_docs = len(corpus); self.chunk2doc = None
        if os.environ.get("EXCISE_DOC_CHUNK"):
            doc_chunk = int(os.environ["EXCISE_DOC_CHUNK"])
        elif len(corpus) > 120_000:
            doc_chunk = min(doc_chunk, 256)
        elif len(corpus) > 60_000:
            doc_chunk = min(doc_chunk, 512)
        self.corpus, self.doc_chunk, self.van = corpus, doc_chunk, []
        enc_idx = self.ml if index_with_lora else self.mv
        self.q_enc = enc_idx if self.first_stage == "index" else self.mv   # see --first-stage
        reuse = enc_idx is self.ml
        self.lor = [] if (precompute_lora and not reuse) else None
        n = len(corpus)
        use_fp16 = os.environ.get("EXCISE_INDEX_FP16") == "1"
        keep_fp32 = not use_fp16
        cast = (lambda V: V.half()) if use_fp16 else (lambda V: V)
        ndev = int(os.environ.get("EXCISE_INDEX_SHARD", torch.cuda.device_count() or 1))
        devs = [f"cuda:{i}" for i in range(ndev)] if ndev > 1 else [self.dev]
        self.store_devs = devs
        est = n * 256 * 128 * (4 if keep_fp32 else 2) / 1e9 * (2 if self.lor is not None else 1)
        mode = "aliased(same encoder)" if reuse else ("precomputed" if self.lor is not None else "at-query-time")
        if verbose:
            print(f"[index] {n} docs, lora_store={mode}, index_with_lora={index_with_lora}, "
                  f"~{est:.1f} GB token-vector store", flush=True)
        _cdir = None
        if os.environ.get("EXCISE_INDEX_CACHE") and self.lor is None:
            _h = hashlib.sha256()
            _h.update(repr((getattr(C, "BACKBONE", ""), bool(use_fp16), doc_chunk,
                            index_with_lora, reuse, n)).encode())
            for _t in corpus:
                _h.update((_t or "").encode("utf-8", "replace"))
            _cdir = os.path.join(os.environ["EXCISE_INDEX_CACHE"], _h.hexdigest()[:24])

        _hit = False
        if _cdir and os.path.exists(os.path.join(_cdir, "COMPLETE")):
            # A cache that half-loads is worse than one that misses: fall back to encoding on ANY problem
            # rather than proceeding with a short store, which would silently change every score.
            try:
                _want = (n + doc_chunk - 1) // doc_chunk
                _meta = json.loads(open(os.path.join(_cdir, "COMPLETE")).read())
                if _meta.get("blocks") != _want or _meta.get("n") != n:
                    raise ValueError(f"cache records {_meta.get('blocks')} blocks for n={_meta.get('n')}, "
                                     f"this corpus needs {_want} for n={n}")
                _tmp = []
                for bi in range(_want):
                    d = devs[bi % len(devs)]
                    _b = torch.load(os.path.join(_cdir, f"{bi:06d}.pt"), map_location="cpu")
                    _tmp.append((_b["v"].to(d), _b["m"].to(d)))
                self.van, _hit = _tmp, True
                if verbose:
                    print(f"[index] loaded {len(self.van)} encoded blocks from cache {_cdir}", flush=True)
            except Exception as _e:
                print(f"[index] cache unusable ({type(_e).__name__}: {_e}) -- re-encoding", flush=True)
                self.van, _hit = [], False
        if not _hit:
            if _cdir:
                os.makedirs(_cdir, exist_ok=True)
            for bi, s in enumerate(range(0, n, doc_chunk)):
                d = devs[bi % len(devs)]                       # this block's home card
                Vv, Mv = emb(enc_idx, corpus[s:s + doc_chunk], False, self.dev)
                _cv = cast(Vv)
                if _cdir:
                    _bp = os.path.join(_cdir, f"{bi:06d}.pt")
                    _tp = f"{_bp}.{os.getpid()}.tmp"
                    torch.save({"v": _cv.cpu(), "m": Mv.cpu()}, _tp)
                    os.replace(_tp, _bp)
                self.van.append((_cv.to(d), Mv.to(d)))
                if self.lor is not None:
                    Vl, Ml = emb(self.ml, corpus[s:s + doc_chunk], False, self.dev)
                    self.lor.append((cast(Vl).to(d), Ml.to(d)))
                del Vv, Mv
            if _cdir:
                open(os.path.join(_cdir, "COMPLETE"), "w").write(
                    json.dumps({"blocks": len(self.van), "n": n, "doc_chunk": doc_chunk,
                                "fp16": bool(use_fp16)}) + "\n")
                if verbose:
                    print(f"[index] cached {len(self.van)} encoded blocks to {_cdir}", flush=True)
        if len(devs) > 1 and verbose:
            print(f"[index] store sharded over {len(devs)} GPUs", flush=True)
        if reuse: self.lor = self.van                                 # identical encoder: alias, never encode twice
        return self

    def free_index(self):
        """Drop the encoded corpus (keeps the loaded models) so the next source can be indexed."""
        self._rc.clear(); self._rc_bytes = 0
        self.van = self.lor = self.corpus = None
        self.chunk2doc = None; self.n_docs = 0
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return self

    def _select(self, base, topk):
        """First-stage candidate selection. Returns (cand, ids): cand = CHUNK indices to gather embeddings/text"""
        if self.chunk2doc is None:
            top = torch.topk(base, min(topk, base.numel())).indices
            return top, top
        nd = self.n_docs
        doc_base = torch.full((nd,), float("-inf"), device=base.device)
        doc_base.scatter_reduce_(0, self.chunk2doc, base, reduce="amax", include_self=True)   # per-doc max score
        ismax = base == doc_base[self.chunk2doc]                                                # chunk == its doc's max
        rep = torch.zeros(nd, dtype=torch.long, device=base.device)
        idxs = torch.nonzero(ismax, as_tuple=True)[0]; rep[self.chunk2doc[idxs]] = idxs         # rep chunk per doc
        ids = torch.topk(doc_base, min(topk, nd)).indices
        return rep[ids], ids

    def _emb_docs_cached(self, idxs):
        """Re-embed candidate documents with the adapter, reusing anything already seen. Exact, not approximate."""
        if self._rc_budget <= 0:
            V, M = emb(self.ml, [self.corpus[i] for i in idxs], False, self.dev)
            return V.float(), M
        need = [i for i in idxs if i not in self._rc]
        fresh = {}
        if need:
            Vm, Mm = emb(self.ml, [self.corpus[i] for i in need], False, self.dev)
            for j, i in enumerate(need):
                n = int(Mm[j].sum().item())
                fresh[i] = Vm[j, :n].float().clone()                       # store UNPADDED
        def _get(i):
            if i in self._rc:
                self._rc.move_to_end(i); return self._rc[i]
            return fresh[i]
        vecs = [_get(i) for i in idxs]
        T = max(v.shape[0] for v in vecs); P = vecs[0].shape[1]
        V = torch.zeros(len(vecs), T, P, device=self.dev)
        M = torch.zeros(len(vecs), T, dtype=torch.bool, device=self.dev)
        for k, v in enumerate(vecs):
            V[k, :v.shape[0]] = v; M[k, :v.shape[0]] = True
        for i, v in fresh.items():                                          # insert AFTER gathering, so nothing
            self._rc[i] = v; self._rc.move_to_end(i)                        # needed by THIS call can be evicted
            self._rc_bytes += v.numel() * v.element_size()
        while self._rc_bytes > self._rc_budget and self._rc:
            _, ev = self._rc.popitem(last=False)
            self._rc_bytes -= ev.numel() * ev.element_size()
        return V, M

    def _gather(self, store, idx):
        vs, ms = [], []
        for i in idx:
            c, loc = i // self.doc_chunk, i % self.doc_chunk; Dv, Dm = store[c]
            # back to the working card and to fp32 before any MaxSim, whatever the block's home or storage
            vs.append(Dv[loc].float().to(self.dev)); ms.append(Dm[loc].to(self.dev))
        T, P = max(v.shape[0] for v in vs), vs[0].shape[1]
        Vk = torch.zeros(len(vs), T, P, device=self.dev); Mk = torch.zeros(len(vs), T, dtype=torch.bool, device=self.dev)
        for j, (v, mm) in enumerate(zip(vs, ms)): Vk[j, :v.shape[0]] = v; Mk[j, :mm.shape[0]] = mm
        return Vk, Mk

    def _demote(self, rr, dz, ezm, top, lam, demote_tau, soft_demote, adaptive=False,
                hard1=False, hard_m=1, hard_guard=False, guard_rule="top_rel", relative=False, qpt=None, zev=None,
                floor_ev=None,
                hard_auto=False, hard_frac=0.6, hard_cap=3, min_hard_n=MIN_HARD_N, two_stage=False, ts_floor=0.35, ts_q=0.75, anomaly_gate=False, anomaly_sigma=0.5,
                xgate=False, xgate_q=0.5, xgate_strict=False, gap_gate=False):
        """Apply the exclusion penalty to candidate scores `rr` IN PLACE; return the demoted doc-id (or None)."""
        scale = float(rr.abs().max()) + 1e-6
        tgt = dz if zev is None else zev                             # what "strongest Z-match" means, under the pooling
        if adaptive:
            nz = float(ezm.sum().clamp(min=1.0))
            ev = (dz / nz) if zev is None else zev                   # per-candidate Z-evidence
            pen = (ev - demote_tau).clamp(min=0)
            if float(pen.max()) <= 0:
                return None
            n_strong = int((ev >= 0.6 * float(ev.max())).sum())      # comparably-strong Z-docs (conservative)
            if n_strong >= 2:                                        # multi-target -> soft global (T3)
                rr -= lam * scale * pen
            else:                                                   # single-target -> hard demote-1 (T1/T2)
                d = int(tgt.argmax()); rr[d] = rr[d] - scale * lam
            return int(top[int(tgt.argmax())])
        if soft_demote:
            nz = float(ezm.sum().clamp(min=1.0))                      # per-token so short/long Z are comparable
            zpt = (dz / nz) if zev is None else zev                   # per-candidate Z-evidence (mean, or pooled)
            if relative and qpt is not None:
                score = zpt - qpt
                pen = (score - demote_tau).clamp(min=0)
            else:
                score = zpt
                pen = (score - demote_tau).clamp(min=0)
            if two_stage:
                _gate = zpt if floor_ev is None else floor_ev
                if float(_gate.max()) < ts_floor:
                    pen = torch.zeros_like(score)
                elif score.numel() >= MIN_ANOM_N:
                    # Stage 2 (EFFECTIVENESS, relative): Use statistical anomaly or static quantile
                    if gap_gate:
                        cut = gap_cut(score)                      # non-parametric; ignores anomaly_sigma
                    elif anomaly_gate:
                        mu = score.float().mean()
                        std = score.float().std() + 1e-6
                        cut = mu + anomaly_sigma * std
                    else:
                        cut = torch.quantile(score.float(), ts_q)
                    pen = (score - cut).clamp(min=0)
                # else: shortlist too small for a distributional cut (see MIN_ANOM_N) -> KEEP the absolute-rule
                # `pen` computed just above. Not a zero penalty: at n<=3 the cut is degenerate, not informative.
            if float(pen.max()) <= 0:
                return None
            top_rel = int(rr.argmax())                               # the ANSWER = most-relevant candidate (pre-penalty)
            rel_pre = rr.detach().clone()                            # see `protected` below: BOTH guard rules read
            rr -= lam * scale * pen                                   # pre-penalty relevance, not just top_rel
        else:
            nz = float(ezm.sum().clamp(min=1.0))
            score = (dz / nz) if zev is None else zev
            if two_stage:
                _gate = score if floor_ev is None else floor_ev
                if float(_gate.max()) < ts_floor:
                    pen = torch.zeros_like(score)
                elif score.numel() >= MIN_ANOM_N:
                    if gap_gate:
                        cut = gap_cut(score)
                    elif anomaly_gate:
                        mu = score.float().mean()
                        std = score.float().std() + 1e-6
                        cut = mu + anomaly_sigma * std
                    else:
                        cut = torch.quantile(score.float(), ts_q)
                    pen = (score - cut).clamp(min=0)
                else:
                    pen = (score - demote_tau).clamp(min=0)   # too few candidates for the cut (MIN_ANOM_N)
            else:
                pen = (score - demote_tau).clamp(min=0)

            if float(pen.max()) <= 0:
                return None
            top_rel = int(rr.argmax())
            rel_pre = rr                                             # hard path: rr is still pre-penalty here

        am = int(tgt.argmax())
        protected = top_rel
        if guard_rule == "rel_not_am":
            rel_ord = torch.argsort(rel_pre, descending=True).tolist()
            protected = next((i for i in rel_ord if i != am), top_rel)
        tiny = tgt.numel() < min_hard_n            # MIN_HARD_N: relative selection degenerates here
        # tiny AND a soft penalty is already acting -> that penalty suffices; skip the degenerate hard drop.
        if (hard1 or not soft_demote) and not (tiny and soft_demote):  # hard-demote the top-m Z-docs
            if tiny:
                sel = torch.topk(tgt, 1).indices.tolist()            # plain demote-1: an argmax, always defined
            elif hard_auto or (two_stage and not soft_demote):
                lim = hard_frac * float(tgt.max()) if hard_auto else 0.0
                sel = [i for i in range(tgt.numel()) if float(tgt[i]) >= lim and float(pen[i]) > 0]
                sel = sorted(sel, key=lambda i: -float(tgt[i]))
                if hard_cap > 0 and hard_auto: sel = sel[:hard_cap]
            else:
                sel = torch.topk(tgt, min(hard_m, tgt.numel())).indices.tolist()  # penalty. m=1 recovers T1;
            _need = 1 if xgate_strict else 2
            if (xgate or xgate_strict) and qpt is not None and qpt.numel() == tgt.numel() and len(sel) >= _need:
                qcut = float(torch.quantile(qpt.float(), xgate_q))
                kept = [i for i in sel if float(qpt[i]) <= qcut]
                if kept or xgate_strict: sel = kept              # strict: an empty drop-set is a legal answer
            for i in sel:
                if float(pen[i]) > 0 and not (hard_guard and not tiny and i == protected):    # GUARD: never hard-demote the
                    rr[i] = -9999.0                                              # GUARANTEE bottom rank
        return int(top[am])                                      # the real Z-matches, so T1 stays 1, no over-demote

    def _pos_qpt(self, query, Z, encoder, Vk, Mk, X=None):
        """Per-candidate per-token relevance to the POSITIVE query (query with the excluded Z words removed)."""
        if X:
            qpos = X
        else:
            qpos = strip_Z(query, Z)
        qv, qm = emb(encoder, [qpos], True, self.dev)
        prel = maxsim(qv.float(), qm.float(), Vk, Mk)[0]
        return prel / float(qm.sum().clamp(min=1.0))

    @staticmethod
    def _adaptive_lam(rr, ev, scale):
        """A per-query demotion strength with NO fitted constant."""
        sr = float(rr.std()); se = float(ev.std())
        if not (sr > 0 and se > 0 and scale > 0):
            return None                                  # degenerate shortlist -> caller keeps fixed lam
        return sr / (scale * se)

    def _lexical_demote(self, rr, topl, top, Z, lam):
        """Training-free, DOMAIN-INVARIANT negation: hard-demote every top-K doc whose TEXT literally contains an"""
        import re
        zw = {w for w in re.findall(r"[a-z0-9]+", Z.lower()) if len(w) >= 3}
        if not zw: return None
        scale = float(rr.abs().max()) + 1e-6
        hit = None
        for j, di in enumerate(topl):
            dw = set(re.findall(r"[a-z0-9]+", self.corpus[di].lower()))
            if zw & dw:
                rr[j] = rr[j] - scale * lam
                if hit is None: hit = int(top[j])                    # first == highest-ranked lexical match
        return hit

    @torch.no_grad()
    def rank(self, query, topk=None, lam=None, demote=True, switch_gate=True, reembed=True, demote_tau=0.0,
             soft_demote=False, adaptive_demote=False, signed_op=False, lexical_neg=False,
             hard1=False, hard_m=1, hard_guard=False, guard_rule="top_rel", relative=False, blend=False, blend_beta=0.5, z_veto=0, z_hard=0, override_Z=None,
             mask_z=False, z_pool="mean", z_pool_m=2, query_only=False, adapt_query=False, adaptive_lam=False, learned_x=False,
             hard_auto=False, hard_frac=0.6, hard_cap=3, min_hard_n=MIN_HARD_N, two_stage=False, ts_floor=0.35, ts_q=0.75, anomaly_gate=False, anomaly_sigma=0.5, score_gap=False, score_gap_sigma=1.0, score_gap_intent_thr=0.3,
             xgate=False, xgate_q=0.5, xgate_strict=False, gap_gate=False, zanchor=False, zanchor_beta=1.0, zanchor_m=3,
             gap_evidence=False, gap_beta=0.25, zmask_q=False, zmask_soft=False):
        """Return (ordered_doc_ids, info)."""
        if self.van is None: raise RuntimeError("ExciseMethod.index(corpus) must be called before rank()")
        topk = topk or C.REEMBED_TOPK; lam = C.DEMOTE_LAM if lam is None else lam
        qvv, qmv = emb(self.q_enc, [query], True, self.dev)
        base = torch.cat([maxsim(qvv.float().to(Dv.device), qmv.float().to(Dv.device),
                                 Dv.float(), Dm)[0].to(self.dev) for Dv, Dm in self.van])
        cand, ids = self._select(base, topk)     # cand = CHUNK idx (gather), ids = DOC idx (return); equal if no chunking
        # Finder BEFORE the re-embed: only then can the Switch gate the re-embed, not just the demotion.
        E_z = None; alpha = None; X_learned = None
        if override_Z is not None:
            Z = override_Z
            has_intent = bool(Z)
        elif self.finder is not None:
            if learned_x and hasattr(self.finder, "extract_XZ") and getattr(self.finder, "xz", False):
                X_learned, Z = self.finder.extract_XZ(query)
            else:
                Z = self.finder.extract_Z(query)
            has_intent = bool(Z)
            if score_gap and not has_intent:
                has_intent = self.finder.fire_score(query) > score_gap_intent_thr
        else:
            Z = ""
            has_intent = False

        if z_veto and Z and len(set(content_tokens(Z))) > z_veto:                                 # topical FALSE-fire veto:
            Z = ""   # a real exclusion flags ~3 tokens; a false-fire on a long argument flags ~10 -> treat as silent
            has_intent = False
        if z_hard and Z and len(set(content_tokens(Z))) > z_hard:                                  # FALSE-fire -> HARD demote-1
            soft_demote = False; hard1 = False   # keeps the LoRA re-embed benefit (touches 1 doc, not all) unlike z_veto
        # `override_Z is not None` included so an INJECTED extractor gets the same switch semantics as the
        # trained detector: an empty extraction returns the frozen order verbatim.
        if switch_gate and (self.finder is not None or override_Z is not None) and not has_intent:                           # silent => vanilla, verbatim
            return ids, {"Z": "", "fired": False, "demoted_id": None, "reembedded": False}
        topl = cand.tolist()
        if query_only:
            Vk_v, Mk_v = self._gather(self.van, topl)
            qv, qm = emb(self.ml, [query], True, self.dev)
            rr = maxsim(qv.float(), qm.float(), Vk_v, Mk_v)[0]
            order = ids[torch.argsort(rr, descending=True)]
            return order, {"Z": Z, "fired": bool(Z), "demoted_id": None, "reembedded": False}
        if not reembed:                                                                          # E1: DEMOTE-ON-FROZEN
            Vk_v, Mk_v = self._gather(self.van, topl)
            if adapt_query:
                _qv, _qm = emb(self.ml, [query], True, self.dev)
                rr = maxsim(_qv.float(), _qm.float(), Vk_v, Mk_v)[0]
            else:
                rr = base[cand].clone()
            demoted = None
            if Z and demote:
                _zenc = self.ml if adapt_query else self.mv          # Z shares the query side's encoder
                if E_z is not None:
                    ez = E_z
                    ezm = torch.ones(ez.shape[0], dtype=torch.bool, device=self.dev)
                    dz = maxsim(ez.unsqueeze(0), ezm.unsqueeze(0), Vk_v, Mk_v)[0]
                else:
                    ez, ezm = emb(_zenc, [Z], True, self.dev)
                    dz = maxsim(ez.float(), ezm.float(), Vk_v, Mk_v)[0]
                if signed_op:                                        # PURE signed operator: rel - lam*Zsim (no relu floor)
                    scale = float(rr.abs().max()) + 1e-6; nz = float(ezm.sum().clamp(min=1.0))
                    lam_eff = lam * alpha if alpha is not None else lam
                    if adaptive_lam:
                        _al = self._adaptive_lam(rr, dz / nz, scale)
                        if _al is not None: lam_eff = _al
                    rr = rr - lam_eff * scale * (dz / nz)
                    demoted = int(ids[int(dz.argmax())])
                else:
                    qpt = self._pos_qpt(query, Z, (self.ml if adapt_query else self.mv), Vk_v, Mk_v, X=X_learned) if (relative or xgate or xgate_strict) else None
                    lam_eff = lam * alpha if alpha is not None else lam
                    if adaptive_lam:
                        _sc = float(rr.abs().max()) + 1e-6
                        _al = self._adaptive_lam(rr, dz / float(ezm.sum().clamp(min=1.0)), _sc)
                        if _al is not None: lam_eff = _al
                    demoted = self._demote(rr, dz, ezm, ids, lam_eff, demote_tau, soft_demote, adaptive_demote,
                                           hard1=hard1, hard_m=hard_m, hard_guard=hard_guard, guard_rule=guard_rule, relative=relative, qpt=qpt,
                                           hard_auto=hard_auto, hard_frac=hard_frac, hard_cap=hard_cap, min_hard_n=min_hard_n,
                                           two_stage=two_stage, ts_floor=ts_floor, ts_q=ts_q, anomaly_gate=anomaly_gate, anomaly_sigma=anomaly_sigma,
                                           xgate=xgate, xgate_q=xgate_q, xgate_strict=xgate_strict, gap_gate=gap_gate)
            if lexical_neg and Z:                                    # domain-invariant lexical backstop (stacks on dense)
                demoted = self._lexical_demote(rr, topl, ids, Z, lam) or demoted
            order = ids[torch.argsort(rr, descending=True)]
            return order, {"Z": Z, "fired": bool(Z), "demoted_id": demoted, "reembedded": False}
        if self.lor is not None:                                                                 # precomputed
            Vk_l, Mk_l = self._gather(self.lor, topl)
        else:                                                                                    # faithful: re-embed the K candidate TEXTS now
            Vk_l, Mk_l = self._emb_docs_cached(topl)
        q_rel = strip_Z(query, Z) if (mask_z and Z) else query
        qvl, qml = emb(self.ml, [q_rel], True, self.dev)
        qw = qml.float()
        if zmask_q and Z and not mask_z:
            zm, zn = (self.finder.z_token_prob(query) if zmask_soft
                      else self.finder.z_token_mask(query))
            L = int(qw.shape[1])
            if zn > L:
                raise RuntimeError(f"--zmask-q: detector tokenised the query to {zn} tokens, the re-embed "
                                   f"encoder to {L}; the mask cannot be aligned and would zero wrong tokens")
            qw = qw.clone()
            qw[0, :zn] = qw[0, :zn] * (1.0 - zm[:zn].to(qw.device))
            if float(qw.sum()) <= 0:                   # Z covered the whole query -> nothing to rank on
                qw = qml.float()
        rr = maxsim(qvl.float(), qw, Vk_l, Mk_l)[0]                                               # LoRA rerank
        if blend:
            fr = base[cand]
            rn = (rr - rr.mean()) / (rr.std() + 1e-6)
            fn = (fr - fr.mean()) / (fr.std() + 1e-6)
            rr = blend_beta * rn + (1.0 - blend_beta) * fn
        demoted = None
        if score_gap and has_intent and not Z and demote:
            fr = base[cand].float()  # frozen scores for these candidates
            fn = (fr - fr.mean()) / (fr.std() + 1e-6)  # z-normalise
            ln = (rr - rr.mean()) / (rr.std() + 1e-6)  # z-normalise
            gap = fn - ln  # positive = frozen liked it more than LoRA
            gap_mu = gap.mean()
            gap_std = gap.std() + 1e-6
            cut = gap_mu + score_gap_sigma * gap_std
            anomalous = gap > cut
            if anomalous.any():
                if soft_demote:
                    scale = float(rr.abs().max()) + 1e-6
                    pen = (gap - cut).clamp(min=0)
                    rr[anomalous] -= lam * scale * pen[anomalous]
                else:
                    # HARD DROP: completely remove the anomalous documents from the top
                    # rankings, perfectly preserving the nDCG of the remaining good docs.
                    if hard_guard:
                        top_rel = int(rr.argmax())
                        anomalous[top_rel] = False
                    rr[anomalous] = -1e9
        if Z and demote:                                                                          # demote-1 / soft
            if E_z is not None:
                ez = E_z
                ezm = torch.ones(ez.shape[0], dtype=torch.bool, device=self.dev)
                dz = maxsim(ez.unsqueeze(0), ezm.unsqueeze(0), Vk_l, Mk_l)[0]
            else:
                ez, ezm = emb(self.ml, [Z], True, self.dev)
                dz = maxsim(ez.float(), ezm.float(), Vk_l, Mk_l)[0]
            zev = None if z_pool == "mean" else pool_z(
                per_token_maxsim(ez.float(), ezm.float(), Vk_l, Mk_l), ezm.float(), z_pool, z_pool_m)
            if zanchor:
                _qx = strip_Z(query, Z)
                _qxv, _qxm = emb(self.ml, [_qx], True, self.dev)
                _xc = maxsim(_qxv.float(), _qxm.float(), Vk_l, Mk_l)[0]           # positive-query match per candidate
                _anch = torch.topk(dz - _xc, min(zanchor_m, dz.numel())).indices.tolist()   # confident Z-docs
                _len = Mk_l.float().sum(1).clamp(min=1.0)
                _asim = torch.zeros_like(dz)
                for _ai in _anch:                                                # doc-doc MaxSim to each anchor
                    _asim = _asim + maxsim(Vk_l, Mk_l.float(), Vk_l[_ai:_ai + 1], Mk_l[_ai:_ai + 1])[:, 0] / _len
                _asim = _asim / max(len(_anch), 1)
                _base = zev if zev is not None else dz
                _as = _asim / (float(_asim.max()) + 1e-6) * (float(_base.max()) + 1e-6)   # rescale to evidence scale
                zev = _base + zanchor_beta * _as                                 # combined evidence (Z-token + relational)
            floor_ev = None
            if gap_evidence and z_pool == "mean":
                _nz = float(ezm.sum().clamp(min=1.0))
                _raw = dz / _nz
                _fr = base[cand].float()
                _fn = (_fr - _fr.mean()) / (_fr.std() + 1e-6)
                _ln = (rr - rr.mean()) / (rr.std() + 1e-6)
                _zn = (_raw - _raw.mean()) / (_raw.std() + 1e-6)
                zev = _zn + gap_beta * (_fn - _ln)
                floor_ev = _raw
            elif gap_evidence:
                raise SystemExit("--gap-evidence requires --z-pool mean (it z-scores the mean-pooled dz)")
            qpt = self._pos_qpt(query, Z, self.ml, Vk_l, Mk_l, X=X_learned) if (relative or xgate or xgate_strict) else None
            lam_eff = lam * alpha if alpha is not None else lam
            if adaptive_lam:
                _sc = float(rr.abs().max()) + 1e-6
                _al = self._adaptive_lam(rr, (zev if zev is not None else dz) / float(ezm.sum().clamp(min=1.0)), _sc)
                if _al is not None: lam_eff = _al
            demoted = self._demote(rr, dz, ezm, ids, lam_eff, demote_tau, soft_demote, adaptive_demote,
                                   hard1=hard1, hard_m=hard_m, hard_guard=hard_guard, guard_rule=guard_rule, relative=relative, qpt=qpt,
                                   zev=zev, floor_ev=floor_ev, hard_auto=hard_auto, hard_frac=hard_frac, hard_cap=hard_cap, min_hard_n=min_hard_n,
                                   two_stage=two_stage, ts_floor=ts_floor, ts_q=ts_q, anomaly_gate=anomaly_gate, anomaly_sigma=anomaly_sigma,
                                   xgate=xgate, xgate_q=xgate_q, xgate_strict=xgate_strict, gap_gate=gap_gate)
        if lexical_neg and Z:                                    # domain-invariant lexical backstop (reembed path too):
            demoted = self._lexical_demote(rr, topl, ids, Z, lam) or demoted   # hard-demote docs literally containing Z
        order = ids[torch.argsort(rr, descending=True)]
        return order, {"Z": Z, "fired": bool(Z), "demoted_id": demoted, "reembedded": True}
