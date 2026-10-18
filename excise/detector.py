"""The detector: a query-side LoRA with a span head. It fires on exclusion queries and returns the X and Z
spans as substrings of the query."""
from __future__ import annotations
import sys, os
# the repository root, so `config` and the packages import from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from excise import config as C
import torch, torch.nn as nn
from excise.detector_backbone import load_backbone

_EDGE = frozenset("""not without except excluding exclude excludes excluded omit omits omitted omitting
    other than apart from aside from besides neither nor minus rather instead of free non
    already covered discussed familiar beyond known documented setting leaving aside
    the a an of and or to in for on with is are was were be been it this that as at by from into
    about their his her its your our them they we you i any some""".split())


def _query_hidden(bb, texts, device):
    """768-d last_hidden_state on the QUERY path (LoRA active), matching bb.encode(is_query=True)."""
    enc = bb.tokenizer([bb.query_prefix + (t or "") for t in texts], padding="max_length",
                       truncation=True, max_length=bb.query_len, return_tensors="pt")
    ids = enc["input_ids"].to(device); am = enc["attention_mask"].to(device)
    pid, mid = bb.tokenizer.pad_token_id, bb.tokenizer.mask_token_id
    if mid is not None: ids = torch.where(ids == pid, torch.full_like(ids, mid), ids)
    return bb.encoder(input_ids=ids, attention_mask=am).last_hidden_state


def _load_lora_checked(bb, sd, path):
    """strict=False silently drops name-mismatched keys -> the finder LoRA would be a no-op (vanilla reps under a"""
    if not sd: raise RuntimeError(f"{path}: empty finder LoRA state dict")
    have = dict(bb.named_parameters())
    missing = [k for k in sd if k not in have]
    if missing:
        raise RuntimeError(f"{path}: {len(missing)} finder-LoRA keys match NO backbone parameter "
                           f"(silent no-op load). e.g. {missing[:3]}")
    res = bb.load_state_dict(sd, strict=False)
    if getattr(res, "unexpected_keys", []):
        raise RuntimeError(f"{path}: unexpected keys {list(res.unexpected_keys)[:3]}")


class DeepFinder:
    def __init__(self, ckpt_path, device="cuda:0", thr=None):
        ck = torch.load(ckpt_path, map_location="cpu")
        self.dev = device; self.thr = C.FINDER_THR if thr is None else thr
        self.bb = load_backbone(device=device, lora=True, lora_r=ck["cfg"]["lora_r"], lora_last_k=ck["cfg"]["lora_last_k"])
        _load_lora_checked(self.bb, ck["lora"], ckpt_path); self.bb.eval()
        # A Sequential (MLP) head stores 0.weight/2.weight, a Linear stores weight. Read the shape off
        # the FIRST and LAST matrices either way, so the tap and the class count are inferred for both.
        _mlp = "weight" not in ck["head"]
        _first = ck["head"]["0.weight"] if _mlp else ck["head"]["weight"]
        _last = ck["head"]["2.weight"] if _mlp else ck["head"]["weight"]
        hdim = _first.shape[1]                                  # 128 (proj) or 768 (hidden) — infer
        self.tap = ck["cfg"].get("tap", "hidden768" if hdim == self.bb.hidden_dim else "proj128")
        n_out = int(_last.shape[0])                         # 1 = bag-of-words, 3 = IOB, 5 = X/Z
        self.iob = bool(ck["cfg"].get("iob", n_out == 3))
        self.xz = bool(ck["cfg"].get("xz", n_out == 5))
        self.head = (nn.Sequential(nn.Linear(hdim, _first.shape[0]), nn.GELU(),
                                   nn.Linear(_first.shape[0], n_out)) if _mlp
                     else nn.Linear(hdim, n_out)).to(device)
        self.head.load_state_dict(ck["head"]); self.head.eval()
        self.spec = set(self.bb.tokenizer.all_special_ids)
        _tq = ck["cfg"].get("query_len")
        if _tq is not None and int(_tq) != int(self.bb.query_len):
            print(f"[finder] WARNING query_len MISMATCH: trained at {_tq}, running at {self.bb.query_len}. "
                  f"Set EXCISE_QUERY_LEN={_tq} to match training.", flush=True)

    @torch.no_grad()
    def extract_Z(self, query):
        """Return the excluded-topic Z string ('' if the finder is silent => no exclusion)."""
        raw, full, ids, offs, plen = self._forward(query)
        return self._decode(self._zprob(raw), full, ids, offs, plen)

    @torch.no_grad()
    def extract_XZ(self, query):
        """-> (X, Z). The POSITIVE topic and the EXCLUDED topic, as separate strings."""
        raw, full, ids, offs, plen = self._forward(query)
        Z = self._decode(self._zprob(raw), full, ids, offs, plen)
        X = self._decode(self._xprob(raw), full, ids, offs, plen) if self.xz else ""
        return X, Z

    @torch.no_grad()
    def _forward(self, query):
        bb = self.bb
        full = bb.query_prefix + (query or "")
        enc = bb.tokenizer(full, truncation=True, max_length=bb.query_len, return_offsets_mapping=True)
        feat = _query_hidden(bb, [query], self.dev) if self.tap == "hidden768" else bb.encode([query], is_query=True, device=self.dev).vectors
        return self.head(feat.float())[0], full, enc["input_ids"], enc["offset_mapping"], len(bb.query_prefix)

    def _decode(self, fire, full, ids, offs, plen, return_spans=False):
        """Per-token probability -> the verbatim substring of the query it selects."""
        n = min(len(ids), int(fire.shape[0]))
        wordish = lambda ch: ch.isalnum() or ch in "'-_+."
        words = {}                                        # (start,end) -> [sum p, count]
        for p in range(n):
            if ids[p] in self.spec:
                continue
            a, b = offs[p]
            if b <= a or b <= plen:                       # empty, or inside the query prefix
                continue
            a = max(a, plen)
            while a > plen and wordish(full[a - 1]): a -= 1
            while b < len(full) and wordish(full[b]): b += 1
            acc = words.setdefault((a, b), [0.0, 0])
            acc[0] += float(fire[p]); acc[1] += 1
        # HYSTERESIS. Z-word RECALL is markedly lower out of domain than in domain, while the
    # emitted length stays near-constant -- the head finds
        hyst = float(os.environ.get("EXCISE_HYST", "1") or 1)
        keys = sorted(words.keys())
        spans, open_ = [], False
        for (a, b) in keys:
            tot, k = words[(a, b)]
            pw = (tot / k) if k else 0.0
            gap = bool(spans) and full[spans[-1][1]:a].strip()      # a non-space gap closes the span
            if gap:
                open_ = False
            if pw > self.thr:
                spans.append([a, b]); open_ = True
            elif open_ and hyst < 1.0 and pw > self.thr * hyst:
                spans.append([a, b])                                 # continuation, lower bar
            else:
                open_ = False
        if not spans:
            return [] if return_spans else ""      # type must follow the flag, not the string path
        spans.sort()
        merged = [spans[0]]
        for a, b in spans[1:]:
            if a <= merged[-1][1] or not full[merged[-1][1]:a].strip():
                merged[-1][1] = max(merged[-1][1], b)     # touching, or separated by whitespace only
            else:
                merged.append([a, b])
        margin = float(os.environ.get("EXCISE_Z_MARGIN", "0") or 0)
        if margin > 0 and len(merged) > 1:
            def _score(a, b):
                ps = [tot / k for (wa, wb), (tot, k) in words.items() if k and wa >= a and wb <= b]
                return sum(ps) / len(ps) if ps else 0.0
            scored = [(_score(a, b), a, b) for a, b in merged]
            best = max(sc for sc, _, _ in scored)
            merged = [[a, b] for sc, a, b in scored if sc >= best - margin]
        out, kept = [], []
        for a, b in merged:
            tp, pos = [], a
            for t in full[a:b].split():
                st = full.index(t, pos, b); tp.append((t, st, st + len(t))); pos = st + len(t)
            while tp and tp[0][0].strip(".,;:!?()\"'").lower() in _EDGE: tp.pop(0)
            while tp and tp[-1][0].strip(".,;:!?()\"'").lower() in _EDGE: tp.pop()
            if tp:
                out.append(" ".join(t for t, _, _ in tp)); kept.append((tp[0][1], tp[-1][2]))
        if return_spans:
            return kept
        return " ".join(out).strip()

    @torch.no_grad()
    def z_token_prob(self, query):
        """-> (p, n): the detector's per-query-token P(Z), UNDECODED."""
        raw, full, ids, offs, plen = self._forward(query)
        pz = self._zprob(raw)
        n = min(len(ids), int(raw.shape[0]))
        p = torch.zeros(n, dtype=torch.float32, device=raw.device)
        for i in range(n):
            if ids[i] in self.spec: continue
            a, b = offs[i]
            if b <= a or b <= plen: continue
            p[i] = float(pz[i])
        return p, n

    @torch.no_grad()
    def z_token_mask(self, query):
        """-> (mask, n_tokens): 1.0 on every QUERY TOKEN POSITION the detector labelled Z, else 0."""
        raw, full, ids, offs, plen = self._forward(query)
        sp = self._decode(self._zprob(raw), full, ids, offs, plen, return_spans=True)
        n = min(len(ids), int(raw.shape[0]))
        m = torch.zeros(n, dtype=torch.float32, device=raw.device)
        if not sp:
            return m, n
        for p in range(n):
            if ids[p] in self.spec:
                continue
            a, b = offs[p]
            if b <= a or b <= plen:
                continue
            # overlap, not containment: a Z word can split into sub-tokens whose offsets straddle the edge
            if any(a < e and b > s for s, e in sp):
                m[p] = 1.0
        return m, n

    def _probs(self, raw):
        """Per-token class distribution. EXCISE_VITERBI=1 replaces the independent softmax with the"""
        import torch as _t
        if os.environ.get("EXCISE_VITERBI", "0") == "1" and raw.shape[-1] > 1:
            return self._viterbi(raw)
        return _t.softmax(raw, dim=-1)

    def _xprob(self, raw):
        """P(token is inside the POSITIVE topic span) = P(B-X)+P(I-X). Zero without a 5-class head."""
        import torch as _t
        if not self.xz:
            return _t.zeros(raw.shape[0], device=raw.device)
        pr = self._probs(raw); return pr[:, 3] + pr[:, 4]

    def _viterbi(self, raw):
        """Constrained argmax over the tag sequence, with impossible transitions forbidden."""
        import torch as _t
        n_out = raw.shape[-1]
        lp = _t.log_softmax(raw, dim=-1)
        T = _t.full((n_out, n_out), float("-inf"), device=raw.device)
        starts = [0] + [c for c in range(1, n_out) if c % 2 == 1]      # O and every B-*
        for prev in range(n_out):
            for nxt in starts:
                T[prev, nxt] = 0.0
            if prev in (1, 2): T[prev, 2] = 0.0                        # I-Z may continue a Z span
            if prev in (3, 4) and n_out > 4: T[prev, 4] = 0.0          # I-X may continue an X span
        L = lp.shape[0]
        dp = lp[0].clone(); bp = _t.zeros((L, n_out), dtype=_t.long, device=raw.device)
        for t in range(1, L):
            sc = dp[:, None] + T
            best, arg = sc.max(0)
            dp = best + lp[t]; bp[t] = arg
        path = [int(dp.argmax())]
        for t in range(L - 1, 0, -1):
            path.append(int(bp[t, path[-1]]))
        path.reverse()
        oh = _t.zeros_like(lp)
        oh[_t.arange(L, device=raw.device), _t.tensor(path, device=raw.device)] = 1.0
        return oh                                                      # hard one-hot, read by _zprob/_xprob

    def _zprob(self, raw):
        """P(token is inside an EXCLUDED span), per token."""
        import torch as _t
        if self.xz:
            pr = self._probs(raw); return pr[:, 1] + pr[:, 2]
        if self.iob:
            return 1.0 - self._probs(raw)[:, 0]
        return _t.sigmoid(raw.squeeze(-1))

    @torch.no_grad()
    def fire_score(self, query):
        """Max fire-prob over non-special query tokens = the value compared to thr in extract_Z. For threshold sweeps"""
        bb = self.bb
        ids = bb.tokenizer(bb.query_prefix + query, truncation=True, max_length=bb.query_len)["input_ids"]; Nr = len(ids)
        feat = _query_hidden(bb, [query], self.dev) if self.tap == "hidden768" else bb.encode([query], is_query=True, device=self.dev).vectors
        raw = self.head(feat.float())[0]
        # Same quantity in both modes -- P(this token is inside an excluded span) -- so tau keeps its meaning
        # and every sweep, budget and comparison stays on the same scale across heads.
        fire = self._zprob(raw)
        vals = [float(fire[p]) for p in range(min(Nr, len(ids))) if ids[p] not in self.spec]
        return max(vals) if vals else 0.0
