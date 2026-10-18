"""Train the detector: a query-side LoRA and a span head that tag X and Z (IOB).

  python -m excise.train_detector --data data/benchmark/train/detector.jsonl --out model/checkpoints/<name> --xz --iob"""
from __future__ import annotations
import argparse, json, os, sys, random
# the repository root, so `config` and the packages import from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from excise import config as C   # sets HF_HOME before the backbone loads
import torch, torch.nn as nn, torch.nn.functional as F
from excise.detector_backbone import load_backbone


def excl_spans(named):
    """named = [[span_text, 'exclude'|...], ...] ; return the excluded span strings."""
    return [x[0] for x in (named or []) if isinstance(x, (list, tuple)) and len(x) >= 2 and x[1] == "exclude" and x[0]]


def query_hidden(bb, texts, device):
    """768-d last_hidden_state on the QUERY path (LoRA active), replicating bb.encode(is_query=True)."""
    enc = bb.tokenizer([bb.query_prefix + (t or "") for t in texts], padding="max_length",
                       truncation=True, max_length=bb.query_len, return_tensors="pt")
    ids = enc["input_ids"].to(device); am = enc["attention_mask"].to(device)
    pid, mid = bb.tokenizer.pad_token_id, bb.tokenizer.mask_token_id
    if mid is not None: ids = torch.where(ids == pid, torch.full_like(ids, mid), ids)   # query expansion
    return bb.encoder(input_ids=ids, attention_mask=am).last_hidden_state               # (B, QL, hidden)


def build_iob(bb, instruction, z_spans):
    """-> (n_real_tokens, per-token label in {0:O, 1:B, 2:I})."""
    full = bb.query_prefix + (instruction or "")
    off = len(bb.query_prefix)
    enc = bb.tokenizer(full, truncation=True, max_length=bb.query_len, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    lab = [0] * len(offs)
    for cs, ce in (z_spans or []):
        cs, ce = cs + off, ce + off
        first = True
        for t, (s, e) in enumerate(offs):
            if e <= s or e <= cs or s >= ce:          # empty token, or no overlap with the span
                continue
            lab[t] = 1 if first else 2
            first = False
    return len(offs), lab


def build_xz(bb, instruction, z_spans, x_spans):
    """-> (n_real_tokens, per-token label in {0:O, 1:B-Z, 2:I-Z, 3:B-X, 4:I-X})."""
    full = bb.query_prefix + (instruction or "")
    off = len(bb.query_prefix)
    enc = bb.tokenizer(full, truncation=True, max_length=bb.query_len, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    lab = [0] * len(offs)
    for spans, B, I in ((x_spans or [], 3, 4), (z_spans or [], 1, 2)):
        for cs, ce in spans:
            cs, ce = cs + off, ce + off
            first = True
            for t, (a_, b_) in enumerate(offs):
                if b_ <= a_ or b_ <= cs or a_ >= ce:
                    continue
                lab[t] = B if first else I
                first = False
    return len(offs), lab


def build_labels(bb, instruction, named, fire_words=None):
    """fire_words (in-distribution scheme): whole-word char mask, robust to subwords / recurring words."""
    import re as _re
    full = bb.query_prefix + (instruction or "")
    enc = bb.tokenizer(full, truncation=True, max_length=bb.query_len, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    lab = [0] * len(ids)
    if fire_words:
        low = full.lower(); mark = [False] * len(low)
        for w in fire_words:
            for mt in _re.finditer(r"\b" + _re.escape(w.lower()) + r"\b", low):
                for c in range(mt.start(), mt.end()): mark[c] = True
        for t, (s, e) in enumerate(offs):
            if e > s and any(mark[s:e]): lab[t] = 1
    else:
        for sp in excl_spans(named):
            lo = full.lower().find(sp.lower())
            if lo >= 0:
                hi = lo + len(sp)
                for t, (s, e) in enumerate(offs):
                    if e > s and e > lo and s < hi:
                        lab[t] = 1
    return len(ids), lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=C.DETECTOR_TRAIN)   # NEW multi-source data
    ap.add_argument("--max-train", type=int, default=20000)
    ap.add_argument("--allow-unbalanced", action="store_true",
                    help='permit finder data with no no-fire rows (or no firing rows)')
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-last-k", type=int, default=6); ap.add_argument("--pos-weight", type=float, default=8.0)
    ap.add_argument("--mlp-head", type=int, default=0, metavar="H",
                    help='replace Linear(d,n_out) with Linear(d,H)-GELU-Linear(H,n_out)')
    ap.add_argument("--xz", action="store_true",
                    help='5-class head O/B-Z/I-Z/B-X/I-X: label the POSITIVE topic as well as the exclusion, so the head learns which span is which')
    ap.add_argument("--iob", action="store_true",
                    help='train a 3-class IOB head (O/B/I) over `z_spans` instead of the 1-class bag-of-words head over `fire_words`. The bag says WHICH tokens are excise-worthy and never where a topic starts or stops, which is why the detector fires on a discontiguous cloud, reaches poor span-exact accuracy, and marks the cue word itself as part of the topic')
    ap.add_argument("--cell-balance", action="store_true",
                    help="reweight FIRE rows so (explicitness x length x #exclusions) matches a realistic target instead of the file's own, heavily skewed, empirical shape")
    ap.add_argument("--cell-cap", type=float, default=12.0, help='max reweight factor, so 12-row cells cannot dominate')
    ap.add_argument("--class-balance", type=float, default=0.0,
                    help='target share of FIRE rows in the loss, e.g')
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    ap.add_argument("--tap", choices=["proj128", "hidden768"], default="proj128")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--allow-unverified", action="store_true",
                    help='train on data with no provenance sidecar (you are asserting you know what this file is)')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    from excise.build_train import read_provenance
    meta = read_provenance(a.data, require=not a.allow_unverified)
    if meta: print(f"[deepfinder] data provenance: {meta['builder']} rows={meta['rows']} "
                   f"records={[os.path.basename(r) for r in meta['records']]}", flush=True)
    dev = a.device; torch.manual_seed(a.seed); random.seed(a.seed)
    rows = [json.loads(l) for l in open(a.data)]
    rows = [r for r in rows if r.get("instruction")]                # keep plain rows (named=[]) as no-fire negatives
    random.shuffle(rows); rows = rows[:a.max_train]
    if not rows: raise SystemExit(f"[deepfinder] no usable rows in {a.data}")
    nfire = sum(1 for r in rows if r.get("fire_words"))
    print(f"[deepfinder] {len(rows)} rows  fire={nfire}  no-fire(plain/twin)={len(rows)-nfire}", flush=True)
    if nfire == 0 or nfire == len(rows):
        if not a.allow_unbalanced:
            raise SystemExit("[deepfinder] finder data must contain BOTH firing and no-fire rows "
                             "(no-fire rows are what train the Switch to stay silent). Pass "
                             "--allow-unbalanced only for the firing-rows-only ablation.")
        print("[deepfinder] --allow-unbalanced: training on firing rows only. The Switch has nothing "
              "teaching it to stay silent and is expected to fire on everything.", flush=True)
    bb = load_backbone(device=dev, lora=True, lora_r=a.lora_r, lora_last_k=a.lora_last_k)
    hdim = bb.hidden_dim if a.tap == "hidden768" else bb.proj_dim
    if a.xz: a.iob = True                                    # 5-class is IOB plus the X classes
    n_out = 5 if a.xz else (3 if a.iob else 1)
    head = (nn.Sequential(nn.Linear(hdim, a.mlp_head), nn.GELU(), nn.Linear(a.mlp_head, n_out))
            if a.mlp_head else nn.Linear(hdim, n_out)).to(dev)
    print(f"[deepfinder] tap={a.tap} head_dim={hdim}", flush=True)
    QL = bb.query_len
    Ns, Labs = [], []                                            # per-row real-length + padded labels
    if a.iob:
        drop = [r for r in rows if r.get("fire_words") and not r.get("iob_ok", False)]
        if drop:
            print(f"[deepfinder] IOB: {len(drop)} fire rows have no derivable span (iob_ok=False) -> "
                  f"excluded from this run; they keep their rows and their fire_words", flush=True)
            rows = [r for r in rows if not (r.get("fire_words") and not r.get("iob_ok", False))]
        miss = sum(1 for r in rows if r.get("fire_words") and not r.get("z_spans"))
        if miss:
            sys.exit(f"--iob needs z_spans on every fire row; {miss} lack it. "
                     f"The released detector data carries them.")
    for r in rows:
        if a.xz:
            N, lab = build_xz(bb, r["instruction"], r.get("z_spans"), r.get("x_spans"))
        elif a.iob:
            N, lab = build_iob(bb, r["instruction"], r.get("z_spans"))
        else:
            N, lab = build_labels(bb, r["instruction"], r.get("named"), r.get("fire_words"))
        lab = lab[:QL] + [0] * (QL - len(lab)); Ns.append(min(N, QL)); Labs.append(lab)
    Labs = torch.tensor(Labs, dtype=(torch.long if a.iob else torch.float), device=dev)
    Ns = torch.tensor(Ns, device=dev)
    if a.iob:
        _b = int((Labs == 1).sum()); _i = int((Labs == 2).sum()); _o = int((Labs == 0).sum())
        if a.xz:
            _xb = int((Labs == 3).sum()); _xi = int((Labs == 4).sum())
            _nx = sum(1 for r in rows if r.get("fire_words") and not r.get("x_ok"))
            print(f"[deepfinder] XZ tokens: O={_o} B-Z={_b} I-Z={_i} B-X={_xb} I-X={_xi}  "
                  f"({len(rows)} rows; {_nx} fire rows have no derivable X and train X-as-O)", flush=True)
        else:
            print(f"[deepfinder] IOB tokens: O={_o} B={_b} I={_i}  ({len(rows)} rows)", flush=True)
    posw = torch.tensor(a.pos_weight, device=dev)
    # ROW weights. Duplicating a fire row makes its tokens count twice in the loss; weighting it by 2 does
    # exactly the same thing without storing the row twice, and without the file carrying repeated text.
    is_fire = torch.tensor([1.0 if r.get("fire_words") else 0.0 for r in rows], device=dev)
    roww = torch.ones(len(rows), device=dev)
    if any(r.get("row_weight") for r in rows):
        rw = torch.tensor([float(r.get("row_weight", 1.0)) for r in rows], device=dev)
        roww = roww * rw
        print(f"[deepfinder] row_weight honoured on {int((rw != 1).sum())} rows "
              f"(total mass {float(rw.sum()):.0f} over {len(rows)} rows)", flush=True)
    if a.class_balance:
        n_fire = float(is_fire.sum()); n_none = float(len(rows)) - n_fire   # NOT `nn` -- that is torch.nn
        t = a.class_balance
        w = (t / (1 - t)) * (n_none / max(n_fire, 1))   # solve  w*n_fire / (w*n_fire + n_none) = t
        roww = torch.where(is_fire > 0, torch.full_like(roww, w), roww)
        print(f"[deepfinder] class-balance {t}: {int(n_fire)} fire rows weighted x{w:.3f} against "
              f"{int(n_none)} no-fire (no duplication)", flush=True)
    if a.cell_balance:
        import re as _re2
        from excise.schema import NEG_CUES as _NC
        _cue = _re2.compile("|".join(_NC), _re2.I)
        _MK = ("already", "covered", "well-known", "familiar", "discussed", "beyond", "setting aside",
               "leaving aside", "well-documented")
        _mk = _re2.compile("|".join(_re2.escape(m) for m in _MK), _re2.I)
        P_LEN = {"short": 0.40, "mid": 0.40, "long": 0.20}
        P_EXP = {"cue": 0.45, "marker": 0.35, "bare": 0.20}
        P_NEX = {"1": 0.60, "2": 0.30, "3+": 0.10}

        def _cell(r):
            q = r.get("instruction") or ""
            e = "cue" if _cue.search(q) else ("marker" if _mk.search(q) else "bare")
            w = len(q.split())
            l = "long" if w > 60 else ("mid" if w > 25 else "short")
            nz = len(r.get("Z") or [])
            return (e, l, "1" if nz <= 1 else ("2" if nz == 2 else "3+"))

        from collections import Counter as _C
        fidx = [i for i, r in enumerate(rows) if r.get("fire_words")]
        cells = {i: _cell(rows[i]) for i in fidx}
        emp = _C(cells.values())
        tgt = {c: P_EXP[c[0]] * P_LEN[c[1]] * P_NEX[c[2]] for c in emp}
        z = sum(tgt.values())
        nf = float(len(fidx))
        wts = {}
        for c, n in emp.items():
            want = tgt[c] / z                                  # renormalised over cells that HAVE rows
            wts[c] = min(a.cell_cap, max(1.0 / a.cell_cap, want / (n / nf)))
        mean_w = sum(wts[cells[i]] for i in fidx) / max(nf, 1)  # keep total fire mass unchanged
        for i in fidx:
            roww[i] = roww[i] * wts[cells[i]] / mean_w
        eff = _C()
        for i in fidx:
            eff[cells[i]] += float(roww[i])
        tot_eff = sum(eff.values())
        print(f"[deepfinder] cell-balance on {len(fidx)} fire rows (cap x{a.cell_cap:g}); "
              f"top shifts empirical -> effective:", flush=True)
        for c in sorted(emp, key=lambda c: -abs(eff[c] / tot_eff - emp[c] / nf))[:8]:
            print(f"    {'/'.join(c):<20s} {emp[c]/nf:6.1%} -> {eff[c]/tot_eff:6.1%}  (x{wts[c]:.2f})", flush=True)

    trainable = [p for p in bb.parameters() if p.requires_grad] + list(head.parameters())
    print(f"[deepfinder] trainable={sum(p.numel() for p in trainable)}", flush=True)
    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=0.01)
    N = len(rows); nb = max(1, N // a.bs)          # N < bs would give nb=0 -> ZeroDivisionError + no training at all
    for ep in range(a.epochs):
        bb.train(); head.train(); perm = torch.randperm(N).tolist(); tot = 0.0
        for b in range(nb):
            idx = perm[b * a.bs:(b + 1) * a.bs]; ii = torch.tensor(idx, device=dev)
            if a.tap == "hidden768":
                feat = query_hidden(bb, [rows[i]["instruction"] for i in idx], dev)            # (B, QL, 768) LoRA, grad
            else:
                feat = bb.encode([rows[i]["instruction"] for i in idx], is_query=True, device=dev).vectors  # (B,QL,128)
            raw = head(feat.float())                                                           # (B, QL, n_out)
            QLb = raw.shape[1]
            posn = torch.arange(QLb, device=dev)[None, :] < Ns[ii][:, None]                    # real-token mask
            lab = Labs[ii][:, :QLb]
            if a.iob:
                cw = torch.tensor([1.0] + [float(a.pos_weight)] * (n_out - 1), device=dev)
                per = F.cross_entropy(raw[posn], lab[posn], weight=cw, reduction="none")
                loss = (per * roww[ii][:, None].expand(-1, QLb)[posn]).mean()
            else:
                logit = raw.squeeze(-1)
                w = roww[ii][:, None].expand_as(logit)[posn]
                loss = F.binary_cross_entropy_with_logits(logit[posn], lab[posn].float(),
                                                          pos_weight=posw, weight=w)
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
        with torch.no_grad():
            if a.iob:
                pr = raw.argmax(-1)                    # `logit` does not exist in IOB mode
                # Z-token recall ONLY, even in 5-class mode: counting X hits here would inflate the
                # number against every 3-class run in the log and make the two incomparable.
                zt = (lab == 1) | (lab == 2)
                zp = (pr == 1) | (pr == 2)
                tp = (zp & zt & posn).sum().item()
                fire_recall = tp / max((zt & posn).sum().item(), 1)
            else:
                pred = (torch.sigmoid(logit) > 0.5) & posn; tp = (pred & (lab > 0.5)).sum().item()
                fire_recall = tp / max((lab > 0.5)[posn].sum().item(), 1)
        print(f"[deepfinder] ep{ep} loss={tot/nb:.4f} Z-token-recall(last-batch)={fire_recall:.3f}", flush=True)
    sd = {n: p.detach().cpu() for n, p in bb.named_parameters() if p.requires_grad}
    torch.save({"lora": sd, "head": head.state_dict(), "cfg": {"lora_r": a.lora_r, "lora_last_k": a.lora_last_k, "tap": a.tap, "iob": bool(a.iob), "xz": bool(a.xz), "mlp_head": int(a.mlp_head),
                    "query_len": int(bb.query_len)},
                "args": vars(a)}, os.path.join(a.out, "deepfinder.pt"))
    print(f"[deepfinder] saved -> {a.out}/deepfinder.pt", flush=True)


if __name__ == "__main__":
    main()
