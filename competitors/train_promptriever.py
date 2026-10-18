"""Fine-tune Promptriever (a LoRA over Mistral-7B-v0.1 read as a bi-encoder) on the 1,900 retrieval triples."""
from __future__ import annotations
import argparse, json, os, sys, random

# the repository root, so `config` and the packages import from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from excise import config as C

REPO = "samaya-ai/promptriever-mistral-v0.1-7b-v1"
BASE = "mistralai/Mistral-7B-v0.1"
QP, DP = "query: ", "passage: "


def pool(model, tok, texts, dev, maxlen, train=True):
    """Encode with the EXACT convention competitors.evaluate_external.PromptrieverEncoder uses."""
    import torch
    b = [t + "</s>" for t in texts]
    enc = tok(b, padding=True, truncation=True, max_length=maxlen, return_tensors="pt").to(dev)
    hs = model(**enc).last_hidden_state
    last = enc["attention_mask"].sum(dim=1) - 1
    h = hs[torch.arange(hs.size(0), device=hs.device), last]
    return torch.nn.functional.normalize(h.float(), p=2, dim=-1)


def run(triples, lr, epochs, bs, maxlen, out, dev, seed=0, eval_only_frac=0.0):
    import torch
    from transformers import AutoTokenizer, AutoModel
    from peft import PeftModel
    torch.manual_seed(seed); random.seed(seed)
    tok = AutoTokenizer.from_pretrained(REPO)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    base = AutoModel.from_pretrained(BASE, torch_dtype=torch.bfloat16)
    # is_trainable=True continues the SHIPPED adapter rather than stacking a second one, which keeps the
    # instruction-following behaviour the row exists to test instead of layering a fresh head over it.
    model = PeftModel.from_pretrained(base, REPO, is_trainable=True).to(dev)
    model.enable_input_require_grads()
    if hasattr(model, "config"): model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()
    assert any(p.requires_grad for p in model.parameters()), "no trainable parameters"
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[pr-ft] trainable params: {sum(p.numel() for p in params)/1e6:.2f}M  lr={lr}", flush=True)
    opt = torch.optim.AdamW(params, lr=lr)
    n_hold = int(len(triples) * eval_only_frac)
    hold, tr = triples[:n_hold], triples[n_hold:]
    for ep in range(epochs):
        random.shuffle(tr)
        tot = nb = 0.0
        for i in range(0, len(tr), bs):
            ch = tr[i:i + bs]
            if len(ch) < 2: continue
            q = pool(model, tok, [QP + r["query"] for r in ch], dev, maxlen)
            d = pool(model, tok, [DP + r["positive"] for r in ch] + [DP + r["negative"] for r in ch], dev, maxlen)
            logits = q @ d.T * 20.0
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(ch), device=dev))
            if not loss.requires_grad:
                raise RuntimeError("loss is detached from the graph -- checkpointing/PEFT wiring is wrong")
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
            if nb % 20 == 0: print(f"  ep{ep} step{int(nb)} loss={tot/nb:.4f}", flush=True)
        print(f"[pr-ft] epoch {ep} mean loss {tot/max(nb,1):.4f}", flush=True)
    acc = None
    if hold:
        model.eval(); ok = 0
        with torch.no_grad():
            for i in range(0, len(hold), bs):
                ch = hold[i:i + bs]
                q = pool(model, tok, [QP + r["query"] for r in ch], dev, maxlen)
                p_ = pool(model, tok, [DP + r["positive"] for r in ch], dev, maxlen)
                n_ = pool(model, tok, [DP + r["negative"] for r in ch], dev, maxlen)
                ok += int(((q * p_).sum(-1) > (q * n_).sum(-1)).sum().item())
        acc = ok / max(len(hold), 1)
        print(f"[pr-ft] held-out triple accuracy @ lr={lr}: {acc:.4f}  (n={len(hold)})", flush=True)
        model.train()
    if out:
        os.makedirs(out, exist_ok=True)
        model.save_pretrained(out); tok.save_pretrained(out)
        print(f"[pr-ft] saved adapter -> {out}", flush=True)
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", default=C.BASELINE_TRIPLES,
                    help='the 1,900 retrieval triples every trainable system trains on')
    ap.add_argument("--out", default="")
    ap.add_argument("--lr", type=float, default=0.0, help='0 = run the held-out LR selection first')
    ap.add_argument("--lrs", default="1e-4,5e-5,2e-5")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--hold-frac", type=float, default=0.1)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.triples) if l.strip()]
    for r in rows: assert {"query", "positive", "negative"} <= r.keys(), f"bad triple: {list(r)}"
    print(f"[pr-ft] {len(rows)} triples from {a.triples}", flush=True)
    dev = "cuda:0"
    lr = a.lr
    if lr <= 0:
        best, bacc = None, -1.0
        for cand in [float(x) for x in a.lrs.split(",")]:
            acc = run(rows, cand, a.epochs, a.bs, a.max_len, "", dev, eval_only_frac=a.hold_frac)
            if acc is not None and acc > bacc: best, bacc = cand, acc
        lr = best
        print(f"[pr-ft] SELECTED lr={lr} (held-out acc {bacc:.4f})", flush=True)
    run(rows, lr, a.epochs, a.bs, a.max_len, a.out, dev, eval_only_frac=0.0)


if __name__ == "__main__":
    main()
