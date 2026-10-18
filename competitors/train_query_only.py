"""Train the query-only baseline: a LoRA adapter on the query side only, scored against the frozen index,
so no re-indexing is needed.

  python -m competitors.train_query_only --triples data/benchmark/train/retrieval_triples.jsonl --out model/checkpoints/<name>"""
from __future__ import annotations
import argparse, json, os, sys, random
# the repository root, so `config` and the packages import from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from excise import config as C
import torch
from peft import LoraConfig
from pylate import models
from excise.encoder import MODEL, _QL, load_model, emb
from excise.lora_targets import detect_lora_targets
from excise.maxsim import weighted_maxsim_matrix as maxsim


def pad_stack(vs, ms, dev):
    T = max(v.shape[0] for v in vs); P = vs[0].shape[1]
    V = torch.zeros(len(vs), T, P, device=dev); M = torch.zeros(len(vs), T, dtype=torch.bool, device=dev)
    for i, (v, m) in enumerate(zip(vs, ms)):
        V[i, :v.shape[0]] = v; M[i, :m.shape[0]] = m
    return V, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", default=C.BASELINE_TRIPLES,
                    help='the 1,900 retrieval triples every trainable system trains on')
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)
    dev = "cuda:0"
    os.makedirs(a.out, exist_ok=True)

    rows = [json.loads(l) for l in open(a.triples) if l.strip()]
    print(f"[qonly] {len(rows)} triples from {a.triples}", flush=True)

    # ---- documents: encoded ONCE with the FROZEN encoder, never adapted (this is the whole point) ----
    frozen = load_model("", dev)
    with torch.no_grad():
        pv, pm = emb(frozen, [r["positive"] or "" for r in rows], False, dev)
        nv, nm = emb(frozen, [r["negative"] or "" for r in rows], False, dev)
    pv, nv = pv.float(), nv.float()
    # pos/neg are padded independently by emb(), so their token dimensions differ (e.g. 2127 vs 1517). Pad both to
    # a common length or the per-batch torch.cat below fails. Padded tokens carry mask=0 and are excluded by MaxSim.
    T = max(pv.shape[1], nv.shape[1])

    def pad_T(V, M):
        if V.shape[1] == T:
            return V, M
        k = T - V.shape[1]
        return (torch.cat([V, torch.zeros(V.shape[0], k, V.shape[2], device=V.device)], 1),
                torch.cat([M, torch.zeros(M.shape[0], k, dtype=torch.bool, device=M.device)], 1))
    pv, pm = pad_T(pv, pm); nv, nm = pad_T(nv, nm)
    print(f"[qonly] frozen doc vectors: pos{tuple(pv.shape)} neg{tuple(nv.shape)} (padded to T={T})", flush=True)
    del frozen
    torch.cuda.empty_cache()

    # ---- query encoder: trainable LoRA on top of the same backbone ----
    m = models.ColBERT(MODEL, model_kwargs={"attn_implementation": "sdpa"}, **_QL)
    if hasattr(m, "_input_length") and not hasattr(m, "_text_length"): m._text_length = m._input_length
    m.add_adapter(LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, target_modules=detect_lora_targets(m),
                             lora_dropout=0.0, bias="none"))
    m = m.to(dev); m.train()
    params = [p for p in m.parameters() if p.requires_grad]
    print(f"[qonly] trainable tensors: {len(params)} ({sum(p.numel() for p in params)/1e6:.2f}M params)", flush=True)
    opt = torch.optim.AdamW(params, lr=a.lr)

    idx = list(range(len(rows)))
    step = 0
    for ep in range(a.epochs):
        random.shuffle(idx)
        for s in range(0, len(idx) - a.batch_size + 1, a.batch_size):
            b = idx[s:s + a.batch_size]
            feats = m.tokenize([rows[i]["query"] for i in b], is_query=True)
            feats = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in feats.items()}
            qv = m(feats)["token_embeddings"].float()
            qw = feats["attention_mask"].float()
            # in-batch contrastive: the gold of each query must outrank every other doc AND its own hard negative
            D = torch.cat([pv[b], nv[b]], 0); DM = torch.cat([pm[b], nm[b]], 0)
            sc = maxsim(qv, qw, D, DM)                                   # (B, 2B)
            loss = torch.nn.functional.cross_entropy(sc, torch.arange(len(b), device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % 20 == 0: print(f"[qonly] ep{ep} step{step} loss={loss.item():.4f}", flush=True)

    lora_sd = {k: v.detach().cpu() for k, v in m.state_dict().items() if "lora" in k.lower()}
    torch.save({"lora": lora_sd, "args": vars(a)}, os.path.join(a.out, "lora.pt"))
    print(f"[qonly] saved {len(lora_sd)} LoRA tensors -> {a.out}/lora.pt", flush=True)
    print("QONLY_TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
