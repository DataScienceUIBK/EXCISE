"""Train the exclusion adapter (contrastive loss plus the Z margin) or a fine-tuned baseline, with PyLate.

  python -m excise.train_adapter --triples data/benchmark/train/adapter_triples.jsonl --out model/checkpoints/<name> --z-margin 0.10 --z-weight 5.0
  python -m excise.train_adapter --triples data/benchmark/train/retrieval_triples.jsonl --out model/checkpoints/<name>   # LoRA baseline; --full for the full fine-tune"""
from __future__ import annotations
import argparse, os, sys, random
# the repository root, so `config` and the packages import from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from excise import config as C   # sets HF_HOME before pylate/transformers
import torch
from datasets import Dataset
from peft import LoraConfig
from pylate import models, losses
from pylate.utils import ColBERTCollator
from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments

MODEL = C.BACKBONE


class DocPadCollator:
    """Wrap ColBERTCollator: pad positive/negative DOCUMENT tensors to a common length"""
    SUF = ("_input_ids", "_attention_mask", "_token_type_ids")

    def __init__(self, base):
        self.base = base

    def __getattr__(self, name):           # forward introspection (valid_label_columns, etc.) to base
        if name == "base":
            raise AttributeError(name)
        return getattr(self.base, name)

    def __call__(self, feats):
        b = self.base(feats)
        keys = [k for k in b if isinstance(b[k], torch.Tensor) and any(k.endswith(s) for s in self.SUF)
                and not (k.startswith("query") or k.startswith("anchor"))]
        if not keys:
            return b
        maxT = max(b[k].shape[1] for k in keys)
        for k in keys:
            v = b[k]; T = v.shape[1]
            if T < maxT:
                b[k] = torch.cat([v, torch.zeros(v.shape[0], maxT - T, dtype=v.dtype)], dim=1)
        return b


def load_triples_jsonl_z(path: str):
    """Triples that carry a `z_query` column (excise/label_z.py)."""
    import json as _j
    q, p_, n, z, lab = [], [], [], [], []
    for line in open(path):
        r = _j.loads(line)
        if not all(r.get(k) for k in ("query", "positive", "negative", "z_query")):
            continue
        q.append(r["query"]); p_.append(r["positive"]); n.append(r["negative"]); z.append(r["z_query"])
        lab.append(float(r.get("label", 1.0)))
    return Dataset.from_dict({"query": q, "positive": p_, "negative": n, "z_query": z, "label": lab})


def load_triples_jsonl(path: str):
    import json
    q, p, n = [], [], []; bad = 0
    for line in open(path):
        r = json.loads(line)
        if not (r.get("query") and r.get("positive") and r.get("negative")): bad += 1; continue
        q.append(r["query"]); p.append(r["positive"]); n.append(r["negative"])
    if not q: raise SystemExit(f"[pyft] {path}: 0 usable triples ({bad} malformed)")
    if bad: print(f"[pyft] WARNING dropped {bad} malformed rows ({bad/(bad+len(q)):.1%})", flush=True)
    print(f"[pyft] loaded {len(q)} triples from {path}", flush=True)
    return Dataset.from_dict({"query": q, "positive": p, "negative": n})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=16,
                    help='EFFECTIVE batch across all visible GPUs, not per-device')
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--full", action="store_true", help='FULL fine-tune the whole backbone (no LoRA adapter)')
    ap.add_argument("--triples", default=C.BASELINE_TRIPLES,
                    help='the (query, positive, negative) training file')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--z-margin", type=float, default=0.0,
                    help='train the channel the OPERATOR demotes by')
    ap.add_argument("--z-weight", type=float, default=1.0, help='weight on the Z-margin term')
    ap.add_argument("--z-no-mask", action="store_true",
                    help='ABLATION: score the Z-margin hinge on every row, including the ones with no exclusion whose z_query is a placeholder')
    ap.add_argument("--allow-unverified", action="store_true",
                    help='train on triples with no provenance sidecar (you are asserting you know what this file is)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from excise.build_train import read_provenance
    meta = read_provenance(a.triples, require=not a.allow_unverified)
    if meta: print(f"[pyft] data provenance: {meta['builder']} rows={meta['rows']} "
                   f"records={[os.path.basename(r) for r in meta['records']]}", flush=True)

    m = models.ColBERT(MODEL, model_kwargs={"attn_implementation": "sdpa"})
    m.document_length = 256   # fixed length -> uniform doc padding (avoids PyLate's
                              # batch-max stack mismatch on variable-length negatives)
    if a.full:
        for p in m.parameters():
            p.requires_grad_(True)
    else:
        from excise.lora_targets import detect_lora_targets
        m.add_adapter(LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r,
                                 target_modules=detect_lora_targets(m), lora_dropout=0.0, bias="none"))
    n_tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    if a.z_margin > 0:
        ds = load_triples_jsonl_z(a.triples)
        if "z_query" not in ds.column_names:
            raise SystemExit("--z-margin needs a z_query column; build it with excise/label_z.py")
    else:
        ds = load_triples_jsonl(a.triples)
    print(f"[pyft] data={a.triples} triples={len(ds)} trainable={n_tr} full={a.full} "
          f"z_margin={a.z_margin} z_weight={a.z_weight} cols={ds.column_names}", flush=True)

    _base = losses.Contrastive(model=m)
    if a.z_margin > 0:
        from excise.z_margin_loss import ContrastiveWithZMargin
        _loss = ContrastiveWithZMargin(m, _base, margin=a.z_margin, weight=a.z_weight,
                                       mask=not a.z_no_mask)
        print(f"[pyft] Z-MARGIN ON: relu({a.z_margin} - (zsim(Z,neg) - zsim(Z,pos))) * {a.z_weight}", flush=True)
    else:
        _loss = _base

    _ngpu = max(1, torch.cuda.device_count())
    if a.batch_size % _ngpu:
        raise SystemExit(f"--batch-size {a.batch_size} is the EFFECTIVE batch and does not divide over "
                         f"{_ngpu} visible GPUs. Pick a multiple of {_ngpu}, or run on one card "
                         f"(CUDA_VISIBLE_DEVICES=0).")
    _per_dev = a.batch_size // _ngpu
    print(f"[pyft] batch: effective={a.batch_size} over {_ngpu} GPU(s) -> per_device={_per_dev}; "
          f"steps/epoch={len(ds)//a.batch_size}", flush=True)
    args = SentenceTransformerTrainingArguments(
        output_dir=os.path.join(a.out, "_hf"), per_device_train_batch_size=_per_dev,
        num_train_epochs=a.epochs, learning_rate=a.lr, warmup_ratio=0.05,
        bf16=True, fp16=False, save_strategy="no", logging_steps=25, report_to=[],
        dataloader_drop_last=True, seed=a.seed)
    trainer = SentenceTransformerTrainer(model=m, args=args, train_dataset=ds,
                                         loss=_loss,
                                         data_collator=DocPadCollator(ColBERTCollator(tokenize_fn=m.tokenize)))
    trainer.train()
    if a.full:
        # save the WHOLE model state (avoid pylate's buggy full save); load via strict=False
        full_sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}
        torch.save({"full": full_sd, "args": vars(a)}, os.path.join(a.out, "lora.pt"))
        print(f"[pyft] saved FULL model ({len(full_sd)} tensors) -> {a.out}/lora.pt", flush=True)
    else:
        # save ONLY the LoRA adapter weights (avoid pylate's buggy full save)
        lora_sd = {k: v.detach().cpu() for k, v in m.state_dict().items() if "lora" in k.lower()}
        torch.save({"lora": lora_sd, "args": vars(a)}, os.path.join(a.out, "lora.pt"))
        print(f"[pyft] saved {len(lora_sd)} LoRA tensors -> {a.out}/lora.pt", flush=True)


if __name__ == "__main__":
    main()
