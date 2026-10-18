"""Fine-tune an external dense retriever or cross-encoder reranker on the 1,900 retrieval triples, with
each model's own query and document prefixes.

  python -m competitors.train_external --model bge-rerank --out model/checkpoints/<name>"""
from __future__ import annotations
import torch
import argparse, json, os, sys
# the repository root, so `config` and the packages import from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from excise import config as C  # noqa: F401  (pins HF caches before any HF import)

from competitors.evaluate_external import MODELS

OUT_ROOT = os.path.join(C.ROOT, "model", "checkpoints", "strong_ft")


def _per_device(bs: int) -> int:
    """`bs` is the EFFECTIVE batch; return the per-device value that preserves it."""
    n = max(1, torch.cuda.device_count())
    if bs % n:
        raise SystemExit(f"batch {bs} is the EFFECTIVE batch and does not divide over {n} visible GPUs. "
                         f"Pick a multiple of {n}, or run on one card (CUDA_VISIBLE_DEVICES=0).")
    return bs // n


def load_triples(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    for r in rows:
        assert {"query", "positive", "negative"} <= r.keys(), f"bad triple: {list(r)}"
    return rows



def train_dense(m, triples, out, epochs, bs, lr):
    if m.get("peft_base"):
        raise SystemExit("[train_external] this model is PEFT over %s -- train it with "
                         "competitors.train_promptriever, which reproduces its pooling conventions."
                         % m["peft_base"])
    import torch
    from datasets import Dataset
    from sentence_transformers import (SentenceTransformer, SentenceTransformerTrainer,
                                       SentenceTransformerTrainingArguments)
    from sentence_transformers.losses import MultipleNegativesRankingLoss
    kw = {"trust_remote_code": True} if m.get("trust_remote_code") else {}
    model = SentenceTransformer(m["repo"], device="cuda", **kw)
    if m.get("max_seq_length"):
        model.max_seq_length = m["max_seq_length"]
    qp, dp = m["qp"], m["dp"]
    ds = Dataset.from_dict({
        "anchor":   [qp + t["query"] for t in triples],
        "positive": [dp + (t["positive"] or "") for t in triples],
        "negative": [dp + (t["negative"] or "") for t in triples],
    })
    if m.get("grad_ckpt"):
        from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
        loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=4)
        print("[train_external] loss=CachedMultipleNegativesRankingLoss(mini_batch_size=4) "
              "gradient_checkpointing=True", flush=True)
    else:
        loss = MultipleNegativesRankingLoss(model)  # in-batch negatives + the provided hard negative
        print("[train_external] loss=MultipleNegativesRankingLoss", flush=True)
    args = SentenceTransformerTrainingArguments(
        output_dir=os.path.join(out, "_trainer"),
        num_train_epochs=epochs, per_device_train_batch_size=_per_device(bs), learning_rate=lr,
        warmup_ratio=0.1, bf16=torch.cuda.is_available(), fp16=False,
        gradient_checkpointing=m.get("grad_ckpt", False),
        logging_steps=20, save_strategy="no", report_to=[], dataloader_drop_last=True,
    )
    SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=loss).train()
    model.save_pretrained(out)
    print(f"[train_external] saved dense FT -> {out}  (n={len(triples)}, ep={epochs}, bs={bs}, lr={lr})")


def train_crossencoder(m, triples, out, epochs, bs, lr):
    """bge-reranker cross-encoder: (query, pos)->1, (query, neg)->0, BCE."""
    import torch
    from datasets import Dataset
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    model = CrossEncoder(m["repo"], num_labels=1, max_length=512, device="cuda")
    q, d, y = [], [], []
    for t in triples:
        q += [t["query"], t["query"]]
        d += [t["positive"] or "", t["negative"] or ""]
        y += [1.0, 0.0]
    ds = Dataset.from_dict({"query": q, "passage": d, "label": y})
    args = CrossEncoderTrainingArguments(
        output_dir=os.path.join(out, "_trainer"),
        num_train_epochs=epochs, per_device_train_batch_size=_per_device(bs), learning_rate=lr,
        warmup_ratio=0.1, bf16=torch.cuda.is_available(), fp16=False,
        gradient_checkpointing=m.get("grad_ckpt", False),
        logging_steps=20, save_strategy="no", report_to=[],
    )
    CrossEncoderTrainer(model=model, args=args, train_dataset=ds,
                        loss=BinaryCrossEntropyLoss(model)).train()
    model.save_pretrained(out)
    print(f"[train_external] saved cross-encoder FT -> {out}  (pairs={len(q)}, ep={epochs}, bs={bs}, lr={lr})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--triples", default=C.BASELINE_TRIPLES,
                    help='the 1,900 retrieval triples every trainable system trains on')
    ap.add_argument("--out", default="",
                    help='checkpoint directory')
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    a = ap.parse_args()
    m = MODELS[a.model]
    out = a.out or os.path.join(OUT_ROOT, a.model)
    os.makedirs(out, exist_ok=True)
    triples = load_triples(a.triples)
    print(f"[train_external] model={a.model} mode={m['mode']} triples={len(triples)} from {a.triples}")
    if m["mode"] == "dense":
        train_dense(m, triples, out, a.epochs, a.bs, a.lr)
    else:
        train_crossencoder(m, triples, out, a.epochs, max(8, a.bs // 2), a.lr)
    print(f"STRONG_FT_TRAIN_DONE model={a.model}")


if __name__ == "__main__":
    main()
