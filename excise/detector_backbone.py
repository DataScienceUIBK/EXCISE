"""The ColBERT encoder the detector is trained on top of."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer


REPO_ID = "lightonai/Reason-ModernColBERT"

# Defaults pulled from config_sentence_transformers.json. Centralised here
# so changes to the published checkpoint surface in one place.
DEFAULT_QUERY_PREFIX = "[Q] "
DEFAULT_DOC_PREFIX = "[D] "
DEFAULT_QUERY_LEN = 128
DEFAULT_DOC_LEN = 8192
DEFAULT_PROJ_DIM = 128
DEFAULT_HIDDEN = 768

DEFAULT_SKIPLIST = list("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


@dataclass
class EncodedBatch:
    """L2-normalised per-token ColBERT vectors plus an attention mask."""

    vectors: torch.Tensor    # (B, T, proj_dim), float
    mask: torch.Tensor       # (B, T), bool — True for tokens that participate

    def __len__(self) -> int:
        return self.vectors.shape[0]


def _resolve_local_dir(repo_id: str = REPO_ID, cache_dir: str | None = None) -> Path:
    local = snapshot_download(repo_id=repo_id, cache_dir=cache_dir)
    return Path(local)


def _native_linear(local_dir: Path) -> dict:
    """The 768->proj projection of a checkpoint that has no `1_Dense/` directory."""
    for name in ("model.safetensors", "pytorch_model.bin"):
        f = local_dir / name
        if not f.exists():
            continue
        sd = load_file(f) if name.endswith(".safetensors") else torch.load(f, map_location="cpu")
        out = {k: v for k, v in sd.items() if k in ("linear.weight", "linear.bias")}
        if "linear.weight" in out:
            return out
        raise RuntimeError(
            f"{f}: no `1_Dense/` directory and no `linear.weight` in the checkpoint -- cannot recover the "
            f"ColBERT projection. Refusing to initialise it at random.")
    raise FileNotFoundError(
        f"{local_dir}: no 1_Dense/config.json and no model weights file to recover `linear.weight` from.")

class ColBERTBackbone(nn.Module):
    """Frozen ModernBERT + Dense projection in ColBERT semantics."""

    def __init__(
        self,
        local_dir: str | Path | None = None,
        repo_id: str = REPO_ID,
        cache_dir: str | None = None,
        attn_implementation: str = "sdpa",
        torch_dtype: torch.dtype | None = torch.bfloat16,
    ) -> None:
        super().__init__()
        if local_dir is None:
            local_dir = _resolve_local_dir(repo_id=repo_id, cache_dir=cache_dir)
        local_dir = Path(local_dir)

        # Tokenizer & encoder.
        self.tokenizer = AutoTokenizer.from_pretrained(local_dir)
        kwargs: dict = {}
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        self.encoder = AutoModel.from_pretrained(local_dir, **kwargs)

        dense_cfg_path = local_dir / "1_Dense" / "config.json"
        if dense_cfg_path.exists():
            dense_cfg = json.loads(dense_cfg_path.read_text())
            dense_state = load_file(local_dir / "1_Dense" / "model.safetensors")
        else:
            dense_state = _native_linear(local_dir)
            w = dense_state["linear.weight"]
            dense_cfg = {"in_features": int(w.shape[1]), "out_features": int(w.shape[0]),
                         "bias": "linear.bias" in dense_state}
            print(f"[backbone] {local_dir.name}: no 1_Dense/ -- projecting with the checkpoint's own "
                  f"`linear.weight` {tuple(w.shape)} (native ColBERT layout)", flush=True)
        self.dense = nn.Linear(
            dense_cfg["in_features"],
            dense_cfg["out_features"],
            bias=bool(dense_cfg.get("bias", False)),
        )
        # pylate stores the weight as `linear.weight` (no bias).
        weight_key = "linear.weight" if "linear.weight" in dense_state else next(iter(dense_state))
        self.dense.weight.data = dense_state[weight_key].to(self.dense.weight.dtype)
        if dense_cfg.get("bias", False):
            self.dense.bias.data = dense_state.get("linear.bias", torch.zeros_like(self.dense.bias))

        # Sentence-transformers config for prefixes / lengths / skiplist.
        st_cfg_path = local_dir / "config_sentence_transformers.json"
        st_cfg = json.loads(st_cfg_path.read_text()) if st_cfg_path.exists() else {}
        self.query_prefix = st_cfg.get("query_prefix", DEFAULT_QUERY_PREFIX)
        self.doc_prefix = st_cfg.get("document_prefix", DEFAULT_DOC_PREFIX)
        self.query_len = int(os.environ.get("EXCISE_QUERY_LEN") or st_cfg.get("query_length", DEFAULT_QUERY_LEN))
        self.doc_len = int(st_cfg.get("document_length", DEFAULT_DOC_LEN))
        _maxpos = int(getattr(self.encoder.config, "max_position_embeddings", 0) or 0)
        if _maxpos and self.doc_len > _maxpos:
            print(f"[backbone] doc_len {self.doc_len} exceeds the encoder's {_maxpos} position embeddings; "
                  f"capping at {_maxpos}", flush=True)
            self.doc_len = _maxpos
        self.proj_dim = self.dense.out_features
        self.hidden_dim = self.dense.in_features
        self.attend_to_expansion = bool(st_cfg.get("attend_to_expansion_tokens", False))

        skiplist_words: Iterable[str] = st_cfg.get("skiplist_words", DEFAULT_SKIPLIST)
        skiplist_ids: set[int] = set()
        for w in skiplist_words:
            ids = self.tokenizer(w, add_special_tokens=False)["input_ids"]
            skiplist_ids.update(int(i) for i in ids)
        self.register_buffer(
            "skiplist_ids",
            torch.tensor(sorted(skiplist_ids), dtype=torch.long),
            persistent=False,
        )

        for p in self.parameters():
            p.requires_grad_(True)

        self._lora_active: bool = False


    def enable_lora(
        self,
        r: int = 8,
        lora_alpha: int = 16,
        target_modules: list[str] | None = None,
        lora_dropout: float = 0.0,
        last_k_layers: int | None = None,
    ) -> int:
        """Wrap the encoder with PEFT LoRA adapters."""
        from peft import LoraConfig, get_peft_model
        if target_modules is None:
            # Backbone-aware: ModernBERT fuses Wqkv+Wo; BERT (colbertv2.0) uses query/key/value/dense.
            from excise.lora_targets import detect_lora_targets
            target_modules = detect_lora_targets(self.encoder)

        if last_k_layers is not None:
            n_layers = base = None
            for path in ("layers", "model.layers", "encoder.layer", "model.encoder.layer"):
                obj = self.encoder
                try:
                    for p in path.split("."):
                        obj = getattr(obj, p)
                except AttributeError:
                    continue
                if hasattr(obj, "__len__") and len(obj) > 1:
                    n_layers, base = len(obj), path
                    break
            if n_layers is None:
                raise RuntimeError(
                    "enable_lora(last_k_layers): could not locate the transformer layer "
                    "list on self.encoder — refusing to silently target ALL layers.")
            if last_k_layers < n_layers:
                start = n_layers - last_k_layers
                idx = "|".join(str(i) for i in range(start, n_layers))
                tm_alt = "|".join(target_modules)
                base_esc = base.replace(".", r"\.")
                target_modules = rf".*{base_esc}\.({idx})\..*({tm_alt})"
                print(f"[enable_lora] last_k={last_k_layers}/{n_layers} via '{base}' "
                      f"-> {target_modules}", flush=True)

        cfg = LoraConfig(
            r=r, lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout, bias="none",
        )
        self.encoder = get_peft_model(self.encoder, cfg)
        self._lora_active = True
        n_trainable = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        return n_trainable


    def encode(
        self,
        texts: list[str],
        is_query: bool,
        device: str | torch.device | None = None,
    ) -> EncodedBatch:
        """Encode a batch of strings to per-token L2-normalised ColBERT vectors."""
        if device is None:
            device = next(self.parameters()).device
        prefix = self.query_prefix if is_query else self.doc_prefix
        max_len = self.query_len if is_query else self.doc_len
        batch_texts = [prefix + (t or "") for t in texts]
        # For queries: pad up to max_len with [MASK] so the encoder can perform
        # query expansion. For docs: pad to longest in batch.
        padding = "max_length" if is_query else True
        enc = self.tokenizer(
            batch_texts,
            padding=padding,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attn_mask = enc["attention_mask"].to(device)

        if is_query and self.tokenizer.mask_token_id is not None:
            # Replace pad tokens with mask tokens for query expansion.
            pad_id = self.tokenizer.pad_token_id
            mask_id = self.tokenizer.mask_token_id
            replace = (input_ids == pad_id)
            input_ids = torch.where(replace, torch.full_like(input_ids, mask_id), input_ids)
            if not self.attend_to_expansion:
                pass

        if self._lora_active and not is_query:
            with torch.no_grad():
                with self.encoder.disable_adapter():
                    out = self.encoder(input_ids=input_ids, attention_mask=attn_mask)
        else:
            out = self.encoder(input_ids=input_ids, attention_mask=attn_mask)
        h = out.last_hidden_state                         # (B, T, hidden)
        v = self.dense(h.to(self.dense.weight.dtype))     # (B, T, proj)
        v = F.normalize(v, dim=-1)

        valid = attn_mask.bool()
        if not is_query:
            # Doc skiplist: zero out punctuation token vectors so they cannot
            # win MaxSim, and mark them invalid in the mask.
            sk = self.skiplist_ids.to(device)
            in_skip = torch.isin(input_ids, sk)
            valid = valid & ~in_skip
            v = v * valid.unsqueeze(-1).to(v.dtype)

        return EncodedBatch(vectors=v, mask=valid)


    @staticmethod
    def maxsim(q: EncodedBatch, d: EncodedBatch) -> torch.Tensor:
        """ColBERT MaxSim between every query and every document."""
        # q.vectors: (Bq, Tq, P); d.vectors: (Bd, Td, P)
        # For each pair (i, j): sum over q_tokens of max over d_tokens of dot.
        sims = torch.einsum("itp,jsp->ijts", q.vectors, d.vectors)   # (Bq, Bd, Tq, Td)
        # Mask invalid d tokens to a very negative number before max.
        d_mask = d.mask[None, :, None, :]                            # (1, Bd, 1, Td)
        sims = sims.masked_fill(~d_mask, float("-inf"))
        per_q_token = sims.max(dim=-1).values                        # (Bq, Bd, Tq)
        # Mask invalid q tokens to 0 before sum.
        q_mask = q.mask[:, None, :].to(per_q_token.dtype)            # (Bq, 1, Tq)
        return (per_q_token * q_mask).sum(dim=-1)


import sys

_root = os.path.dirname(os.path.abspath(__file__))
while _root != os.path.dirname(_root) and not os.path.exists(os.path.join(_root, "excise/config.py")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)
from excise import config as C   # noqa: E402


def load_backbone(device="cuda:0", lora=True, lora_r=8, lora_last_k=6, dtype=None):
    import torch
    # bfloat16 is the GPU default; on CPU it is slow/patchily supported -> float32 so the arch can be run
    # (and therefore VERIFIED) without a GPU.
    if dtype is None:
        dtype = torch.float32 if str(device).startswith("cpu") else torch.bfloat16
    bb = ColBERTBackbone(repo_id=C.BACKBONE, attn_implementation="sdpa", torch_dtype=dtype).to(device)
    for p in bb.parameters(): p.requires_grad_(False)
    bb.eval()
    if lora:
        bb.enable_lora(r=lora_r, last_k_layers=lora_last_k); bb.eval()
    return bb
