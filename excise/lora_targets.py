"""The linear layers a LoRA adapter attaches to, per backbone architecture."""
from __future__ import annotations


def detect_lora_targets(module) -> list[str]:
    leaf = {n.split(".")[-1] for n, m in module.named_modules()
            if m.__class__.__name__ == "Linear"}
    if "Wqkv" in leaf:
        return ["Wqkv", "Wo"]                          # ModernBERT
    if {"query", "key", "value"} <= leaf:
        return ["query", "key", "value", "dense"]      # BERT (colbertv2.0)
    raise ValueError(f"cannot infer LoRA target modules from Linear leaves: {sorted(leaf)}")
