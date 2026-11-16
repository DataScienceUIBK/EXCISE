"""The PyLate ColBERT encoder, with or without the exclusion adapter. Documents are re-embedded with the
adapter at query time only; the index on disk stays frozen."""
from __future__ import annotations
import sys, os
# the repository root, so `config` and the packages import from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from excise import config as C                                              # sets HF_HOME before pylate/transformers
import numpy as np, torch
from pylate import models
from peft import LoraConfig
from excise.lora_targets import detect_lora_targets

MODEL = C.BACKBONE                                              # lightonai/Reason-ModernColBERT
_QL = {"query_length": C.BACKBONE_QUERY_LEN} if C.BACKBONE_QUERY_LEN else {}   # colbertv2 native q-len 32 -> bump for fair long queries
# ColBERTv2 ships doc_maxlen=180 in its metadata, shorter than the 200-word chunks it is given; EXCISE_DOC_LEN
# (run.py sets 300 for it, the length the paper states) keeps the whole chunk.
_DL = {"document_length": int(os.environ["EXCISE_DOC_LEN"])} if os.environ.get("EXCISE_DOC_LEN") else {}


def _load_checked(module, sd, what):
    """strict=False silently DROPS every key whose name does not match -> the adapter would load as a no-op and"""
    res = module.load_state_dict(sd, strict=False)
    unexpected = list(getattr(res, "unexpected_keys", []))
    if unexpected:
        raise RuntimeError(f"{what}: {len(unexpected)} checkpoint keys matched NO parameter "
                           f"(silent no-op load). e.g. {unexpected[:3]}")
    if not sd:
        raise RuntimeError(f"{what}: empty state dict")


def load_model(lora_path, device):
    m = models.ColBERT(MODEL, model_kwargs={"attn_implementation": "sdpa"}, **_QL, **_DL)
    if hasattr(m, "_input_length") and not hasattr(m, "_text_length"): m._text_length = m._input_length
    elif hasattr(m, "_text_length") and not hasattr(m, "_input_length"): m._input_length = m._text_length
    if lora_path:
        r = torch.load(lora_path, map_location="cpu")
        if "full" in r:
            _load_checked(m, r["full"], f"reembed full {lora_path}")
        else:
            lr = r["args"].get("lora_r", 8)
            m.add_adapter(LoraConfig(r=lr, lora_alpha=2 * lr, target_modules=detect_lora_targets(m), lora_dropout=0.0, bias="none"))
            _load_checked(m, r["lora"], f"reembed lora {lora_path}")
            _assert_lora_active(m, lora_path)
    return m.to(device).eval()


def _assert_lora_active(m, path):
    bs = [p.detach() for n, p in m.named_parameters() if "lora_B" in n]
    if not bs:
        raise RuntimeError(f"{path}: adapter added but no lora_B parameters found")
    if max(float(p.abs().max()) for p in bs) == 0.0:
        raise RuntimeError(f"{path}: every lora_B is zero -> the adapter is the IDENTITY (load was a no-op)")


@torch.no_grad()
def emb(m, texts, is_query, device, batch=64):
    cap = int(getattr(m, "document_length", 8192) or 8192)
    max_len = min(cap, max((len(t) // 3 for t in texts), default=1) + 8)
    bs = max(1, min(batch, int(5e8 // (max_len * max_len))))
    e = m.encode(texts, is_query=is_query, show_progress_bar=False, batch_size=bs)
    P = np.asarray(e[0]).shape[-1]; T = max(np.asarray(x).shape[0] for x in e); N = len(e)
    V = torch.zeros(N, T, P); M = torch.zeros(N, T, dtype=torch.bool)
    for i, x in enumerate(e):
        x = torch.as_tensor(np.asarray(x)); V[i, :x.shape[0]] = x; M[i, :x.shape[0]] = True
    return V.to(device), M.to(device)
