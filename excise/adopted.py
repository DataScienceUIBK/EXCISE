"""Write excise/adopted.json: the released detector and adapter per backbone, with the arguments each checkpoint
recorded when it was trained and the detector threshold stored beside it.

  python -m excise.adopted            # print
  python -m excise.adopted --write    # also write excise/adopted.json"""
import json, os, sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK = os.environ.get("EXCISE_CKPT_ROOT", os.path.join(ROOT, "model", "checkpoints"))

# backbone -> (adapter, detector, Hugging Face backbone)
ADOPTED = {
    "reason":    ("adapter_reason",    "detector_reason",    "lightonai/Reason-ModernColBERT"),
    "gte":       ("adapter_gte",       "detector_gte",       "lightonai/GTE-ModernColBERT-v1"),
    "colbertv2": ("adapter_colbertv2", "detector_colbertv2", "colbert-ir/colbertv2.0"),
}

OPERATOR = ["--soft-demote", "--lam 0.97", "--two-stage", "--ts-floor 0.35",
            "--anomaly-gate", "--anomaly-sigma 0.5", "--faithful", "--topk 100",
            "--hard1", "--hard-auto", "--hard-guard", "--hard-guard-rule top_rel",
            "--hard-frac 0.6", "--hard-cap 3"]


def ckpt_dir(name):
    for sub in ("", "excise"):
        p = os.path.join(CK, sub, name)
        if os.path.isdir(p):
            return p
    return os.path.join(CK, name)


def args_of(path):
    """The arguments a checkpoint recorded at training time."""
    try:
        return dict(torch.load(path, map_location="cpu", weights_only=False).get("args", {}))
    except Exception as e:
        raise SystemExit(f"cannot read {path}: {e}\nSet EXCISE_CKPT_ROOT to where the checkpoints live.")


def thr_of(name):
    p = os.path.join(ckpt_dir(name), "thr.txt")
    return open(p).read().strip() if os.path.exists(p) else None


def build():
    out = {"generated_from": "checkpoint args + thr.txt", "operator": OPERATOR, "backbones": {}}
    for bb, (ad, fd, hf) in ADOPTED.items():
        a = args_of(os.path.join(ckpt_dir(ad), "lora.pt"))
        d = args_of(os.path.join(ckpt_dir(fd), "deepfinder.pt"))
        out["backbones"][bb] = {
            "hf_backbone": hf,
            "adapter": {"checkpoint": ad, "lr": a.get("lr"), "z_margin": a.get("z_margin"),
                        "z_weight": a.get("z_weight"), "epochs": a.get("epochs"), "lora_r": a.get("lora_r"),
                        "batch_size": a.get("batch_size"), "seed": a.get("seed")},
            "detector": {"checkpoint": fd, "lr": d.get("lr"), "epochs": d.get("epochs"), "bs": d.get("bs"),
                         "lora_r": d.get("lora_r"), "pos_weight": d.get("pos_weight"), "xz": d.get("xz"),
                         "iob": d.get("iob"), "tap": d.get("tap"), "seed": d.get("seed"), "tau": thr_of(fd)},
        }
    return out


if __name__ == "__main__":
    cfg = build()
    print(json.dumps(cfg, indent=1))
    if "--write" in sys.argv:
        p = os.path.join(ROOT, "excise", "adopted.json")
        json.dump(cfg, open(p, "w"), indent=1)
        print(f"\nwrote {p}", file=sys.stderr)
