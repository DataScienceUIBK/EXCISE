"""EXCISE: paths, the backbone, the released checkpoints and the operator constants."""
from __future__ import annotations
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))       # the repository root
OUT = os.path.join(ROOT, "data", "out")                 # generated files
os.makedirs(OUT, exist_ok=True)

# ---- Hugging Face cache: the backbones and the BEIR datasets are downloaded into it on first use ----
HF_HOME = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = os.path.join(HF_HOME, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_HOME, "datasets")

# ---- training data ----
SEED = 0
TWIN_FRAC = 0.45                                        # training-set balance
FINDER_PLAINNEG_FRAC = 0.45                             # share of no-exclusion queries in the detector data

# ---- the backbone ----
BACKBONE = os.environ.get("EXCISE_BACKBONE", "lightonai/Reason-ModernColBERT")
BACKBONE_QUERY_LEN = int(os.environ["EXCISE_QUERY_LEN"]) if os.environ.get("EXCISE_QUERY_LEN") else None


def backbone_tag(name=None):
    """Filesystem-safe short id of a backbone, used in checkpoint names."""
    return (name or BACKBONE).split("/")[-1].replace(".", "_").lower()


# sentence-transformers 5.4 renamed the backbone submodule, so the adapters load only below 5.4
# (requirements.txt pins 5.3.0).
PYTHON = os.environ.get("PYTHON") or __import__("sys").executable

# ---- the released checkpoints (excise/adopted.json names them per backbone; excise/run.py passes them explicitly) ----
_TAG = backbone_tag()
REEMBED_LORA = os.environ.get(
    "EXCISE_REEMBED_LORA", os.path.join(ROOT, "model", "checkpoints",
                 "adapter_" + {"reason-moderncolbert": "reason", "gte-moderncolbert-v1": "gte",
                              "colbertv2_0": "colbertv2"}.get(_TAG, _TAG), "lora.pt"))
FINDER_CKPT = os.environ.get(
    "EXCISE_FINDER_CKPT", os.path.join(ROOT, "model", "checkpoints", "detector_reason", "deepfinder.pt"))

# Each backbone is read within its own configured length; documents longer than the encoder are windowed
# and MaxSim-pooled.
BACKBONES = {
    "reason":    dict(repo="lightonai/Reason-ModernColBERT",  budget_words=10000, chunk_words=512, query_len=None),
    "gte":       dict(repo="lightonai/GTE-ModernColBERT-v1",  budget_words=1000,  chunk_words=200, query_len=128),
    "colbertv2": dict(repo="colbert-ir/colbertv2.0",          budget_words=1000,  chunk_words=200, query_len=128),
}

# The five systems of each backbone band. The fine-tunes rebuild the corpus index; EXCISE does not.
SYSTEMS = {
    "frozen":  dict(ckpt=None,            re_index=False, note="the reference"),
    "lora_ft": dict(ckpt="loraft_{tag}",  re_index=True,  note="LoRA fine-tune, re-indexed"),
    "full_ft": dict(ckpt="fullft_{tag}",  re_index=True,  note="full encoder fine-tune, re-indexed"),
    "qonly":   dict(ckpt="qonly_{tag}",   re_index=False, note="query-only adapter over the frozen index"),
    "excise":  dict(ckpt=None, re_index=False,
                    note="this operator: the detector and Z-margin adapter excise/adopted.json names per backbone; "
                         "excise.run builds the arm from that file"),
}

BASELINE_TRIPLES = os.path.join(ROOT, "data", "benchmark", "train", "retrieval_triples.jsonl")
OPERATOR_TRIPLES = os.path.join(ROOT, "data", "benchmark", "train", "adapter_triples.jsonl")
PARITY_TRIPLES = OPERATOR_TRIPLES
DETECTOR_TRAIN = os.path.join(ROOT, "data", "benchmark", "train", "detector.jsonl")

# ---- evaluation ----
SELECTION_RECORDS = [os.path.join(ROOT, "data", "benchmark", "train", f)
                     for f in ("excluir.jsonl", "fiqa.jsonl")]
TEST_COLLECTIONS = ["edgar", "eurlex", "esgenius", "trec-covid", "excluir", "fiqa"]
OOD_COLLECTIONS = ["edgar", "eurlex", "esgenius", "trec-covid"]
NOHARM_SUITES = ["scifact", "nfcorpus", "arguana"]


def _thr_from_sidecar(default=0.85):
    """The detector's threshold, read from the thr.txt beside its checkpoint."""
    p = os.path.join(os.path.dirname(FINDER_CKPT), "thr.txt")
    try:
        return float(open(p).read().strip())
    except Exception:
        return default


FINDER_THR = float(os.environ["EXCISE_FINDER_THR"]) if os.environ.get("EXCISE_FINDER_THR") else _thr_from_sidecar()

OPERATOR = dict(
    lam=0.97,            # soft-penalty strength
    topk=100,            # candidates re-embedded at query time; the index is never rebuilt
    two_stage=True,      # an absolute floor, then a cut relative to the shortlist
    ts_floor=0.35,       # stage 1: act only when some candidate carries real Z-evidence
    anomaly_gate=True,   # stage 2: cut at mu + sigma * std of the shortlist's Z-evidence
    anomaly_sigma=0.5,
    soft_demote=True,
    faithful=True,       # re-embed the candidate texts at query time
)
# Hard demotion sends the strongest Z-matches to the bottom of the shortlist; the guard keeps the top
# candidate when it is the best match for X, since the gold document can mention the excluded topic too.
HARD_DEMOTE = dict(hard1=True, hard_auto=True, hard_guard=True,
                   hard_guard_rule="top_rel", hard_frac=0.6, hard_cap=3)

# The adapter's objective: the contrastive loss plus a margin on MaxSim(Z, d), the quantity the operator
# demotes by. lr is the fallback; the five-fold selection writes its choice to data/selection/lr_selected.txt.
Z_MARGIN = dict(margin=0.10, weight=5.0, lora_r=8, epochs=3, lr=1e-3, effective_batch=16)


def selected_lr(kind, name, default=None):
    """The five-fold cross-validation choice for one system, or `default` if selection has not run for it."""
    path = os.path.join(ROOT, "data", "selection", "lr_selected.txt")
    if not os.path.exists(path):
        path = os.path.join(OUT, "lr_selected.txt")
    if not os.path.exists(path):
        return default
    best = None
    for line in open(path):
        f = line.split()
        if len(f) >= 4 and f[0] == kind and f[1] == name:
            best = f[2]          # the last line wins
    return float(best) if best else default


def adapter_lr(backbone):
    """The adapter learning rate used for this backbone."""
    return selected_lr("adapter", backbone, Z_MARGIN["lr"])


REEMBED_TOPK = 100                                      # candidates re-embedded at query time
DEMOTE_LAM = 0.97                                       # soft-penalty strength

# ---- SLURM: account, partition, QoS and excluded nodes come from the environment ----
SLURM = dict(account=os.environ.get("SBATCH_ACCOUNT", ""), qos=os.environ.get("SBATCH_QOS", ""),
             partition=os.environ.get("SBATCH_PARTITION", ""), exclude=os.environ.get("SBATCH_EXCLUDE", ""))
