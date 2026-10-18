#!/usr/bin/env python3
"""Run one system on one X-BENCH collection with the released checkpoints and constants. One GPU.

  python -m excise.run --list
  python -m excise.run --backbone reason --dataset excluir --system excise --dry
  python -m excise.run --backbone reason --dataset excluir --system excise

The EXCISE arm is built from excise/adopted.json, which is generated from the checkpoints themselves.
`scripts/slurm_example.sh` wraps this for a scheduler."""
from __future__ import annotations
import argparse, glob, json, os, shlex, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK = os.environ.get("EXCISE_CKPT_ROOT", os.path.join(ROOT, "model", "checkpoints"))
COLLECTIONS = ["excluir", "fiqa", "trec-covid", "esgenius", "edgar", "eurlex"]

# the shipped operator, as the README lists it
OPERATOR = ["--lam", "0.97", "--soft-demote", "--two-stage", "--ts-floor", "0.35",
            "--anomaly-gate", "--anomaly-sigma", "0.5", "--faithful",
            "--hard1", "--hard-auto", "--hard-guard", "--hard-guard-rule", "top_rel",
            "--hard-frac", "0.6", "--hard-cap", "3"]

BUDGET = {"reason":    {"EXCISE_CHUNK_WORDS": "512", "EXCISE_MAX_DOC_WORDS": "10000"},
          "gte":       {"EXCISE_QUERY_LEN": "128", "EXCISE_CHUNK_WORDS": "200", "EXCISE_MAX_DOC_WORDS": "1000"},
          "colbertv2": {"EXCISE_QUERY_LEN": "128", "EXCISE_CHUNK_WORDS": "200", "EXCISE_MAX_DOC_WORDS": "1000"}}

def adopted():
    p = os.path.join(ROOT, "excise", "adopted.json")
    if not os.path.exists(p):
        sys.exit("excise/adopted.json missing — run: python -m excise.adopted --write")
    return json.load(open(p))["backbones"]

def records(ds):
    """The collection's test files, relative to the repository root (the evaluators run from there)."""
    files = sorted(glob.glob(os.path.join(ROOT, "data", "benchmark", "test", ds, "*.jsonl")))
    if not files: sys.exit(f"no test records for {ds}")
    return [os.path.relpath(f, ROOT) for f in files]

def build(backbone, ds, system, ft=None):
    A = adopted()
    if backbone not in A: sys.exit(f"unknown backbone {backbone}; have {', '.join(A)}")
    b = A[backbone]; tag = b["hf_backbone"].split("/")[-1].lower().replace(".", "_")
    env = {"EXCISE_BACKBONE": b["hf_backbone"], **BUDGET[backbone]}
    rec = records(ds)

    if system == "excise":
        ad = os.path.join(CK, b["adapter"]["checkpoint"], "lora.pt")
        fd = os.path.join(CK, b["detector"]["checkpoint"], "deepfinder.pt")
        tau = b["detector"]["tau"]
        if tau is None: sys.exit(f"{backbone}: detector has no thr.txt")
        cmd = ["-m", "excise.evaluate", "--records", *rec,
               "--reembed-lora", ad, "--finder", fd, "--thr", str(tau)] + OPERATOR
    elif system == "frozen":
        cmd = ["-m", "excise.evaluate", "--records", *rec,
               "--reembed-lora", "", "--finder", ""]
    elif system in ("loraft", "fullft", "qonly"):
        ck = os.path.join(CK, f"{system}_{tag}", "lora.pt")
        extra = ["--query-only"] if system == "qonly" else ["--reindex"]
        cmd = ["-m", "excise.evaluate", "--records", *rec,
               "--reembed-lora", ck, "--finder", ""] + extra
    else:
        sys.exit(f"unknown system {system}")
    return env, cmd

SYSTEMS = ["excise", "frozen", "loraft", "fullft", "qonly"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--backbone"); ap.add_argument("--dataset"); ap.add_argument("--system")
    ap.add_argument("--dry", action="store_true", help='print the command, do not run it')
    a, extra = ap.parse_known_args()      # any other flag (e.g. --dump-orders FILE) is passed to the evaluator

    if a.list:
        A = adopted()
        print("backbones:  " + ", ".join(A))
        print("datasets:   " + ", ".join(COLLECTIONS))
        print("systems:    " + ", ".join(SYSTEMS))
        print("\nadopted per backbone (from excise/adopted.json, generated from the checkpoints):")
        for k, b in A.items():
            print(f"  {k:10s} {b['adapter']['checkpoint']:38s} + {b['detector']['checkpoint']}  tau={b['detector']['tau']}")
        sys.exit(0)

    if not (a.backbone and a.dataset and a.system): ap.error("need --backbone, --dataset and --system (or --list)")
    if a.dataset not in COLLECTIONS: ap.error(f"dataset must be one of {COLLECTIONS}")
    env, cmd = build(a.backbone, a.dataset, a.system)
    cmd += extra
    py = os.environ.get("PYTHON") or os.environ.get("PY") or sys.executable
    line = " ".join(f"{k}={v}" for k, v in env.items()) + " " + shlex.join([py] + cmd)   # pastes into a shell as is
    if a.dry:
        print(line); sys.exit(0)
    e = dict(os.environ, EXCISE_ROOT=ROOT, **env)
    sys.exit(subprocess.call([py] + cmd, cwd=ROOT, env=e))
