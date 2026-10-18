#!/usr/bin/env python3
"""Download the corpora and the checkpoints from Hugging Face and check each corpus against its sha256 pin.

  python scripts/download.py --what corpora
  python scripts/download.py --what checkpoints [--tier excise|baselines|all]
  python scripts/download.py --verify"""
import argparse, hashlib, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINS = os.path.join(ROOT, "data", "benchmark", "CORPUS_REQUIRED.json")

def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()

def local_path(ds, spec):
    return os.path.join(ROOT, "data", "datasets", ds, spec["file"])

def verify():
    pins = json.load(open(PINS))["datasets"]
    bad = missing = 0
    for ds, spec in sorted(pins.items()):
        p = local_path(ds, spec)
        if not os.path.exists(p):
            print(f"  MISSING  {ds:12s} {spec['file']}"); missing += 1; continue
        size = os.path.getsize(p)
        if size != spec["bytes"]:
            print(f"  SIZE     {ds:12s} {size} bytes, pinned {spec['bytes']}"); bad += 1; continue
        got = sha256(p)
        if got != spec["sha256"]:
            print(f"  SHA256   {ds:12s} {got[:16]}… != pinned {spec['sha256'][:16]}…"); bad += 1
        else:
            print(f"  ok       {ds:12s} {spec['n']:>7,} docs")
    print(f"\n  {len(pins)-bad-missing}/{len(pins)} verified, {bad} mismatched, {missing} missing")
    return 1 if (bad or missing) else 0

HUB = {"corpora": "mohammed933/EXCISE-corpora", "checkpoints": "mohammed933/EXCISE-checkpoints"}   # the published Hugging Face repositories


def recorded_repo(what):
    return HUB["corpora" if what == "corpora" else "checkpoints"]


TIERS = ("excise", "baselines", "externals")


def fetch(what, repo, tier="excise"):
    repo = repo or recorded_repo(what)
    if not repo:
        sys.exit(f"--what {what} needs --repo <hub-id>")
    from huggingface_hub import snapshot_download
    dest = os.path.join(ROOT, "data", "datasets") if what == "corpora" else os.path.join(ROOT, "model", "checkpoints")
    os.makedirs(dest, exist_ok=True)
    tiers = TIERS if tier == "all" else (tier,)
    print(f"  fetching {what}{'' if what == 'corpora' else ' (' + ', '.join(tiers) + ')'} from {repo} -> {os.path.relpath(dest, ROOT)}")
    snapshot_download(repo_id=repo, repo_type="dataset" if what == "corpora" else "model", local_dir=dest,
                      allow_patterns=["*/corpus.jsonl"] if what == "corpora" else [f"{t}/*" for t in tiers] + ["README.md"])
    if what == "corpora":
        print("\n  verifying against the pins:"); return verify()
    # The Hub repository is tiered (excise/, baselines/, externals/); the code reads a flat
    # model/checkpoints/<name>/. Link each fetched checkpoint directory to the flat path.
    linked = 0
    for t in tiers:
        tdir = os.path.join(dest, t)
        if not os.path.isdir(tdir):
            continue
        for name in sorted(os.listdir(tdir)):
            flat = os.path.join(dest, name)
            if os.path.isdir(os.path.join(tdir, name)) and not os.path.exists(flat):
                os.symlink(os.path.join(t, name), flat); linked += 1
    print(f"  {linked} checkpoint director{'y' if linked == 1 else 'ies'} linked flat under {os.path.relpath(dest, ROOT)}/")
    return 0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["corpora", "checkpoints", "all"])
    ap.add_argument("--repo", default=None, help='Hugging Face repo id to fetch from')
    ap.add_argument("--verify", action="store_true", help='check local corpora against the sha256 pins')
    ap.add_argument("--tier", default="excise", choices=list(TIERS) + ["all"],
                    help='which checkpoints: excise (the operator, 25 MB')
    a = ap.parse_args()
    if not (a.what or a.verify): ap.error("give --verify or --what")
    rc = 0
    for w in (["corpora", "checkpoints"] if a.what == "all" else ([a.what] if a.what else [])):
        rc |= fetch(w, a.repo, a.tier)
    if a.verify: rc |= verify()
    sys.exit(rc)
