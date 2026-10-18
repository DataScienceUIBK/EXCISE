#!/bin/bash
#SBATCH --job-name=excise
#SBATCH --gres=gpu:1 --time=02:00:00
#SBATCH --output=logs/%x_%j.log
# One arm on one GPU through excise.run. Account, partition and QoS are read by sbatch from
# SBATCH_ACCOUNT, SBATCH_PARTITION and SBATCH_QOS in the submitting shell, so nothing cluster-specific
# is written here.
#
#   export EXCISE_ROOT=$PWD HF_HOME=/path/to/hf_cache SBATCH_ACCOUNT=... SBATCH_PARTITION=... SBATCH_QOS=...
#   sbatch scripts/slurm_example.sh reason excluir excise
set -u
cd "${EXCISE_ROOT:?export EXCISE_ROOT to this repository}"
export HF_HOME="${HF_HOME:?export HF_HOME to the cache that holds the backbone}" HF_HUB_OFFLINE=1
mkdir -p logs
python -m excise.run --backbone "${1:?backbone: reason | gte | colbertv2}" --dataset "${2:?collection}" \
    --system "${3:?system: excise | frozen | loraft | fullft | qonly}"
