#!/bin/bash
# Shared environment for HPC (SLURM) jobs on hpc.ii.pw.edu.pl.
# Source this from each SLURM script:  source hpc/env.sh
#
# It (1) loads the cluster modules, (2) activates the project venv, and
# (3) exports the UPGRADED model configuration (stronger backbone + longer
# context) that the code reads via environment variables (see src/config/
# settings.py). Because every stage reads the same config, the classifier,
# topic, and per-user embedding steps stay consistent automatically.

# --- 1. Cluster modules (verified against `module avail` on hpc.ii.pw.edu.pl) -
# SLURM batch jobs run in a non-login shell where Lmod is not initialised,
# so source it explicitly before using `module`.
if ! command -v module >/dev/null 2>&1; then source /etc/profile.d/z00_lmod.sh; fi
module purge
module load python/3.11.15
module load cuda/12.9

# --- 2. Project virtual environment -------------------------------------
source "$HOME/recsys-env/bin/activate"

# --- 3. Upgraded model configuration (96 GB GPU lets us scale up) -------
# Stronger backbone + longer context than the 8 GB laptop run.
export MBTI_MODEL_NAME="roberta-base"   # stronger drop-in; 768-dim
# Alternative (often best for classification): microsoft/deberta-v3-base
#   -> requires `pip install sentencepiece` and a fast tokenizer.
export MBTI_MAX_LENGTH=256              # was 128 on the laptop
export MBTI_BATCH_SIZE=64               # larger batch on the big GPU

# --- 4. Offline HuggingFace (compute nodes usually have no internet) -----
# Pre-download the backbone ONCE on the login node (see hpc/README.md),
# then jobs run fully offline from the cache.
export HF_HOME="$HOME/.cache/huggingface"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# --- 5. Reproducibility pins (harmless on Linux, kept for parity) --------
export NUMBA_DISABLE_INTEL_SVML=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[env] backbone=$MBTI_MODEL_NAME max_length=$MBTI_MAX_LENGTH batch=$MBTI_BATCH_SIZE"
