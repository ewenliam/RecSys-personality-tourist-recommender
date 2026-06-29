#!/bin/bash
#SBATCH --job-name=mbti-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# Prefer the 96 GB Blackwell node for the upgraded backbone/context.
# Uncomment if you specifically want it (may queue longer):
# #SBATCH --nodelist=h86

# Stage 1: train the BERT-MBTI classifier with the upgraded configuration
# (stronger backbone + longer context, exported by hpc/env.sh).

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source hpc/env.sh

echo "Training classifier: backbone=$MBTI_MODEL_NAME max_length=$MBTI_MAX_LENGTH"
srun python scripts/train_mbti.py --epochs 4 --batch-size "$MBTI_BATCH_SIZE"
