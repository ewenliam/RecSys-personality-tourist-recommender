#!/bin/bash
#SBATCH --job-name=mbti-deleaked
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# bert-base-uncased/128 fits on any GPU node, so no --nodelist is needed.
# Uncomment for the fast 96 GB node if it is free:
# #SBATCH --nodelist=h86

# De-leaked classifier retrain: identical to the canonical run, but with the
# explicit MBTI label terms (type codes, axis pair codes, generic vocabulary)
# stripped from the training text. An audit found 87% of users write their own
# type in their posts, so this run measures how much of the honest 80.1% acc /
# 75.2% balanced accuracy depended on the classifier reading the label off the
# text. Checkpoints go to models/checkpoints/bert_mbti_deleaked (the canonical
# bert_mbti checkpoints are left untouched). Honest per-user test metrics print
# at the end of the log; compare them to the leaked baseline.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source hpc/env.sh   # canonical bert-base-uncased/128, venv, offline HF cache

echo "De-leaked training: backbone=$MBTI_MODEL_NAME max_length=$MBTI_MAX_LENGTH"
srun python scripts/train_mbti.py --epochs 4 --batch-size "$MBTI_BATCH_SIZE" \
    --strip-mbti-terms
