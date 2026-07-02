#!/bin/bash
#SBATCH --job-name=mbti-bertvar
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -w h86
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# bert-base-uncased seed variance (option B): 3 seeds at the canonical config,
# to check whether deberta-v3@256's +0.009 balanced-acc edge is inside bert's
# own run-to-run noise. Each run is wrapped in `timeout 3600` so a hang (as on
# job 61916) self-kills at 1h and the job moves to the next seed instead of
# stalling for hours. Healthy bert run is ~35 min.

set -o pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results
source hpc/env.sh   # modules + venv + offline HF; config overridden just below

export MBTI_MODEL_NAME="bert-base-uncased" MBTI_MAX_LENGTH=128 MBTI_BATCH_SIZE=32
RESULTS="results/bertvar_metrics.csv"
echo "model,seed,mean_acc,mean_balanced_acc,exact16" > "$RESULTS"

for s in 42 123 7; do
  echo "================ bert-base-uncased 128/32 seed=$s ================"
  log="logs/bertvar_seed${s}.log"
  if timeout 3600 python scripts/train_mbti.py --epochs 4 --batch-size 32 --seed "$s" 2>&1 | tee "$log"; then
    acc=$(grep "MEAN per-user accuracy" "$log" | tail -1 | sed -E 's/.*= ([0-9.]+).*/\1/')
    bal=$(grep "MEAN per-user balanced accuracy" "$log" | tail -1 | sed -E 's/.*= ([0-9.]+).*/\1/')
    ex=$(grep "EXACT 16-type" "$log" | tail -1 | sed -E 's/.*= ([0-9.]+).*/\1/')
    echo "bert-base-uncased,$s,${acc:-NA},${bal:-NA},${ex:-NA}" >> "$RESULTS"
  else
    echo "bert-base-uncased,$s,FAILED_OR_TIMEOUT,,," >> "$RESULTS"
  fi
done

echo "================ SUMMARY ================"
cat "$RESULTS"
