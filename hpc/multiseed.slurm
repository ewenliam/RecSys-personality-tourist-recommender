#!/bin/bash
#SBATCH --job-name=mbti-multiseed
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH -w h86
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Multi-seed robustness: is deberta-v3@256's balanced-acc edge over bert@128
# real across seeds, or single-seed noise? Runs each model at its own config
# for seeds {42,123,7}, SEQUENTIALLY (each run finishes train+eval before the
# next, so best_model.pt is never clobbered mid-run). Metrics -> CSV.

set -o pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs results
source hpc/env.sh   # modules + venv + offline HF; backbone overridden per-run

SEEDS="42 123 7"
RESULTS="results/multiseed_metrics.csv"
echo "model,max_length,batch,seed,mean_acc,mean_balanced_acc,exact16" > "$RESULTS"

run_model () {
  model="$1"; maxlen="$2"; batch="$3"
  export MBTI_MODEL_NAME="$model" MBTI_MAX_LENGTH="$maxlen" MBTI_BATCH_SIZE="$batch"
  tag=$(echo "$model" | tr '/' '_')
  for s in $SEEDS; do
    echo "================ $model maxlen=$maxlen batch=$batch seed=$s ================"
    log="logs/ms_${tag}_${maxlen}_seed${s}.log"
    if python scripts/train_mbti.py --epochs 4 --batch-size "$batch" --seed "$s" 2>&1 | tee "$log"; then
      acc=$(grep "MEAN per-user accuracy" "$log" | tail -1 | sed -E 's/.*= ([0-9.]+).*/\1/')
      bal=$(grep "MEAN per-user balanced accuracy" "$log" | tail -1 | sed -E 's/.*= ([0-9.]+).*/\1/')
      ex=$(grep "EXACT 16-type" "$log" | tail -1 | sed -E 's/.*= ([0-9.]+).*/\1/')
      echo "$model,$maxlen,$batch,$s,${acc:-NA},${bal:-NA},${ex:-NA}" >> "$RESULTS"
    else
      echo "$model,$maxlen,$batch,$s,FAILED,FAILED,FAILED" >> "$RESULTS"
    fi
  done
}

run_model "bert-base-uncased" 128 32
run_model "microsoft/deberta-v3-base" 256 64

echo "================ SUMMARY ================"
cat "$RESULTS"
