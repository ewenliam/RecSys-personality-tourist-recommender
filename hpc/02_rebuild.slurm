#!/bin/bash
#SBATCH --job-name=recsys-rebuild
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Stage 2: rebuild the whole downstream pipeline against the new classifier
# (topics -> per-user MBTI -> GNN -> 5-seed robustness). Uses the same
# upgraded backbone/context via hpc/env.sh, so all stages stay consistent.
#
# Submit this AFTER stage 1 finishes. To chain automatically:
#   JID=$(sbatch --parsable hpc/01_train_mbti.slurm)
#   sbatch --dependency=afterok:$JID hpc/02_rebuild.slurm

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source hpc/env.sh

srun python scripts/rebuild_pipeline.py --seeds 42,123,7,2024,99
