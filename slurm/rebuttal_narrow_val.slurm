#!/bin/bash
#SBATCH --job-name=nxro_narrow_val
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=slurm/logs/narrow_val_%a_%j.out
#SBATCH --error=slurm/logs/narrow_val_%a_%j.err
#SBATCH --array=0-14

# =============================================================================
# Sweep B: Narrow val split (3yr instead of 6yr)
# Train 1979-1998, Val 1999-2001, Test 2002-2022
# 5 models (linear, res, attentive, graph_pyg, pure_transformer) x 3 seeds = 15 jobs
# =============================================================================

set -euo pipefail

WORK_DIR="${WORK_DIR:-/home/fredxu_squareup_com/Github/Ocean_TimeSeries_Graph}"
cd "${WORK_DIR}"
source ~/workspace/bin/activate-hermit 2>/dev/null || true
PYTHON="${WORK_DIR}/.venv/bin/python"

MODELS=(linear res attentive graph_pyg pure_transformer)
SEEDS=(42 123 256)

TASK_ID=${SLURM_ARRAY_TASK_ID}
MODEL_IDX=$((TASK_ID / 3))
SEED_IDX=$((TASK_ID % 3))

MODEL=${MODELS[$MODEL_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "=== Narrow Val: model=${MODEL} seed=${SEED} (task ${TASK_ID}) ==="

${PYTHON} NXRO_train_out_of_sample.py \
    --model ${MODEL} \
    --nc_path data/XRO_indices_oras5.nc \
    --seed ${SEED} \
    --train_start 1979-01 \
    --train_end 2001-12 \
    --val_start 1999-01 \
    --val_end 2001-12 \
    --test_start 2002-01 \
    --test_end 2022-12 \
    --epochs 2000 \
    --batch_size 128 \
    --lr 1e-3 \
    --extra_train_nc none \
    --run_tag "narrow_val" \
    --base_results_dir results_rebuttal_narrow_val

echo "=== Done: model=${MODEL} seed=${SEED} ==="
