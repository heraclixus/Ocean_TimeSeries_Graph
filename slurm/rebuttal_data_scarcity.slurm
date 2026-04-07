#!/bin/bash
#SBATCH --job-name=nxro_scarcity
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=slurm/logs/data_scarcity_%a_%j.out
#SBATCH --error=slurm/logs/data_scarcity_%a_%j.err
#SBATCH --array=0-59

# =============================================================================
# Rebuttal Experiment 4: Data scarcity curve
# Addresses reviewer i9br (how much data is needed?)
#
# 4 models (linear, res, attentive, graph_pyg) x 5 train sizes x 3 seeds = 60 jobs
# Train sizes: 10yr, 13yr, 16yr, 19yr, 23yr (full)
# All use val split (last 3 years of training period), test 2002-2022
# =============================================================================

set -euo pipefail

WORK_DIR="${WORK_DIR:-/home/fredxu_squareup_com/Github/Ocean_TimeSeries_Graph}"
cd "${WORK_DIR}"
source ~/workspace/bin/activate-hermit 2>/dev/null || true
PYTHON="${WORK_DIR}/.venv/bin/python"

MODELS=(linear res attentive graph_pyg)
# Training end dates corresponding to ~10, 13, 16, 19, 23 years of data
# Validation always uses last 3 years of the nominal training period
TRAIN_ENDS=("1988-12" "1991-12" "1994-12" "1997-12" "2001-12")
VAL_STARTS=("1986-01" "1989-01" "1992-01" "1995-01" "1999-01")
VAL_ENDS=("1988-12" "1991-12" "1994-12" "1997-12" "2001-12")
SIZE_TAGS=("10yr" "13yr" "16yr" "19yr" "23yr")
SEEDS=(42 123 256)

TASK_ID=${SLURM_ARRAY_TASK_ID}
MODEL_IDX=$((TASK_ID / 15))
REM=$((TASK_ID % 15))
SIZE_IDX=$((REM / 3))
SEED_IDX=$((REM % 3))

MODEL=${MODELS[$MODEL_IDX]}
TRAIN_END=${TRAIN_ENDS[$SIZE_IDX]}
VAL_START=${VAL_STARTS[$SIZE_IDX]}
VAL_END=${VAL_ENDS[$SIZE_IDX]}
SIZE_TAG=${SIZE_TAGS[$SIZE_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "=== Data Scarcity: model=${MODEL} size=${SIZE_TAG} seed=${SEED} (task ${TASK_ID}) ==="

${PYTHON} NXRO_train_out_of_sample.py \
    --model ${MODEL} \
    --nc_path data/XRO_indices_oras5.nc \
    --seed ${SEED} \
    --train_start 1979-01 \
    --train_end "${TRAIN_END}" \
    --val_start "${VAL_START}" \
    --val_end "${VAL_END}" \
    --test_start 2002-01 \
    --test_end 2022-12 \
    --epochs 2000 \
    --batch_size 128 \
    --lr 1e-3 \
    --extra_train_nc none \
    --run_tag "${SIZE_TAG}" \
    --base_results_dir results_rebuttal_data_scarcity

echo "=== Done: model=${MODEL} size=${SIZE_TAG} seed=${SEED} ==="
