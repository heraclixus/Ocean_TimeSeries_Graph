#!/bin/bash
#SBATCH --job-name=nxro_mlp_tiny
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=slurm/logs/mlp_tiny_%a_%j.out
#SBATCH --error=slurm/logs/mlp_tiny_%a_%j.err
#SBATCH --array=0-29

# =============================================================================
# Sweep: Very small MLPs to find one that works under val split
# 5 hidden x 2 weight_decay x 3 seeds = 30 jobs
# hidden: 4, 8, 12, 16, 32
# weight_decay: 0.01, 0.05
# =============================================================================

set -euo pipefail

WORK_DIR="${WORK_DIR:-/home/fredxu_squareup_com/Github/Ocean_TimeSeries_Graph}"
cd "${WORK_DIR}"
source ~/workspace/bin/activate-hermit 2>/dev/null || true
PYTHON="${WORK_DIR}/.venv/bin/python"

HIDDEN_VALUES=(4 8 12 16 32)
WD_VALUES=(0.01 0.05)
SEEDS=(42 123 256)

TASK_ID=${SLURM_ARRAY_TASK_ID}
# 30 = 5*2*3
SEED_IDX=$((TASK_ID % 3))
REM=$((TASK_ID / 3))
WD_IDX=$((REM % 2))
H_IDX=$((REM / 2))

HIDDEN=${HIDDEN_VALUES[$H_IDX]}
WD=${WD_VALUES[$WD_IDX]}
SEED=${SEEDS[$SEED_IDX]}

TAG="h${HIDDEN}_wd${WD}"
echo "=== MLP Tiny: hidden=${HIDDEN} wd=${WD} seed=${SEED} (task ${TASK_ID}) ==="

${PYTHON} NXRO_train_out_of_sample.py \
    --model res \
    --nc_path data/XRO_indices_oras5.nc \
    --seed ${SEED} \
    --train_start 1979-01 \
    --train_end 2001-12 \
    --val_start 1996-01 \
    --val_end 2001-12 \
    --test_start 2002-01 \
    --test_end 2022-12 \
    --epochs 2000 \
    --batch_size 128 \
    --lr 1e-3 \
    --weight_decay ${WD} \
    --res_reg 1e-4 \
    --hidden_mlp ${HIDDEN} \
    --extra_train_nc none \
    --run_tag "${TAG}" \
    --base_results_dir results_rebuttal_mlp_tiny

echo "=== Done: ${TAG} seed=${SEED} ==="
