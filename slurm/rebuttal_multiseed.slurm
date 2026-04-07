#!/bin/bash
#SBATCH --job-name=nxro_multiseed
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=slurm/logs/multiseed_%a_%j.out
#SBATCH --error=slurm/logs/multiseed_%a_%j.err
#SBATCH --array=0-59

# =============================================================================
# Rebuttal Experiment 1: Multi-seed runs with train/val/test split
# Addresses reviewer jHw1 (model selection leakage, multi-seed variance)
#
# 6 core models x 10 seeds = 60 jobs
# Models: linear, res (MLP), attentive, graph_pyg (GNN), pure_neural_ode, pure_transformer
# Split: train 1979-1995, val 1996-2001, test 2002-2022
# =============================================================================

set -euo pipefail

WORK_DIR="${WORK_DIR:-/home/fredxu_squareup_com/Github/Ocean_TimeSeries_Graph}"
cd "${WORK_DIR}"
source ~/workspace/bin/activate-hermit 2>/dev/null || true
PYTHON="${WORK_DIR}/.venv/bin/python"

# 6 models x 10 seeds = 60 configs
MODELS=(linear res attentive graph_pyg pure_neural_ode pure_transformer)
SEEDS=(42 123 256 314 512 777 1024 1337 2024 9999)

TASK_ID=${SLURM_ARRAY_TASK_ID}
MODEL_IDX=$((TASK_ID / 10))
SEED_IDX=$((TASK_ID % 10))

MODEL=${MODELS[$MODEL_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "=== Rebuttal Multi-Seed: model=${MODEL} seed=${SEED} (task ${TASK_ID}) ==="

# Common args
COMMON_ARGS=(
    --nc_path data/XRO_indices_oras5.nc
    --seed ${SEED}
    --train_start 1979-01
    --train_end 2001-12
    --val_start 1996-01
    --val_end 2001-12
    --test_start 2002-01
    --test_end 2022-12
    --epochs 2000
    --batch_size 128
    --lr 1e-3
    --extra_train_nc none
    --stochastic
    --members 100
    --train_noise_stage2
    --base_results_dir results_rebuttal_multiseed
)

${PYTHON} NXRO_train_out_of_sample.py \
    --model ${MODEL} \
    "${COMMON_ARGS[@]}"

echo "=== Done: model=${MODEL} seed=${SEED} ==="
