#!/bin/bash
#SBATCH --job-name=nxro_mlp_best
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=slurm/logs/mlp_best_%a_%j.out
#SBATCH --error=slurm/logs/mlp_best_%a_%j.err
#SBATCH --array=0-9

# Best tiny MLP config: hidden=4, wd=0.05, 10 seeds
set -euo pipefail

WORK_DIR="${WORK_DIR:-/home/fredxu_squareup_com/Github/Ocean_TimeSeries_Graph}"
cd "${WORK_DIR}"
source ~/workspace/bin/activate-hermit 2>/dev/null || true
PYTHON="${WORK_DIR}/.venv/bin/python"

SEEDS=(42 123 256 314 512 777 1024 1337 2024 9999)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "=== MLP Best (h=4 wd=0.05): seed=${SEED} ==="

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
    --weight_decay 0.05 \
    --res_reg 1e-4 \
    --hidden_mlp 4 \
    --extra_train_nc none \
    --stochastic \
    --members 100 \
    --train_noise_stage2 \
    --run_tag "h4_wd005" \
    --base_results_dir results_rebuttal_mlp_best

echo "=== Done: seed=${SEED} ==="
