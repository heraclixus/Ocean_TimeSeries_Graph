#!/bin/bash
#SBATCH --job-name=nxro_gate_abl
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=slurm/logs/gate_ablation_%a_%j.out
#SBATCH --error=slurm/logs/gate_ablation_%a_%j.err
#SBATCH --array=0-19

# =============================================================================
# Rebuttal Experiment 2: Seasonal gate ablation (SPB analysis)
# Addresses reviewer cGpF (empirical validation of SPB mitigation)
#
# 2 models (attentive, graph_pyg) x 2 conditions (with/without gate) x 5 seeds = 20 jobs
# =============================================================================

set -euo pipefail

WORK_DIR="${WORK_DIR:-/home/fredxu_squareup_com/Github/Ocean_TimeSeries_Graph}"
cd "${WORK_DIR}"
source ~/workspace/bin/activate-hermit 2>/dev/null || true
PYTHON="${WORK_DIR}/.venv/bin/python"

MODELS=(attentive graph_pyg)
GATE_FLAGS=("" "--disable_seasonal_gate")
GATE_TAGS=("with_gate" "no_gate")
SEEDS=(42 123 256 314 512)

TASK_ID=${SLURM_ARRAY_TASK_ID}
MODEL_IDX=$((TASK_ID / 10))
REM=$((TASK_ID % 10))
GATE_IDX=$((REM / 5))
SEED_IDX=$((REM % 5))

MODEL=${MODELS[$MODEL_IDX]}
GATE_FLAG=${GATE_FLAGS[$GATE_IDX]}
GATE_TAG=${GATE_TAGS[$GATE_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "=== Gate Ablation: model=${MODEL} gate=${GATE_TAG} seed=${SEED} (task ${TASK_ID}) ==="

${PYTHON} NXRO_train_out_of_sample.py \
    --model ${MODEL} \
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
    --extra_train_nc none \
    --run_tag "${GATE_TAG}" \
    --base_results_dir results_rebuttal_gate_ablation \
    ${GATE_FLAG}

echo "=== Done: model=${MODEL} gate=${GATE_TAG} seed=${SEED} ==="
