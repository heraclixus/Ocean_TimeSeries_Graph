#!/bin/bash
#SBATCH --job-name=nxro_stoch_abl
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=slurm/logs/stochastic_ablation_%a_%j.out
#SBATCH --error=slurm/logs/stochastic_ablation_%a_%j.err
#SBATCH --array=0-29

# =============================================================================
# Rebuttal Experiment 3: Controlled stochastic ablation
# Addresses reviewer jHw1 (isolate drift vs noise improvement)
#
# 3 models (res/MLP, attentive, graph_pyg) x 2 noise modes x 5 seeds = 30 jobs
# Noise modes: post-hoc AR(1) vs Stage 2 likelihood-optimized AR(1)
# All use same val split for fair comparison
# =============================================================================

set -euo pipefail

WORK_DIR="${WORK_DIR:-/home/fredxu_squareup_com/Github/Ocean_TimeSeries_Graph}"
cd "${WORK_DIR}"
source ~/workspace/bin/activate-hermit 2>/dev/null || true
PYTHON="${WORK_DIR}/.venv/bin/python"

MODELS=(res attentive graph_pyg)
# 0 = post-hoc only, 1 = stage2 optimized
NOISE_MODES=(0 1)
NOISE_TAGS=("posthoc" "stage2")
SEEDS=(42 123 256 314 512)

TASK_ID=${SLURM_ARRAY_TASK_ID}
MODEL_IDX=$((TASK_ID / 10))
REM=$((TASK_ID % 10))
NOISE_IDX=$((REM / 5))
SEED_IDX=$((REM % 5))

MODEL=${MODELS[$MODEL_IDX]}
NOISE_MODE=${NOISE_MODES[$NOISE_IDX]}
NOISE_TAG=${NOISE_TAGS[$NOISE_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "=== Stochastic Ablation: model=${MODEL} noise=${NOISE_TAG} seed=${SEED} (task ${TASK_ID}) ==="

STAGE2_FLAG=""
if [ "$NOISE_MODE" -eq 1 ]; then
    STAGE2_FLAG="--train_noise_stage2"
fi

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
    --stochastic \
    --members 100 \
    ${STAGE2_FLAG} \
    --run_tag "${NOISE_TAG}" \
    --base_results_dir results_rebuttal_stochastic_ablation

echo "=== Done: model=${MODEL} noise=${NOISE_TAG} seed=${SEED} ==="
