#!/bin/bash
#SBATCH --job-name=arx_bs2
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/arx_bs2_%a_%j.out
#SBATCH --error=slurm/logs/arx_bs2_%a_%j.err
#SBATCH --array=0-17

# Arxiv OOD boost v2 - memory-safe configs (hidden<=96)
# Target: beat GNNSafe++ AUROC 74.77, AUPR 83.21
# Current best: AUROC 73.36 (energy_ft, h=64, P=2, 200ep)
#
# Sweep: P x epochs x scoring (energy_ft vs gnnsafepp vs energy_mi_prop)
# Fixed: hidden=96, layers=3, K=4, S=6

set -euo pipefail

WORK_DIR="${WORK_DIR:-/home/fredxu_squareup_com/Github/TFEGNN}"
cd "${WORK_DIR}"
source ~/workspace/bin/activate-hermit 2>/dev/null || true

# 18 configs: 3 P x 2 epochs x 3 scoring
P_VALUES=(2 3 4)
EPOCH_VALUES=(350 600)
# 0=energy_ft, 1=gnnsafepp, 2=energy_mi_prop
SCORE_CONFIGS=("gcn_dssres_energy_ft" "gcn_dssres_gnnsafepp" "gcn_dssres_gnnsafepp_energy_mi_prop")

TASK_ID=${SLURM_ARRAY_TASK_ID}
P_IDX=$((TASK_ID / 6))
REM=$((TASK_ID % 6))
E_IDX=$((REM / 3))
S_IDX=$((REM % 3))

P=${P_VALUES[$P_IDX]}
EPOCHS=${EPOCH_VALUES[$E_IDX]}
CONFIG=${SCORE_CONFIGS[$S_IDX]}

TAG="arxiv_boost2_P${P}_e${EPOCHS}_${CONFIG##*_}"

echo "ARX_BS2: P=${P} epochs=${EPOCHS} config=${CONFIG} (task ${TASK_ID})"

EXTRA_FLAGS=""
if [[ "$CONFIG" == *"energy_mi_prop"* ]]; then
    EXTRA_FLAGS="--use_random_gates --P_gate 1"
fi

.venv/bin/python scripts/local_ood_density_compare.py \
    --dataset arxiv \
    --ood_type structure \
    --configs ${CONFIG} \
    --runs 3 \
    --epochs ${EPOCHS} \
    --hidden_channels 96 \
    --num_layers 3 \
    --P ${P} \
    --K_lp 4 --K_hp 4 \
    --S 6 \
    --dropout 0.5 \
    --lr 0.01 \
    --weight_decay 5e-4 \
    --dss_lambda_reg 0.01 \
    --warmup_base_epochs 50 \
    --residual_scale_init 0.1 \
    --lamda 1.0 \
    --m_in -5 --m_out -1 \
    --K 2 --alpha 0.5 \
    --tag "${TAG}" \
    ${EXTRA_FLAGS}
