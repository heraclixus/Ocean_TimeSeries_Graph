#!/bin/bash
#SBATCH --job-name=nxro_skill
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=slurm/logs/skill_curves_%j.out
#SBATCH --error=slurm/logs/skill_curves_%j.err

set -euo pipefail

WORK_DIR="${WORK_DIR:-/home/fredxu_squareup_com/Github/Ocean_TimeSeries_Graph}"
cd "${WORK_DIR}"
source ~/workspace/bin/activate-hermit 2>/dev/null || true
PYTHON="${WORK_DIR}/.venv/bin/python"
export PYTHONPATH="${WORK_DIR}:${PYTHONPATH:-}"

echo "=== Generating skill curve figure ==="
${PYTHON} scripts/generate_rebuttal_skill_curves.py \
    --device cuda \
    --results_dir results_rebuttal_multiseed \
    --mlp_dir results_rebuttal_mlp_best

echo "=== Done ==="
