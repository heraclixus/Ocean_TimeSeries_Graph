#!/bin/bash
# =============================================================================
# Master script: rsync to server, submit all rebuttal experiments, and
# optionally rsync results back.
#
# Usage:
#   ./slurm/submit_rebuttal.sh push       # rsync code to server
#   ./slurm/submit_rebuttal.sh submit     # ssh + sbatch all jobs
#   ./slurm/submit_rebuttal.sh pull       # rsync results back
#   ./slurm/submit_rebuttal.sh all        # push + submit
#   ./slurm/submit_rebuttal.sh status     # check job status
# =============================================================================

set -euo pipefail

SERVER="slurmsmall-prod"
REMOTE_DIR="/home/fredxu_squareup_com/Github/Ocean_TimeSeries_Graph"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

ACTION="${1:-all}"

push_code() {
    echo "=== Syncing code to ${SERVER}:${REMOTE_DIR} ==="
    rsync -avz --delete \
        --exclude='results_*' \
        --exclude='*.pt' \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='*.pyc' \
        --exclude='.venv' \
        --exclude='slurm/logs/*.out' \
        --exclude='slurm/logs/*.err' \
        "${LOCAL_DIR}/" "${SERVER}:${REMOTE_DIR}/"
    # Ensure logs directory exists
    ssh "${SERVER}" "mkdir -p ${REMOTE_DIR}/slurm/logs"
    echo "=== Push complete ==="
}

submit_jobs() {
    echo "=== Submitting rebuttal experiments ==="
    ssh "${SERVER}" bash -l <<REMOTE_EOF
cd ${REMOTE_DIR}
mkdir -p slurm/logs

echo "--- Experiment 1: Multi-seed core models (60 jobs) ---"
JOB1=\$(sbatch --parsable slurm/rebuttal_multiseed.slurm)
echo "  Submitted: \${JOB1}"

echo "--- Experiment 2: Seasonal gate ablation (20 jobs) ---"
JOB2=\$(sbatch --parsable slurm/rebuttal_seasonal_gate_ablation.slurm)
echo "  Submitted: \${JOB2}"

echo "--- Experiment 3: Stochastic ablation (30 jobs) ---"
JOB3=\$(sbatch --parsable slurm/rebuttal_stochastic_ablation.slurm)
echo "  Submitted: \${JOB3}"

echo "--- Experiment 4: Data scarcity curve (60 jobs) ---"
JOB4=\$(sbatch --parsable slurm/rebuttal_data_scarcity.slurm)
echo "  Submitted: \${JOB4}"

echo ""
echo "=== All jobs submitted (170 total). Monitor with: squeue -u \$USER ==="
REMOTE_EOF
}

pull_results() {
    echo "=== Pulling results from ${SERVER} ==="
    for dir in results_rebuttal_multiseed results_rebuttal_gate_ablation \
               results_rebuttal_stochastic_ablation results_rebuttal_data_scarcity; do
        echo "  Syncing ${dir}/ ..."
        rsync -avz "${SERVER}:${REMOTE_DIR}/${dir}/" "${LOCAL_DIR}/${dir}/" 2>/dev/null || \
            echo "  (${dir} not found yet)"
    done
    # Also pull slurm logs
    rsync -avz "${SERVER}:${REMOTE_DIR}/slurm/logs/" "${LOCAL_DIR}/slurm/logs/" 2>/dev/null || true
    echo "=== Pull complete ==="
}

check_status() {
    echo "=== Job status on ${SERVER} ==="
    ssh "${SERVER}" "squeue -u \$USER -o '%.10i %.20j %.8T %.10M %.6D %R' | head -40"
    echo ""
    ssh "${SERVER}" "echo 'Completed rebuttal results:' && ls -d ${REMOTE_DIR}/results_rebuttal_*/ 2>/dev/null | while read d; do echo \"  \$(basename \$d): \$(find \$d -name '*_summary.json' | wc -l) runs done\"; done"
}

case "${ACTION}" in
    push)    push_code ;;
    submit)  submit_jobs ;;
    pull)    pull_results ;;
    status)  check_status ;;
    all)     push_code && submit_jobs ;;
    *)       echo "Usage: $0 {push|submit|pull|status|all}" && exit 1 ;;
esac
