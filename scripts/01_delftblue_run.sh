#!/bin/bash
set -euo pipefail

module load 2025
module load python

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
REQ_FILE="$PROJECT_ROOT/requirements.txt"
RUN_VENV="$HOME/venvs/ar_hyperparameter_search"
WHEELHOUSE="$HOME/wheelhouse/ar_hyperparameter_search"
STAMP_FILE="$RUN_VENV/.requirements_installed"
WANDB_ROOT="$HOME/wandb_runs"

mkdir -p "$PROJECT_ROOT/logs" "$WANDB_ROOT"

if [ ! -f "$REQ_FILE" ]; then
    echo "ERROR: requirements.txt not found at $REQ_FILE"
    exit 1
fi

if [ ! -d "$RUN_VENV" ]; then
    echo "ERROR: runtime venv $RUN_VENV not found."
    echo "Run 01_setup_wandb_env.sh first on a login node."
    exit 1
fi

#SBATCH --job-name="ar_hyperparameter_search"
#SBATCH --partition=compute
#SBATCH --time=23:59:59
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=13
#SBATCH --mem-per-cpu=2G
#SBATCH --account=education-eemcs-bsc-ti
#SBATCH --output=logs/%x-%j.out


source "$RUN_VENV/bin/activate"

# Install from the local wheelhouse only.
# Do not load py-numpy here if numpy is already in requirements.txt.
if [ ! -f "$STAMP_FILE" ] || [ "$REQ_FILE" -nt "$STAMP_FILE" ]; then
    python -m pip install --upgrade pip
    python -m pip install --no-index --find-links="$WHEELHOUSE" -r "$REQ_FILE"
    touch "$STAMP_FILE"
fi

# W&B offline on compute nodes
export WANDB_MODE=offline
export WANDB_PROJECT=ar_hyperparameter_search
export WANDB_DIR="$WANDB_ROOT/$SLURM_JOB_ID"
export WANDB_RUN_GROUP="slurm-${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1

mkdir -p "$WANDB_DIR"

# Linux path, not Windows path
srun python -u src/wandb_runner.py