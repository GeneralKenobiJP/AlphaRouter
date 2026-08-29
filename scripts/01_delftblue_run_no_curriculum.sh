#!/bin/bash
#SBATCH --job-name=ar_hyperparameter_search
#SBATCH --partition=compute
#SBATCH --time=23:59:59
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=2G
#SBATCH --account=education-eemcs-bsc-ti
#SBATCH --output=%x-%j.out

set -euo pipefail

module load 2025
module load python

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"
REQ_FILE="$PROJECT_ROOT/requirements.txt"
RUN_VENV="$HOME/venvs/ar_hyperparameter_search"
WHEELHOUSE="$HOME/wheelhouse/ar_hyperparameter_search"
STAMP_FILE="$RUN_VENV/.requirements_installed"
WANDB_ROOT="/scratch/$USER/wandb_runs"
RUNS="${RUNS:-36}"
WORKERS="${WORKERS:-$SLURM_CPUS_PER_TASK}"
DEVICE="${DEVICE:-cpu}"

if [ ! -f "$REQ_FILE" ]; then
    echo "ERROR: requirements.txt not found at $REQ_FILE"
    exit 1
fi

if [ ! -d "$RUN_VENV" ]; then
    echo "ERROR: runtime venv $RUN_VENV not found."
    echo "Run 00_delftblue_setup_fixed.sh first on a login node."
    exit 1
fi

cd "$PROJECT_ROOT"
mkdir -p "$WANDB_ROOT"

source "$RUN_VENV/bin/activate"

if [ ! -f "$STAMP_FILE" ] || [ "$REQ_FILE" -nt "$STAMP_FILE" ]; then
    python -m pip install --upgrade pip
    python -m pip install --no-index --find-links="$WHEELHOUSE" -r "$REQ_FILE"
    touch "$STAMP_FILE"
fi

export WANDB_MODE=offline
export WANDB_PROJECT=alpharouter
export WANDB_DIR="$WANDB_ROOT/$SLURM_JOB_ID"
export WANDB_RUN_GROUP="slurm-${SLURM_JOB_ID}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "$WANDB_DIR"

srun -n 1 python -u src/wandb_runner_offline_v3.py --runs "$RUNS" --workers "$WORKERS" --device "$DEVICE"
#9739373