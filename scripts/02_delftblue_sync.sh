#!/bin/bash
set -euo pipefail

module load 2025
module load python

BOOTSTRAP_VENV="$HOME/venvs/ar_bootstrap"
WANDB_ROOT="${1:-$HOME/wandb_runs}"

if [ ! -d "$BOOTSTRAP_VENV" ]; then
    echo "ERROR: bootstrap venv $BOOTSTRAP_VENV not found."
    echo "Run 01_setup_wandb_env.sh first on a login node."
    exit 1
fi

if [ ! -d "$WANDB_ROOT" ]; then
    echo "ERROR: W&B offline directory $WANDB_ROOT not found."
    exit 1
fi

source "$BOOTSTRAP_VENV/bin/activate"

# Optional sanity check for stored credentials
wandb login --verify=True >/dev/null

# Sync everything under the offline run root
wandb sync --include-offline "$WANDB_ROOT"