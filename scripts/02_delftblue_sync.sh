#!/bin/bash
set -euo pipefail

module load 2025
module load python

BOOTSTRAP_VENV="$HOME/venvs/ar_bootstrap"
WANDB_ROOT="${1:-/scratch/$USER/wandb_runs}"

if [ ! -d "$BOOTSTRAP_VENV" ]; then
    echo "ERROR: bootstrap venv $BOOTSTRAP_VENV not found."
    echo "Run 00_delftblue_setup.sh first on a login node."
    exit 1
fi

if [ ! -d "$WANDB_ROOT" ]; then
    echo "ERROR: W&B offline directory $WANDB_ROOT not found."
    exit 1
fi

source "$BOOTSTRAP_VENV/bin/activate"
wandb login --verify >/dev/null

found=0
synced=0

# Find actual run directories, not just the parent wandb directory.
while IFS= read -r -d '' run_dir; do
    found=1
    echo "Syncing $run_dir"
    if wandb sync "$run_dir"; then
        synced=$((synced + 1))
    else
        echo "WARNING: failed to sync $run_dir" >&2
    fi
done < <(find "$WANDB_ROOT" -type d \( -name 'offline-run-*' -o -name 'run-*' \) -print0)

if [ "$found" -eq 0 ]; then
    echo "No W&B run directories found under $WANDB_ROOT"
    echo "Inspect with: find $WANDB_ROOT -maxdepth 4 | sort"
    exit 1
fi

echo "Synced $synced run(s)."
