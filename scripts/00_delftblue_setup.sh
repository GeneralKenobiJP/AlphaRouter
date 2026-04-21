#!/bin/bash
set -euo pipefail

module load 2025
module load python

PROJECT_ROOT="${1:-$PWD}"
REQ_FILE="$PROJECT_ROOT/requirements.txt"
BOOTSTRAP_VENV="$HOME/venvs/ar_bootstrap"
RUN_VENV="$HOME/venvs/ar_hyperparameter_search"
WHEELHOUSE="$HOME/wheelhouse/ar_hyperparameter_search"

if [ ! -f "$REQ_FILE" ]; then
    echo "ERROR: requirements.txt not found at $REQ_FILE"
    exit 1
fi

mkdir -p "$HOME/venvs" "$WHEELHOUSE"

# Small bootstrap env used only to build wheels and run W&B CLI on login nodes
if [ ! -d "$BOOTSTRAP_VENV" ]; then
    python -m venv "$BOOTSTRAP_VENV"
fi
source "$BOOTSTRAP_VENV/bin/activate"

python -m pip install --upgrade pip wheel wandb

# Build/download all requirement wheels while on a node with internet access
python -m pip wheel -r "$REQ_FILE" -w "$WHEELHOUSE"

# Create the runtime venv once; packages get installed into it by the job script if needed
if [ ! -d "$RUN_VENV" ]; then
    python -m venv "$RUN_VENV"
fi

# Authenticate W&B once and store credentials locally
wandb login --verify=True

echo
echo "Setup complete."
echo "Bootstrap venv : $BOOTSTRAP_VENV"
echo "Runtime venv   : $RUN_VENV"
echo "Wheelhouse     : $WHEELHOUSE"