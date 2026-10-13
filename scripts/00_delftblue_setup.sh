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

if [ ! -d "$BOOTSTRAP_VENV" ]; then
    python -m venv "$BOOTSTRAP_VENV"
fi
source "$BOOTSTRAP_VENV/bin/activate"

python -m pip install --upgrade pip wheel wandb
python -m pip wheel -r "$REQ_FILE" -w "$WHEELHOUSE"

if [ ! -d "$RUN_VENV" ]; then
    python -m venv "$RUN_VENV"
fi

wandb login --verify

echo
echo "Setup complete."
echo "Bootstrap venv : $BOOTSTRAP_VENV"
echo "Runtime venv   : $RUN_VENV"
echo "Wheelhouse     : $WHEELHOUSE"
