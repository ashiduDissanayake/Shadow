#!/usr/bin/env zsh
set -euo pipefail

cd "${0:A:h}/.."  # go to pipeline/

PY=python3.11
if ! command -v $PY >/dev/null 2>&1; then
  echo "Python 3.11 not found as 'python3.11'. Please install it (e.g., via pyenv or Homebrew)."
  exit 1
fi

VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
  $PY -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
pip install -r requirements.txt
echo "Venv ready at pipeline/$VENV_DIR"
