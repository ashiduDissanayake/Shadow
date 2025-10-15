#!/usr/bin/env zsh
set -euo pipefail

cd "${0:A:h}/.."  # go to pipeline/
source .venv/bin/activate
python feature_extraction_s2.py
