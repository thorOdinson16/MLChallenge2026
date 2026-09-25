#!/usr/bin/env bash
# Run a pipeline module with the right interpreter, e.g.:  ./run.sh ber.io
# BER_PYTHON can point at any Python with requirements.txt installed.
set -euo pipefail
cd "$(dirname "$0")/src"
PY="${BER_PYTHON:-/c/Users/abyud/miniconda3/envs/torch/python.exe}"
export PYTHONNOUSERSITE=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 USE_TF=0 TRANSFORMERS_NO_TF=1
exec "$PY" -W ignore::DeprecationWarning -m "$@"
