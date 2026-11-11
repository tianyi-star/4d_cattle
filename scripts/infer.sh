#!/usr/bin/env bash
# Inference entry for walking replay and future-gait prediction.
# Pass a checkpoint produced by your own training run:
#   bash scripts/infer.sh --checkpoint /path/to/checkpoint.pt --sequence seq_id
# This repository does not release a weight file.

set -euo pipefail

if command -v python3 >/dev/null 2>&1; then
  python_cmd="python3"
elif command -v python >/dev/null 2>&1; then
  python_cmd="python"
else
  echo "Python is not installed." >&2
  exit 1
fi

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${DIR}"
export PYTHONPATH="${DIR}/src:${PYTHONPATH:-}"

"${python_cmd}" -m cattle4d.engine.infer \
  --config "${DIR}/configs/default.yaml" \
  --paths "${DIR}/configs/path.yaml" \
  "$@"
