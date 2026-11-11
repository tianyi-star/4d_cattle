#!/usr/bin/env bash
# Train entry. Extra arguments are forwarded, for example: --dry-run
# The Python module validates the config and stops inside fit(), which has
# no optimizer step until an experiment is filled in.

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

"${python_cmd}" -m cattle4d.engine.train \
  --config "${DIR}/configs/default.yaml" \
  --paths "${DIR}/configs/path.yaml" \
  "$@"
