#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "$REPO_DIR"

if [[ -z "${PYTHON_BIN:-}" && -x "${REPO_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${REPO_DIR}/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

"${PYTHON_BIN}" -m ruff check \
  generator \
  scripts \
  sources \
  tests \
  validator \
  "$@"
