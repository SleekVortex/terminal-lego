#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-glm52-opencode-compat-proxy-30004}"
BIND="${OPENCODE_COMPAT_PROXY_BIND:-172.16.0.1}"
PORT="${OPENCODE_COMPAT_PROXY_PORT:-30004}"
TARGET="${OPENCODE_COMPAT_PROXY_TARGET:-http://172.16.0.1:30003}"
LOG_PATH="${OPENCODE_COMPAT_PROXY_LOG:-/tmp/opencode-compat-proxy.jsonl}"
DISABLE_THINKING="${OPENCODE_COMPAT_DISABLE_THINKING:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROXY_SCRIPT="${REPO_DIR}/configs/opencode/openai_compat_proxy.py"

extra_args=()
case "${DISABLE_THINKING}" in
  1|true|TRUE|yes|YES)
    extra_args+=(--disable-thinking)
    ;;
esac

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"
else
  tmux new -d -s "${SESSION}" \
    "python3 ${PROXY_SCRIPT} \
      --bind ${BIND} \
      --port ${PORT} \
      --target ${TARGET} \
      --log-path ${LOG_PATH} \
      ${extra_args[*]}"
fi

sleep 1
if ! curl -fsS --max-time 10 "http://${BIND}:${PORT}/v1/models" >/dev/null; then
  echo "ERROR: OpenCode compat proxy did not answer on ${BIND}:${PORT}" >&2
  tmux capture-pane -pt "${SESSION}" -S -80 >&2 || true
  exit 1
fi

echo "OpenCode compat proxy ready: ${BIND}:${PORT} -> ${TARGET}"
