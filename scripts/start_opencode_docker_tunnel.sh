#!/usr/bin/env bash
set -euo pipefail

SESSION="${SESSION:-glm52-opencode-docker-tunnel-30003}"
LOCAL_BIND="${LOCAL_BIND:-172.16.0.1}"
LOCAL_PORT="${LOCAL_PORT:-30003}"
REMOTE_TARGET="${REMOTE_TARGET:-localhost:30001}"
SSH_HOST="${SSH_HOST:-lm-mpi-job-3aed4232-c8b6-43e6-9294-1ff68df3c96d-mpimaster-0.ai0001071-01058@ssh-sr008-jupyter.ai.cloud.ru}"
SSH_KEY="${SSH_KEY:-/home/avzavodov/projects/mlspace__private_key.txt}"
SSH_PORT="${SSH_PORT:-2222}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}"
  exit 0
fi

tmux new -d -s "${SESSION}" \
  "ssh -N -T -g \
    -L ${LOCAL_BIND}:${LOCAL_PORT}:${REMOTE_TARGET} \
    -p ${SSH_PORT} \
    -i ${SSH_KEY} \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o StrictHostKeyChecking=no \
    -o ExitOnForwardFailure=yes \
    ${SSH_HOST}"

echo "Started ${SESSION}: ${LOCAL_BIND}:${LOCAL_PORT} -> ${REMOTE_TARGET}"
