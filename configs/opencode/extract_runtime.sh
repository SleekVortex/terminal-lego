#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-terminal-lego/opencode-tools:1.17.13}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="${OPENCODE_RUNTIME_DIR:-${SCRIPT_DIR}/runtime}"

container_id="$(docker create "${IMAGE}")"
cleanup() {
  docker rm -f "${container_id}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

rm -rf "${RUNTIME_DIR}"
mkdir -p "${RUNTIME_DIR}"
docker cp "${container_id}:/opt/terminal-lego/opencode/." "${RUNTIME_DIR}/"
test -x "${RUNTIME_DIR}/bin/opencode"
"${RUNTIME_DIR}/bin/opencode" --version
echo "OpenCode runtime extracted to ${RUNTIME_DIR}"
