#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/generate_solutions.sh TASKS_DIR [JOBS_DIR] [extra solution_generator args...]

Environment:
  AGENT                   Harbor agent. Default: terminus-2
  MODEL_NAME              Model name passed to the agent.
  HARBOR_ENV              Harbor environment. Default: docker
  PYTHON_BIN              Python executable. Default: python3
  OPENAI_API_BASE         OpenAI-compatible API base.
  OPENAI_API_KEY          API key. Default: EMPTY
  JOB_NAME                Harbor job name. Default: solution-rollouts-<timestamp>
  N_ATTEMPTS              Attempts per task. Default: 1
  N_CONCURRENT            Concurrent trials. Default: 1
  MAX_RETRIES             Harbor retry count. Default: 0
  MATERIALIZE_INSTRUCTION Extra instruction path for /logs/artifacts/solve.sh.

Examples:
  AGENT=terminus-2 MODEL_NAME=openai/glm-5.2-fp8 \
  OPENAI_API_BASE=http://localhost:30002/v1 OPENAI_API_KEY=EMPTY \
  MATERIALIZE_INSTRUCTION=/home/avzavodov/projects/swe-lego/materialize_solution.md \
  scripts/generate_solutions.sh ./validated ./runs
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

TASKS_DIR="$1"
JOBS_DIR="${2:-./runs}"
if [[ $# -ge 2 ]]; then
  shift 2
else
  shift 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
JOB_NAME="${JOB_NAME:-solution-rollouts-$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

args=(
  --tasks-dir "${TASKS_DIR}"
  --jobs-dir "${JOBS_DIR}"
  --job-name "${JOB_NAME}"
  --agent "${AGENT:-terminus-2}"
  --env "${HARBOR_ENV:-docker}"
  --n-attempts "${N_ATTEMPTS:-1}"
  --n-concurrent "${N_CONCURRENT:-1}"
  --max-retries "${MAX_RETRIES:-0}"
  --api-key "${OPENAI_API_KEY}"
  --trajectory-raw-content
  --trajectory-linear-history
)

if [[ -n "${MODEL_NAME:-}" ]]; then
  args+=(--model "${MODEL_NAME}")
fi

if [[ -n "${OPENAI_API_BASE:-}" ]]; then
  args+=(--api-base "${OPENAI_API_BASE}")
fi

if [[ -n "${MATERIALIZE_INSTRUCTION:-}" ]]; then
  args+=(--extra-instruction-path "${MATERIALIZE_INSTRUCTION}")
fi

"${PYTHON_BIN}" "${REPO_DIR}/generator/solution_generator.py" "${args[@]}" "$@"
