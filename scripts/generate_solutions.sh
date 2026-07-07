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
  DOCKER_NETWORK_STRATEGY Docker network strategy. Default: bridge
                          Values: bridge, compose
  OPENCODE_RUNTIME_DIR    Preinstalled OpenCode runtime dir for
                          AGENT=preinstalled-opencode.
                          Default: configs/opencode/runtime
  CLEANUP_DOCKER          Remove leftover Docker compose containers/networks
                          for this job on exit. Default: 1
  OPENAI_API_BASE         OpenAI-compatible API base.
  OPENAI_API_KEY          API key. Default: EMPTY
  JOB_NAME                Harbor job name. Default: solution-runs-<timestamp>
  N_ATTEMPTS              Attempts per task. Default: 1
  N_CONCURRENT            Concurrent trials. Default: 1
  MAX_RETRIES             Harbor retry count. Default: 0
  MATERIALIZE_INSTRUCTION Extra instruction path for /logs/artifacts/solve.sh.
                          Default: prompts/solution_generator/materialize_solution.md

Examples:
  AGENT=terminus-2 MODEL_NAME=openai/glm-5.2-fp8 \
  OPENAI_API_BASE=http://localhost:30002/v1 OPENAI_API_KEY=EMPTY \
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
MATERIALIZE_INSTRUCTION="${MATERIALIZE_INSTRUCTION:-${REPO_DIR}/prompts/solution_generator/materialize_solution.md}"

make_abs_path() {
  case "$1" in
    /*) printf "%s\n" "$1" ;;
    *) printf "%s\n" "$(pwd)/$1" ;;
  esac
}

TASKS_DIR="$(make_abs_path "$TASKS_DIR")"
JOBS_DIR="$(make_abs_path "$JOBS_DIR")"

OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
JOB_NAME="${JOB_NAME:-solution-runs-$(date -u +%Y%m%dT%H%M%SZ)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DOCKER_NETWORK_STRATEGY="${DOCKER_NETWORK_STRATEGY:-bridge}"
CLEANUP_DOCKER="${CLEANUP_DOCKER:-1}"
OPENCODE_RUNTIME_DIR="${OPENCODE_RUNTIME_DIR:-${REPO_DIR}/configs/opencode/runtime}"

if [[ "${AGENT:-terminus-2}" == "preinstalled-opencode" ]]; then
  OPENAI_API_BASE="${OPENAI_API_BASE:-http://host.docker.internal:30003/v1}"
fi

if [[ -n "${OPENAI_API_BASE:-}" && -z "${OPENAI_BASE_URL:-}" ]]; then
  OPENAI_BASE_URL="${OPENAI_API_BASE}"
  export OPENAI_BASE_URL
fi

compose_project_name() {
  local name="$1"
  name="$(printf "%s" "$name" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_-]/-/g')"
  if [[ ! "$name" =~ ^[a-z0-9] ]]; then
    name="0$name"
  fi
  printf "%s\n" "$name"
}

cleanup_compose_projects() {
  local job_dir="$1"
  [[ -n "$job_dir" && -d "$job_dir" ]] || return 0
  command -v docker >/dev/null 2>&1 || return 0

  local projects=()
  local dir base project existing
  while IFS= read -r dir; do
    base="${dir##*/}"
    project="$(compose_project_name "$base")"
    existing=" ${projects[*]} "
    if [[ "$existing" != *" $project "* ]]; then
      projects+=("$project")
    fi
  done < <(find "$job_dir" -mindepth 0 -maxdepth 2 -type f \( -name config.json -o -name result.json \) -exec dirname {} \; 2>/dev/null | sort -u)

  ((${#projects[@]} > 0)) || return 0

  local project_file
  project_file="$(mktemp)"
  printf "%s\n" "${projects[@]}" > "$project_file"

  local containers=()
  local networks=()
  local resource_id project_label
  while read -r resource_id project_label; do
    [[ -n "${resource_id:-}" && -n "${project_label:-}" ]] || continue
    if grep -Fxq "$project_label" "$project_file"; then
      containers+=("$resource_id")
    fi
  done < <(docker ps -a --format '{{.ID}} {{.Label "com.docker.compose.project"}}' 2>/dev/null || true)

  if ((${#containers[@]} > 0)); then
    docker rm -f "${containers[@]}" >/dev/null 2>&1 || true
  fi

  while read -r resource_id project_label; do
    [[ -n "${resource_id:-}" && -n "${project_label:-}" ]] || continue
    if grep -Fxq "$project_label" "$project_file"; then
      networks+=("$resource_id")
    fi
  done < <(docker network ls --format '{{.ID}} {{.Label "com.docker.compose.project"}}' 2>/dev/null || true)

  if ((${#networks[@]} > 0)); then
    docker network rm "${networks[@]}" >/dev/null 2>&1 || true
  fi

  rm -f "$project_file"
}

cleanup() {
  local status=$?
  trap - EXIT
  case "$CLEANUP_DOCKER" in
    1|true|TRUE|yes|YES)
      cleanup_compose_projects "${JOBS_DIR%/}/${JOB_NAME}" || true
      ;;
  esac
  exit "$status"
}
trap cleanup EXIT

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

if [[ "${AGENT:-terminus-2}" == "preinstalled-opencode" ]]; then
  if [[ ! -x "${OPENCODE_RUNTIME_DIR}/bin/opencode" ]]; then
    echo "ERROR: AGENT=preinstalled-opencode requires executable:" >&2
    echo "  ${OPENCODE_RUNTIME_DIR}/bin/opencode" >&2
    echo "Build/extract it with configs/opencode/extract_runtime.sh first." >&2
    exit 2
  fi
  export OPENCODE_RUNTIME_DIR
  args+=(--extra-docker-compose "${REPO_DIR}/configs/opencode/docker-compose-runtime.yaml")
fi

case "$DOCKER_NETWORK_STRATEGY" in
  bridge)
    args+=(--extra-docker-compose "${REPO_DIR}/configs/harbor/docker-compose-bridge-network.yaml")
    ;;
  compose|default|off|none)
    ;;
  *)
    echo "ERROR: DOCKER_NETWORK_STRATEGY must be one of: bridge, compose" >&2
    exit 2
    ;;
esac

if [[ -n "${MATERIALIZE_INSTRUCTION:-}" ]]; then
  args+=(--extra-instruction-path "${MATERIALIZE_INSTRUCTION}")
fi

cd "$REPO_DIR"
"${PYTHON_BIN}" -m solvers.run_solutions "${args[@]}" "$@"
