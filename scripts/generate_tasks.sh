#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/generate_tasks.sh INPUT_JSONL OUTPUT_DIR

Environment:
  SEED_JSON              Seed output path. Default: OUTPUT_DIR/seed.json
  CANDIDATES_DIR         Generated tasks dir. Default: OUTPUT_DIR/candidates
  MIN_SCORE              Optional StackOverflow score filter.
  CATEGORY               Optional Terminal-Bench category filter. Repeat by comma.
  PER_CATEGORY           Optional top N rows per category, e.g. 2.
  SAMPLE_SIZE            Optional benchmark-distribution sample size.
  DISTRIBUTION           Distribution name for SAMPLE_SIZE. Default: terminal-bench-2
  LIMIT                  Optional first/top rows limit.
  START                  Optional start offset for first rows mode. Default: 0
  SORT_BY_SCORE          1 to rank by SO score. Default: 1
  WORKERS                Task generator workers. Default: 1
  MODEL_NAME             LLM model for task generation.
  OPENAI_API_BASE        OpenAI-compatible API base.
  OPENAI_API_KEY         API key. Default: EMPTY

Any extra CLI arguments are passed to generator/task_generator.py.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 2
fi

INPUT_JSONL="$1"
OUTPUT_DIR="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

SEED_JSON="${SEED_JSON:-${OUTPUT_DIR}/seed.json}"
CANDIDATES_DIR="${CANDIDATES_DIR:-${OUTPUT_DIR}/candidates}"
START="${START:-0}"
SORT_BY_SCORE="${SORT_BY_SCORE:-1}"
WORKERS="${WORKERS:-1}"
OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

mkdir -p "${OUTPUT_DIR}" "${CANDIDATES_DIR}"

prepare_args=(
  --input "${INPUT_JSONL}"
  --output "${SEED_JSON}"
  --format generator-json
  --start "${START}"
)

if [[ -n "${MIN_SCORE:-}" ]]; then
  prepare_args+=(--min-score "${MIN_SCORE}")
fi

if [[ -n "${CATEGORY:-}" ]]; then
  IFS=',' read -ra categories <<< "${CATEGORY}"
  for category in "${categories[@]}"; do
    [[ -n "${category}" ]] && prepare_args+=(--category "${category}")
  done
fi

if [[ -n "${PER_CATEGORY:-}" ]]; then
  prepare_args+=(--per-category "${PER_CATEGORY}")
elif [[ -n "${SAMPLE_SIZE:-}" ]]; then
  prepare_args+=(--sample-size "${SAMPLE_SIZE}" --distribution "${DISTRIBUTION:-terminal-bench-2}")
elif [[ -n "${LIMIT:-}" ]]; then
  prepare_args+=(--limit "${LIMIT}")
fi

if [[ "${SORT_BY_SCORE}" != "0" ]]; then
  prepare_args+=(--sort-by-score)
fi

python "${REPO_DIR}/stackoverflow/prepare_dataset.py" "${prepare_args[@]}"

generator_args=(
  --input "${SEED_JSON}"
  --output "${CANDIDATES_DIR}"
  --workers "${WORKERS}"
  --api-key "${OPENAI_API_KEY}"
)

if [[ -n "${OPENAI_API_BASE:-}" ]]; then
  generator_args+=(--api-base "${OPENAI_API_BASE}")
fi

if [[ -n "${MODEL_NAME:-}" ]]; then
  generator_args+=(--model "${MODEL_NAME}")
fi

python "${REPO_DIR}/generator/task_generator.py" "${generator_args[@]}" "$@"

echo "Seed: ${SEED_JSON}"
echo "Candidates: ${CANDIDATES_DIR}"

