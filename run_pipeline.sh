#!/bin/bash
set -euo pipefail

# ============================================================================
# Terminal-Lego Pipeline
# Full loop: Scrape SO → Generate tasks → Docker validation
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"

# Load .env if present
if [ -f "${SCRIPT_DIR}/.env" ]; then
    set -a
    source "${SCRIPT_DIR}/.env"
    set +a
fi

# Configuration (override via environment or .env)
: "${OPENAI_API_KEY:?Error: OPENAI_API_KEY not set. Copy .env.example to .env and fill in your keys.}"
: "${OPENAI_API_BASE:=https://api.openai.com/v1}"
: "${SO_API_KEY:=}"
: "${MODEL_NAME:=claude-opus-4-6}"
: "${GEN_WORKERS:=16}"
: "${VAL_WORKERS:=8}"
: "${VAL_TIMEOUT:=300}"
: "${CRAWL_COUNT:=5000}"

export OPENAI_API_KEY OPENAI_API_BASE MODEL_NAME

mkdir -p "${DATA_DIR}"

# Find next round number
ROUND=1
while [ -f "${DATA_DIR}/so_data_r${ROUND}.json" ] || \
      [ -d "${DATA_DIR}/candidates_r${ROUND}" ] || \
      [ -d "${DATA_DIR}/validated_r${ROUND}" ]; do
    ROUND=$((ROUND + 1))
done

echo "============================================================"
echo "Terminal-Lego Pipeline — starting from round ${ROUND}"
echo "  Each round: crawl ${CRAWL_COUNT} SO → generate tasks → validate"
echo "  Model: ${MODEL_NAME}"
echo "============================================================"

while true; do
    SO_DATA="${DATA_DIR}/so_data_r${ROUND}.json"
    CANDIDATES_DIR="${DATA_DIR}/candidates_r${ROUND}"
    VALIDATED_DIR="${DATA_DIR}/validated_r${ROUND}"
    LOG_FILE="${DATA_DIR}/run_r${ROUND}.log"

    echo ""
    echo "============================================================"
    echo "Round ${ROUND} — $(date)"
    echo "============================================================"

    # Phase 1: Scrape StackOverflow
    echo "[R${ROUND}] Phase 1: Crawling ${CRAWL_COUNT} SO questions..." | tee -a "${LOG_FILE}"
    python3 "${SCRIPT_DIR}/scraper/so_scraper.py" \
        --round "${ROUND}" \
        --output "${DATA_DIR}" \
        --count "${CRAWL_COUNT}" \
        --dedup-dir "${DATA_DIR}" \
        ${SO_API_KEY:+--api-key "${SO_API_KEY}"} \
        2>&1 | tee -a "${LOG_FILE}"

    if [ ! -f "${SO_DATA}" ]; then
        echo "[R${ROUND}] ERROR: Crawl failed. Retrying in 60s..." | tee -a "${LOG_FILE}"
        sleep 60
        continue
    fi

    SO_COUNT=$(python3 -c "import json; d=json.load(open('${SO_DATA}')); print(len(d['questions']))" 2>/dev/null || echo "0")
    echo "[R${ROUND}] Crawled ${SO_COUNT} questions" | tee -a "${LOG_FILE}"

    if [ "${SO_COUNT}" -lt 100 ]; then
        echo "[R${ROUND}] Too few questions (${SO_COUNT}), waiting 300s..." | tee -a "${LOG_FILE}"
        sleep 300
        ROUND=$((ROUND + 1))
        continue
    fi

    # Phase 2: Generate tasks
    mkdir -p "${CANDIDATES_DIR}"
    echo "[R${ROUND}] Phase 2: Generating tasks from ${SO_COUNT} questions..." | tee -a "${LOG_FILE}"
    python3 "${SCRIPT_DIR}/generator/task_generator.py" \
        --input "${SO_DATA}" \
        --output "${CANDIDATES_DIR}" \
        --workers "${GEN_WORKERS}" \
        --limit "${SO_COUNT}" \
        --api-base "${OPENAI_API_BASE}" \
        --api-key "${OPENAI_API_KEY}" \
        --model "${MODEL_NAME}" \
        2>&1 | tee -a "${LOG_FILE}"

    CANDIDATE_COUNT=$(find "${CANDIDATES_DIR}" -maxdepth 1 -type d -name "task_*" | wc -l)
    echo "[R${ROUND}] Generated ${CANDIDATE_COUNT} candidates" | tee -a "${LOG_FILE}"

    # Phase 3: Docker validation
    mkdir -p "${VALIDATED_DIR}"
    echo "[R${ROUND}] Phase 3: Validating ${CANDIDATE_COUNT} candidates..." | tee -a "${LOG_FILE}"
    python3 "${SCRIPT_DIR}/validator/validate_tasks.py" \
        --input "${CANDIDATES_DIR}" \
        --output "${VALIDATED_DIR}" \
        --workers "${VAL_WORKERS}" \
        --timeout "${VAL_TIMEOUT}" \
        2>&1 | tee -a "${LOG_FILE}"

    VALIDATED_COUNT=$(find "${VALIDATED_DIR}" -maxdepth 1 -type d -name "task_*" | wc -l)
    PASS_RATE=$(python3 -c "
c=${CANDIDATE_COUNT}; v=${VALIDATED_COUNT}
print(f'{v}/{c} = {v/c*100:.1f}%' if c > 0 else 'N/A')
" 2>/dev/null || echo "N/A")

    echo "[R${ROUND}] Done: ${PASS_RATE}" | tee -a "${LOG_FILE}"
    echo "[R${ROUND}] Finished at $(date)" | tee -a "${LOG_FILE}"

    ROUND=$((ROUND + 1))
    echo "Starting next round in 30s..."
    sleep 30
done
