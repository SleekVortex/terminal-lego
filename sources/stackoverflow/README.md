# StackOverflow dump datasets

This directory contains the local-dump StackOverflow preparation flow used before
`python -m generator.task_generator`.

The expected input is JSONL produced from `Posts.xml` with at least:

```json
{
  "question_id": 123,
  "title": "...",
  "body": "...",
  "tags": ["python", "regex"],
  "score": 42,
  "accepted_answer_id": 456,
  "accepted_answer": {
    "answer_id": 456,
    "body": "...",
    "score": 50
  }
}
```

Optional fields such as `categories` and `selected_category` are preserved. If
they are missing, `prepare_dataset.py` classifies the row from tags using
`terminal_bench_tag_taxonomy.json`.

## Create a generator seed

Balanced smoke seed from an already-filtered JSONL:

```bash
python -m sources.stackoverflow.prepare_dataset \
  --input /data/avzavodov/stackoverflow_posts_xml/sets/so_posts_6y_score_ge_0_our_categories_top100k_tbench_dist_20260702T170248+0300/questions.jsonl \
  --output ./data/so_seed_balanced32.json \
  --format generator-json \
  --per-category 2 \
  --sort-by-score
```

Then generate tasks:

```bash
python -m generator.task_generator \
  --input ./data/so_seed_balanced32.json \
  --output ./data/candidates_so_seed_balanced32 \
  --workers 1 \
  --api-base "$OPENAI_API_BASE" \
  --api-key "$OPENAI_API_KEY" \
  --model "$MODEL_NAME"
```

## Create a 100k benchmark-distribution JSONL

```bash
python -m sources.stackoverflow.prepare_dataset \
  --input /data/avzavodov/stackoverflow_posts_xml/sets/so_posts_6y_score_ge_0_our_categories_jsonl_20260702T165727+0300/questions.jsonl \
  --output ./data/so_top100k_tbench_dist.jsonl \
  --format jsonl \
  --sample-size 100000 \
  --distribution terminal-bench-2 \
  --min-score 0
```

The sampler writes a single `selected_category` per row and preserves the full
multi-label `categories` list.
