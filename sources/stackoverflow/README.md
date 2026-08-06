# StackOverflow dump datasets

This directory contains the StackOverflow preparation code used by the canonical
`scripts.run_task_pipeline` entrypoint.

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

## Generate and validate tasks

The production pipeline prepares the seed, generates tasks, and immediately
validates each generated candidate:

```bash
.venv/bin/python -m scripts.run_task_pipeline \
  --input /data/avzavodov/stackoverflow_posts_xml/sets/so_posts_6y_score_ge_0_our_categories_top100k_tbench_dist_20260702T170248+0300/questions.jsonl \
  --output ./terminal-lego-work/task-generation/so-balanced32 \
  --per-category 2 \
  --generate-workers 32 \
  --validate-workers 32
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
