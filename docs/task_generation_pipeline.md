# Task generation

Terminal-Lego has one production task pipeline:

```bash
.venv/bin/python -m scripts.run_task_pipeline \
  --input /absolute/path/to/questions.jsonl \
  --output ./terminal-lego-work/task-generation/run-name \
  --generate-workers 32 \
  --validate-workers 32 \
  --timeout 300 \
  --api-base "$OPENAI_API_BASE" \
  --api-key "$OPENAI_API_KEY" \
  --model "$MODEL_NAME"
```

It performs exactly three operations:

```text
StackOverflow JSONL
  -> normalize and select seed questions
  -> generate each task with the existing TaskGenerator
  -> immediately validate its reference solution in Docker
```

Generation and validation use independent worker pools and overlap. Failed
tasks are reported and skipped. The pipeline does not regenerate Dockerfiles,
repair tasks, or top up to a target accepted count.

## Seed selection

`sources/stackoverflow/prepare_dataset.py` converts input JSONL into the
generator contract:

```json
{
  "metadata": {"format": "generator-json"},
  "questions": []
}
```

`scripts.run_task_pipeline` forwards these optional filters:

- `--start N`;
- `--limit N`;
- `--min-score N`;
- `--category NAME`;
- `--per-category N`;
- `--sample-size N --distribution terminal-bench-2`;
- `--sort-by-score` or `--no-sort-by-score`.

Only questions with an accepted answer are sent to the generator.

## Existing generation logic

`generator/task_generator.py` runs the unchanged generation cascade:

```text
instruction
  -> environment
  -> reference solution
  -> tests
  -> Dockerfile
  -> final instruction rewrite
  -> task directory
```

Prompts remain under `prompts/task_generator/`. Static reviewers under
`generator/reviewers/` remain part of generation and reject malformed solution,
test, and Dockerfile outputs before a task is written.

Instruction rewriting is configured through:

```text
INSTRUCTION_REWRITE_MODE=off|concise|lossy|mixed
LOSSY_INSTRUCTION_RATIO=0.25
```

The default is deterministic `mixed`. This behavior is not changed by the
streaming validator.

## Immediate golden validation

When generation of one task finishes, `scripts.run_task_pipeline` submits its
directory to the existing `TaskValidator`. The validator:

1. builds `environment/Dockerfile`;
2. runs `solution/solve.sh` in the fresh task container;
3. runs `tests/test.sh`;
4. reads `/logs/verifier/reward.txt`;
5. copies the complete task to `accepted/` only for `reward == 1`.

Validation never changes a candidate. `failed`, `build_failed`, and `error`
results remain in the report and are not retried.

## Output contract

```text
run-name/
  seed.json
  candidates/
    task_XXXXX/
    generation_summary.json
  accepted/
    task_XXXXX/
    validation_report.json
    validate_tasks.log
    validation_logs/
  pipeline_logs/
  pipeline_state/
    01_prepare_seed.json
    02_generate_and_validate.json
  pipeline_summary.json
```

Every accepted directory has the native Harbor task layout:

```text
task_XXXXX/
  instruction.md
  task.toml
  environment/Dockerfile
  environment/task_file/...
  solution/solve.sh
  tests/test.sh
  tests/test_outputs.py
```

The complete `accepted/` directory can be passed directly to:

```bash
.venv/bin/harbor run -p /absolute/path/to/accepted ...
```

## Resume

Add `--resume` to reuse a completed seed stage, skip already complete candidate
question IDs, and validate existing complete candidates before newly generated
ones. A completed combined generation/validation marker skips the whole stage.

## Direct diagnostic commands

The underlying CLIs remain available for focused debugging:

```bash
.venv/bin/python -m generator.task_generator --help
.venv/bin/python -m validator.validate_tasks --help
.venv/bin/python -m sources.stackoverflow.prepare_dataset --help
```

They are implementation and diagnostic surfaces. The production dataset path
is `scripts.run_task_pipeline`.
