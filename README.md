<h1 align="center">Terminal-Lego</h1>

<p align="center">
<a href="https://huggingface.co/datasets/StephYang/Terminal-Lego-15k">Instances</a>
•
<a href="https://huggingface.co/datasets/SWE-Lego/Terminal-Lego-Traj-Deepseek-V3-2-15k">Trajectories</a>
•
<a href="https://github.com/SWE-Lego/terminal-lego">Code</a>
•
<a href="https://arxiv.org/pdf/2606.03461v1">Paper</a>
</p>

Terminal-Lego generates standalone terminal tasks from StackOverflow-derived
questions, validates every generated task with its reference solution, and
uses Harbor for agent solution rollouts.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/harbor --version
```

The repository pins `harbor==0.20.0`. Always use the executable from `.venv`.

Configure an OpenAI-compatible generation endpoint:

```bash
export OPENAI_API_BASE=http://127.0.0.1:30114/v1
export OPENAI_API_KEY=EMPTY
export MODEL_NAME=MiniMaxAI/MiniMax-M3
```

## Generate and validate tasks

There is one production entrypoint:

```bash
.venv/bin/python -m scripts.run_task_pipeline \
  --input /absolute/path/to/questions.jsonl \
  --output ./terminal-lego-work/task-generation/run-name \
  --generate-workers 32 \
  --validate-workers 32 \
  --timeout 300
```

The existing task-generation logic is unchanged. As soon as one candidate is
written, it is submitted to Docker validation. A task is copied to
`accepted/` only when its reference solution receives `reward == 1`.

The run writes:

```text
run-name/
  seed.json
  candidates/
  accepted/
  pipeline_logs/
  pipeline_state/
  pipeline_summary.json
```

Failed generation and validation attempts are reported and skipped. There are
no repair, regeneration, or target-count loops.

See `docs/task_generation_pipeline.md` for the task contract and filtering
options.

## Generate agent solutions

Pass the complete accepted dataset directly to Harbor:

```bash
export HARBOR_MODEL=openai/MiniMaxAI/MiniMax-M3
export HARBOR_TASKS=/absolute/path/to/run-name/accepted
export HARBOR_JOBS=/absolute/path/to/solution-generation

OPENAI_API_BASE=http://127.0.0.1:30114/v1 \
OPENAI_API_KEY=EMPTY \
.venv/bin/harbor run \
  -p "$HARBOR_TASKS" \
  -a terminus-2 \
  -m "$HARBOR_MODEL" \
  -n 32 \
  -o "$HARBOR_JOBS" \
  --job-name terminus-2-full
```

Harbor owns task discovery, workers, Docker lifecycle, verification, retries,
resume, and artifacts. This repository has no solver wrapper or collector.
Use another built-in Harbor agent by changing `-a`.

Resume an interrupted job with:

```bash
.venv/bin/harbor job resume -p "$HARBOR_JOBS/terminus-2-full"
```

See `docs/solution_generation_pipeline.md` for the exact Harbor contract.

## Task format

```text
task_00001/
  instruction.md
  task.toml
  environment/
    Dockerfile
    task_file/
  solution/
    solve.sh
  tests/
    test.sh
    test_outputs.py
```

This is directly consumable by `harbor run -p <accepted-dir>`.

## Checks

```bash
.venv/bin/pytest -q
scripts/lint.sh
```

Docker is required for task validation and Harbor runs.

## License

MIT
