# Solution generation with Harbor

Terminal-Lego does not implement a solution runner. It produces accepted task
directories and delegates solution generation directly to `harbor==0.20.0`.

## Input contract

Pass the complete accepted dataset to Harbor:

```text
accepted/
  task_00001/
    instruction.md
    task.toml
    environment/Dockerfile
    environment/task_file/...
    solution/solve.sh
    tests/test.sh
    tests/test_outputs.py
```

Harbor discovers the direct task subdirectories under the path supplied with
`-p`. Production runs do not use a task-name filter or a second scheduler.

## Terminus 2

`terminus-2` calls the model from the Harbor host process:

```bash
export MODEL_NAME=openai/MiniMaxAI/MiniMax-M3
export TASKS_DIR=/absolute/path/to/accepted
export JOBS_DIR=/absolute/path/to/solution-generation

OPENAI_API_BASE=http://127.0.0.1:30114/v1 \
OPENAI_API_KEY=EMPTY \
.venv/bin/harbor run \
  -p "$TASKS_DIR" \
  -a terminus-2 \
  -m "$MODEL_NAME" \
  -n 32 \
  -o "$JOBS_DIR" \
  --job-name terminus-2-full
```

## mini-swe-agent

`mini-swe-agent` runs inside the task container, so its API base must be
reachable from Docker. On the tested CPU setup, the task memory limit also had
to be raised to 4096 MB for agent installation:

```bash
export MODEL_NAME=openai/MiniMaxAI/MiniMax-M3
export TASKS_DIR=/absolute/path/to/accepted
export JOBS_DIR=/absolute/path/to/solution-generation
export DOCKER_API_BASE=http://172.16.0.1:30115/v1

OPENAI_API_KEY=EMPTY \
.venv/bin/harbor run \
  -p "$TASKS_DIR" \
  -a mini-swe-agent \
  -m "$MODEL_NAME" \
  -n 32 \
  -o "$JOBS_DIR" \
  --job-name mini-swe-agent-full \
  --agent-env OPENAI_API_KEY=EMPTY \
  --agent-env OPENAI_API_BASE="$DOCKER_API_BASE" \
  --override-memory-mb 4096
```

Both commands use only native Harbor options. There are no repository-local
agent adapters, Docker Compose overlays, runtime bundles, or compatibility
proxies.

## Other agents

Harbor 0.20 ships other agents such as `opencode`, `langgraph`, `swe-agent`,
`aider`, and `codex`. Select one by changing `-a` and add agent-specific kwargs
only after a native smoke run demonstrates a concrete requirement.

DeepAgent is not a separately registered Harbor agent. Harbor provides the
generic `langgraph` agent; a DeepAgent graph project should only be added if it
becomes an explicit benchmark requirement.

## Workers, retries, and resume

- `-n N` controls concurrent trials.
- Harbor defaults to one attempt per task and zero retries.
- Use `-k` or `-r` only when a run explicitly requires different values.
- Resume Harbor's own job state:

```bash
.venv/bin/harbor job resume -p "$JOBS_DIR/<job-name>"
```

Do not reconstruct completed-task lists or run a separate collector.

## Output contract

Harbor artifacts are the only solution-generation output:

```text
<jobs-dir>/<job-name>/
  config.json
  result.json
  <trial-name>/
    config.json
    result.json
    agent/
      trajectory.json
    verifier/
      reward.txt
```

Some agents write additional native logs. Terminal-Lego does not create a
second summary, accepted/failed JSONL files, quota state, or batch state.
