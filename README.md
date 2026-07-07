<h1 align="center"> Terminal-Lego: What Makes Interaction Trajectories Effective for Training Terminal Agents? </h1>

<p align="center">
<a href="https://huggingface.co/datasets/StephYang/Terminal-Lego-15k" > 🤗 Instances </a>
•
<a href="https://huggingface.co/datasets/SWE-Lego/Terminal-Lego-Traj-Deepseek-V3-2-15k" > 🤗 Trajectories </a>
•
<a href="https://huggingface.co/StephYang/Terminal-Lego-Qwen3-8B" > 🤗 Terminal-Lego-Qwen3-8B/32B</a>
•
<a href="https://github.com/SWE-Lego/terminal-lego" > 🧑‍💻 Code</a>
•
<a href="https://arxiv.org/pdf/2606.03461v1" > 📖 Paper</a>
</p>
<p align="center">
<a href="https://stephen0808.github.io/terminal-lego.github.io/#" > 📖 Project</a>
</p>

## 1. Quick Start

### 1.1 Install dependencies

```bash
pip install -r requirements.txt
```

### 1.2 Lint

```bash
scripts/lint.sh
```

### 1.3 Configure

```bash
cp .env.example .env
# Edit .env with your API keys
```

Required environment variables:
- `OPENAI_API_KEY` — API key for an OpenAI-compatible LLM service
- `OPENAI_API_BASE` — API base URL (default: `https://api.openai.com/v1`)
- `MODEL_NAME` — Model to use (default: `claude-opus-4-6`)

## 2. Generate Tasks From StackOverflow JSONL

For local StackOverflow dumps, use `sources/stackoverflow/prepare_dataset.py` before
task generation. It reads dump-derived JSONL rows, classifies rows into
Terminal-Bench categories from tags, and writes either JSONL or the generator
contract `{"metadata": ..., "questions": [...]}`.

Balanced smoke seed:

```bash
python -m sources.stackoverflow.prepare_dataset \
    --input /data/avzavodov/stackoverflow_posts_xml/sets/so_posts_6y_score_ge_0_our_categories_top100k_tbench_dist_20260702T170248+0300/questions.jsonl \
    --output ./data/so_seed_balanced32.json \
    --format generator-json \
    --per-category 2 \
    --sort-by-score
```

100k set close to the Terminal-Bench 2.0 category distribution:

```bash
python -m sources.stackoverflow.prepare_dataset \
    --input /data/avzavodov/stackoverflow_posts_xml/sets/so_posts_6y_score_ge_0_our_categories_jsonl_20260702T165727+0300/questions.jsonl \
    --output ./data/so_top100k_tbench_dist.jsonl \
    --format jsonl \
    --sample-size 100000 \
    --distribution terminal-bench-2 \
    --min-score 0
```

The tag taxonomy and benchmark distribution are in
`sources/stackoverflow/terminal_bench_tag_taxonomy.json`.

For task generation from StackOverflow JSONL, the convenience wrapper is:

```bash
PER_CATEGORY=2 \
MODEL_NAME=openai/glm-5.2-fp8 \
OPENAI_API_BASE=http://localhost:30002/v1 \
OPENAI_API_KEY=EMPTY \
scripts/generate_tasks.sh \
    /data/avzavodov/stackoverflow_posts_xml/sets/so_posts_6y_score_ge_0_our_categories_top100k_tbench_dist_20260702T170248+0300/questions.jsonl \
    ./data/so_smoke
```

## 3. Validate Tasks

Validation requires Docker.

```bash
python -m validator.validate_tasks \
    --input ./data/candidates_r1 \
    --output ./data/validated_r1 \
    --workers 8 \
    --timeout 300
```

## 4. Generate Agent Solution Rollouts

After validation, run Harbor-supported agents on accepted task directories:

```bash
AGENT=terminus-2 \
MODEL_NAME=openai/glm-5.2-fp8 \
OPENAI_API_BASE=http://localhost:30002/v1 \
OPENAI_API_KEY=EMPTY \
scripts/generate_solutions.sh \
    ./data/validated_r1 \
    ./data/solution_runs
```

`solvers/run_solutions.py` uses Harbor directly, so the same interface can
run other Harbor agents and environments:

```bash
python -m solvers.run_solutions \
    --tasks-dir ./data/validated_r1 \
    --jobs-dir ./data/solution_runs \
    --agent mini-swe-agent \
    --model openai/gpt-4.1 \
    --env docker \
    --n-concurrent 4
```

Outputs are written under `<jobs-dir>/<job-name>/`:

```
config.json
result.json
solution_generation_summary.json
accepted_trajectories.jsonl
failed_trials.jsonl
<trial_name>/agent/trajectory.json
<trial_name>/artifacts/logs/artifacts/solve.sh
<trial_name>/verifier/reward.txt
```

## 5. Output Format

Each generated task has this structure:

```
task_00001/
├── instruction.md          # Task description
├── task.toml               # Metadata (difficulty, category, tags)
├── environment/
│   ├── Dockerfile          # Container setup
│   └── task_file/          # Input files
├── solution/
│   └── solve.sh            # Reference solution
└── tests/
    ├── test.sh             # Test runner
    └── test_outputs.py     # pytest assertions
```

## 6. Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKERS` | 1 | Parallel threads for task generation wrapper |
| `VAL_WORKERS` | 8 | Parallel threads for validation |
| `VAL_TIMEOUT` | 300 | Docker timeout per step (seconds) |
| `MODEL_NAME` | claude-opus-4-6 | LLM model name |

## 7. Requirements

- Python 3.10+
- Docker (for validation step)
- An OpenAI-compatible API endpoint

## 8. License

MIT
