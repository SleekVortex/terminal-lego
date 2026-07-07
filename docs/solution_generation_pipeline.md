# Solution Generation Pipeline

Документ описывает текущую реализацию генерации agent rollouts / решений для уже готовых Terminal-Lego задач.

## Основные Файлы

- `scripts/generate_solutions.sh` - convenience wrapper для запуска solution generation.
- `rollouts/solution_generator.py` - Python CLI, который строит Harbor `JobConfig`, запускает job и собирает summary.
- `prompts/solution_generator/materialize_solution.md` - extra instruction для агента: материализовать итоговое решение в `/logs/artifacts/solve.sh`.
- `configs/harbor/docker-compose-bridge-network.yaml` - опциональный Harbor compose override для `network_mode: bridge`.
- `configs/opencode/` - preinstalled OpenCode runtime config для запуска OpenCode без скачивания nvm/npm внутри каждого trial.

## Общая Схема

```text
validated Terminal-Lego tasks
  -> scripts/generate_solutions.sh
  -> rollouts/solution_generator.py
  -> Harbor Job
  -> agent runs task in Docker environment
  -> verifier runs tests
  -> result.json + trajectory + materialized solve.sh
  -> solution_generation_summary.json
  -> accepted_trajectories.jsonl / failed_trials.jsonl
```

Task generation и solution generation разделены. На вход solution generation подается уже готовая папка задач:

```text
validated/
  task_00001/
    instruction.md
    task.toml
    environment/Dockerfile
    environment/task_file/...
    tests/test.sh
    tests/test_outputs.py
```

`solution/solve.sh` в задаче может существовать как reference solution, но agent rollout не должен читать reference solution. Для agent solution используется отдельный materialized artifact:

```text
<trial>/artifacts/logs/artifacts/solve.sh
```

## Запуск Через Wrapper

Базовый запуск:

```bash
AGENT=terminus-2 \
MODEL_NAME=openai/glm-5.2-fp8 \
OPENAI_API_BASE=http://localhost:30002/v1 \
OPENAI_API_KEY=EMPTY \
scripts/generate_solutions.sh ./validated ./runs
```

Wrapper принимает:

```bash
scripts/generate_solutions.sh TASKS_DIR [JOBS_DIR] [extra solution_generator args...]
```

Значения по умолчанию:

- `AGENT=terminus-2`;
- `HARBOR_ENV=docker`;
- `PYTHON_BIN=python3`;
- `OPENAI_API_KEY=EMPTY`;
- `JOB_NAME=solution-rollouts-<utc timestamp>`;
- `N_ATTEMPTS=1`;
- `N_CONCURRENT=1`;
- `MAX_RETRIES=0`;
- `DOCKER_NETWORK_STRATEGY=bridge`;
- `CLEANUP_DOCKER=1`;
- `MATERIALIZE_INSTRUCTION=prompts/solution_generator/materialize_solution.md`.
- `OPENCODE_RUNTIME_DIR=configs/opencode/runtime` при `AGENT=preinstalled-opencode`.

Wrapper собирает аргументы и вызывает:

```bash
python rollouts/solution_generator.py ...
```

Все extra CLI args после `TASKS_DIR [JOBS_DIR]` пробрасываются в `solution_generator.py`.

## Preinstalled OpenCode

Обычный Harbor agent `opencode` устанавливает nvm, Node и `opencode-ai` внутри каждого task container. Для массовых rollouts это слишком медленно и зависит от GitHub/npm во время каждого trial.

Для этого добавлен agent mode:

```bash
AGENT=preinstalled-opencode
```

Он использует Harbor `OpenCode.run()` и стандартный парсер trajectory, но заменяет сетевой install на проверку уже смонтированного runtime:

```text
/opt/terminal-lego/opencode/bin/opencode
```

Runtime собирается отдельно через `configs/opencode/Dockerfile`, переносится на `cpu`, затем извлекается в:

```text
configs/opencode/runtime/
```

Wrapper при `AGENT=preinstalled-opencode` автоматически добавляет:

```text
--extra-docker-compose configs/opencode/docker-compose-runtime.yaml
```

и проверяет, что существует executable:

```text
${OPENCODE_RUNTIME_DIR}/bin/opencode
```

Пример запуска:

```bash
AGENT=preinstalled-opencode \
MODEL_NAME=openai/glm-5.2-fp8 \
OPENAI_API_KEY=EMPTY \
DOCKER_NETWORK_STRATEGY=bridge \
scripts/generate_solutions.sh ./validated ./runs
```

Для Docker bridge нужен отдельный tunnel, который слушает Docker gateway, например:

```bash
scripts/start_opencode_docker_tunnel.sh
```

Этот tunnel не заменяет и не останавливает существующий `127.0.0.1:30002`; он добавляет отдельный listener `172.16.0.1:30003` для контейнеров.

Для GLM 5.2 `PreinstalledOpenCode` регистрирует в OpenCode custom provider:

- provider id: `glm`;
- provider package: `@ai-sdk/openai-compatible`;
- default baseURL для Docker bridge: `http://host.docker.internal:30003/v1`;
- model: `glm/glm-5.2-fp8`;
- `interleaved.field = reasoning_content`, чтобы OpenCode читал reasoning из streaming chunk-ов GLM.
- `agent.build.prompt` с Terminal-Lego system prompt для solution generation.

Внешний `MODEL_NAME=openai/glm-5.2-fp8` остается совместимым алиасом для остальных wrapper-ов, но перед запуском `opencode` агент меняет его на `glm/glm-5.2-fp8`. Это заставляет OpenCode использовать `/v1/chat/completions`, а не OpenAI Responses API.

Если нужен другой endpoint, можно явно передать:

```text
OPENAI_API_BASE=http://host.docker.internal:30003/v1
```

Trajectory для `preinstalled-opencode` дополняется первым step-ом:

```json
{
  "source": "system",
  "message": "..."
}
```

Этот system step содержит тот же текст, который записывается в OpenCode config как `agent.build.prompt`. После него идет `source=user` с task instruction, затем `source=agent` шаги из OpenCode stream. Это нужно, чтобы rollout был самодостаточным для просмотра и последующего train-data conversion.

## Materialize Solution Prompt

`prompts/solution_generator/materialize_solution.md` добавляется в Harbor job как `extra_instruction_path`.

Его задача - заставить агента в конце работы записать executable, non-interactive, idempotent Bash script:

```text
/logs/artifacts/solve.sh
```

Контракт prompt-а:

- первая строка должна быть `#!/usr/bin/env bash`;
- script должен воспроизводить успешные изменения на свежей копии task image;
- script может зависеть только от файлов в fresh image и разрешенного network access;
- script не должен читать `/tests`, `/solution`, `/logs/verifier`, reward files или hidden reference solution;
- script должен завершаться non-zero on failure;
- после записи надо выполнить `chmod +x /logs/artifacts/solve.sh`.

Именно наличие этого файла используется при классификации accepted rollouts.

## Harbor Integration

`rollouts/solution_generator.py` импортирует Harbor runtime:

```python
from harbor.job import Job
from harbor.models.job.config import DatasetConfig, JobConfig, RetryConfig
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig, VerifierConfig
```

Если Harbor не установлен, CLI завершится с ошибкой:

```text
Harbor is required for solution generation. Install dependencies with `pip install -r requirements.txt`.
```

Текущий `requirements.txt` содержит:

```text
requests>=2.28.0
harbor>=0.15.0
pytest>=8.0.0
```

## Построение JobConfig

Главная функция: `build_job_config(args)`.

### Task / Dataset Selection

`--tasks-dir` может быть:

- single task directory, если внутри есть `task.toml`;
- dataset directory, если внутри лежат task directories.

Если это single task, создается:

```python
TaskConfig(path=tasks_dir)
```

Если это dataset, создается:

```python
DatasetConfig(
    path=tasks_dir,
    task_names=args.include_task_name or None,
    exclude_task_names=args.exclude_task_name or None,
    n_tasks=args.n_tasks,
)
```

Поддерживаются:

- `--include-task-name`;
- `--exclude-task-name`;
- `--n-tasks`.

### AgentConfig

Agent задается через:

- `--agent`;
- `--model`;
- `--api-base`;
- `--api-key`;
- `--agent-kwarg KEY=VALUE`;
- `--agent-env KEY=VALUE`;
- `--agent-include-logs`;
- `--agent-exclude-logs`.

`parse_key_value()` парсит `KEY=VALUE`, а значения приводит к типам:

- `true` / `false` -> bool;
- `none` / `null` -> None;
- JSON object/list -> `json.loads`;
- int / float;
- иначе string.

Для `terminus` и `terminus-2` выставляются defaults:

```python
parser_name = "json"
enable_summarize = True
record_terminal_session = True
```

Если указаны trajectory flags:

- `--trajectory-raw-content`;
- `--trajectory-linear-history`;
- `--trajectory-config-json`;

они собираются в `agent.kwargs["trajectory_config"]`.

Wrapper всегда добавляет:

```text
--trajectory-raw-content
--trajectory-linear-history
```

Для `preinstalled-opencode` вместо `AgentConfig.name` используется Harbor `AgentConfig.import_path`:

```text
rollouts.agents.preinstalled_opencode:PreinstalledOpenCode
```

Это позволяет не patch-ить Harbor package и при этом переиспользовать его OpenCode run/trajectory implementation.

### EnvironmentConfig

Environment задается через:

- `--env`, default `docker`;
- `--force-build`;
- `--delete / --no-delete`, default `delete=True`;
- `--extra-docker-compose`;
- `--environment-kwarg KEY=VALUE`;
- `--environment-env KEY=VALUE`.

Wrapper при `DOCKER_NETWORK_STRATEGY=bridge` добавляет:

```text
--extra-docker-compose configs/harbor/docker-compose-bridge-network.yaml
```

Этот compose override задает:

```yaml
services:
  main:
    network_mode: bridge
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

Цель - не плодить отдельную Docker compose network на каждый worker/trial и иметь доступ к host endpoint через `host.docker.internal`.

Если `DOCKER_NETWORK_STRATEGY=compose`, override не добавляется, Harbor использует default compose behavior.

### VerifierConfig

Verifier задается через:

- `--disable-verification`;
- `--verifier-env KEY=VALUE`;
- `--verifier-include-logs`;
- `--verifier-exclude-logs`.

Verifier запускается Harbor-ом после agent run и читает rewards из task tests.

### Retry / Attempts / Timeouts

JobConfig получает:

- `n_attempts`;
- `n_concurrent_trials`;
- `RetryConfig(max_retries=args.max_retries)`;
- `timeout_multiplier`;
- `agent_timeout_multiplier`;
- `verifier_timeout_multiplier`;
- `environment_build_timeout_multiplier`.

Wrapper defaults:

```text
N_ATTEMPTS=1
N_CONCURRENT=1
MAX_RETRIES=0
```

## Dry Run

`--dry-run-config PATH` не запускает Harbor job. Вместо этого пишет serialized Harbor config:

```bash
python rollouts/solution_generator.py \
  --tasks-dir ./validated \
  --dry-run-config ./job_config.json
```

Это полезно для проверки, какие agents, env kwargs, datasets, tasks и extra instruction paths реально попадут в Harbor.

## Выполнение Job

Основной runtime:

```python
async def run_job(config):
    harbor = import_harbor()
    Job = harbor["Job"]
    job = await Job.create(config)
    return await job.run()
```

После `asyncio.run(run_job(config))` код считает, что Harbor записал результаты в:

```text
<jobs_dir>/<job_name>/
```

Дальше запускается post-processing:

```python
summary = summarize_job(job_dir, args.reward_threshold)
```

## Docker Cleanup В Wrapper

`scripts/generate_solutions.sh` ставит `trap cleanup EXIT`.

Если `CLEANUP_DOCKER=1`, wrapper:

1. ищет trial directories внутри `<JOBS_DIR>/<JOB_NAME>`;
2. из имен trial directories строит Docker compose project names;
3. удаляет containers с matching label `com.docker.compose.project`;
4. удаляет matching Docker networks.

Это best-effort cleanup: ошибки удаления игнорируются, чтобы wrapper не скрывал основной exit status.

## Classification И Acceptance Logic

`summarize_job(job_dir, reward_threshold)` проходит по всем:

```text
<job_dir>/*/result.json
```

Для каждого trial извлекаются:

- `task_name`;
- `trial_name`;
- `reward`;
- `exception_info`;
- `exception_type`;
- путь к `agent/trajectory.json`, если есть;
- путь к `artifacts/logs/artifacts/solve.sh`, если есть.

Reward читается из:

```python
result["verifier_result"]["rewards"]["reward"]
```

`reward_from_result()` приводит reward к finite float. `nan`, отсутствующее значение или непарсируемая строка дают `None`.

Trial считается accepted, если:

```text
reward >= reward_threshold
and artifacts/logs/artifacts/solve.sh exists
```

Default threshold:

```text
--reward-threshold 1.0
```

Отдельный случай:

```text
reward >= threshold
solve.sh exists
exception_type == AgentTimeoutError
```

Такой trial все равно accepted, но получает:

```json
{
  "accepted_with_timeout": true,
  "original_exception_type": "AgentTimeoutError"
}
```

Если reward прошел threshold, но materialized `solve.sh` отсутствует, trial rejected:

```json
{
  "rejection_reason": "reward_threshold_met_missing_solve_sh"
}
```

Если reward ниже threshold или reward отсутствует, trial rejected без специальной причины.

## Output Artifacts

После job пишутся:

```text
<jobs_dir>/<job_name>/solution_generation_summary.json
<jobs_dir>/<job_name>/accepted_trajectories.jsonl
<jobs_dir>/<job_name>/failed_trials.jsonl
```

`solution_generation_summary.json` содержит:

- job dir;
- total trials;
- accepted trials;
- failed trials;
- reward threshold;
- полный список trial rows;
- пути к summary/accepted/failed файлам.

`accepted_trajectories.jsonl` содержит только accepted rows. Это основной файл для дальнейшего использования agent rollouts.

Каждая строка включает:

```json
{
  "task_name": "...",
  "trial_name": "...",
  "reward": 1.0,
  "accepted": true,
  "accepted_with_timeout": false,
  "original_exception_type": null,
  "rejection_reason": null,
  "exception_type": null,
  "exception_info": null,
  "result_path": ".../result.json",
  "trajectory_path": ".../agent/trajectory.json",
  "materialized_solution_path": ".../artifacts/logs/artifacts/solve.sh"
}
```

`failed_trials.jsonl` содержит rejected rows с тем же schema.

## Отличие Reference Solution И Agent Solution

В task directory есть:

```text
solution/solve.sh
```

Это reference solution, generated на стадии task generation. Оно используется для Docker validation candidate tasks.

Во время solution generation агент должен решить задачу сам. Для сохранения результата используется:

```text
<trial>/artifacts/logs/artifacts/solve.sh
```

Этот файл создается агентом благодаря `prompts/solution_generator/materialize_solution.md`. Именно этот файл считается materialized agent solution.

## Важные Ограничения Текущей Реализации

- `solution_generator.py` не реализует harness logic сам; он делегирует запуск Harbor.
- Качество и формат trajectory зависят от выбранного Harbor agent.
- Multi-harness поддержка ограничена тем, какие agents/environments реально поддерживает установленный Harbor.
- Accepted classification требует не только reward, но и materialized `/logs/artifacts/solve.sh`.
- `accepted_with_timeout` означает, что verifier reward и solve.sh есть, но Harbor result сохранил `AgentTimeoutError`.
- Docker cleanup работает по compose project labels и является best-effort.
