# Multi-Agent Task Generation Implementation

Документ описывает текущую реализацию генерации Terminal-Lego задач на мультиагентном пайплайне. Монолитный `generator/task_generator.py` заменен на маленькие модули: CLI entrypoint, LLM client, artifact agents, reviewers, writer, Docker validator и controlled repair loop.

## Цели

Pipeline должен:

- генерировать standalone Terminal-Bench-compatible задачи из локального StackOverflow-derived JSONL;
- явно разделять генерацию спецификации, окружения, решения, тестов и Dockerfile;
- не принимать задачу без `reward == 1.0` на golden solution;
- сохранять подробные логи build/solve/test для failed задач;
- автоматически чинить типовые ошибки через controlled repair loop;
- не ослаблять verifier tests только потому, что reference solution не прошел;
- легко читаться: маленькие модули, явные контракты данных, минимум скрытой магии.

## Базовые Принципы

1. **Один агент - один артефакт.**
   Модель не пишет все файлы одним ответом. `InstructionAgent`, `EnvironmentAgent`, `SolutionAgent`, `TestAgent` и `DockerfileAgent` отвечают за отдельные артефакты, а `TaskGenerator` только координирует порядок.

2. **Tests проверяются статически и Docker-валидацией.**
   `TestAgent` получает reference solution как контекст ожидаемых postconditions, но static review запрещает тестам запускать `solve.sh` и импортировать implementation modules. Итоговый oracle - Docker validation, где reference solution должен пройти независимый verifier.

3. **Verifier runner детерминированный.**
   `tests/test.sh` остается статическим offline runner: Python 3.12+, `pytest`, без `curl`, `uv`, `uvx`, `pip`, GitHub/PyPI во время проверки.

4. **Docker validation - главный oracle.**
   Static review только отсеивает очевидный мусор. Accepted task - это только задача, где golden solution проходит verifier и пишет `reward.txt = 1`.

5. **Repair loop controlled, не free-for-all.**
   Агенту нельзя сразу менять все файлы. Failure classifier определяет, какие файлы разрешено чинить в конкретной попытке.

6. **Минимум новых зависимостей.**
   На MVP используем текущий стек repo: stdlib, `requests`, `pytest`, Docker CLI, текущий OpenAI-compatible endpoint. Новые библиотеки добавляются только если они реально уменьшают сложность.

## Используемые Библиотеки И Инструменты

Текущий `requirements.txt`:

```text
requests>=2.28.0
harbor>=0.15.0
pytest>=8.0.0
```

MVP использует:

- `dataclasses` - схемы внутренних контрактов без внешних зависимостей;
- `typing` / `TypedDict` / `Literal` - читаемые типы;
- `pathlib` - работа с task directories;
- `json` - persistence контрактов и отчетов;
- `argparse` - CLI scripts;
- `logging` - структурированные stage logs;
- `concurrent.futures.ThreadPoolExecutor` - parallel LLM calls и Docker subprocess orchestration;
- `subprocess` - Docker build/run/cp/exec/rm/rmi;
- `shutil` - копирование accepted/repaired task directories;
- `ast` - static review `tests/test_outputs.py`;
- `re` - lightweight parsing Dockerfile/COPY и log classification;
- `requests` - существующий OpenAI-compatible LLM client;
- `pytest` - unit tests и verifier runtime;
- `docker` CLI - без Python Docker SDK на MVP.

Почему не добавлять сразу Pydantic/Celery/Docker SDK:

- `dataclasses + explicit validate_*()` достаточно для внутренних JSON contracts;
- pipeline CPU/I/O-bound, а не требует distributed queue на старте;
- Docker CLI уже используется в `validator/validate_tasks.py`, проще не менять execution model;
- меньше зависимостей - меньше проблем внутри generated Docker images и на `cpu`.

Если позже понадобится строгая runtime schema validation, можно добавить `pydantic`, но только после стабилизации JSON contracts.

## Высокоуровневая Схема

```text
StackOverflow JSONL / seed.json
  -> SeedLoader
  -> InstructionAgent
  -> EnvironmentAgent
  -> SolutionAgent
  -> SolutionReviewer
  -> TestAgent
  -> TestReviewer
  -> DockerfileAgent
  -> DockerfileReviewer
  -> TaskWriter
  -> DockerValidator
  -> accepted
       or
     FailureClassifier
       -> RepairLoop
       -> accepted / rejected
```

## Контракты Данных

Контракты должны жить отдельно от orchestration-кода.

Файл:

```text
generator/contracts.py
```

Основные dataclasses:

```python
@dataclass
class TaskSpec:
    task_id: str
    question_id: int
    title: str
    category: str
    categories: list[str]
    tags: list[str]
    instruction: str
    expected_effects: list[str]
    required_runtimes: list[str]
    input_artifacts: list[str]
    output_artifacts: list[str]
    constraints: list[str]
    forbidden: list[str]


@dataclass
class EnvironmentSpec:
    files: dict[str, str]
    directories: list[str]


@dataclass
class GeneratedTask:
    task_name: str
    task_spec: TaskSpec
    environment: EnvironmentSpec
    solution_sh: str
    test_outputs_py: str
    dockerfile: str
    difficulty: str


@dataclass
class ValidationResult:
    task_name: str
    status: str
    reward: float | None
    build_time: float
    solve_time: float
    test_time: float
    error: str | None
    log_dir: str


@dataclass
class FailureDiagnosis:
    task_name: str
    failure_class: str
    confidence: float
    evidence: list[str]
    allowed_repairs: list[str]


@dataclass
class RepairAttempt:
    task_name: str
    attempt: int
    failure_class: str
    changed_files: list[str]
    validation_result: ValidationResult
```

Каждый dataclass получает:

- `to_dict()`;
- `from_dict()`;
- `validate()` с понятными ошибками.

Для MVP не нужен сложный schema framework: ошибки контракта должны быть читаемыми и проверяться unit tests.

## Разделение Кода

Предлагаемая структура:

```text
generator/
  contracts.py
  llm_client.py
  orchestrator.py
  task_writer.py
  agents/
    __init__.py
    base.py
    task_generation.py
  reviewers/
    __init__.py
    solution_review.py
    test_review.py
    dockerfile_review.py
  failure/
    __init__.py
    classifier.py
  repair/
    __init__.py
    loop.py
    agent.py

rollouts/
  solution_generator.py
  agents/
    __init__.py
    preinstalled_opencode.py

validator/
  validate_tasks.py
  docker_runner.py
  validation_logs.py

scripts/
  generate_tasks.sh
  diagnose_failed_tasks.py
  repair_failed_tasks.py
  regenerate_dockerfiles.py
  validation_first_pipeline.py

prompts/
  task_generator/
    ...
  repair/
    task_repair.md
```

### `generator/llm_client.py`

Выносит текущую `call_llm_api()` из `task_generator.py`.

Ответственность:

- OpenAI-compatible `POST /chat/completions`;
- retries;
- timeout;
- token usage tracking;
- optional `check_truncation`;
- единая запись telemetry по стадиям.

Существующий код можно перенести почти напрямую, чтобы не плодить второй client.

### `generator/agents/base.py`

Базовый класс для LLM agents:

```python
class LLMAgent:
    name: str
    prompt_template: str

    def build_prompt(self, context: dict) -> str: ...
    def parse_response(self, response: str): ...
    def run(self, context: dict): ...
```

Здесь не должно быть business logic reviewer-ов. Agent только строит prompt, вызывает LLM, парсит output.

### `generator/reviewers/*`

Reviewer-ы - deterministic code, без LLM.

Примеры:

- `test_review.py`:
  - `ast.parse`;
  - no non-stdlib imports;
  - no `solve.sh` execution;
  - no implementation imports.

- `dockerfile_review.py`:
  - есть `FROM`;
  - нет пустого `apt-get install`;
  - нет `uv/uvx`;
  - COPY sources существуют;
  - все environment artifacts copied;
  - есть Python 3.12+ и pytest для verifier.

- `solution_review.py`:
  - shell syntax check через `bash -n`;
  - no interactive commands;
  - no forbidden network calls;
  - common path sanity.

### `generator/orchestrator.py`

Главный state machine одного task:

```text
generate_task(seed):
  instruction = InstructionAgent.run(seed)

  env = EnvironmentAgent.run(seed, instruction)

  solution = SolutionAgent.run(seed, instruction, env)
  review(solution)

  tests = TestAgent.run(seed, instruction, env, solution)
  review(tests)

  dockerfile = DockerfileAgent.run(seed, instruction, env, solution, tests)
  review(dockerfile)

  TaskWriter.write(...)
```

Orchestrator не содержит prompt text и не содержит Docker subprocess logic.

### `validator/docker_runner.py`

Нужно вынести Docker operation из `validate_tasks.py`:

```python
class DockerRunner:
    def build(self, task_dir: Path, image_tag: str) -> CommandResult: ...
    def start(self, image_tag: str, container_name: str) -> CommandResult: ...
    def copy_solution(self, ...): ...
    def run_solution(self, ...): ...
    def copy_tests(self, ...): ...
    def run_tests(self, ...): ...
    def read_reward(self, ...): ...
    def cleanup(self, ...): ...
```

Это сделает validator тестируемым без реального Docker через mocked runner.

### `validator/validation_logs.py`

Обязательная доработка для repair loop.

Для каждой задачи сохранять:

```text
validation_logs/task_XXXXX/
  build.stdout
  build.stderr
  run.stdout
  run.stderr
  solve.stdout
  solve.stderr
  test.stdout
  test.stderr
  reward.stdout
  result.json
```

`validation_report.json` должен ссылаться на `log_dir`.

## Агенты

### 1. `InstructionAgent`

Вход:

- StackOverflow title/body/tags;

Выход:

- `instruction.md`.

Назначение:

- сформулировать standalone terminal task;
- убрать зависимость от StackOverflow context;
- не писать environment, solution, tests или Dockerfile.

### 2. `EnvironmentAgent`

Вход:

- StackOverflow title/tags;
- generated instruction.

Выход:

- `EnvironmentSpec`.

Ограничения:

- все paths relative to `environment/`;
- основная директория ввода - `task_file`;
- нет абсолютных путей;
- нет сетевых ссылок как обязательного runtime input;
- маленькие deterministic files.

### 3. `SolutionAgent`

Вход:

- generated instruction;
- `EnvironmentSpec` summary;
- accepted answer / source hints.

Выход:

- `solution/solve.sh`.

Ограничения:

- non-interactive bash;
- работает из `/app`;
- создает только expected outputs;
- не меняет tests;
- не требует сети, если это не часть задачи.

### 4. `TestAgent`

Вход:

- generated instruction;
- `EnvironmentSpec` summary;
- reference solution as context for expected postconditions.

Выход:

- `tests/test_outputs.py`.

Ограничения:

- only Python stdlib + `pytest`;
- black-box postcondition checks;
- no calls to `solve.sh`;
- no imports from generated implementation;
- no random/time-sensitive checks.

### 5. `DockerfileAgent`

Вход:

- generated instruction;
- environment artifact list;
- generated solution;
- generated tests.

Выход:

- `environment/Dockerfile`.

Ограничения:

- runtime image соответствует задаче;
- verifier Python 3.12+ добавляется deterministic helper-ом;
- `COPY` покрывает все generated artifacts;
- no verifier-time dependency download.

### 6. `FailureClassifier`

Вход:

- `ValidationResult`;
- `validation_logs/task_XXXXX/*`.

Выход:

- `FailureDiagnosis`.

Классы:

| class | evidence | allowed repairs |
|---|---|---|
| `build_failed` | docker build stderr | Dockerfile |
| `runtime_missing` | `command not found`, missing `/usr/share/dotnet`, missing `node`, etc. | Dockerfile |
| `solution_error` | solve returncode != 0 | solution |
| `path_mismatch` | `No such file or directory`, expected path absent | solution, environment |
| `test_assertion_or_error` | pytest failed after solve success | solution first, tests only if invalid |
| `test_invalid` | test imports invalid libs, executes solution, syntax error | tests |
| `timeout` | build/solve/test timeout | Dockerfile or solution depending stage |

### 7. `RepairAgent`

Один общий coordinator и специализированные prompts:

- `repair_solution.md`;
- `repair_dockerfile.md`;
- `repair_environment.md`;
- `repair_tests.md`.

На MVP лучше разрешить:

- Dockerfile repair;
- solution repair;
- environment path repair.

Tests repair включать только если classifier уверен в `test_invalid`.

## Repair Loop

Algorithm:

```text
for failed_task in failed_tasks:
  diagnosis = FailureClassifier(...)
  current_task_dir = failed_task

  for attempt in 1..MAX_REPAIR_ATTEMPTS:
    allowed_files = policy(diagnosis)
    repaired = RepairAgent.run(current_task_dir, diagnosis, allowed_files)
    StaticReview(repaired changed files)
    result = DockerValidator.validate_one(repaired)

    if result.reward == 1.0:
      copy to repaired_accepted/
      stop

    diagnosis = FailureClassifier(result.logs)

  copy to rejected/ with final diagnosis
```

Default `MAX_REPAIR_ATTEMPTS = 2` для MVP. Третью попытку можно включить позже, если статистика покажет окупаемость.

## Persistence Layout

Каждый run должен быть самодостаточным:

```text
terminal-lego-work/task-generation/<run_id>/
  seed.json
  candidates/
  generation/
    task_specs/
    agent_outputs/
    token_usage.json
    generation_summary.json
  validation/
    baseline/
      task_XXXXX/
      validation_logs/
      validation_report.json
      run.log
    failed_tasks.jsonl
    repair/
      attempts/
        task_XXXXX/attempt_01/
        task_XXXXX/attempt_02/
      repaired_validated/
      repair_report.json
    final_accepted/
      task_XXXXX/
      final_accepted_report.json
```

Нельзя перетирать исходные candidates во время repair. Repair работает в sibling output, чтобы можно было сравнить исходную и repaired версию.

## CLI

### Generate

```bash
python scripts/generate_tasks_multi_agent.py \
  --seed seed.json \
  --output terminal-lego-work/task-generation/<run_id>/candidates \
  --workers 24 \
  --api-base http://127.0.0.1:30002/v1 \
  --model glm-5.2-fp8 \
  --resume
```

Resume semantics:

- source id сравнивается с уже записанными task metadata;
- существующие task dirs не перезатираются;
- incomplete task dirs можно помечать как `partial` и регенерировать только если явно указан `--repair-partials`.

### Validate

```bash
python validator/validate_tasks.py \
  --input candidates \
  --output validation/baseline \
  --workers 64 \
  --timeout 300 \
  --save-logs
```

`--save-logs` должен стать default, потому что repair loop без логов бесполезен.

### Collect Failures

```bash
python scripts/validation_first_pipeline.py collect-failures \
  --validation-report validation/baseline/validation_report.json \
  --tasks-dir candidates \
  --output validation/failed_tasks.jsonl
```

### Repair

```bash
python scripts/repair_failed_tasks.py \
  --tasks-dir candidates \
  --failed-tasks validation/failed_tasks.jsonl \
  --validation-log-dir validation/baseline/validation_logs \
  --output validation/repair \
  --workers 16 \
  --max-attempts 2
```

### Merge Accepted

```bash
python scripts/validation_first_pipeline.py merge-accepted \
  --baseline-validated-dir validation/baseline \
  --regenerated-validated-dir validation/repair/repaired_validated \
  --output-dir validation/final_accepted
```

## Parallelism

Использовать `ThreadPoolExecutor`, а не `ProcessPoolExecutor`, потому что:

- LLM calls - network I/O;
- Docker operations - subprocess I/O;
- CPU-heavy Python code внутри orchestrator почти отсутствует;
- thread workers проще останавливать и логировать.

Раздельные worker pools:

- task generation: 16-32 workers, зависит от LLM endpoint;
- validation: 64 workers, если Docker daemon и диск выдерживают;
- repair: 8-16 workers, потому что каждый repair делает LLM call + validation.

Нужно иметь backpressure:

- max active Docker containers;
- max build concurrency;
- optional semaphore на `docker build`, отдельно от `docker run`.

Практический вариант:

```python
build_semaphore = threading.Semaphore(max_builds)
container_semaphore = threading.Semaphore(max_containers)
```

Это лучше, чем тупо увеличивать общий worker count, если Docker build начинает душить диск/сеть.

## Acceptance Policy

Task accepted only if:

- all required files exist;
- Docker build succeeds;
- container starts;
- `solution/solve.sh` exits successfully;
- `tests/test.sh` exits successfully;
- `/logs/verifier/reward.txt == 1`;
- task directory copied to `validated` or `final_accepted`.

Task rejected if:

- no valid Dockerfile after repair attempts;
- solution cannot be repaired within attempts;
- tests are invalid and cannot be repaired without weakening task semantics;
- task requires external service/network that is not part of deterministic environment;
- task is too broad or repo-scale unsupported by current generator.

## Test Strategy

Unit tests:

- contracts serialization/validation;
- prompt parsing per agent;
- environment reviewer;
- solution reviewer;
- test reviewer;
- dockerfile reviewer;
- failure classifier;
- repair policy;
- resume semantics;
- merge accepted semantics.

Integration tests without real LLM:

- fake LLM responses for full happy path;
- fake invalid Dockerfile -> retry -> valid Dockerfile;
- fake invalid test import -> retry;
- fake solution path mismatch -> repair solution -> accepted.

Docker smoke tests:

- one small bash task accepted;
- one task with runtime missing classified as `runtime_missing`;
- one task with failed pytest classified as `test_assertion_or_error`;
- one task with invalid Docker tag naming to prevent regression like `task_00000`.

## Implementation Phases

### Phase 1. Make Validator Repair-Ready

- Extend `validator/validate_tasks.py` to save per-task logs.
- Move Docker command execution into `validator/docker_runner.py`.
- Add `ValidationResult.log_dir`.
- Add tests for log persistence.

This is the highest-priority phase because current failed tasks only say `reward=0`.

### Phase 2. Extract Reviewers

- Move `_review_tests`, `_review_dockerfile`, `_ensure_verifier_deps` from `task_generator.py` into reviewer modules.
- Remove old implementation hooks from `task_generator.py`; tests should import reviewer modules directly.
- Add focused tests per reviewer.

### Phase 3. Add Contracts And Artifact Agents

- Add `generator/contracts.py`.
- Add `generator/agents/task_generation.py`.
- Move orchestration into `generator/orchestrator.py`.
- Keep only the generated Terminal-Bench task directory format unchanged.

### Phase 4. Multi-Agent Orchestrator

- Add `generator/orchestrator.py`.
- Split environment/solution/tests/dockerfile agents.
- Keep task prompts under `prompts/task_generator/` and repair prompt under `prompts/repair/`.
- Add fake-LLM integration tests.

### Phase 5. Repair Loop

- Add `FailureClassifier`.
- Add `RepairAgent`.
- Add `scripts/repair_failed_tasks.py`.
- Support changed-file allowlist.
- Validate repaired task in isolated sibling dir.

### Phase 6. Metrics And Reports

- Add per-stage pass/fail stats.
- Add failure class distribution.
- Add repair success rate by class.
- Add token/cost per accepted task.
- Add category-level acceptance rates.

## Что Не Делать В MVP

- Не давать агенту свободно менять все файлы задачи.
- Не чинить тесты при обычном `solution_error`.
- Не запускать dependency downloads внутри `tests/test.sh`.
- Не перезаписывать исходные candidates во время repair.
- Не добавлять distributed queue до появления реальной необходимости.
- Не добавлять много внешних библиотек ради схем, если dataclasses достаточно.
- Не считать static review accepted signal: accepted только через Docker validation.

## Итоговая Картина

Система должна выглядеть как controlled production pipeline:

```text
generate structured task
  -> write candidate
  -> validate with Docker
  -> diagnose failures with logs
  -> repair allowed files only
  -> validate again
  -> merge accepted
```

Такой подход сохраняет сильную сторону текущего генератора - массовую генерацию маленьких terminal tasks - но добавляет недостающий слой качества: detailed validation logs, typed contracts, deterministic reviewers and targeted repair agents.
