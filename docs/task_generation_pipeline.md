# Task Generation Pipeline

Документ описывает текущую реализацию генерации Terminal-Lego задач из StackOverflow-derived датасета.

## Основные Файлы

- `scripts/generate_tasks.sh` - convenience wrapper для подготовки seed и запуска генератора.
- `sources/stackoverflow/prepare_dataset.py` - нормализация JSONL, фильтрация, сортировка и выборка seed.
- `generator/task_generator.py` - CLI entrypoint генерации задач.
- `generator/task_builder.py` - координатор одной задачи без prompt text и Docker subprocess logic.
- `generator/agents/task_generation.py` - LLM agents для `instruction`, `environment`, `solution`, `tests`, `Dockerfile`.
- `generator/repair/agent.py` и `generator/repair/loop.py` - controlled repair loop для failed задач.
- `prompts/task_generator/*.md` - prompt templates для стадий генерации.
- `prompts/repair/task_repair.md` - prompt template для targeted repair.
- `validator/validate_tasks.py` - отдельная Docker round-trip валидация generated candidates.
- `validator/docker_runner.py` и `validator/validation_logs.py` - Docker execution и per-task logs.

## Общая Схема

```text
StackOverflow JSONL
  -> sources/stackoverflow/prepare_dataset.py
  -> seed.json, формат {"metadata": ..., "questions": [...]}
  -> generator/task_generator.py
  -> candidates/task_XXXXX/
  -> validator/validate_tasks.py, опционально
  -> validated/task_XXXXX/
```

`scripts/generate_tasks.sh` автоматизирует первые две стадии:

```bash
scripts/generate_tasks.sh INPUT_JSONL OUTPUT_DIR
```

Wrapper пишет:

- `OUTPUT_DIR/seed.json`, если не переопределен `SEED_JSON`;
- `OUTPUT_DIR/candidates`, если не переопределен `CANDIDATES_DIR`.

## Подготовка Seed

`sources/stackoverflow/prepare_dataset.py` принимает JSONL, где каждая строка - StackOverflow question row с `accepted_answer`.

Поддерживаемые входные поля нормализуются в контракт генератора:

```json
{
  "question_id": 123,
  "title": "...",
  "body": "...",
  "tags": ["bash", "json"],
  "score": 42,
  "view_count": 1000,
  "answer_count": 3,
  "accepted_answer_id": 456,
  "accepted_answer": {
    "answer_id": 456,
    "body": "...",
    "score": 50
  },
  "link": "https://stackoverflow.com/questions/123",
  "categories": ["system-administration", "data-processing"],
  "selected_category": "system-administration"
}
```

Если `categories` отсутствуют, они восстанавливаются через `sources/stackoverflow/tag_taxonomy.py` по тегам. `selected_category` берется из входа, иначе из первой найденной категории.

Фильтры и режимы выборки:

- `--min-score` / `MIN_SCORE` - отбрасывает вопросы ниже score.
- `--category` / `CATEGORY` - оставляет строки, пересекающиеся с указанными категориями.
- `--allow-uncategorized` - разрешает строки без категории.
- `--sort-by-score` / `SORT_BY_SCORE=1` - выбирает лучшие строки по composite score.
- `--limit` / `LIMIT` - ограничивает количество строк.
- `--start` / `START` - offset после фильтрации для first/top mode.
- `--per-category` / `PER_CATEGORY` - top N на категорию.
- `--sample-size` / `SAMPLE_SIZE` + `--distribution terminal-bench-2` - выборка близко к Terminal-Bench 2.0 distribution.

Composite score сортировки:

```text
question score,
accepted answer score,
view count,
answer count,
question_id as stable tie-breaker
```

Выход для генератора - JSON object:

```json
{
  "metadata": {
    "source": "...",
    "format": "generator-json",
    "total": 1000,
    "with_answers": 1000,
    "updated": "..."
  },
  "questions": [...]
}
```

## Загрузка Prompt Templates

`generator/task_generator.py` не хранит большие prompts в коде. Они читаются из:

```text
prompts/task_generator/system.md
prompts/task_generator/instruction.md
prompts/task_generator/environment.md
prompts/task_generator/solution.md
prompts/task_generator/tests.md
prompts/task_generator/dockerfile.md
```

Кодовый контракт:

```python
PROMPT_TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "prompts" / "task_generator"
```

Файлы читаются через `.read_text(...).rstrip("\n")`, поэтому финальный trailing newline в prompt-файле не попадает в LLM request.

## LLM Client

`call_llm_api()` отправляет OpenAI-compatible `POST /chat/completions`.

Настройки по умолчанию:

- `DEFAULT_API_BASE = "https://api.openai.com/v1"`;
- `DEFAULT_MODEL = "claude-opus-4-6"`;
- `MAX_RETRIES = 3`;
- `REQUEST_TIMEOUT = 300`;
- `REQUEST_INTERVAL = 0.2`.

`max_tokens` по умолчанию не отправляется в request payload. Это снимает client-side cap на completion length и позволяет endpoint/model server самому ограничивать ответ по доступному контексту. Явный лимит все еще можно передать через `call_llm_api(..., max_tokens=...)`, если он понадобится для отдельной стадии.

`temperature` по умолчанию тоже не отправляется. В новой реализации стадии передают `temperature=None`, а `llm_client.call_llm_api()` не добавляет это поле в payload. Явную температуру можно вернуть для отдельной стадии, передав `call_llm_api(..., temperature=...)`.

Конфиг берется из CLI или env:

- `--api-base` или `OPENAI_API_BASE`;
- `--api-key` или `OPENAI_API_KEY`;
- `--model` или `MODEL_NAME`.

Каждая LLM стадия пишет telemetry в `api_token_usage.json`:

- model;
- task name;
- stage;
- attempt;
- status;
- prompt/completion/total tokens;
- duration;
- success/failure;
- агрегаты `by_stage`.

Для `instruction` включен `check_truncation=True`. При подозрении на обрезанный JSON/fenced block запрос повторяется, если остались retry.

## Генерация Task Directory

`TaskGenerator.generate()` выполняет каскадно:

```text
instruction -> environment -> solution -> difficulty -> tests -> dockerfile -> write_files
```

Если критическая стадия не вернула результат, задача считается failed и не записывается.

### 1. `instruction.md`

Метод: `_generate_instruction()`.

В prompt передаются:

- `title`;
- `tags`;
- HTML-cleaned `body`.

HTML чистится через `clean_html()`:

- `<pre><code>...</code></pre>` превращается в fenced code block;
- inline `<code>` превращается в backticks;
- ссылки сохраняются как Markdown links;
- списки, абзацы, `<strong>`, `<em>` сохраняются в Markdown-like виде;
- остальной HTML удаляется.

Temperature по умолчанию не передается.

Ответ принимается как raw Markdown. Если модель обернула его в ```markdown, wrapper удаляется.

### 2. `environment/`

Метод: `_generate_environment(instruction)`.

Prompt просит JSON:

```json
{
  "files": {
    "relative/path/filename": "file content"
  },
  "directories": ["task_file", "task_file/input"]
}
```

Temperature по умолчанию не передается.

Парсинг:

1. сначала ищется fenced ```json block;
2. если не найден или не парсится, пробуется весь response как JSON;
3. если не получилось, используется fallback:

```json
{"files": {}, "directories": ["task_file"]}
```

Все пути интерпретируются относительно `environment/`.

### 3. `solution/solve.sh`

Метод: `_generate_solution(instruction, env_data)`.

Prompt получает:

- generated `instruction`;
- cleaned accepted answer body;
- tags;
- список доступных environment files.

Список environment files формируется `format_env_file_list()`:

```text
[dir]  /app/task_file/input/
[file] /app/task_file/input/data.txt
       Content: first 200 chars...
```

Temperature по умолчанию не передается.

Ответ извлекается из ```bash или ```sh block. Если fenced block нет, берется весь response.

### 4. Difficulty

Метод: `_assess_difficulty(instruction)`.

Сложность сейчас определяется детерминированно, без LLM:

```text
easy:
  body_len <= 1200
  and body_len + answer_len <= 3000
  and code_blocks <= 1

hard:
  body_len >= 5000
  or body_len + answer_len >= 10000
  or code_blocks >= 4

medium:
  все остальное
```

`code_blocks` считается по cleaned question body: `body.count("```") // 2`.

### 5. `tests/`

Метод: `_generate_tests(instruction, env_data, solution)`.

Prompt просит JSON:

```json
{
  "test_sh": "",
  "test_outputs_py": "import pytest\n..."
}
```

`test_sh` из LLM response не используется. Генератор всегда записывает статический offline runner `STATIC_TEST_SH`, который:

- не вызывает `curl`, `uv`, `uvx`, `pip`, `apt-get`, GitHub или PyPI;
- выбирает Python 3.12+ interpreter, в котором доступен `pytest`, с приоритетом
  `PYTHON_BIN`, `python3.13`, `python3.12`, `/usr/local/bin/python3`,
  `python3`, `/usr/bin/python3`;
- запускает `python -m pytest /tests/test_outputs.py -rA`;
- пишет `1` или `0` в `/logs/verifier/reward.txt`.

Temperature по умолчанию не передается.

Логика attempts:

1. максимум 3 генерации;
2. JSON извлекается из fenced ```json или из всего response;
3. `test_outputs_py` должен существовать;
4. `ast.parse(test_outputs_py, feature_version=(3, 12))` должен пройти;
5. `_review_tests()` выполняет static review.

Static review сейчас проверяет:

- `test_outputs.py` не должен запускать `solve.sh` через `subprocess.run/call/check_call/check_output/Popen`, `os.system`, `os.popen`;
- imports кроме `pytest` должны быть из Python stdlib. Нестандартные библиотеки вроде `pandas`, `numpy`, `cv2` отклоняются.

Prompt для этой стадии явно синхронизирован с reviewer-ом: `tests.md` требует black-box postcondition checks, запрещает импортировать implementation modules (`solution`, `app`, `main`, `model`) и запрещает third-party imports (`yaml`, `pydantic`, `fastapi`, `IPython`, etc.). Reference solution передается только как context для понимания expected postconditions.

Если review прошел, тесты принимаются, а `test_sh` принудительно заменяется на `STATIC_TEST_SH`. Если review отклонил, candidate сохраняется уже со статическим runner-ом. Если ни один из 3 attempts не прошел, но были syntactically valid candidates, возвращается первый rejected candidate. Если candidates нет, стадия failed.

### 6. `environment/Dockerfile`

Метод: `_generate_dockerfile(instruction, env_data, solution, test_data)`.

Prompt получает:

- generated `instruction`;
- tags;
- список generated environment artifacts из `_format_env_file_list(env_data)`;
- generated `solution/solve.sh`;
- generated `tests/test_outputs.py`.

Temperature по умолчанию не передается.

Prompt выбирает Docker base image под runtime задачи, а не под verifier:

- Python tasks: `python:3.12-slim-bookworm` или новее;
- Node.js tasks: `node:20-slim`;
- Java tasks: `openjdk:17-slim` или Maven/Gradle image;
- Go tasks: `golang:1.21-bookworm`;
- Rust/PHP/Swift/CUDA/.NET tasks: соответствующий runtime image;
- general Linux/shell tasks: `ubuntu:22.04`.

Python 3.12+ и `pytest` для verifier добавляются генератором отдельно, поэтому Dockerfile не должен выбирать Python base только ради `/tests/test.sh`.

Список артефактов нужен, чтобы Dockerfile видел весь build context: например `requirements.txt`, `package.json`, `setup.sh`, `init.sh`, config files и `task_file/...`. Reference solution и verifier test code передаются как context для выбора runtime packages и base image.

Ответ извлекается из ```dockerfile или generic fenced block. Если ответ отсутствует, не извлекается или не проходит static review, стадия retry-ится до `DOCKERFILE_MAX_ATTEMPTS = 3`. Fallback Dockerfile больше не записывается: если после retry нет валидного Dockerfile, задача считается failed и не попадает в candidates.

После извлечения Dockerfile генератор дополнительно вызывает `_ensure_verifier_deps()`. Он гарантирует build-time наличие Python 3.12+ и `pytest` для verifier:

- если Dockerfile уже основан на Python 3.12+ image, добавляет `RUN python -m pip install --no-cache-dir pytest` до `COPY ./task_file`, если pytest не установлен;
- если Dockerfile основан на другом image, добавляет multi-stage `verifier_python` на `python:3.12-slim-bookworm` и копирует `/usr/local` в final image до `COPY ./task_file`.

Перед записью `environment/Dockerfile` `_write_files()` повторно применяет `_ensure_verifier_deps()`. Это write-time guard: даже если отдельный путь генерации вернул Dockerfile без verifier Python/pytest, на диск сохраняется уже пропатченный Dockerfile.

Это нужно, чтобы `/tests/test.sh` не скачивал зависимости во время проверки и всегда исполнял `test_outputs.py` новым Python.

Static review Dockerfile проверяет:

- есть `FROM`;
- нет пустого `apt-get install -y && ...`;
- нет `astral.sh/uv` и `uvx`;
- есть Python 3.12+ и `pytest` для verifier после `_ensure_verifier_deps()`;
- `COPY` не ссылается на несуществующие build-context paths;
- generated environment artifacts копируются в image.

## Запись Файлов

`_write_files()` создает:

```text
task_XXXXX/
  instruction.md
  task.toml
  environment/
    Dockerfile
    task_file/...
  solution/
    solve.sh
  tests/
    __init__.py
    test.sh
    test_outputs.py
```

`task_name`:

- если generator получил `index`, используется `task_{index:05d}`;
- иначе slug из title + question id.

`task.toml` содержит:

- metadata: author, difficulty, category, tags, categories, source URL, source score;
- verifier timeout: 300 sec;
- agent timeout: 600 sec;
- environment build timeout: 120 sec;
- cpus: 1;
- memory: 1G;
- storage: 5G.

Category выбирается так:

1. `selected_category`, если есть;
2. первая category из `categories`;
3. fallback `general`;
4. для некоторых тегов есть старый local mapping, например `linux -> system-administration`, `docker -> containerization`.

## Параллелизм И Итоговые Артефакты

`task_generator.py` запускает `ThreadPoolExecutor(max_workers=args.workers)`.

В конце пишет:

```text
CANDIDATES_DIR/generation_summary.json
task_generator.log
api_token_usage.json
```

`generation_summary.json` содержит:

- total processed;
- success count;
- error count;
- skip count;
- elapsed seconds;
- model;
- timestamp;
- token usage summary.

## Валидация Generated Candidates

Валидация не входит в `scripts/generate_tasks.sh`; она запускается отдельно через:

```bash
python -m validator.validate_tasks \
  --input ./candidates \
  --output ./validated \
  --workers 8 \
  --timeout 300
```

`validator/validate_tasks.py` делает Docker round-trip:

1. проверяет наличие `environment/Dockerfile`, `solution/solve.sh`, `tests/test.sh`;
2. собирает image:

```bash
docker build -t tl-validate-<task> -f environment/Dockerfile environment/
```

3. запускает container с `--memory 1g --cpus 1`;
4. копирует `solution/solve.sh` в `/app/solve.sh`;
5. запускает reference solution в `/app`;
6. копирует `tests/` в `/tests`;
7. запускает `/tests/test.sh` в `/app`;
8. читает `/logs/verifier/reward.txt`;
9. если reward == 1, копирует task directory в validated output.

В конце пишет:

```text
validated/validation_report.json
validated/validate_tasks.log
```

## Validation-First Flow

Для большого generated set есть верхнеуровневый runner полного цикла:

```bash
python -m scripts.run_task_pipeline \
  --input ./questions.jsonl \
  --output ./terminal-lego-work/task-pipeline/run-name \
  --generate-workers 24 \
  --validate-workers 64 \
  --dockerfile-workers 8 \
  --repair-workers 4 \
  --timeout 300 \
  --resume
```

Он выполняет:

```text
1. prepare StackOverflow seed
2. generate candidates
3. validate baseline candidates
4. collect failed tasks
5. regenerate Dockerfile for eligible failures
6. validate regenerated failed tasks
7. diagnose remaining failures
8. run repair loop with validation
9. merge accepted baseline + regenerated + repaired tasks
```

Финальный датасет accepted tasks лежит в:

```text
RUN_DIR/accepted/
RUN_DIR/pipeline_summary.json
```

Ниже описаны те же стадии как отдельные ручные команды.

Сначала запускается baseline validation текущих candidates:

```bash
python -m validator.validate_tasks \
  --input ./candidates \
  --output ./validation/baseline_current_dockerfiles \
  --workers 8 \
  --timeout 300
```

Затем failed tasks собираются в отдельный JSONL:

```bash
python scripts/validation_first_pipeline.py collect-failures \
  --validation-report ./validation/baseline_current_dockerfiles/validation_report.json \
  --tasks-dir ./candidates \
  --output ./validation/failed_tasks.jsonl
```

`failed_tasks.jsonl` содержит `task`, baseline status/reward/error и `regen_eligible`. Dockerfile regeneration запускается только по `regen_eligible` задачам:

```bash
python -m scripts.regenerate_dockerfiles \
  --tasks-dir ./candidates \
  --task-list ./validation/failed_tasks.jsonl \
  --report-dir ./validation/dockerfile_regen \
  --workers 8
```

`scripts/regenerate_dockerfiles.py` использует тот же strict Dockerfile review, что и `TaskGenerator`. Если новая генерация не проходит review или API падает, старый Dockerfile сохраняется, а строка получает status `kept_old`; скрипт не пишет сомнительный fallback поверх старого файла.

После повторной validation regenerated failed tasks accepted задачи объединяются:

```bash
python scripts/validation_first_pipeline.py merge-accepted \
  --baseline-validated-dir ./validation/baseline_current_dockerfiles \
  --regenerated-validated-dir ./validation/regenerated_dockerfiles \
  --repair-validated-dir ./repair/repaired_validated \
  --output-dir ./validation/final_accepted
```

## Важные Ограничения Текущей Реализации

- Task generation генерирует небольшие standalone terminal tasks, обычно без полноценного repository context.
- Tests генерируются моделью, но проходят только static review до Docker validation.
- `test.sh` не генерируется LLM-ом: он всегда статический offline runner.
- Нестандартные Python imports в `test_outputs.py` static review отклоняет.
- Валидация проверяет reference solution, а не agent solution.
- `scripts/generate_tasks.sh` принимает JSONL, но `generator/task_generator.py` напрямую принимает только generator JSON с полем `questions`.
