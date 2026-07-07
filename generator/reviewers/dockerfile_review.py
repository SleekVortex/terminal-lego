from __future__ import annotations

import json
import re
from typing import Any, Optional


VERIFIER_PYTHON_STAGE = """FROM python:3.12-slim-bookworm AS verifier_python
RUN python -m pip install --no-cache-dir pytest
"""

VERIFIER_PYTHON_COPY = """COPY --from=verifier_python /usr/local /usr/local
ENV PATH="/usr/local/bin:${PATH}"
"""

VERIFIER_PYTEST_INSTALL = "RUN python -m pip install --no-cache-dir pytest"


def has_python_312_plus(dockerfile: str) -> bool:
    if re.search(r"(?im)^FROM\s+python:3\.(?:1[2-9]|[2-9][0-9])", dockerfile):
        return True
    if re.search(r"\bpython3\.(?:1[2-9]|[2-9][0-9])\b", dockerfile):
        return True
    return False


def has_pytest_for_verifier(dockerfile: str) -> bool:
    return bool(
        re.search(r"\bpython(?:3(?:\.\d+)?)?\s+-m\s+pip\s+install\b[^\n\\]*\bpytest\b", dockerfile)
        or re.search(r"\bpip3?\s+install\b[^\n\\]*\bpytest\b", dockerfile)
    )


def insert_before_task_file_copy(dockerfile: str, insertion: str) -> str:
    copy_match = re.search(r"(?im)^COPY\s+\./task_file\s+/app/task_file\s*$", dockerfile)
    if copy_match:
        return (
            dockerfile[: copy_match.start()].rstrip()
            + "\n\n"
            + insertion.rstrip()
            + "\n"
            + dockerfile[copy_match.start() :].lstrip()
        )
    return dockerfile.rstrip() + "\n\n" + insertion.rstrip()


def ensure_verifier_deps(dockerfile: str) -> str:
    text = dockerfile.strip()
    if has_python_312_plus(text) and has_pytest_for_verifier(text):
        return text
    if has_python_312_plus(text):
        return insert_before_task_file_copy(text, VERIFIER_PYTEST_INSTALL)
    if VERIFIER_PYTHON_STAGE in text or "AS verifier_python" in text:
        return text
    return VERIFIER_PYTHON_STAGE.rstrip() + "\n\n" + insert_before_task_file_copy(text, VERIFIER_PYTHON_COPY)


def extract_dockerfile_from_response(response: str) -> Optional[str]:
    match = re.search(r"```dockerfile\n?(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\n?(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def normalize_copy_source(source: str) -> str:
    source = source.strip().strip('"').strip("'")
    while source.startswith("./"):
        source = source[2:]
    return source.rstrip("/")


def env_artifact_paths(env_data: Optional[dict[str, Any]]) -> set[str]:
    if not env_data:
        return {"task_file"}
    paths = {normalize_copy_source(path) for path in env_data.get("directories", []) if path}
    paths.update(normalize_copy_source(path) for path in env_data.get("files", {}) if path)
    paths.discard("")
    if not paths:
        paths.add("task_file")
    return paths


def env_path_exists(source: str, env_paths: set[str]) -> bool:
    source = normalize_copy_source(source)
    if not source or source == ".":
        return True
    if source in env_paths:
        return True
    prefix = source + "/"
    return any(path.startswith(prefix) for path in env_paths)


def parse_copy_sources(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.upper().startswith("COPY "):
        return []
    rest = stripped[5:].strip()
    if not rest or "--from=" in rest or rest.startswith("<<"):
        return []
    while rest.startswith("--"):
        parts = rest.split(None, 1)
        if len(parts) == 1:
            return []
        rest = parts[1].strip()
    if rest.startswith("["):
        try:
            values = json.loads(rest)
        except json.JSONDecodeError:
            return []
        if isinstance(values, list) and len(values) >= 2:
            return [str(value) for value in values[:-1]]
        return []
    parts = rest.split()
    if len(parts) < 2:
        return []
    return parts[:-1]


def review_dockerfile(dockerfile: str, env_data: Optional[dict[str, Any]] = None) -> dict[str, object]:
    issues: list[str] = []
    text = dockerfile.strip()
    if not text:
        return {"pass": False, "issues": ["Dockerfile is empty"]}
    if not re.search(r"(?im)^FROM\s+\S+", text):
        issues.append("Dockerfile must contain a FROM instruction")
    if re.search(r"(?is)apt-get\s+install\s+-y(?:\s+--no-install-recommends)?\s*(?:&&|;)", text):
        issues.append("empty apt-get install command")
    if "astral.sh/uv" in text or re.search(r"(?im)\buvx\b", text):
        issues.append("Dockerfile must not install or run uv/uvx test-runtime tooling")
    if not (has_python_312_plus(text) and has_pytest_for_verifier(text)):
        issues.append("Dockerfile must include Python 3.12+ and pytest for verifier")

    env_paths = env_artifact_paths(env_data)
    copy_sources: list[str] = []
    for line in text.splitlines():
        copy_sources.extend(parse_copy_sources(line))
    normalized_sources = {normalize_copy_source(source) for source in copy_sources}
    copies_all_context = "." in normalized_sources

    missing_sources = [
        source for source in sorted(normalized_sources)
        if source and not env_path_exists(source, env_paths)
    ]
    if missing_sources:
        issues.append("COPY references missing build-context paths: " + ", ".join(missing_sources[:5]))

    root_paths = sorted({path.split("/", 1)[0] for path in env_paths if path})
    missing_root_paths = [
        path for path in root_paths
        if path not in normalized_sources
        and not copies_all_context
        and not any(src and path.startswith(src + "/") for src in normalized_sources)
    ]
    if missing_root_paths:
        issues.append("Dockerfile does not copy environment artifacts: " + ", ".join(missing_root_paths[:5]))

    return {"pass": not issues, "issues": issues}
