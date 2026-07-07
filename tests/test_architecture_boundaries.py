from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _python_files(root: Path) -> list[Path]:
    return [
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts and path.is_file()
    ]


def test_task_generation_layer_does_not_depend_on_solver_runtime() -> None:
    for path in _python_files(REPO_ROOT / "generator"):
        text = path.read_text(encoding="utf-8")
        assert "harbor." not in text, path
        assert "solvers." not in text, path
        assert "preinstalled_opencode" not in text, path


def test_solver_layer_does_not_depend_on_task_generation_agents() -> None:
    for path in _python_files(REPO_ROOT / "solvers"):
        text = path.read_text(encoding="utf-8")
        assert "generator.agents" not in text, path
        assert "generator.orchestrator" not in text, path
        assert "generator.task_generator" not in text, path


def test_module_entrypoints_do_not_patch_python_import_path() -> None:
    entrypoints = [
        REPO_ROOT / "generator" / "task_generator.py",
        REPO_ROOT / "solvers" / "run_solutions.py",
    ]
    for path in entrypoints:
        text = path.read_text(encoding="utf-8")
        assert "sys.path" not in text, path
        assert "REPO_ROOT" not in text, path
