from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


def review_solution_shell(solution: str) -> dict[str, object]:
    issues: list[str] = []
    if not solution.strip():
        return {"pass": False, "issues": ["solve.sh is empty"]}
    if "read -p" in solution or "select " in solution:
        issues.append("solve.sh appears interactive")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "solve.sh"
        path.write_text(solution, encoding="utf-8")
        result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
        if result.returncode != 0:
            issues.append("bash syntax error: " + (result.stderr.strip() or result.stdout.strip()))
    return {"pass": not issues, "issues": issues}
