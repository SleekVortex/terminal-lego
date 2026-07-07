from __future__ import annotations

import ast
import sys


TEST_OUTPUTS_FEATURE_VERSION = (3, 12)

STATIC_TEST_SH = """#!/usr/bin/env bash
set +e

mkdir -p /logs/verifier

if [ "$PWD" = "/" ]; then
    echo "Error: No working directory set."
    echo 0 > /logs/verifier/reward.txt
    exit 1
fi

find_python_with_pytest() {
    for candidate in \
        "${PYTHON_BIN:-}" \
        python3.13 \
        python3.12 \
        /usr/local/bin/python3 \
        python3 \
        /usr/bin/python3
    do
        [ -n "$candidate" ] || continue

        if command -v "$candidate" >/dev/null 2>&1; then
            bin="$(command -v "$candidate")"
        elif [ -x "$candidate" ]; then
            bin="$candidate"
        else
            continue
        fi

        if "$bin" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 12) else 1)
PY
        then
            :
        else
            continue
        fi

        if "$bin" -m pytest --version >/dev/null 2>&1; then
            printf '%s\n' "$bin"
            return 0
        fi
    done

    return 1
}

PYTHON_BIN="$(find_python_with_pytest)"
if [ -z "$PYTHON_BIN" ]; then
    echo "Error: no Python 3.12+ with pytest is installed."
    echo 0 > /logs/verifier/reward.txt
    exit 1
fi

if ! "$PYTHON_BIN" -m pytest --version >/dev/null 2>&1; then
    echo "Error: pytest is not installed for $PYTHON_BIN."
    echo 0 > /logs/verifier/reward.txt
    exit 1
fi

"$PYTHON_BIN" -m pytest /tests/test_outputs.py -rA
status=$?

if [ "$status" -eq 0 ]; then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi

exit "$status"
"""


def parse_test_outputs_py(test_code: str) -> ast.AST:
    return ast.parse(test_code, feature_version=TEST_OUTPUTS_FEATURE_VERSION)


def review_tests(test_code: str) -> dict[str, object]:
    issues: list[str] = []
    try:
        tree = parse_test_outputs_py(test_code)
    except SyntaxError as exc:
        return {"pass": False, "issues": [f"Python syntax error: {exc.msg}"]}

    forbidden_runners = {"run", "call", "check_call", "check_output", "Popen"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            func_name = None
            owner_name = None
            if isinstance(func, ast.Attribute):
                func_name = func.attr
                if isinstance(func.value, ast.Name):
                    owner_name = func.value.id
            elif isinstance(func, ast.Name):
                func_name = func.id
            call_text = ast.get_source_segment(test_code, node) or ""
            if "solve.sh" in call_text and (
                (owner_name == "subprocess" and func_name in forbidden_runners)
                or (owner_name == "os" and func_name in {"system", "popen"})
                or func_name in {"system", "popen"}
            ):
                issues.append("test_outputs.py must not execute solve.sh")

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module_names = []
            if isinstance(node, ast.Import):
                module_names = [alias.name.split(".", 1)[0] for alias in node.names]
            elif node.module:
                module_names = [node.module.split(".", 1)[0]]
            for module_name in module_names:
                if module_name == "pytest":
                    continue
                if module_name not in sys.stdlib_module_names:
                    issues.append(f"non-stdlib import in test_outputs.py: {module_name}")

    return {"pass": not issues, "issues": issues}
