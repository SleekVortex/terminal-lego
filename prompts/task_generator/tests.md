Based on the following Terminal Bench task, its environment, and its reference solution, generate test code.

**Generate test output JSON.**

1. **test.sh** - ignored by the generator. Return an empty string for this field.
The generator writes a deterministic offline runner. Do not include curl, uv, uvx,
pip, apt-get, GitHub, PyPI, or any network install commands in test.sh.
The generated test_outputs.py is executed with Python 3.12+.

The runner used by the generator is equivalent to:
```bash
#!/usr/bin/env bash
set +e

if [ "$PWD" = "/" ]; then
    echo "Error: No working directory set."
    echo 0 > /logs/verifier/reward.txt
    exit 1
fi

python3.12 -m pytest /tests/test_outputs.py -rA
status=$?

if [ "$status" -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi

exit "$status"
```

2. **test_outputs.py** - pytest test file that verifies the state AFTER solve.sh has already been executed:

**CRITICAL RULES for test_outputs.py:**
- solve.sh has ALREADY been executed before test_outputs.py runs. Do NOT call solve.sh again.
- Tests should ONLY check the resulting state: output files, file contents, directory structures, etc.
- Use os.path.exists(), open().read(), subprocess.run() for verification commands — but NEVER run solve.sh.

**IMPORT AND BLACK-BOX RULES for test_outputs.py:**
- test_outputs.py is a black-box postcondition verifier. It must not import or execute implementation code.
- Use only Python standard library imports and pytest.
- Python 3.12 syntax is allowed.
- Do NOT import third-party packages such as yaml, pandas, numpy, pydantic, fastapi, IPython, tomli, cv2, PIL, openai, yt_dlp, or tiktoken.
- Do NOT import task/application modules such as solution, app, main, model, openai_client, download_video, logging_util, or any module created by solve.sh.
- Verify behavior through filesystem state, file contents, subprocess command output, and paths under /app.
- If a file format would normally need a third-party parser, use text/regex checks or stdlib modules such as json, xml.etree.ElementTree, csv, configparser, pathlib, sqlite3, zipfile, tarfile, gzip, hashlib, subprocess, and tomllib when applicable.

**ROBUSTNESS RULES:**
- When checking output files that list paths, use `line.endswith("filename")` or `os.path.basename(line)`.
- When checking file contents, use `in` operator or regex, NOT exact equality.
- When counting lines, allow for trailing newlines: `len([l for l in content.strip().splitlines() if l.strip()])`.
- Prefer checking that files/directories EXIST on disk over parsing output text.

**Output format (JSON):**
```json
{{
    "test_sh": "",
    "test_outputs_py": "import pytest\\n..."
}}
```

**Task instruction:**
{instruction}

**Environment files in the container:**
{env_file_list}

**Reference solution context (solve.sh, already executed before tests):**
Use this only to understand expected postconditions. Do not import it, execute it, or test its internal implementation.
```bash
{solution}
```

**Tags:** {tags}

Please output only JSON, wrapped with ```json```. Ensure Python code syntax is correct.
