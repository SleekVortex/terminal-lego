Based on the following Terminal Bench task, its environment, and its reference solution, generate test code.

**Generate two files:**

1. **test.sh** - Test runner script, format:
```bash
#!/bin/bash

apt-get update && apt-get install -y curl

curl -LsSf https://astral.sh/uv/0.9.5/install.sh | sh
source $HOME/.local/bin/env

if [ "$PWD" = "/" ]; then
    echo "Error: No working directory set."
    exit 1
fi

uvx \\
  -p 3.13 \\
  -w pytest==8.4.1 \\
  -w pytest-json-ctrf==0.3.5 \\
  pytest --ctrf /logs/verifier/ctrf.json /tests/test_outputs.py -rA

if [ $? -eq 0 ]; then
  echo 1 > /logs/verifier/reward.txt
else
  echo 0 > /logs/verifier/reward.txt
fi
```

2. **test_outputs.py** - pytest test file that verifies the state AFTER solve.sh has already been executed:

**CRITICAL RULES for test_outputs.py:**
- solve.sh has ALREADY been executed before test_outputs.py runs. Do NOT call solve.sh again.
- Tests should ONLY check the resulting state: output files, file contents, directory structures, etc.
- Use os.path.exists(), open().read(), subprocess.run() for verification commands — but NEVER run solve.sh.

**ROBUSTNESS RULES:**
- When checking output files that list paths, use `line.endswith("filename")` or `os.path.basename(line)`.
- When checking file contents, use `in` operator or regex, NOT exact equality.
- When counting lines, allow for trailing newlines: `len([l for l in content.strip().splitlines() if l.strip()])`.
- Prefer checking that files/directories EXIST on disk over parsing output text.

**Output format (JSON):**
```json
{{
    "test_sh": "#!/bin/bash\\n...",
    "test_outputs_py": "import pytest\\n..."
}}
```

**Task instruction:**
{instruction}

**Environment files in the container:**
{env_file_list}

**Reference solution (solve.sh) that will be executed:**
```bash
{solution}
```

**Tags:** {tags}

Please output only JSON, wrapped with ```json```. Ensure Python code syntax is correct.
