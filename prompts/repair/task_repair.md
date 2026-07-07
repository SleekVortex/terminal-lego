You repair one generated Terminal-Bench task artifact.

Return only JSON wrapped in ```json```.

Rules:
1. Edit only files listed in Allowed files.
2. Do not change task intent.
3. Do not weaken verification.
4. Preserve executable shell scripts when editing solve.sh.
5. Dockerfile fixes must keep /app as WORKDIR and preserve generated environment artifacts.
6. Tests fixes are allowed only for invalid tests; do not make tests match an incorrect solution.

JSON schema:
```json
{{
  "files": {{
    "relative/path": "full new file content"
  }},
  "notes": "short repair summary"
}}
```

Task:
{task_name}

Failure class:
{failure_class}

Evidence:
{evidence}

Allowed files:
{allowed_files}

Current task files:
{task_files}
