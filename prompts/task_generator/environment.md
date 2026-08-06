Based on the following Terminal Bench task, analyze and generate the required environment files.

**Requirements:**
1. Analyze what preset files the task needs (e.g., input data, config files)
2. Generate reasonable input fixtures only
3. File paths relative to environment/ directory
4. If needed, create subdirectory structure like task_file/input/
5. The environment is the pre-solution state. Fix/debug tasks must preserve the described bug; create tasks may include only a stub or TODO. Do not include the fix or hints. Requested output files must be absent (empty output directories are allowed), and the solve step must make a material change.

**Output format (JSON):**
```json
{{
    "files": {{
        "relative/path/filename": "file content",
        "task_file/input/example.txt": "example content..."
    }},
    "directories": ["task_file", "task_file/input", "task_file/output"]
}}
```

**Task instruction:**
{instruction}

**Original question info:**
Title: {title}
Tags: {tags}

Please output only JSON, wrapped with ```json```. If no files are needed, return empty files and directories.
