Based on the following Terminal Bench task, analyze and generate the required environment files.

**Requirements:**
1. Analyze what preset files the task needs (e.g., input data, config files)
2. Generate reasonable test data
3. File paths relative to environment/ directory
4. If needed, create subdirectory structure like task_file/input/

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
