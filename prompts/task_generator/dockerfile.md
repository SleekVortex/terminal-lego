Based on the following Terminal Bench task, generate a Dockerfile for the environment.

**Requirements:**
1. Choose an appropriate base image:
   - Prefer `python:3.12-slim-bookworm` when possible.
   - Python tasks: `python:3.12-slim-bookworm` or newer.
   - Node.js tasks: `node:20-slim` plus Python 3.12+ verifier dependencies, or install Node.js on `python:3.12-slim-bookworm`.
   - Java tasks: `openjdk:17-slim` plus Python 3.12+ verifier dependencies, or install Java on `python:3.12-slim-bookworm`.
   - Go tasks: `golang:1.21-bookworm` plus Python 3.12+ verifier dependencies, or install Go on `python:3.12-slim-bookworm`.
   - General Linux/shell tasks: `python:3.12-slim-bookworm`.
2. Install necessary packages
3. Set WORKDIR to /app
4. COPY ./task_file /app/task_file
5. The image must include offline verifier dependencies: Python 3.12+ and `pytest`
   installed for that interpreter. Do not install verifier dependencies at test runtime.

**Output format:**
```dockerfile
FROM <base_image>
WORKDIR /app
RUN apt-get update && apt-get install -y <packages> && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir pytest
COPY ./task_file /app/task_file
```

**Task instruction:**
{instruction}

**Tags:** {tags}

Please output only Dockerfile content, wrapped with ```dockerfile```.
