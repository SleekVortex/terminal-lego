Based on the following Terminal Bench task, generate a Dockerfile for the environment.

**Requirements:**
1. Choose an appropriate base image:
   - Choose the base image for the task runtime first.
   - Do not choose a Python image only because the verifier uses Python.
   - Python tasks: `python:3.12-slim-bookworm` or newer.
   - Node.js tasks: `node:20-slim`.
   - Java tasks: `openjdk:17-slim` or a Maven/Gradle image when build tooling is needed.
   - Go tasks: `golang:1.21-bookworm`.
   - Rust tasks: a Rust image or install the Rust toolchain explicitly.
   - PHP tasks: a PHP CLI image.
   - Swift tasks: a Swift image.
   - CUDA/GPU tasks: an NVIDIA CUDA image when CUDA tooling is required.
   - .NET tasks: a .NET SDK image.
   - General Linux/shell tasks: `ubuntu:22.04`.
2. Install necessary packages
3. Set WORKDIR to /app
4. COPY ./task_file /app/task_file
5. The generator will add offline verifier dependencies separately: Python 3.12+ and
   `pytest` installed for that interpreter. Do not install verifier dependencies at test runtime.
6. Use the generated artifact context below to include task runtime dependencies and copy any
   required non-task_file build-context files such as requirements.txt, package.json, setup.sh,
   init.sh, or config files.

**Output format:**
```dockerfile
FROM <base_image>
WORKDIR /app
RUN apt-get update && apt-get install -y <packages> && rm -rf /var/lib/apt/lists/*
COPY ./task_file /app/task_file
```

**Task instruction:**
{instruction}

**Tags:** {tags}

**Generated environment artifacts available in the Docker build context:**
{env_file_list}

**Generated reference solution:**
```bash
{solution}
```

**Generated verifier test_outputs.py:**
```python
{test_outputs_py}
```

Please output only Dockerfile content, wrapped with ```dockerfile```.
