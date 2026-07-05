Based on the following Terminal Bench task, generate a Dockerfile for the environment.

**Requirements:**
1. Choose an appropriate base image:
   - Python tasks: `python:3.13-slim-bookworm`
   - Node.js tasks: `node:20-slim`
   - Java tasks: `openjdk:17-slim`
   - Go tasks: `golang:1.21-bookworm`
   - General Linux/shell tasks: `ubuntu:22.04`
2. Install necessary packages
3. Set WORKDIR to /app
4. COPY ./task_file /app/task_file

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

Please output only Dockerfile content, wrapped with ```dockerfile```.
