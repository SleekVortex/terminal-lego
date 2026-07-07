# Preinstalled OpenCode Runtime

Harbor's stock `opencode` agent installs nvm, Node, and OpenCode inside every
task container. For Terminal-Lego rollouts this directory provides a reusable
Linux/amd64 OpenCode runtime and a compose override that mounts it into the task
container.

Build the runtime image on a machine where GitHub release assets are reachable.
If this requires a proxy, set it in the same shell session as `docker buildx`:

```bash
export HTTP_PROXY=http://5.101.116.13:8888
export HTTPS_PROXY=http://5.101.116.13:8888
docker buildx build \
  --platform linux/amd64 \
  --build-arg HTTP_PROXY=$HTTP_PROXY \
  --build-arg HTTPS_PROXY=$HTTPS_PROXY \
  --build-arg http_proxy=$HTTP_PROXY \
  --build-arg https_proxy=$HTTPS_PROXY \
  -t terminal-lego/opencode-tools:1.17.13 \
  -f configs/opencode/Dockerfile \
  --load \
  .
docker save terminal-lego/opencode-tools:1.17.13 \
  | gzip > /tmp/opencode-tools-1.17.13-linux-amd64.tar.gz
scp /tmp/opencode-tools-1.17.13-linux-amd64.tar.gz \
  cpu:/data/avzavodov/opencode-tools-1.17.13-linux-amd64.tar.gz
```

Load and extract it on `cpu`:

```bash
gunzip -c /data/avzavodov/opencode-tools-1.17.13-linux-amd64.tar.gz \
  | docker load
bash configs/opencode/extract_runtime.sh terminal-lego/opencode-tools:1.17.13
```

The extracted runtime lives in `configs/opencode/runtime/` and is intentionally
ignored by git.

For Docker bridge containers, expose the model endpoint through a separate SSH
tunnel that listens on the Docker bridge gateway. Do not stop the existing
`127.0.0.1:30002` tunnel.

```bash
tmux new -d -s glm52-opencode-docker-tunnel-30003 \
  'ssh -N -T -g \
    -L 172.16.0.1:30003:localhost:30001 \
    -p 2222 \
    -i /home/avzavodov/projects/mlspace__private_key.txt \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o StrictHostKeyChecking=no \
    -o ExitOnForwardFailure=yes \
    lm-mpi-job-3aed4232-c8b6-43e6-9294-1ff68df3c96d-mpimaster-0.ai0001071-01058@ssh-sr008-jupyter.ai.cloud.ru'
```

For GLM 5.2, `PreinstalledOpenCode` registers a native OpenCode custom provider:

- provider id: `glm`
- provider package: `@ai-sdk/openai-compatible`
- default base URL: `http://host.docker.internal:30003/v1`
- model: `glm/glm-5.2-fp8`
- `agent.build.prompt`: Terminal-Lego system prompt for solution generation

The wrapper can still receive `MODEL_NAME=openai/glm-5.2-fp8`; the agent rewrites
that OpenAI-style alias to `glm/glm-5.2-fp8` before invoking `opencode`. This
keeps OpenCode on `/v1/chat/completions` and avoids the OpenAI Responses API
path.

The same system prompt is prepended to `agent/trajectory.json` as the first
`source="system"` step, followed by the user task instruction and the agent
steps from OpenCode's JSON stream.

Smoke run:

```bash
AGENT=preinstalled-opencode \
MODEL_NAME=openai/glm-5.2-fp8 \
OPENAI_API_KEY=EMPTY \
DOCKER_NETWORK_STRATEGY=bridge \
scripts/generate_solutions.sh ./validated ./runs
```
