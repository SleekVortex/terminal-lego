from __future__ import annotations

import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class CommandResult:
    cmd: list[str]
    returncode: int | str
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    timeout: Optional[int] = None

    @classmethod
    def from_completed(cls, result: subprocess.CompletedProcess) -> "CommandResult":
        return cls(
            cmd=list(result.args),
            returncode=result.returncode,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
        )

    @classmethod
    def from_timeout(cls, cmd: list[str], exc: subprocess.TimeoutExpired, timeout: int) -> "CommandResult":
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        return cls(cmd=cmd, returncode="timeout", stdout=stdout, stderr=stderr, timed_out=True, timeout=timeout)

    def to_dict(self) -> dict:
        return asdict(self)


def docker_safe_name(value: str) -> str:
    safe = re.sub(r"[^a-z0-9.-]+", "-", value.lower())
    safe = re.sub(r"-+", "-", safe).strip(".-")
    return safe or "task"


class DockerRunner:
    def __init__(self, task_name: str, pid: Optional[int] = None):
        self.task_name = task_name
        self.safe_task_name = docker_safe_name(task_name)
        self.pid = pid or os.getpid()
        self.image_tag = f"tl-validate-{self.safe_task_name}"
        self.container_name = f"tl-val-{self.safe_task_name}-{self.pid}"

    def run_command(self, cmd: list[str], timeout: int = 300) -> CommandResult:
        try:
            completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            return CommandResult.from_completed(completed)
        except subprocess.TimeoutExpired as exc:
            return CommandResult.from_timeout(cmd, exc, timeout)

    def build(self, dockerfile: Path, env_dir: Path, timeout: int) -> CommandResult:
        return self.run_command(["docker", "build", "-t", self.image_tag, "-f", str(dockerfile), str(env_dir)], timeout)

    def start(self) -> CommandResult:
        return self.run_command(
            [
                "docker",
                "run",
                "-d",
                "--name",
                self.container_name,
                "--memory",
                "1g",
                "--cpus",
                "1",
                self.image_tag,
                "sleep",
                "infinity",
            ],
            timeout=30,
        )

    def exec(self, cmd: list[str], timeout: int = 300, workdir: Optional[str] = None) -> CommandResult:
        docker_cmd = ["docker", "exec"]
        if workdir:
            docker_cmd.extend(["-w", workdir])
        docker_cmd.append(self.container_name)
        docker_cmd.extend(cmd)
        return self.run_command(docker_cmd, timeout)

    def copy_to_container(self, source: str, destination: str, timeout: int = 10) -> CommandResult:
        return self.run_command(["docker", "cp", source, f"{self.container_name}:{destination}"], timeout)

    def read_reward(self) -> CommandResult:
        return self.exec(["cat", "/logs/verifier/reward.txt"], timeout=10)

    def cleanup(self) -> None:
        subprocess.run(["docker", "rm", "-f", self.container_name], capture_output=True, timeout=30)
        subprocess.run(["docker", "rmi", "-f", self.image_tag], capture_output=True, timeout=30)
