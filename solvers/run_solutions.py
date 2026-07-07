#!/usr/bin/env python3
"""Run solver agents for Terminal-Lego tasks using Harbor.

This module intentionally delegates harness-specific execution to Harbor. The
same CLI can run Terminus, mini-swe-agent, OpenHands, Codex, SWE-Agent, and any
other Harbor-supported agent/environment combination.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Sequence

from solvers.config_builder import build_job_config
from solvers.harbor_runner import run_job
from solvers.summary import summarize_job


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", required=True, type=Path, help="Task or dataset directory")
    parser.add_argument("--jobs-dir", type=Path, default=Path("jobs"))
    parser.add_argument(
        "--job-name",
        default=f"solution-runs-{time.strftime('%Y%m%dT%H%M%S')}",
    )
    parser.add_argument("--agent", default=os.environ.get("AGENT", "terminus-2"))
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME"))
    parser.add_argument("--env", default=os.environ.get("HARBOR_ENV", "docker"))
    parser.add_argument("--api-base", default=os.environ.get("OPENAI_API_BASE"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--n-attempts", type=int, default=int(os.environ.get("N_ATTEMPTS", "1")))
    parser.add_argument("--n-concurrent", type=int, default=int(os.environ.get("N_CONCURRENT", "1")))
    parser.add_argument("--max-retries", type=int, default=int(os.environ.get("MAX_RETRIES", "0")))
    parser.add_argument("--n-tasks", type=int, default=None)
    parser.add_argument("--timeout-multiplier", type=float, default=1.0)
    parser.add_argument("--agent-timeout-multiplier", type=float, default=None)
    parser.add_argument("--verifier-timeout-multiplier", type=float, default=None)
    parser.add_argument("--environment-build-timeout-multiplier", type=float, default=None)
    parser.add_argument("--force-build", action="store_true")
    parser.add_argument("--delete", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--extra-docker-compose", action="append", type=Path, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--disable-verification", action="store_true")
    parser.add_argument("--reward-threshold", type=float, default=1.0)
    parser.add_argument("--extra-instruction-path", action="append", type=Path, default=[])
    parser.add_argument("--include-task-name", action="append", default=[])
    parser.add_argument("--exclude-task-name", action="append", default=[])
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--agent-kwarg", action="append", default=[])
    parser.add_argument("--agent-env", action="append", default=[])
    parser.add_argument("--agent-include-logs", action="append", default=[])
    parser.add_argument("--agent-exclude-logs", action="append", default=[])
    parser.add_argument("--environment-kwarg", action="append", default=[])
    parser.add_argument("--environment-env", action="append", default=[])
    parser.add_argument("--verifier-env", action="append", default=[])
    parser.add_argument("--verifier-include-logs", action="append", default=[])
    parser.add_argument("--verifier-exclude-logs", action="append", default=[])
    parser.add_argument("--trajectory-raw-content", action="store_true")
    parser.add_argument("--trajectory-linear-history", action="store_true")
    parser.add_argument("--trajectory-config-json", default=None)
    parser.add_argument("--dry-run-config", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = build_job_config(args)

    if args.dry_run_config:
        args.dry_run_config.parent.mkdir(parents=True, exist_ok=True)
        args.dry_run_config.write_text(
            config.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote Harbor job config to {args.dry_run_config}")
        return 0

    asyncio.run(run_job(config))

    job_dir = config.jobs_dir / config.job_name
    summary = summarize_job(job_dir, args.reward_threshold)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
