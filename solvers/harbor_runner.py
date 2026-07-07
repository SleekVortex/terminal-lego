"""Harbor import and execution boundary for solver runs."""

from __future__ import annotations


def import_harbor():
    try:
        from harbor.job import Job
        from harbor.models.job.config import DatasetConfig, JobConfig, RetryConfig
        from harbor.models.trial.config import (
            AgentConfig,
            EnvironmentConfig,
            TaskConfig,
            VerifierConfig,
        )
    except ImportError as exc:
        raise SystemExit(
            "Harbor is required for solution generation. Install dependencies "
            "with `pip install -r requirements.txt`."
        ) from exc

    return {
        "Job": Job,
        "JobConfig": JobConfig,
        "RetryConfig": RetryConfig,
        "DatasetConfig": DatasetConfig,
        "TaskConfig": TaskConfig,
        "AgentConfig": AgentConfig,
        "EnvironmentConfig": EnvironmentConfig,
        "VerifierConfig": VerifierConfig,
    }


async def run_job(config):
    harbor = import_harbor()
    Job = harbor["Job"]
    job = await Job.create(config)
    return await job.run()
