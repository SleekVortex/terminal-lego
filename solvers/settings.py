"""Settings for solver execution integrations."""

from __future__ import annotations

from dataclasses import dataclass


PREINSTALLED_OPENCODE_AGENT = "preinstalled-opencode"
DEEPAGENT_AGENT = "deepagent"


@dataclass(frozen=True)
class AgentImportTarget:
    module: str
    class_name: str

    @property
    def import_path(self) -> str:
        return f"{self.module}:{self.class_name}"


SOLVER_AGENT_IMPORTS = {
    DEEPAGENT_AGENT: AgentImportTarget(
        module="solvers.agents.deepagent",
        class_name="DeepAgent",
    ),
    PREINSTALLED_OPENCODE_AGENT: AgentImportTarget(
        module="solvers.agents.preinstalled_opencode",
        class_name="PreinstalledOpenCode",
    ),
}


def import_path_for_agent(agent_name: str) -> str:
    return SOLVER_AGENT_IMPORTS[agent_name].import_path
