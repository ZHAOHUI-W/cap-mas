"""Contracts for reproducible, phase-separated experiment artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentRunConfig:
    """Non-secret controls that define one CAP-MAS experiment run."""

    run_id: str
    task_id: str
    task: str
    seed: int
    protocol: str
    proposal_mode: str
    execution_mode: str
    model: str
    endpoint_host: str
    policy_agents: int
    max_workers: int
    llm_deadline_ms: int
    llm_max_output_tokens: int
    llm_max_retries: int
    llm_proposal_retries: int
    schema_mode: str
    manager_plan_fallback: bool = False
    policy_strategies: tuple[str, ...] = ()
    geometry_mode: str = "disabled"
    geometry_deadline_ms: int = 50
    geometry_depth_subsample: int = 16
    rehearsal_mode: str = "disabled"
    preview_backend: str = "none"
    privilege_mode: str = "realistic_sensor"
    artifact_dir: str = ""
    runner_version: str = "capmas-0.1"

    def __post_init__(self) -> None:
        if not self.run_id or not self.task_id or not self.task or not self.model:
            raise ValueError("experiment identity fields must not be empty")
        if self.protocol not in {"legacy", "staged"}:
            raise ValueError("experiment protocol must be legacy or staged")
        if self.proposal_mode not in {"subgoal_serial", "ready_wave"}:
            raise ValueError("experiment proposal mode must be subgoal_serial or ready_wave")
        if self.execution_mode not in {"fixed_graph", "rolling"}:
            raise ValueError("experiment execution mode must be fixed_graph or rolling")
        if not self.endpoint_host:
            raise ValueError("experiment endpoint host must not be empty")
        if self.policy_agents <= 0 or self.max_workers <= 0:
            raise ValueError("experiment agent and worker counts must be positive")
        if self.llm_deadline_ms <= 0 or self.llm_max_output_tokens <= 0:
            raise ValueError("experiment LLM budgets must be positive")
        if self.llm_max_retries < 0 or self.llm_proposal_retries < 0:
            raise ValueError("experiment retry budgets must not be negative")
        if self.schema_mode not in {"strict_provider_schema", "local_json_validation", "none"}:
            raise ValueError("unsupported experiment schema mode")
        if self.geometry_mode not in {"disabled", "shadow", "online_bounded"}:
            raise ValueError("unsupported geometry mode")
        if self.geometry_deadline_ms <= 0:
            raise ValueError("geometry deadline must be positive")
        if self.geometry_depth_subsample <= 0:
            raise ValueError("geometry depth subsample must be positive")
        if self.rehearsal_mode not in {"disabled", "shadow", "online_bounded"}:
            raise ValueError("unsupported rehearsal mode")
        if self.privilege_mode not in {"realistic_sensor", "diagnostic_privileged"}:
            raise ValueError("unsupported privilege mode")
        if self.policy_strategies:
            if len(self.policy_strategies) != self.policy_agents:
                raise ValueError(
                    "policy_strategies must contain one value per policy agent"
                )
            allowed = {"balanced", "safety", "robust", "efficient"}
            unknown = sorted(set(self.policy_strategies) - allowed)
            if unknown:
                raise ValueError(
                    f"unsupported policy strategies: {', '.join(unknown)}"
                )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary without credentials or raw prompts."""
        return asdict(self)
