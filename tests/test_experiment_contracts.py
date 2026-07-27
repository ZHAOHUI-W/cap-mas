import pytest

from capmas.contracts.experiment import ExperimentRunConfig


def test_experiment_run_config_is_json_safe_and_records_non_secret_controls() -> None:
    config = ExperimentRunConfig(
        run_id="run-1",
        task_id="libero_spatial_0",
        task="Place bowl on plate",
        seed=1,
        protocol="staged",
        proposal_mode="subgoal_serial",
        execution_mode="fixed_graph",
        model="gpt-5.5",
        endpoint_host="api.example.test",
        policy_agents=2,
        max_workers=2,
        llm_deadline_ms=60_000,
        llm_max_output_tokens=1536,
        llm_max_retries=2,
        llm_proposal_retries=0,
        schema_mode="strict_provider_schema",
        policy_strategies=("balanced", "safety"),
    )

    payload = config.to_dict()

    assert payload["task_id"] == "libero_spatial_0"
    assert payload["policy_agents"] == 2
    assert payload["schema_mode"] == "strict_provider_schema"
    assert payload["policy_strategies"] == ("balanced", "safety")
    assert "api_key" not in payload
    assert "api-key" not in payload


def test_experiment_run_config_rejects_invalid_phase_controls() -> None:
    with pytest.raises(ValueError, match="protocol"):
        ExperimentRunConfig(
            run_id="run-1",
            task_id="task",
            task="task",
            seed=1,
            protocol="unknown",
            proposal_mode="subgoal_serial",
            execution_mode="fixed_graph",
            model="gpt-5.5",
            endpoint_host="example.test",
            policy_agents=1,
            max_workers=1,
            llm_deadline_ms=1,
            llm_max_output_tokens=1,
            llm_max_retries=0,
            llm_proposal_retries=0,
            schema_mode="strict_provider_schema",
            policy_strategies=("balanced", "unknown"),
        )
