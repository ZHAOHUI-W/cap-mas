from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def _runner_module():
    path = Path(__file__).parents[1] / "scripts" / "run_libero_b3_llm.py"
    spec = importlib.util.spec_from_file_location("run_libero_b3_llm", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_evaluator_accepts_policy_checkpoint_name() -> None:
    runner = _runner_module()
    predicates = ("object_in_gripper(akita_black_bowl)", "gripper_closed()")
    subgraph = SimpleNamespace(
        checkpoints=(
            SimpleNamespace(
                name="pick_bowl_valid",
                predicates=predicates,
                validate=True,
            ),
        )
    )
    node = SimpleNamespace(node_id="chk_pick_bowl", postconditions=predicates)
    context = SimpleNamespace(scene=object())

    class Verifier:
        def goal_satisfied(self, actual_predicates, scene):
            assert actual_predicates == predicates
            assert scene is context.scene
            return True

    assert runner._evaluate_checkpoint(subgraph, node, context, Verifier()) == "success"

