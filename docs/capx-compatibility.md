# CAP-X Compatibility and Adapter Boundary

## 1. Purpose

CAP-MAS must be able to run the CAP-X API path unchanged enough to support fair comparisons. Compatibility is an adapter boundary, not permission to expose CAP-X's unrestricted execution model to the new runtime.

## 2. Preserved API surface

The first LIBERO adapter should wrap the existing API names where available:

~~~text
get_observation
get_object_pose
sample_grasp_pose
goto_pose
open_gripper
close_gripper
goto_home_joint_position
get_object_3d_points_and_masks_from_language
get_oriented_bounding_box_from_3d_points
~~~

Optional CAP-X/CuRobo functions remain feature-gated until their action contracts are defined.

## 3. Adapter modes

| Mode | Purpose | Agent access |
| --- | --- | --- |
| capx_legacy | Reproduce CAP-X behavior | CAP-X code execution path |
| capx_typed | Same API wrapped as typed skills | Registered skills only |
| capmas_contract | Full contract and verification runtime | Typed skills and snapshots |

## 3.1 YAML and API reuse

`capmas.backends.capx_libero_factory.build_capx_runtime_from_yaml()` accepts an
existing CAP-X YAML. By default it reads `env.cfg.low_level` and calls CAP-X's
own `instantiate()` on that node, so the CAP-X `CodeExecutionEnv` and its
arbitrary Python executor are not constructed. The configured API names are
then resolved through CAP-X's `get_api()` registry, instantiated against the
same low-level environment, and exposed to CAP-MAS only through allowlisted
`CAPXTypedSkill` bindings.

If a caller explicitly needs the existing high-level CAP-X wrapper for a
compatibility test, it can opt into `instantiate_code_env=True`; this is not
the default CAP-MAS execution path.

## 4. Required CAP-X parity

- Same task config, suite, task ID, initial state seeds, and horizon.
- Same API servers and model backend where applicable.
- Same observation images and robot proprioception at comparable boundaries.
- Same action primitives before adding new CAP-MAS modules.
- Separate reporting for code-generation success, task completion, reward, and agent-visible verification.

## 5. Known semantic differences to record

CAP-X currently executes Python through an in-process exec path and exposes low-level objects in some configurations. CAP-MAS intentionally does not reproduce this permissive behavior in its main mode. Any gain may therefore combine coordination and constrainability; the experiments must isolate them with capx_typed and validator-only ablations.

CAP-X multi-turn regeneration can replace future code after physical actions have already occurred. CAP-MAS treats physical history as immutable and replans from a new observed snapshot.

## 6. Adapter interface

~~~python
class RobotBackend(Protocol):
    def reset(
        self,
        seed: int | None = None,
        options: Mapping[str, object] | None = None,
    ) -> EpisodeStart: ...
    def observe(self) -> SceneSnapshot: ...
    def execute_skill(
        self,
        skill: TypedSkill,
        args: dict[str, object],
        budget: ExecutionBudget,
    ) -> SkillExecutionResult: ...
    def stop(self, lease: ActionLease) -> None: ...
    def evaluator_success(self) -> bool: ...  # evaluator-only
~~~

The adapter must expose evaluator-only success separately from the agent-visible observation stream.

The CAP-X `ApiBase` is not itself the CAP-MAS backend interface. Its mixed
perception/control methods are converted into separate allowlisted typed skill
bindings and observation/perception adapters. The core package does not import
CAP-X or receive its environment handle.
