# CAP-MAS Whole-System Structure

This document gives one end-to-end view of CAP-MAS. It is the map from a natural-language task to robot execution, verification, recovery, memory, skill evolution, and CAP-X comparison.

## 1. End-to-end architecture

~~~mermaid
flowchart TB
    U[User task and constraints] --> API[Task API]
    API --> M[Mission Manager]

    subgraph AGENT[Agent Plane - event triggered]
        M --> SG[Subgoal Graph]
        SG --> P[Policy Agent]
        P --> AC[Action Contract]
        AC --> V[Verifier Agent]
        V -->|approve| SCH[Runtime Scheduler]
        V -->|reject| R[Recovery Agent]
        R --> M
        MON[Execution Monitor] --> M
        C[Critic and Memory Skill Evolver] -. trace analysis .-> MSB[Memory Skill Registry]
        C -. boundary tests and priors .-> MEM[Persistent Memory]
        MC[Memory Controller] -. selects Top-K .-> MSB
        ME[Memory Executor] -. proposes updates .-> MEM
        RS[Robot Skill Evolver] -. later sequential phase .-> RSB[Robot Skill Registry]
        MEM --> M
        MEM --> P
        MSB --> MC --> ME
    end

    subgraph CONTRACT[Contract and Runtime Plane]
        SCH --> VAL[Schema and Resource Validator]
        VAL --> LEASE[Action Lease Manager]
        LEASE --> EX[Typed Skill Executor]
        EX --> MON
        ST[Versioned State Store] --> V
        ST --> P
        EX --> ST
    end

    subgraph WORLD[World Model Plane - asynchronous]
        SENS[RGB-D and Proprioception] --> SYNC[Sensor Synchronizer]
        SYNC --> GEO[Fast Geometry and FK]
        GEO --> TRACK[Object Tracking and Prediction]
        TRACK --> MAP[Incremental Local 3D Map]
        MAP --> SNAP[Scene Snapshot]
        SNAP --> ST
        SNAP --> TRIG[Confidence and Event Triggers]
        TRIG --> SEM[Semantic Perception Agent]
        SEM --> MAP
    end

    subgraph CONTROL[Control Plane - real time]
        EX --> CHUNK[Bounded Action Chunk]
        CHUNK --> SAFE[Safety Controller]
        SAFE --> ROBOT[Franka Robot or Simulator]
        ROBOT --> SENS
    end

    subgraph BACKEND[Backend and Compatibility Plane]
        CAPX[CAP-X Adapter]
        LIBERO[LIBERO-PRO Backend]
        ROBOSUITE[Robosuite Backend]
        BEHAVIOR[BEHAVIOR Backend]
        CAPX --> LIBERO
        CAPX --> ROBOSUITE
        CAPX --> BEHAVIOR
        LIBERO --> ROBOT
        ROBOSUITE --> ROBOT
        BEHAVIOR --> ROBOT
    end

    subgraph EVAL[Evaluation Plane - offline or evaluator only]
        TRACE[Normalized Episode Trace]
        METRIC[Metrics and Failure Analysis]
        ABL[Module Ablation Controller]
        GT[Privileged Evaluator]
        MON --> TRACE
        TRACE --> METRIC
        ABL -. configures .-> AGENT
        ABL -. configures .-> CONTRACT
        ABL -. configures .-> WORLD
        GT --> METRIC
        ROBOT --> GT
    end
~~~

## 2. Main data flow

~~~text
Task
  -> Mission Manager
  -> Subgoal Contract
  -> Policy Agent
  -> Action Contract(parent scene version)
  -> Validator
  -> Verifier
  -> Action Lease
  -> Typed Skill Executor
  -> Bounded Robot Action
  -> New Sensor Data
  -> New Scene Snapshot(version + 1)
  -> Postcondition Verification
       -> commit and advance
       -> or classify failure and recover
~~~

The physical robot is changed only through the Typed Skill Executor after validation, verification, and lease acquisition.

## 3. Layered module tree

~~~text
CAP-MAS
├── Interface Layer
│   ├── Task API
│   ├── Agent artifact API
│   └── Evaluation report API
│
├── Agent Plane
│   ├── Mission Manager
│   ├── Perception Agent
│   ├── Policy Agent
│   ├── Verifier Agent
│   ├── Recovery Agent
│   ├── Execution Monitor
│   ├── Critic and Memory Skill Evolver
│   └── Robot Skill Evolver
│
├── Contract and Runtime Plane
│   ├── Message envelope
│   ├── SceneSnapshot
│   ├── SubgoalContract
│   ├── ActionContract
│   ├── VerificationResult
│   ├── Failure taxonomy
│   ├── Versioned State Store
│   ├── Event Bus
│   ├── Runtime Scheduler
│   ├── Action Lease Manager
│   └── Checkpoint Manager
│
├── World Model Plane
│   ├── Sensor Synchronizer
│   ├── Forward Kinematics
│   ├── Fast Geometric Estimator
│   ├── Object Tracker
│   ├── Motion Predictor
│   ├── Incremental Local 3D Map
│   ├── Scene Snapshot Publisher
│   └── Semantic Perception Trigger
│
├── Skill and Execution Plane
│   ├── Typed Skill Schema
│   ├── Active Robot Skill Registry
│   ├── Quarantine Robot Skill Registry
│   ├── Active Memory Skill Registry
│   ├── Quarantine Memory Skill Registry
│   ├── Skill Validator
│   ├── Shadow Executor
│   ├── Typed Executor
│   ├── Safety Monitor
│   └── Execution Trace Recorder
│
├── Backend Plane
│   ├── CAP-X Legacy Adapter
│   ├── CAP-X Typed Adapter
│   ├── LIBERO-PRO Adapter
│   ├── Robosuite Adapter
│   └── BEHAVIOR Adapter
│
├── Memory Plane
│   ├── Episode Working Memory
│   ├── Experience Memory
│   ├── Semantic and Procedural Memory
│   ├── Memory Skill Bank
│   ├── Memory Controller and Executor
│   ├── Failure Case Store
│   ├── Cross-Task Priors
│   └── Persistent Role and Topology Cache
│
└── Evaluation Plane
    ├── CAP-X Baseline Runner
    ├── CAP-MAS Runner
    ├── Budget Matcher
    ├── Ablation Controller
    ├── Metric Aggregator
    ├── Failure Taxonomy Reports
    └── OOD and Regression Evaluator
~~~

## 4. Timing and process boundaries

~~~mermaid
flowchart LR
    subgraph RT[Real-time process]
        FK[Joint state and FK]
        GEOM[Geometry and local map]
        SAFETY[Safety controller]
        ACT[Robot action]
        FK --> GEOM --> SAFETY --> ACT
    end

    subgraph ASYNC[Asynchronous process]
        TRACK[Object tracker]
        SEM[Semantic perception]
        SNAP[Snapshot publisher]
        TRACK --> SNAP
        SEM --> SNAP
    end

    subgraph LLM[Agent process]
        PLAN[Manager and policy]
        VERIFY[Verifier]
        RECOVER[Recovery]
        PLAN --> VERIFY --> RECOVER
    end

    ACT --> FK
    GEOM --> SNAP
    SNAP --> PLAN
    SNAP --> VERIFY
    PLAN --> SAFETY
~~~

### Hard boundary

The real-time process must not import or wait for LLM clients, network APIs, semantic VLM inference, skill evolution, or experiment reporting.

### Soft boundary

The asynchronous process may publish newer snapshots while an agent is reasoning. Every agent proposal therefore carries the scene version it used. The runtime rejects stale proposals instead of silently applying them.

## 5. State and authority flow

| State or capability | Owner | Readers | Writers |
| --- | --- | --- | --- |
| Raw sensor frames | Sensor process | World model | Sensor driver |
| Scene snapshots | State store | Agents, verifier, controller | World model publisher |
| Subgoal graph | Mission Manager | Policy, recovery, evaluator | Mission Manager |
| Action contract | Policy and runtime | Validator, verifier, executor | Policy Agent |
| Actuator lease | Runtime | Executor, safety controller | Lease Manager |
| Physical robot state | Robot backend | World model, executor, evaluator | Robot controller |
| Active Robot Skill registry | Skill runtime | Policy, validator, executor | Robot Skill Promotion manager |
| Active Memory Skill registry | Memory runtime | Memory Controller, Executor | Memory Skill Promotion manager |
| Quarantine registries | Evolver runtimes | Shadow executors, evaluator | Critic and corresponding evolver |
| Privileged task success | Evaluator | Evaluation process | Simulator or task evaluator |

Privileged task success is intentionally outside the agent observation boundary.

## 6. Communication topology

The default communication graph is sparse:

~~~text
Scene Snapshot
      |
      v
Mission Manager -> MissionGraph -> Local Policy Agents -> GraphValidator -> Arbiter
      ^                                                               |          |
      |                                                               v          v
Recovery Agent <- Execution Monitor <- Typed Executor <- ActionContract <- Verifier
      ^
      |
Critic and Memory Skill Evolver
~~~

Agents communicate through typed artifacts, not an unrestricted shared conversation. Natural language may be included as an explanatory field, but it is never the authoritative state representation.

The typed artifact that connects the multi-agent roles is the `MissionGraph`.
The Manager owns the global graph; local Policy Agents own candidate
`SubgraphSpec` artifacts. `GraphValidator` performs static graph checks before a
candidate reaches the Arbiter, and only the selected candidate is lowered into
an `ActionContract` for Verifier approval. The graph is therefore a shared
workspace and an executable specification, not merely a visualization of the
conversation.

```text
MissionGraph
  -> parallel local SubgraphSpec candidates
  -> GraphValidator
  -> Arbiter
  -> ActionContract(parent_scene_version)
  -> Verifier -> ActionLease -> Executor
```

Parallel candidate generation is read-only. A single Executor owns the robot's
exclusive actuator lease; branches that share an exclusive resource are not
allowed to execute concurrently. GaP-style parallel rehearsal belongs to the
offline/asynchronous evolution plane and is not placed in the servo loop.

## 7. CAP-X comparison boundary

~~~text
CAP-X Legacy
  Task prompt -> single agent -> Python exec -> CAP-X API -> environment

CAP-X Typed
  Task prompt -> single policy -> typed CAP-X skills -> contract validator -> environment

CAP-MAS Contract
  Task prompt -> Manager -> MissionGraph -> local Policy Agents -> GraphValidator
  -> Arbiter -> Verifier -> typed skills
  -> lease-controlled executor -> environment
  -> asynchronous snapshots -> recovery and memory
~~~

The same CAP-X API functions should be wrapped where possible:

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

## 8. Ablation map

~~~mermaid
flowchart TB
    ROOT[Full CAP-MAS]
    ROOT --> A1[Coordination ablations]
    ROOT --> A2[Perception ablations]
    ROOT --> A3[Execution ablations]
    ROOT --> A4[Skill ablations]
    ROOT --> A5[Memory and topology ablations]

    A1 --> X1[CAP-X single agent]
    A1 --> X2[No version checks]
    A1 --> X3[No verifier]
    A1 --> X4[Natural-language messages]

    A2 --> Y1[CAP-X observation callback]
    A2 --> Y2[Synchronous semantic perception]
    A2 --> Y3[No object tracking]
    A2 --> Y4[Full-map rebuild]

    A3 --> Z1[CAP-X exec]
    A3 --> Z2[No action lease]
    A3 --> Z3[Unbounded action chunk]

    A4 --> K1[No skill evolution]
    A4 --> K2[CAP-X occurrence library]
    A4 --> K3[Quarantine without promotion]
    A4 --> K4[Safe-boundary activation]

    A5 --> Q1[Fixed graph]
    A5 --> Q2[Adaptive sparse graph]
    A5 --> Q3[No persistent memory]
~~~

Every ablation should change one architectural seam at a time. The CAP-X legacy baseline is a separate execution mode, not a code path hidden inside the full runtime.

## 9. Single-episode timeline

~~~text
t0  Reset episode and create episode epoch
t1  Publish initial SceneSnapshot(v0)
t2  Manager creates MissionGraph and subgoal budget
t3  Local Policy Agents propose SubgraphSpec candidates
t4  GraphValidator and Arbiter select a candidate
t5  Candidate lowers to ActionContract(parent=v0); Validator and Verifier approve
t6  Executor acquires lease and runs bounded action chunk
t7  World model publishes SceneSnapshot(v1)
t8  Monitor verifies observable postconditions

    success -> commit v1 and continue graph
    failure -> invalidate graph suffix, classify failure, recover from v1

t9  At checkpoint: optionally activate a shadow-validated Memory Skill candidate
t10 At episode end: archive trace, update memory, run Memory Skill promotion tests
t11 Later phase: freeze memory, then run Robot Skill promotion tests
~~~

## 10. What this structure is intended to prove

The central claim is not simply that more agents perform better. The intended test is that explicit contracts and controlled authority make performance degrade more slowly as task horizon increases, while keeping perception latency outside the control deadline and preserving fair CAP-X comparisons.
