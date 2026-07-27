# ADR-0005: Dedicated LLM Backend Boundary

## Status

Accepted for the initial research prototype.

## Context

CAP-MAS must compare its multi-agent architecture with CAP-X under the same model and API budget. Agent reasoning also needs to be replaceable with local, mock, and replay backends without changing robot execution.

## Decision

All deliberative Agent inference goes through LLMBackend. Robot actions go through RobotBackend and the contract runtime. LLMBackend returns raw text, parsed artifacts, schema validity, usage, latency, and errors.

The control plane cannot call LLMBackend. A missed Agent deadline produces an explicit timeout artifact and invokes a fallback or recovery policy.

## Consequences

Positive: model/provider changes are isolated; CAP-X parity, usage accounting, replay, and latency ablations are possible.

Negative: each Agent needs schema-aware parsing and error handling. The project must maintain both model-call traces and runtime traces.

## Alternatives rejected

- Letting each Agent call a provider SDK directly.
- Allowing an LLM response to call RobotBackend directly.
- Treating malformed output as FINISH.
