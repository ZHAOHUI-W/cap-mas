Status: ready-for-agent
Type: task
Labels: ready-for-agent

# Implement staged Multi-Policy experiment observability

Implement P3.1a from `spec.md`: normalized LLM call traces, run configuration,
and artifact-level metrics. Preserve the current strict structured-output
execution behavior and do not change physical scheduling.

## Acceptance Criteria

- Each successful LLM request emits exactly one `LLMCallTrace`.
- A schema compatibility fallback is visible as one completed trace with
  `structured_output_fallback=true` and a request-attempt count greater than
  one.
- A terminal transport/provider failure emits a failed trace with status code
  and sanitized error type/message.
- The collector is safe for concurrent Policy calls and returns deterministic
  ordering.
- The LIBERO LLM artifact records `run_config` and `llm_calls` without secrets.
- Existing 105 tests remain green and the client transport tests still cover
  strict request format and fallback behavior.
