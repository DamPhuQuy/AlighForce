# Code-Mode Tool Batching

<code_mode_protocol version="2.0">

Batch calls when it reduces avoidable round trips or keeps large intermediate results out of context. A direct call is usually clearer for one small action. Independent read-only calls may run in parallel with individual error boundaries and bounded output; keep dependent operations and mutations strictly sequential so each postcondition can be verified before the next step. Follow [tool-calling-protocol.md](tool-calling-protocol.md) for stateful mutation checks and timeout recovery.

Useful cases include bounded polling, paginated reads, or filtering structured results. In a script or code-mode tool:

- Set a timeout or retry bound and retain the meaningful failure status.
- Return the fields needed for the decision, plus enough provenance to inspect a surprising result. Avoid dumping unbounded logs or silently discarding errors.
- Keep destructive, external, or privileged actions subject to the same authorization and scope checks as direct tool calls. Batching does not bypass tool permissions or approval boundaries.
- Avoid creating a temporary script when the native tool, a filter, or a short command is simpler. If a scratch file is needed, use an authorized location and remove only files created for this task when safe.

Do not assume a fixed token or latency saving. Compare actual calls, tokens, time, and errors only when profiling is requested under [`observability-and-evals.md`](observability-and-evals.md). During review, investigate batching only if a real tool-output or latency problem occurred.

</code_mode_protocol>
