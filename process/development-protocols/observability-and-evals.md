# Observability and Evaluation

## Routine evidence

For normal tasks, keep relevant verifier results and a short continuation record. Do not require profiling files, cost estimates or seven QA layers for every edit. Log sanitized references rather than private outputs or credentials.

For substantial changes, capture acceptance result, regression result, scope violations, failed retries, and tool outcome classifications (success, validation error, permission denied, stale schema, mutation timeout) following [tool-calling-protocol.md](tool-calling-protocol.md). When the host exposes usage, record input/output tokens, cached input, model/version and cost units. If unavailable, mark unknown; UTF-8 byte estimates from the offline packer are not measured billed tokens or cache hit rates.

## Evaluate strength and efficiency together

Use paired baseline/candidate tasks at the same repository revisions with fixed model/version, tool permissions, verifier, time/token limits and environment. Include small docs/config edits, bug fixes, cross-file features, stale memory, graph staleness, noisy tool output, repeated failures, successful polling, context handoff and integration conflicts. Include multiple repositories and random seeds when measuring model behavior.

The primary outcome is independently checked acceptance with no new regressions or unauthorized effects. Also report false loop denials, escaped loops, continuation failures, source-backed localization, total wall time, p50/p95 latency, tool calls, tokens/cost, human interruptions and artifact overhead. Treat quality and safety as constraints before optimizing speed. Report failures and variability, not only the best run.

A useful ablation compares the small kernel alone, then context/memory, graph, loop hook, and independent review. Do not attribute a combined improvement to each component. Graph indexing, worker startup, reviewer calls, cache writes and log compaction all count toward end-to-end cost. Use thresholds from the project's workload rather than importing paper percentages as guarantees.

## What the packaged checks establish

The package's Node suite exercises scaffold/update, scope/policy, IDE sync and worktree behavior. `npm run test:swe-runtime` exercises deterministic Python utilities and native hook trajectories, including corrupt state, concurrent outcomes and changed workspaces. `npm run eval:harness` evaluates offline policy fixtures. These are engineering regressions, not a SWE-bench run or evidence of a model success-rate improvement.

Performance claims require before/after measurements on representative tasks. For changes limited to instructions, compare against the user's prior instructions and include documentation/token overhead. Keep a candidate only if acceptance holds and its measured trade-off is useful; no instruction framework is universally optimal.
