# Delegation and Integration (SACH-2026 Framework)

Use one agent unless independent slices or consequential architectural decisions justify coordination overhead. Delegation requires actual host support and authorization; an instruction file cannot create agents, discard their history or enforce write ownership by itself.

## Targeted Review and Task Delegation

Start with a single agent. Delegate work only when four criteria hold: (1) the sub-task is structurally independent; (2) the host exposes authentic process or context isolation; (3) the estimated token/time budget accommodates child startup; and (4) the expected quality gain exceeds coordination and integration overhead.

For consequential research or architectural choices, submit proposals to targeted adversarial challenge lenses (security boundaries, failure recovery, scalability) with concrete counterexamples rather than decorative persona quotas. Follow [adversarial-debate-protocol.md](adversarial-debate-protocol.md).

## Dependency frontier

Represent substantial work as tasks with stable IDs, explicit dependencies, allowed files, verifier and failure policy. Validate that every parent exists and the graph is acyclic. A pending/ready task is dispatchable only when all parents have completed with relevant evidence. Running, failed, blocked and completed tasks are not automatically returned to READY. Retry requires an explicit replan; never silently resurrect a pruned task.

Use `.agents/harness.py validate` for consistency, not as a scheduler. The host/coordinator owns dispatch, resource limits, result acceptance and integration. Investigation loops may be cyclic; task prerequisites must not be.

## Boomerang handoff

Spawn fresh-context children when supported. Pass acceptance/spec, source pointers, declared files, baseline revision, worktree/branch, verifier, time/tool limits and return format. Do not pass the parent's failed reasoning transcript. State which assumptions require revalidation.

A child returns a patch/commit reference and base revision, changed paths, actual verifier command/result, unresolved risks and acceptance changes. Keep narrative concise (a 1,500-token target is a heuristic). Retain evidence artifacts for inspection. The parent verifies and integrates the change, then tests the integrated state. A diff stat and self-reported success alone are insufficient.

## Resource and ownership limits

Default to no nested delegation and at most three concurrent children, reduced to host capacity. Separate write ownership or use distinct worktrees; changes in independent worktrees may touch the same logical file only with an explicit integration owner and conflict plan. Read-only tasks need no worktree unless the host requires one. Isolate ports, temporary directories, test databases and mutable caches where shared state could collide. Never treat a worktree as a security sandbox.

## Review and failure

For consequential changes, give an independent reviewer acceptance, raw diff, environment and evidence, then ask for concrete counterexamples. Limit exchange to two rounds unless new evidence justifies more. A reviewer can be wrong; resolve findings against source and execution.

After failure, preserve evidence and choose REPLAN (new hypothesis/smaller slice), PRUNE (optional branch), or ESCALATE (material unresolved decision/authorization). Keep independent useful work moving. Checkpoint before context handoff using [context and memory](context-memory-protocol.md). Follow [worktree hygiene](worktree-isolation-protocol.md) before cleanup.
