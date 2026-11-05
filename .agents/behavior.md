# Agent Behavior Protocol (SACH-2026 Framework)

## Task routing

Use `DELEGATED` by default: investigate, implement, verify and report without routine mode switches. `PAIR` honors checkpoints requested by the user. Neither label authorizes new tools or external effects.

| Task | Minimum sufficient process |
|---|---|
| Small reversible edit | Intent, intended files, relevant verification; keep the record in the conversation if no handoff is needed |
| Small task / Bug fix | Intent and scope in `.agents/TASK.md` or conversation, observable failure reproducer, passing patch; direct verification |
| Feature or refactor | Acceptance and invariants first, coherent slices, focused and integration checks |
| Large or consequential change (Big Task) | `process/features/active/{task-slug}/task.md` (from `.agents/TASK.md.template`), machine contracts `task.json` / `spec.json` for DAG slices and EARS/BDD; on completion move folder to `process/features/completed/{task-slug}/` |

Do not impose a failing runtime test on pure documentation changes. Do not create a new specification, DAG, matrix or independent review merely to satisfy a template. A concise decision with evidence is sufficient when alternatives are obvious.

## Working state and ownership

Task artifacts (`.agents/TASK.md` or `process/features/active/{task-slug}/task.md`) contain intent, scope under `<allowed_files>`, decisions, evidence references, failed hypotheses and next actions. `.agents/TASK.md.template` serves as unified scratchpad guidance. A completed checkbox is a claim until linked to concrete verifier evidence.

When automation needs structured state, `task.json` (or `.agents/task.json`) owns machine status/dependencies/goals; `spec.json` (or `.agents/spec.json`) owns requirements and scenarios. The markdown references these files instead of copying their status fields. Validate them with `python3 .agents/harness.py validate`. Never treat defaults in a JSON Schema as values already written to a document.

## Delegation when it pays for itself

Start with one agent. Delegate work only when four conditions are satisfied: (1) the sub-task is structurally independent; (2) the host exposes authentic process or context isolation; (3) the estimated token and time budget accommodates child startup; and (4) the expected quality gain exceeds coordination and integration overhead. Never invent an unavailable subagent capability.

When conducting complex research or architectural planning, challenge proposals with targeted review lenses (such as security boundaries, resilience under partial failure, or performance trade-offs) tailored to the actual uncertainties of the task. Rely on falsifiable counterexamples and testable edge cases rather than arbitrary persona quotas or narrative claims of Pareto optimality. Follow [adversarial-debate-protocol.md](../process/development-protocols/adversarial-debate-protocol.md).

Use fresh-context children where supported. Supply the requirement, necessary source pointers, allowed files, verifier, branch/base revision, worktree path, resource limits and return contract. Report whether the host actually omitted parent history. Avoid nested delegation; cap concurrent writers at three or the smaller host limit. Do not grant overlapping write ownership on the integration checkout.

Return a bounded result: changed paths/commit or patch reference, base revision, actual verifier results, remaining uncertainty and acceptance delta. A diff stat alone is not an integrable result. Preserve raw evidence on disk; a compact return does not mean logs were deleted by the runtime.

Use separate worktrees for concurrent writers or disruptive experiments. Worktrees share repository metadata and may share credentials, services, ports and caches; use a container/VM and host permissions when security isolation is required. Test the integrated result after applying a child's change.

## Independent review

For consequential behavior, use an independent reviewer if available, supplied with acceptance, raw diff, environment and test evidence rather than the Worker's persuasive explanation. Ask it to falsify the result with concrete counterexamples. Independence reduces shared context bias; it does not eliminate error or replace execution.

Bound review to two rounds by default. Resolve concrete findings or document remaining disagreement; escalate only if a material decision cannot be made within the user's authorization. If no independent agent exists, perform a separate evidence-based review and state that limitation.

## Context and memory

Follow [context-memory-protocol.md](../process/development-protocols/context-memory-protocol.md). Keep the entry point small, retrieve source just in time, bound new output before ingestion, and checkpoint at coherent boundaries. Do not attempt to edit hidden system prompts, erase past tool messages or terminate/resume sessions through capabilities the host does not expose.
