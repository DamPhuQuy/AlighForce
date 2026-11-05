# Agent Control Layer

Start with `AGENTS.md`; load these files only for the active need. Project-specific memory is owned and reviewed by the repository.

| File | Purpose |
|---|---|
| [`behavior.md`](behavior.md) | Task calibration, state ownership, delegation and review |
| [`guardrails.md`](guardrails.md) | Scope, failed-trajectory recovery and completion evidence |
| [`conventions/naming.md`](conventions/naming.md) | Naming and structural hygiene |
| [`conventions/repo-learnings.md`](conventions/repo-learnings.md) | Source-backed reusable lessons and invalidation |

The executable helpers (harness.py, hooks/, schemas/) live directly in `.agents/`; engineering procedures live in `process/development-protocols/`. Runtime capabilities and limitations are documented in [runtime-harness-protocol.md](../process/development-protocols/runtime-harness-protocol.md). These instructions do not themselves intercept tools or configure model context.
