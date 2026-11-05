# SACH Operational Workspace & Task Orchestration

Start with [`AGENTS.md`](../AGENTS.md). SACH uses evidence, verified tests, and calibrated task execution without bloated template baggage or mandatory phase switches.

## 1. Task Lifecycle & Calibration

- **Small Tasks:** Bounded edits, bug fixes, or minor enhancements. Keep intent, `<allowed_files>`, and verification directly in conversation or `.agents/TASK.md`. Zero directory overhead.
- **Big Tasks (Consequential / Architectural / Multi-slice):**
  - Initiated by `"big task: <goal>"` or complex feature requests.
  - Coordinate creation of task directory in `process/features/active/{task-slug}/`.
  - Initialize `task.md` using [`.agents/TASK.md.template`](../.agents/TASK.md.template) as working memory.
  - For multi-step task DAGs and EARS/BDD requirements, define machine contracts `task.json` and `spec.json` (schemas in [`.agents/schemas/`](../.agents/schemas/)).
  - For deep research or architectural decisions, leverage [research dossiers](development-protocols/research-harness-protocol.md) and [adversarial multi-expert debate](development-protocols/adversarial-debate-protocol.md).
  - When verified, all binary goals satisfied, and contracts validated (`python3 .agents/harness.py validate --task task.json --spec spec.json`), mark complete by moving the folder from `process/features/active/{task-slug}/` to `process/features/completed/{task-slug}/`.

## 2. Runtime Harness & Tooling (`.agents/`)

- Offline harness utilities live in [`.agents/harness.py`](../.agents/harness.py):
  - Validates contracts (`validate`).
  - Assembles bounded context packs (`context`).
  - Checks memory freshness against SHA256 hashes (`memory-check`).
  - Queries bounded relationship graphs (`graph-query`).
  - Generates multi-dimensional web research queries (`research-queries`).
  - Validates research dossiers and computes Evidence Confidence Index (`research-dossier`).
  - Evaluates multi-expert adversarial debate ledgers and Pareto synthesis (`debate-eval`).
- Machine contracts (JSON schemas) live in [`.agents/schemas/`](../.agents/schemas/):
  - `spec.schema.json`: EARS requirements and BDD scenarios.
  - `task.schema.json`: Directed Acyclic Graph (DAG) of slices, execution frontier, binary goals, and verifiers.
- The native loop guard hook lives in [`.agents/hooks/pre-tool-guard.py`](../.agents/hooks/pre-tool-guard.py).
- Package `sync` automatically registers loop hooks in `.claude/settings.json` when available.

Use the [protocol index](development-protocols/all-development-protocols.md) (including [adversarial-debate-protocol.md](development-protocols/adversarial-debate-protocol.md) and [research-harness-protocol.md](development-protocols/research-harness-protocol.md)) and [context router](context/all-context.md) on demand. Working style (`DELEGATED`/`PAIR`) does not expand tool permissions. Git worktrees isolate working files, while process/network/credential isolation belongs to the execution environment.
