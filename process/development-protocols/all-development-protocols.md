# Development Protocols Index (SACH-2026 Framework)

<protocols_index version="4.0" framework="SACH">

Start with [`AGENTS.md`](../../AGENTS.md) and the active task record or [`.agents/TASK.md`](../../.agents/TASK.md.template). Open only the specific protocol relevant to your active execution step:

| Need | Protocol | Focus |
|---|---|---|
| Context and Memory | [`context-memory-protocol.md`](context-memory-protocol.md) | Bounded retrieval, cache boundaries, source freshness and continuation |
| Runtime Harness | [`runtime-harness-protocol.md`](runtime-harness-protocol.md) | Contract validation, context packs, native hook registration and limits |
| Graph Engineering | [`graph-engineering-protocol.md`](graph-engineering-protocol.md) | Bounded neighborhoods, source validation and impact |
| Boomerang Orchestration & Subagent Delegation | [`orchestration.md`](orchestration.md) | Clean context windows, DAG execution frontier, Worker-Judge separation |
| Worktree Sandboxing & Isolation | [`worktree-isolation-protocol.md`](worktree-isolation-protocol.md) | Safe `.agent-worktrees/`, quarantine dirty trees, zero data loss |
| Task-Sized Verification & Log Compaction | [`harness-verification-protocol.md`](harness-verification-protocol.md) | Behavioral oracles, static checks, evidence and regression |
| Code Hygiene & Implementation Standards | [`implementation-standards.md`](implementation-standards.md) | Clean code, idiomatic repo patterns, type safety |
| LSP & MCP Tool Guidance | [`mcp-lsp-protocol.md`](mcp-lsp-protocol.md) | AST navigation, def-use slicing, tool selection |
| Tool Calling & Capability Contracts | [`tool-calling-protocol.md`](tool-calling-protocol.md) | State-aware execution, failure classification, mutation timeouts |
| Deep Investigation & Research Harness | [`research-harness-protocol.md`](research-harness-protocol.md) | Source classification, claim ledger, epistemic status, R&D exit contract |
| Targeted Review & Adversarial Challenge | [`adversarial-debate-protocol.md`](adversarial-debate-protocol.md) | Substantive counterexamples, trade-offs matrix, risk mitigations |
| Architectural Decisions | [`innovation-harness-protocol.md`](innovation-harness-protocol.md) | Repo-native choices, proportional trade-offs, falsification |
| Observability & Evaluation | [`observability-and-evals.md`](observability-and-evals.md) | Token accounting, latency tracking, trajectory evals |
| Subprocess Script Batching | [`code-mode-batching.md`](code-mode-batching.md) | Batch chatty tool calls to save token overhead |
| Progressive QA Selection | [`seven-layer-qa-protocol.md`](seven-layer-qa-protocol.md) | Progressive fail-fast verification pipeline |

</protocols_index>
