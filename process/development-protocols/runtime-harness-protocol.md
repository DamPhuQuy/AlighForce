# Portable Runtime Harness

## Capability boundaries

`.agents/harness.py` requires Python 3.10+ standard library and runs offline. It prints JSON, returns nonzero for invalid input, never executes a command from a spec, and does not call a model or change live context. The separate loop hook writes only bounded session state. Host tool permissions remain authoritative.

| Component | Executable behavior | Limit |
|---|---|---|
| `validate` | Contract structure, nonempty clauses, scenario requirement references/coverage, task references/cycles/frontier, scope path hygiene and completion fields | Checks the bundled schema vocabulary and consistency; does not authenticate logs or run acceptance tests |
| `context` | Static block preservation, reserved capacity, required entries and priority-based optional selection | Offline content assembly using UTF-8 bytes as a conservative token bound; provider framing is excluded |
| `memory-check` | Source-file hash and expiry checks | Freshness, not semantic truth or injection immunity |
| `graph-query` | Bounded graphify-compatible neighborhood metadata | Existing graph required; no indexing or automatic freshness proof |
| `research-dossier` | Structural validation of dossier v2, URL/DOI syntax, source-claim linkage; `--strict` checks execution readiness | Structural consistency only; does not browse the web, verify remote facts, or confirm peer review status |
| `debate-eval` | Review ledger validation, substantive counterexamples, trade-offs alignment; rejects empty attacks `{}`; `--strict` enforces v2 | Offline consistency check; does not prove model independence or guarantee Pareto optimality |
| `capability-check` | Capability catalog schema validation, triggers/non-triggers, side-effect profiles, idempotency flags | Offline catalog contract check; does not execute tools, grant host permissions, or inspect live environment |
| Loop hook | Native pre/post event handling and failed-trajectory deduplication | Installed adapters only; not a shell/security sandbox or a full cognitive monitor |

## Machine contracts for substantial tasks

Use the schemas in `.agents/schemas/` only when machine validation benefits the task. Keep task status in task JSON and narrative decisions in `TASK.md`; avoid duplicate status records. A minimal valid pair:

```json
{
  "spec_id": "SPEC-EXAMPLE", "title": "Example behavior",
  "ears_requirements": [{"id":"REQ-1","pattern":"event_driven","trigger":"valid request","system_response":"return a result","full_statement":"WHEN a valid request arrives THE SYSTEM SHALL return a result"}],
  "bdd_scenarios": [{"id":"SCN-1","name":"valid request","given":["service ready"],"when":"valid request arrives","then":["result returned"],"requirement_ids":["REQ-1"]}]
}
```

```json
{
  "task_id":"TASK-EXAMPLE", "title":"Implement example", "status":"READY",
  "spec_ref":"SPEC-EXAMPLE", "dependencies":[], "allowed_files":["src/example.ts","test/example.test.ts",".agents/TASK.md",".agents/task.json"],
  "binary_goals":[{"id":"GOAL-1","description":"acceptance test passes","passed":false}],
  "verifier":{"command":"npm test -- test/example.test.ts","timeout_seconds":30}
}
```

Run `python3 .agents/harness.py validate --spec .agents/spec.json --task .agents/task.json`. Repeat `--task`/`--spec` for the complete dependency set. Unknown dependencies, cycles, empty criteria and unlinked requirements fail. A completed task needs all goals passed and `verifier.evidence_ref`; the reference itself remains a claim until inspected. Optional-feature selection is represented in the clause precondition. EARS templates and BDD are semi-formal requirements; use a dedicated proof/model-checking tool for actual formal verification.

## Claude Code hook adapter

Run the package's `sync` command after scaffolding to register PreToolUse, PostToolUse and PostToolUseFailure in `.claude/settings.json` if the guard script exists. Sync preserves other settings and existing hooks, adds its own command once, and refuses malformed settings. It does not enable hooks for other IDEs. Review installed settings and confirm Python is available; do not assume every host emits identical shell response fields.

The command reads native JSON stdin using `tool_name`, `tool_input`, `cwd`, `session_id`, `hook_event_name`, and post-event outcome fields. Allow paths return `{}` so normal permissions apply. Denials use `hookSpecificOutput.permissionDecision="deny"`; pre-event infrastructure errors exit 2 with stderr. No `allow` permission override is emitted.

State lives under `.agents/state/guard-<session-and-cwd-hash>.json`; exclude `.agents/state/` and `.agents/logs/` from version control. Updates use an OS file lock and atomic replace. SHA-256 identifies exact operational input, not intent. Known successful polling is allowed. Failed retries after source changes are reconsidered via a revision/status/stat signature. Session state stores hashes and flags, not raw tool parameters or private outputs. A failure event stores the last ten completed events; duplicate post notifications with a tool-use ID are ignored.

To diagnose, inspect the stderr and relevant state. After resolving corruption or a stale session, reset only the selected state: `python3 .agents/hooks/pre-tool-guard.py --state-file <actual-session-file> --reset`. Do not reset repeatedly to repeat a failed hypothesis. Post-event errors are diagnostic: an already run tool cannot be retroactively blocked.

## Scope verification and legacy compatibility

`verify-gate G2`/`G3` discover task files in `process/features/active/`, `process/general-plans/active/`, or `.agents/TASK.md`; pass an explicit file when needed. G2 checks declared scope/DAG tags; G3 audits current Git changes. These commands do not run BDD tests or certify requirements. Keep task/state files within declared scope. G0/G1 remain for legacy artifacts and are not mandatory phases of SACH.

The portable policy's `DELEGATED` write mode and `OBSERVE` read mode describe an advisory capability decision. Legacy phase labels are accepted through explicit aliases. Pair checkpoints are a conversation preference, not permission to mutate. External tools bypassing the package are governed by the host, not by this policy manifest.

If Git/revision state is unavailable (including a repository without an initial commit), the hook records outcomes without denying equivalent retries based on an unknown workspace. A trusted host adapter may supply an explicit `progress_key`. Diagnostic limitations never authorize an action; normal host permissions still apply.
