# Tool Calling & Capability Execution Protocol (SACH-2026 Framework)

<protocol name="tool-calling" version="1.0" framework="SACH">

Agents select and execute tools based on observable contracts, declared capabilities, and side-effect profiles. Never assume a tool is available, idempotent, or authorized without verifying runtime capability metadata.

## 1. Capability Discovery & Routing Lifecycle

The execution lifecycle follows seven explicit stages:

```
[1. Discover] ──► [2. Check Authority] ──► [3. Select / Abstain] ──► [4. Validate Input]
                                                                              │
                                                                              ▼
[7. Verify Postcondition] ◄── [6. Classify Outcome] ◄── [5. Host Execution]
```

1. **Discover**: Inspect runtime capability declarations or `.agents/capabilities.json`. Match intent against declared triggers and non-triggers.
2. **Check Authority & Availability**: Verify tool is registered with the host and permitted in the current security boundary. If unavailable, do not retry blindly—switch immediately to the declared fallback.
3. **Select or Abstain**: If a direct source read or native command is sufficient, **abstain** from tool invocation to conserve context and latency.
4. **Validate Arguments**: Validate input types, bounds, and preconditions before dispatch.
5. **Host Execution**: Execute strictly through authorized host APIs. Never simulate tool output or bypass host permission prompts.
6. **Classify Outcome**: Map runtime response to one of seven deterministic outcome categories (see Section 2).
7. **Verify Postcondition**: Confirm expected state changes on the filesystem or service. A successful tool exit code does not substitute for postcondition verification.

## 2. Deterministic Outcome Classification

Every tool call returns an outcome classified into one of seven categories:

| Category | Definition | Recovery Policy |
|---|---|---|
| `success` | Postconditions verified; payload meets schema | Continue execution; record evidence reference |
| `validation_error` | Input arguments violated constraints or schema | Correct arguments locally; do not repeat same payload |
| `permission_denied` | Host adapter or user denied permission | **Abstain immediately**; do not attempt evasion or repeated prompts |
| `stale_schema` | Tool schema or API contract changed | Refresh capability metadata via `capability-check` before retrying |
| `transient_network` | Connection reset or socket timeout prior to write | Retry with exponential backoff up to retry limit |
| `semantic_failure` | Target not found or search returned zero matches | Re-evaluate hypothesis; change search terms or approach |
| `unknown_mutation_timeout` | Timeout after mutation command dispatched | **STOP & VERIFY STATE**. Check git diff, filesystem, or service state before retrying. **Never assume writes are idempotent.** |

## 3. Stateful Mutations & Timeout Guard

Writing files, applying patches, or calling stateful APIs carries non-idempotent risks:
- **Precondition check**: Verify file exists, git worktree is clean, or prerequisite record is present.
- **Postcondition check**: Read back written file, inspect diff, or verify service status.
- **Timeout after write**: If a timeout occurs during a write or execution step, do **not** re-issue the write. First inspect `git status`, check file timestamps, or query status endpoints to determine whether the mutation occurred before the timeout severed the connection.

## 4. Read-Only Batching vs. Sequential Mutations

- **Independent Reads**: Subprocess batching (e.g. multiple symbol lookups or file reads) is permitted only for read-only operations with bounded outputs and individual error boundaries.
- **Stateful Writes**: Must remain strictly sequential with verified postconditions between calls. Never batch mutations where subsequent operations depend on the unverified success of earlier mutations.

## 5. Machine Validation

Validate workspace capability manifests against the catalog schema:

```bash
python3 .agents/harness.py capability-check --input .agents/capabilities.json
```

</protocol>
