# Targeted Adversarial Review & Challenge Protocol (SACH-2026 Framework)

<protocol name="adversarial-debate" version="4.1" framework="SACH">

Single-agent reasoning on complex or architectural tasks is vulnerable to confirmation bias and missed edge cases. Rather than spawning a fixed quorum of decorative personas, conduct targeted adversarial review focused directly on the technical uncertainties and failure modes of the task.

## 1. Targeted Review Calibration

Review effort scales with task uncertainty and irreversible impact, not rigid persona formulas:

- **Routine / Localized edits**: Self-check against acceptance criteria and reproduction tests; no multi-expert ledger required.
- **Consequential / Architectural changes**: Identify specific threat surfaces (e.g. security boundaries, partial failures, schema drift, race conditions) and submit them to targeted falsification challenges.

## 2. Review Lens Mandates & Substantive Challenges

Every recorded challenge must provide substantive, falsifiable counterexamples. Polite agreement or empty attack objects are rejected:

1. **Security & Authorization**: Probes input sanitization, privilege escalation, credential leakage, and boundary enforcement.
2. **Resilience & State Recovery**: Probes crash recovery, timeouts, partial mutations, network partitions, and idempotency.
3. **Performance & Scalability**: Probes latency bounds, algorithmic complexity, lock contention, and memory ceilings.
4. **Simplicity & Maintainability**: Probes accidental complexity, unwarranted dependencies, and high refactoring cost.
5. **Red Team Challenger**: Actively seeks unstated assumptions, boundary off-by-ones, and worst-case timing races.

Each challenge in `debate-ledger.json` requires:
- `target`: The specific design element, assumption, or interface under test.
- `counterexample`: A concrete input, scenario, or sequence demonstrating the flaw.
- `expected`: The specified or safe system behavior.
- `observed_or_predicted`: The observed or modeled failure symptom.
- `evidence_ref`: Reference to source, reproducible test, or authoritative specification.
- `disposition`: `accepted_mitigated` (with mitigation plan), `rejected_with_evidence`, or `pending`.

## 3. Trade-off Synthesis and Mitigation Plan

Non-trivial architectural choices involve inherent trade-offs:
1. **Trade-offs Matrix**: Compare candidate approaches against concrete requirements (correctness, operational overhead, simplicity, latency).
2. **Justified Selection**: Record why the selected approach is appropriate for the context.
3. **Mitigation Plan**: Explicitly address the negative consequences of the selected approach.

## 4. Machine Validation

Validate the debate ledger offline with the runtime harness:

```bash
# Validate debate ledger v2 with substantive counterexamples and trade-off alignment
python3 .agents/harness.py debate-eval --input process/features/active/<slug>/debate-ledger.json --strict
```

A valid debate ledger v2 requires:
- Non-empty challenge records with substantive counterexamples (empty `{}` objects are rejected).
- Unique reviewer roles without duplicates.
- Selected approach matches an evaluated candidate in the trade-offs matrix.

</protocol>
