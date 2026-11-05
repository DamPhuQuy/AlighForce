# Research and Investigation (SACH-2026 Framework)

<protocol name="research-harness" version="4.0" framework="SACH">

Start from the requested outcome and an observable question. Investigate read-only until there is enough evidence to choose the next authorized change; authoring a scoped reproduction test is allowed. Do not make Research a mandatory mode or prohibit useful early hypotheses.

## 1. Internal Codebase Investigation

1. **Locate entry points**: Use lexical and symbol search, then inspect exact definitions, imports, and related tests.
2. **Def-Use Slicing**: Trace the failure backward to its inputs/assignments and the candidate change forward to consumers. Use LSP/AST or an existing graph when available; distinguish a static approximation from a measured runtime path.
3. **Falsifiable hypothesis**: For a bug, run a reproducer whose failure matches the reported symptom; dependency/install failures are separate environment problems. For features, define acceptance before implementation. For docs, use relevant source and consistency checks.
4. **Bounded context**: Stop expanding context when the uncertainty needed for the next step is resolved. Record facts with source locations/results and keep assumptions explicit. An unsuccessful hypothesis narrows the search; do not erase it from the continuation record.

## 2. Research Evidence Classification & Claim Ledger

When internal context is insufficient, conduct structured research. Never infer publication authority or peer review merely from a domain name (such as arXiv). Classify each source by its actual provenance and ground claims with precise references:

### Source Categories & Publication Status

- **`spec_standard` (Official Standards & Specs)**: Language specifications, RFCs, W3C, ISO, or formal protocols. Canonical for syntactic and behavioral contracts.
- **`peer_reviewed_paper` (Formally Refereed Research)**: Papers published in peer-reviewed conferences or journals (ACM, IEEE, USENIX). *Preprints (e.g. arXiv) are unrefereed; mark `publication_status: "preprint"` unless a peer-reviewed venue is cited.*
- **`vendor_docs` (Official Vendor Documentation)**: Primary documentation, changelogs, migration notes from runtime or library maintainers.
- **`industry_benchmark` (Empirical Industry Studies)**: Documented, repeatable architectural analyses from high-scale engineering organizations.
- **`community_anecdotal` (Community Discussions)**: Developer forums, blog posts, and tutorials. Treat as hypotheses requiring local empirical confirmation.
- **`local_empirical` (Workspace Reproductions)**: Falsifiable reproducers, isolated benchmarks, or spikes executed directly in the repository environment.

### Claim Ledger & Epistemic Status

Every technical claim in `research-dossier.json` must be tracked in the claim ledger with an explicit epistemic status:
- **`observed`**: Directly reproduced or verified in the local environment.
- **`reported`**: Stated by an authoritative source (cites source ID and section/location).
- **`inferred`**: Deduced logically from established facts; subject to verification.
- **`unverified`**: Unproven hypothesis; must not serve as the sole justification for architectural change.

### R&D Exit Contract

Research is complete when:
1. The **decision question** is answered with justified rationale.
2. Unresolved uncertainties are bounded and will not block execution.
3. Every adoption or rejection references documented claims.
4. The **execution readiness** block declares `ready: true` and links required evidence references.
5. Structural validation passes via `python3 .agents/harness.py research-dossier --input <path> --strict`.

## 3. Harness Tooling & Machine Validation

Generate targeted search queries and validate research dossiers using the offline harness:

```bash
# 1. Generate multi-dimensional search queries for primary docs, benchmarks, modern tech, and CVEs
python3 .agents/harness.py research-queries --topic "distributed lock redis lease" --domain software

# 2. Validate research dossier v2 structurally and check execution readiness
python3 .agents/harness.py research-dossier --input process/features/active/<slug>/research-dossier.json --strict
```

Follow [graph retrieval](graph-engineering-protocol.md) for bounded relationships, [adversarial debate](adversarial-debate-protocol.md) for multi-expert stress testing, and [verification](harness-verification-protocol.md) for empirical evidence.

</protocol>
