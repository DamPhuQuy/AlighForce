# Graph Retrieval and Change Impact

## Choose the cheapest sufficient representation

Begin with lexical search and repository conventions. Use LSP/AST definitions and direct imports when symbol identity matters. Use a repository graph when relationships span files or repeated searches fail. Data-flow or taint questions may require a real CFG/PDG/CPG tool; a call graph is not a substitute. Do not claim an ARISE, LocAgent or Joern capability merely because its name appears in a protocol.

Separate the **code graph** (definitions, references, imports, inheritance) from the **task DAG** (dependencies and integration order). A task DAG must be acyclic; investigation/replan loops belong to a bounded execution state machine, not cyclic task dependencies.

## Bounded retrieval

Identify a small set of exact seed symbols or file paths from the requirement/failure. Prefer definitions and direct imports first, then expand callers/callees one hop; use two hops only if needed. Limit the result by nodes and output budget. Request source previews before full bodies. Stop expanding once the root-cause hypothesis, affected consumers and verifier can be grounded.

Use qualified symbol IDs (path, enclosing scope, name) to avoid conflating common names. Respect edge direction/type when tracing impact; undirected proximity is only a navigation hint. Labels inferred by an LLM must remain distinguishable from AST/LSP-extracted edges and runtime observations.

If a graphify-compatible `graph.json` already exists, the portable helper supports:

```bash
python3 .agents/harness.py graph-query --input graphify-out/graph.json --term verifyGate --hops 1 --max-nodes 20
```

It uses case-insensitive label terms, at most three seeds, undirected bounded neighborhood search, 0–2 hops and 1–50 nodes. `--max-chars` bounds the JSON response (default 12,000 characters); truncation and ignored dangling edges are reported. It returns bounded metadata and edges, not raw code or instructions. This lightweight adapter does not generate AST/CPG graphs, perform semantic embeddings, rank by PageRank or prove a data-flow path. Empty matches are reported as empty results. Expand terms from the actual graph vocabulary if necessary; never invent a graph relation to fill a gap.

## Freshness and evidence

Before using a result, compare its source path/line with the current checkout and inspect changed files. Record index revision and extractor version when the upstream index exposes them. Dirty worktree edits can invalidate a graph built at the same HEAD. Refresh affected nodes/import edges using the actual indexer when supported; otherwise fall back to current source/LSP. The helper explicitly labels freshness unverified.

For a proposed mutation, inspect upstream definitions and downstream consumers; select tests covering that impact. Record source locations and confidence in the task. Graph proximity is a candidate list, not proof of causation or coverage. Generated code, dynamic imports, reflection and cross-language boundaries may be missing.

Measure usefulness by correct localization, source-backed conclusions, downstream test success, query latency and tokens spent. Enable graph tooling only when its end-to-end benefit exceeds indexing/retrieval overhead on representative project tasks.
