# Progressive Quality Checks

Select the layers that address the change's risk. This is a menu, not seven mandatory ceremonies. Resolve a failure before relying on a higher-layer result that depends on it.

| Layer | Evidence | When useful |
|---|---|---|
| L1 Syntax and types | Parse, lint, typecheck | Code/config changes |
| L2 Focused behavior | Reproducer or acceptance check | Changed observable behavior |
| L3 Boundary cases | Invalid input, empty state, invariants | Contract-sensitive behavior |
| L4 Integration | Module/service/client interactions | Shared dependencies/interfaces |
| L5 User journey | Real UI or end-to-end exercise | User-facing flows |
| L6 Scope and security | Diff, permissions, secrets/trust boundaries; `verify-gate G3` | Consequential changes |
| L7 Performance and regression | Representative workload, affected/full suite | Broad impact or performance claims |

For bug fixes, confirm the reported symptom on baseline and its absence on candidate. Features need acceptance defined first; documentation/mechanical edits use appropriate static checks. Keep environment failures separate from behavioral failures. Never alter an assertion just to obtain a passing result. Record what ran, what passed and what remains unverified.
