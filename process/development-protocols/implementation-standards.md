# Implementation Standards

<implementation_standards version="2.0">

Use the repository's language, architecture, formatter, and test conventions. Apply the rules below to the affected code; do not refactor unrelated code to satisfy a template ideal.

## Types and contracts

<strict_typing>
Preserve existing type and schema contracts. Add explicit types where the language and repository convention call for them, especially at public boundaries and when a change would otherwise hide an unsafe value. Do not add a broad lint or type suppression merely to make a check pass. If a narrow suppression is necessary, explain it beside the code and verify the behavior.
</strict_typing>

## Architecture

<clean_architecture>
Respect the dependency boundaries already established in the repository. Keep domain rules separate from infrastructure when that distinction exists and matters to this change. Introduce a port, adapter, interface, or dependency injection only when it solves a concrete coupling or testing problem. Simple utilities need no architecture framework.
</clean_architecture>

## Behavior and errors

- Validate untrusted input at its actual boundary. Preserve documented error behavior and avoid silently swallowing failures.
- Check affected edge cases such as empty input, limits, cancellation, and concurrent updates when relevant.
- Keep logs useful and free of secrets. Remove temporary debugging output introduced for the task.
- Keep edits scoped to the requested result. A regression test may update an existing test when its expectation is demonstrably wrong; never weaken an assertion simply to obtain a pass.
- Update dependencies or lockfiles only when the requested outcome requires it and the change is within authorized scope.

## Resource cleanup

<deterministic_teardown>
Close resources the changed code opens, including files, sockets, transactions, and temporary test state. Use the language's normal cleanup mechanism. Inspect the final diff for scratch files and unintended edits before completion.
</deterministic_teardown>

</implementation_standards>
