# Planning Context

<planning_context version="2.0">

## Outcome and slicing

Use INVEST as a check on a proposed unit of work, not a scorecard every task must fill: Independent, Negotiable, Valuable, Estimable, Small enough to review, and Testable. State the observable outcome, relevant invariants, expected files or boundaries, and a verifier before editing. If a material unknown remains, investigate it first.

<vertical_slicing>
For larger work, divide the plan by user visible behavior or independently verifiable engineering outcomes. A slice should leave the repository in a useful, checkable state. Record dependencies between slices and an integration check when parallel work is possible. A small task may need only one step; do not manufacture multiple slices to meet a quota.
</vertical_slicing>

## Scope and verification

Declare the files expected to change in the active task contract. Add a necessary companion file with a reason before editing it. Choose focused checks based on affected behavior and impact; broaden when interfaces, data, or security boundaries change. Review the actual diff and verifier results after implementation.

<rollback_strategy>
For consequential changes, identify how to recover from a failed slice before starting it: revert a known agent-owned hunk, restore from a known checkpoint, disable a feature flag, or use a tested migration reversal as appropriate. Preserve pre-existing user and other agent edits. Do not run git checkout, git restore, reset, stash, or deletion across files whose ownership is uncertain. If a check fails repeatedly, preserve evidence, diagnose, and choose a safe next step; an attempt count alone does not authorize rollback.
</rollback_strategy>

Estimate effort or use story points only if the team uses them. Local historical data is more useful than a universal hours-per-point rule.

</planning_context>
