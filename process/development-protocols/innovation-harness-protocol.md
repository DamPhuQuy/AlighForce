# Design Decisions and Trade-offs

Prefer the repository's existing abstractions and utilities when they solve the problem. Identify constraints and a falsifiable failure case before committing to an unfamiliar architecture or dependency. A minimal fix may be the only sensible option; do not manufacture alternatives.

For a consequential decision, compare feasible approaches by correctness, compatibility, complexity, security, performance, verification and recovery. Connect the chosen R&D decision directly to the immediate execution slice: declare the exact scope under `<allowed_files>`, specify the automated verifier command, and outline the recovery/rollback plan if empirical verification fails. A short explanation in `TASK.md` is sufficient when clear; subjective score matrices do not replace executable proof.

Challenge the favored approach: which assumption is unverified, what boundary or race can break it, which consumers change, and how can agent-owned work be recovered safely? Run a small authorized experiment when evidence resolves uncertainty more cheaply than debate.

Record the chosen approach, supporting evidence, rejected options and conditions that would justify reconsideration. Replan when an assumption is falsified. Do not reintroduce mandatory Innovation/Research phase gates or routine human confirmation through a secondary protocol.
