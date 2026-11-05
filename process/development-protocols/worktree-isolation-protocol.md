# Worktree Isolation and Recovery

Use a separate checkout for concurrent writers or a disruptive experiment. Read-only investigation usually needs no worktree. A worktree isolates working files while sharing Git metadata; it does not restrict processes, credentials, network access, services, ports or databases. Use host/container/VM isolation when required.

Create task-specific branches and sanitized paths under `.agent-worktrees/` by default; ignore that directory in Git. Record base revision, owner, allowed files and intended integration branch. Verify the actual CLI path layout rather than assuming a task/slice nesting that the tool does not implement.

Before integration, inspect the complete branch diff and ensure the child result references the expected base. Apply only intended changes and resolve conflicts with current user edits. Run the relevant verifier on the integrated result, not merely before the merge.

Never force-remove a dirty worktree or pop another person's shared stash. Inspect tracked/untracked files and preserve valuable changes. Commit only authorized agent-owned work; otherwise leave the checkout intact with a clear handoff. If quarantine is appropriate, use `git worktree move` so Git's registration remains consistent, and record the destination. The package cleanup command preserves dirty work; it does not automatically quarantine or merge branches.

Remove a clean checkout only after its changes are integrated or intentionally retained on a branch. Delete only merged agent branches with normal safe Git checks. An unresolved cleanup is preferable to losing work.
