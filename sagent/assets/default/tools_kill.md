Convenience alias for `TaskControl(action='kill')`.

Terminate background tasks, runaway processes, or subagents.
Supports selective (`cascade: false`, one_for_one) or cascading (`cascade: true`, one_for_all) teardown.

Arguments:
- `target` (or `id`, required) -- task ID (e.g. `job-1`), tool name (e.g. `Bash`), subagent label, or `all`.
- `cascade` (optional, default false) -- when true, recursively terminates all descendant subagents and child tasks.
