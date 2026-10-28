Manage background tasks, runaway processes, and subagent swarms.

Operations:
- `list` -- show all active background tasks with ID, PID, tool name, phase (running/sleeping/completed), elapsed time, and command.
- `tail` -- view recent live stdout/stderr lines for a running or completed task.
- `kill` -- terminate a background task, process group, or subagent by ID, tool name, or 'all'. Supports selective (`cascade: false`, one_for_one) or cascading (`cascade: true`, one_for_all) teardown.
- `pause` -- temporarily suspend a running process group (SIGSTOP).
- `resume` -- resume a suspended process group (SIGCONT).
- `foreground` -- await task completion and return its result.

Arguments:
- `action` (required) -- one of `list`, `tail`, `kill`, `pause`, `resume`, `foreground`.
- `target` (or `id`) -- task ID (e.g. `job-1`), tool name (e.g. `Bash`), subagent label, or `all`.
- `lines` (optional, default 20) -- number of output lines to tail.
- `cascade` (optional, default false) -- when killing, recursively terminate child subagents and their tasks.
