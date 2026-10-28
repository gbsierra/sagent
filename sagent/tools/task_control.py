"""``TaskControl`` tool: Inspect, tail output, pause, resume, or terminate tasks.

Task management tool for background jobs, processes, and subagents.
Provides:
- ``list``: inspect active background jobs with PID, command, phase, elapsed.
- ``tail``: live recent stdout/stderr output lines for running or finished tasks.
- ``kill``: terminate task by ID, tool name, agent label, or 'all', with selective
  (``cascade: false``, one_for_one) or cascading (``cascade: true``, one_for_all) teardown.
- ``pause`` / ``resume``: signal process groups with SIGSTOP / SIGCONT.
- ``foreground``: await task completion and return result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import asyncio
import contextlib
import os
import signal
import time

from treekle import PlainTree, immutable

from sagent.agent.state import (
    AgentLike,
    agent_registry,
    current_agent_var,
)
from sagent.agent.supervisor import get_default_supervisor
from sagent.tools.background_task import (
    _await_detached,
    _find_history_result,
    _splice_from_inbox_items,
    cancel_persistent_subagent,
)
from sagent.tools.core import load_tool_description
from sagent.types.runtime import (
    CANCELLED_PLACEHOLDER,
    DetachedResult,
    ToolResult,
    ToolResultKind,
)


if TYPE_CHECKING:
    from collections.abc import Mapping


__all__ = [
    "Kill",
    "TaskControl",
    "get_downstream_subagents",
    "terminate_subagent_tree",
]


def get_downstream_subagents(agent: AgentLike) -> list[AgentLike]:
    """Return the list of all descendant agents in this agent's downstream subtree."""
    descendants: list[AgentLike] = []
    to_visit: list[AgentLike] = [agent]
    visited_ids: set[int] = set()
    while to_visit:
        curr = to_visit.pop(0)
        curr_id = id(curr)
        if curr_id in visited_ids:
            continue
        visited_ids.add(curr_id)
        if curr is not agent:
            descendants.append(curr)
        bg = getattr(curr, "background", None)
        if isinstance(bg, dict):
            for job in bg.values():
                if getattr(job, "kind", None) == "subagent":
                    label = getattr(job, "queue_id", "").removeprefix("persistent:")
                    child = agent_registry.get(label)
                    if child is not None and id(child) not in visited_ids:
                        to_visit.append(child)
    return descendants


class TaskControl:
    """Tool: list, tail, kill, pause, resume, or foreground tasks."""

    name: str = "TaskControl"
    tool_id: str = "application/x-tool-taskcontrol"
    clearable_results: bool = False
    description: str = load_tool_description("TaskControl")
    directive_schema: Mapping[str, PlainTree] = immutable(
        {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "tail", "kill", "pause", "resume", "foreground"],
                    "description": "Action to perform: list running tasks, tail output, kill/cancel, pause, resume, or foreground.",
                },
                "target": {
                    "type": "string",
                    "description": "Target task ID (e.g. 'job-1'), tool name ('Bash'), agent label, or 'all'.",
                },
                "id": {
                    "type": "string",
                    "description": "Alias for target.",
                },
                "lines": {
                    "type": "integer",
                    "default": 20,
                    "description": "Number of recent stdout/stderr lines to return for tail (default 20).",
                },
                "cascade": {
                    "type": "boolean",
                    "default": False,
                    "description": "When true, recursively terminates all descendant subagents and child tasks (one_for_all). Defaults to false (one_for_one).",
                },
            },
            "required": ["action"],
        },
    )

    def summary(self, args: Mapping[str, object]) -> str:
        """Return a short label for this task control operation."""
        action = str(args.get("action") or args.get("operation") or "?")
        target = str(args.get("target") or args.get("id") or "")
        if target:
            return f"TaskControl {action} {target}"
        return f"TaskControl {action}"

    def prompt(self) -> str:
        """Return task control guidance for the system prompt."""
        return (
            "Use TaskControl (or alias Kill) to manage running background tasks and processes:\n"
            "- `TaskControl(action='list')`: inspect active jobs with PID, command, phase, elapsed.\n"
            "- `TaskControl(action='tail', target='job-1')`: view recent output lines from a running task.\n"
            "- `TaskControl(action='kill', target='job-1')`: cancel a task (selective one_for_one by default; pass cascade=true to terminate subagent tree).\n"
            "- `TaskControl(action='pause'|'resume', target='job-1')`: temporarily suspend/resume a task."
        )

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        """Run in parallel: task control operations have no shared serialization resource."""
        del args
        return None

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Dispatch a task control operation."""
        # Support both 'action' and legacy 'operation'
        action = str(args.get("action") or args.get("operation") or "").lower()
        target = str(args.get("target") or args.get("id") or "")
        cascade = bool(args.get("cascade", False))
        lines_count = int(args.get("lines", 20))

        if action == "cancel":
            action = "kill"

        agent = current_agent_var.get(None)
        if agent is None:
            return ToolResult(
                call_id="",
                content="BackgroundTask: no active agent" if action in ("cancel", "list", "foreground") else "TaskControl: no active agent",
                is_error=True,
            )

        if action == "list":
            return self._list(agent)
        if action == "tail":
            return self._tail(agent, target, lines_count)
        if action == "kill":
            return await self._kill(agent, target, cascade=cascade)
        if action == "pause":
            return self._pause(agent, target)
        if action == "resume":
            return self._resume(agent, target)
        if action == "foreground":
            return await self._foreground(agent, target)

        return ToolResult(
            call_id="",
            content=f"Unknown operation: {action!r}" if "operation" in args else f"Unknown action: {action!r}",
            is_error=True,
        )

    def _list(self, agent: AgentLike) -> ToolResult:
        """Render the background task listing with PID, Command, and Elapsed."""
        jobs = {q: j for q, j in agent.background.items() if not j.hidden}
        if not jobs:
            return ToolResult(call_id="", content="No background tasks.")

        lines: list[str] = []
        now = time.time()
        for qid, job in jobs.items():
            elapsed = now - job.started
            if job.task.cancelled():
                phase = "cancelled"
            elif job.task.done():
                try:
                    phase = "failed" if job.task.exception() is not None else "completed"
                except asyncio.CancelledError:
                    phase = "cancelled"
            elif job.delay_sec > 0 and elapsed < job.delay_sec:
                phase = "sleeping"
            else:
                phase = "running"
            pid_str = str(job.pid) if job.pid is not None else "-"
            cmd_str = job.command.replace("\n", " ").strip()
            if len(cmd_str) > 35:
                cmd_str = cmd_str[:32] + "..."
            lines.append(
                f"{qid:<8s}  {pid_str:<7s}  {job.tool_name:<16s}  {phase:<10s}  {elapsed:>4.0f}s  {cmd_str}",
            )

        header = f"{'ID':<8s}  {'PID':<7s}  {'TOOL':<16s}  {'PHASE':<10s}  {'ELAPSED':<6s}  COMMAND"
        return ToolResult(call_id="", content=header + "\n" + "\n".join(lines))

    def _tail(self, agent: AgentLike, target: str, lines_count: int = 20) -> ToolResult:
        """Return the recent output lines of a running or completed task."""
        if not target:
            return ToolResult(call_id="", content="tail requires a target or id", is_error=True)

        job = agent.background.get(target)
        if job is not None and not job.hidden:
            if job.stdout_tail:
                tail_lines = job.stdout_tail[-lines_count:]
                output = "".join(tail_lines).strip()
                if output:
                    return ToolResult(call_id="", content=output)

            # Check if finished and in history
            call_id = job.call_id or job.queue_id
            spliced = _find_history_result(agent, call_id)
            if spliced is not None and spliced.content:
                lines = spliced.content.splitlines(keepends=True)[-lines_count:]
                return ToolResult(call_id="", content="".join(lines).strip())

            return ToolResult(call_id="", content=f"(No output produced yet for {target})")

        # Fallback: check history directly for call_id == target
        spliced = _find_history_result(agent, target)
        if spliced is not None and spliced.content:
            lines = spliced.content.splitlines(keepends=True)[-lines_count:]
            return ToolResult(call_id="", content="".join(lines).strip())

        # Fallback 2: check runtime inbox for DetachedResult
        inbox = getattr(getattr(agent, "runtime", None), "inbox", None)
        if inbox is not None:
            queue_items = getattr(getattr(inbox, "_queue", None), "_queue", None)
            items = list(queue_items) if queue_items is not None else (list(inbox) if hasattr(inbox, "__iter__") else [])
            for item in items:
                if isinstance(item, DetachedResult):
                    c_id = getattr(item, "call_id", getattr(item.result, "call_id", ""))
                    if c_id in (target, f"job-{target}"):
                        lines = item.result.content.splitlines(keepends=True)[-lines_count:]
                        return ToolResult(call_id="", content="".join(lines).strip())

        return ToolResult(call_id="", content=f"No such job: {target}", is_error=True)

    async def _kill(
        self,
        agent: AgentLike,
        target: str,
        *,
        cascade: bool = False,
    ) -> ToolResult:
        """Terminate target by job ID, tool name, agent label, or 'all'."""
        if not target:
            return ToolResult(call_id="", content="cancel requires an id", is_error=True)

        supervisor = get_default_supervisor()
        downstream = get_downstream_subagents(agent)
        downstream_ids = {id(a) for a in downstream}

        if target == "all":
            # Terminate all tasks owned by this agent
            visible_jobs = [j for j in agent.background.values() if not j.hidden]
            for job in visible_jobs:
                if job.pgid or job.pid:
                    await supervisor.kill_process_group(job.pgid or job.pid)
                _ = job.task.cancel()
                call_id = job.call_id or job.queue_id
                agent.runtime.inbox.push_back(
                    DetachedResult(
                        result=ToolResult(
                            call_id=call_id,
                            content=CANCELLED_PLACEHOLDER,
                            is_error=True,
                            kind=ToolResultKind.CANCELLED,
                            failure_attribution={
                                "trigger": "user_kill",
                                "signal": "SIGKILL",
                                "exit_code": 143,
                                "job_id": job.queue_id,
                            },
                        ),
                    ),
                )
                agent.cancel_background(job.queue_id)

            if cascade:
                # Scoped Ownership: only recursively tear down downstream subagents
                for child in downstream:
                    terminate_subagent_tree(child)

            return ToolResult(call_id="", content="Terminated all background tasks.")

        # 1. Check if target matches a background job ID
        job = agent.background.get(target)
        if job is not None and not job.hidden:
            if job.kind == "subagent":
                label = target.removeprefix("persistent:") if target.startswith("persistent:") else job.queue_id.removeprefix("persistent:")
                child = agent_registry.get(label)
                if child is not None:
                    if id(child) not in downstream_ids and child is not agent:
                        return ToolResult(
                            call_id="",
                            content=f"Permission denied: cannot kill agent {label!r} outside your downstream subtree (scoped ownership).",
                            is_error=True,
                        )
                    if cascade:
                        terminate_subagent_tree(child)
                cancel_persistent_subagent(agent, label)
                return ToolResult(call_id="", content=f"Cancelled: {job.tool_name} ({target})")

            # Tool job
            if job.pgid or job.pid:
                await supervisor.kill_process_group(job.pgid or job.pid)
            _ = job.task.cancel()
            call_id = job.call_id or job.queue_id
            agent.runtime.inbox.push_back(
                DetachedResult(
                    result=ToolResult(
                        call_id=call_id,
                        content=CANCELLED_PLACEHOLDER,
                        is_error=True,
                        kind=ToolResultKind.CANCELLED,
                        failure_attribution={
                            "trigger": "user_kill",
                            "signal": "SIGKILL",
                            "exit_code": 143,
                            "job_id": target,
                        },
                    ),
                ),
            )
            agent.cancel_background(target)
            return ToolResult(call_id="", content=f"Cancelled: {job.tool_name} ({target})")

        # 2. Check if target matches an agent in agent_registry
        child = agent_registry.get(target)
        if child is not None:
            if id(child) not in downstream_ids and child is not agent:
                return ToolResult(
                    call_id="",
                    content=f"Permission denied: cannot kill agent {target!r} outside your downstream subtree (scoped ownership).",
                    is_error=True,
                )
            if cascade:
                terminate_subagent_tree(child)
            else:
                child.shutdown(force=True)
            return ToolResult(call_id="", content=f"Killed subagent: {target}")

        # 3. Check if target matches a tool name (case-insensitive, e.g. 'bash')
        matched_jobs = [
            qid for qid, j in agent.background.items()
            if not j.hidden and j.tool_name.lower() == target.lower()
        ]
        if matched_jobs:
            for qid in matched_jobs:
                j = agent.background.get(qid)
                if j is not None:
                    if j.pgid or j.pid:
                        await supervisor.kill_process_group(j.pgid or j.pid)
                    _ = j.task.cancel()
                    call_id = j.call_id or j.queue_id
                    agent.runtime.inbox.push_back(
                        DetachedResult(
                            result=ToolResult(
                                call_id=call_id,
                                content=CANCELLED_PLACEHOLDER,
                                is_error=True,
                                kind=ToolResultKind.CANCELLED,
                                failure_attribution={
                                    "trigger": "user_kill",
                                    "signal": "SIGKILL",
                                    "exit_code": 143,
                                    "job_id": qid,
                                },
                            ),
                        ),
                    )
                    agent.cancel_background(qid)
            return ToolResult(call_id="", content=f"Killed {len(matched_jobs)} {target} task(s).")

        return ToolResult(call_id="", content=f"No such job: {target}", is_error=True)

    def _pause(self, agent: AgentLike, target: str) -> ToolResult:
        """Suspend process group via SIGSTOP."""
        if not target:
            return ToolResult(call_id="", content="pause requires a target", is_error=True)
        job = agent.background.get(target)
        pgid = job.pgid or job.pid if job else None
        if pgid is None:
            with contextlib.suppress(ValueError):
                pgid = int(target)
        if pgid is None:
            return ToolResult(call_id="", content=f"No process group found for target: {target}", is_error=True)
        try:
            os.killpg(pgid, signal.SIGSTOP)
            return ToolResult(call_id="", content=f"Paused process group {pgid} ({target}).")
        except OSError as exc:
            return ToolResult(call_id="", content=f"Failed to pause {target}: {exc}", is_error=True)

    def _resume(self, agent: AgentLike, target: str) -> ToolResult:
        """Resume suspended process group via SIGCONT."""
        if not target:
            return ToolResult(call_id="", content="resume requires a target", is_error=True)
        job = agent.background.get(target)
        pgid = job.pgid or job.pid if job else None
        if pgid is None:
            with contextlib.suppress(ValueError):
                pgid = int(target)
        if pgid is None:
            return ToolResult(call_id="", content=f"No process group found for target: {target}", is_error=True)
        try:
            os.killpg(pgid, signal.SIGCONT)
            return ToolResult(call_id="", content=f"Resumed process group {pgid} ({target}).")
        except OSError as exc:
            return ToolResult(call_id="", content=f"Failed to resume {target}: {exc}", is_error=True)

    async def _foreground(self, agent: AgentLike, target: str) -> ToolResult:
        """Resolve a tracked background task to its result (same as BackgroundTask.foreground)."""
        if not target:
            return ToolResult(call_id="", content="foreground requires an id", is_error=True)
        job = agent.background.get(target)
        if job is None or job.hidden:
            # Check if job already completed and result is in history or inbox
            call_id = target
            if hasattr(agent, "_call_id_for_job"):
                call_id = agent._call_id_for_job(target)  # noqa: SLF001

            spliced = _find_history_result(agent, call_id)
            if spliced is None and call_id != target:
                spliced = _find_history_result(agent, target)
            if spliced is not None:
                return ToolResult(call_id="", content=spliced.content, is_error=spliced.is_error)

            inbox = getattr(getattr(agent, "runtime", None), "inbox", None)
            if inbox is not None and hasattr(inbox, "drain_nowait"):
                spliced = _splice_from_inbox_items(agent, call_id, inbox.drain_nowait())
                if spliced is None and call_id != target:
                    spliced = _splice_from_inbox_items(agent, target, inbox.drain_nowait())
                if spliced is not None:
                    return ToolResult(call_id="", content=spliced.content, is_error=spliced.is_error)

            return ToolResult(call_id="", content=f"No such job: {target}", is_error=True)
        if job.kind == "subagent" and job.lifecycle == "serviced":
            return ToolResult(
                call_id="",
                content=(
                    "Serviced subagent jobs cannot be foregrounded; use AgentSend "
                    "to message the child or BackgroundTask cancel to stop it."
                ),
                is_error=True,
            )
        call_id = job.call_id or job.queue_id
        spliced = _find_history_result(agent, call_id)
        if spliced is None:
            spliced = await _await_detached(agent, call_id, job)
        agent.cancel_background(target)
        return ToolResult(
            call_id="",
            content=spliced.content,
            is_error=spliced.is_error,
        )


class Kill(TaskControl):
    """Convenience alias tool for TaskControl with default action 'kill'."""

    name: str = "Kill"
    tool_id: str = "application/x-tool-kill"
    description: str = load_tool_description("Kill")
    directive_schema: Mapping[str, PlainTree] = immutable(
        {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Target task ID (e.g. 'job-1'), tool name ('Bash'), agent label, or 'all'.",
                },
                "id": {
                    "type": "string",
                    "description": "Alias for target.",
                },
                "cascade": {
                    "type": "boolean",
                    "default": False,
                    "description": "When true, recursively terminates all descendant subagents and child tasks (one_for_all). Defaults to false (one_for_one).",
                },
            },
            "required": ["target"],
        },
    )

    def summary(self, args: Mapping[str, object]) -> str:
        """Render brief summary of kill invocation."""
        target = str(args.get("target") or args.get("id") or "")
        return f"Kill {target}".strip()

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        """Execute Kill tool by delegating to TaskControl with action='kill'."""
        modified_args = dict(args)
        if "action" not in modified_args:
            modified_args["action"] = "kill"
        return await super().run(modified_args)


def terminate_subagent_tree(agent: AgentLike) -> None:
    """Recursively terminate an agent and all descendant subagents and background tasks."""
    get_default_supervisor()
    # Snapshot children to avoid dictionary mutation during iteration
    child_jobs = list(agent.background.items())
    for qid, job in child_jobs:
        if job.kind == "subagent":
            label = job.queue_id.removeprefix("persistent:")
            child = agent_registry.get(label)
            if child is not None and child is not agent:
                terminate_subagent_tree(child)
        elif job.pgid or job.pid:
            # Kill process group synchronously or signal
            with contextlib.suppress(OSError):
                os.killpg(job.pgid or job.pid, signal.SIGKILL)
        _ = job.task.cancel()
        agent.cancel_background(qid)

    for label, reg_agent in list(agent_registry.items()):
        if reg_agent is agent:
            agent_registry.pop(label, None)

    agent.shutdown(force=True, cascade=True)
