"""TaskSupervisor: Process group isolation, tracking, and kill ladder.

Decoupled from the main agent event loop to guarantee that runaway tools,
subagents, or worker processes can be inspected and cleanly terminated
without leaving orphaned zombie processes.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import asyncio
import atexit
import contextlib
import contextvars
import ctypes
import dataclasses
import logging
import os
import signal
import subprocess
import time

from sagent.agent.state import agent_label_var, agent_registry


if TYPE_CHECKING:
    from collections.abc import Mapping

    from sagent.agent.state import AgentLike

logger = logging.getLogger(__name__)

current_job_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_job_id",
    default="",
)


@dataclasses.dataclass(slots=True)
class ProcessEntry:
    """Record of an active OS process / process group spawned by a tool."""

    call_id: str
    pid: int
    pgid: int
    command: str = ""
    tool_name: str = ""
    agent_label: str = ""
    started: float = dataclasses.field(default_factory=time.time)
    process: asyncio.subprocess.Process | subprocess.Popen[bytes] | None = None
    stdout_tail: list[str] = dataclasses.field(default_factory=list)


def _init_child_subreaper() -> None:
    """Register this process as a subreaper on Linux so orphaned child trees reparent here."""
    with contextlib.suppress(AttributeError, OSError):
        pr_set_child_subreaper = 36
        libc = ctypes.CDLL(None)
        if hasattr(libc, "prctl"):
            _ = libc.prctl(pr_set_child_subreaper, 1, 0, 0, 0)


def resolve_agent_label(agent: AgentLike | None) -> str:
    """Resolve the stable label for an agent across registry, property, and contextvar."""
    if agent is None:
        return agent_label_var.get("")
    if hasattr(agent, "label") and agent.label:
        return str(agent.label)
    if hasattr(agent, "_registered_label"):
        reg = agent._registered_label()  # noqa: SLF001
        if reg:
            return reg
    for label, reg_agent in agent_registry.items():
        if reg_agent is agent:
            return label
    return agent_label_var.get("")


class TaskSupervisor:
    """Out-of-band process supervisor and kill ladder engine."""

    def __init__(self, agent_label: str = "") -> None:
        self.agent_label: str = agent_label
        self._processes: dict[str, ProcessEntry] = {}
        _init_child_subreaper()
        # Register atexit handler so no processes leak if the process exits abruptly
        atexit.register(self.cleanup_all)

    def register_process(
        self,
        call_id: str,
        pid: int,
        pgid: int | None = None,
        *,
        command: str = "",
        tool_name: str = "",
        agent_label: str = "",
        process: asyncio.subprocess.Process | subprocess.Popen[bytes] | None = None,
        stdout_tail: list[str] | None = None,
    ) -> ProcessEntry:
        """Register a spawned process and its process group."""
        resolved_pgid = pgid if pgid is not None else pid
        entry = ProcessEntry(
            call_id=call_id,
            pid=pid,
            pgid=resolved_pgid,
            command=command,
            tool_name=tool_name,
            agent_label=agent_label or self.agent_label,
            process=process,
            stdout_tail=stdout_tail if stdout_tail is not None else [],
        )
        self._processes[call_id] = entry
        logger.debug(
            "TaskSupervisor: registered call_id=%s pid=%d pgid=%d (tool=%s)",
            call_id,
            pid,
            resolved_pgid,
            tool_name,
        )
        return entry

    def unregister_process(self, call_id: str) -> ProcessEntry | None:
        """Remove a process record on normal completion."""
        return self._processes.pop(call_id, None)

    def get_process(self, call_id: str) -> ProcessEntry | None:
        """Find a process entry by call_id."""
        return self._processes.get(call_id)

    def get_by_pid(self, pid: int) -> ProcessEntry | None:
        """Find a process entry by pid or pgid."""
        for entry in self._processes.values():
            if pid in (entry.pid, entry.pgid):
                return entry
        return None

    def all_processes(self) -> Mapping[str, ProcessEntry]:
        """Return read-only snapshot of tracked processes."""
        return dict(self._processes)

    @staticmethod
    def is_pgid_alive(pgid: int) -> bool:
        """Check if any process in process group ``pgid`` is still alive."""
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # Alive, but owned by another entity or restricted
            return True
        except OSError:
            return False

    @staticmethod
    def is_pid_alive(pid: int) -> bool:
        """Check if a specific PID is still alive and not a zombie."""
        try:
            stat_path = Path(f"/proc/{pid}/stat")
            if stat_path.exists():
                content = stat_path.read_text(encoding="utf-8", errors="replace")
                # Status flag is after the closing parenthesis: e.g. (bash) S ...
                state = content.rpartition(")")[2].split()[0]
                return state != "Z"
            # Fall back to kill 0
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, FileNotFoundError):
            return False
        except OSError:
            return False

    @staticmethod
    def reap_zombies() -> list[int]:
        """Reap any dead child processes without blocking."""
        reaped: list[int] = []
        while True:
            try:
                pid, status = os.waitpid(-1, os.WNOHANG)
                if pid <= 0:
                    break
                reaped.append(pid)
                logger.debug("Reaped zombie pid=%d status=%d", pid, status)
            except (ChildProcessError, OSError):
                break
        return reaped

    async def kill_process_group(
        self,
        pgid: int,
        *,
        grace_sec: float = 0.5,
    ) -> bool:
        """SIGTERM -> wait grace_sec -> SIGKILL kill ladder for process group.

        Returns True if the process group was successfully terminated or already dead.
        """
        if not self.is_pgid_alive(pgid):
            self.reap_zombies()
            return True

        # Phase 1: SIGTERM
        try:
            os.killpg(pgid, signal.SIGTERM)
            logger.debug("Sent SIGTERM to pgid=%d", pgid)
        except ProcessLookupError:
            self.reap_zombies()
            return True
        except OSError as exc:
            logger.warning("Error sending SIGTERM to pgid=%d: %s", pgid, exc)

        # Non-blocking wait for grace_sec
        deadline = time.monotonic() + grace_sec
        while time.monotonic() < deadline:
            self.reap_zombies()
            if not self.is_pgid_alive(pgid):
                return True
            await asyncio.sleep(0.02)

        # Phase 2: SIGKILL if still alive
        if self.is_pgid_alive(pgid):
            logger.warning(
                "pgid=%d survived SIGTERM for %.2fs; escalating to SIGKILL",
                pgid,
                grace_sec,
            )
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError as exc:
                logger.warning("Error sending SIGKILL to pgid=%d: %s", pgid, exc)

        # Brief settle time and final reap
        await asyncio.sleep(0.05)
        self.reap_zombies()
        return not self.is_pgid_alive(pgid)

    async def kill_task(self, call_id: str, *, grace_sec: float = 0.5) -> bool:
        """Terminate a task and its process group by call_id."""
        entry = self._processes.pop(call_id, None)
        if entry is None:
            return False
        return await self.kill_process_group(entry.pgid, grace_sec=grace_sec)

    def kill_active_foreground(self, agent: AgentLike | None = None) -> list[int]:
        """Emergency brake: immediately terminate running foreground process groups."""
        killed_pids: list[int] = []
        agent_label = resolve_agent_label(agent)
        for _call_id, entry in list(self._processes.items()):
            if agent_label and entry.agent_label and entry.agent_label != agent_label:
                continue
            if self.is_pgid_alive(entry.pgid):
                try:
                    os.killpg(entry.pgid, signal.SIGKILL)
                    killed_pids.append(entry.pgid)
                except OSError:
                    pass
        self.reap_zombies()
        return killed_pids

    def has_active_foreground(self, agent: AgentLike | None = None) -> bool:
        """Return True if there are live foreground processes registered for this agent."""
        agent_label = resolve_agent_label(agent)
        for entry in self._processes.values():
            if agent_label and entry.agent_label and entry.agent_label != agent_label:
                continue
            if self.is_pgid_alive(entry.pgid):
                return True
        return False

    def cleanup_all(self) -> None:
        """Clean up all tracked process groups synchronously via SIGKILL (for atexit/shutdown)."""
        for call_id, entry in list(self._processes.items()):
            try:
                if self.is_pgid_alive(entry.pgid):
                    os.killpg(entry.pgid, signal.SIGKILL)
            except OSError:
                pass
            self._processes.pop(call_id, None)
        self.reap_zombies()

    async def aclose(self) -> None:
        """Async cleanup awaiting termination of all tracked process groups."""
        for call_id, entry in list(self._processes.items()):
            await self.kill_process_group(entry.pgid, grace_sec=0.2)
            self._processes.pop(call_id, None)
        self.reap_zombies()


# Global default supervisor shared across the process
_DEFAULT_SUPERVISOR: TaskSupervisor = TaskSupervisor()


def get_default_supervisor() -> TaskSupervisor:
    """Return the global default TaskSupervisor instance."""
    return _DEFAULT_SUPERVISOR
