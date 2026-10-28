"""Unit tests for TaskSupervisor.

Verifies process registration, PID/PGID tracking, kill ladder (SIGTERM -> SIGKILL),
zombie reaping, foreground preemption, and process group lifecycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import asyncio
import os
import subprocess

import pytest

from sagent.agent.agent import Agent
from sagent.agent.agent_test import StubModel
from sagent.agent.background import BackgroundTaskEntry
from sagent.agent.state import agent_registry
from sagent.agent.supervisor import TaskSupervisor
from sagent.tools.task_control import terminate_subagent_tree
from sagent.types.runtime import AssistantMessage


if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
async def test_supervisor_registration_and_lookup() -> None:
    """TaskSupervisor registers, finds, and unregisters processes by call_id and pid."""
    sup = TaskSupervisor(agent_label="test-agent")
    entry = sup.register_process(
        call_id="call-123",
        pid=99999,
        pgid=99999,
        command="test command",
        tool_name="TestTool",
    )
    assert entry.call_id == "call-123"
    assert entry.pid == 99999
    assert entry.agent_label == "test-agent"
    assert sup.get_process("call-123") is entry
    assert sup.get_by_pid(99999) is entry
    assert "call-123" in sup.all_processes()

    unreg = sup.unregister_process("call-123")
    assert unreg is entry
    assert sup.get_process("call-123") is None
    assert sup.get_by_pid(99999) is None


@pytest.mark.asyncio
async def test_supervisor_kill_process_group_live_process() -> None:
    """TaskSupervisor kill_process_group cleanly terminates a live process and its PGID."""
    sup = TaskSupervisor(agent_label="test-agent")
    # Launch real subprocess in dedicated session/process group
    proc = subprocess.Popen(  # noqa: ASYNC220
        ["sleep", "100"],  # noqa: S607
        start_new_session=True,
    )
    pid = proc.pid
    pgid = os.getpgid(pid)

    assert sup.is_pid_alive(pid)
    assert sup.is_pgid_alive(pgid)

    sup.register_process(
        call_id="sleep-100",
        pid=pid,
        pgid=pgid,
        command="sleep 100",
        tool_name="Bash",
        process=proc,
    )

    success = await sup.kill_process_group(pgid, grace_sec=0.1)
    assert success is True
    assert not sup.is_pgid_alive(pgid)
    assert not sup.is_pid_alive(pid)


@pytest.mark.asyncio
async def test_supervisor_kill_active_foreground_scoped_by_agent() -> None:
    """kill_active_foreground terminates only processes matching agent label."""
    sup = TaskSupervisor(agent_label="default")
    # Process 1 for agent-A
    p1 = subprocess.Popen(["sleep", "100"], start_new_session=True)  # noqa: ASYNC220, S607
    pg1 = os.getpgid(p1.pid)
    sup.register_process(
        call_id="p1",
        pid=p1.pid,
        pgid=pg1,
        agent_label="agent-A",
        process=p1,
    )

    # Process 2 for agent-B
    p2 = subprocess.Popen(["sleep", "100"], start_new_session=True)  # noqa: ASYNC220, S607
    pg2 = os.getpgid(p2.pid)
    sup.register_process(
        call_id="p2",
        pid=p2.pid,
        pgid=pg2,
        agent_label="agent-B",
        process=p2,
    )

    class _MockAgent:
        label = "agent-A"

    killed = sup.kill_active_foreground(_MockAgent())
    assert pg1 in killed

    # p1 should be dead, p2 still alive
    await asyncio.sleep(0.05)
    assert not sup.is_pid_alive(p1.pid)
    assert sup.is_pid_alive(p2.pid)

    # Cleanup p2
    await sup.kill_process_group(pg2, grace_sec=0.05)
    assert not sup.is_pid_alive(p2.pid)


@pytest.mark.asyncio
async def test_supervisor_reap_zombies() -> None:
    """reap_zombies non-blockingly collects exited child processes."""
    p = subprocess.Popen(["true"])  # noqa: ASYNC220, S607
    p.wait()
    reaped = TaskSupervisor.reap_zombies()
    assert isinstance(reaped, list)


@pytest.mark.asyncio
async def test_terminate_subagent_tree_recursive_teardown(tmp_path: Path) -> None:
    """terminate_subagent_tree recursively shuts down an agent and its child registry entries."""
    parent = Agent(
        model=StubModel(responses=[AssistantMessage(text="done")]),
        session_dir=tmp_path / "parent",
    )
    child = Agent(
        model=StubModel(responses=[AssistantMessage(text="done")]),
        session_dir=tmp_path / "child",
    )

    child_task = asyncio.create_task(child.serve_forever())
    agent_registry["child-agent"] = child

    parent.register_background(
        "persistent:child-agent",
        BackgroundTaskEntry(
            task=child_task,
            tool_name="serviced-agent",
            queue_id="child-agent",
            started=0.0,
            kind="subagent",
            lifecycle="serviced",
            persistent_run_id="run-child",
        ),
    )

    try:
        await asyncio.sleep(0.01)
        terminate_subagent_tree(parent)
        await asyncio.sleep(0.02)

        assert parent._shutting_down is True
        assert child._shutting_down is True
        assert child_task.done()
        assert "child-agent" not in agent_registry
    finally:
        child_task.cancel()
        agent_registry.clear()

