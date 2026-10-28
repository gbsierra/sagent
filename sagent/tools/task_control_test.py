"""Tests for TaskControl and Kill tools.

Verifies list, tail, kill, pause, resume, and foreground against running processes.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import asyncio
import os

import pytest

from sagent.agent.agent import Agent
from sagent.agent.agent_test import StubModel
from sagent.agent.runtime import AgentRuntime
from sagent.agent.session_io import _entry_from_json, _entry_to_json
from sagent.agent.state import current_agent_var
from sagent.providers.anthropic.api import _build_messages as build_anthropic_messages
from sagent.providers.anthropic.api_test import _make_request as make_anthropic_request
from sagent.providers.openai.compat import build_messages as build_openai_messages
from sagent.providers.openai.compat_test import _make_request as make_openai_request
from sagent.testing import with_fake_agent
from sagent.tools.bash import Bash
from sagent.tools.task_control import Kill, TaskControl
from sagent.types.runtime import (
    CANCELLED_PLACEHOLDER,
    AssistantMessage,
    ToolCall,
    ToolResult,
    ToolResultKind,
)


@pytest.mark.asyncio
async def test_task_control_list_empty() -> None:
    """TaskControl list with no jobs returns 'No background tasks.'."""
    tc = TaskControl()
    with with_fake_agent():
        res = await tc.run({"action": "list"})
    assert res.content == "No background tasks."
    assert not res.is_error


@pytest.mark.asyncio
async def test_task_control_list_active_processes(tmp_path: Path) -> None:
    """TaskControl list renders PID, tool name, phase, and command of real OS process."""
    agent = Agent(
        model=StubModel(responses=[AssistantMessage(text="done")]),
        tools=[Bash(), TaskControl()],
        session_dir=tmp_path / "session",
    )
    token = current_agent_var.set(agent)
    try:
        bash_tool = agent.runtime.tools_map["Bash"]
        res = await bash_tool.run({"command": "sleep 20", "background": True})
        assert "id=job-1" in res.content
        assert "pid=" in res.content

        job_entry = agent.background["job-1"]
        for _ in range(50):
            if job_entry.pid is not None:
                break
            await asyncio.sleep(0.01)
        assert job_entry.pid is not None

        tc = TaskControl()
        list_res = await tc.run({"action": "list"})
        assert not list_res.is_error
        assert "ID" in list_res.content
        assert "PID" in list_res.content
        assert "COMMAND" in list_res.content
        assert "job-1" in list_res.content
        assert str(job_entry.pid) in list_res.content
        assert "sleep 20" in list_res.content
        assert "running" in list_res.content
    finally:
        # cleanup
        agent.cancel_background("job-1")
        current_agent_var.reset(token)
        await agent.aclose()


@pytest.mark.asyncio
async def test_task_control_tail_streaming_output(tmp_path: Path) -> None:
    """TaskControl tail extracts recent stdout lines from a live running process."""
    agent = Agent(
        model=StubModel(responses=[AssistantMessage(text="done")]),
        tools=[Bash(), TaskControl()],
        session_dir=tmp_path / "session",
    )
    token = current_agent_var.set(agent)
    try:
        bash_tool = agent.runtime.tools_map["Bash"]
        # Script that prints 5 lines with delay
        cmd = 'python3 -c "import time; [print(f\'STREAM_LINE_{i}\', flush=True) or time.sleep(0.05) for i in range(5)]; time.sleep(5)"'
        res = await bash_tool.run({"command": cmd, "background": True})
        assert "id=job-1" in res.content

        job_entry = agent.background["job-1"]
        # Wait until at least 3 lines are buffered
        for _ in range(100):
            if len(job_entry.stdout_tail) >= 3:
                break
            await asyncio.sleep(0.02)
        assert len(job_entry.stdout_tail) >= 3

        tc = TaskControl()
        tail_res = await tc.run({"action": "tail", "target": "job-1", "lines": 2})
        assert not tail_res.is_error
        lines = tail_res.content.splitlines()
        assert len(lines) <= 2
        assert any("STREAM_LINE_" in line for line in lines)
    finally:
        agent.cancel_background("job-1")
        current_agent_var.reset(token)
        await agent.aclose()


@pytest.mark.asyncio
async def test_task_control_tail_errors() -> None:
    """TaskControl tail handles missing target and unknown job gracefully."""
    tc = TaskControl()
    with with_fake_agent():
        res_no_target = await tc.run({"action": "tail"})
        assert res_no_target.is_error
        assert "tail requires a target or id" in res_no_target.content

        res_unknown = await tc.run({"action": "tail", "target": "ghost"})
        assert res_unknown.is_error
        assert "No such job: ghost" in res_unknown.content


@pytest.mark.asyncio
async def test_task_control_pause_and_resume(tmp_path: Path) -> None:
    """TaskControl pause (SIGSTOP) and resume (SIGCONT) on a real OS process."""
    agent = Agent(
        model=StubModel(responses=[AssistantMessage(text="done")]),
        tools=[Bash(), TaskControl()],
        session_dir=tmp_path / "session",
    )
    token = current_agent_var.set(agent)
    try:
        bash_tool = agent.runtime.tools_map["Bash"]
        res = await bash_tool.run({"command": "sleep 30", "background": True})
        assert "id=job-1" in res.content

        job_entry = agent.background["job-1"]
        for _ in range(50):
            if job_entry.pid is not None:
                break
            await asyncio.sleep(0.01)
        pid = job_entry.pid
        assert pid is not None

        status_path = Path(f"/proc/{pid}/status")

        tc = TaskControl()
        # 1. Pause
        pause_res = await tc.run({"action": "pause", "target": "job-1"})
        assert not pause_res.is_error
        assert "Paused process group" in pause_res.content

        # Inspect /proc/<pid>/status on Linux: State line should contain 'T (stopped)'
        await asyncio.sleep(0.02)
        if status_path.exists():  # noqa: ASYNC240
            status_text = status_path.read_text()  # noqa: ASYNC240
            assert "T (stopped)" in status_text

        # 2. Resume
        resume_res = await tc.run({"action": "resume", "target": "job-1"})
        assert not resume_res.is_error
        assert "Resumed process group" in resume_res.content

        # Inspect /proc/<pid>/status on Linux: State should no longer be stopped
        await asyncio.sleep(0.02)
        if status_path.exists():  # noqa: ASYNC240
            status_text_resumed = status_path.read_text()  # noqa: ASYNC240
            assert "T (stopped)" not in status_text_resumed
    finally:
        agent.cancel_background("job-1")
        current_agent_var.reset(token)
        await agent.aclose()


@pytest.mark.asyncio
async def test_task_control_kill_by_job_id(tmp_path: Path) -> None:
    """TaskControl kill terminates the real OS process and cleans up background registry."""
    agent = Agent(
        model=StubModel(responses=[AssistantMessage(text="done")]),
        tools=[Bash(), TaskControl()],
        session_dir=tmp_path / "session",
    )
    token = current_agent_var.set(agent)
    try:
        bash_tool = agent.runtime.tools_map["Bash"]
        res = await bash_tool.run({"command": "sleep 30", "background": True})
        assert "id=job-1" in res.content

        job_entry = agent.background["job-1"]
        for _ in range(50):
            if job_entry.pid is not None:
                break
            await asyncio.sleep(0.01)
        pid = job_entry.pid
        assert pid is not None
        # Verify process is running
        os.kill(pid, 0)

        tc = TaskControl()
        kill_res = await tc.run({"action": "kill", "target": "job-1"})
        assert not kill_res.is_error
        assert "Cancelled: Bash (job-1)" in kill_res.content
        assert "job-1" not in agent.background

        # Verify process is terminated in OS
        is_alive = True
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except OSError:
                is_alive = False
                break
            await asyncio.sleep(0.02)
        assert not is_alive, f"Process {pid} should have been terminated by TaskControl kill"
    finally:
        current_agent_var.reset(token)
        await agent.aclose()


@pytest.mark.asyncio
async def test_task_control_kill_by_tool_name(tmp_path: Path) -> None:
    """TaskControl kill by tool name ('Bash') terminates all matching tasks."""
    agent = Agent(
        model=StubModel(responses=[AssistantMessage(text="done")]),
        tools=[Bash(), TaskControl()],
        session_dir=tmp_path / "session",
    )
    token = current_agent_var.set(agent)
    try:
        bash_tool = agent.runtime.tools_map["Bash"]
        _ = await bash_tool.run({"command": "sleep 30", "background": True})
        _ = await bash_tool.run({"command": "sleep 31", "background": True})

        assert "job-1" in agent.background
        assert "job-2" in agent.background

        for _ in range(50):
            if agent.background["job-1"].pid and agent.background["job-2"].pid:
                break
            await asyncio.sleep(0.01)
        pid1 = agent.background["job-1"].pid
        pid2 = agent.background["job-2"].pid
        assert pid1 is not None
        assert pid2 is not None

        tc = TaskControl()
        kill_res = await tc.run({"action": "kill", "target": "bash"})
        assert not kill_res.is_error
        assert "Killed 2 bash task(s)" in kill_res.content
        assert len(agent.background) == 0

        # Verify both processes are terminated
        await asyncio.sleep(0.05)
        with pytest.raises(ProcessLookupError):
            os.kill(pid1, 0)
        with pytest.raises(ProcessLookupError):
            os.kill(pid2, 0)
    finally:
        current_agent_var.reset(token)
        await agent.aclose()


@pytest.mark.asyncio
async def test_kill_tool_alias(tmp_path: Path) -> None:
    """Kill tool alias defaults action to 'kill' and terminates process."""
    agent = Agent(
        model=StubModel(responses=[AssistantMessage(text="done")]),
        tools=[Bash(), Kill()],
        session_dir=tmp_path / "session",
    )
    token = current_agent_var.set(agent)
    try:
        bash_tool = agent.runtime.tools_map["Bash"]
        res = await bash_tool.run({"command": "sleep 30", "background": True})
        assert "id=job-1" in res.content

        job_entry = agent.background["job-1"]
        for _ in range(50):
            if job_entry.pid is not None:
                break
            await asyncio.sleep(0.01)
        pid = job_entry.pid
        assert pid is not None

        kill_tool = Kill()
        kill_res = await kill_tool.run({"target": "job-1"})
        assert not kill_res.is_error
        assert "Cancelled: Bash (job-1)" in kill_res.content
        assert "job-1" not in agent.background

        await asyncio.sleep(0.05)
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        current_agent_var.reset(token)
        await agent.aclose()


@pytest.mark.asyncio
async def test_task_control_foreground(tmp_path: Path) -> None:
    """TaskControl foreground awaits completion of background task and returns result."""
    agent = Agent(
        model=StubModel(responses=[AssistantMessage(text="done")]),
        tools=[Bash(), TaskControl()],
        session_dir=tmp_path / "session",
    )
    token = current_agent_var.set(agent)
    try:
        bash_tool = agent.runtime.tools_map["Bash"]
        res = await bash_tool.run({"command": "echo 'completed foreground task'", "background": True})
        assert "id=job-1" in res.content

        tc = TaskControl()
        fg_res = await tc.run({"action": "foreground", "target": "job-1"})
        assert not fg_res.is_error
        assert "completed foreground task" in fg_res.content
        assert "job-1" not in agent.background
    finally:
        current_agent_var.reset(token)
        await agent.aclose()


@pytest.mark.asyncio
async def test_killed_tool_wire_valid_anthropic_and_openai() -> None:
    """A killed tool call produces a paired CANCELLED result accepted by API serializers."""
    call_id = "toolu_01928374"
    tc = ToolCall(id=call_id, name="Bash", args={"command": "sleep 100"})
    asst = AssistantMessage(tool_calls=(tc,))

    cancelled_result = ToolResult(
        call_id=call_id,
        content=CANCELLED_PLACEHOLDER,
        is_error=True,
        kind=ToolResultKind.CANCELLED,
        failure_attribution={
            "signal": "SIGKILL",
            "trigger": "user_kill",
            "exit_code": -9,
            "job_id": "job-1",
        },
    )

    history = [asst, cancelled_result]

    # 1. Anthropic wire validation
    anthropic_req = make_anthropic_request(history)
    anthropic_msgs = build_anthropic_messages(anthropic_req)

    assert len(anthropic_msgs) == 2
    assert anthropic_msgs[0]["role"] == "assistant"
    assert anthropic_msgs[1]["role"] == "user"

    user_content = cast(list[dict[str, object]], anthropic_msgs[1]["content"])
    tool_results = [b for b in user_content if b.get("type") == "tool_result"]
    assert len(tool_results) == 1
    assert tool_results[0]["is_error"] is True
    assert tool_results[0]["content"] == CANCELLED_PLACEHOLDER

    asst_content = cast(list[dict[str, object]], anthropic_msgs[0]["content"])
    tool_uses = [b for b in asst_content if b.get("type") == "tool_use"]
    assert len(tool_uses) == 1
    assert tool_results[0]["tool_use_id"] == tool_uses[0]["id"]

    # 2. OpenAI wire validation
    openai_req = make_openai_request(messages=history)
    openai_msgs = build_openai_messages(openai_req)

    assert len(openai_msgs) == 2
    assert openai_msgs[0]["role"] == "assistant"
    assert openai_msgs[1]["role"] == "tool"
    assert openai_msgs[1]["tool_call_id"] == openai_msgs[0]["tool_calls"][0]["id"]
    assert CANCELLED_PLACEHOLDER in str(openai_msgs[1]["content"])

    # 3. Tape roundtrip validation
    serialized = _entry_to_json(cancelled_result)
    assert serialized["failure_attribution"] == {
        "signal": "SIGKILL",
        "trigger": "user_kill",
        "exit_code": -9,
        "job_id": "job-1",
    }
    reloaded = _entry_from_json(serialized)
    assert isinstance(reloaded, ToolResult)
    assert reloaded.kind == ToolResultKind.CANCELLED
    assert reloaded.is_error is True
    assert reloaded.failure_attribution == cancelled_result.failure_attribution


@pytest.mark.asyncio
async def test_runtime_kill_all_tools_generates_paired_wire_history() -> None:
    """runtime._stop_all_tools(mode='kill') generates paired synthetic results with failure attribution."""
    rt = AgentRuntime(model=StubModel(responses=[]))
    cid1 = "call-1"
    cid2 = "call-2"

    async def hang() -> None:
        await asyncio.sleep(100)

    task1 = asyncio.create_task(hang())
    task2 = asyncio.create_task(hang())

    rt.running_tools[cid1] = task1
    rt.running_tools[cid2] = task2
    rt.cohort.add(cid1)
    rt.cohort.add(cid2)

    rt._stop_all_tools(mode="kill")

    results = [rec.event for rec in rt.tape if isinstance(rec.event, ToolResult)]
    assert len(results) == 2
    cids = {r.call_id for r in results}
    assert cids == {cid1, cid2}

    for res in results:
        assert res.kind == ToolResultKind.CANCELLED
        assert res.is_error is True
        assert res.content == CANCELLED_PLACEHOLDER
        assert res.failure_attribution is not None
        assert res.failure_attribution["trigger"] == "runtime_kill"

    await asyncio.sleep(0)
    assert task1.cancelled()
    assert task2.cancelled()
