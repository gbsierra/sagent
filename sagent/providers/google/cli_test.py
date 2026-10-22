"""Tests for ``providers.google_cli``: wire-format + lifecycle."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import asyncio
import inspect

import pytest

from sagent.agent.retry import validate_model_response
from sagent.providers.google import cli
from sagent.providers.google.api import Google
from sagent.providers.google.cli import (
    GoogleCLI,
    _dispatch_session_update,
    _gemini_cli_home,
    _google_subprocess_env,
    _GoogleCLIModel,
    _GoogleCLIProcState,
    _user_prompt_blocks,
)
from sagent.providers.lib.cli_respawn import hash_system
from sagent.providers.lib.hotspare import HotSpare
from sagent.providers.lib.subproc import (
    Subproc,
    SubprocessTransportError,
)
from sagent.types.model import ModelRequest, ModelTerminationError
from sagent.types.runtime import (
    AssistantMessage,
    ModelResponsePartial,
    ModelResponseThinking,
    RuntimeEvent,
    ToolCall,
    ToolResult,
    UserMessage,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from sagent.lib.codec import MutablePlainTree


def _which_missing(_name: str) -> str | None:
    """Pretend Gemini is not installed."""
    return None


def _which_gemini(_name: str) -> str:
    """Return a stable fake Gemini executable path."""
    return "/usr/bin/gemini"


def test_google_cli_does_not_import_subscription_provider() -> None:
    source = inspect.getsource(cli)
    assert "providers.google_sub" not in source


def test_from_cli_rejects_missing_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``from_credentials`` fails fast if ``gemini`` is not on ``PATH``."""
    monkeypatch.setattr(
        "sagent.providers.google.cli.shutil.which",
        _which_missing,
    )
    with pytest.raises(RuntimeError, match="not on PATH"):
        GoogleCLI.from_credentials()


def test_from_cli_with_native_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sagent.providers.google.cli.shutil.which",
        _which_gemini,
    )
    provider = GoogleCLI.from_credentials()
    assert provider.account is None
    assert GoogleCLI.from_credentials(account="work").account == "work"


def test_google_cli_login_runs_native_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``/login`` delegates OAuth storage and refresh to Gemini CLI."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    captured: dict[str, object] = {}

    def fake_run(*args: object, **kwargs: object) -> object:
        captured["args"] = args
        captured["env"] = kwargs["env"]
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(
        "sagent.providers.google.cli.shutil.which",
        _which_gemini,
    )
    monkeypatch.setattr(
        "sagent.providers.google.cli.subprocess.run",
        fake_run,
    )

    GoogleCLI.login(account="work")

    assert captured["args"] == (["/usr/bin/gemini", "login"],)
    env = cast(dict[str, str], captured["env"])
    assert env["GEMINI_CLI_HOME"] == str(_gemini_cli_home("work"))


def test_google_subprocess_env_keeps_native_home_and_isolates_auth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Auth isolation uses the Gemini setting, not a copied temporary HOME."""
    operator_home = str(tmp_path / "operator-home")
    monkeypatch.setenv("HOME", operator_home)
    monkeypatch.setenv("USERPROFILE", operator_home)
    monkeypatch.setenv("GEMINI_CLI_HOME", "/operator/gemini")
    config_home = tmp_path / "sagent-gemini-work"
    system_path = tmp_path / "system.md"

    env = _google_subprocess_env(config_home, system_path=system_path)

    assert env["HOME"] == operator_home
    assert env["USERPROFILE"] == operator_home
    assert env["GEMINI_CLI_HOME"] == str(config_home)
    assert env["GEMINI_SYSTEM_MD"] == str(system_path)
    assert env["GEMINI_DEFAULT_AUTH_TYPE"] == "oauth-personal"
    for key in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
    ):
        monkeypatch.setenv(key, "not-a-real-secret")
        assert key not in _google_subprocess_env(None)


def test_gemini_cli_home_default_is_native_and_named_is_sagent_owned(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("GEMINI_CLI_HOME", str(tmp_path / "native"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert _gemini_cli_home(None) is None
    assert _gemini_cli_home("default") is None
    assert _gemini_cli_home("work") == (
        tmp_path / "xdg" / "rekursiv-ai" / "sagent" / "gemini" / "work"
    )


def test_from_key_delegates_to_google() -> None:
    """``GoogleCLI.from_key`` returns a plain ``Google`` provider for API keys."""
    fallback = GoogleCLI.from_key("AIza-not-real")
    assert not isinstance(fallback, GoogleCLI)


def test_model_unknown_raises() -> None:
    """An unsupported model id surfaces with the known-model list."""
    provider = GoogleCLI()
    with pytest.raises(ValueError, match="Unknown model"):
        _ = provider.model("not-a-gemini")


def test_model_uses_default_when_unset() -> None:
    provider = GoogleCLI()
    model = provider.model()
    assert model.capability.model_id == provider.catalog.resolve("default")[0].model_id


def test_catalog_inherits_from_google() -> None:
    assert tuple(GoogleCLI.catalog.models) == tuple(Google.catalog.models)


def test_utility_model_picks_flash_lite() -> None:
    """``utility`` resolves the same row on the CLI as on the API."""
    provider = GoogleCLI()
    model = provider.model("utility")
    assert model.capability.model_id == provider.catalog.resolve("utility")[0].model_id


def test_model_capabilities() -> None:
    """The CLI transport narrows the row it inherits from the API."""
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    assert model.capability.thinking.budget != frozenset({"none"})
    assert model.capability.thinking.effort == frozenset({"none"})
    assert model.capability.cache_ttl_sec == frozenset({0.0})
    assert model.capability.manage_context_server_side == frozenset({True})
    assert model.capability.account_auth is True


def test_max_image_limits() -> None:
    """Gemini has no per-image pixel/byte cap; only the 100 MB total applies."""
    model = GoogleCLI().model("gemini-2.5-flash")
    assert model.limits.max_image_edge_px == 0
    assert model.limits.max_image_bytes == 0
    assert model.limits.max_request_bytes == 100 * 1024 * 1024


def test_is_context_overflow_text_markers() -> None:
    """Overflow classification covers Gemini's documented overflow strings."""
    model = GoogleCLI().model("gemini-2.5-flash")
    assert model.is_context_overflow(RuntimeError("Input too long")) is True
    assert model.is_context_overflow(RuntimeError("exceeds the maximum")) is True
    assert model.is_context_overflow(RuntimeError("other")) is False


def test_byte_limit_not_classified_as_context_overflow() -> None:
    """A request-byte-limit error must not be swallowed by ``"too large"``.

    The bare ``"too large"`` substring would otherwise mis-route a byte
    wire-limit error into token-overflow recovery; the shared byte guard
    must short-circuit it first, uniform with the HTTP siblings.
    """
    model = GoogleCLI().model("gemini-2.5-flash")
    assert model.is_context_overflow(RuntimeError("request entity too large")) is False


def test_user_prompt_blocks_text_only() -> None:
    """Plain text turns into a single ``{type:text}`` block."""
    blocks = _user_prompt_blocks(
        UserMessage(text="hi"),
        max_image_dim=3072,
        max_image_bytes=20 * 1024 * 1024,
    )
    assert blocks == [{"type": "text", "text": "hi"}]


def test_user_prompt_blocks_empty_user_emits_placeholder() -> None:
    """An empty user message still emits one block to satisfy ACP shape."""
    blocks = _user_prompt_blocks(
        UserMessage(text=""),
        max_image_dim=3072,
        max_image_bytes=20 * 1024 * 1024,
    )
    assert blocks == [{"type": "text", "text": ""}]


def test_dispatch_session_update_routes_text_and_thinking() -> None:
    """``agent_message_chunk`` / ``agent_thought_chunk`` fan to the callbacks."""
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    text_chunks: list[str] = []
    thinking_chunks: list[str] = []

    def _sink(ev: RuntimeEvent) -> None:
        if isinstance(ev, ModelResponsePartial):
            text_chunks.append(ev.text)
        elif isinstance(ev, ModelResponseThinking):
            thinking_chunks.append(ev.text)

    message_chunk: dict[str, MutablePlainTree] = {
        "update": {
            "sessionUpdate": "agent_message_chunk",
            "content": {"text": "hello"},
        },
    }
    thought_chunk: dict[str, MutablePlainTree] = {
        "update": {
            "sessionUpdate": "agent_thought_chunk",
            "content": {"text": "thinking..."},
        },
    }
    _dispatch_session_update(
        message_chunk,
        text_parts,
        thinking_parts,
        _sink,
    )
    _dispatch_session_update(
        thought_chunk,
        text_parts,
        thinking_parts,
        _sink,
    )
    assert text_parts == ["hello"]
    assert thinking_parts == ["thinking..."]
    assert text_chunks == ["hello"]
    assert thinking_chunks == ["thinking..."]


def test_dispatch_session_update_ignores_unknown_kinds() -> None:
    """``tool_call_update`` and other kinds are dropped without side effects."""
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    unknown_update: dict[str, MutablePlainTree] = {
        "update": {"sessionUpdate": "tool_call_update", "id": 1},
    }
    _dispatch_session_update(
        unknown_update,
        text_parts,
        thinking_parts,
        None,
    )
    assert text_parts == []
    assert thinking_parts == []


def test_approx_request_tokens_sums_user_and_assistant() -> None:
    """The walker counts both user and assistant text entries."""
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    request = ModelRequest(
        messages=[
            UserMessage(text="aaaa" * 4),
            AssistantMessage(text="bbbb"),
            UserMessage(text="cccc"),
        ],
    )
    estimate = model.approx_request_tokens(request)
    # ``len("a"*16) // 4 = 4`` + ``4 // 4 = 1`` + ``4 // 4 = 1`` = 6.
    assert estimate == 6


def test_should_respawn_triggers() -> None:
    """Each respawn trigger from §1.4 is detected on the next request."""
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")

    class _DummyActive:
        pass

    model._hot_spare._active = cast(Subproc, _DummyActive())
    user = UserMessage(text="hi")
    request = ModelRequest(messages=[user])

    model._system_hash = hash_system("different")
    assert model._should_respawn(request) is True

    model._system_hash = hash_system("")
    model._sent_history_head = UserMessage(text="other")
    assert model._should_respawn(request) is True

    model._sent_history_head = user
    model._turn_count = 200
    assert model._should_respawn(request) is True

    model._turn_count = 0
    model._last_input_tokens = model.limits.max_request_tokens
    assert model._should_respawn(request) is True

    model._last_input_tokens = 0
    assert model._should_respawn(request) is False


@pytest.mark.asyncio
async def test_stream_eof_respawns_and_resets_sent_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Subprocess EOF is transport failure and resets Google send state."""
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    respawn_count = 0

    class _DeadProc:
        async def write_line(self, line: str) -> None:
            del line

        async def read_json_line(self, *, skip_non_json: bool = False) -> None:
            del skip_non_json

    class _HotSpare:
        active = cast(Subproc | None, object())

        async def acquire(self) -> Subproc:
            return cast(Subproc, _DeadProc())

        async def respawn_after_transport_failure(self) -> Subproc:
            nonlocal respawn_count
            respawn_count += 1
            return cast(Subproc, _DeadProc())

    model._system_hash = hash_system("")
    model._session_id = "session"
    monkeypatch.setattr(model, "_hot_spare", _HotSpare())
    request = ModelRequest(messages=[UserMessage(text="hi")])
    with pytest.raises(Exception, match="stdout closed"):
        await model.stream(request)
    assert respawn_count == 1
    assert model._last_sent_index == 0
    assert model._sent_history_head is None


@pytest.mark.asyncio
async def test_stream_read_timeout_respawns_and_resets_sent_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    respawn_count = 0

    class _StalledProc:
        async def write_line(self, line: str) -> None:
            del line

        async def read_json_line(self, *, skip_non_json: bool = False) -> object:
            del skip_non_json
            raise SubprocessTransportError("subprocess stdout idle timeout")

    class _HotSpare:
        active = cast(Subproc | None, object())

        async def acquire(self) -> Subproc:
            return cast(Subproc, _StalledProc())

        async def respawn_after_transport_failure(self) -> Subproc:
            nonlocal respawn_count
            respawn_count += 1
            return cast(Subproc, _StalledProc())

    model._system_hash = hash_system("")
    model._session_id = "session"
    monkeypatch.setattr(model, "_hot_spare", _HotSpare())
    request = ModelRequest(messages=[UserMessage(text="hi")])
    with pytest.raises(Exception, match="stdout idle timeout"):
        await model.stream(request)
    assert respawn_count == 1
    assert model._last_sent_index == 0
    assert model._sent_history_head is None


@pytest.mark.asyncio
async def test_stream_repeated_transport_failures_trip_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated subprocess transport failures stop respawning indefinitely."""
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    respawn_count = 0

    class _DeadProc:
        async def write_line(self, line: str) -> None:
            del line

        async def read_json_line(self, *, skip_non_json: bool = False) -> None:
            del skip_non_json

    class _HotSpare:
        active = cast(Subproc | None, object())

        async def acquire(self) -> Subproc:
            return cast(Subproc, _DeadProc())

        async def respawn_after_transport_failure(self) -> Subproc:
            nonlocal respawn_count
            respawn_count += 1
            if respawn_count == 2:
                raise RuntimeError("transport failure budget exhausted")
            return cast(Subproc, _DeadProc())

    model._system_hash = hash_system("")
    model._session_id = "session"
    monkeypatch.setattr(model, "_hot_spare", _HotSpare())
    request = ModelRequest(messages=[UserMessage(text="hi")])

    with pytest.raises(Exception, match="stdout closed"):
        await model.stream(request)
    with pytest.raises(RuntimeError, match="transport failure budget exhausted"):
        await model.stream(request)

    assert respawn_count == 2
    assert model._last_sent_index == 0
    assert model._sent_history_head is None


def test_should_respawn_skips_when_no_active() -> None:
    """Before the first acquire, the model has no active and skips respawn."""
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    assert model._should_respawn(ModelRequest(messages=[UserMessage(text="hi")])) is (
        False
    )


@pytest.mark.asyncio
async def test_stream_system_change_discards_warmed_old_system_spare(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    spawned_systems: list[str] = []
    used_systems: list[str] = []
    warmed = asyncio.Event()

    class _Proc:
        async def close(self) -> None:
            pass

    async def spawn_initialized() -> Subproc:
        proc = cast(Subproc, _Proc())
        spawned_systems.append(model._pending_system)
        if len(spawned_systems) == 2:
            warmed.set()
        state = _GoogleCLIProcState(
            proc=proc,
            session_id=f"session-{model._pending_system}",
            tmpdir=Path.cwd(),
            system_hash=hash_system(model._pending_system),
        )
        _GoogleCLIModel._attach_proc_state(state)
        return proc

    async def send_prompt(
        proc: Subproc,
        prompt_blocks: list[dict[str, MutablePlainTree]],
        text_parts: list[str],
        thinking_parts: list[str],
        publish: Callable[[RuntimeEvent], None] | None,
    ) -> str | None:
        del proc, prompt_blocks, text_parts, thinking_parts, publish
        used_systems.append(model._system_hash)
        return "end_turn"

    monkeypatch.setattr(model, "_send_prompt", send_prompt)
    model._hot_spare = HotSpare(spawn_initialized)

    _ = await model.stream(ModelRequest(messages=[UserMessage(text="a")], system="A"))
    await warmed.wait()
    _ = await model.stream(ModelRequest(messages=[UserMessage(text="b")], system="B"))

    assert used_systems == [hash_system("A"), hash_system("B")]
    assert spawned_systems == ["A", "A", "B"]
    await model.close()


@pytest.mark.asyncio
async def test_hot_spare_warmup_does_not_overwrite_active_session_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    ready = asyncio.Event()
    warmed = asyncio.Event()

    class _DummyProc:
        session_ids: list[str]

        def __init__(self) -> None:
            self.session_ids = []

        async def close(self) -> None:
            pass

    async def spawn_initialized() -> Subproc:
        proc = cast(Subproc, _DummyProc())
        session_id = "active-session"
        if not ready.is_set():
            ready.set()
        else:
            session_id = "spare-session"
            warmed.set()
        state = _GoogleCLIProcState(
            proc=proc,
            session_id=session_id,
            tmpdir=Path.cwd(),
            system_hash=hash_system(""),
        )
        _GoogleCLIModel._attach_proc_state(state)
        return proc

    async def send_prompt(
        proc: Subproc,
        prompt_blocks: list[dict[str, MutablePlainTree]],
        text_parts: list[str],
        thinking_parts: list[str],
        publish: Callable[[RuntimeEvent], None] | None,
    ) -> str | None:
        del prompt_blocks, text_parts, thinking_parts, publish
        cast(_DummyProc, proc).session_ids.append(model._session_id)
        return "end_turn"

    monkeypatch.setattr(model, "_send_prompt", send_prompt)
    model._hot_spare = HotSpare(spawn_initialized)
    proc = await model._hot_spare.acquire()
    await warmed.wait()
    response = await model.stream(ModelRequest(messages=[UserMessage(text="hi")]))
    assert cast(_DummyProc, proc).session_ids == ["active-session"]
    assert response.message_id == "active-session"
    await model.close()


@pytest.mark.asyncio
async def test_exchange_turn_skips_assistant_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Respawn replay sends only user-like entries to the CLI subprocess."""
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    prompts: list[list[dict[str, MutablePlainTree]]] = []

    async def send_prompt(
        proc: Subproc,
        prompt_blocks: list[dict[str, MutablePlainTree]],
        text_parts: list[str],
        thinking_parts: list[str],
        publish: Callable[[RuntimeEvent], None] | None,
    ) -> str | None:
        del proc, text_parts, thinking_parts, publish
        prompts.append(prompt_blocks)
        return "end_turn"

    monkeypatch.setattr(model, "_send_prompt", send_prompt)
    response = await model._exchange_turn(
        cast(Subproc, object()),
        ModelRequest(
            messages=[
                UserMessage(text="first"),
                AssistantMessage(text="hidden"),
                UserMessage(text="second"),
            ],
        ),
        publish=None,
    )

    assert [[block["text"] for block in prompt] for prompt in prompts] == [
        ["first\n\nsecond"],
    ]
    assert response.stop_reason == "model_finished"


@pytest.mark.asyncio
async def test_stream_returns_response_without_credential_writeback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    success_count = 0

    class _Proc:
        pass

    class _HotSpare:
        active = cast(Subproc | None, object())

        async def acquire(self) -> Subproc:
            return cast(Subproc, _Proc())

        def record_success(self) -> None:
            nonlocal success_count
            success_count += 1

    async def send_prompt(
        proc: Subproc,
        prompt_blocks: list[dict[str, MutablePlainTree]],
        text_parts: list[str],
        thinking_parts: list[str],
        publish: Callable[[RuntimeEvent], None] | None,
    ) -> str | None:
        del proc, prompt_blocks, thinking_parts, publish
        text_parts.append("ok")
        return "end_turn"

    user = UserMessage(text="hi")
    model._system_hash = hash_system("")
    model._session_id = "session"
    monkeypatch.setattr(model, "_hot_spare", _HotSpare())
    monkeypatch.setattr(model, "_send_prompt", send_prompt)

    response = await model.stream(ModelRequest(messages=[user]))

    assert response.message.text == "ok"
    assert response.message_id == "session"
    assert success_count == 1
    assert model._last_sent_index == 1
    assert model._sent_history_head is user
    assert model._turn_count == 1


@pytest.mark.asyncio
async def test_respawn_resets_active_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    model._turn_count = 100
    model._last_input_tokens = model.limits.max_request_tokens

    class _Proc:
        pass

    class _HotSpare:
        active = cast(Subproc | None, object())

        async def acquire(self) -> Subproc:
            return cast(Subproc, _Proc())

        async def respawn(self) -> Subproc:
            return cast(Subproc, _Proc())

        def record_success(self) -> None:
            pass

    async def send_prompt(
        proc: Subproc,
        prompt_blocks: list[dict[str, MutablePlainTree]],
        text_parts: list[str],
        thinking_parts: list[str],
        publish: Callable[[RuntimeEvent], None] | None,
    ) -> str | None:
        del proc, prompt_blocks, text_parts, thinking_parts, publish
        return "end_turn"

    model._system_hash = hash_system("")
    model._session_id = "session"
    monkeypatch.setattr(model, "_hot_spare", _HotSpare())
    monkeypatch.setattr(model, "_send_prompt", send_prompt)

    request = ModelRequest(messages=[UserMessage(text="hi")])
    response = await model.stream(request)

    assert response.message_id == "session"
    assert model._turn_count == 1
    assert model._last_input_tokens == 0
    assert model._should_respawn(request) is False


@pytest.mark.asyncio
async def test_exchange_turn_returns_current_output_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    text_callbacks: list[str] = []

    async def send_prompt(
        proc: Subproc,
        prompt_blocks: list[dict[str, MutablePlainTree]],
        text_parts: list[str],
        thinking_parts: list[str],
        publish: Callable[[RuntimeEvent], None] | None,
    ) -> str | None:
        del proc, thinking_parts
        text = str(prompt_blocks[0]["text"])
        text_parts.append(text)
        if publish is not None:
            publish(ModelResponsePartial(text))
        return "end_turn"

    def _sink(ev: RuntimeEvent) -> None:
        if isinstance(ev, ModelResponsePartial):
            text_callbacks.append(ev.text)

    monkeypatch.setattr(model, "_send_prompt", send_prompt)
    response = await model._exchange_turn(
        cast(Subproc, object()),
        ModelRequest(messages=[UserMessage(text="first"), UserMessage(text="current")]),
        publish=_sink,
    )

    assert response.message.text == "first\n\ncurrent"
    assert text_callbacks == ["first\n\ncurrent"]


@pytest.mark.asyncio
async def test_terminal_json_rpc_error_respawns_and_resets_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    respawn_count = 0

    class _ErrorProc:
        async def write_line(self, line: str) -> None:
            del line

        async def read_json_line(
            self,
            *,
            skip_non_json: bool = False,
        ) -> dict[str, MutablePlainTree]:
            del skip_non_json
            return {"id": 1, "error": {"message": "boom"}}

    class _HotSpare:
        active = cast(Subproc | None, object())

        async def acquire(self) -> Subproc:
            return cast(Subproc, _ErrorProc())

        async def respawn_after_transport_failure(self) -> Subproc:
            nonlocal respawn_count
            respawn_count += 1
            return cast(Subproc, _ErrorProc())

    user = UserMessage(text="hi")
    model._system_hash = hash_system("")
    model._session_id = "session"
    model._last_sent_index = 0
    model._sent_history_head = user
    model._turn_count = 9
    model._last_input_tokens = 123
    monkeypatch.setattr(model, "_hot_spare", _HotSpare())

    with pytest.raises(SubprocessTransportError, match="JSON-RPC error"):
        _ = await model.stream(ModelRequest(messages=[user]))

    assert respawn_count == 1
    assert model._last_sent_index == 0
    assert model._sent_history_head is None
    assert model._turn_count == 0
    assert model._last_input_tokens == 0


@pytest.mark.asyncio
async def test_exchange_turn_skips_tool_results_from_api_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """History recorded by an API provider carries tool results; skip them."""
    model = GoogleCLI().model("gemini-2.5-flash")
    prompts: list[list[dict[str, MutablePlainTree]]] = []

    async def send_prompt(
        proc: Subproc,
        prompt_blocks: list[dict[str, MutablePlainTree]],
        text_parts: list[str],
        thinking_parts: list[str],
        publish: Callable[[RuntimeEvent], None] | None,
    ) -> str | None:
        del proc, text_parts, thinking_parts, publish
        prompts.append(prompt_blocks)
        return "end_turn"

    monkeypatch.setattr(model, "_send_prompt", send_prompt)
    await model._exchange_turn(
        cast(Subproc, object()),
        ModelRequest(
            messages=[
                UserMessage(text="list files"),
                AssistantMessage(
                    text="",
                    tool_calls=(ToolCall(id="c1", name="Bash", args={}),),
                ),
                ToolResult(call_id="c1", content="a.txt"),
                UserMessage(text="thanks"),
            ],
        ),
        publish=None,
    )
    assert [[block["text"] for block in prompt] for prompt in prompts] == [
        ["list files\n\nthanks"],
    ]


@pytest.mark.asyncio
async def test_exchange_turn_without_input_does_not_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = GoogleCLI().model("gemini-2.5-flash")
    model._last_sent_index = 1

    async def send_prompt(*args: object) -> str | None:
        raise AssertionError(f"unexpected prompt {args!r}")

    monkeypatch.setattr(model, "_send_prompt", send_prompt)
    response = await model._exchange_turn(
        cast(Subproc, object()),
        ModelRequest(messages=[UserMessage(text="seen")]),
        publish=None,
    )
    assert response.message.text == ""


@pytest.mark.asyncio
async def test_cancelled_prompt_respawns_before_next_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cancelled prompt keeps streaming in the CLI; that process is retired."""
    model = GoogleCLI().model("gemini-2.5-flash")
    respawns = 0

    class _CancelProc:
        async def write_line(self, line: str) -> None:
            del line

        async def read_json_line(self, *, skip_non_json: bool = False) -> object:
            del skip_non_json
            raise asyncio.CancelledError

    class _HotSpare:
        active = cast(Subproc | None, object())

        async def acquire(self) -> Subproc:
            return cast(Subproc, _CancelProc())

        async def respawn_after_transport_failure(self) -> Subproc:
            nonlocal respawns
            respawns += 1
            return cast(Subproc, _CancelProc())

    user = UserMessage(text="hi")
    model._system_hash = hash_system("")
    model._session_id = "session"
    model._sent_history_head = user
    monkeypatch.setattr(model, "_hot_spare", _HotSpare())
    with pytest.raises(asyncio.CancelledError):
        await model.stream(ModelRequest(messages=[user]))
    assert respawns == 1
    assert model._last_sent_index == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply",
    [None, {"id": 1, "error": {"message": "auth failed"}}],
)
async def test_handshake_failure_is_a_transport_error(
    reply: dict[str, MutablePlainTree] | None,
) -> None:
    """A dead or erroring handshake must take the respawn path."""

    class _Proc:
        async def write_line(self, line: str) -> None:
            del line

        async def read_json_line(self, *, skip_non_json: bool = False) -> object:
            del skip_non_json
            return reply

    with pytest.raises(SubprocessTransportError):
        await cli._rpc_call(cast(Subproc, _Proc()), 1, "initialize", {})


def test_build_response_estimates_tokens_and_cost() -> None:
    """The Code Assist response is reconstructed with estimated usage + cost."""
    provider = GoogleCLI()
    model = provider.model("gemini-2.5-flash")
    request = ModelRequest(messages=[UserMessage(text="aaaa" * 4)])
    response = model._build_response(
        text_parts=["hello"],
        thinking_parts=["thinking..."],
        stop_reason="end_turn",
        request=request,
    )
    assert response.message.text == "hello"
    assert response.stop_reason == "model_finished"
    assert response.tokens.request == 4
    assert response.tokens.response == 1
    assert response.total_cost >= 0.0
    assert len(response.message.thinking_blocks) == 1


@pytest.mark.parametrize(
    ("raw", "accepted"),
    [("end_turn", True), ("max_turn_requests", True), ("refusal", False)],
)
def test_build_response_normalizes_acp_stop_reasons(raw: str, accepted: bool) -> None:
    """ACP reports ``end_turn`` for every normal turn; it must not fail the turn."""
    model = GoogleCLI().model("gemini-2.5-flash")
    response = model._build_response(
        text_parts=["hello"],
        thinking_parts=[],
        stop_reason=raw,
        request=ModelRequest(messages=[UserMessage(text="hi")]),
    )
    if accepted:
        validate_model_response(response)
    else:
        with pytest.raises(ModelTerminationError):
            validate_model_response(response)


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
