"""``GoogleCLI`` provider: wraps the user's installed ``gemini`` CLI in ACP mode.

Speaks the Agent Client Protocol (JSON-RPC over NDJSON) to a
persistent ``gemini --experimental-acp`` subprocess that rides the
user's Gemini Code Assist subscription. Tool calls flow through the
in-process MCP bridge (``providers.lib.mcp_bridge``); the CLI
registers the bridge via ``session/new.params.mcpServers`` with
``type: "http"`` and dispatches sagent tools by URL.

The terminal ACP event for each user turn is the JSON-RPC response to
``session/prompt`` (matched by id). Streaming chunks arrive as
``session/update`` notifications with ``agent_message_chunk`` /
``agent_thought_chunk`` payloads in between.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast, override

import asyncio
import base64
import dataclasses
import json
import os
import shutil
import subprocess
import tempfile

from sagent.catalog.google import cli, models
from sagent.catalog.table import ModelCatalog
from sagent.lib.codec import MutablePlainTree, from_plain
from sagent.lib.userdirs import data_dir
from sagent.providers.google.api import Google, gemini_image_tokens
from sagent.providers.lib.cli_respawn import (
    CLISubprocessModel,
    empty_turn_response,
    hash_system,
)
from sagent.providers.lib.errors import (
    error_status_code,
    is_context_overflow_text,
    is_request_too_large,
)
from sagent.providers.lib.hotspare import HotSpare
from sagent.providers.lib.mcp_bridge import ToolsBridge
from sagent.providers.lib.model_base import ModelDefaults
from sagent.providers.lib.oauth import resolve_account
from sagent.providers.lib.stop_reason import normalize_stop_reason
from sagent.providers.lib.subproc import (
    Subproc,
    SubprocessTransportError,
)
from sagent.types.cost import TokenCount
from sagent.types.model import (
    ModelRequest,
    ModelResponse,
)
from sagent.types.runtime import (
    AgentSendMessage,
    AssistantMessage,
    ModelResponsePartial,
    ModelResponseThinking,
    RuntimeEvent,
    UserMessage,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from sagent.lib import image
    from sagent.types.capability import ModelCapability, ModelSettings
else:
    from wrapt import lazy_import

    image = lazy_import("sagent.lib.image")


def _gemini_cli_home(account: str | None) -> Path | None:
    """Return the native Gemini CLI home for an account.

    The default account leaves ``GEMINI_CLI_HOME`` untouched so the CLI uses
    the operator's existing login. Named accounts get persistent sagent-owned
    homes, which lets the Gemini CLI own the OAuth files and refresh lifecycle.
    """
    name = resolve_account(account)
    if name == "default":
        return None
    return data_dir() / "rekursiv-ai" / "sagent" / "gemini" / name


class GoogleCLI:
    """Provider that drives the user's installed ``gemini`` CLI subprocess.

    Shares the Gemini model catalog. Auth is owned by the Gemini CLI. The default
    account uses the operator's native CLI configuration; named accounts use a
    sagent-owned ``GEMINI_CLI_HOME``.
    Cost figures are estimated from public per-token pricing since ACP
    does not surface per-turn usage on ``session/prompt`` responses.
    """

    catalog = ModelCatalog(models=models(), transport=cli())

    def __init__(self, *, account: str | None = None) -> None:
        self._account = account

    @classmethod
    def from_key(cls, api_key: str) -> Google:
        """Create an API-key provider (delegates to :class:`Google`).

        The CLI-wrapping provider is incompatible with API-key auth, so
        this returns a plain :class:`Google` instance.

        Args:
          api_key: Google AI Studio API key.

        Returns:
          provider: ``Google`` provider instance.

        """
        return Google.from_key(api_key)

    @classmethod
    def from_credentials(cls, *, account: str | None = None) -> GoogleCLI:
        """Build a provider that uses the local ``gemini`` CLI's login.

        Args:
          account: Named account slot. ``None`` uses the native CLI account.

        Returns:
          provider: Configured CLI-wrapping provider.

        Raises:
          RuntimeError: If ``gemini`` is not on ``PATH``.

        """
        _gemini_cli_home(account)
        if shutil.which("gemini") is None:
            raise RuntimeError(
                "GoogleCLI: `gemini` is not on PATH; install the Gemini CLI.",
            )
        return cls(account=account)

    @classmethod
    def login(cls, *, account: str | None = None) -> None:
        """Run the Gemini CLI's native login command for an account.

        Args:
          account: Account slot. ``None`` uses the operator's native CLI
              configuration; named accounts get an isolated
              ``GEMINI_CLI_HOME``.

        Raises:
          RuntimeError: If the CLI is missing or login fails.

        """
        binary = shutil.which("gemini")
        if binary is None:
            raise RuntimeError(
                "GoogleCLI: `gemini` is not on PATH; install the Gemini CLI.",
            )
        proc = subprocess.run(  # noqa: S603 -- The CLI argv uses a resolved binary and fixed subcommand.
            [binary, "login"],
            env=_google_subprocess_env(_gemini_cli_home(account)),
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Gemini CLI login failed with exit code {proc.returncode}.",
            )

    def model(
        self,
        model_id: str | None = None,
    ) -> _GoogleCLIModel:
        """Build a CLI-backed model.

        Args:
          model_id: Gemini model id; ``None`` uses catalog key ``default``.

        Returns:
          model: Backend wrapping a managed ``gemini --experimental-acp`` subprocess.

        Raises:
          ValueError: If the resolved id is not in the catalog.

        """
        mid = model_id if model_id is not None else "default"
        capability, settings = self.catalog.resolve(mid)
        return _GoogleCLIModel(
            provider=self,
            capability=capability,
            settings=settings,
        )

    async def close_sdk(self) -> None:
        """Do nothing: each model owns and closes its own subprocess."""

    @property
    def account(self) -> str | None:
        """Named account slot, ``None`` for the operator's native login."""
        return self._account


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _GoogleCLIProcState:
    """ACP session state owned by one Gemini CLI subprocess."""

    proc: Subproc
    session_id: str
    tmpdir: Path
    system_hash: str


def _google_subprocess_env(
    config_home: Path | None,
    *,
    system_path: Path | None = None,
    settings_path: Path | None = None,
) -> dict[str, str]:
    """Build the env for ``gemini`` without copying or parsing OAuth files."""
    drop = {
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_GENAI_USE_GCA",
        "GEMINI_CLI_USE_COMPUTE_ADC",
        "GOOGLE_APPLICATION_CREDENTIALS",
    }
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(
        {
            "GEMINI_DEFAULT_AUTH_TYPE": "oauth-personal",
            "GEMINI_FORCE_FILE_STORAGE": "true",
            "GEMINI_YOLO_MODE": "true",
            "GEMINI_FOLDER_TRUST": "true",
            "GEMINI_CLI_TRUST_WORKSPACE": "true",
            "GEMINI_CLI_NO_RELAUNCH": "true",
            "GEMINI_STRICT_TELEMETRY_LIMITS": "true",
        },
    )
    if config_home is not None:
        env["GEMINI_CLI_HOME"] = str(config_home)
    if system_path is not None:
        env["GEMINI_SYSTEM_MD"] = str(system_path)
    if settings_path is not None:
        env["GEMINI_CLI_SYSTEM_SETTINGS_PATH"] = str(settings_path)
    return env


_GEMINI_SETTINGS: dict[str, MutablePlainTree] = {
    "security": {"auth": {"selectedType": "oauth-personal"}},
    "privacy": {"usageStatisticsEnabled": False},
    "telemetry": {
        "enabled": False,
        "logPrompts": False,
        "useCollector": False,
    },
    "general": {
        "checkpointing": {"enabled": False},
        "enableAutoUpdate": False,
        "enableAutoUpdateNotification": False,
        "enableNotifications": False,
    },
    "ui": {
        "hideTips": True,
        "hideBanner": True,
        "hideContextSummary": True,
        "hideSandboxStatus": True,
        "hideModelInfo": True,
        "showMemoryUsage": False,
    },
    "context": {
        "loadMemoryFromIncludeDirectories": False,
        "discoveryMaxDirs": 0,
    },
    "tools": {
        "excludeTools": ["*"],
        "useWriteTodos": False,
        "toolSandboxing": False,
        "blockGitExtensions": True,
        "allowedExtensions": [],
    },
    "mcp": {"allowed": [], "excluded": ["*"]},
    "advanced": {
        "autoConfigureMemory": False,
        "agentSessionNoninteractiveEnabled": False,
        "agentSessionInteractiveEnabled": False,
        "extensionManagement": False,
        "extensionConfig": False,
        "extensionRegistry": False,
        "extensionReloading": False,
        "jitContext": False,
        "taskTracker": False,
        "modelSteering": False,
        "memoryV2": False,
        "autoMemory": False,
        "contextManagement": False,
    },
    "experimental": {"compressionThreshold": 1.0},
}


async def _rpc_call(
    proc: Subproc,
    request_id: int,
    method: str,
    params: dict[str, object],
) -> dict[str, MutablePlainTree]:
    """Send one JSON-RPC request and return the matching ``result`` payload."""
    await _rpc_send(proc, request_id, method, params)
    while True:
        msg = await proc.read_json_line(skip_non_json=True)
        if msg is None:
            raise SubprocessTransportError(
                f"GoogleCLI: stdout closed waiting for {method}",
            )
        if msg.get("id") == request_id:
            if "error" in msg:
                raise SubprocessTransportError(
                    f"GoogleCLI: {method} error: {msg['error']}",
                )
            return cast(dict[str, MutablePlainTree], msg.get("result") or {})


async def _rpc_send(
    proc: Subproc,
    request_id: int,
    method: str,
    params: dict[str, object],
) -> None:
    """Serialise one JSON-RPC request and write it to the subprocess stdin."""
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    }
    await proc.write_line(json.dumps(payload))


def _user_prompt_blocks(
    entry: AgentSendMessage | UserMessage,
    max_image_dim: int,
    max_image_bytes: int,
) -> list[dict[str, MutablePlainTree]]:
    """Build ACP ``[{type:text}|{type:image}]`` blocks for a ``UserMessage``."""
    blocks: list[dict[str, MutablePlainTree]] = []
    if entry.text:
        blocks.append({"type": "text", "text": entry.text})
    for att in entry.attachments:
        if not att.descriptor.startswith("image/"):
            continue
        raw, mime = image.resize(
            att.data,
            max_dim=max_image_dim,
            max_bytes=max_image_bytes,
        )
        blocks.append(
            {
                "type": "image",
                "data": base64.b64encode(raw).decode(),
                "mimeType": mime,
            },
        )
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    return blocks


def _dispatch_session_update(
    params: dict[str, MutablePlainTree],
    text_parts: list[str],
    thinking_parts: list[str],
    publish: Callable[[RuntimeEvent], None] | None,
) -> None:
    """Route one ``session/update`` notification payload."""
    update = cast(dict[str, MutablePlainTree], params.get("update") or {})
    kind = update.get("sessionUpdate")
    if kind == "agent_message_chunk":
        text = cast(
            str,
            cast(dict[str, MutablePlainTree], update.get("content") or {}).get("text")
            or "",
        )
        if text:
            text_parts.append(text)
            if publish is not None:
                publish(ModelResponsePartial(text))
    elif kind == "agent_thought_chunk":
        text = cast(
            str,
            cast(dict[str, MutablePlainTree], update.get("content") or {}).get("text")
            or "",
        )
        if text:
            thinking_parts.append(text)
            if publish is not None:
                publish(ModelResponseThinking(text))


class _GoogleCLIModel(CLISubprocessModel, ModelDefaults):
    """``gemini --experimental-acp`` subprocess wrapped as a sagent ``Model``.

    Args:
      provider: Owning :class:`GoogleCLI`.
      capability: Resolved model and transport capabilities.
      settings: Selections validated against ``capability``.

    """

    def __init__(
        self,
        *,
        provider: GoogleCLI,
        capability: ModelCapability,
        settings: ModelSettings,
    ) -> None:
        super().__init__()
        self._provider = provider
        self._capability = capability
        self._settings = settings
        self._tools_bridge: ToolsBridge | None = None
        self._next_rpc_id = 1
        self._session_id: str = ""
        self._hot_spare = HotSpare(
            self._spawn_initialized_proc,
            close_partial=self._close_warming_proc,
        )
        self._pending_system: str = ""

    @property
    @override
    def capability(self) -> ModelCapability:
        """What this model offers on this transport."""
        return self._capability

    @property
    @override
    def settings(self) -> ModelSettings:
        """What this instance chose."""
        return self._settings

    @override
    def approx_text_tokens(self, text: str) -> int:
        """Estimate locally from the catalog's measured character ratio."""
        return int(len(text) / self.capability.approx_chars_per_token)

    @override
    def approx_image_tokens(self, data: bytes) -> int:
        """Local estimate via Gemini's tile heuristic."""
        return gemini_image_tokens(data)

    def is_context_overflow(self, error: Exception) -> bool:
        """Classify whether an error means the prompt exceeded the token window.

        Excludes the request-byte wire-limit via the shared classifier first
        (uniform with the HTTP providers): a byte-limit error routes to
        byte-overflow recovery, not the ``/model`` larger-window remediation
        the byte ceiling ignores. Without this guard the bare ``"too large"``
        marker would swallow a byte-limit error into token overflow.

        Args:
          error: Exception raised by the call path.

        Returns:
          overflow: ``True`` for known token-overflow markers in the message.

        """
        if is_request_too_large(error_status_code(error), str(error)):
            return False
        return is_context_overflow_text(str(error))

    @override
    async def stream(
        self,
        request: ModelRequest,
        publish: Callable[[RuntimeEvent], None] | None = None,
    ) -> ModelResponse:
        """Drive one user turn through the ``gemini`` ACP subprocess.

        Args:
          request: Conversation + tools + system prompt for the turn.
          publish: Per-chunk event sink; ``None`` disables live streaming.

        Returns:
          response: Parsed model response with cost estimated from pricing.

        Raises:
          RuntimeError: The subprocess closed before responding, or the
              server returned a JSON-RPC ``error`` field.

        """
        self._pending_system = request.system
        if self._hot_spare.active is not None:
            self._promote_proc_state(self._hot_spare.active)
        if self._should_respawn(request):
            if hash_system(request.system) != self._system_hash:
                await self._hot_spare.discard_spare()
            await self._hot_spare.respawn()
            self._reset_active_state()
        proc = await self._hot_spare.acquire()
        self._promote_proc_state(proc)
        self._sync_tools_bridge(request)

        try:
            response = await self._exchange_turn(proc, request, publish)
            self._last_sent_index = len(request.messages)
        except (SubprocessTransportError, asyncio.CancelledError):
            # A cancelled ``session/prompt`` keeps running in the CLI and its
            # ``session/update`` chunks would leak into the next turn's read,
            # so the subprocess is replaced exactly as after a transport error.
            self._reset_active_state()
            await self._hot_spare.respawn_after_transport_failure()
            raise
        self._hot_spare.record_success()
        self._turn_count += 1
        return response

    @override
    async def close(self) -> None:
        """Tear down the subprocess pool and the MCP bridge."""
        await self._hot_spare.close()
        if self._tools_bridge is not None:
            await self._tools_bridge.stop()
            self._tools_bridge = None

    def _should_respawn(self, request: ModelRequest) -> bool:
        """Inspect the trigger list (§1.4) for this request."""
        if self._hot_spare.active is None:
            return False
        return self._respawn_due(
            request.messages,
            system=request.system,
            max_request_tokens=self.limits.max_request_tokens,
        )

    def _sync_tools_bridge(self, request: ModelRequest) -> None:
        """Refresh the MCP bridge's tool registry to match the request."""
        if self._tools_bridge is not None:
            self._tools_bridge.update_tools(list(request.tools))

    def _promote_proc_state(self, proc: Subproc) -> None:
        """Copy state from the active subprocess wrapper onto model fields."""
        state = self._proc_state(proc)
        if state is None:
            return
        self._session_id = state.session_id
        self._system_hash = state.system_hash

    @classmethod
    def _attach_proc_state(cls, state: _GoogleCLIProcState) -> None:
        """Attach Google CLI state to its subprocess wrapper."""
        state.proc.sagent_google_cli_state = state

    @classmethod
    def _proc_state(cls, proc: Subproc) -> _GoogleCLIProcState | None:
        """Return Google CLI state carried by ``proc``, if any."""
        state = getattr(proc, "sagent_google_cli_state", None)
        if isinstance(state, _GoogleCLIProcState):
            return state
        return None

    async def _exchange_turn(
        self,
        proc: Subproc,
        request: ModelRequest,
        publish: Callable[[RuntimeEvent], None] | None,
    ) -> ModelResponse:
        """Send the unseen input as one ``session/prompt`` and build the reply."""
        entry = self._unsent_input(request.messages)
        if entry is None:
            return empty_turn_response()
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        stop_reason = await self._send_prompt(
            proc,
            _user_prompt_blocks(
                entry,
                self.limits.max_image_edge_px,
                self.limits.max_image_bytes,
            ),
            text_parts,
            thinking_parts,
            publish,
        )
        return self._build_response(text_parts, thinking_parts, stop_reason, request)

    async def _send_prompt(
        self,
        proc: Subproc,
        prompt_blocks: list[dict[str, MutablePlainTree]],
        text_parts: list[str],
        thinking_parts: list[str],
        publish: Callable[[RuntimeEvent], None] | None,
    ) -> str | None:
        """Issue one ``session/prompt`` RPC and drain until its response."""
        request_id = self._allocate_rpc_id()
        await _rpc_send(
            proc,
            request_id,
            "session/prompt",
            {"sessionId": self._session_id, "prompt": prompt_blocks},
        )
        while True:
            msg = await proc.read_json_line(skip_non_json=True)
            if msg is None:
                raise SubprocessTransportError(
                    "GoogleCLI: subprocess stdout closed before response",
                )
            if msg.get("id") == request_id:
                if "error" in msg:
                    raise SubprocessTransportError(
                        f"GoogleCLI: JSON-RPC error: {msg['error']}",
                    )
                result = cast(dict[str, MutablePlainTree], msg.get("result") or {})
                return cast(str | None, result.get("stopReason"))
            if msg.get("method") == "session/update":
                _dispatch_session_update(
                    cast(dict[str, MutablePlainTree], msg.get("params") or {}),
                    text_parts,
                    thinking_parts,
                    publish,
                )

    def _build_response(
        self,
        text_parts: list[str],
        thinking_parts: list[str],
        stop_reason: str | None,
        request: ModelRequest,
    ) -> ModelResponse:
        """Assemble a ``ModelResponse`` with estimated token usage and cost."""
        full_text = "".join(text_parts)
        input_tokens = self.approx_request_tokens(request)
        output_tokens = self.approx_text_tokens(full_text)
        tokens = TokenCount(request=input_tokens, response=output_tokens)
        self._last_input_tokens = input_tokens
        thinking_blocks = (
            ({"type": "thinking", "thinking": "".join(thinking_parts)},)
            if thinking_parts
            else ()
        )
        return ModelResponse(
            message=AssistantMessage(
                text=full_text,
                thinking_blocks=thinking_blocks,
                tool_calls=(),
            ),
            tokens=tokens,
            stop_reason=normalize_stop_reason(
                stop_reason,
                kind="acp",
                has_tool_use=False,
            ),
            message_id=self._session_id,
            request_id=self._session_id,
            spend=self.spend(tokens),
        )

    async def _spawn_initialized_proc(self) -> Subproc:
        """Spawn ``gemini`` and attach isolated ACP state to the subprocess."""
        state = await self._spawn_initialized()
        self._attach_proc_state(state)
        return state.proc

    async def _spawn_initialized(self) -> _GoogleCLIProcState:
        """Spawn ``gemini`` and run the ACP handshake to a live ``sessionId``."""
        if self._tools_bridge is None:
            self._tools_bridge = ToolsBridge(tools=[])
            await self._tools_bridge.start()
        system = self._pending_system
        account = self._provider.account
        tmpdir = Path(tempfile.mkdtemp(prefix="sagent-google-cli-"))
        workdir = tmpdir / "workdir"
        system_path = tmpdir / "system.md"
        settings_path = tmpdir / "settings.json"
        try:
            workdir.mkdir()
            system_path.write_text(system, encoding="utf-8")
            settings_path.write_text(json.dumps(_GEMINI_SETTINGS), encoding="utf-8")
        except BaseException:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise
        system_hash = hash_system(system)
        model_id = self.capability.model_id
        proc = Subproc(
            ["gemini", "--experimental-acp", "--model", model_id],
            env=_google_subprocess_env(
                _gemini_cli_home(account),
                system_path=system_path,
                settings_path=settings_path,
            ),
            tmpdir=tmpdir,
            cwd=workdir,
        )
        self._warming_proc = proc
        try:
            await proc.start()
            session_id = await self._acp_handshake(proc, workdir)
        except BaseException:
            await proc.close()
            self._warming_proc = None
            raise
        state = _GoogleCLIProcState(
            proc=proc,
            session_id=session_id,
            tmpdir=tmpdir,
            system_hash=system_hash,
        )
        self._warming_proc = None
        return state

    async def _acp_handshake(self, proc: Subproc, workdir: Path) -> str:
        """Run ``initialize`` → ``authenticate`` → ``session/new`` (§3.2)."""
        if self._tools_bridge is None:
            raise ValueError("Expected self._tools_bridge is not None.")
        await _rpc_call(
            proc,
            self._allocate_rpc_id(),
            "initialize",
            {"protocolVersion": 1, "clientCapabilities": {}},
        )
        await _rpc_call(
            proc,
            self._allocate_rpc_id(),
            "authenticate",
            {"methodId": "oauth-personal"},
        )
        result = await _rpc_call(
            proc,
            self._allocate_rpc_id(),
            "session/new",
            {
                "cwd": str(workdir),
                "mcpServers": [
                    {
                        "name": self._tools_bridge.server_name,
                        "type": "http",
                        "url": self._tools_bridge.url,
                        "headers": [],
                    },
                ],
            },
        )
        session_id = from_plain(result.get("sessionId"), str, default="")
        if not session_id:
            raise RuntimeError(
                f"GoogleCLI: session/new returned no sessionId: {result}",
            )
        return session_id

    def _allocate_rpc_id(self) -> int:
        """Allocate the next JSON-RPC id, incrementing in place."""
        rid = self._next_rpc_id
        self._next_rpc_id += 1
        return rid
