"""Tests for ``providers.moonshot``: Moonshot OpenAI-compat surface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest

from sagent.bin.cli import DEFAULT_TOOLS, resolve_tools
from sagent.lib.codec import MutablePlainTree, from_plain, immutable
from sagent.providers.moonshot.api import Moonshot, _MoonshotModel
from sagent.providers.moonshot.wire import _DROPPED_SCHEMA_KEYS
from sagent.types.capability import (
    ModelSettings,
    ThinkingBudget,
    ThinkingEffort,
)
from sagent.types.model import ModelRequest
from sagent.types.runtime import ToolResult, UserMessage


if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from sagent.lib.codec import PlainTree
    from sagent.types.tools import Tool


def test_moonshot_from_key() -> None:
    p = Moonshot.from_key("sk-moon-test")
    assert p.api_key == "sk-moon-test"
    assert p.base_url == "https://api.moonshot.ai/v1"


def test_moonshot_from_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="API key not configured"):
        Moonshot.from_env()


def test_moonshot_from_env_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MOONSHOT_API_KEY", "sk-moon-env")
    p = Moonshot.from_env()
    assert p.api_key == "sk-moon-env"


def test_moonshot_default_model() -> None:
    p = Moonshot.from_key("k")
    m = p.model()
    assert m.capability.model_id == p.catalog.resolve("default")[0].model_id
    assert m.capability.thinking.output == frozenset({"none", "text"})


def test_moonshot_unknown_model_raises() -> None:
    p = Moonshot.from_key("k")
    with pytest.raises(ValueError, match="Unknown model"):
        _ = p.model("not-kimi")


def test_moonshot_known_models_have_pricing_and_limits() -> None:
    p = Moonshot.from_key("k")
    for mid in Moonshot.catalog.models:
        m = p.model(mid)
        assert m.limits.max_request_tokens > 0
        assert m.limits.max_response_tokens > 0


def test_moonshot_base_url_override_via_from_key() -> None:
    p = Moonshot.from_key("k", base_url="http://localhost:8000/v1")
    assert p.base_url == "http://localhost:8000/v1"


def test_moonshot_offers_no_prompt_cache() -> None:
    m = Moonshot.from_key("k").model("kimi-k2.6")
    assert m.capability.cache_ttl_sec == frozenset({0.0})


def _body(
    model_id: str,
    *,
    budget: ThinkingBudget | None = None,
    effort: ThinkingEffort | None = None,
    cap: int | None = None,
    tools: Sequence[Tool] = (),
) -> dict[str, MutablePlainTree]:
    m = cast(_MoonshotModel, Moonshot.from_key("k").model(model_id))
    settings = ModelSettings.narrowest(m.capability)
    if budget is not None:
        settings.thinking_budget = budget
    if effort is not None:
        settings.thinking_effort = effort
    m._settings = settings
    return m._build_body(
        ModelRequest(
            messages=[UserMessage(text="x")],
            max_response_tokens=cap,
            tools=tools,
        ),
        stream=False,
    )


@pytest.mark.parametrize("model_id", sorted(Moonshot.catalog.models))
def test_no_request_carries_the_fixed_temperature(model_id: str) -> None:
    assert "temperature" not in _body(model_id)


def test_k3_sends_max_verbatim_rather_than_openais_high() -> None:
    assert _body("kimi-k3", effort="max")["reasoning_effort"] == "max"


def test_k3_with_no_effort_leaves_the_server_default() -> None:
    body = _body("kimi-k3")
    assert "reasoning_effort" not in body
    assert "thinking" not in body


@pytest.mark.parametrize(
    ("budget", "wire"),
    [("none", "disabled"), ("auto", "enabled")],
)
def test_k2_6_sends_its_thinking_switch(budget: ThinkingBudget, wire: str) -> None:
    assert _body("kimi-k2.6", budget=budget)["thinking"] == {"type": wire}


def test_k2_7_code_is_never_sent_the_thinking_parameter() -> None:
    assert "thinking" not in _body("kimi-k2.7-code")


def test_the_wire_carries_the_vendor_id() -> None:
    assert _body("kimi-k2.7-code-highspeed")["model"] == "kimi-k2.7-code-highspeed"


@pytest.mark.parametrize("model_id", sorted(Moonshot.catalog.models))
def test_the_output_cap_never_rides_the_deprecated_field(model_id: str) -> None:
    """``max_tokens`` is "Deprecated, please refer to max_completion_tokens".

    The chat API lists the deprecated spelling on every request schema
    (ChatRequestCommon for K3, ChatRequestBase for K2.x), and the base
    builder picks the field by OpenAI's effort rule, so a K2 row would
    otherwise send the one Kimi asks callers not to use.
    """
    body = _body(model_id, cap=1024)
    assert body["max_completion_tokens"] == 1024
    assert "max_tokens" not in body


class _StubTool:
    name: str = "Echo"
    tool_id: str = "application/x-tool-echo"
    description: str = "Echo"
    directive_schema: Mapping[str, PlainTree] = {
        "type": "object",
        "additionalProperties": False,
    }
    clearable_results: bool = False

    def summary(self, args: Mapping[str, object]) -> str:
        del args
        return ""

    def prompt(self) -> str | None:
        return None

    def serialize_key(self, args: Mapping[str, object]) -> str | None:
        del args
        return None

    async def run(self, args: Mapping[str, object]) -> ToolResult:
        del args
        return ToolResult(call_id="", content="")


class _UnionTool(_StubTool):
    """Tool whose directive schema uses a JSON Schema type array."""

    name = "Ids"
    directive_schema: Mapping[str, PlainTree] = immutable(
        {
            "type": "object",
            "properties": {
                "ids": {"type": ["array", "string"], "items": {"type": "string"}},
            },
            "required": ["ids"],
        },
    )


def _parameters_of(tool: object) -> dict[str, MutablePlainTree]:
    """Return the one ``function.parameters`` a request builds for ``tool``."""
    body = _body("kimi-k3", tools=[cast(Any, tool)])
    wire_tools = from_plain(body["tools"], list[dict[str, MutablePlainTree]])
    function = from_plain(wire_tools[0]["function"], dict[str, MutablePlainTree])
    return from_plain(function["parameters"], dict[str, MutablePlainTree])


def test_a_type_array_reaches_the_wire_as_the_anyof_union() -> None:
    """The ``ids`` union survives to ``parameters`` as MFJS spells it."""
    parameters = _parameters_of(_UnionTool())
    properties = from_plain(parameters["properties"], dict[str, MutablePlainTree])
    ids = from_plain(properties["ids"], dict[str, MutablePlainTree])
    assert ids == {
        "anyOf": [
            {"type": "array", "items": {"type": "string"}},
            {"type": "string"},
        ],
    }


def _mfjs_offences(node: object, path: str) -> list[str]:
    """Return where a schema leaves the MFJS subset Kimi validates against."""
    if isinstance(node, list):
        return [
            offence
            for index, item in enumerate(node)
            for offence in _mfjs_offences(item, f"{path}[{index}]")
        ]
    if not isinstance(node, dict):
        return []
    offences: list[str] = []
    if isinstance(node.get("type"), list):
        offences.append(f"{path}.type={node['type']!r}")
    offences.extend(f"{path}.{key}" for key in node if key in _DROPPED_SCHEMA_KEYS)
    for key, value in node.items():
        if key == "properties" and isinstance(value, dict):
            # A property named like a dropped key is a NAME, not a schema
            # key; only the schemas it maps to are checked.
            offences.extend(
                offence
                for name, sub in value.items()
                for offence in _mfjs_offences(sub, f"{path}.{name}")
            )
        else:
            offences.extend(_mfjs_offences(value, f"{path}.{key}"))
    return offences


def test_kimi_default_tool_parameters_stay_inside_mfjs() -> None:
    """Every default tool's schema survives into a body Kimi accepts.

    ``parameters`` must conform to MFJS, where ``type`` is a single string;
    the paper tools' ``["array", "string"]`` ``ids`` is outside that subset.
    This drives the real tool set through the real body builder, so a tool
    that regresses the wire schema fails here rather than at the vendor.
    """
    tools = resolve_tools(list(DEFAULT_TOOLS))
    body = _body("kimi-k3", tools=tuple(tools))
    wire_tools = from_plain(body["tools"], list[dict[str, MutablePlainTree]])
    names: list[MutablePlainTree] = []
    offences: list[str] = []
    for tool in wire_tools:
        function = from_plain(tool["function"], dict[str, MutablePlainTree])
        names.append(function["name"])
        offences.extend(
            f"{function['name']}{path}"
            for path in _mfjs_offences(function["parameters"], "")
        )
    assert names == [t.name for t in tools]
    assert offences == []


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
