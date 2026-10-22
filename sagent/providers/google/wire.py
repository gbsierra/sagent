"""How a Gemini request spells Sagent's thinking and tool selections.

Sources:

- thinking: https://ai.google.dev/gemini-api/docs/generate-content/thinking
- tool schemas:
  https://ai.google.dev/gemini-api/docs/generate-content/function-calling
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, cast

from sagent.lib.codec import mutable
from sagent.providers.lib.thinking_budget import EFFORT_BUDGET_TOKENS


if TYPE_CHECKING:
    from collections.abc import Mapping

    from sagent.lib.codec import MutablePlainTree, PlainTree
    from sagent.types.capability import (
        ModelCapability,
        ModelSettings,
        ThinkingEffort,
    )


__all__ = ["thinking_config", "tool_schema"]


def thinking_config(
    capability: ModelCapability,
    settings: ModelSettings,
) -> dict[str, MutablePlainTree] | None:
    """Return the ``generationConfig.thinkingConfig`` a selection sends.

    Budget ``none`` omits the config and effort ``none`` omits the level or
    cap, so the model's own default applies in both cases. Gemini 3 rows take
    a ``thinkingLevel`` under either budget mode, its levels being dynamic
    already; 2.5 rows take a ``thinkingBudget`` token cap, ``-1`` under
    ``auto``.

    Args:
      capability: The row the request is built against.
      settings: The selection in effect.

    Returns:
      config: The wire object, or ``None`` when the request sends none.

    """
    if not settings.thinking_active:
        return None
    config: dict[str, MutablePlainTree] = {
        "includeThoughts": settings.thinking_output == "text",
    }
    effort = settings.thinking_effort
    if capability.model_id in _THINKING_LEVEL_MODELS:
        # Gemini 3 accepts ``thinkingBudget`` only for backwards compatibility.
        if effort != "default":
            config["thinkingLevel"] = _thinking_level(effort)
    elif settings.thinking_budget == "auto":
        config["thinkingBudget"] = -1
    elif effort != "default":
        config["thinkingBudget"] = EFFORT_BUDGET_TOKENS[effort]
    return config


def _thinking_level(effort: ThinkingEffort) -> str:
    """Return the Gemini 3 ``thinkingLevel`` an effort sends."""
    match effort:
        case "min":
            return "minimal"
        case "low" | "medium" | "high":
            return effort
        case "default" | "none" | "xhigh" | "max":
            raise ValueError(f"effort {effort!r} has no Gemini thinkingLevel")


_THINKING_LEVEL_MODELS: Final = frozenset(
    {
        "gemini-3.1-pro-preview",
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3-flash-preview",
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
    },
)
"""Exact routes whose API contract uses ``thinkingLevel``."""


# A directive schema is provider-neutral JSON Schema; these are the two
# constructs Gemini's Schema proto has no field for. Everything else passes
# through, because everything else is a documented field.
_DROPPED_SCHEMA_KEYS: Final = frozenset({"additionalProperties"})
_SCHEMA_LIST_KEYS: Final = frozenset({"anyOf", "oneOf", "allOf", "prefixItems"})


def tool_schema(schema: Mapping[str, PlainTree]) -> MutablePlainTree:
    """Return a directive schema spelled the way Gemini's Schema proto accepts.

    Gemini validates ``functionDeclarations[].parameters`` against a proto
    rather than against JSON Schema: "Only a subset of the OpenAPI schema is
    supported". ``additionalProperties`` is not one of its fields at all, and
    ``type`` is a single enum rather than JSON Schema's array of them -- the
    paper tools' ``ids``, declared ``["array", "string"]``, is a 400 for the
    whole request, not for that one property.

    Sagent's own schemas stay provider-neutral (``validate_tool_input``
    accepts either spelling of ``ids``), so the narrowing happens here, on the
    way out. A type array collapses to its first non-null member, which is the
    member any sibling ``items``/``properties`` describe; an additional
    ``null`` member becomes ``nullable``.

    Args:
      schema: The tool's directive schema.

    Returns:
      wire: The nested schema Gemini accepts.

    References:
      https://ai.google.dev/gemini-api/docs/generate-content/function-calling
        "Only a subset of the OpenAPI schema is supported."

    """
    # Directive schemas are ``immutable``: mapping proxies and tuples, shared
    # by every request. ``mutable`` gives this one its own dicts and lists.
    return _schema_node(mutable(schema))


def _schema_node(node: object) -> MutablePlainTree:
    """Translate one JSON Schema node, and everything below it."""
    if isinstance(node, list):
        return [_schema_node(item) for item in node]
    if not isinstance(node, dict):
        return cast("MutablePlainTree", node)
    out: dict[str, MutablePlainTree] = {}
    for raw_key, value in node.items():
        key = str(raw_key)
        if key in _DROPPED_SCHEMA_KEYS:
            continue
        # ``properties`` maps names to schemas; recursing into the map itself
        # would read the property NAMES as schema keys.
        if key == "properties" and isinstance(value, dict):
            out[key] = {str(k): _schema_node(v) for k, v in value.items()}
        elif key in _SCHEMA_LIST_KEYS and isinstance(value, list):
            out[key] = [_schema_node(item) for item in value]
        elif key == "items":
            out[key] = _schema_node(value)
        else:
            out[key] = cast("MutablePlainTree", value)
    types = _declared_types(node.get("type"))
    if types:
        out["type"] = types[0]
    if len(types) > 1 and types[1] == "null":
        out["nullable"] = True
    return out


def _declared_types(value: object) -> list[str]:
    """Return a node's type spellings, most specific first."""
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    names = [item for item in value if isinstance(item, str) and item != "null"]
    return [*names, "null"] if any(item == "null" for item in value) else names
