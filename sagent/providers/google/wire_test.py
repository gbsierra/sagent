"""Tests for the Gemini thinking and tool-schema wire."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from sagent.catalog import google
from sagent.lib.codec import immutable
from sagent.providers.google.wire import thinking_config, tool_schema
from sagent.types.capability import (
    ModelSettings,
    ThinkingBudget,
    ThinkingEffort,
)


def _wire(
    model_id: str,
    *,
    effort: ThinkingEffort = "default",
    budget: ThinkingBudget = "fixed",
) -> object:
    """Return the ``thinkingConfig`` one selection sends on ``model_id``."""
    row = google.models()[model_id]
    settings = ModelSettings(
        capability=row,
        thinking_effort=effort,
        thinking_budget=budget,
        thinking_output="text",
    )
    return thinking_config(row, settings)


@pytest.mark.parametrize(
    "model_id",
    ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
)
def test_gemini_2_5_takes_a_budget_not_a_level(model_id: str) -> None:
    assert _wire(model_id, effort="max") == {
        "includeThoughts": True,
        "thinkingBudget": 24_576,
    }
    assert _wire(model_id, budget="auto") == {
        "includeThoughts": True,
        "thinkingBudget": -1,
    }


@pytest.mark.parametrize(
    ("effort", "level"),
    [("min", "minimal"), ("low", "low"), ("medium", "medium"), ("high", "high")],
)
def test_gemini_3_sends_the_effort_as_a_level(
    effort: ThinkingEffort,
    level: str,
) -> None:
    assert _wire("gemini-3.6-flash", effort=effort) == {
        "includeThoughts": True,
        "thinkingLevel": level,
    }


@pytest.mark.parametrize("budget", ["auto", "fixed"])
def test_gemini_3_never_sends_a_token_budget(budget: ThinkingBudget) -> None:
    """``thinkingBudget`` is backward-compat only on Gemini 3."""
    sent = _wire("gemini-3.1-pro-preview", effort="high", budget=budget)
    assert sent == {"includeThoughts": True, "thinkingLevel": "high"}


@pytest.mark.parametrize("model_id", ["gemini-3.8-flash", "gemini-2.5-pro"])
def test_default_effort_leaves_the_models_default_in_place(model_id: str) -> None:
    """No level and no cap: never an explicit "off" a model may reject."""
    assert _wire(model_id) == {"includeThoughts": True}


def test_budget_none_sends_no_thinking_config() -> None:
    assert _wire("gemini-2.5-flash", budget="none") is None


def test_tool_schema_drops_additional_properties_recursively() -> None:
    """``additionalProperties`` is not a Schema proto field, at any depth."""
    schema = immutable(
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "nested": {"type": "object", "additionalProperties": False},
            },
            "items": [{"additionalProperties": False, "type": "string"}],
        },
    )
    wire = tool_schema(schema)
    properties = wire["properties"]
    assert isinstance(properties, dict)
    nested = properties["nested"]
    assert isinstance(nested, dict)
    assert "additionalProperties" not in wire
    assert "additionalProperties" not in nested
    items = wire["items"]
    assert isinstance(items, list)
    assert "additionalProperties" not in items[0]


def test_tool_schema_collapses_a_type_array_to_its_first_member() -> None:
    """``["array", "string"]`` has no proto spelling; the array is kept.

    The paper tools declare ``ids`` this way so one bare id or many are both
    legal, and that union is what ``validate_tool_input`` still accepts.
    """
    schema = immutable(
        {
            "type": "object",
            "properties": {
                "ids": {
                    "type": ["array", "string"],
                    "items": {"type": "string"},
                    "description": "One id, or several.",
                },
            },
            "required": ["ids"],
        },
    )
    assert tool_schema(schema) == {
        "type": "object",
        "properties": {
            "ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "One id, or several.",
            },
        },
        "required": ["ids"],
    }


def test_tool_schema_turns_a_null_member_into_nullable() -> None:
    """A nullable union is the one union the proto can say."""
    schema = immutable({"type": ["string", "null"], "description": "Optional."})
    assert tool_schema(schema) == {
        "type": "string",
        "nullable": True,
        "description": "Optional.",
    }


def test_tool_schema_keeps_a_property_named_additional_properties() -> None:
    """Only the schema key is dropped, never a property whose name matches."""
    schema = immutable(
        {
            "type": "object",
            "properties": {"additionalProperties": {"type": "boolean"}},
        },
    )
    wire = tool_schema(schema)
    properties = wire["properties"]
    assert isinstance(properties, dict)
    assert properties["additionalProperties"] == {"type": "boolean"}


def test_tool_schema_leaves_the_shared_directive_schema_untouched() -> None:
    """One frozen schema is reused by every request; the narrowing is a copy."""
    schema = immutable(
        {
            "type": "object",
            "properties": {"ids": {"type": ["array", "string"]}},
            "additionalProperties": False,
        },
    )
    tool_schema(schema)
    properties = schema["properties"]
    assert isinstance(properties, Mapping)
    ids = properties["ids"]
    assert isinstance(ids, Mapping)
    assert ids["type"] == ("array", "string")
    assert schema["additionalProperties"] is False


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
