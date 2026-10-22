"""Tests for the Kimi MFJS tool-schema wire."""

from __future__ import annotations

from collections.abc import Mapping

from sagent.lib.codec import immutable
from sagent.providers.moonshot.wire import tool_schema


def test_a_type_array_becomes_the_anyof_union_the_subset_spells() -> None:
    """MFJS ``type`` is one string; its own union idiom is ``anyOf``.

    The paper tools declare ``ids`` this way so one bare id or many are
    both legal. ``items`` moves into the array branch -- the shape the
    spec's worked example writes -- so the string branch carries nothing
    it cannot use, and either spelling of ``ids`` stays legal to Kimi.
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
            "additionalProperties": False,
        },
    )
    assert tool_schema(schema) == {
        "type": "object",
        "properties": {
            "ids": {
                "anyOf": [
                    {"type": "array", "items": {"type": "string"}},
                    {"type": "string"},
                ],
                "description": "One id, or several.",
            },
        },
        "required": ["ids"],
        "additionalProperties": False,
    }


def test_additional_properties_is_inside_the_subset_and_survives() -> None:
    """Unlike Gemini, MFJS documents ``additionalProperties``."""
    schema = immutable(
        {
            "type": "object",
            "properties": {
                "nested": {"type": "object", "additionalProperties": False},
            },
            "additionalProperties": False,
        },
    )
    wire = tool_schema(schema)
    assert isinstance(wire, dict)
    assert wire["additionalProperties"] is False
    properties = wire["properties"]
    assert isinstance(properties, dict)
    nested = properties["nested"]
    assert isinstance(nested, dict)
    assert nested["additionalProperties"] is False


def test_a_null_member_is_its_own_branch() -> None:
    """``{"type": "null"}`` is a legal branch; MFJS has no ``nullable``."""
    schema = immutable({"type": ["string", "null"], "description": "Optional."})
    assert tool_schema(schema) == {
        "anyOf": [{"type": "string"}, {"type": "null"}],
        "description": "Optional.",
    }


def test_a_single_member_list_collapses_to_the_plain_string() -> None:
    schema = immutable({"type": ["string"]})
    assert tool_schema(schema) == {"type": "string"}


def test_number_bounds_land_in_the_number_branch_only() -> None:
    """``minimum`` is a number's constraint; the string branch drops it."""
    schema = immutable(
        {"type": ["string", "integer"], "minimum": 1, "description": "A count."},
    )
    assert tool_schema(schema) == {
        "anyOf": [{"type": "string"}, {"type": "integer", "minimum": 1}],
        "description": "A count.",
    }


def test_a_union_beside_an_anyof_narrows_every_branch() -> None:
    """A type array and a prior ``anyOf`` intersect, never merge.

    Merging the two lists would turn "a string that also satisfies A" into
    "a string, or anything satisfying A", so the prior list rides inside
    each branch instead. No sagent tool emits this shape today; the guard
    keeps the projection exact if one ever does.
    """
    schema = immutable(
        {
            "type": ["string", "integer"],
            "anyOf": [{"maxLength": 4}, {"maximum": 4}],
        },
    )
    wire = tool_schema(schema)
    assert isinstance(wire, dict)
    branches = wire["anyOf"]
    assert isinstance(branches, list)
    assert branches == [
        {"type": "string", "anyOf": [{"maxLength": 4}, {"maximum": 4}]},
        {"type": "integer", "anyOf": [{"maxLength": 4}, {"maximum": 4}]},
    ]


def test_fields_the_subset_names_unsupported_are_dropped_recursively() -> None:
    """``title``/``$comment``/``format`` annotations and tuple forms go."""
    schema = immutable(
        {
            "type": "object",
            "title": "Root",
            "$comment": "note",
            "properties": {
                "when": {"type": "string", "format": "date-time", "title": "When"},
                "pair": {
                    "type": ["array", "null"],
                    "prefixItems": [{"type": "string"}],
                },
            },
        },
    )
    wire = tool_schema(schema)
    assert isinstance(wire, dict)
    assert "title" not in wire
    assert "$comment" not in wire
    properties = wire["properties"]
    assert isinstance(properties, dict)
    when = properties["when"]
    assert isinstance(when, dict)
    assert when == {"type": "string"}
    pair = properties["pair"]
    assert isinstance(pair, dict)
    assert pair == {"anyOf": [{"type": "array"}, {"type": "null"}]}


def test_unlisted_validators_stay_to_guide_the_model() -> None:
    """The subset tolerates unlisted keywords; ``minimum`` still steers Kimi."""
    schema = immutable({"type": "integer", "minimum": 1})
    assert tool_schema(schema) == {"type": "integer", "minimum": 1}


def test_a_property_named_like_a_dropped_key_survives() -> None:
    """Only schema keys are dropped, never a property whose name matches."""
    schema = immutable(
        {
            "type": "object",
            "properties": {"title": {"type": "string"}, "format": {"type": "string"}},
        },
    )
    wire = tool_schema(schema)
    assert isinstance(wire, dict)
    properties = wire["properties"]
    assert isinstance(properties, dict)
    assert properties == {"title": {"type": "string"}, "format": {"type": "string"}}


def test_the_shared_directive_schema_stays_untouched() -> None:
    """One frozen schema is reused by every request; the projection copies."""
    schema = immutable(
        {
            "type": "object",
            "properties": {
                "ids": {"type": ["array", "string"], "items": {"type": "string"}},
            },
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
