"""How a Kimi request spells Sagent's tool schemas.

Sources:

- tool parameters:
  https://platform.kimi.ai/docs/api/chat
- the subset they must conform to, Moonshot Flavored JSON Schema (MFJS):
  https://github.com/MoonshotAI/walle/blob/main/docs/mfjs-spec.md
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast

from sagent.lib.codec import mutable


if TYPE_CHECKING:
    from collections.abc import Mapping

    from sagent.lib.codec import MutablePlainTree, PlainTree


__all__ = ["tool_schema"]


def tool_schema(schema: Mapping[str, PlainTree]) -> MutablePlainTree:
    """Return a directive schema spelled the way Kimi's MFJS subset accepts.

    The chat API requires ``tools[].function.parameters`` to "conform to
    the MFJS (Moonshot Flavored JSON Schema) specification", so what
    leaves the provider-neutral directive schema has to fit that subset.
    Two of Sagent's constructs need translating:

    - ``type`` there is a single string, so the paper tools'
      ``["array", "string"]`` ``ids`` is not spellable. The subset's own
      union idiom is ``anyOf`` -- its worked example is a
      string-or-array-of-strings parameter -- so a type array becomes one
      branch per member, and the keys only one member can use (``items``
      for ``array``) ride inside that branch, the shape the example
      writes. Unlike Gemini's first-member collapse, either shape of
      ``ids`` stays legal to Kimi.
    - ``additionalProperties`` is *inside* the subset, so unlike Gemini it
      is kept.

    Fields the spec names as unsupported are dropped. Unlisted ones
    (``minimum``, ``minLength``) are kept: the spec says they "may not be
    directly blocked" and the model still reads them as guidance.

    Args:
      schema: The tool's directive schema.

    Returns:
      wire: The nested schema Kimi accepts.

    """
    # Directive schemas are ``immutable``: mapping proxies and tuples, shared
    # by every request. ``mutable`` gives this one its own dicts and lists.
    return _schema_node(mutable(schema))


# What MFJS names as unsupported: the ``title``/``$comment``/``format``
# annotations, the ``prefixItems``/``unevaluatedItems`` tuple forms, and the
# exclusive/contains validators. Sagent's schemas emit none of these; dropping
# them keeps a future tool from shipping vocabulary the validator documents
# rejecting.
_DROPPED_SCHEMA_KEYS: Final = frozenset(
    {
        "title",
        "$comment",
        "format",
        "prefixItems",
        "unevaluatedItems",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minContains",
        "maxContains",
    },
)
_SCHEMA_LIST_KEYS: Final = frozenset({"anyOf", "oneOf", "allOf"})
_SCHEMA_MAP_KEYS: Final = frozenset({"properties", "$defs"})

# Which keys one ``type`` member can use, by JSON Schema's own applicability:
# ``items`` says nothing about a string, so a union branch carries only what
# its member can constrain. Keys no member can use, and keys any instance can
# use (``enum``, ``const``, ``description``), stay on the union's parent,
# where they constrain every branch alike. Every key here is either recursed
# into by ``_schema_node`` or holds a plain value, so nothing lands in a
# branch unprojected.
_TYPE_SCOPED_KEYS: Final[Mapping[str, frozenset[str]]] = MappingProxyType(
    {
        "object": frozenset(
            {
                "properties",
                "required",
                "additionalProperties",
                "minProperties",
                "maxProperties",
            },
        ),
        "array": frozenset({"items", "minItems", "maxItems", "uniqueItems"}),
        "string": frozenset({"minLength", "maxLength", "pattern"}),
        "number": frozenset({"minimum", "maximum", "multipleOf"}),
        "integer": frozenset({"minimum", "maximum", "multipleOf"}),
    },
)


def _schema_node(node: object) -> MutablePlainTree:
    """Translate one JSON Schema node, and everything below it."""
    if isinstance(node, list):
        items = cast("list[object]", node)
        return [_schema_node(item) for item in items]
    if not isinstance(node, dict):
        return cast("MutablePlainTree", node)
    entries = cast("dict[str, object]", node)
    out: dict[str, MutablePlainTree] = {}
    for raw_key, value in entries.items():
        key = str(raw_key)
        if key in _DROPPED_SCHEMA_KEYS:
            continue
        # ``properties`` and ``$defs`` map names to schemas; recursing into
        # the map itself would read the property NAMES as schema keys.
        if key in _SCHEMA_MAP_KEYS and isinstance(value, dict):
            schema_map = cast("dict[str, object]", value)
            out[key] = {str(k): _schema_node(v) for k, v in schema_map.items()}
        elif key in _SCHEMA_LIST_KEYS and isinstance(value, list):
            items = cast("list[object]", value)
            out[key] = [_schema_node(item) for item in items]
        elif key in ("items", "additionalProperties") and isinstance(
            value,
            (dict, list),
        ):
            # Both take a schema (``additionalProperties`` also takes a bool,
            # which is neither and passes through below).
            schema = cast("dict[str, object] | list[object]", value)
            out[key] = _schema_node(schema)
        else:
            out[key] = cast("MutablePlainTree", value)
    return _union_of_types(out)


def _union_of_types(out: dict[str, MutablePlainTree]) -> MutablePlainTree:
    """Spell a JSON Schema type array as the ``anyOf`` union MFJS writes.

    A lone member collapses back to the plain string MFJS's ``type`` is;
    more members become one branch each, with the keys only that member
    can use moved into its branch and the rest left constraining the union
    itself.
    """
    types = out.get("type")
    if not isinstance(types, list):
        return out
    members = [t for t in types if isinstance(t, str)]
    if not members:
        # An empty member list declares nothing, and MFJS's ``type`` is
        # optional; shipping the empty list would ship the violation.
        del out["type"]
        return out
    if len(members) == 1:
        out["type"] = members[0]
        return out
    branches: list[MutablePlainTree] = []
    moved: set[str] = set()
    for member in members:
        branch: dict[str, MutablePlainTree] = {"type": member}
        for key in _TYPE_SCOPED_KEYS.get(member, ()):
            if key in out:
                branch[key] = out[key]
                moved.add(key)
        branches.append(branch)
    del out["type"]
    for key in moved:
        del out[key]
    prior = out.pop("anyOf", None)
    if prior is not None:
        # The schema already narrowed itself with an ``anyOf``; that list
        # constrains the union's every branch, so it rides inside each one
        # rather than being overwritten, or merged, which would turn "both"
        # into "either".
        for candidate in branches:
            if isinstance(candidate, dict):
                candidate["anyOf"] = prior
    out["anyOf"] = branches
    return out
