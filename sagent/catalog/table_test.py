"""Tests for exact, role, alias, and retired model-name resolution."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from types import MappingProxyType

import pytest

from sagent.catalog.provenance import Source
from sagent.catalog.table import (
    CONTEXT_TAGS,
    ModelCatalog,
    ModelTable,
    UnknownModelError,
    UnsupportedTagError,
    base_model_id,
    split_model_id,
)
from sagent.types.capability import (
    ContextTag,
    ModelCapability,
    ModelLimits,
    ModelSettings,
    ThinkingCapability,
)
from sagent.types.cost import PriceCatalog, PriceKey, TokenPrice


def _opus(model_id: str = "claude-opus-4-8") -> ModelCapability:
    return ModelCapability(
        model_id=model_id,
        context=MappingProxyType(
            {
                "": ModelLimits(max_request_tokens=200_000, max_image_bytes=5_000_000),
                "+1m": ModelLimits(
                    max_request_tokens=1_000_000,
                    max_image_bytes=5_000_000,
                ),
            },
        ),
        prices=PriceCatalog(
            {
                PriceKey("auto"): TokenPrice(
                    request=5.0,
                    response=0.0,
                    cache_write=0.0,
                    cache_write_1h=0.0,
                    cache_read=0.0,
                ),
            },
        ),
        thinking=ThinkingCapability(effort=frozenset({"none", "max"})),
        service_tier=frozenset({"auto", "default"}),
    )


def _cli() -> ModelCapability:
    return ModelCapability(
        thinking=ThinkingCapability(
            effort=frozenset({"none"}),
            output=frozenset({"none", "text"}),
        ),
        manage_context_server_side=frozenset({True}),
    )


def _table() -> ModelTable:
    return ModelTable(
        rows=(
            _opus("claude-opus-5-5"),
            _opus("claude-opus-4-8"),
            _opus("claude-sonnet-4-5"),
            _opus("claude-sonnet-4-5-20250929"),
        ),
        roles={"default": "claude-opus-5-5", "utility": "claude-sonnet-4-5"},
        aliases={"opus": "claude-opus-5-5"},
        renamed={"opus-4.8": "claude-opus-4-8"},
    )


def _resolve(
    model_id: str,
    *,
    rows: tuple[ModelCapability, ...] = (),
    transport: ModelCapability | None = None,
) -> tuple[ModelCapability, ModelSettings]:
    table = ModelTable(rows=rows or (_opus(),))
    return ModelCatalog(
        models=table,
        transport=transport or ModelCapability(),
    ).resolve(model_id)


# ---- ModelTable lookup -----------------------------------------------------


@pytest.mark.parametrize(
    ("key", "model_id"),
    [
        ("claude-opus-4-8", "claude-opus-4-8"),
        ("claude-sonnet-4-5", "claude-sonnet-4-5"),
        ("claude-sonnet-4-5-20250929", "claude-sonnet-4-5-20250929"),
        ("default", "claude-opus-5-5"),
        ("utility", "claude-sonnet-4-5"),
        ("opus", "claude-opus-5-5"),
        ("opus-4.8", "claude-opus-4-8"),
    ],
)
def test_a_key_names_a_vendor_id_a_role_an_alias_or_a_retired_name(
    key: str,
    model_id: str,
) -> None:
    assert _table()[key].model_id == model_id


@pytest.mark.parametrize(
    "key",
    ["op", "claude-opus", "claude-opus-4", "sonnet", "claude-sonnet-4-5x", "haiku"],
)
def test_no_spelling_is_guessed_into_a_row(key: str) -> None:
    """A family or prefix names nothing: only the vendor says what is newest."""
    assert key not in _table()
    assert _table().get(key) is None
    with pytest.raises(KeyError):
        _ = _table()[key]


def test_an_alias_and_its_snapshot_are_separate_rows() -> None:
    """Each sends exactly what was named, and each prices at its own row."""
    table = _table()
    assert table["claude-sonnet-4-5"] is not table["claude-sonnet-4-5-20250929"]
    assert table.exact("claude-sonnet-4-5-20250929") is table.rows[3]


def test_context_tags_are_the_alias_members_without_the_default() -> None:
    assert CONTEXT_TAGS == ("+200k", "+272k", "+1m")


def test_iteration_lists_vendor_ids_only() -> None:
    table = _table()
    assert list(table) == [
        "claude-opus-5-5",
        "claude-opus-4-8",
        "claude-sonnet-4-5",
        "claude-sonnet-4-5-20250929",
    ]
    assert len(table) == 4
    assert "default" in table
    assert "default" not in list(table)
    assert "opus" in table
    assert "opus" not in list(table)
    assert "opus-4.8" in table
    assert "opus-4.8" not in list(table)


@pytest.mark.parametrize(
    ("name", "model_id"),
    [
        ("claude-opus-4-8", "claude-opus-4-8"),
        ("default", None),
        ("opus", None),
        ("opus-4.8", None),
        ("claude-opus", None),
    ],
)
def test_exact_matches_only_a_vendor_id(name: str, model_id: str | None) -> None:
    """A server reports vendor ids, never a Sagent role or retired name."""
    row = _table().exact(name)
    assert (row.model_id if row is not None else None) == model_id


_PRICING = Source("https://example.com/pricing", date(2026, 10, 3))
_NEWER = Source("https://example.com/pricing", date(2026, 10, 18))


def test_a_row_source_overrides_the_tables_for_that_fact_only() -> None:
    table = replace(
        _table(),
        sources={"prices": _PRICING},
        row_sources={"claude-opus-4-8": {"prices": _NEWER}},
    )
    assert table.source("claude-opus-4-8", "prices") == _NEWER
    assert table.source("claude-opus-5-5", "prices") == _PRICING
    assert table.source("claude-opus-4-8", "limits") is None


def test_a_source_for_no_row_is_rejected() -> None:
    with pytest.raises(ValueError, match="sources for no model: default"):
        replace(_table(), row_sources={"default": {"prices": _NEWER}})


@pytest.mark.parametrize(
    ("roles", "aliases", "renamed", "message"),
    [
        ({"default": "haiku"}, {}, {}, "role 'default' names no model: 'haiku'"),
        ({}, {"opus": "nope"}, {}, "alias 'opus' names no model: 'nope'"),
        (
            {},
            {},
            {"opus-4.8": "nope"},
            "retired name 'opus-4.8' names no model: 'nope'",
        ),
        (
            {"claude-opus-4-8": "claude-opus-4-8"},
            {},
            {},
            "role 'claude-opus-4-8' shadows a model id",
        ),
        (
            {"x": "claude-opus-4-8"},
            {"x": "claude-opus-4-8"},
            {},
            "both a role and an alias: x",
        ),
        (
            {},
            {"x": "claude-opus-4-8"},
            {"x": "claude-opus-4-8"},
            "both an alias and a retired name: x",
        ),
    ],
)
def test_a_name_map_must_point_at_a_row_and_shadow_nothing(
    roles: dict[str, str],
    aliases: dict[str, str],
    renamed: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _ = ModelTable(
            rows=(_opus(),),
            roles=roles,
            aliases=aliases,
            renamed=renamed,
        )


def test_duplicate_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"duplicate model id 'claude-opus-4-8'"):
        _ = ModelTable(rows=(_opus(), _opus()))


def test_name_maps_are_frozen() -> None:
    roles = {"default": "claude-opus-4-8"}
    aliases = {"opus": "claude-opus-4-8"}
    table = ModelTable(rows=(_opus(),), roles=roles, aliases=aliases)
    roles["default"] = "nope"
    aliases["opus"] = "nope"
    assert table.roles["default"] == "claude-opus-4-8"
    assert table.aliases["opus"] == "claude-opus-4-8"
    assert type(table.roles) is MappingProxyType
    assert type(table.aliases) is MappingProxyType
    assert type(table.renamed) is MappingProxyType


# ---- model-id tags ---------------------------------------------------------


@pytest.mark.parametrize(
    ("model_id", "base", "tags"),
    [
        ("claude-opus-4-8", "claude-opus-4-8", frozenset[ContextTag]()),
        ("claude-opus-4-8+1m", "claude-opus-4-8", frozenset({"+1m"})),
        ("claude-opus-4-8+200k", "claude-opus-4-8", frozenset({"+200k"})),
        ("gpt-6-astra+272k", "gpt-6-astra", frozenset({"+272k"})),
        ("Opus-4.8+1M", "Opus-4.8", frozenset({"+1m"})),
        ("model+unknown", "model+unknown", frozenset[ContextTag]()),
        # ``+fast`` was a second spelling of ``service_tier="priority"``.
        ("opus-5+fast", "opus-5+fast", frozenset[ContextTag]()),
        ("Opus-4.7+200k+1M", "Opus-4.7", frozenset({"+200k", "+1m"})),
    ],
)
def test_split_model_id(model_id: str, base: str, tags: frozenset[ContextTag]) -> None:
    assert split_model_id(model_id) == (base, tags)
    assert base_model_id(model_id) == base


def test_context_tags_derive_from_the_literal() -> None:
    """A tag the type admits but the tuple omits would be unparseable."""
    assert set(CONTEXT_TAGS) == {"+1m", "+272k", "+200k"}


# ---- ModelCatalog.resolve --------------------------------------------------


def test_resolve_returns_capability_and_settings_as_peers() -> None:
    capability, settings = _resolve("claude-opus-4-8+1m")
    assert settings.context == "+1m"
    assert capability.context.keys() == {"", "+1m"}
    assert settings.capability == capability


def test_resolve_keeps_the_whole_context_table() -> None:
    _, settings = _resolve("claude-opus-4-8")
    assert settings.limits.max_request_tokens == 200_000
    wide = ModelSettings(capability=settings.capability, context="+1m")
    assert wide.limits.max_request_tokens == 1_000_000


def test_resolve_meets_the_transport() -> None:
    capability, _ = _resolve("claude-opus-4-8", transport=_cli())
    assert capability.thinking.effort == frozenset({"default", "none"})
    assert capability.manage_context_server_side == frozenset({True})


def test_served_returns_its_own_row_met_with_the_transport() -> None:
    catalog = ModelCatalog(models=_table(), transport=_cli())
    requested, _ = catalog.resolve("claude-sonnet-4-5")
    served = catalog.served(requested, "claude-sonnet-4-5-20250929")
    assert served.model_id == "claude-sonnet-4-5-20250929"
    assert served.manage_context_server_side == frozenset({True})


@pytest.mark.parametrize("reported", ["", "claude-opus-4-8", "default", "gpt-6-sol"])
def test_served_keeps_the_request_unless_this_catalog_lists_the_id(
    reported: str,
) -> None:
    """Blank, the requested id, a role, or another catalog's id: no swap."""
    catalog = ModelCatalog(models=_table(), transport=ModelCapability())
    requested, _ = catalog.resolve("claude-opus-4-8")
    assert catalog.served(requested, reported) is requested


def test_resolve_never_grants_what_the_row_lacks() -> None:
    capability, _ = _resolve(
        "claude-opus-4-8",
        transport=ModelCapability(
            thinking=ThinkingCapability(
                effort=frozenset({"none", "min", "low", "medium", "high", "max"}),
            ),
        ),
    )
    assert capability.thinking.effort == frozenset({"default", "none", "max"})


def test_resolve_rejects_a_context_the_model_lacks_naming_those_it_offers() -> None:
    row = replace(
        _opus(),
        context=MappingProxyType(
            {**_opus().context, "+200k": ModelLimits(max_request_tokens=200_000)},
        ),
    )
    with pytest.raises(UnsupportedTagError) as raised:
        _ = _resolve("claude-opus-4-8+272k", rows=(row,))
    assert str(raised.value) == (
        "Model 'claude-opus-4-8+272k' has no +272k context; offers: +1m, +200k"
    )


def test_resolve_says_none_when_the_model_offers_no_tagged_context() -> None:
    row = replace(_opus(), context=MappingProxyType({"": ModelLimits()}))
    with pytest.raises(UnsupportedTagError) as raised:
        _ = _resolve("claude-opus-4-8+200k", rows=(row,))
    assert str(raised.value) == (
        "Model 'claude-opus-4-8+200k' has no +200k context; offers: (none)"
    )


@pytest.mark.parametrize(
    "model_id",
    ["claude-opus-4-8+1m+200k", "claude-opus-4-8+200k+1m"],
)
def test_resolve_rejects_conflicting_context_tags(model_id: str) -> None:
    with pytest.raises(UnsupportedTagError) as raised:
        _ = _resolve(model_id)
    assert str(raised.value) == (
        f"Model {model_id!r} selects conflicting contexts: +1m, +200k"
    )


def test_resolve_accepts_explicit_1m_when_the_default_is_already_1m() -> None:
    row = replace(
        _opus(),
        context=MappingProxyType(
            {
                "": ModelLimits(
                    max_request_tokens=1_000_000,
                    max_response_tokens=128_000,
                ),
            },
        ),
    )
    capability, settings = _resolve("claude-opus-4-8+1m", rows=(row,))
    assert settings.context == "+1m"
    assert settings.limits == capability.context[""]
    untagged, _ = _resolve("claude-opus-4-8", rows=(row,))
    assert untagged.context.keys() == {""}


def test_a_transport_cannot_remove_a_context_window() -> None:
    """Windows are the model's; a transport restricts knobs, not physics."""
    narrow = ModelCapability(context=MappingProxyType({"": ModelLimits()}))
    _, settings = _resolve("claude-opus-4-8+1m", transport=narrow)
    assert settings.limits.max_request_tokens == 1_000_000


def test_resolve_names_the_known_models_roles_and_aliases_on_a_miss() -> None:
    catalog = ModelCatalog(models=_table(), transport=ModelCapability())
    with pytest.raises(UnknownModelError) as raised:
        _ = catalog.resolve("unknown")
    assert str(raised.value) == (
        "Unknown model 'unknown'.\n"
        "\n"
        "Aliases:\n"
        "  opus -> claude-opus-5-5\n"
        "\n"
        "Roles:\n"
        "  default -> claude-opus-5-5\n"
        "  utility -> claude-sonnet-4-5\n"
        "\n"
        "Known models:\n"
        "  claude-opus-5-5, claude-opus-4-8, claude-sonnet-4-5, "
        "claude-sonnet-4-5-20250929"
    )


@pytest.mark.parametrize(
    ("name", "model_id", "context"),
    [
        ("utility", "claude-sonnet-4-5", ""),
        ("opus", "claude-opus-5-5", ""),
        ("default+1m", "claude-opus-5-5", "+1m"),
        ("opus+1m", "claude-opus-5-5", "+1m"),
        ("opus-4.8+1m", "claude-opus-4-8", "+1m"),
        ("claude-opus-4-8", "claude-opus-4-8", ""),
    ],
)
def test_resolve_returns_the_vendor_id_whichever_name_reached_it(
    name: str,
    model_id: str,
    context: ContextTag,
) -> None:
    """The resolved id is what is sent, shown, and saved; a role never is."""
    catalog = ModelCatalog(models=_table(), transport=ModelCapability())
    capability, settings = catalog.resolve(name)
    assert capability.model_id == model_id
    assert settings.context == context


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
