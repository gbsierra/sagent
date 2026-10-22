"""Tests for ``types.capability``: the meet, and validation on assignment."""

from __future__ import annotations

from dataclasses import fields, replace
from types import MappingProxyType
from typing import Final, cast

import pytest

from sagent.types.capability import (
    InputModality,
    LimitCondition,
    LimitOverlay,
    ModelCapability,
    ModelLimits,
    ModelSettings,
    ThinkingCapability,
    ThinkingEffort,
    format_modalities,
)
from sagent.types.cost import PriceCatalog, PriceKey, TokenPrice


def _row() -> ModelCapability:
    return ModelCapability(
        model_id="m",
        context=MappingProxyType(
            {
                "": ModelLimits(max_request_tokens=200_000),
                "+1m": ModelLimits(max_request_tokens=1_000_000),
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
        thinking=ThinkingCapability(
            effort=frozenset({"none", "low", "high"}),
            budget=frozenset({"none", "auto"}),
            output=frozenset({"none", "text", "redacted"}),
            control="budget",
        ),
        service_tier=frozenset({"auto", "default", "priority"}),
        cache_ttl_sec=frozenset({300.0, 3600.0}),
        manage_context_server_side=frozenset({False, True}),
        retries_internally=True,
    )


# ---- the meet --------------------------------------------------------------


def test_every_field_is_default_constructible() -> None:
    assert ModelCapability().model_id == ""
    assert ModelSettings().context == ""


def test_a_bare_transport_keeps_the_axis_a_row_cannot_know() -> None:
    """A row does not know who rolls history, so it must not pin that axis.

    Defaulting it narrow made every CLI meet empty: the CLI offers
    ``manage_context_server_side={True}``, and ``{False} & {True}`` is a
    model nothing can be requested from.
    """
    met = _row() & ModelCapability()
    assert met.manage_context_server_side == _row().manage_context_server_side
    # The thinking and tier axes stay narrow: a row DOES know what it can
    # think and which tiers the vendor sells it at.
    assert met.thinking.effort == frozenset({"default"})
    assert met.service_tier == frozenset({"auto", "default"})
    assert met.cache_ttl_sec == frozenset({0.0})


def test_the_meet_only_removes() -> None:
    transport = ModelCapability(
        thinking=ThinkingCapability(
            effort=frozenset({"none", "low", "medium", "high", "max"}),
            budget=frozenset({"none", "auto", "fixed"}),
            output=frozenset({"none", "text", "redacted"}),
        ),
        service_tier=frozenset({"auto", "default", "flex", "priority"}),
        cache_ttl_sec=frozenset({300.0, 3600.0, 7200.0}),
        manage_context_server_side=frozenset({False, True}),
        retries_internally=True,
    )
    met = _row() & transport
    assert met.thinking.effort == frozenset({"default", "none", "low", "high"})
    assert met.service_tier == frozenset({"auto", "default", "priority"})


def test_the_meet_cannot_grant() -> None:
    met = ModelCapability(
        thinking=ThinkingCapability(effort=frozenset({"none"})),
    ) & ModelCapability(
        thinking=ThinkingCapability(effort=frozenset({"none", "max"})),
    )
    assert met.thinking.effort == frozenset({"default", "none"})


def test_the_meet_narrows_input_modalities_to_what_both_accept() -> None:
    """A transport with no PDF channel cannot inherit the row's ``pdf`` fact."""
    row = ModelCapability(inputs=frozenset({"text", "image", "pdf"}))
    transport = ModelCapability(inputs=frozenset({"text", "image"}))
    assert (row & transport).inputs == frozenset({"text", "image"})


def test_the_meet_cannot_grant_an_input_modality() -> None:
    """A wire that can send video cannot give a text-only row the ability."""
    row = ModelCapability(inputs=frozenset({"text"}))
    transport = ModelCapability(inputs=frozenset({"text", "image", "audio", "video"}))
    assert (row & transport).inputs == frozenset({"text"})


def test_a_row_with_no_input_modality_is_rejected() -> None:
    """An empty set is the meet's absorbing element, not a valid declaration."""
    with pytest.raises(ValueError, match="at least one modality"):
        _ = ModelCapability(inputs=frozenset())


def test_an_unknown_input_modality_is_rejected() -> None:
    """A typo must fail at the row, not silently drop every attachment."""
    with pytest.raises(ValueError, match="unknown input modality: images"):
        _ = ModelCapability(
            inputs=cast("frozenset[InputModality]", frozenset({"images"}))
        )


def test_format_modalities_reads_in_declaration_order() -> None:
    assert format_modalities(frozenset({"pdf", "text", "image"})) == "text, image, pdf"
    assert format_modalities(frozenset({"video", "text"})) == "text, video"
    assert format_modalities(frozenset()) == "(none)"


def test_the_transport_declares_the_cache_lifetimes() -> None:
    """Not intersected: only the transport knows if it writes a breakpoint."""
    offered = frozenset({300.0, 3600.0})
    assert (_row() & ModelCapability(cache_ttl_sec=offered)).cache_ttl_sec == offered
    assert (_row() & ModelCapability()).cache_ttl_sec == frozenset({0.0})


def test_the_meet_passes_context_and_prices_through() -> None:
    met = _row() & ModelCapability()
    assert met.context.keys() == {"", "+1m"}
    assert met.prices == _row().prices


def test_the_transport_declares_retries_and_auth() -> None:
    """Not intersected: a catalog row cannot know who bills or retries it.

    Conjoining made a transport's ``account_auth=True`` unreachable, because
    every row leaves it at the ``False`` default.
    """
    row = _row()
    assert (row & ModelCapability(retries_internally=True)).retries_internally is True
    assert (row & ModelCapability()).retries_internally is False
    assert (row & ModelCapability(account_auth=True)).account_auth is True
    assert (row & ModelCapability()).account_auth is False


# ---- assignment validates --------------------------------------------------


def test_every_settings_axis_names_a_capability_field() -> None:
    """The membership check is total only if the names line up."""
    capability_names = {f.name for f in fields(ModelCapability)}
    thinking_names = {
        f"thinking_{f.name}" for f in fields(ThinkingCapability) if f.name != "control"
    }
    for f in fields(ModelSettings):
        if f.name == "capability":
            continue
        assert f.name in capability_names | thinking_names, f.name


def test_a_valid_selection_constructs() -> None:
    ModelSettings(
        capability=_row(),
        context="+1m",
        thinking_effort="high",
        thinking_budget="auto",
        thinking_output="redacted",
        service_tier="priority",
        cache_ttl_sec=300.0,
        manage_context_server_side=True,
    )


def test_defaults_construct_against_a_default_capability() -> None:
    ModelSettings(capability=ModelCapability())


def test_assignment_rejects_what_the_model_withholds() -> None:
    """The whole point of the setter: a stale knob never reaches the wire."""
    settings = ModelSettings.narrowest(_row())
    with pytest.raises(ValueError, match="thinking_effort='medium'"):
        settings.thinking_effort = "medium"
    assert settings.thinking_effort == "default"


def test_construction_rejects_what_the_model_withholds() -> None:
    with pytest.raises(ValueError, match="thinking_effort='medium'"):
        ModelSettings(capability=_row(), cache_ttl_sec=300.0, thinking_effort="medium")


def test_an_unoffered_context_is_rejected() -> None:
    with pytest.raises(ValueError, match="context='\\+200k'"):
        ModelSettings(capability=_row(), cache_ttl_sec=300.0, context="+200k")


def test_the_cache_axis_validates_by_membership() -> None:
    """Vendors sell discrete lifetimes, so a bound admitted unspellable values."""
    ModelSettings(capability=_row(), cache_ttl_sec=300.0)
    ModelSettings(capability=_row(), cache_ttl_sec=3600.0)
    with pytest.raises(ValueError, match="not offered by m"):
        ModelSettings(capability=_row(), cache_ttl_sec=1800.0)


def test_a_boolean_axis_validates_by_membership() -> None:
    row = ModelCapability(manage_context_server_side=frozenset({False}))
    ModelSettings(capability=row, manage_context_server_side=False)
    with pytest.raises(ValueError, match="manage_context_server_side=True"):
        ModelSettings(capability=row, manage_context_server_side=True)


def test_the_error_names_the_model_and_what_it_offers() -> None:
    with pytest.raises(ValueError, match="not offered by m") as excinfo:
        ModelSettings(capability=_row(), cache_ttl_sec=300.0, service_tier="flex")
    assert "'priority'" in str(excinfo.value)


_OFF = ModelCapability(
    thinking=ThinkingCapability(
        effort=frozenset({"none"}),
        budget=frozenset({"none"}),
        output=frozenset({"none"}),
    ),
    service_tier=frozenset({"auto"}),
    cache_ttl_sec=frozenset({0.0}),
    manage_context_server_side=frozenset({False}),
)

_UNOFFERED: Final = (
    ("context", "+200k"),
    ("thinking_effort", "medium"),
    ("thinking_budget", "fixed"),
    ("thinking_output", "text"),
    ("service_tier", "flex"),
    ("cache_ttl_sec", 300.0),
    ("manage_context_server_side", True),
)


@pytest.mark.parametrize(("name", "value"), _UNOFFERED, ids=[n for n, _ in _UNOFFERED])
def test_assignment_checks_every_axis(name: str, value: object) -> None:
    settings = ModelSettings.narrowest(_OFF)
    with pytest.raises(ValueError, match=name):
        setattr(settings, name, value)


def test_the_axis_cases_cover_every_settings_axis() -> None:
    """Without this, a new ``ModelSettings`` axis is untested but green."""
    axes = {f.name for f in fields(ModelSettings)} - {"capability"}
    assert {n for n, _ in _UNOFFERED} == axes


def test_the_capability_itself_is_assignable() -> None:
    """It is the authority, not a choice, so it cannot validate against itself."""
    settings = ModelSettings.narrowest(_OFF)
    settings.capability = _row()
    settings.thinking_effort = "high"


# ---- take / adopt ----------------------------------------------------------


def test_take_applies_what_is_offered() -> None:
    settings = ModelSettings.narrowest(_row())
    settings.take(thinking_effort="high", thinking_budget="auto")
    assert settings.thinking_effort == "high"
    assert settings.thinking_budget == "auto"


def test_take_drops_what_is_not_offered_instead_of_raising() -> None:
    """A swap or resume must not FAIL on a knob the new model withholds."""
    settings = ModelSettings.narrowest(_row())
    settings.take(thinking_effort="max", service_tier="flex")
    assert settings.thinking_effort == "high"
    assert settings.service_tier == "auto"


def _efforts(*offered: ThinkingEffort) -> ModelCapability:
    return replace(_row(), thinking=ThinkingCapability(effort=frozenset(offered)))


@pytest.mark.parametrize(
    ("offered", "wanted", "landed"),
    [
        (("none", "low", "high"), "max", "high"),
        (("low", "medium", "high", "xhigh", "max"), "min", "low"),
        (("none", "low", "high"), "medium", "low"),
        (("none", "low", "high", "max"), "xhigh", "high"),
        (("medium", "high", "xhigh"), "max", "xhigh"),
        (("none", "min", "low", "medium", "high"), "max", "high"),
    ],
)
def test_take_lands_an_unoffered_effort_on_the_nearest_rung(
    offered: tuple[ThinkingEffort, ...],
    wanted: ThinkingEffort,
    landed: ThinkingEffort,
) -> None:
    """A tie between two rungs rounds down, toward the cheaper one."""
    settings = ModelSettings.narrowest(_efforts(*offered))
    settings.take(thinking_effort=wanted)
    assert settings.thinking_effort == landed


def test_take_keeps_an_offered_effort_exactly() -> None:
    settings = ModelSettings.narrowest(_efforts("none", "low", "high"))
    settings.take(thinking_effort="low")
    assert settings.thinking_effort == "low"


def test_take_never_promotes_none_into_thinking() -> None:
    """Unsupported explicit ``none`` falls back to provider default."""
    settings = ModelSettings.narrowest(_efforts("medium", "high"))
    settings.thinking_effort = "high"
    settings.take(thinking_effort="none")
    assert settings.thinking_effort == "default"


def test_take_falls_to_default_when_the_row_has_no_effort_knob() -> None:
    settings = ModelSettings.narrowest(_OFF)
    settings.take(thinking_effort="max")
    assert settings.thinking_effort == "default"


def test_take_defaults_an_effort_that_is_off_the_ladder() -> None:
    settings = ModelSettings.narrowest(_efforts("low", "high"))
    settings.take(thinking_effort="extreme")
    assert settings.thinking_effort == "default"


def test_capability_swap_carries_effort_to_the_nearest_rung() -> None:
    settings = ModelSettings.narrowest(_efforts("none", "low", "medium", "max"))
    settings.thinking_effort = "max"
    settings.capability = _efforts("none", "low", "high")
    assert settings.thinking_effort == "high"


def test_take_rejects_a_name_that_is_not_an_axis() -> None:
    with pytest.raises(ValueError, match="not settings axes: nonsense"):
        ModelSettings().take(nonsense=1)


def test_adopt_carries_the_selection_across_a_swap() -> None:
    old = ModelSettings(
        capability=_row(),
        cache_ttl_sec=300.0,
        thinking_effort="high",
        service_tier="priority",
    )
    new = ModelSettings(capability=_row(), cache_ttl_sec=300.0, context="+1m")
    new.adopt(old)
    assert new.thinking_effort == "high"
    assert new.service_tier == "priority"


def test_adopt_keeps_the_incoming_context_tag() -> None:
    """The new model id already named its window; the swap must not undo it."""
    old = ModelSettings(capability=_row(), cache_ttl_sec=300.0, context="")
    new = ModelSettings(capability=_row(), cache_ttl_sec=300.0, context="+1m")
    new.adopt(old)
    assert new.context == "+1m"


def test_adopt_drops_a_knob_the_new_model_rejects() -> None:
    old = ModelSettings(capability=_row(), cache_ttl_sec=300.0, thinking_effort="high")
    new = ModelSettings.narrowest(_OFF)
    new.adopt(old)
    assert new.thinking_effort == "default"


# ---- narrowest -------------------------------------------------------------


def test_narrowest_is_the_unset_value_of_every_axis_that_offers_one() -> None:
    assert ModelSettings.narrowest(_row()) == ModelSettings(
        capability=_row(),
        cache_ttl_sec=300.0,
    )


def test_narrowest_selects_the_cache_the_wire_already_ships() -> None:
    """``0.0`` claimed no caching while a ``5m`` breakpoint shipped anyway."""
    assert ModelSettings.narrowest(_row()).cache_ttl_sec == 300.0
    assert ModelSettings.narrowest(ModelCapability()).cache_ttl_sec == 0.0
    only_long = ModelCapability(cache_ttl_sec=frozenset({3600.0}))
    assert ModelSettings.narrowest(only_long).cache_ttl_sec == 3600.0


def test_narrowest_carries_the_context_the_id_selected() -> None:
    assert ModelSettings.narrowest(_row(), context="+1m").context == "+1m"


def test_narrowest_carries_the_capability_it_narrowed() -> None:
    """Without it every later write would validate against nothing."""
    assert ModelSettings.narrowest(_row()).capability == _row()


def test_narrowest_keeps_provider_default_and_climbs_other_axes() -> None:
    row = ModelCapability(
        thinking=ThinkingCapability(effort=frozenset({"low", "high"})),
        manage_context_server_side=frozenset({True}),
    )
    settings = ModelSettings.narrowest(row)
    assert settings.thinking_effort == "default"
    assert settings.manage_context_server_side is True


@pytest.mark.parametrize(
    ("models", "transport"),
    [(_row(), ModelCapability()), (_row(), _row())],
    ids=["bare", "self"],
)
def test_narrowest_builds_against_what_it_narrowed(
    models: ModelCapability,
    transport: ModelCapability,
) -> None:
    """Construction validates, so an unselectable narrowest would raise here."""
    capability = models & transport
    assert ModelSettings.narrowest(capability).capability == capability


def test_thinking_capability_always_offers_provider_default() -> None:
    capability = ModelCapability(thinking=ThinkingCapability(effort=frozenset()))
    assert capability.thinking.effort == frozenset({"default"})
    assert ModelSettings.narrowest(capability).thinking_effort == "default"


# ---- limits ----------------------------------------------------------------


def test_limits_are_derived_not_cached() -> None:
    """A cached row let a spec claim one context and carry another's."""
    row = _row()
    assert ModelSettings.narrowest(row).limits.max_request_tokens == 200_000
    assert (
        ModelSettings.narrowest(row, context="+1m").limits.max_request_tokens
        == 1_000_000
    )


def test_limit_overlay_resolves_from_selected_thinking_state() -> None:
    thinking_limits = ModelLimits(max_request_tokens=180_000)
    row = replace(
        _row(),
        limit_overlays=(
            LimitOverlay(
                when=LimitCondition(thinking=True),
                limits=thinking_limits,
            ),
        ),
    )
    settings = ModelSettings.narrowest(row)
    assert settings.limits.max_request_tokens == 200_000
    settings.thinking_budget = "auto"
    assert settings.limits == thinking_limits


def test_more_specific_limit_overlay_wins_without_order_dependence() -> None:
    thinking = LimitOverlay(
        when=LimitCondition(thinking=True),
        limits=ModelLimits(max_request_tokens=180_000),
    )
    thinking_priority = LimitOverlay(
        when=LimitCondition(thinking=True, service_tier="priority"),
        limits=ModelLimits(max_request_tokens=160_000),
    )
    row = replace(_row(), limit_overlays=(thinking_priority, thinking))
    settings = ModelSettings.narrowest(row)
    settings.thinking_budget = "auto"
    settings.service_tier = "priority"
    assert settings.limits.max_request_tokens == 160_000


def test_equal_specificity_limit_overlays_are_ambiguous() -> None:
    row = replace(
        _row(),
        limit_overlays=(
            LimitOverlay(
                when=LimitCondition(thinking=True),
                limits=ModelLimits(max_request_tokens=180_000),
            ),
            LimitOverlay(
                when=LimitCondition(service_tier="priority"),
                limits=ModelLimits(max_request_tokens=170_000),
            ),
        ),
    )
    settings = ModelSettings.narrowest(row)
    settings.thinking_budget = "auto"
    settings.service_tier = "priority"
    with pytest.raises(ValueError, match="ambiguous limit overlays for m"):
        _ = settings.limits


def test_transport_intersection_preserves_model_limit_overlays() -> None:
    overlay = LimitOverlay(
        when=LimitCondition(thinking=True),
        limits=ModelLimits(max_request_tokens=180_000),
    )
    row = replace(_row(), limit_overlays=(overlay,))
    assert (row & ModelCapability()).limit_overlays == (overlay,)


def test_duplicate_limit_conditions_fail_when_the_row_is_built() -> None:
    condition = LimitCondition(thinking=True)
    with pytest.raises(ValueError, match="duplicate limit condition for m"):
        replace(
            _row(),
            limit_overlays=(
                LimitOverlay(when=condition, limits=ModelLimits()),
                LimitOverlay(when=condition, limits=ModelLimits()),
            ),
        )


def test_settings_over_an_empty_axis_cannot_be_constructed() -> None:
    """So ``take`` never meets a capability whose ``narrowest`` raises."""
    with pytest.raises(ValueError, match=r"cache_ttl_sec=0\.0 is not offered"):
        ModelSettings(capability=ModelCapability(cache_ttl_sec=frozenset()))


def test_take_rejects_capability_as_a_non_axis() -> None:
    with pytest.raises(ValueError, match="not settings axes: capability"):
        ModelSettings().take(capability=ModelCapability())


def test_capability_assignment_renarrows_context_and_all_axes() -> None:
    settings = ModelSettings.narrowest(_row(), context="+1m")
    settings.thinking_effort = "high"
    settings.capability = _OFF
    assert settings.context == ""
    assert settings.limits == ModelLimits()
    assert settings.thinking_effort == "default"
    assert settings.cache_ttl_sec == 0.0


def test_invalid_capability_assignment_leaves_settings_unchanged() -> None:
    settings = ModelSettings.narrowest(_row(), context="+1m")
    with pytest.raises(ValueError, match="cache_ttl_sec offers nothing"):
        settings.capability = ModelCapability(cache_ttl_sec=frozenset())
    assert settings.capability == _row()
    assert settings.context == "+1m"


def test_equal_capabilities_are_hashable_independent_of_mapping_order() -> None:
    assert hash(ModelCapability()) == hash(ModelCapability())
    assert hash(_row()) == hash(_row())


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
