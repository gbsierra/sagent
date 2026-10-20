"""Anthropic catalog rows, checked against the published docs (2026-10-03)."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from sagent.catalog import anthropic
from sagent.types.cost import PriceKey, TokenCount, TokenPrice


def test_sonnet_5_5_bills_its_published_card() -> None:
    """https://platform.claude.com/docs/en/about-claude/pricing (read 2026-10-18)."""
    assert anthropic.models()["claude-sonnet-5-5"].prices[
        PriceKey("auto")
    ] == TokenPrice(
        request=2.0,
        response=10.0,
        cache_write=2.5,
        cache_write_1h=4.0,
        cache_read=0.1,
    )


@pytest.mark.parametrize(
    ("tokens", "card"),
    [
        (
            TokenCount(request=100_000),
            TokenPrice(
                request=0.1,
                response=0.5,
                cache_write=0.125,
                cache_write_1h=0.2,
                cache_read=0.01,
            ),
        ),
        # Cache reads count toward the 100k threshold.
        (
            TokenCount(request=1, cache_read=100_000),
            TokenPrice(
                request=0.5,
                response=2.5,
                cache_write=0.625,
                cache_write_1h=1.0,
                cache_read=0.05,
            ),
        ),
    ],
)
def test_haiku_5_5_bills_the_band_its_whole_prompt_selects(
    tokens: TokenCount,
    card: TokenPrice,
) -> None:
    """https://platform.claude.com/docs/en/about-claude/pricing#long-context-pricing."""
    prices = anthropic.models()["claude-haiku-5-5"].prices
    at = date(2026, 10, 17)
    assert prices.rate(service_tier="auto", prompt_tokens=tokens.prompt, at=at) == card


def test_haiku_5_5_states_its_published_specs() -> None:
    """https://platform.claude.com/docs/en/models/haiku-5-5/overview."""
    row = anthropic.models()["claude-haiku-5-5"]
    limits = row.context[""]
    assert (limits.max_request_tokens, limits.max_response_tokens) == (
        1_000_000,
        128_000,
    )
    assert limits.max_image_edge_px == 2576
    assert row.knowledge_cutoff == "June 2026"
    assert row.thinking.effort == frozenset(
        {"default", "low", "medium", "high", "xhigh", "max"},
    )
    assert "priority" not in row.service_tier


def test_sonnet_5_5_states_its_published_windows() -> None:
    rows = anthropic.models()
    assert rows["utility"].model_id == "claude-sonnet-5-5"
    limits = rows["claude-sonnet-5-5"].context[""]
    assert (limits.max_request_tokens, limits.max_response_tokens) == (
        1_000_000,
        128_000,
    )


@pytest.mark.parametrize(
    ("model_id", "cutoff"),
    [
        ("claude-sonnet-5-5", "June 2026"),
        ("claude-mythos-5-1", "June 2026"),
        ("claude-mythos-5", "January 2026"),
    ],
)
def test_new_rows_state_their_published_cutoff(model_id: str, cutoff: str) -> None:
    assert anthropic.models()[model_id].knowledge_cutoff == cutoff


@pytest.mark.parametrize(
    ("mythos", "fable"),
    [
        ("claude-mythos-5-1", "claude-fable-5-1"),
        ("claude-mythos-5", "claude-fable-5"),
    ],
)
def test_mythos_shares_fables_specs_and_prices(mythos: str, fable: str) -> None:
    """Each Mythos overview page: "It shares Claude Fable ...'s specifications"."""
    rows = anthropic.models()
    assert rows[mythos].prices == rows[fable].prices
    assert rows[mythos].context == rows[fable].context
    assert rows[mythos].thinking == rows[fable].thinking
    assert "priority" not in rows[mythos].service_tier


@pytest.mark.parametrize(
    ("alias", "snapshot"),
    [
        ("claude-opus-4-5", "claude-opus-4-5-20251101"),
        ("claude-sonnet-4-5", "claude-sonnet-4-5-20250929"),
        ("claude-haiku-4-5", "claude-haiku-4-5-20251001"),
    ],
)
def test_a_dated_snapshot_carries_its_aliass_facts(alias: str, snapshot: str) -> None:
    """The API reports the snapshot it served; it must price like the alias."""
    rows = anthropic.models()
    assert rows.exact(snapshot) is not None
    assert rows[snapshot] == replace(rows[alias], model_id=snapshot)


@pytest.mark.parametrize("name", ["opus", "sonnet", "haiku", "fable", "mythos"])
def test_a_family_word_is_not_a_model(name: str) -> None:
    """Only the vendor says which row a family means; Sagent no longer guesses."""
    assert name not in anthropic.models()


def test_a_retired_sagent_name_still_reaches_its_row() -> None:
    """A session saved as ``opus-4.7`` loads as ``claude-opus-4-7``."""
    assert anthropic.models()["opus-4.7"].model_id == "claude-opus-4-7"


def test_no_row_needs_a_context_beta() -> None:
    """For every 1M model, "1M is the default: you don't need a beta header"."""
    for model_id, row in anthropic.models().items():
        for tag, limits in row.context.items():
            assert not limits.request_betas, (model_id, tag)


def test_image_cap_leaves_room_for_base64_under_10_mb() -> None:
    """The vision page caps 10 MB base64-encoded; the cap applies to raw bytes."""
    for row in anthropic.models().values():
        raw = row.context[""].max_image_bytes
        assert raw == 7_500_000
        assert -(-raw // 3) * 4 <= 10_000_000


@pytest.mark.parametrize(
    "model_id",
    [
        "claude-fable-5-1",
        "claude-fable-5",
        "claude-mythos-5-1",
        "claude-mythos-5",
        "claude-opus-5-5",
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-sonnet-5-5",
        "claude-sonnet-5",
        "claude-haiku-5-5",
    ],
)
def test_omitted_display_models_still_offer_summarized_text(model_id: str) -> None:
    """``display`` defaults to omitted here; ``summarized`` opts back in."""
    output = anthropic.models()[model_id].thinking.output
    assert output == frozenset({"none", "text", "redacted"})


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
