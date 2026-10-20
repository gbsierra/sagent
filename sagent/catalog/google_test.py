"""Tests for the Gemini catalog's published facts."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pytest

from sagent.catalog import google


if TYPE_CHECKING:
    from sagent.types.capability import ModelCapability


@pytest.mark.parametrize(
    ("retired", "vendor_id"),
    [
        ("gemini-pro-3.1", "gemini-3.1-pro-preview"),
        ("gemini-pro-2.5", "gemini-2.5-pro"),
        ("gemini-flash-3.8", "gemini-3.8-flash"),
        ("gemini-flash-3.7", "gemini-3.7-flash"),
        ("gemini-flash-3.6", "gemini-3.6-flash"),
        ("gemini-flash-3.5", "gemini-3.5-flash"),
        ("gemini-flash-3.0", "gemini-3-flash-preview"),
        ("gemini-flash-2.5", "gemini-2.5-flash"),
        ("gemini-flash-lite-3.5", "gemini-3.5-flash-lite"),
        ("gemini-flash-lite-3.1", "gemini-3.1-flash-lite"),
        ("gemini-flash-lite-2.5", "gemini-2.5-flash-lite"),
    ],
)
def test_a_retired_sagent_name_reaches_its_vendor_row(
    retired: str,
    vendor_id: str,
) -> None:
    """A session saved under the old spelling loads as the vendor id."""
    rows = google.models()
    assert rows[retired].model_id == vendor_id
    assert rows.exact(retired) is None


@pytest.mark.parametrize(
    "model_id",
    ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
)
def test_shut_down_models_are_gone(model_id: str) -> None:
    assert model_id not in google.models()


def test_roles_follow_googles_guidance_for_new_projects() -> None:
    rows = google.models()
    assert rows["default"].model_id == "gemini-3.1-pro-preview"
    assert rows["utility"].model_id == "gemini-3.5-flash-lite"


@pytest.mark.parametrize(
    ("model_id", "standard"),
    [
        ("gemini-3.8-flash", (0.75, 3.75, 0.075)),
        ("gemini-3.7-flash", (0.75, 3.75, 0.075)),
        ("gemini-3.6-flash", (0.75, 3.75, 0.075)),
        ("gemini-3.5-flash", (1.5, 9.0, 0.15)),
        ("gemini-3.5-flash-lite", (0.3, 2.5, 0.03)),
        ("gemini-3.1-flash-lite", (0.25, 1.5, 0.025)),
        ("gemini-3-flash-preview", (0.5, 3.0, 0.05)),
        ("gemini-2.5-flash", (0.3, 2.5, 0.03)),
        ("gemini-2.5-flash-lite", (0.1, 0.4, 0.01)),
    ],
)
def test_a_row_bills_the_published_2026_rate(
    model_id: str,
    standard: tuple[float, float, float],
) -> None:
    price = google.models()[model_id].prices.rate(
        service_tier="auto",
        prompt_tokens=1_000,
        at=date(2026, 12, 31),
    )
    assert (price.request, price.response, price.cache_read) == standard


@pytest.mark.parametrize(
    "model_id",
    ["gemini-3.8-flash", "gemini-3.7-flash", "gemini-3.6-flash"],
)
def test_the_introductory_flash_rate_doubles_on_2027_01_01(model_id: str) -> None:
    prices = google.models()[model_id].prices
    last = prices.rate(service_tier="auto", prompt_tokens=1, at=date(2026, 12, 31))
    first = prices.rate(service_tier="auto", prompt_tokens=1, at=date(2027, 1, 1))
    assert (last.request, last.response, last.cache_read) == (0.75, 3.75, 0.075)
    assert (first.request, first.response, first.cache_read) == (1.5, 7.5, 0.15)


@pytest.mark.parametrize("model_id", list(google.models()))
def test_every_row_has_the_published_windows(model_id: str) -> None:
    limits = google.models()[model_id].context[""]
    assert limits.max_request_tokens == 1_048_576
    assert limits.max_response_tokens == 65_536
    assert limits.max_request_bytes == 100 * 1024 * 1024


@pytest.mark.parametrize(
    ("model_id", "cutoff"),
    [
        ("gemini-3.1-pro-preview", "January 2025"),
        ("gemini-3.1-flash-lite", "January 2025"),
        ("gemini-3-flash-preview", "January 2025"),
        ("gemini-2.5-pro", "January 2025"),
        ("gemini-2.5-flash", "January 2025"),
        ("gemini-2.5-flash-lite", "January 2025"),
        ("gemini-3.8-flash", None),
        ("gemini-3.7-flash", None),
        ("gemini-3.6-flash", None),
        ("gemini-3.5-flash", None),
        ("gemini-3.5-flash-lite", None),
    ],
)
def test_knowledge_cutoff_is_stated_only_where_published(
    model_id: str,
    cutoff: str | None,
) -> None:
    assert google.models()[model_id].knowledge_cutoff == cutoff


@pytest.mark.parametrize(
    ("model_id", "efforts"),
    [
        ("gemini-3.8-flash", {"default", "low", "medium", "high"}),
        ("gemini-3.7-flash", {"default", "low", "medium", "high"}),
        ("gemini-3.1-pro-preview", {"default", "low", "medium", "high"}),
        ("gemini-3.6-flash", {"default", "min", "low", "medium", "high"}),
        ("gemini-3.5-flash", {"default", "min", "low", "medium", "high"}),
        ("gemini-3.5-flash-lite", {"default", "min", "low", "medium", "high"}),
        ("gemini-3.1-flash-lite", {"default", "min", "low", "medium", "high"}),
        ("gemini-3-flash-preview", {"default", "min", "low", "medium", "high"}),
    ],
)
def test_a_gemini_3_row_offers_only_the_levels_it_accepts(
    model_id: str,
    efforts: set[str],
) -> None:
    """A level the model rejects is a 400; no level past ``high`` exists."""
    row = google.models()[model_id]
    assert row.thinking.effort == efforts
    assert row.effort_as_level


@pytest.mark.parametrize(
    "transport",
    [google.api(), google.cli(), google.subscription()],
    ids=["api", "cli", "subscription"],
)
def test_no_transport_offers_redacted_thinking(transport: ModelCapability) -> None:
    """Gemini has no server-side redaction to select."""
    assert "redacted" not in transport.thinking.output


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
