"""Tests for the Responses reasoning-effort and service-tier wire."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from sagent.providers.openai.wire import reasoning_effort, served_tier


if TYPE_CHECKING:
    from sagent.types.capability import ThinkingEffort
    from sagent.types.cost import ServiceTier


@pytest.mark.parametrize(
    ("reported", "tier"),
    [
        (None, "auto"),
        ("auto", "auto"),
        ("default", "auto"),
        ("flex", "flex"),
        ("priority", "priority"),
        ("fast", "priority"),
        ("ultrafast", "ultrafast"),
        # Scale Tier is prepaid capacity billed at the standard card.
        ("scale", "auto"),
    ],
)
def test_openai_bills_the_tier_it_reports_serving(
    reported: str | None,
    tier: ServiceTier,
) -> None:
    assert served_tier(reported) == tier


def test_an_unknown_reported_tier_is_unknown_not_an_error() -> None:
    """The response is already paid for; the caller picks the fallback rate."""
    assert served_tier("hyperfast") is None


@pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
def test_reasoning_effort_sends_the_selected_level_verbatim(
    effort: ThinkingEffort,
) -> None:
    assert reasoning_effort(effort) == effort


def test_default_effort_has_no_wire_spelling() -> None:
    with pytest.raises(ValueError, match=r"^effort 'default' is not sent"):
        _ = reasoning_effort("default")


def test_none_and_min_have_real_openai_wire_spellings() -> None:
    assert reasoning_effort("none") == "none"
    assert reasoning_effort("min") == "minimal"


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
