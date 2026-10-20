"""Tests for the Gemini thinking wire."""

from __future__ import annotations

import pytest

from sagent.catalog import google
from sagent.providers.google.wire import thinking_config
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
    assert not google.models()[model_id].effort_as_level
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


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
