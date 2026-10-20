"""DashScope catalog rows against Model Studio's published tables (2026-10-03)."""

from __future__ import annotations

from datetime import date

import pytest

from sagent.catalog import dashscope
from sagent.types.cost import PriceKey


_TODAY = date(2026, 10, 3)


@pytest.mark.parametrize(
    ("model_id", "prompt_tokens", "published"),
    [
        ("qwen3.8-max", 1_000_000, (2.0, 6.0)),
        ("qwen3.8-flash", 1_000, (0.15, 0.47)),
        ("qwen3.7-max", 1_000, (2.5, 7.5)),
        ("qwen3.7-plus", 256_000, (0.4, 1.6)),
        ("qwen3.7-plus", 256_001, (1.2, 4.8)),
        ("qwen3.7-flash", 32_000, (0.03, 0.13)),
        ("qwen3.7-flash", 32_001, (0.10, 0.40)),
        ("qwen3.7-flash", 256_001, (0.20, 0.80)),
        ("qwen3.6-plus", 256_001, (2.0, 6.0)),
        ("qwen3.6-flash", 1_000, (0.25, 1.5)),
        ("qwen3.6-max-preview", 128_001, (2.0, 12.0)),
        ("qwen3-coder-480b-a35b-instruct", 128_001, (4.5, 22.5)),
        ("qwen3-235b-a22b-thinking-2507", 1_000, (0.23, 2.3)),
        ("qwen3-32b", 1_000, (0.16, 0.64)),
        ("qwen-plus", 256_001, (1.2, 3.6)),
        ("qwen-turbo", 1_000, (0.05, 0.2)),
    ],
)
def test_a_prompt_bills_its_whole_band(
    model_id: str,
    prompt_tokens: int,
    published: tuple[float, float],
) -> None:
    """Every token in a request bills at its tier's rate (model-pricing)."""
    price = dashscope.models()[model_id].prices.rate(
        service_tier="auto",
        prompt_tokens=prompt_tokens,
        at=_TODAY,
    )
    assert (price.request, price.response) == published


@pytest.mark.parametrize(
    ("model_id", "cache_read"),
    [
        ("qwen3.7-plus", 0.08),
        ("qwen-turbo", 0.01),
        # Unpublished (console only): billed at the input rate.
        ("qwen3.8-max", 2.0),
        # Explicit-cache only in Singapore: no implicit discount.
        ("qwen3.6-plus", 0.5),
    ],
)
def test_cache_hits_bill_the_published_rate(model_id: str, cache_read: float) -> None:
    price = dashscope.models()[model_id].prices[PriceKey("auto")]
    assert price.cache_read == pytest.approx(cache_read)


def test_roles_name_the_vendor_recommendations() -> None:
    rows = dashscope.models()
    assert rows["default"].model_id == "qwen3.7-plus"
    assert rows["utility"].model_id == "qwen3.8-flash"


def test_qwen_3_8_takes_its_effort_as_a_level() -> None:
    rows = dashscope.models()
    assert rows["qwen3.8-max"].effort_as_level
    assert rows["qwen3.8-max"].thinking.effort == {
        "default",
        "none",
        "low",
        "medium",
        "xhigh",
    }
    assert not rows["qwen3.7-plus"].effort_as_level


@pytest.mark.parametrize(
    ("model_id", "budget"),
    [
        ("qwen-turbo", {"none", "auto", "fixed"}),
        ("qwen3.8-2.4t-a95b", {"auto"}),
        ("qwen3-235b-a22b-thinking-2507", {"auto", "fixed"}),
        ("qwen3-235b-a22b-instruct-2507", {"none"}),
    ],
)
def test_thinking_switches_match_the_published_modes(
    model_id: str,
    budget: set[str],
) -> None:
    """``qwen-turbo`` is hybrid; ``-thinking`` ids reject the off switch."""
    assert dashscope.models()[model_id].thinking.budget == budget


@pytest.mark.parametrize(
    ("retired", "vendor_id"),
    [
        ("qwen-max-3.8", "qwen3.8-max"),
        ("qwen-2400b-a95b-3.8", "qwen3.8-2.4t-a95b"),
        ("qwen-max-3.6", "qwen3.6-max-preview"),
        ("qwen-235b-a22b-instruct-3.0", "qwen3-235b-a22b-instruct-2507"),
        ("qwen-plus-3.0", "qwen-plus"),
    ],
)
def test_a_retired_sagent_name_reaches_its_vendor_row(
    retired: str,
    vendor_id: str,
) -> None:
    """A session saved under the old spelling loads as the vendor id."""
    rows = dashscope.models()
    assert rows[retired].model_id == vendor_id
    assert rows.exact(retired) is None


def test_a_family_word_is_not_a_model() -> None:
    """``qwen-max`` is no row's id here; only the vendor says what is newest."""
    assert "qwen-max" not in dashscope.models()


def test_qwen_plus_is_its_own_vendor_row() -> None:
    """``qwen-plus`` is a vendor id, never a family pointing at ``qwen3.7-plus``."""
    rows = dashscope.models()
    assert rows["qwen-plus"].model_id == "qwen-plus"
    assert rows["qwen-plus"] is not rows["qwen3.7-plus"]


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
