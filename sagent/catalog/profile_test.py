"""Tests for generated and augmented model profiles."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pytest

from sagent.catalog.profile import (
    FieldSource,
    ModeCondition,
    ModelProfile,
    ModeOverlay,
    ProfileError,
    load_generated_profiles,
    load_profile_augmentations,
)


if TYPE_CHECKING:
    from pathlib import Path


def test_missing_facts_remain_unknown() -> None:
    profile = ModelProfile(provider_id="openai", model_id="listed-only")

    assert profile.context_tokens is None
    assert profile.input_tokens is None
    assert profile.output_tokens is None
    assert profile.capabilities is None
    assert profile.prices is None


def test_source_requires_complete_vendor_evidence() -> None:
    with pytest.raises(ProfileError, match="https"):
        FieldSource(url="http://example.com", checked=date(2026, 10, 20), note="fact")
    with pytest.raises(ProfileError, match="note"):
        FieldSource(url="https://example.com", checked=date(2026, 10, 20), note="")


def test_mode_condition_must_select_an_axis() -> None:
    with pytest.raises(ProfileError, match="at least one"):
        ModeCondition()


def test_more_specific_overlay_wins() -> None:
    profile = ModelProfile(
        provider_id="alibaba",
        model_id="qwen3.8-max",
        input_tokens=991_808,
        modes=(
            ModeOverlay(
                when=ModeCondition(thinking=True),
                input_tokens=983_616,
            ),
            ModeOverlay(
                when=ModeCondition(thinking=True, effort="xhigh"),
                input_tokens=900_000,
            ),
        ),
    )

    assert (
        profile.resolve(
            context="",
            thinking=True,
            effort="medium",
            service_tier="auto",
        ).input_tokens
        == 983_616
    )
    assert (
        profile.resolve(
            context="",
            thinking=True,
            effort="xhigh",
            service_tier="auto",
        ).input_tokens
        == 900_000
    )


def test_equal_specificity_match_is_not_order_dependent() -> None:
    profile = ModelProfile(
        provider_id="example",
        model_id="ambiguous",
        modes=(
            ModeOverlay(when=ModeCondition(thinking=True), input_tokens=10),
            ModeOverlay(when=ModeCondition(effort="high"), input_tokens=20),
        ),
    )

    with pytest.raises(ProfileError, match="equal-specificity"):
        profile.resolve(
            context="",
            thinking=True,
            effort="high",
            service_tier="auto",
        )


def test_structured_provider_modes_map_to_sagent_service_tiers() -> None:
    baseline = {
        "providers": {
            "openai": {
                "models": {
                    "m": {
                        "id": "m",
                        "experimental": {
                            "modes": {
                                "arbitrary-upstream-label": {
                                    "cost": {"input": 2, "output": 4},
                                    "provider": {"body": {"service_tier": "priority"}},
                                }
                            }
                        },
                    }
                }
            }
        }
    }

    profile = load_generated_profiles(baseline)["openai"]["m"]

    assert profile.modes[0].when.service_tier == "priority"
    assert profile.modes[0].prices is not None
    assert profile.modes[0].prices.input == 2.0
    assert profile.unrecognized_modes == ()


def test_unknown_mode_is_reported_and_never_inferred_from_its_name() -> None:
    baseline = {
        "providers": {
            "openai": {
                "models": {
                    "m": {
                        "id": "m",
                        "experimental": {
                            "modes": {
                                "flex": {
                                    "provider": {"body": {"not_service_tier": "flex"}}
                                },
                                "pro": {
                                    "provider": {"body": {"reasoning": {"mode": "pro"}}}
                                },
                            }
                        },
                    }
                }
            }
        }
    }

    profile = load_generated_profiles(baseline)["openai"]["m"]

    assert profile.modes == ()
    assert profile.unrecognized_modes == ("flex", "pro")


def test_augmentation_inherits_source_and_applies_conditional_fact(
    tmp_path: Path,
) -> None:
    baseline = {
        "providers": {
            "alibaba": {"models": {"qwen": {"id": "qwen", "limit": {"input": 100}}}}
        }
    }
    path = tmp_path / "profiles.toml"
    path.write_text(
        """
[provider.alibaba]
source_url = "https://vendor.example/models"
checked = 2026-10-20

[provider.alibaba.model.qwen]
input_tokens = 90
note = "The vendor page publishes a lower normal-mode input ceiling."

[provider.alibaba.model.qwen.mode.thinking]
when_thinking = true
input_tokens = 80
note = "The same table publishes a lower ceiling while thinking."
"""
    )

    profiles = load_profile_augmentations(path, load_generated_profiles(baseline))
    profile = profiles["alibaba"]["qwen"]

    assert profile.input_tokens == 90
    assert profile.sources["input_tokens"].url == "https://vendor.example/models"
    assert profile.modes[0].source is not None
    assert (
        profile.resolve(
            context="", thinking=True, effort="default", service_tier="auto"
        ).input_tokens
        == 80
    )


def test_correction_without_note_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "profiles.toml"
    path.write_text(
        """
[provider.openai]
source_url = "https://vendor.example/models"
checked = 2026-10-20
[provider.openai.model.m]
input_tokens = 90
"""
    )
    profiles = load_generated_profiles(
        {"providers": {"openai": {"models": {"m": {"id": "m"}}}}}
    )

    with pytest.raises(ProfileError, match="requires a factual note"):
        load_profile_augmentations(path, profiles)
