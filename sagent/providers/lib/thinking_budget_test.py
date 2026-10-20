"""Tests for the shared effort-to-token-cap table."""

from __future__ import annotations

from typing import Final, TypeAliasType, cast, get_args

from sagent.providers.lib.thinking_budget import EFFORT_BUDGET_TOKENS
from sagent.types.capability import ThinkingEffort


_EFFORTS: Final = cast(
    "tuple[ThinkingEffort, ...]",
    get_args(cast(TypeAliasType, ThinkingEffort.__value__)),
)


def test_every_explicit_thinking_effort_has_a_cap() -> None:
    assert set(EFFORT_BUDGET_TOKENS) == set(_EFFORTS) - {"default", "none"}


def test_caps_grow_with_effort() -> None:
    """A higher effort must never send a smaller cap."""
    caps = [
        EFFORT_BUDGET_TOKENS[e] for e in _EFFORTS if e not in {"default", "none"}
    ]
    assert caps == sorted(caps)
    assert len(set(caps)) == len(caps)


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
