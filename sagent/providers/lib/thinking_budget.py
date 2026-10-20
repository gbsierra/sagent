"""Token caps sent for each thinking effort by transports that take a cap.

Gemini 2.5's ``thinkingBudget`` and DashScope's ``thinking_budget`` both bound
reasoning in tokens rather than by level. Neither vendor publishes a mapping
from Sagent's effort scale, so one table serves both: a higher effort must
never send a smaller cap, and two copies would drift apart.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final


if TYPE_CHECKING:
    from collections.abc import Mapping

    from sagent.types.capability import ThinkingEffort


__all__ = ["EFFORT_BUDGET_TOKENS"]


# No ``default`` or ``none``: provider policy omits the cap, and a request that
# does not think omits it rather than sending a zero some models reject.
EFFORT_BUDGET_TOKENS: Final[Mapping[ThinkingEffort, int]] = MappingProxyType(
    {
        "min": 1024,
        "low": 4096,
        "medium": 8192,
        "high": 16384,
        "xhigh": 20480,
        "max": 24576,
    },
)
"""Reasoning-token cap for each explicit thinking effort."""
