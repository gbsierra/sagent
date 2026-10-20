"""How the Responses API spells reasoning effort and reports service tiers.

Sources:
  https://developers.openai.com/api/docs/guides/reasoning
  https://developers.openai.com/api/docs/guides/flex-processing
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal


if TYPE_CHECKING:
    from sagent.types.capability import ThinkingEffort
    from sagent.types.cost import ServiceTier


__all__ = ["reasoning_effort", "served_tier"]


def reasoning_effort(
    effort: ThinkingEffort,
) -> Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]:
    """Return the Responses ``reasoning.effort`` a selected level sends.

    Every level a row offers is spelled as the API spells it; the row has
    already rejected any level the model does not take.

    Args:
      effort: Selected catalog effort.

    Returns:
      wire: Responses reasoning effort.

    Raises:
      ValueError: ``effort`` is ``default``, which omits ``reasoning``.

    """
    match effort:
        case "default":
            raise ValueError(f"effort {effort!r} is not sent as reasoning.effort")
        case "min":
            return "minimal"
        case "none" | "low" | "medium" | "high" | "xhigh" | "max":
            return effort


def served_tier(reported: str | None) -> ServiceTier | None:
    """Map the ``service_tier`` a response reports to the tier it bills at.

    Args:
      reported: The response's ``service_tier``; absent means standard.

    Returns:
      tier: Catalog tier whose card priced the request; ``None`` for a tier
        this catalog does not know. Not an error: the response is already paid
        for, so the caller decides which rate to fall back to.

    """
    match reported:
        # Scale Tier is prepaid capacity, metered at the standard card.
        case None | "auto" | "default" | "scale":
            return "auto"
        case "flex":
            return "flex"
        case "priority" | "fast":
            return "priority"
        case "ultrafast":
            return "ultrafast"
        case _:
            return None
