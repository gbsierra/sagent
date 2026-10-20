"""How a Gemini request spells Sagent's thinking selection.

Source: https://ai.google.dev/gemini-api/docs/generate-content/thinking
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sagent.providers.lib.thinking_budget import EFFORT_BUDGET_TOKENS


if TYPE_CHECKING:
    from sagent.lib.codec import MutablePlainTree
    from sagent.types.capability import (
        ModelCapability,
        ModelSettings,
        ThinkingEffort,
    )


__all__ = ["thinking_config"]


def thinking_config(
    capability: ModelCapability,
    settings: ModelSettings,
) -> dict[str, MutablePlainTree] | None:
    """Return the ``generationConfig.thinkingConfig`` a selection sends.

    Budget ``none`` omits the config and effort ``none`` omits the level or
    cap, so the model's own default applies in both cases. Gemini 3 rows take
    a ``thinkingLevel`` under either budget mode, its levels being dynamic
    already; 2.5 rows take a ``thinkingBudget`` token cap, ``-1`` under
    ``auto``.

    Args:
      capability: The row the request is built against.
      settings: The selection in effect.

    Returns:
      config: The wire object, or ``None`` when the request sends none.

    """
    if settings.thinking_budget == "none":
        return None
    config: dict[str, MutablePlainTree] = {
        "includeThoughts": settings.thinking_output == "text",
    }
    effort = settings.thinking_effort
    if capability.effort_as_level:
        # Gemini 3 accepts ``thinkingBudget`` only for backwards compatibility.
        if effort != "default":
            config["thinkingLevel"] = _thinking_level(effort)
    elif settings.thinking_budget == "auto":
        config["thinkingBudget"] = -1
    elif effort != "default":
        config["thinkingBudget"] = EFFORT_BUDGET_TOKENS[effort]
    return config


def _thinking_level(effort: ThinkingEffort) -> str:
    """Return the Gemini 3 ``thinkingLevel`` an effort sends."""
    match effort:
        case "min":
            return "minimal"
        case "low" | "medium" | "high":
            return effort
        case "default" | "none" | "xhigh" | "max":
            raise ValueError(f"effort {effort!r} has no Gemini thinkingLevel")
