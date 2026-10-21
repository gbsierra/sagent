"""Moonshot/Kimi model catalog, expressed as ``ModelCapability`` rows.

Sources: limits, prices, and thinking per ``_SOURCES``; also
  https://platform.kimi.ai/docs/guide/kimi-k3-quickstart
  https://platform.kimi.ai/docs/guide/kimi-k2-6-quickstart

A row carries only what the MODEL can do; caching, retry, and auth mode
are transport facts declared by ``openai.compatible()``.

Thinking: ``kimi-k3`` always reasons and takes ``reasoning_effort`` low /
high / max; ``kimi-k2.7-code`` always reasons with no knob; ``kimi-k2.6``
reasons by default and accepts ``thinking: {"type": "disabled"}``, so its
budget offers ``none`` (off) and ``auto`` (on).
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from sagent.catalog.generated.moonshot import MODELS as GENERATED_MODELS
from sagent.catalog.provenance import Source
from sagent.catalog.table import ModelTable
from sagent.types.capability import (
    ContextTag,
    ModelCapability,
    ModelLimits,
    ThinkingCapability,
)
from sagent.types.cost import (
    PriceCatalog,
    PriceKey,
    TokenPrice,
)


if TYPE_CHECKING:
    from collections.abc import Mapping

    from sagent.catalog.provenance import Fact


__all__ = ["models", "source_models"]


def source_models() -> ModelTable:
    """Return every model this vendor serves, keyed by base id.

    Returns:
      models: Capability per base model id.

    """
    always_thinks = ThinkingCapability(
        output=frozenset({"none", "text"}),
        control="always",
    )
    code = ModelCapability(
        model_id="kimi-k2.7-code",
        context=_limits(),
        prices=_prices(_card(request=0.95, response=4.0, cache_read=0.19)),
        thinking=always_thinks,
    )
    rows = (
        ModelCapability(
            model_id="kimi-k3",
            # "max_completion_tokens defaults to 131072 and can be set up to
            # 1048576".
            context=_limits(max_tokens=1_048_576, output_tokens=1_048_576),
            prices=_prices(
                _card(
                    request=3.0,
                    response=15.0,
                    cache_read=0.3,
                    cache_write=3.0,
                    cache_write_1h=6.0,
                ),
            ),
            # ``default`` sends no knob; the effective server default is max.
            thinking=replace(
                always_thinks,
                effort=frozenset({"low", "high", "max"}),
            ),
        ),
        ModelCapability(
            model_id="kimi-k2.6",
            context=_limits(),
            prices=_prices(_card(request=0.95, response=4.0, cache_read=0.16)),
            thinking=replace(
                always_thinks,
                budget=frozenset({"none", "auto"}),
                control="budget",
            ),
        ),
        code,
        replace(
            code,
            model_id="kimi-k2.7-code-highspeed",
            prices=_prices(_card(request=1.9, response=8.0, cache_read=0.38)),
        ),
    )
    # Kimi K3 publishes max as its omitted reasoning_effort default.
    rows = tuple(
        replace(row, default_effort="max") if row.model_id == "kimi-k3" else row
        for row in rows
    )
    # A tier is offered exactly when it is priced.
    rows = tuple(replace(row, service_tier=row.prices.service_tiers) for row in rows)
    return ModelTable(
        rows=rows,
        # docs/models: "Please use the latest Kimi model kimi-k3"; K2.6 is
        # the general-purpose model at a fifth of K3's input price.
        roles={"default": "kimi-k3", "utility": "kimi-k2.6"},
        # Sagent's own spellings before rows were keyed by vendor id; kept so
        # saved sessions and settings load. Never extend.
        renamed={
            "kimi-3.0": "kimi-k3",
            "kimi-2.6": "kimi-k2.6",
            "kimi-code-2.7": "kimi-k2.7-code",
            "kimi-code-highspeed-2.7": "kimi-k2.7-code-highspeed",
        },
        sources=_SOURCES,
    )


def models() -> ModelTable:
    """Return the checked-in generated Moonshot catalog."""
    return GENERATED_MODELS


_SOURCES: Final[Mapping[Fact, Source]] = MappingProxyType(
    {
        "limits": Source("https://platform.kimi.ai/docs/models", date(2026, 10, 3)),
        "prices": Source(
            "https://platform.kimi.ai/docs/pricing/chat",
            date(2026, 10, 3),
        ),
        "thinking": Source(
            "https://platform.kimi.ai/docs/guide/use-thinking-models",
            date(2026, 10, 3),
        ),
    },
)


def _card(
    *,
    request: float,
    response: float,
    cache_read: float,
    cache_write: float = 0.0,
    cache_write_1h: float = 0.0,
) -> TokenPrice:
    """Return one published price-table row; K2 bills no cache writes."""
    return TokenPrice(
        request=request,
        response=response,
        cache_write=cache_write,
        cache_write_1h=cache_write_1h,
        cache_read=cache_read,
    )


def _prices(standard: TokenPrice) -> PriceCatalog:
    """Return the one published card; K3 is "flat ... no tiering by context"."""
    return PriceCatalog({PriceKey("auto"): standard})


# K2.x publishes only its 32,768 default output, not a ceiling. The request body is
# capped at 100 MB ("ensure that the request body size does not exceed 100M").
def _limits(
    *,
    max_tokens: int = 262_144,
    output_tokens: int = 32_768,
) -> Mapping[ContextTag, ModelLimits]:
    """One context tag: Moonshot ships no window variants."""
    return MappingProxyType(
        {
            "": ModelLimits(
                max_request_tokens=max_tokens,
                max_response_tokens=output_tokens,
                max_request_bytes=100 * 1024 * 1024,
            ),
        },
    )
