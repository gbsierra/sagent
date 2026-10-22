"""llama.cpp (local; every rate is zero) model catalog, expressed as ``ModelCapability`` rows.

Source: n/a -- self-hosted

A row carries only what the MODEL can do; caching, retry, and auth mode are
transport facts declared by ``openai.compatible()``. A llama.cpp server
publishes no image pixel or byte ceiling and preprocesses images itself,
so ``ModelLimits`` carries only the two token windows.

Every row states ``inputs={"text"}``. The rows name specific text checkpoints,
and a llama.cpp server only accepts image bytes when it was loaded with a
matching ``--mmproj`` projector, which no catalog row can observe. Declaring
``text`` keeps a pasted screenshot from being posted as opaque base64 that the
server would decode as prompt text; the request boundary replaces it with a
notice naming what was dropped.
"""

from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType
from typing import TYPE_CHECKING

from sagent.catalog.table import ModelTable
from sagent.types.capability import ContextTag, ModelCapability, ModelLimits
from sagent.types.cost import (
    PriceCatalog,
    PriceKey,
    TokenPrice,
)


if TYPE_CHECKING:
    from collections.abc import Mapping


__all__ = ["models"]


def models() -> ModelTable:
    """Return every model this vendor serves, keyed by base id.

    Returns:
      models: Capability per base model id.

    """
    # The reference row: every other model states only how it differs from it.
    # A local server bills nothing, but a missing price row would raise.
    default = ModelCapability(
        model_id="qwen3.6-27b-12gb",
        inputs=frozenset({"text"}),
        context=_limits(),
        prices=PriceCatalog(
            {
                PriceKey("auto"): TokenPrice(
                    request=0.0,
                    response=0.0,
                    cache_write=0.0,
                    cache_write_1h=0.0,
                    cache_read=0.0,
                ),
            },
        ),
    )
    rows = (
        default,
        replace(
            default,
            model_id="qwen3.6-27b-mtp-64k",
            context=_limits(max_tokens=65_536, output_tokens=4_096),
        ),
        # Whatever the server happens to load; its version is unknowable, so
        # ``0.0`` marks "unversioned" and the wire id stays the server's alias.
        replace(
            default,
            model_id="local",
            context=_limits(max_tokens=32_768, output_tokens=4_096),
        ),
    )
    # A tier is offered exactly when it is priced.
    rows = tuple(replace(row, service_tier=row.prices.service_tiers) for row in rows)
    return ModelTable(
        rows=rows,
        roles={"default": default.model_id, "utility": default.model_id},
        # Sagent's own spellings before rows were keyed by vendor id; kept so
        # saved sessions and settings load. Never extend.
        renamed={
            "qwen-27b-12gb-3.6": "qwen3.6-27b-12gb",
            "qwen-27b-mtp-64k-3.6": "qwen3.6-27b-mtp-64k",
            "local-0.0": "local",
        },
    )


def _limits(
    *,
    max_tokens: int = 16_384,
    output_tokens: int = 1_024,
) -> Mapping[ContextTag, ModelLimits]:
    """One context tag: a local server ships no window variants."""
    return MappingProxyType(
        {
            "": ModelLimits(
                max_request_tokens=max_tokens,
                max_response_tokens=output_tokens,
            ),
        },
    )
