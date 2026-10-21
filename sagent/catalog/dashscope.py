"""DashScope/Qwen model catalog, expressed as ``ModelCapability`` rows.

Sources (international / Singapore deployment, list prices): each live
model's page at https://www.qwencloud.com/models/<id> (read 2026-10-20)
for its limits and flat prices, ``_SOURCES`` for the rest; also
  https://www.alibabacloud.com/help/en/model-studio/context-cache
  https://www.alibabacloud.com/help/en/model-studio/vision

A row carries only what the MODEL can do; caching, retry, and auth mode
are transport facts declared by ``openai.compatible()``. The model pages
state the caps the API enforces: a 991,808-token input cap inside the
advertised 1M window (thinking mode trims it to 983K), and one output
cap per family, 131,072 for Qwen 3.7 / 3.8 and 65,536 for 3.6. Rows
whose page is gone keep the 2026-10-03 read of the text-generation table.

Thinking axes: ``budget`` ``none`` is ``enable_thinking=false``; ``auto``
thinks at the model's default depth; ``fixed`` caps it with
``thinking_budget``. Qwen 3.8's provider wire mapping takes the effort as
``reasoning_effort`` instead and rejects ``thinking_budget`` alongside it.
Effort ``none`` sends no depth knob at all.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast

from sagent.catalog.generated.dashscope import MODELS as GENERATED_MODELS
from sagent.catalog.provenance import Fact, Source
from sagent.catalog.table import ModelTable
from sagent.types.capability import (
    ContextTag,
    LimitCondition,
    LimitOverlay,
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

__all__ = ["models", "source_models"]


def source_models() -> ModelTable:
    """Return every model this vendor serves, keyed by base id.

    Returns:
      models: Capability per base model id.

    """
    hybrid = ThinkingCapability(
        effort=frozenset({"none", "min", "low", "medium", "high", "xhigh", "max"}),
        budget=frozenset({"none", "auto", "fixed"}),
        output=frozenset({"none", "text"}),
        control="budget",
    )
    # Qwen 3.8 accepts ``reasoning_effort`` low / medium / xhigh; the other
    # spellings it folds onto these, so offering them would only alias.
    levels = ThinkingCapability(
        effort=frozenset({"none", "low", "medium", "xhigh"}),
        budget=frozenset({"none", "auto"}),
        output=frozenset({"none", "text"}),
        control="budget",
    )
    # Images ride as base64 data URIs, capped at 20 MB as a STRING; base64
    # inflates 4/3, so raw bytes above ~15 MB would be rejected. Every live
    # model page states the input cap the API enforces, 991,808 of the 1M
    # window, and the output cap per family: 131,072 for Qwen 3.7 / 3.8,
    # 65,536 for 3.6. Thinking mode trims the input cap to 983K.
    vision = _limits(max_tokens=991_808, output_tokens=131_072, image_bytes=14_000_000)
    vision_64k = _limits(
        max_tokens=991_808,
        output_tokens=65_536,
        image_bytes=14_000_000,
    )
    # Text-only rows: qwen3.7-max "currently offers a pure-text-only
    # interface", and the 2.4T release ships no image input.
    text = _limits(max_tokens=991_808, output_tokens=131_072)
    rows = (
        ModelCapability(
            model_id="qwen3.8-max",
            context=vision,
            limit_overlays=_thinking_limit(vision),
            prices=_prices((0, 2.0, 6.0), cache_read=0.25),
            thinking=levels,
        ),
        ModelCapability(
            model_id="qwen3.8-flash",
            context=vision,
            limit_overlays=_thinking_limit(vision),
            prices=_prices((0, 0.15, 0.47), cache_read=0.016),
            thinking=levels,
        ),
        ModelCapability(
            model_id="qwen3.8-27b",
            # The model page lists image and video input for this release.
            context=vision,
            limit_overlays=_thinking_limit(vision),
            prices=_prices((0, 0.5, 3.0), cache_read=0.1),
            thinking=levels,
        ),
        ModelCapability(
            model_id="qwen3.8-2.4t-a95b",
            context=text,
            limit_overlays=_thinking_limit(text),
            prices=_prices((0, 2.0, 6.0), cache_read=0.25),
            # "Supports thinking mode only": no ``enable_thinking=false``.
            thinking=replace(
                levels,
                budget=frozenset({"auto"}),
                control="always",
            ),
        ),
        ModelCapability(
            model_id="qwen3.7-max",
            context=text,
            limit_overlays=_thinking_limit(text),
            prices=_prices((0, 2.5, 7.5), cache_hit=0.2),
            thinking=hybrid,
        ),
        ModelCapability(
            model_id="qwen3.7-plus",
            context=vision,
            limit_overlays=_thinking_limit(vision),
            prices=_prices((0, 0.4, 1.6), (256_000, 1.2, 4.8), cache_hit=0.2),
            thinking=hybrid,
        ),
        ModelCapability(
            model_id="qwen3.7-flash",
            context=vision,
            limit_overlays=_thinking_limit(vision),
            prices=_prices(
                (0, 0.03, 0.13),
                (32_000, 0.10, 0.40),
                (256_000, 0.20, 0.80),
                cache_hit=0.2,
            ),
            thinking=hybrid,
        ),
        # The qwen3.6 rows support only EXPLICIT cache in Singapore, which
        # needs a ``cache_control`` marker this transport never writes.
        ModelCapability(
            model_id="qwen3.6-max-preview",
            context=_limits(max_tokens=262_144),
            prices=_prices((0, 1.3, 7.8), (128_000, 2.0, 12.0), cache_hit=1.0),
            thinking=hybrid,
        ),
        ModelCapability(
            model_id="qwen3.6-plus",
            context=vision_64k,
            limit_overlays=_thinking_limit(vision_64k),
            prices=_prices((0, 0.5, 3.0), (256_000, 2.0, 6.0), cache_hit=1.0),
            thinking=hybrid,
        ),
        ModelCapability(
            model_id="qwen3.6-flash",
            context=vision_64k,
            limit_overlays=_thinking_limit(vision_64k),
            prices=_prices((0, 0.25, 1.5), (256_000, 1.0, 4.0), cache_hit=1.0),
            thinking=hybrid,
        ),
        # ``-instruct`` and ``-coder`` reject the toggle, so every thinking
        # axis offers only its off value; their open-source deployments carry
        # no implicit cache.
        ModelCapability(
            model_id="qwen3-235b-a22b-instruct-2507",
            context=_limits(max_tokens=262_144),
            prices=_prices((0, 0.23, 0.92), cache_hit=1.0),
        ),
        ModelCapability(
            model_id="qwen3-235b-a22b-thinking-2507",
            context=_limits(max_tokens=262_144, output_tokens=32_768),
            prices=_prices((0, 0.23, 2.3), cache_hit=1.0),
            # These ids reject ``enable_thinking=false``.
            thinking=replace(
                hybrid,
                effort=hybrid.effort - {"none"},
                budget=hybrid.budget - {"none"},
            ),
        ),
        ModelCapability(
            model_id="qwen3-30b-a3b-instruct-2507",
            context=_limits(max_tokens=262_144),
            prices=_prices((0, 0.2, 0.8), cache_hit=1.0),
        ),
        ModelCapability(
            model_id="qwen3-32b",
            context=_limits(max_tokens=262_144),
            prices=_prices((0, 0.16, 0.64), cache_hit=1.0),
            thinking=hybrid,
        ),
        ModelCapability(
            model_id="qwen3-coder-480b-a35b-instruct",
            context=_limits(max_tokens=262_144),
            prices=_prices(
                (0, 1.5, 7.5),
                (32_000, 2.7, 13.5),
                (128_000, 4.5, 22.5),
                cache_hit=1.0,
            ),
        ),
        # The unversioned aliases are named by the generation deep-thinking
        # files them under ("Qwen3 ... Commercial version"). ``qwen-max`` is
        # absent: no page states its generation.
        ModelCapability(
            model_id="qwen-plus",
            context=_limits(output_tokens=32_768),
            # The page states a 995K input cap but not the exact figure, so
            # the row keeps the 1M context. Non-thinking card; thinking
            # output bills $4 / $12, which no ``PriceKey`` axis can select.
            prices=_prices((0, 0.4, 1.2), (256_000, 1.2, 3.6), cache_hit=0.2),
            thinking=hybrid,
        ),
        ModelCapability(
            model_id="qwen-turbo",
            context=_limits(output_tokens=32_768),
            # Non-thinking card; thinking output bills $0.5, which no
            # ``PriceKey`` axis can select.
            prices=_prices((0, 0.05, 0.2), cache_hit=0.2),
            thinking=hybrid,
        ),
    )
    # Qwen 3.8 publishes xhigh when reasoning_effort is omitted. Older rows
    # publish a dynamic/default token budget rather than a named effort.
    # https://help.aliyun.com/en/model-studio/qwen-api-via-openai-responses
    rows = tuple(
        replace(row, default_effort="xhigh")
        if row.model_id
        in {
            "qwen3.8-max",
            "qwen3.8-flash",
            "qwen3.8-27b",
            "qwen3.8-2.4t-a95b",
        }
        else row
        for row in rows
    )
    # A tier is offered exactly when it is priced.
    rows = tuple(replace(row, service_tier=row.prices.service_tiers) for row in rows)
    return ModelTable(
        rows=rows,
        # text-generation-model: "Start with qwen3.7-plus ... To cut costs,
        # switch to qwen3.8-flash".
        roles={"default": "qwen3.7-plus", "utility": "qwen3.8-flash"},
        # Sagent's own spellings before rows were keyed by vendor id; kept so
        # saved sessions and settings load. Never extend.
        renamed={
            "qwen-max-3.8": "qwen3.8-max",
            "qwen-flash-3.8": "qwen3.8-flash",
            "qwen-27b-3.8": "qwen3.8-27b",
            "qwen-2400b-a95b-3.8": "qwen3.8-2.4t-a95b",
            "qwen-max-3.7": "qwen3.7-max",
            "qwen-plus-3.7": "qwen3.7-plus",
            "qwen-flash-3.7": "qwen3.7-flash",
            "qwen-max-3.6": "qwen3.6-max-preview",
            "qwen-plus-3.6": "qwen3.6-plus",
            "qwen-flash-3.6": "qwen3.6-flash",
            "qwen-235b-a22b-instruct-3.0": "qwen3-235b-a22b-instruct-2507",
            "qwen-235b-a22b-thinking-3.0": "qwen3-235b-a22b-thinking-2507",
            "qwen-30b-a3b-instruct-3.0": "qwen3-30b-a3b-instruct-2507",
            "qwen-32b-3.0": "qwen3-32b",
            "qwen-coder-480b-a35b-3.0": "qwen3-coder-480b-a35b-instruct",
            "qwen-plus-3.0": "qwen-plus",
            "qwen-turbo-3.0": "qwen-turbo",
        },
        sources=_SOURCES,
        # The live rows read their limits, and where the page states every
        # band their prices, from each model's own qwencloud page.
        row_sources=_PAGE_READ,
    )


def models() -> ModelTable:
    """Return the checked-in generated DashScope catalog."""
    return GENERATED_MODELS


_DOCS: Final = "https://www.alibabacloud.com/help/en/model-studio"

_SOURCES: Final[Mapping[Fact, Source]] = MappingProxyType(
    {
        "limits": Source(f"{_DOCS}/text-generation-model", date(2026, 10, 3)),
        "prices": Source(f"{_DOCS}/model-pricing", date(2026, 10, 3)),
        "thinking": Source(f"{_DOCS}/deep-thinking", date(2026, 10, 3)),
    },
)


def _page(mid: str, *facts: Fact) -> Mapping[Fact, Source]:
    """Return ``facts`` sourced from the model's qwencloud page (2026-10-20)."""
    page = Source(f"https://www.qwencloud.com/models/{mid}", date(2026, 10, 20))
    return MappingProxyType(dict.fromkeys(facts, page))


# Tiered rows keep the table-wide pricing source for their higher bands,
# which no model page states.
_PAGE_READ: Final[Mapping[str, Mapping[Fact, Source]]] = MappingProxyType(
    {
        mid: _page(mid, *cast(tuple[Fact, ...], facts))
        for mid, facts in (
            ("qwen3.8-max", ("limits", "prices")),
            ("qwen3.8-flash", ("limits", "prices")),
            ("qwen3.8-27b", ("limits", "prices")),
            ("qwen3.8-2.4t-a95b", ("limits", "prices")),
            ("qwen3.7-max", ("limits", "prices")),
            ("qwen3.7-plus", ("limits",)),
            ("qwen3.7-flash", ("limits",)),
            ("qwen3.6-plus", ("limits",)),
            ("qwen3.6-flash", ("limits",)),
        )
    },
)


# Implicit cache creation bills as plain input, and this transport never writes the
# explicit marker, so a reported write bills at the input rate. A hit bills the
# price the model page publishes (``cache_read``) or the input rate times the
# rate the vendor states (``cache_hit``; Qwen 3.7 hits bill 20%).
def _prices(
    *bands: tuple[int, float, float],
    cache_hit: float | None = None,
    cache_read: float | None = None,
) -> PriceCatalog:
    """Return the published bands; each bills a whole prompt larger than its floor."""
    if (cache_hit is None) == (cache_read is None):
        msg = "pass exactly one of cache_hit and cache_read"
        raise ValueError(msg)

    def hit(request: float) -> float:
        return cache_read if cache_read is not None else request * (cache_hit or 0.0)

    return PriceCatalog(
        {
            PriceKey("auto", floor): TokenPrice(
                request=request,
                response=response,
                cache_write=request,
                cache_write_1h=0.0,
                cache_read=hit(request),
            )
            for floor, request, response in bands
        },
    )


def _thinking_limit(
    limits: Mapping[ContextTag, ModelLimits],
) -> tuple[LimitOverlay, ...]:
    """Return the model-page input ceiling used while thinking is enabled."""
    base = limits[""]
    return (
        LimitOverlay(
            when=LimitCondition(thinking=True),
            limits=replace(base, max_request_tokens=983_616),
        ),
    )


def _limits(
    *,
    max_tokens: int = 1_000_000,
    output_tokens: int = 65_536,
    image_bytes: int = 0,
) -> Mapping[ContextTag, ModelLimits]:
    """One context tag: DashScope ships no window variants."""
    return MappingProxyType(
        {
            "": ModelLimits(
                max_request_tokens=max_tokens,
                max_response_tokens=output_tokens,
                max_image_bytes=image_bytes,
            ),
        },
    )
