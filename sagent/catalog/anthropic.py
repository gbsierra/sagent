"""Anthropic model catalog, expressed as ``ModelCapability`` rows.

Sources: limits, prices, and thinking per ``_SOURCES`` (a few rows override
it in :func:`models`); also
  - Fast mode: https://docs.anthropic.com/en/docs/build-with-claude/fast-mode
  - Vision: https://platform.claude.com/docs/en/build-with-claude/vision
  - Request size: https://platform.claude.com/docs/en/api/overview

:func:`models` is the API-key view. Other transports narrow it with ``&``
(see :func:`cli`), which can only remove.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from sagent.catalog.generated.anthropic import MODELS as GENERATED_MODELS
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
    from collections.abc import (
        Mapping,
    )

    from sagent.catalog.provenance import Fact


__all__ = [
    "CACHE_TTL_SEC",
    "api",
    "cli",
    "models",
    "source_models",
    "subscription",
]


# No ``0``: these transports write a breakpoint on every request, so "do not
# cache" is not a selection this vendor offers.
CACHE_TTL_SEC: Final = frozenset({300.0, 3600.0})


# Thinking capability per https://platform.claude.com/docs/en/build-with-claude/thinking
# (verified 2026-10-03). ``auto`` is ``thinking.type=adaptive`` (no budget);
# ``fixed`` is ``enabled`` + ``budget_tokens``. Output ``text`` sends
# ``display: "summarized"`` and ``redacted`` sends ``"omitted"``; every row
# offers both, because 4.7-and-later default to ``omitted`` but accept an
# explicit ``summarized``.
#
# Budget and output sets state ``none``: a model that CAN think can also skip
# an explicit thinking request, and each axis is total, so omitting ``none``
# would make the provider default unselectable. Effort sets omit ``none`` --
# the Anthropic wire cannot spell it -- and the capability type adds
# ``default`` itself.
#
# Knowledge cutoffs are the vendor's "reliable knowledge cutoff", and prices
# its published table (input, output, 5m/1h cache write, cache hit), both
# verified on 2026-10-03.
def source_models() -> ModelTable:
    """Return every Anthropic model, as the API-key transport sees it.

    Returns:
      models: Capability per base model id.

    """
    # The reference row: every other model states only how it differs from it.
    default = ModelCapability(
        model_id="claude-opus-5-5",
        knowledge_cutoff="June 2026",
        approx_chars_per_token=2.38,
        context=_limits(),
        prices=_prices(
            _card(
                request=4.0,
                response=20.0,
                cache_write=5.0,
                cache_write_1h=8.0,
                cache_read=0.2,
            ),
            fast_request=8.0,
            fast_response=40.0,
        ),
        thinking=ThinkingCapability(
            effort=frozenset({"low", "medium", "high", "xhigh", "max"}),
            budget=frozenset({"none", "auto"}),
            output=frozenset({"none", "text", "redacted"}),
            control="budget",
        ),
    )
    # The 4.6-and-earlier reasoning contract: a fixed budget alongside adaptive.
    extended = replace(
        default.thinking,
        budget=frozenset({"none", "auto", "fixed"}),
    )
    fable_5_1 = replace(
        default,
        model_id="claude-fable-5-1",
        prices=_prices(
            _card(
                request=10.0,
                response=50.0,
                cache_write=12.5,
                cache_write_1h=20.0,
                cache_read=0.25,
            ),
        ),
    )
    fable_5_0 = replace(
        default,
        model_id="claude-fable-5",
        knowledge_cutoff="January 2026",
        prices=_prices(
            _card(
                request=10.0,
                response=50.0,
                cache_write=12.5,
                cache_write_1h=20.0,
                cache_read=1.0,
            ),
        ),
    )
    rows = (
        fable_5_1,
        fable_5_0,
        # Invitation-only (Project Glasswing); same specs and prices as Fable.
        replace(fable_5_1, model_id="claude-mythos-5-1"),
        replace(fable_5_0, model_id="claude-mythos-5"),
        default,
        replace(
            default,
            model_id="claude-opus-5",
            knowledge_cutoff="May 2026",
            prices=_prices(
                _card(
                    request=5.0,
                    response=25.0,
                    cache_write=6.25,
                    cache_write_1h=10.0,
                    cache_read=0.5,
                ),
                fast_request=10.0,
                fast_response=50.0,
            ),
        ),
        replace(
            default,
            model_id="claude-opus-4-8",
            knowledge_cutoff="January 2026",
            prices=_prices(
                _card(
                    request=5.0,
                    response=25.0,
                    cache_write=6.25,
                    cache_write_1h=10.0,
                    cache_read=0.5,
                ),
                fast_request=10.0,
                fast_response=50.0,
            ),
        ),
        # No fast price row on 4-7 or 4-6: 4-7 lost fast mode on 2026-07-24 and
        # the API now rejects ``speed="fast"``; 4-6 never shipped it and bills
        # standard, so a fast row would misprice both.
        replace(
            default,
            model_id="claude-opus-4-7",
            knowledge_cutoff="January 2026",
            prices=_prices(
                _card(
                    request=5.0,
                    response=25.0,
                    cache_write=6.25,
                    cache_write_1h=10.0,
                    cache_read=0.5,
                ),
            ),
        ),
        replace(
            default,
            model_id="claude-opus-4-6",
            knowledge_cutoff="May 2025",
            approx_chars_per_token=3.12,
            context=_limits(image_edge_px=1568),
            prices=_prices(
                _card(
                    request=5.0,
                    response=25.0,
                    cache_write=6.25,
                    cache_write_1h=10.0,
                    cache_read=0.5,
                ),
            ),
            thinking=replace(
                extended,
                effort=frozenset({"low", "medium", "high", "max"}),
            ),
        ),
        replace(
            default,
            model_id="claude-opus-4-5",
            knowledge_cutoff="May 2025",
            approx_chars_per_token=3.12,
            context=_limits(
                max_tokens=200_000,
                output_tokens=64_000,
                image_edge_px=1568,
            ),
            prices=_prices(
                _card(
                    request=5.0,
                    response=25.0,
                    cache_write=6.25,
                    cache_write_1h=10.0,
                    cache_read=0.5,
                ),
            ),
            thinking=replace(
                extended,
                effort=frozenset({"low", "medium", "high"}),
                budget=frozenset({"none", "fixed"}),
            ),
        ),
        replace(
            default,
            model_id="claude-sonnet-5-5",
            prices=_prices(
                # Cache hits are 0.05x input here, not the standard 0.1x.
                _card(
                    request=2.0,
                    response=10.0,
                    cache_write=2.5,
                    cache_write_1h=4.0,
                    cache_read=0.1,
                ),
            ),
        ),
        replace(
            default,
            model_id="claude-sonnet-5",
            knowledge_cutoff="January 2026",
            prices=_prices(
                _card(
                    request=2.0,
                    response=10.0,
                    cache_write=2.5,
                    cache_write_1h=4.0,
                    cache_read=0.2,
                ),
            ),
        ),
        replace(
            default,
            model_id="claude-sonnet-4-6",
            knowledge_cutoff="August 2025",
            approx_chars_per_token=3.12,
            context=_limits(image_edge_px=1568),
            prices=_prices(
                _card(
                    request=3.0,
                    response=15.0,
                    cache_write=3.75,
                    cache_write_1h=6.0,
                    cache_read=0.3,
                ),
            ),
            thinking=replace(
                extended,
                effort=frozenset({"low", "medium", "high", "max"}),
            ),
        ),
        # Deprecated 2026-09-30, retiring 2026-11-30 (replacement
        # ``claude-sonnet-5-5``); requests fail after that date. Source:
        # https://platform.claude.com/docs/en/about-claude/model-deprecations
        # (read 2026-10-17).
        replace(
            default,
            model_id="claude-sonnet-4-5",
            knowledge_cutoff="January 2025",
            approx_chars_per_token=3.12,
            context=_limits(
                max_tokens=200_000,
                output_tokens=64_000,
                image_edge_px=1568,
            ),
            prices=_prices(
                _card(
                    request=3.0,
                    response=15.0,
                    cache_write=3.75,
                    cache_write_1h=6.0,
                    cache_read=0.3,
                ),
            ),
            thinking=replace(
                extended,
                effort=frozenset(),
                budget=frozenset({"none", "fixed"}),
            ),
        ),
        # The one row priced by prompt length: a request whose prompt (cache
        # reads and writes included) is over 100,000 tokens pays the upper
        # band. No fast mode. Sources (both read 2026-10-17):
        # https://platform.claude.com/docs/en/models/haiku-5-5/overview
        # https://platform.claude.com/docs/en/about-claude/pricing#long-context-pricing
        replace(
            default,
            model_id="claude-haiku-5-5",
            prices=PriceCatalog(
                {
                    PriceKey("auto"): _card(
                        request=0.1,
                        response=0.5,
                        cache_write=0.125,
                        cache_write_1h=0.2,
                        cache_read=0.01,
                    ),
                    PriceKey("auto", effective_prompt_tokens=100_000): _card(
                        request=0.5,
                        response=2.5,
                        cache_write=0.625,
                        cache_write_1h=1.0,
                        cache_read=0.05,
                    ),
                },
            ),
        ),
        replace(
            default,
            model_id="claude-haiku-4-5",
            knowledge_cutoff="February 2025",
            approx_chars_per_token=3.12,
            context=_limits(
                max_tokens=200_000,
                output_tokens=64_000,
                image_edge_px=1568,
            ),
            prices=_prices(
                _card(
                    request=1.0,
                    response=5.0,
                    cache_write=1.25,
                    cache_write_1h=2.0,
                    cache_read=0.1,
                ),
            ),
            thinking=replace(
                extended,
                effort=frozenset(),
                budget=frozenset({"none", "fixed"}),
            ),
        ),
    )
    # 4.5 and earlier ids are dateless aliases of dated snapshots, and the API
    # reports the snapshot it served; 4.6 and later are dateless and pinned.
    # Source: https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions
    # (checked 2026-10-17). The haiku pair is also what ``claude`` 2.1.227's
    # initialize handshake reports for its ``haiku`` alias (probed 2026-10-17).
    by_id = {row.model_id: row for row in rows}
    rows += tuple(
        replace(by_id[alias], model_id=snapshot)
        for alias, snapshot in (
            ("claude-opus-4-5", "claude-opus-4-5-20251101"),
            ("claude-sonnet-4-5", "claude-sonnet-4-5-20250929"),
            ("claude-haiku-4-5", "claude-haiku-4-5-20251001"),
        )
    )
    # output_config.effort defaults to high on every supported model except
    # Opus 5.5 and Haiku 5.5, whose published default is medium.
    # https://platform.claude.com/docs/en/build-with-claude/effort
    medium_defaults = {"claude-opus-5-5", "claude-haiku-5-5"}
    rows = tuple(
        replace(
            row,
            default_effort=(
                "medium"
                if row.model_id in medium_defaults
                else "high"
                if row.thinking.effort - {"default", "none"}
                else None
            ),
        )
        for row in rows
    )
    # A tier is offered exactly when it is priced.
    rows = tuple(replace(row, service_tier=row.prices.service_tiers) for row in rows)
    return ModelTable(
        rows=rows,
        roles={
            "best": "claude-fable-5-1",
            "default": "claude-opus-5-5",
            "utility": "claude-sonnet-5-5",
        },
        renamed=_RENAMED,
        sources=_SOURCES,
        row_sources={
            "claude-sonnet-5-5": {"prices": Source(_PRICING, date(2026, 10, 18))},
            "claude-haiku-5-5": {
                "limits": Source(
                    "https://platform.claude.com/docs/en/models/haiku-5-5/overview",
                    date(2026, 10, 17),
                ),
                "prices": Source(
                    f"{_PRICING}#long-context-pricing",
                    date(2026, 10, 17),
                ),
            },
        },
    )


def models() -> ModelTable:
    """Return the checked-in generated Anthropic catalog."""
    return GENERATED_MODELS


_PRICING: Final = "https://platform.claude.com/docs/en/about-claude/pricing"

_SOURCES: Final[Mapping[Fact, Source]] = MappingProxyType(
    {
        "limits": Source("https://docs.anthropic.com/en/docs/about-claude/models"),
        "prices": Source(_PRICING, date(2026, 10, 3)),
        "thinking": Source(
            "https://platform.claude.com/docs/en/build-with-claude/thinking",
            date(2026, 10, 3),
        ),
    },
)


# Sagent's own spellings before rows were keyed by vendor id; kept so saved
# sessions and settings load. Never extend: a new row has only its vendor id.
_RENAMED: Final = MappingProxyType(
    {
        "fable-5.1": "claude-fable-5-1",
        "fable-5.0": "claude-fable-5",
        "mythos-5.1": "claude-mythos-5-1",
        "mythos-5.0": "claude-mythos-5",
        "opus-5.5": "claude-opus-5-5",
        "opus-5.0": "claude-opus-5",
        "opus-4.8": "claude-opus-4-8",
        "opus-4.7": "claude-opus-4-7",
        "opus-4.6": "claude-opus-4-6",
        "opus-4.5": "claude-opus-4-5",
        "sonnet-5.5": "claude-sonnet-5-5",
        "sonnet-5.0": "claude-sonnet-5",
        "sonnet-4.6": "claude-sonnet-4-6",
        "sonnet-4.5": "claude-sonnet-4-5",
        "haiku-4.5": "claude-haiku-4-5",
    },
)


# A row carries only what the MODEL can do; caching, retry, and auth mode are
# transport facts. ``&`` can only remove, so a row asserting ``False`` would
# pin every transport -- these declare it per transport instead.


def api() -> ModelCapability:
    """Return what the Messages API adds on top of a model row.

    One model, three ways in::

        models()["claude-opus-5"] & api()          # API key: full tiers, cache
        models()["claude-opus-5"] & subscription() # same wire, OAuth-billed
        models()["claude-opus-5"] & cli()          # subprocess: no effort/tier

    Returns:
      capability: The API transport's restrictions.

    """
    return ModelCapability(
        thinking=ThinkingCapability(
            effort=frozenset({"none", "min", "low", "medium", "high", "xhigh", "max"}),
            budget=frozenset({"none", "auto", "fixed"}),
            output=frozenset({"none", "text", "redacted"}),
        ),
        cache_ttl_sec=CACHE_TTL_SEC,
        manage_context_server_side=frozenset({False, True}),
        retries_internally=True,
        # No ``flex``: the Messages API takes only ``auto`` / ``standard_only``
        # (https://platform.claude.com/docs/en/api/messages/create), which map to
        # ``auto`` / ``default`` here. ``priority`` is the fast-mode beta.
        service_tier=frozenset({"auto", "default", "priority"}),
    )


def cli() -> ModelCapability:
    """Return what the ``claude`` subprocess narrows a model row to.

    One model, three ways in::

        models()["claude-opus-5"] & api()          # API key: full tiers, cache
        models()["claude-opus-5"] & subscription() # same wire, OAuth-billed
        models()["claude-opus-5"] & cli()          # subprocess: no effort/tier

    Returns:
      capability: The CLI transport's restrictions.

    """
    return ModelCapability(
        thinking=ThinkingCapability(
            effort=frozenset(),
            budget=frozenset({"none", "auto", "fixed"}),
            output=frozenset({"none", "text"}),
        ),
        manage_context_server_side=frozenset({True}),
    )


def subscription() -> ModelCapability:
    """Return what OAuth billing narrows a model row to.

    One model, three ways in::

        models()["claude-opus-5"] & api()          # API key: full tiers, cache
        models()["claude-opus-5"] & subscription() # same wire, OAuth-billed
        models()["claude-opus-5"] & cli()          # subprocess: no effort/tier

    Returns:
      capability: The subscription transport's restrictions.

    """
    return replace(api(), account_auth=True)


def _card(
    *,
    request: float,
    response: float,
    cache_write: float,
    cache_write_1h: float,
    cache_read: float,
) -> TokenPrice:
    """Return one published price-table row: input, output, 5m/1h writes, hits."""
    return TokenPrice(
        request=request,
        response=response,
        cache_write=cache_write,
        cache_write_1h=cache_write_1h,
        cache_read=cache_read,
    )


# Fast mode publishes only input and output; "prompt caching multipliers apply
# on top of fast mode pricing", so each cache rate keeps its ratio to input.
def _prices(
    standard: TokenPrice,
    *,
    fast_request: float = 0.0,
    fast_response: float = 0.0,
) -> PriceCatalog:
    """Return the standard card, plus the fast-mode card when the model has one."""
    cards = {PriceKey("auto"): standard}
    if fast_request:
        scale = fast_request / standard.request
        cards[PriceKey("priority")] = _card(
            request=fast_request,
            response=fast_response,
            cache_write=standard.cache_write * scale,
            cache_write_1h=standard.cache_write_1h * scale,
            cache_read=standard.cache_read * scale,
        )
    return PriceCatalog(cards)


def _limits(
    *,
    max_tokens: int = 1_000_000,
    output_tokens: int = 128_000,
    image_edge_px: int = 2576,
) -> Mapping[ContextTag, ModelLimits]:
    """Serve ``max_tokens`` by default; ``+200k`` selects 200k when it is larger."""
    limits = ModelLimits(
        max_request_tokens=max_tokens,
        max_response_tokens=output_tokens,
        max_request_bytes=32 * 1024 * 1024,
        max_image_edge_px=image_edge_px,
        # The vision page caps 10 MB *base64-encoded*; this cap is applied to raw
        # bytes before encoding, which inflates them by 4/3.
        max_image_bytes=10 * 1000 * 1000 * 3 // 4,
    )
    context: dict[ContextTag, ModelLimits] = {"": limits}
    if max_tokens > 200_000:
        context["+200k"] = replace(limits, max_request_tokens=200_000)
    return MappingProxyType(context)
