"""Google (Gemini) model catalog, expressed as ``ModelCapability`` rows.

Sources: limits, prices, and thinking per ``_SOURCES``; also
  https://ai.google.dev/gemini-api/docs/file-input-methods

:func:`models` is the API-key view. Other transports narrow it with ``&``
(see :func:`cli`), which can only remove.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from sagent.catalog.generated.google import MODELS as GENERATED_MODELS
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


__all__ = [
    "api",
    "cli",
    "models",
    "source_models",
    "subscription",
]


# Gemini surfaces readable thought parts and offers no server-side redaction.
# A row carries only what the MODEL can do; caching, retry, auth mode, and the
# fast path are transport facts, since ``&`` can only remove and a row saying
# ``False`` would pin every transport.
def source_models() -> ModelTable:
    """Return every Gemini model, as the API-key transport sees it.

    Returns:
      models: Capability per base model id.

    """
    # Every row gets ``default`` from ``ThinkingCapability``: it means "send
    # no level", so the model's provider-owned default applies.
    levels = ThinkingCapability(
        effort=frozenset({"min", "low", "medium", "high"}),
        budget=frozenset({"none", "auto", "fixed"}),
        output=frozenset({"none", "text"}),
        control="budget",
    )
    # "minimal is not supported and returns an error" on these.
    no_minimal = replace(levels, effort=frozenset({"low", "medium", "high"}))
    budgets = replace(
        levels,
        effort=frozenset({"min", "low", "medium", "high", "xhigh", "max"}),
    )
    # The reference row: every other model states only how it differs from it.
    pro = ModelCapability(
        model_id="gemini-3.1-pro-preview",
        knowledge_cutoff="January 2025",
        inputs=frozenset({"text", "image", "audio", "video", "pdf"}),
        context=_limits(),
        prices=_prices(
            _card(request=2.0, response=12.0, cache_read=0.2),
            over_200k=_card(request=4.0, response=18.0, cache_read=0.4),
        ),
        thinking=no_minimal,
    )
    # 2.5 cannot take ``thinkingLevel``; every effort maps to a token cap.
    legacy = replace(pro, thinking=budgets)
    # The 3.6-3.8 Flash cards are introductory through 2026-12-31.
    flash = replace(
        pro,
        knowledge_cutoff=None,
        prices=_prices(
            _card(request=0.75, response=3.75, cache_read=0.075),
            regular_from_2027=_card(request=1.5, response=7.5, cache_read=0.15),
        ),
    )
    rows = (
        pro,
        replace(
            legacy,
            model_id="gemini-2.5-pro",
            prices=_prices(
                _card(request=1.25, response=10.0, cache_read=0.125),
                over_200k=_card(request=2.5, response=15.0, cache_read=0.25),
            ),
        ),
        replace(flash, model_id="gemini-3.8-flash"),
        replace(flash, model_id="gemini-3.7-flash"),
        replace(
            flash,
            model_id="gemini-3.6-flash",
            thinking=levels,
        ),
        replace(
            flash,
            model_id="gemini-3.5-flash",
            prices=_prices(_card(request=1.5, response=9.0, cache_read=0.15)),
            thinking=levels,
        ),
        # Legacy (/pricing): replaced by gemini-3.6-flash; no shutdown date yet.
        replace(
            pro,
            model_id="gemini-3-flash-preview",
            prices=_prices(_card(request=0.5, response=3.0, cache_read=0.05)),
            thinking=levels,
        ),
        replace(
            legacy,
            model_id="gemini-2.5-flash",
            prices=_prices(_card(request=0.3, response=2.5, cache_read=0.03)),
        ),
        replace(
            pro,
            model_id="gemini-3.5-flash-lite",
            knowledge_cutoff=None,
            prices=_prices(_card(request=0.3, response=2.5, cache_read=0.03)),
            thinking=levels,
        ),
        # Shuts down 2027-05-07. Audio input bills $0.50, not the text rate.
        replace(
            pro,
            model_id="gemini-3.1-flash-lite",
            prices=_prices(_card(request=0.25, response=1.5, cache_read=0.025)),
            thinking=levels,
        ),
        replace(
            legacy,
            model_id="gemini-2.5-flash-lite",
            prices=_prices(_card(request=0.1, response=0.4, cache_read=0.01)),
        ),
    )
    # Gemini publishes the effective level used when thinkingLevel is absent.
    # The 2.5 Pro/Flash rows publish only dynamic "On", not a named level, so
    # their effective label intentionally remains unknown; Flash-Lite is Off.
    # https://ai.google.dev/gemini-api/docs/thinking
    default_efforts = {
        "gemini-3.1-pro-preview": "high",
        "gemini-3.8-flash": "medium",
        "gemini-3.7-flash": "medium",
        "gemini-3.6-flash": "medium",
        "gemini-3.5-flash": "medium",
        "gemini-3-flash-preview": "high",
        "gemini-3.5-flash-lite": "min",
        "gemini-3.1-flash-lite": "min",
    }
    rows = tuple(
        replace(row, default_effort=default_efforts.get(row.model_id)) for row in rows
    )
    # A tier is offered exactly when it is priced.
    rows = tuple(replace(row, service_tier=row.prices.service_tiers) for row in rows)
    # /models: "For any new projects, use ... 3.5 Flash-Lite or 3.8 Flash".
    return ModelTable(
        rows=rows,
        roles={"default": "gemini-3.1-pro-preview", "utility": "gemini-3.5-flash-lite"},
        # Sagent's own spellings before rows were keyed by vendor id; kept so
        # saved sessions and settings load. Never extend.
        renamed={
            "gemini-pro-3.1": "gemini-3.1-pro-preview",
            "gemini-pro-2.5": "gemini-2.5-pro",
            "gemini-flash-3.8": "gemini-3.8-flash",
            "gemini-flash-3.7": "gemini-3.7-flash",
            "gemini-flash-3.6": "gemini-3.6-flash",
            "gemini-flash-3.5": "gemini-3.5-flash",
            "gemini-flash-3.0": "gemini-3-flash-preview",
            "gemini-flash-2.5": "gemini-2.5-flash",
            "gemini-flash-lite-3.5": "gemini-3.5-flash-lite",
            "gemini-flash-lite-3.1": "gemini-3.1-flash-lite",
            "gemini-flash-lite-2.5": "gemini-2.5-flash-lite",
        },
        sources=_SOURCES,
    )


def models() -> ModelTable:
    """Return the checked-in generated Google catalog."""
    return GENERATED_MODELS


_SOURCES: Final[Mapping[Fact, Source]] = MappingProxyType(
    {
        "inputs": Source(
            "https://ai.google.dev/gemini-api/docs/file-input-methods",
            date(2026, 10, 7),
        ),
        "limits": Source("https://ai.google.dev/gemini-api/docs/models"),
        "prices": Source("https://ai.google.dev/gemini-api/docs/pricing"),
        "thinking": Source(
            "https://ai.google.dev/gemini-api/docs/generate-content/thinking",
        ),
    },
)


def api() -> ModelCapability:
    """Return what the REST API adds on top of a model row.

    Nothing; named so the three transports read alike at their call sites.

    One model, three ways in::

        models()["gemini-3.1-pro-preview"] & api()          # REST, key auth
        models()["gemini-3.1-pro-preview"] & subscription() # same wire, OAuth
        models()["gemini-3.1-pro-preview"] & cli()          # ACP: no effort knob

    Returns:
      capability: The API transport's restrictions.

    """
    # Every axis stated: ``&`` can only remove, and a defaulted axis is the
    # narrow value, so an omitted one would strip the model's real capability.
    return ModelCapability(
        inputs=frozenset({"text", "image", "pdf"}),
        thinking=ThinkingCapability(
            effort=frozenset({"none", "min", "low", "medium", "high", "xhigh", "max"}),
            budget=frozenset({"none", "auto", "fixed"}),
            output=frozenset({"none", "text"}),
        ),
    )


def cli() -> ModelCapability:
    """Return what the ``gemini`` subprocess narrows a model row to.

    One model, three ways in::

        models()["gemini-3.1-pro-preview"] & api()          # REST, key auth
        models()["gemini-3.1-pro-preview"] & subscription() # same wire, OAuth
        models()["gemini-3.1-pro-preview"] & cli()          # ACP: no effort knob

    Returns:
      capability: The CLI transport's restrictions.

    """
    return ModelCapability(
        inputs=frozenset({"text", "image"}),
        thinking=ThinkingCapability(
            effort=frozenset({"none"}),
            budget=frozenset({"none", "auto", "fixed"}),
            output=frozenset({"none", "text"}),
        ),
        manage_context_server_side=frozenset({True}),
        account_auth=True,
    )


def subscription() -> ModelCapability:
    """Return what OAuth billing narrows a model row to.

    One model, three ways in::

        models()["gemini-3.1-pro-preview"] & api()          # REST, key auth
        models()["gemini-3.1-pro-preview"] & subscription() # same wire, OAuth
        models()["gemini-3.1-pro-preview"] & cli()          # ACP: no effort knob

    Returns:
      capability: The subscription transport's restrictions.

    """
    return ModelCapability(
        inputs=frozenset({"text", "image", "pdf"}),
        thinking=ThinkingCapability(
            effort=frozenset({"none", "min", "low", "medium", "high", "xhigh", "max"}),
            budget=frozenset({"none", "auto", "fixed"}),
            output=frozenset({"none", "text"}),
        ),
        account_auth=True,
    )


def _card(*, request: float, response: float, cache_read: float) -> TokenPrice:
    """Return one published price-table row; Gemini bills no cache writes."""
    return TokenPrice(
        request=request,
        response=response,
        cache_write=0.0,
        cache_write_1h=0.0,
        cache_read=cache_read,
    )


# The REST ``generateContent`` wire this catalog serves sends no
# ``service_tier``, so only the standard tab applies; the page's flex and
# priority tabs belong to the Interactions API. Audio input is billed at a
# separate rate the single input meter cannot express; the text rate is kept.
def _prices(
    standard: TokenPrice,
    *,
    over_200k: TokenPrice | None = None,
    regular_from_2027: TokenPrice | None = None,
) -> PriceCatalog:
    """Return the standard card, plus its >200k band and 2027 reprice if published."""
    cards = {PriceKey("auto"): standard}
    if over_200k is not None:
        cards[PriceKey("auto", 200_000)] = over_200k
    if regular_from_2027 is not None:
        cards[PriceKey("auto", effective_date=date(2027, 1, 1))] = regular_from_2027
    return PriceCatalog(cards)


def _limits() -> Mapping[ContextTag, ModelLimits]:
    """No per-image cap: images are tiled server-side, so only bytes bound."""
    return MappingProxyType(
        {
            "": ModelLimits(
                max_request_tokens=1_048_576,
                max_response_tokens=65_536,
                # /file-input-methods and the changelog say 100 MB inline;
                # /image-understanding still says 20 MB.
                max_request_bytes=100 * 1024 * 1024,
            ),
        },
    )
