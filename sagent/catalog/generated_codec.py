"""Strict plain-data boundary for checked-in generated model catalogs.

Generated modules contain data, not executable provider logic.  The normal
dataclass codec handles the catalog's typed structure; the two hooks below
replace Python reduce recipes with explicit representations for the only
non-dataclass values in a table: dates and price catalogs.
"""

from __future__ import annotations

from datetime import date
from types import MappingProxyType
from typing import TypeGuard, cast, get_args

from sagent.catalog.table import ModelTable
from sagent.lib.codec import Hooks, PlainTree, from_plain, to_plain
from sagent.types.cost import PriceCatalog, PriceKey, ServiceTier, TokenPrice


__all__ = ["decode_catalog", "encode_catalog"]


def _encode_prices(catalog: PriceCatalog) -> list[list[object]]:
    """Encode prices without pickle/import recipes."""
    return [
        [
            key.service_tier,
            key.effective_prompt_tokens,
            key.effective_date.isoformat(),
            price.request,
            price.response,
            price.cache_write,
            price.cache_write_1h,
            price.cache_read,
            price.tokens_per_unit,
        ]
        for key, price in catalog.items()
    ]


def _decode_prices(raw: object) -> PriceCatalog:
    """Decode the exact price row shape emitted by :func:`_encode_prices`."""
    if not isinstance(raw, list):
        raise TypeError("generated price catalog must be a list")
    prices: dict[PriceKey, TokenPrice] = {}
    service_tiers = get_args(ServiceTier.__value__)
    for index, raw_item in enumerate(cast(list[object], raw)):
        if not isinstance(raw_item, list):
            raise TypeError(f"generated price row {index} must have 9 fields")
        item = cast(list[object], raw_item)
        if len(item) != 9:
            raise TypeError(f"generated price row {index} must have 9 fields")
        tier, threshold, effective, request, response, write, write_1h, read, unit = (
            item
        )
        if not isinstance(tier, str) or tier not in service_tiers:
            raise TypeError(f"generated price row {index} has invalid tier")
        if type(threshold) is not int or not isinstance(effective, str):
            raise TypeError(f"generated price row {index} has invalid condition")
        rates = (request, response, write, write_1h, read)
        if not all(_is_number(value) for value in rates):
            raise TypeError(f"generated price row {index} has invalid rates")
        typed_rates = cast(tuple[int | float, ...], rates)
        if type(unit) is not int or unit <= 0:
            raise TypeError(f"generated price row {index} has invalid unit")
        key = PriceKey(
            cast(ServiceTier, tier),
            threshold,
            date.fromisoformat(effective),
        )
        prices[key] = TokenPrice(
            request=float(typed_rates[0]),
            response=float(typed_rates[1]),
            cache_write=float(typed_rates[2]),
            cache_write_1h=float(typed_rates[3]),
            cache_read=float(typed_rates[4]),
            tokens_per_unit=unit,
        )
    return PriceCatalog(prices)


def _is_number(value: object) -> TypeGuard[int | float]:
    """Accept real numeric literals but not bool's integer subclass."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


_HOOKS: Hooks = MappingProxyType(
    {
        PriceCatalog: (_encode_prices, _decode_prices),
        date: (date.isoformat, date.fromisoformat),
    },
)


def encode_catalog(table: ModelTable) -> PlainTree:
    """Return deterministic, inert data for one complete provider table."""
    # Encode constructor inputs only. ``ModelTable._by_id`` is a derived index;
    # serializing it would duplicate every row and make generated output depend
    # on an implementation detail that ``__post_init__`` already rebuilds.
    constructor = {
        "rows": table.rows,
        "roles": table.roles,
        "aliases": table.aliases,
        "renamed": table.renamed,
        "sources": table.sources,
        "row_sources": table.row_sources,
    }
    return to_plain(constructor, dialect="python", mutable=True, hooks=_HOOKS)


def decode_catalog(data: PlainTree) -> ModelTable:
    """Reconstruct a generated table without permitting arbitrary imports."""
    return from_plain(data, ModelTable, hooks=_HOOKS)
