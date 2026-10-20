"""Where the catalogs disagree with models.dev, and which disagreements are known.

models.dev (https://models.dev/api.json) is a community aggregate: a seed
and a drift detector, never a source of truth. A disagreement means a page
should be reopened, not that either side is right. One that has been
reopened and resolved in Sagent's favor is listed in :data:`ACCEPTED`,
pinned to the value models.dev showed then, so a later change on its side
surfaces again.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import math

import httpx2

from sagent.catalog import anthropic, dashscope, google, minimax, moonshot, openai
from sagent.catalog.provenance import Source
from sagent.catalog.sources.live import describe
from sagent.lib.codec import from_plain
from sagent.types.cost import PriceKey


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from datetime import date

    from sagent.catalog.provenance import Fact
    from sagent.catalog.sources.live import Warn
    from sagent.catalog.table import ModelTable


__all__ = [
    "ACCEPTED",
    "CATALOGS",
    "URL",
    "Drift",
    "compare",
    "fetch",
    "format_value",
    "report",
    "stale",
]


URL: Final = "https://models.dev/api.json"

# Keyed by models.dev provider. Each catalog prices the international
# deployment, never ``-cn``, whose rates are converted from CNY.
CATALOGS: Final[Mapping[str, Callable[[], ModelTable]]] = MappingProxyType(
    {
        "anthropic": anthropic.models,
        "openai": openai.models,
        "google": google.models,
        "alibaba": dashscope.models,
        "moonshotai": moonshot.models,
        "minimax": minimax.models,
    },
)


@dataclass(frozen=True, slots=True, kw_only=True)
class Drift:
    """One value Sagent and models.dev state differently."""

    provider: str
    """models.dev provider key."""
    model_id: str
    field: str
    """models.dev's name for the value (``limit.output``, ``cost.cache_read``)."""
    ours: float
    theirs: float
    source: Source | None
    """Where Sagent read the value."""
    accepted: str = ""
    """Why it is known and fine; empty while it needs review."""


_WINDOW: Final = (
    "models.dev subtracts max output from the window; the vendor page states "
    "no separate input cap (gpt-5.5.md read 2026-10-18)"
)
_IMPLICIT_WRITE: Final = (
    "models.dev lists the explicit-cache write; this transport never writes "
    "the explicit marker, and implicit cache creation bills as input"
)
_IMPLICIT_HIT: Final = (
    "models.dev lists the explicit-cache hit (10% of input); implicit hits bill 20%"
)
_EXPLICIT_ONLY: Final = (
    "explicit cache only in Singapore, and this transport never writes the "
    "marker, so no hit is discounted"
)
_K2_OUTPUT: Final = (
    "the vendor states only K2's 32,768 default output; models.dev lists the window"
)
_MINIMAX_OUTPUT: Final = (
    "text-chat-openai caps max_completion_tokens at 524,288 for M3 and "
    "204,800 for the rest (read 2026-10-18)"
)
_QWEN_INPUT: Final = (
    "models.dev lists the 1M context as the input cap; the model page states "
    "991,808 (read 2026-10-20)"
)
_DELISTED: Final = (
    "the vendor's pages no longer list this model (read 2026-10-20); Sagent "
    "keeps the 2026-10-03 read"
)

ACCEPTED: Final[Mapping[tuple[str, str], tuple[float, str]]] = MappingProxyType(
    {
        ("gpt-5.5", "limit.input"): (922_000, _WINDOW),
        ("gpt-5.5-pro", "limit.input"): (922_000, _WINDOW),
        ("gpt-5.4", "limit.input"): (922_000, _WINDOW),
        ("gpt-5.4-pro", "limit.input"): (922_000, _WINDOW),
        ("gpt-5.2", "limit.input"): (272_000, _WINDOW),
        ("gpt-5.2-pro", "limit.input"): (272_000, _WINDOW),
        ("gpt-5.1", "limit.input"): (272_000, _WINDOW),
        ("qwen3.8-max", "limit.input"): (1_000_000, _QWEN_INPUT),
        ("qwen3.8-flash", "limit.input"): (1_000_000, _QWEN_INPUT),
        ("qwen3.7-max", "limit.input"): (1_000_000, _QWEN_INPUT),
        ("qwen3.7-plus", "limit.input"): (1_000_000, _QWEN_INPUT),
        ("qwen3.6-plus", "limit.input"): (1_000_000, _QWEN_INPUT),
        ("qwen3.6-flash", "limit.input"): (1_000_000, _QWEN_INPUT),
        ("qwen3.8-max", "cost.cache_write"): (2.5, _IMPLICIT_WRITE),
        ("qwen3.8-flash", "cost.cache_write"): (0.2, _IMPLICIT_WRITE),
        ("qwen3.7-max", "cost.cache_write"): (3.125, _IMPLICIT_WRITE),
        ("qwen3.7-plus", "cost.cache_write"): (0.5, _IMPLICIT_WRITE),
        ("qwen3.7-flash", "cost.cache_write"): (0.0375, _IMPLICIT_WRITE),
        ("qwen3.6-max-preview", "cost.cache_write"): (1.625, _IMPLICIT_WRITE),
        ("qwen3.6-plus", "cost.cache_write"): (0.625, _IMPLICIT_WRITE),
        ("qwen3.6-flash", "cost.cache_write"): (0.3125, _IMPLICIT_WRITE),
        ("qwen3.7-plus", "cost.cache_read"): (0.04, _IMPLICIT_HIT),
        ("qwen3.7-flash", "cost.cache_read"): (0.003, _IMPLICIT_HIT),
        ("qwen3.6-max-preview", "cost.cache_read"): (0.13, _EXPLICIT_ONLY),
        ("qwen3.6-plus", "cost.cache_read"): (0.05, _EXPLICIT_ONLY),
        ("qwen3.6-flash", "cost.cache_read"): (0.025, _EXPLICIT_ONLY),
        ("qwen3-32b", "limit.input"): (131_072, _DELISTED),
        ("qwen3-32b", "limit.output"): (16_384, _DELISTED),
        ("qwen3-32b", "cost.input"): (0.7, _DELISTED),
        ("qwen3-32b", "cost.output"): (2.8, _DELISTED),
        ("qwen-turbo", "limit.output"): (16_384, _DELISTED),
        ("kimi-k2.6", "limit.output"): (262_144, _K2_OUTPUT),
        ("kimi-k2.7-code", "limit.output"): (262_144, _K2_OUTPUT),
        ("kimi-k2.7-code-highspeed", "limit.output"): (262_144, _K2_OUTPUT),
        ("MiniMax-M2.7", "limit.output"): (131_072, _MINIMAX_OUTPUT),
        ("MiniMax-M2.7-highspeed", "limit.output"): (131_072, _MINIMAX_OUTPUT),
        ("MiniMax-M2.5", "limit.output"): (131_072, _MINIMAX_OUTPUT),
        ("MiniMax-M2.5-highspeed", "limit.output"): (131_072, _MINIMAX_OUTPUT),
        ("MiniMax-M2.1", "limit.output"): (131_072, _MINIMAX_OUTPUT),
        ("MiniMax-M2", "limit.output"): (131_072, _MINIMAX_OUTPUT),
    },
)
"""``(model id, field)`` to the models.dev value reviewed and why it is fine."""


async def fetch(*, warn: Warn) -> dict[str, dict[str, object]] | None:
    """Download every provider's model rows.

    Args:
      warn: Receives the failure when the download or parse fails.

    Returns:
      models: models.dev provider key to its rows by model id; ``None`` on
        failure.

    """
    async with httpx2.AsyncClient(timeout=60, follow_redirects=True) as client:
        try:
            r = await client.get(URL)
            r.raise_for_status()
            body = r.json()
        except httpx2.HTTPError as e:
            warn(f"models.dev: {describe(e)}")
            return None
        except ValueError:
            warn("models.dev: response is not JSON")
            return None
    body = from_plain(body, dict[str, object], default={})
    out: dict[str, dict[str, object]] = {}
    for key, raw in body.items():
        provider = from_plain(raw, dict[str, object], default={})
        out[key] = from_plain(provider.get("models"), dict[str, object], default={})
    return out


def compare(
    provider: str,
    table: ModelTable,
    upstream: Mapping[str, object],
) -> list[Drift]:
    """List every value of ``table`` that ``upstream`` states differently.

    Only rows both sides list are compared: models.dev omits invitation-only
    models and keeps retired ones, so a missing row says nothing.

    Args:
      provider: models.dev provider key, for the report.
      table: The catalog to check.
      upstream: models.dev rows for ``provider``, by model id.

    Returns:
      drifts: Each disagreement, triaged or not, in table order.

    """
    out: list[Drift] = []
    for mid in table:
        raw = upstream.get(mid)
        if raw is None:
            continue
        row = from_plain(raw, dict[str, object], default={})
        limit = from_plain(row.get("limit"), dict[str, object], default={})
        cost = from_plain(row.get("cost"), dict[str, object], default={})
        cap = table[mid]
        window = cap.context[""]
        differ = partial(_differ, out, provider, mid, table)
        differ(
            "limits",
            [
                (
                    "limit.input",
                    window.max_request_tokens,
                    limit.get("input", limit.get("context")),
                ),
                ("limit.output", window.max_response_tokens, limit.get("output")),
            ],
        )
        card = cap.prices.get(PriceKey("auto"))
        if card is not None:
            differ(
                "prices",
                [
                    ("cost.input", card.request, cost.get("input")),
                    ("cost.output", card.response, cost.get("output")),
                    ("cost.cache_read", card.cache_read, cost.get("cache_read")),
                    ("cost.cache_write", card.cache_write, cost.get("cache_write")),
                ],
            )
    return out


def _differ(  # noqa: PLR0917 -- partial-applied per row.
    out: list[Drift],
    provider: str,
    mid: str,
    table: ModelTable,
    fact: Fact,
    pairs: list[tuple[str, float, object]],
) -> None:
    """Append a :class:`Drift` for each ``(field, ours, theirs)`` that differs."""
    for name, ours, raw_theirs in pairs:
        theirs = from_plain(raw_theirs, float, default=math.nan)
        if math.isnan(theirs) or math.isclose(ours, theirs, abs_tol=1e-12):
            continue
        pinned, reason = ACCEPTED.get((mid, name), (math.nan, ""))
        out.append(
            Drift(
                provider=provider,
                model_id=mid,
                field=name,
                ours=ours,
                theirs=theirs,
                source=table.source(mid, fact),
                accepted=reason if math.isclose(pinned, theirs) else "",
            ),
        )


def stale(drifts: list[Drift]) -> list[tuple[str, str]]:
    """Return the :data:`ACCEPTED` entries no current disagreement matched.

    Either models.dev now agrees, or it changed its value: both need a person.
    """
    triaged = {(d.model_id, d.field) for d in drifts if d.accepted}
    return sorted(ACCEPTED.keys() - triaged)


def report(drifts: list[Drift], *, at: date) -> str:
    """Render the disagreements needing review as a Markdown issue body.

    Args:
      drifts: Every disagreement :func:`compare` found.
      at: When models.dev was read.

    Returns:
      text: The body; empty when nothing needs review.

    """
    open_ = [d for d in drifts if not d.accepted]
    triaged = sum(1 for d in drifts if d.accepted)
    leftover = stale(drifts)
    if not open_ and not leftover:
        return ""
    lines = [
        "# Model catalog drift",
        "",
        (
            f"Compared against {URL} on {at.isoformat()}. models.dev is not a "
            "source of truth: reopen Sagent's source, then fix the catalog or "
            "add the value to `ACCEPTED` in "
            "`sagent/catalog/sources/models_dev.py` with the reason."
        ),
        "",
        f"{len(open_)} disagreement(s) need review; {triaged} are already triaged.",
    ]
    if open_:
        lines += [
            "",
            (
                "| Provider | Model | Field | Sagent | models.dev | "
                "Sagent's source | Checked |"
            ),
            "| --- | --- | --- | ---: | ---: | --- | --- |",
        ]
        for d in open_:
            url = d.source.url if d.source else "(none)"
            checked = (
                d.source.checked.isoformat() if d.source and d.source.checked else "?"
            )
            cells = (
                d.provider,
                f"`{d.model_id}`",
                d.field,
                format_value(d.ours),
                format_value(d.theirs),
                url,
                checked,
            )
            lines.append(f"| {' | '.join(cells)} |")
    if leftover:
        lines += [
            "",
            "## Triage that no longer matches",
            "",
            (
                "These `ACCEPTED` entries match no current disagreement: "
                "models.dev now agrees or shows another value. Recheck, then "
                "remove or re-pin:"
            ),
            "",
        ]
        lines += [f"- `{mid}` {field}" for mid, field in leftover]
    return "\n".join(lines) + "\n"


def format_value(value: float) -> str:
    """Show a token count with separators and a price as written."""
    if float(value).is_integer() and value >= 1000:
        return f"{int(value):,}"
    return f"{value:g}"
