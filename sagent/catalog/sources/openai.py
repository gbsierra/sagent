"""OpenAI model limits, scraped from the per-model documentation pages.

Source: https://developers.openai.com/api/docs/models/<model>.md. The
markdown rendering states "N context window", "Maximum input tokens: N"
where the model has one, and "N max output tokens". The HTML page omits the
input cap, and the catalog stores that cap, not the window.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import re

import httpx2

from sagent.catalog.sources.live import LiveLimits, describe


if TYPE_CHECKING:
    from sagent.catalog.sources.live import Warn


__all__ = ["fetch", "parse_page"]


async def fetch(model_ids: list[str], *, warn: Warn) -> dict[str, LiveLimits]:
    """Scrape each model's limits from its documentation page.

    Args:
      model_ids: Model ids to look up.
      warn: Receives each page that failed or did not parse.

    Returns:
      limits: Map of model id to its token limits.

    """
    out: dict[str, LiveLimits] = {}
    doc = "https://developers.openai.com/api/docs/models"
    async with httpx2.AsyncClient(timeout=30, follow_redirects=True) as client:
        for mid in model_ids:
            url = f"{doc}/{mid}.md"
            try:
                r = await client.get(url)
                r.raise_for_status()
            except httpx2.HTTPError as e:
                warn(f"{mid}: {describe(e)}")
                continue
            limits = parse_page(r.text)
            if limits:
                out[mid] = limits
            else:
                warn(f"{mid}: could not parse limits from {url}")
    return out


def parse_page(page: str) -> LiveLimits | None:
    """Extract the input cap and max output tokens from one model page.

    The input cap is "Maximum input tokens" where the page states one, else
    the context window.
    """
    cleaned = re.sub(r"<!--.*?-->", " ", page, flags=re.DOTALL)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    request = re.search(r"Maximum\s+input\s+tokens:?\s+([\d,]+)", cleaned) or (
        re.search(r"([\d,]+)\s+context\s+window", cleaned)
    )
    out = re.search(r"([\d,]+)\s+max\s+output\s+tokens", cleaned)
    if request and out:
        return LiveLimits(
            max_request_tokens=_num(request.group(1)),
            max_response_tokens=_num(out.group(1)),
        )
    return None


def _num(s: str) -> int:
    """Parse a comma-grouped integer literal."""
    return int(s.replace(",", ""))
