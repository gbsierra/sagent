"""Tests for ``catalog.sources.openai``: model-page scraping via stubbed HTTP."""

from __future__ import annotations

from collections.abc import Callable

import httpx2
import pytest

from sagent.catalog.sources import openai
from sagent.catalog.sources.live import LiveLimits


type Serve = Callable[
    [Callable[[httpx2.Request], httpx2.Response]],
    list[httpx2.Request],
]


def _limits(request: int, response: int) -> LiveLimits:
    return LiveLimits(max_request_tokens=request, max_response_tokens=response)


def test_parses_limits_with_markup_and_commas() -> None:
    html = """
    <!-- hidden
    <b>999,999 context window</b> -->
    <div>128,000 context window</div>
    <span>16,384 max output tokens</span>
    """
    assert openai.parse_page(html) == _limits(128_000, 16_384)


def test_prefers_maximum_input_tokens_over_context_window() -> None:
    page = (
        "- 1,050,000 context window\n"
        "- Maximum input tokens: 922,000\n"
        "- 128,000 max output tokens\n"
    )
    assert openai.parse_page(page) == _limits(922_000, 128_000)


def test_returns_none_without_both_limits() -> None:
    assert openai.parse_page("128,000 context window") is None


@pytest.mark.anyio
async def test_fetches_parseable_pages(serve: Serve) -> None:
    pages = {
        "ok": httpx2.Response(200, text="1,000 context window 200 max output tokens"),
        "bad": httpx2.Response(200, text="not parseable"),
        "missing": httpx2.Response(404),
    }

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path.endswith("/down.md"):
            raise httpx2.ConnectError("refused")
        return pages[request.url.path.rsplit("/", 1)[1].removesuffix(".md")]

    seen = serve(handler)
    warnings: list[str] = []
    limits = await openai.fetch(["ok", "bad", "missing", "down"], warn=warnings.append)
    assert limits == {"ok": _limits(1000, 200)}
    # The markdown rendering states "Maximum input tokens"; the HTML omits it.
    assert all(request.url.path.endswith(".md") for request in seen)
    assert warnings[0].startswith("bad: could not parse limits from ")
    assert warnings[1:] == ["missing: HTTP 404", "down: ConnectError"]


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
