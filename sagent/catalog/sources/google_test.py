"""Tests for ``catalog.sources.google``: the Gemini model list via stubbed HTTP."""

from __future__ import annotations

from collections.abc import Callable

import httpx2
import pytest

from sagent.catalog.sources import google
from sagent.catalog.sources.live import LiveLimits


type Serve = Callable[
    [Callable[[httpx2.Request], httpx2.Response]],
    list[httpx2.Request],
]


def _limits(request: int, response: int) -> LiveLimits:
    return LiveLimits(max_request_tokens=request, max_response_tokens=response)


@pytest.mark.anyio
async def test_fetches_model_limits(serve: Serve) -> None:
    body = {
        "models": [
            {
                "name": "models/gemini-test",
                "inputTokenLimit": 10,
                "outputTokenLimit": 20,
            },
            {"name": "models/incomplete", "inputTokenLimit": 1},
        ],
    }
    serve(lambda _: httpx2.Response(200, json=body))
    assert await google.fetch("key", warn=pytest.fail) == {
        "gemini-test": _limits(10, 20),
    }


@pytest.mark.anyio
async def test_follows_next_page_token(serve: Serve) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.params.get("pageToken") == "p2":
            model = {"name": "models/b", "inputTokenLimit": 3, "outputTokenLimit": 4}
            return httpx2.Response(200, json={"models": [model]})
        model = {"name": "models/a", "inputTokenLimit": 1, "outputTokenLimit": 2}
        return httpx2.Response(200, json={"models": [model], "nextPageToken": "p2"})

    serve(handler)
    assert await google.fetch("key", warn=pytest.fail) == {
        "a": _limits(1, 2),
        "b": _limits(3, 4),
    }


@pytest.mark.anyio
async def test_key_rides_a_header_never_the_url(serve: Serve) -> None:
    seen = serve(lambda _: httpx2.Response(403))
    warnings: list[str] = []
    assert await google.fetch("s3cr3t-key", warn=warnings.append) == {}
    assert seen
    assert all("s3cr3t-key" not in str(request.url) for request in seen)
    assert seen[0].headers["x-goog-api-key"] == "s3cr3t-key"
    assert warnings == ["models list: HTTP 403"]


@pytest.mark.anyio
async def test_network_error_warns_instead_of_raising(serve: Serve) -> None:
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError(f"refused {request.url.host}")

    serve(refuse)
    warnings: list[str] = []
    assert await google.fetch("key", warn=warnings.append) == {}
    assert warnings == ["models list: ConnectError"]


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
