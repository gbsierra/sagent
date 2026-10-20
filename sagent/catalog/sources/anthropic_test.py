"""Tests for ``catalog.sources.anthropic``: the Models API via stubbed HTTP."""

from __future__ import annotations

from collections.abc import Callable

import httpx2
import pytest

from sagent.catalog.sources import anthropic
from sagent.catalog.sources.live import LiveLimits


type Serve = Callable[
    [Callable[[httpx2.Request], httpx2.Response]],
    list[httpx2.Request],
]


def _limits(request: int, response: int) -> LiveLimits:
    return LiveLimits(max_request_tokens=request, max_response_tokens=response)


def _model_info(model_id: str, **extra: object) -> dict[str, object]:
    """One ModelInfo carrying the limits ``fetch`` requires."""
    return {"id": model_id, "max_input_tokens": 100, "max_tokens": 10, **extra}


def _model_list(*infos: dict[str, object], has_more: bool = False) -> httpx2.Response:
    last = str(infos[-1]["id"]) if infos else None
    return httpx2.Response(
        200,
        json={"data": list(infos), "has_more": has_more, "last_id": last},
    )


@pytest.mark.anyio
async def test_lists_every_model_across_pages(serve: Serve) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.params.get("after_id") == "a":
            return _model_list(_model_info("b"))
        return _model_list(_model_info("a"), has_more=True)

    seen = serve(handler)
    assert await anthropic.fetch("key", [], warn=pytest.fail) == {
        "a": _limits(100, 10),
        "b": _limits(100, 10),
    }
    assert all(request.url.params["limit"] == "1000" for request in seen)


@pytest.mark.anyio
async def test_asks_by_id_for_an_alias_the_list_omits(serve: Serve) -> None:
    """The list names the dated snapshot; ``/models/{alias}`` says which."""
    snapshot = "claude-haiku-4-5-20251001"

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/v1/models":
            return _model_list(_model_info(snapshot))
        return httpx2.Response(200, json=_model_info(snapshot))

    seen = serve(handler)
    live = await anthropic.fetch(
        "key",
        [snapshot, "claude-haiku-4-5"],
        warn=pytest.fail,
    )
    assert live["claude-haiku-4-5"].served_as == snapshot
    assert live[snapshot].served_as == ""
    assert [r.url.path for r in seen] == [
        "/v1/models",
        "/v1/models/claude-haiku-4-5",
    ]


@pytest.mark.anyio
async def test_reads_lifecycle_and_the_effort_levels_marked_supported(
    serve: Serve,
) -> None:
    effort = {
        "supported": True,
        "low": {"supported": True},
        "high": {"supported": True},
        "max": {"supported": False},
    }
    info = _model_info(
        "old",
        lifecycle="deprecated",
        retires_at="2026-11-30T00:00:00Z",
        capabilities={"effort": effort},
    )
    serve(lambda _: _model_list(info, _model_info("plain")))
    live = await anthropic.fetch("key", [], warn=pytest.fail)
    assert live["old"].effort == frozenset({"low", "high"})
    assert live["old"].lifecycle == "deprecated"
    assert live["old"].retires_at == "2026-11-30T00:00:00Z"
    assert live["plain"].effort is None
    assert live["plain"].lifecycle == ""


@pytest.mark.anyio
async def test_a_model_without_effort_supports_no_level(serve: Serve) -> None:
    info = _model_info("m", capabilities={"effort": {"supported": False}})
    serve(lambda _: _model_list(info))
    live = await anthropic.fetch("key", [], warn=pytest.fail)
    assert live["m"].effort == frozenset()


@pytest.mark.anyio
async def test_fetches_and_warns(serve: Serve) -> None:
    bodies: dict[str, httpx2.Response] = {
        "ok": httpx2.Response(200, json=_model_info("ok")),
        "bad": httpx2.Response(200, json={"max_input_tokens": 0, "max_tokens": 10}),
        "err": httpx2.Response(500),
        "missing": httpx2.Response(404),
    }

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/v1/models":
            return httpx2.Response(503)
        if request.url.path.endswith("/down"):
            raise httpx2.ConnectError("refused")
        return bodies[request.url.path.rsplit("/", 1)[1]]

    serve(handler)
    warnings: list[str] = []
    limits = await anthropic.fetch(
        "key",
        ["ok", "bad", "err", "missing", "down"],
        warn=warnings.append,
    )
    assert limits == {"ok": _limits(100, 10)}
    assert warnings == [
        "models list: HTTP 503",
        "bad: missing limits in API response",
        "err: HTTP 500",
        "missing: not found in API",
        "down: ConnectError",
    ]


@pytest.mark.anyio
async def test_is_queried_by_vendor_id(serve: Serve) -> None:
    seen = serve(lambda _: httpx2.Response(404))
    await anthropic.fetch("key", ["claude-opus-5-5"], warn=lambda _: None)
    assert "/v1/models/claude-opus-5-5" in {request.url.path for request in seen}


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
