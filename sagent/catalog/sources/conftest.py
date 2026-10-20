# Pytest fixtures are referenced by parameter name, not imported.
"""Fixtures for the source readers: every HTTP request answered in-process."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import httpx2
import pytest


type Handler = Callable[[httpx2.Request], httpx2.Response]


@pytest.fixture
def serve(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Handler], list[httpx2.Request]]:
    """Route every ``httpx2.AsyncClient`` through a handler; return the requests."""

    def install(handler: Handler) -> list[httpx2.Request]:
        seen: list[httpx2.Request] = []

        def record(request: httpx2.Request) -> httpx2.Response:
            seen.append(request)
            return handler(request)

        monkeypatch.setattr(
            httpx2,
            "AsyncClient",
            partial(httpx2.AsyncClient, transport=httpx2.MockTransport(record)),
        )
        return seen

    return install
