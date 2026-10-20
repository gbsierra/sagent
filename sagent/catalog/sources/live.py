"""What a vendor's own API or docs report for one model, read live."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import httpx2


__all__ = ["LiveLimits", "Warn", "describe"]


type Warn = Callable[[str], None]
"""Receives one problem a fetch met and carried on past."""


@dataclass(frozen=True, slots=True, kw_only=True)
class LiveLimits:
    """What a live source reported for one model."""

    max_request_tokens: int
    max_response_tokens: int
    effort: frozenset[str] | None = None
    """Effort levels the vendor says the model accepts; ``None`` when unreported."""
    lifecycle: str = ""
    """Vendor lifecycle stage (``active``, ``deprecated``); empty when unreported."""
    retires_at: str = ""
    """Scheduled retirement, as the vendor wrote it; empty when unscheduled."""
    served_as: str = ""
    """The id the vendor answered with, when it differs from the one asked."""


def describe(e: httpx2.HTTPError) -> str:
    """Name a request failure without echoing its URL, which may carry a key."""
    if isinstance(e, httpx2.HTTPStatusError):
        return f"HTTP {e.response.status_code}"
    return type(e).__name__
