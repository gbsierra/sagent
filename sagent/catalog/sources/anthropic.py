"""What the Anthropic Models API reports for each model.

Source: GET /v1/models lists the account's active and deprecated models,
paged by ``after_id`` = the previous page's ``last_id`` while ``has_more``
(``limit`` 1..1000). Each ModelInfo carries ``max_input_tokens``,
``max_tokens``, ``lifecycle``, ``retires_at``, and ``capabilities.effort``.
GET /v1/models/{model_id} answers for one id, including an alias.
https://platform.claude.com/docs/en/api/models/list (read 2026-10-17).
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import httpx2

from sagent.catalog.sources.live import LiveLimits, describe
from sagent.lib.codec import from_plain


if TYPE_CHECKING:
    from collections.abc import Mapping

    from sagent.catalog.sources.live import Warn


__all__ = ["fetch"]


async def fetch(
    api_key: str,
    model_ids: list[str],
    *,
    warn: Warn,
) -> dict[str, LiveLimits]:
    """Fetch what the API reports for every model it lists.

    A catalog id the list omits (a dateless alias of a dated snapshot) is
    then asked for by id, and recorded with the id the API answered as.

    Args:
      api_key: Anthropic API key.
      model_ids: Catalog ids to look up when the list omits them.
      warn: Receives each failure the fetch carried on past.

    Returns:
      limits: Map of model id to what the API reported; partial on failure.

    """
    out: dict[str, LiveLimits] = {}
    api = "https://api.anthropic.com/v1/models"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    async with httpx2.AsyncClient(timeout=30) as client:
        after_id = ""
        while True:
            params = {"limit": "1000"} | ({"after_id": after_id} if after_id else {})
            try:
                r = await client.get(api, headers=headers, params=params)
                r.raise_for_status()
            except httpx2.HTTPError as e:
                warn(f"models list: {describe(e)}")
                break
            body = from_plain(r.json(), dict[str, object])
            for raw in from_plain(body.get("data"), list[object], default=[]):
                info = from_plain(raw, dict[str, object])
                mid = from_plain(info.get("id"), str, default="")
                live = _model_info(mid, info, warn)
                if live is not None:
                    out[mid] = live
            after_id = from_plain(body.get("last_id"), str, default="")
            if (
                not from_plain(body.get("has_more"), bool, default=False)
                or not after_id
            ):
                break
        for mid in model_ids:
            if mid in out:
                continue
            try:
                r = await client.get(f"{api}/{mid}", headers=headers)
                r.raise_for_status()
            except httpx2.HTTPStatusError as e:
                if e.response.status_code == 404:
                    warn(f"{mid}: not found in API")
                else:
                    warn(f"{mid}: {describe(e)}")
                continue
            except httpx2.HTTPError as e:
                warn(f"{mid}: {describe(e)}")
                continue
            info = from_plain(r.json(), dict[str, object])
            live = _model_info(mid, info, warn)
            if live is not None:
                answered = from_plain(info.get("id"), str, default=mid)
                if answered != mid:
                    live = replace(live, served_as=answered)
                out[mid] = live
    return out


def _model_info(mid: str, info: Mapping[str, object], warn: Warn) -> LiveLimits | None:
    """Read one ModelInfo; warn and return ``None`` when it states no limits."""
    max_input = from_plain(info.get("max_input_tokens"), int, default=0)
    max_output = from_plain(info.get("max_tokens"), int, default=0)
    if not (max_input and max_output):
        warn(f"{mid}: missing limits in API response")
        return None
    return LiveLimits(
        max_request_tokens=max_input,
        max_response_tokens=max_output,
        effort=_effort(info.get("capabilities")),
        lifecycle=from_plain(info.get("lifecycle"), str, default=""),
        retires_at=from_plain(info.get("retires_at"), str, default=""),
    )


def _effort(capabilities: object) -> frozenset[str] | None:
    """Return the levels ``capabilities.effort`` marks supported, if it is stated.

    The object is ``{"supported": bool, "<level>": {"supported": bool}, ...}``;
    a model without effort reports ``supported: false`` and no level.
    """
    caps = from_plain(capabilities, dict[str, object], default={})
    effort = from_plain(caps.get("effort"), dict[str, object], default={})
    if not effort:
        return None
    return frozenset(level for level, entry in effort.items() if _supported(entry))


def _supported(entry: object) -> bool:
    """Whether a capability entry is ``{"supported": true, ...}``."""
    match entry:
        case {"supported": True}:
            return True
        case _:
            return False
