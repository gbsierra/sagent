"""Gemini model limits, from the Generative Language API's model list.

Source: https://ai.google.dev/api/models#method:-models.list, which
returns ``inputTokenLimit`` and ``outputTokenLimit`` per model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx2

from sagent.catalog.sources.live import LiveLimits, describe
from sagent.lib.codec import from_plain


if TYPE_CHECKING:
    from sagent.catalog.sources.live import Warn


__all__ = ["fetch"]


async def fetch(api_key: str, *, warn: Warn) -> dict[str, LiveLimits]:
    """Fetch every listed model's limits.

    Args:
      api_key: Google API key.
      warn: Receives each failure the fetch carried on past.

    Returns:
      limits: Map of model id to its token limits; partial when a page fails.

    """
    api = "https://generativelanguage.googleapis.com/v1beta/models"
    out: dict[str, LiveLimits] = {}
    page_token = ""
    # The key rides a header: in the query string it would be echoed by every
    # HTTPStatusError message.
    headers = {"x-goog-api-key": api_key}
    async with httpx2.AsyncClient(timeout=30) as client:
        while True:
            params = {"pageToken": page_token} if page_token else {}
            try:
                r = await client.get(api, headers=headers, params=params)
                r.raise_for_status()
            except httpx2.HTTPError as e:
                warn(f"models list: {describe(e)}")
                return out
            body = from_plain(r.json(), dict[str, object])
            for raw_model in from_plain(body.get("models"), list[object], default=[]):
                model = from_plain(raw_model, dict[str, object])
                short = from_plain(model.get("name"), str, default="").removeprefix(
                    "models/",
                )
                inp = from_plain(model.get("inputTokenLimit"), int, default=0)
                outp = from_plain(model.get("outputTokenLimit"), int, default=0)
                if inp and outp:
                    out[short] = LiveLimits(
                        max_request_tokens=inp,
                        max_response_tokens=outp,
                    )
            page_token = from_plain(body.get("nextPageToken"), str, default="")
            if not page_token:
                return out
