"""Tests for ``catalog.sources.models_dev``: drift detection and triage."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date
from types import MappingProxyType

import httpx2
import pytest

from sagent.catalog.provenance import Source
from sagent.catalog.sources import models_dev
from sagent.catalog.table import ModelTable
from sagent.types.capability import ModelCapability, ModelLimits
from sagent.types.cost import PriceCatalog, PriceKey, TokenPrice


type Serve = Callable[
    [Callable[[httpx2.Request], httpx2.Response]],
    list[httpx2.Request],
]

_LIMITS = Source("https://example.com/models", date(2026, 10, 3))


def _table(model_id: str = "m", *, output: int = 64_000) -> ModelTable:
    row = ModelCapability(
        model_id=model_id,
        context=MappingProxyType(
            {"": ModelLimits(max_request_tokens=200_000, max_response_tokens=output)},
        ),
        prices=PriceCatalog(
            {
                PriceKey("auto"): TokenPrice(
                    request=1.0,
                    response=5.0,
                    cache_write=1.25,
                    cache_write_1h=2.0,
                    cache_read=0.1,
                ),
            },
        ),
    )
    return ModelTable(rows=(row,), sources={"limits": _LIMITS})


def _upstream(**limit: int) -> dict[str, object]:
    return {
        "m": {
            "limit": {"context": 200_000, "output": 64_000} | limit,
            "cost": {"input": 1, "output": 5, "cache_read": 0.1, "cache_write": 1.25},
        },
    }


def test_agreeing_rows_report_nothing() -> None:
    assert models_dev.compare("p", _table(), _upstream()) == []


def test_a_disagreement_names_both_values_and_our_source() -> None:
    (drift,) = models_dev.compare("p", _table(), _upstream(output=32_000))
    assert drift == models_dev.Drift(
        provider="p",
        model_id="m",
        field="limit.output",
        ours=64_000,
        theirs=32_000,
        source=_LIMITS,
    )


def test_input_is_compared_to_the_window_when_upstream_states_no_input() -> None:
    (drift,) = models_dev.compare("p", _table(), _upstream(context=100_000))
    assert drift.field == "limit.input"


def test_rows_upstream_lacks_and_fields_it_omits_are_skipped() -> None:
    assert models_dev.compare("p", _table("x"), _upstream()) == []
    assert models_dev.compare("p", _table(), {"m": {"limit": {}}}) == []


def test_triage_holds_only_for_the_value_it_was_pinned_to() -> None:
    (mid, field), (pinned, reason) = next(iter(models_dev.ACCEPTED.items()))
    assert field == "limit.input"
    table = _table(mid)
    upstream = {mid: {"limit": {"input": pinned}}}
    (drift,) = models_dev.compare("p", table, upstream)
    assert drift.accepted == reason
    (moved,) = models_dev.compare("p", table, {mid: {"limit": {"input": pinned - 1}}})
    assert moved.accepted == ""


def test_every_triaged_entry_names_a_catalog_row() -> None:
    known = {mid for build in models_dev.CATALOGS.values() for mid in build()}
    assert {mid for mid, _ in models_dev.ACCEPTED} <= known


def test_the_report_lists_open_drift_and_stale_triage() -> None:
    drifts = models_dev.compare("p", _table(), _upstream(output=32_000))
    body = models_dev.report(drifts, at=date(2026, 10, 18))
    assert body.startswith("# Model catalog drift\n")
    assert "on 2026-10-18" in body
    assert (
        "| p | `m` | limit.output | 64,000 | 32,000 | https://example.com/models "
        "| 2026-10-03 |"
    ) in body
    # No drift matched any triaged entry, so every one is listed as stale.
    assert len(models_dev.stale(drifts)) == len(models_dev.ACCEPTED)
    assert "## Triage that no longer matches" in body


def test_the_report_is_empty_when_nothing_needs_review() -> None:
    drifts = [
        models_dev.Drift(
            provider="p",
            model_id=mid,
            field=field,
            ours=0,
            theirs=pinned,
            source=None,
            accepted=reason,
        )
        for (mid, field), (pinned, reason) in models_dev.ACCEPTED.items()
    ]
    assert models_dev.report(drifts, at=date(2026, 10, 18)) == ""


@pytest.mark.anyio
async def test_fetch_returns_each_providers_rows(serve: Serve) -> None:
    body = {"openai": {"id": "openai", "models": {"gpt-x": {"id": "gpt-x"}}}}
    seen = serve(lambda _: httpx2.Response(200, json=body))
    assert await models_dev.fetch(warn=pytest.fail) == {
        "openai": {"gpt-x": {"id": "gpt-x"}},
    }
    assert [str(r.url) for r in seen] == [models_dev.URL]


@pytest.mark.anyio
async def test_fetch_failure_warns_and_returns_none(serve: Serve) -> None:
    serve(lambda _: httpx2.Response(502))
    warnings: list[str] = []
    assert await models_dev.fetch(warn=warnings.append) is None
    assert warnings == ["models.dev: HTTP 502"]


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
