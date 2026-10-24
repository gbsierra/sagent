#!/bin/sh
# ruff: noqa: EXE003, D300, D205 -- Polyglot shell/Python script.
# fmt: off
'''' 2>/dev/null #
exec uv --quiet --project "$(dirname "$0")" run --frozen --no-sync python3 "$0" "$@"
Verify provider model limits.

Checks that every provider's catalog entries have correct
max_request_tokens and max_response_tokens by querying live APIs or
scraping official documentation pages. Where the vendor reports them,
also checks effort levels and warns about deprecated models. A model the
vendor lists but the catalog lacks is a problem.

Usage::

    uv --quiet --project . run python -m sagent.bin.verify_models
    uv --quiet --project . run python -m sagent.bin.verify_models --provider google
    uv --quiet --project . run python -m sagent.bin.verify_models --provider openai
    uv --quiet --project . run python -m sagent.bin.verify_models --snapshot live.json
    uv --quiet --project . run python -m sagent.bin.verify_models --drift --report drift.md

Every row is looked up and reported by its vendor id (``claude-opus-5-5``),
never by a Sagent role or retired name (``default``, ``opus-5.5``). A run that
verifies no row of a requested provider fails: a missing key or an
unreachable API is not a pass.

``--drift`` instead compares every hosted catalog with models.dev, needs no
key, and fails only on a disagreement not yet triaged; ``--report`` writes
those as a Markdown issue body.

Sources:
- Readers: sagent/catalog/sources
- Google: GET /v1beta/models (returns inputTokenLimit, outputTokenLimit)
- OpenAI: scrapes https://developers.openai.com/api/docs/models/<id>.md
- Anthropic: GET /v1/models, then /v1/models/<id> for ids it omits
  (max_tokens, max_input_tokens, lifecycle, retires_at, capabilities.effort)
- Drift: https://models.dev/api.json

Cross-reference: https://github.com/taylorwilsdon/llm-context-limits
'''
# fmt: on

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import argparse
import asyncio
import json
import os
import pathlib
import sys

from sagent import providers
from sagent.catalog.sources import (
    anthropic as anthropic_source,
    google as google_source,
    models_dev,
    openai as openai_source,
)
from sagent.providers.anthropic.api import Anthropic
from sagent.providers.google.api import Google
from sagent.providers.openai.api import OpenAI
from sagent.types.providers import ModelResolver


if TYPE_CHECKING:
    from collections.abc import Mapping

    from sagent.catalog.sources.live import LiveLimits
    from sagent.catalog.table import ModelCatalog
    from sagent.types.capability import ModelCapability


def compare(
    provider_name: str,
    known: Mapping[str, ModelCapability],
    live: dict[str, LiveLimits],
) -> int:
    """Compare catalog entries against live API limits.

    A catalog row the live source did not report is listed as unverified;
    if no row at all was verified, that is itself an error.

    Args:
      provider_name: Display name for log output.
      known: Resolved model catalog, keyed by vendor wire id.
      live: LiveLimits fetched from the live API, keyed by vendor wire id.

    Returns:
      error_count: Number of problems found.

    """
    errors = 0
    verified = 0
    all_ids = sorted(set(known) | set(live))
    for mid in all_ids:
        k = known.get(mid)
        lv = live.get(mid)
        if lv is not None:
            if lv.served_as:
                _out(f"  {provider_name}.{mid}: the API answers as {lv.served_as}")
            if lv.lifecycle not in ("", "active"):
                when = f", retires {lv.retires_at}" if lv.retires_at else ""
                _warn(f"{provider_name}.{mid}: lifecycle {lv.lifecycle}{when}")
        if k is None:
            _out(f"  {provider_name}.{mid}: in API but not in the catalog")
            if lv:
                _out(
                    f"    API: req={lv.max_request_tokens:,}"
                    f" resp={lv.max_response_tokens:,}",
                )
            errors += 1
            continue
        if lv is None:
            _out(f"  {provider_name}.{mid}: not verified")
            continue
        verified += 1
        # The untagged context is the one the vendor API reports; ``+1m``
        # is an opt-in the model list does not enumerate.
        base = k.context[""]
        k_req = base.max_request_tokens
        k_resp = base.max_response_tokens
        if k_req != lv.max_request_tokens:
            _out(
                f"  {provider_name}.{mid}: max_request_tokens"
                f" code={k_req:,} api={lv.max_request_tokens:,}",
            )
            errors += 1
        if k_resp != lv.max_response_tokens:
            _out(
                f"  {provider_name}.{mid}: max_response_tokens"
                f" code={k_resp:,} api={lv.max_response_tokens:,}",
            )
            errors += 1
        # A selected level is sent verbatim -- ``output_config.effort`` on
        # Anthropic, ``reasoning.effort`` on OpenAI, which spells ``min``
        # ``minimal`` -- so the vendor's list and the row must match exactly:
        # a level only the catalog offers is a guaranteed 400, one only the
        # vendor lists is hidden. ``default`` means omit the knob, never a
        # wire value; a source that spells effort otherwise (Google's
        # ``thinkingLevel``, where ``none`` omits) must map before reporting.
        k_effort = k.thinking.effort - {"default"}
        if lv.effort is not None and k_effort != lv.effort:
            _out(
                f"  {provider_name}.{mid}: effort"
                f" code={sorted(k_effort)} api={sorted(lv.effort)}",
            )
            errors += 1
    if known and not verified:
        errors += 1
    if errors or verified < len(known):
        _out(f"  {provider_name}: {verified}/{len(known)} models verified")
    else:
        _out(f"  {provider_name}: all {len(known)} models OK")
    return errors


def audit_catalogs() -> int:
    """Check every provider catalog for self-consistency, offline.

    Catches the drift a live query cannot: an empty price catalog raises at
    first bill rather than at import; a zero window silently disables the
    compaction trigger.

    Returns:
      error_count: Number of problems found.

    """
    errors = 0
    for name in sorted(providers.PROVIDER_NAMES):
        cls = getattr(providers, name, None)
        if not isinstance(cls, ModelResolver):
            continue
        rows = {
            model_id: cls.catalog.resolve(model_id)[0]
            for model_id in cls.catalog.models
        }
        for mid, cap in rows.items():
            if not cap.prices:
                _out(f"  {name}.{mid}: no price rows -- spend() would raise")
                errors += 1
            for tag, lim in cap.context.items():
                where = f"{name}.{mid}{tag}"
                if lim.max_request_tokens <= 0:
                    _out(f"  {where}: max_request_tokens is 0")
                    errors += 1
                if lim.max_response_tokens <= 0:
                    _out(f"  {where}: max_response_tokens is 0")
                    errors += 1
    if not errors:
        _out("  catalogs: all rows OK")
    return errors


def snapshot_json(
    fetched: Mapping[str, Mapping[str, LiveLimits]],
    *,
    at: datetime,
) -> str:
    """Render what each live source reported as stable, reviewable JSON.

    Args:
      fetched: Per provider, what its live source reported per model id.
      at: When the sources were read.

    Returns:
      text: Sorted, indented JSON; unreported fields are omitted.

    """
    providers_out: dict[str, dict[str, dict[str, object]]] = {}
    for provider_name, live in fetched.items():
        rows: dict[str, dict[str, object]] = {}
        for mid, lv in live.items():
            row: dict[str, object] = {
                "max_request_tokens": lv.max_request_tokens,
                "max_response_tokens": lv.max_response_tokens,
            }
            if lv.effort is not None:
                row["effort"] = sorted(lv.effort)
            for name in ("lifecycle", "retires_at", "served_as"):
                if value := getattr(lv, name):
                    row[name] = value
            rows[mid] = row
        providers_out[provider_name] = rows
    body = {"fetched_at": at.isoformat(), "providers": providers_out}
    return json.dumps(body, indent=2, sort_keys=True) + "\n"


async def check_drift(report_path: pathlib.Path | None) -> int:
    """Compare every hosted catalog with models.dev.

    Args:
      report_path: Where to write the Markdown issue body when anything
        needs review; nothing is written otherwise.

    Returns:
      error_count: Disagreements needing review, plus stale triage entries,
        plus providers models.dev no longer lists; ``1`` when models.dev
        could not be read.

    """
    upstream = await models_dev.fetch(warn=_warn)
    if upstream is None:
        return 1
    drifts: list[models_dev.Drift] = []
    errors = 0
    for key, table in models_dev.CATALOGS.items():
        rows = upstream.get(key)
        # A provider key vanishing or emptying upstream must not read as "no
        # drift": Anthropic and Google carry no triage pins, so nothing else
        # would notice their comparisons silently stopping.
        if rows is None or not rows:
            _out(f"  {key}: models.dev lists no models; provider key moved?")
            errors += 1
            continue
        drifts.extend(models_dev.compare(key, table(), rows))
    for d in drifts:
        if not d.accepted:
            _out(
                f"  {d.provider}.{d.model_id}: {d.field}"
                f" sagent={models_dev.format_value(d.ours)}"
                f" models.dev={models_dev.format_value(d.theirs)}",
            )
    body = models_dev.report(drifts, at=datetime.now(UTC).date())
    triaged = sum(1 for d in drifts if d.accepted)
    _out(f"  {triaged} known disagreement(s) triaged")
    if body and report_path is not None:
        await asyncio.to_thread(report_path.write_text, body)
        _out(f"\nReport written to {report_path}")
    leftover = models_dev.stale(drifts)
    for mid, field in leftover:
        _out(f"  {mid}: {field} triage matches no current disagreement")
    return errors + sum(1 for d in drifts if not d.accepted) + len(leftover)


def main() -> int:
    """Run the program; return the process exit code."""
    return asyncio.run(_run())


def _out(msg: str) -> None:
    sys.stdout.write(msg + "\n")


def _warn(msg: str) -> None:
    _out(f"  [warn] {msg}")


async def _run() -> int:
    """Verify all providers' model catalogs against live APIs."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 2)[2] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--provider",
        choices=["google", "openai", "anthropic", "all"],
        default="all",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Only run the offline catalog audit; skip every network query.",
    )
    parser.add_argument(
        "--snapshot",
        type=pathlib.Path,
        help="Write what each live source reported, with the time, to this JSON file.",
    )
    parser.add_argument(
        "--drift",
        action="store_true",
        help="Compare every hosted catalog with models.dev instead of vendor APIs.",
    )
    parser.add_argument(
        "--report",
        type=pathlib.Path,
        help="With --drift, write what needs review as Markdown to this file.",
    )
    args = parser.parse_args()
    target = args.provider
    _out("Catalog audit (offline):")
    total_errors = audit_catalogs()
    if args.offline:
        if total_errors:
            _out(f"\n{total_errors} problem(s) found.")
        return 1 if total_errors else 0

    if args.drift:
        _out("models.dev drift:")
        total_errors += await check_drift(args.report)
        _out(
            f"\n{total_errors} problem(s) found."
            if total_errors
            else "\nNo untriaged drift.",
        )
        return 1 if total_errors else 0

    fetched: dict[str, dict[str, LiveLimits]] = {}
    if target in ("all", "google"):
        _out("Google (API query):")
        key = os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            _out("  [error] GOOGLE_API_KEY not set")
            total_errors += 1
        else:
            live = fetched["Google"] = await google_source.fetch(key, warn=_warn)
            total_errors += compare("Google", _canonical_models(Google.catalog), live)

    if target in ("all", "openai"):
        _out("OpenAI (doc scrape):")
        known = _canonical_models(OpenAI.catalog)
        live = fetched["OpenAI"] = await openai_source.fetch(list(known), warn=_warn)
        total_errors += compare("OpenAI", known, live)

    if target in ("all", "anthropic"):
        _out("Anthropic (API query):")
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            _out("  [error] ANTHROPIC_API_KEY not set")
            total_errors += 1
        else:
            known = _canonical_models(Anthropic.catalog)
            live = fetched["Anthropic"] = await anthropic_source.fetch(
                key,
                list(known),
                warn=_warn,
            )
            total_errors += compare("Anthropic", known, live)

    if args.snapshot is not None:
        args.snapshot.write_text(snapshot_json(fetched, at=datetime.now(UTC)))
        _out(f"\nSnapshot written to {args.snapshot}")

    if total_errors:
        _out(f"\n{total_errors} problem(s) found.")
    else:
        _out("\nAll limits verified.")
    return 1 if total_errors else 0


def _canonical_models(catalog: ModelCatalog) -> dict[str, ModelCapability]:
    """Return one resolved row per model, keyed by the id its vendor serves it as."""
    rows = (catalog.resolve(model_id)[0] for model_id in catalog.models)
    return {row.model_id: row for row in rows}


if __name__ == "__main__":
    raise SystemExit(main())
# vim: ft=python
