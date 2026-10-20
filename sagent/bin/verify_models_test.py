"""Tests for ``bin.verify_models``: provider limits checking via stubbed HTTP."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from types import MappingProxyType
from typing import TYPE_CHECKING

import json
import sys

import pytest

from sagent.bin import verify_models
from sagent.catalog.sources import (
    anthropic as anthropic_source,
    google as google_source,
    models_dev,
    openai as openai_source,
)
from sagent.catalog.sources.live import LiveLimits
from sagent.providers.anthropic.api import Anthropic
from sagent.providers.google.api import Google
from sagent.providers.openai.api import OpenAI
from sagent.types.capability import ModelCapability, ModelLimits, ThinkingCapability


if TYPE_CHECKING:
    from pathlib import Path

    from sagent.catalog.table import ModelCatalog


def _cap(request: int, response: int) -> ModelCapability:
    """Build a capability carrying only the two limits ``compare`` reads."""
    return ModelCapability(
        context=MappingProxyType(
            {"": ModelLimits(max_request_tokens=request, max_response_tokens=response)},
        ),
    )


def _limits(request: int, response: int) -> LiveLimits:
    return LiveLimits(
        max_request_tokens=request,
        max_response_tokens=response,
    )


class TestCanonicalModels:
    @pytest.mark.parametrize("catalog", [Anthropic.catalog, OpenAI.catalog])
    def test_keys_every_row_by_its_vendor_id(self, catalog: ModelCatalog) -> None:
        known = verify_models._canonical_models(catalog)
        assert list(known) == list(catalog.models)
        assert all(cap.model_id == mid for mid, cap in known.items())
        names = (
            catalog.models.roles.keys()
            | catalog.models.aliases.keys()
            | catalog.models.renamed.keys()
        )
        assert not names & known.keys()

    def test_an_alias_and_its_snapshot_are_each_verified(self) -> None:
        known = verify_models._canonical_models(Anthropic.catalog)
        assert {"claude-haiku-4-5", "claude-haiku-4-5-20251001"} <= known.keys()


class TestCompare:
    def test_reports_unknown_and_mismatches(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        known = {
            "same": _cap(10, 20),
            "diff": _cap(1, 2),
            "dead": _cap(1, 2),
        }
        live = {
            "same": _limits(10, 20),
            "diff": _limits(3, 4),
            "new": _limits(5, 6),
        }
        assert verify_models.compare("Provider", known, live) == 3
        out = capsys.readouterr().out
        assert "Provider.new: in API but not in the catalog" in out
        assert "Provider.dead: not verified" in out
        assert "max_request_tokens" in out
        assert "max_response_tokens" in out
        assert "2/3 models verified" in out

    def test_reports_all_ok(self, capsys: pytest.CaptureFixture[str]) -> None:
        live = {"m": _limits(1, 2)}
        assert verify_models.compare("Provider", {"m": _cap(1, 2)}, live) == 0
        assert "all 1 models OK" in capsys.readouterr().out

    @pytest.mark.parametrize(
        ("catalog_effort", "live_effort", "errors"),
        [
            # Both sides list the same levels, ``none`` included.
            (
                frozenset({"none", "low", "high"}),
                frozenset({"none", "low", "high"}),
                0,
            ),
            # Anthropic has no ``none`` effort -- thinking off rides the
            # separate ``thinking`` parameter -- so a row offering it sends
            # a level the API rejects.
            (
                frozenset({"none", "low", "high"}),
                frozenset({"low", "high"}),
                1,
            ),
            # A level only the vendor lists is one the catalog hides.
            (
                frozenset({"low", "high"}),
                frozenset({"low", "high", "max"}),
                1,
            ),
            # A vendor that states no effort list is not checked.
            (frozenset({"low", "high"}), None, 0),
        ],
    )
    def test_effort_must_match_the_vendors_level_list_exactly(
        self,
        capsys: pytest.CaptureFixture[str],
        catalog_effort: frozenset[str],
        live_effort: frozenset[str] | None,
        errors: int,
    ) -> None:
        known = {
            "m": replace(
                _cap(1, 2),
                thinking=ThinkingCapability(effort=catalog_effort),
            ),
        }
        live = {"m": replace(_limits(1, 2), effort=live_effort)}
        assert verify_models.compare("P", known, live) == errors
        assert ("P.m: effort" in capsys.readouterr().out) == bool(errors)

    def test_notes_the_served_id_and_warns_on_a_retiring_model(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        live = {
            "alias": replace(_limits(1, 2), served_as="alias-2026"),
            "old": replace(
                _limits(1, 2),
                lifecycle="deprecated",
                retires_at="2026-11-30",
            ),
            "fine": replace(_limits(1, 2), lifecycle="active"),
        }
        known = {mid: _cap(1, 2) for mid in live}
        assert verify_models.compare("P", known, live) == 0
        lines = capsys.readouterr().out.splitlines()
        assert "  P.alias: the API answers as alias-2026" in lines
        assert "  [warn] P.old: lifecycle deprecated, retires 2026-11-30" in lines
        assert not any("P.fine: lifecycle" in line for line in lines)

    def test_zero_coverage_is_an_error(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        assert verify_models.compare("P", {"m": _cap(1, 2)}, {}) == 1
        out = capsys.readouterr().out
        assert "all 1 models OK" not in out
        assert "0/1 models verified" in out


def _exact(known: dict[str, ModelCapability]) -> dict[str, LiveLimits]:
    """Live limits that agree with every catalog row."""
    return {
        mid: _limits(
            cap.context[""].max_request_tokens,
            cap.context[""].max_response_tokens,
        )
        for mid, cap in known.items()
    }


class TestSnapshot:
    def test_records_what_was_reported_and_when(self) -> None:
        live = {
            "Anthropic": {
                "claude-haiku-4-5": replace(
                    _limits(200_000, 64_000),
                    effort=frozenset(),
                    lifecycle="active",
                    served_as="claude-haiku-4-5-20251001",
                ),
            },
            "OpenAI": {"gpt-4.1": _limits(1_047_576, 32_768)},
        }
        at = datetime(2026, 10, 17, 12, 0, tzinfo=UTC)
        text = verify_models.snapshot_json(live, at=at)
        assert json.loads(text) == {
            "fetched_at": "2026-10-17T12:00:00+00:00",
            "providers": {
                "Anthropic": {
                    "claude-haiku-4-5": {
                        "max_request_tokens": 200_000,
                        "max_response_tokens": 64_000,
                        "effort": [],
                        "lifecycle": "active",
                        "served_as": "claude-haiku-4-5-20251001",
                    },
                },
                "OpenAI": {
                    "gpt-4.1": {
                        "max_request_tokens": 1_047_576,
                        "max_response_tokens": 32_768,
                    },
                },
            },
        }
        assert text == verify_models.snapshot_json(live, at=at)

    @pytest.mark.anyio
    async def test_run_writes_the_snapshot_it_was_asked_for(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "live.json"
        monkeypatch.setattr(
            sys,
            "argv",
            ["verify", "--provider", "openai", "--snapshot", str(path)],
        )

        async def fake_fetch_openai(
            ids: list[str],
            **_: object,
        ) -> dict[str, LiveLimits]:
            del ids
            return {"gpt-4.1": _limits(1, 2)}

        monkeypatch.setattr(openai_source, "fetch", fake_fetch_openai)
        await verify_models._run()
        written = json.loads(path.read_text())
        assert set(written["providers"]) == {"OpenAI"}
        assert written["providers"]["OpenAI"]["gpt-4.1"]["max_request_tokens"] == 1


class TestRun:
    @pytest.mark.anyio
    async def test_default_runs_every_provider_and_sums_problems(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["verify"])
        monkeypatch.setenv("GOOGLE_API_KEY", "gkey")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "akey")
        google = verify_models._canonical_models(Google.catalog)
        openai = verify_models._canonical_models(OpenAI.catalog)
        anthropic = verify_models._canonical_models(Anthropic.catalog)
        wrong = next(iter(openai))

        async def fake_google(key: str, **_: object) -> dict[str, LiveLimits]:
            assert key == "gkey"
            return {**_exact(google), "zz-new": _limits(1, 1)}

        async def fake_openai(ids: list[str], **_: object) -> dict[str, LiveLimits]:
            assert ids == list(openai)
            return {**_exact(openai), wrong: _limits(1, 1)}

        async def fake_anthropic(
            key: str,
            ids: list[str],
            **_: object,
        ) -> dict[str, LiveLimits]:
            assert key == "akey"
            assert ids == list(anthropic)
            return {**_exact(anthropic), "zz-new": _limits(1, 1)}

        monkeypatch.setattr(google_source, "fetch", fake_google)
        monkeypatch.setattr(openai_source, "fetch", fake_openai)
        monkeypatch.setattr(anthropic_source, "fetch", fake_anthropic)
        monkeypatch.setattr(verify_models, "audit_catalogs", lambda: 1)
        assert await verify_models._run() == 1
        lines = capsys.readouterr().out.splitlines()
        assert lines[0] == "Catalog audit (offline):"
        for line in (
            "Google (API query):",
            "  Google.zz-new: in API but not in the catalog",
            "OpenAI (doc scrape):",
            f"  OpenAI.{wrong}: max_request_tokens",
            "Anthropic (API query):",
            "  Anthropic.zz-new: in API but not in the catalog",
        ):
            assert any(got.startswith(line) for got in lines), line
        assert lines[-1] == "5 problem(s) found."

    @pytest.mark.anyio
    async def test_missing_keys_each_count_once(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["verify", "--provider", "all"])
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        openai = verify_models._canonical_models(OpenAI.catalog)

        async def fake_openai(ids: list[str], **_: object) -> dict[str, LiveLimits]:
            del ids
            return _exact(openai)

        monkeypatch.setattr(openai_source, "fetch", fake_openai)
        monkeypatch.setattr(verify_models, "audit_catalogs", lambda: 1)
        assert await verify_models._run() == 1
        lines = capsys.readouterr().out.splitlines()
        assert "  [error] GOOGLE_API_KEY not set" in lines
        assert "  [error] ANTHROPIC_API_KEY not set" in lines
        assert lines[-1] == "3 problem(s) found."

    @pytest.mark.anyio
    async def test_offline_skips_the_network(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["verify", "--offline"])
        monkeypatch.delattr(openai_source, "fetch")
        assert await verify_models._run() == 0
        assert capsys.readouterr().out.splitlines() == [
            "Catalog audit (offline):",
            "  catalogs: all rows OK",
        ]

    @pytest.mark.anyio
    async def test_offline_reports_audit_problems(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["verify", "--offline"])
        monkeypatch.setattr(verify_models, "audit_catalogs", lambda: 2)
        assert await verify_models._run() == 1
        assert capsys.readouterr().out.splitlines()[-1] == "2 problem(s) found."

    @pytest.mark.anyio
    async def test_help_shows_the_docstring_body(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["verify", "--help"])
        with pytest.raises(SystemExit) as exit_info:
            await verify_models._run()
        assert exit_info.value.code == 0
        out = capsys.readouterr().out
        assert "\nVerify provider model limits.\n" in out
        assert "\nSources:\n" in out
        assert "exec uv" not in out
        assert "{google,openai,anthropic,all}" in out

    @pytest.mark.anyio
    async def test_rejects_an_unknown_provider(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["verify", "--provider", "bogus"])
        with pytest.raises(SystemExit) as exit_info:
            await verify_models._run()
        assert exit_info.value.code == 2


class TestMain:
    @pytest.mark.anyio
    async def test_missing_api_key_fails_the_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["verify", "--provider", "google"])
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        assert await verify_models._run() == 1
        out = capsys.readouterr().out
        assert "GOOGLE_API_KEY not set" in out
        assert "All limits verified" not in out

    @pytest.mark.anyio
    async def test_returns_nonzero_on_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["verify", "--provider", "openai"])

        async def fake_fetch_openai(
            ids: list[str],
            **_: object,
        ) -> dict[str, LiveLimits]:
            return {mid: _limits(1, 1) for mid in ids}

        monkeypatch.setattr(openai_source, "fetch", fake_fetch_openai)
        assert await verify_models._run() == 1

    @pytest.mark.anyio
    async def test_matching_limits_pass(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["verify", "--provider", "openai"])
        known = verify_models._canonical_models(OpenAI.catalog)

        async def fake_fetch_openai(
            ids: list[str],
            **_: object,
        ) -> dict[str, LiveLimits]:
            return {
                mid: _limits(
                    known[mid].context[""].max_request_tokens,
                    known[mid].context[""].max_response_tokens,
                )
                for mid in ids
            }

        monkeypatch.setattr(openai_source, "fetch", fake_fetch_openai)
        assert await verify_models._run() == 0
        assert capsys.readouterr().out.splitlines()[-1] == "All limits verified."

    @pytest.mark.anyio
    async def test_google_with_no_live_rows_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["verify", "--provider", "google"])
        monkeypatch.setenv("GOOGLE_API_KEY", "x")

        async def fake_fetch_google(
            key: str,
            **_: object,
        ) -> dict[str, LiveLimits]:
            del key
            return {}

        monkeypatch.setattr(google_source, "fetch", fake_fetch_google)
        assert await verify_models._run() == 1
        assert "All limits verified" not in capsys.readouterr().out

    @pytest.mark.anyio
    async def test_missing_anthropic_key_fails_the_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["verify", "--provider", "anthropic"])
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert await verify_models._run() == 1
        assert "ANTHROPIC_API_KEY not set" in capsys.readouterr().out

    @pytest.mark.anyio
    async def test_runs_anthropic_by_wire_id_when_key_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(sys, "argv", ["verify", "--provider", "anthropic"])
        monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
        queried: list[str] = []

        async def fake_fetch_anthropic(
            key: str,
            ids: list[str],
            **_: object,
        ) -> dict[str, LiveLimits]:
            del key
            queried.extend(ids)
            return {}

        monkeypatch.setattr(anthropic_source, "fetch", fake_fetch_anthropic)
        assert await verify_models._run() == 1
        assert "claude-opus-5-5" in queried


type _Upstream = dict[str, dict[str, dict[str, dict[str, float]]]]


def _triaged_upstream() -> _Upstream:
    """models.dev rows showing exactly the values already triaged."""
    out: _Upstream = {}
    for (mid, field), (pinned, _) in models_dev.ACCEPTED.items():
        key = next(k for k, build in models_dev.CATALOGS.items() if mid in build())
        group, name = field.split(".")
        out.setdefault(key, {}).setdefault(mid, {}).setdefault(group, {})[name] = pinned
    # A provider key absent upstream is an error, so every compared provider
    # lists something; a row matching no catalog id compares nothing.
    for key in models_dev.CATALOGS:
        out.setdefault(key, {"__placeholder__": {}})
    return out


class TestDrift:
    @pytest.mark.anyio
    async def test_only_triaged_drift_passes_and_writes_no_report(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        report = tmp_path / "drift.md"
        monkeypatch.setattr(
            sys,
            "argv",
            ["verify", "--drift", "--report", str(report)],
        )

        async def fake_fetch(**_: object) -> _Upstream:
            return _triaged_upstream()

        monkeypatch.setattr(models_dev, "fetch", fake_fetch)
        monkeypatch.delattr(openai_source, "fetch")
        assert await verify_models._run() == 0
        assert not report.exists()
        out = capsys.readouterr().out
        assert f"  {len(models_dev.ACCEPTED)} known disagreement(s) triaged" in out
        assert out.splitlines()[-1] == "No untriaged drift."

    @pytest.mark.anyio
    async def test_untriaged_drift_fails_and_writes_the_report(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        report = tmp_path / "drift.md"
        monkeypatch.setattr(
            sys,
            "argv",
            ["verify", "--drift", "--report", str(report)],
        )
        upstream = _triaged_upstream()
        upstream["anthropic"] = {"claude-opus-5-5": {"cost": {"input": 99}}}

        async def fake_fetch(**_: object) -> _Upstream:
            return upstream

        monkeypatch.setattr(models_dev, "fetch", fake_fetch)
        assert await verify_models._run() == 1
        assert "`claude-opus-5-5` | cost.input" in report.read_text()
        out = capsys.readouterr().out
        assert "  anthropic.claude-opus-5-5: cost.input sagent=4 models.dev=99" in out
        assert out.splitlines()[-1] == "1 problem(s) found."

    @pytest.mark.anyio
    async def test_a_provider_absent_upstream_is_an_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        """A renamed or dropped provider key must not read as "no drift"."""
        report = tmp_path / "drift.md"
        monkeypatch.setattr(
            sys,
            "argv",
            ["verify", "--drift", "--report", str(report)],
        )
        upstream = _triaged_upstream()
        del upstream["anthropic"]

        async def fake_fetch(**_: object) -> _Upstream:
            return upstream

        monkeypatch.setattr(models_dev, "fetch", fake_fetch)
        assert await verify_models._run() == 1
        assert "  anthropic: models.dev lists no models; provider key moved?" in (
            capsys.readouterr().out
        )

    @pytest.mark.anyio
    async def test_an_unreadable_models_dev_fails_without_a_report(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        report = tmp_path / "drift.md"
        monkeypatch.setattr(
            sys,
            "argv",
            ["verify", "--drift", "--report", str(report)],
        )

        async def fake_fetch(**_: object) -> None:
            return None

        monkeypatch.setattr(models_dev, "fetch", fake_fetch)
        assert await verify_models._run() == 1
        assert not report.exists()


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
