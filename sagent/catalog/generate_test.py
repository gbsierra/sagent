"""Tests for deterministic models.dev baseline generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import json

import pytest

from sagent.catalog.generate import (
    SUPPORTED_PROVIDERS,
    GenerationError,
    generate,
    normalize,
    render,
)


if TYPE_CHECKING:
    from pathlib import Path


def _body() -> bytes:
    providers = {
        provider: {
            "name": provider.title(),
            "models": {
                f"{provider}-model": {
                    "id": f"{provider}-model",
                    "name": "Model",
                    "description": "not part of the normalized contract",
                    "reasoning": True,
                    "modalities": {"input": ["text"], "output": ["text"]},
                    "limit": {"context": 100, "input": 90, "output": 10},
                    "cost": {"input": 1.0, "output": 2.0},
                },
            },
        }
        for provider in SUPPORTED_PROVIDERS
    }
    return json.dumps(providers, separators=(",", ":")).encode()


def test_normalize_keeps_only_mapped_fields() -> None:
    baseline = normalize(_body(), minimum_provider_count=len(SUPPORTED_PROVIDERS))
    providers = baseline["providers"]
    assert isinstance(providers, dict)
    openai = providers["openai"]
    assert isinstance(openai, dict)
    models = openai["models"]
    assert isinstance(models, dict)
    model = models["openai-model"]
    assert isinstance(model, dict)

    assert model["limit"] == {"context": 100, "input": 90, "output": 10}
    assert "description" not in model


def test_normalize_rejects_small_or_missing_provider_sets() -> None:
    with pytest.raises(GenerationError, match="expected at least"):
        normalize(b"{}")

    decoded = json.loads(_body())
    del decoded["openai"]
    with pytest.raises(GenerationError, match=r"openai.*missing"):
        normalize(
            json.dumps(decoded).encode(),
            minimum_provider_count=len(SUPPORTED_PROVIDERS) - 1,
        )


def test_normalize_rejects_key_id_disagreement() -> None:
    decoded = json.loads(_body())
    decoded["openai"]["models"]["openai-model"]["id"] = "other"

    with pytest.raises(GenerationError, match="reports id"):
        normalize(
            json.dumps(decoded).encode(),
            minimum_provider_count=len(SUPPORTED_PROVIDERS),
        )


def test_same_bytes_render_identically() -> None:
    baseline = normalize(_body(), minimum_provider_count=len(SUPPORTED_PROVIDERS))
    assert render(baseline) == render(baseline)
    assert "description" not in render(baseline)


def test_generate_atomically_writes_importable_module(tmp_path: Path) -> None:
    output = tmp_path / "models_dev.py"
    # The public generator enforces the production truncation guard, so make a
    # production-sized fixture by padding providers it will ignore.
    decoded = json.loads(_body())
    for i in range(100):
        decoded[f"unused-{i}"] = {"models": {}}
    body = json.dumps(decoded).encode()

    generate(body, output)

    text = output.read_text()
    assert "DO NOT EDIT" in text
    assert "source_sha256" in text
    compile(text, str(output), "exec")
    for provider_id in (
        "anthropic",
        "dashscope",
        "google",
        "minimax",
        "moonshot",
        "openai",
    ):
        provider_output = tmp_path / f"{provider_id}.py"
        provider_text = provider_output.read_text()
        assert "decode_catalog(DATA)" in provider_text
        compile(provider_text, str(provider_output), "exec")
