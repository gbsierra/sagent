"""Generate Sagent's checked-in models.dev baseline.

Normal Sagent startup never imports an HTTP client from this module.  Catalog
refresh is an explicit build/maintenance command; the result is deterministic
for identical input bytes.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Final, cast

import argparse
import hashlib
import json
import pprint

import httpx2

from sagent.catalog import anthropic, dashscope, google, minimax, moonshot, openai
from sagent.catalog.generated_codec import encode_catalog
from sagent.lib.atomic_file import atomic_write_bytes


if TYPE_CHECKING:
    from collections.abc import Mapping

    from sagent.catalog.table import ModelTable


__all__ = [
    "MODELS_DEV_URL",
    "SUPPORTED_PROVIDERS",
    "generate",
    "generate_catalogs",
    "main",
    "normalize",
    "render",
    "render_catalog",
]


MODELS_DEV_URL: Final = "https://models.dev/api.json"
SCHEMA_VERSION: Final = 1
MINIMUM_PROVIDER_COUNT: Final = 100
SUPPORTED_PROVIDERS: Final[tuple[str, ...]] = (
    "alibaba",
    "anthropic",
    "google",
    "minimax",
    "moonshotai",
    "openai",
)
DEFAULT_OUTPUT: Final = Path(__file__).with_name("generated") / "models_dev.py"

_MODEL_FIELDS: Final = (
    "id",
    "name",
    "family",
    "status",
    "release_date",
    "last_updated",
    "knowledge",
    "open_weights",
    "attachment",
    "reasoning",
    "reasoning_options",
    "tool_call",
    "structured_output",
    "temperature",
    "modalities",
    "limit",
    "cost",
    "experimental",
    "canonical_model_id",
)


class GenerationError(ValueError):
    """models.dev input cannot produce a trustworthy baseline."""


def normalize(
    body: bytes, *, minimum_provider_count: int = MINIMUM_PROVIDER_COUNT
) -> dict[str, object]:
    """Validate and normalize models.dev JSON bytes.

    Args:
      body: Exact downloaded JSON bytes.
      minimum_provider_count: Truncation guard; lowered only by unit fixtures.

    Returns:
      baseline: Supported providers and their normalized model mappings.

    Raises:
      GenerationError: The response is malformed, truncated, or missing a
        supported provider.

    """
    try:
        decoded_obj: object = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise GenerationError("models.dev response is not valid JSON") from e
    if not isinstance(decoded_obj, dict):
        raise GenerationError("models.dev response must be an object")
    decoded = cast(dict[str, object], decoded_obj)
    if len(decoded) < minimum_provider_count:
        raise GenerationError(
            f"models.dev response has {len(decoded)} providers; expected at least {minimum_provider_count}",
        )

    providers: dict[str, object] = {}
    for provider_id in SUPPORTED_PROVIDERS:
        raw_provider = decoded.get(provider_id)
        if not isinstance(raw_provider, dict):
            raise GenerationError(f"models.dev provider {provider_id!r} is missing")
        provider = cast(dict[str, object], raw_provider)
        raw_models = provider.get("models")
        if not isinstance(raw_models, dict) or not raw_models:
            raise GenerationError(f"models.dev provider {provider_id!r} has no models")
        for model_id in cast(dict[object, object], raw_models):
            if not isinstance(model_id, str):
                raise GenerationError(
                    f"models.dev provider {provider_id!r} has a non-string model id"
                )
        model_map = cast(dict[str, object], raw_models)
        models: dict[str, object] = {}
        for model_id in sorted(model_map):
            raw_model = model_map[model_id]
            if not isinstance(raw_model, dict):
                raise GenerationError(f"{provider_id}.{model_id} must be an object")
            model = cast(dict[str, object], raw_model)
            reported_id = model.get("id")
            if reported_id != model_id:
                raise GenerationError(
                    f"{provider_id}.{model_id} reports id {reported_id!r}",
                )
            models[model_id] = _normalize_model(model)
        providers[provider_id] = {
            "id": provider_id,
            "name": provider.get("name")
            if isinstance(provider.get("name"), str)
            else None,
            "models": models,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "source_url": MODELS_DEV_URL,
        "source_sha256": hashlib.sha256(body).hexdigest(),
        "providers": providers,
    }


def _normalize_model(raw: Mapping[str, object]) -> dict[str, object]:
    """Keep only explicitly mapped models.dev fields in stable key order."""
    out: dict[str, object] = {}
    for name in _MODEL_FIELDS:
        if name not in raw:
            continue
        value = raw[name]
        if value is None or isinstance(value, (bool, int, float, str)):
            out[name] = value
        elif isinstance(value, (list, dict)):
            # Round-trip through JSON to reject non-JSON fixture objects and
            # detach the generated value from the decoded input mapping.
            out[name] = json.loads(json.dumps(value, sort_keys=True))
        else:
            raise GenerationError(f"model field {name!r} is not JSON-compatible")
    return out


def render(baseline: Mapping[str, object]) -> str:
    """Render one deterministic Python module."""
    literal = pprint.pformat(dict(baseline), width=100, sort_dicts=True)
    return (
        "# ruff: noqa: I001, Q000\n"
        "# fmt: off\n"
        '"""Generated models.dev baseline.  DO NOT EDIT.\n\n'
        f"Source: {MODELS_DEV_URL}\n"
        "Regenerate: python -m sagent.catalog.generate\n"
        '"""\n\n'
        "from typing import Final\n\n"
        f"BASELINE: Final[dict[str, object]] = {literal}\n"
    )


def render_catalog(
    provider_id: str,
    table: ModelTable,
) -> str:
    """Render one final, runtime-ready provider catalog as inert data."""
    # Preserve traversal order: the codec's safe ``py/id`` back-references
    # must follow the value they reference.
    literal = pprint.pformat(encode_catalog(table), width=100, sort_dicts=False)
    return (
        "# ruff: noqa: I001, Q000\n"
        "# fmt: off\n"
        f'"""Generated {provider_id} catalog.  DO NOT EDIT.\n\n'
        "Regenerate: python -m sagent.catalog.generate\n"
        '"""\n\n'
        "from typing import Final, cast\n\n"
        "from sagent.catalog.generated_codec import decode_catalog\n"
        "from sagent.lib.codec import PlainTree\n\n"
        f"DATA: Final[PlainTree] = cast('PlainTree', {literal})\n\n"
        "MODELS = decode_catalog(DATA)\n"
    )


def _source_tables() -> Mapping[str, ModelTable]:
    """Return typed, provider-reviewed generation inputs."""
    return {
        "anthropic": anthropic.source_models(),
        "dashscope": dashscope.source_models(),
        "google": google.source_models(),
        "minimax": minimax.source_models(),
        "moonshot": moonshot.source_models(),
        "openai": openai.source_models(),
    }


def generate_catalogs(baseline: Mapping[str, object], output_dir: Path) -> None:
    """Atomically write every final hosted catalog after full normalization."""
    providers = baseline.get("providers")
    if not isinstance(providers, dict) or any(
        provider_id not in providers for provider_id in SUPPORTED_PROVIDERS
    ):
        raise GenerationError("normalized baseline is missing a supported provider")
    for provider_id, table in _source_tables().items():
        content = render_catalog(provider_id, table).encode()
        atomic_write_bytes(output_dir / f"{provider_id}.py", content)


def generate(body: bytes, output: Path) -> None:
    """Normalize ``body`` and atomically replace every generated artifact."""
    baseline = normalize(body)
    atomic_write_bytes(output, render(baseline).encode())
    generate_catalogs(baseline, output.parent)


def _fetch() -> bytes:
    """Download the aggregate with a bounded request."""
    with httpx2.Client(timeout=60, follow_redirects=True) as client:
        response = client.get(MODELS_DEV_URL)
        response.raise_for_status()
        return response.content


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="Read saved models.dev JSON instead of using the network.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    body = args.input.read_bytes() if args.input is not None else _fetch()
    output = cast(Path, args.output)
    generate(body, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
