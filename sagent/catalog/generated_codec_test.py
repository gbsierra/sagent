"""Tests for the checked-in generated catalog boundary."""

from __future__ import annotations

from typing import Protocol

import pytest

from sagent.catalog import anthropic, dashscope, google, minimax, moonshot, openai
from sagent.catalog.generated_codec import decode_catalog, encode_catalog
from sagent.catalog.table import ModelTable
from sagent.lib.codec import ReadError


class _Catalog(Protocol):
    def models(self) -> ModelTable: ...

    def source_models(self) -> ModelTable: ...


@pytest.mark.parametrize(
    "catalog",
    [
        anthropic,
        dashscope,
        google,
        minimax,
        moonshot,
        openai,
    ],
)
def test_every_hosted_catalog_round_trips_as_inert_data(catalog: _Catalog) -> None:
    table = catalog.models()
    assert isinstance(table, ModelTable)
    encoded = encode_catalog(table)
    assert decode_catalog(encoded) == table


@pytest.mark.parametrize(
    "catalog",
    [anthropic, dashscope, google, minimax, moonshot, openai],
)
def test_generated_catalog_matches_reviewed_source(catalog: _Catalog) -> None:
    """A source edit must regenerate its checked-in runtime artifact."""
    assert catalog.models() == catalog.source_models()


def test_generated_data_cannot_request_an_import() -> None:
    with pytest.raises(ReadError):
        decode_catalog({"py/reduce": ["builtins.eval", ["1 + 1"]]})
