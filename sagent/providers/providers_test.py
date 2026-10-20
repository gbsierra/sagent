"""Tests for ``providers.providers``: build_provider + infer_provider + names."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import Mock, patch

import sys

import pytest


if TYPE_CHECKING:
    from collections.abc import Callable

from sagent import providers
from sagent.providers import (
    PROVIDER_NAMES,
    Anthropic,
    DashScope,
    Google,
    LlamaCpp,
    MiniMax,
    Moonshot,
    OpenAI,
    SelfHosted,
    build_provider,
    infer_provider,
)
from sagent.providers.providers import (
    PROVIDER_ALIASES,
    UnknownProviderError,
    build_provider_with_account_fallback,
    canonical_provider,
    provider_class,
    require_provider,
)
from sagent.types.cost import PriceKey, TokenCount
from sagent.types.providers import ModelResolver


# ``provider_class`` only resolves names in ``PROVIDER_NAMES``, so a stub class set on
# the facade would be invisible; patch the lookup itself instead.
def _register_stub_provider(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    cls: type,
) -> None:
    module = sys.modules["sagent.providers.providers"]
    original = cast("Callable[[str], type | None]", module.provider_class)

    def _lookup(provider_name: str) -> type | None:
        return cls if provider_name == name else original(provider_name)

    monkeypatch.setattr(module, "provider_class", _lookup)


def test_provider_names_contains_core_providers() -> None:
    for name in ("Anthropic", "Google", "OpenAI", "DashScope", "MiniMax"):
        assert name in PROVIDER_NAMES


def test_provider_names_includes_self_hosted_and_compat() -> None:
    assert "SelfHosted" in PROVIDER_NAMES
    assert "OpenAICompat" in PROVIDER_NAMES
    assert "LlamaCpp" in PROVIDER_NAMES


@pytest.mark.parametrize(
    ("model_id", "provider"),
    [
        ("claude-sonnet-4-6", "Anthropic"),
        ("gemini-3-flash-preview", "Google"),
        ("gpt-5.5", "OpenAI"),
        ("o3-mini", "OpenAI"),
        ("kimi-k2.6", "Moonshot"),
        ("qwen3.6-plus", "DashScope"),
        ("MiniMax-M2.7", "MiniMax"),
    ],
)
def test_infer_provider_from_prefix(model_id: str, provider: str) -> None:
    result = infer_provider(model_id, current_provider="None")
    assert result is not None
    assert result[0] == provider
    assert result[1] == "env"


def test_infer_provider_returns_none_when_already_matches() -> None:
    assert infer_provider("claude-sonnet-4-6", current_provider="Anthropic") is None


def test_infer_provider_uses_the_shared_model_alias_registry() -> None:
    assert infer_provider("sol", current_provider="Google") == ("OpenAI", "env")
    assert infer_provider("sol", current_provider="OpenAI") is None
    assert infer_provider("sol", current_provider="OpenAISubscription") is None


@pytest.mark.parametrize(
    ("model_id", "provider"),
    [("claude-opus-4-8", "Anthropic"), ("gpt-6-luna", "OpenAI")],
)
def test_infer_provider_uses_short_catalog_ids(model_id: str, provider: str) -> None:
    assert infer_provider(model_id, current_provider="Google") == (provider, "env")


def test_infer_provider_preserves_same_family_variant() -> None:
    """A same-vendor model swap keeps the current provider variant, never
    downgrading it to the bare API-key sibling.

    An agent already on a variant provider (``AnthropicCLI``,
    ``OpenAISubscription``, ...) that swaps to another model of the SAME
    vendor family must keep its provider/auth: re-inferring to the bare
    ``Anthropic``/``OpenAI`` + ``env`` path would demand an API key the
    operator never set (they authenticate via the CLI/subscription). The
    guard is a catalog check -- the current provider's catalog resolves the
    id -- so it holds regardless of which override entries the account table
    happens to carry. The public (exported) build relies on this for
    ``OpenAISubscription``, which its override table does not enumerate.
    """
    assert infer_provider("claude-haiku-4-5", current_provider="AnthropicCLI") is None
    assert infer_provider("gpt-5.5", current_provider="OpenAISubscription") is None
    # Cross-vendor from a variant caller still resolves to the new vendor.
    assert infer_provider("gpt-5.5", current_provider="AnthropicCLI") == (
        "OpenAI",
        "env",
    )


def test_infer_provider_prefers_account_variant_for_account_callers() -> None:
    """A claude-prefixed model id resolves to the ACCOUNT provider when the
    caller already runs on one.

    ``AgentSelf(model_id="claude-...")`` from an AnthropicCLI-backed
    agent must not silently build a fresh API-key ``Anthropic`` provider
    (which would demand ``ANTHROPIC_API_KEY`` even though the operator
    authenticates via the CLI subscription). Same-provider stays a
    no-op; API-key callers are unaffected.
    """
    # CLI-backed caller asking for another claude model: the override maps
    # Anthropic -> AnthropicCLI, which matches the current provider, so no
    # rebuild at all. Without the override this returned ("Anthropic",
    # "env") -- a fresh API-key provider.
    assert infer_provider("claude-haiku-4-5", current_provider="AnthropicCLI") is None
    # Cross-vendor from a CLI caller still maps to the vendor's API path.
    assert infer_provider("gpt-5.5", current_provider="AnthropicCLI") == (
        "OpenAI",
        "env",
    )


def test_infer_provider_returns_none_for_unknown_prefix() -> None:
    assert infer_provider("mystery-model", current_provider="OpenAI") is None


def test_infer_provider_local_path_returns_self_hosted() -> None:
    result = infer_provider("/opt/models/qwen", current_provider="None")
    assert result == ("SelfHosted", "/opt/models/qwen")


@pytest.mark.parametrize("path", ["./local", "../parent", "~/snapshot"])
def test_infer_provider_other_local_paths(path: str) -> None:
    result = infer_provider(path, current_provider="None")
    assert result == ("SelfHosted", path)


def test_build_provider_unknown_provider_raises() -> None:
    with pytest.raises(AttributeError, match="unknown provider"):
        build_provider("DoesNotExist")


def test_build_provider_from_env_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env-key")
    p = build_provider("Anthropic", "env")
    assert isinstance(p, Anthropic)


@pytest.mark.parametrize(
    ("provider_name", "cls", "env_var"),
    [
        ("Google", Google, "GOOGLE_API_KEY"),
        ("OpenAI", OpenAI, "OPENAI_API_KEY"),
        ("DashScope", DashScope, "DASHSCOPE_API_KEY"),
        ("MiniMax", MiniMax, "MINIMAX_API_KEY"),
        ("Moonshot", Moonshot, "MOONSHOT_API_KEY"),
    ],
)
def test_build_provider_from_env_dispatch_per_provider(
    monkeypatch: pytest.MonkeyPatch,
    provider_name: str,
    cls: type,
    env_var: str,
) -> None:
    monkeypatch.setenv(env_var, "test-key")
    p = build_provider(provider_name, "env")
    assert isinstance(p, cls)


def test_build_provider_llamacpp_from_key_direct() -> None:
    # ``from_key`` factories take their key positionally at the class --
    # ``build_provider`` dispatches auth methods, never credentials.
    assert isinstance(LlamaCpp.from_key("local"), LlamaCpp)


def test_build_provider_account_kw_threaded_when_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If a provider's ``from_<auth>`` accepts ``account``, it gets forwarded."""
    # Build a stub provider class with a from_credentials(account=) signature.
    calls: list[dict[str, object]] = []

    class StubProv:
        @classmethod
        def from_credentials(cls, *, account: str | None = None) -> StubProv:
            calls.append({"account": account})
            return cls()

    _register_stub_provider(monkeypatch, "StubProv", StubProv)
    out = build_provider("StubProv", "credentials", account="work")
    assert isinstance(out, StubProv)
    assert calls == [{"account": "work"}]


def test_build_provider_with_account_fallback_retains_working_account() -> None:
    provider = Mock()
    with patch(
        "sagent.providers.providers.build_provider",
        return_value=provider,
    ) as build:
        result = build_provider_with_account_fallback(
            "StubProv",
            "credentials",
            account="work",
            fallback_to_default=True,
        )
    assert result == (provider, "work")
    build.assert_called_once_with("StubProv", "credentials", account="work")


def test_build_provider_with_account_fallback_retries_default() -> None:
    provider = Mock()
    with patch(
        "sagent.providers.providers.build_provider",
        side_effect=[FileNotFoundError("missing"), provider],
    ) as build:
        result = build_provider_with_account_fallback(
            "StubProv",
            "credentials",
            account="work",
            fallback_to_default=True,
        )
    assert result == (provider, None)
    assert build.call_args_list == [
        (("StubProv", "credentials"), {"account": "work"}),
        (("StubProv", "credentials"), {"account": None}),
    ]


@pytest.mark.parametrize("account", [None, "", "default"])
def test_build_provider_with_account_fallback_does_not_retry_default_like_account(
    account: str | None,
) -> None:
    with (
        patch(
            "sagent.providers.providers.build_provider",
            side_effect=FileNotFoundError("missing"),
        ) as build,
        pytest.raises(FileNotFoundError, match="missing"),
    ):
        _ = build_provider_with_account_fallback(
            "StubProv",
            "credentials",
            account=account,
            fallback_to_default=True,
        )
    build.assert_called_once_with("StubProv", "credentials", account=account)


def test_build_provider_with_account_fallback_respects_disabled_fallback() -> None:
    with (
        patch(
            "sagent.providers.providers.build_provider",
            side_effect=FileNotFoundError("missing"),
        ) as build,
        pytest.raises(FileNotFoundError, match="missing"),
    ):
        _ = build_provider_with_account_fallback(
            "StubProv",
            "credentials",
            account="work",
            fallback_to_default=False,
        )
    build.assert_called_once_with("StubProv", "credentials", account="work")


def test_build_provider_missing_auth_method_raises_even_with_from_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubProvKeyOnly:
        @classmethod
        def from_key(cls, api_key: str) -> StubProvKeyOnly:
            del api_key
            return cls()

    _register_stub_provider(monkeypatch, "StubProvKeyOnly", StubProvKeyOnly)
    with pytest.raises(AttributeError, match="no ``from_literal-key-here``"):
        build_provider("StubProvKeyOnly", "literal-key-here")


def test_build_provider_no_match_no_from_key_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BareProv:
        pass

    _register_stub_provider(monkeypatch, "BareProv", BareProv)
    with pytest.raises(AttributeError, match="no ``from_unsupported``"):
        build_provider("BareProv", "unsupported")


def test_key_only_provider_has_no_zero_argument_default_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StubProvKeyOnly:
        @classmethod
        def from_key(cls, api_key: str) -> StubProvKeyOnly:
            del api_key
            return cls()

    _register_stub_provider(monkeypatch, "StubProvKeyOnly", StubProvKeyOnly)
    with pytest.raises(AttributeError, match="no default auth method"):
        providers.default_auth_for_provider("StubProvKeyOnly")


def test_self_hosted_imported_via_dispatch() -> None:
    # We exercise that ``SelfHosted`` is exposed; we don't load HF weights.
    assert SelfHosted.__name__ == "SelfHosted"


def test_build_provider_defers_to_the_factory_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construction knobs are the factory's; ``build_provider`` invents none."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env-key")
    p = build_provider("Anthropic", "env")
    assert isinstance(p, Anthropic)
    assert p.server_side_context_management is False


def test_build_provider_forwards_account_only_where_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A factory without an ``account`` parameter must not receive one."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    assert isinstance(build_provider("Google", "env", account="ignored"), Google)


@pytest.mark.parametrize(
    "provider_name",
    sorted(
        name
        for name in PROVIDER_NAMES
        if isinstance((cls := getattr(providers, name, None)), ModelResolver)
        and cls.catalog.models
    ),
)
def test_a_catalog_backed_provider_names_a_cheaper_utility_model(
    provider_name: str,
) -> None:
    """``utility`` must name a row, and a cheaper one than ``default``."""
    cls = getattr(providers, provider_name)
    assert not hasattr(cls, "resolve_model")
    assert not hasattr(cls, "model_ids")
    default_row, _ = cls.catalog.resolve("default")
    utility_row, _ = cls.catalog.resolve("utility")
    assert not hasattr(cls, "CAPABILITIES")
    assert not hasattr(cls, "TRANSPORT")
    if utility_row.model_id == default_row.model_id:
        pytest.skip("single-model catalog: utility and default coincide")
    tokens = TokenCount(request=1_000_000, response=100_000)
    assert (utility_row.prices[PriceKey("auto")] * tokens).total < (
        default_row.prices[PriceKey("auto")] * tokens
    ).total, (
        f"{provider_name} utility {utility_row.model_id!r} is not cheaper than"
        f" its default {default_row.model_id!r}"
    )


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("oai", "OpenAI"),
        ("oaisub", "OpenAISubscription"),
        ("ant", "Anthropic"),
        ("antsub", "AnthropicCLI"),
    ],
)
def test_provider_aliases_resolve_to_class_names(alias: str, canonical: str) -> None:
    assert PROVIDER_ALIASES[alias] == canonical
    assert canonical_provider(alias) == canonical
    assert require_provider(alias) == canonical
    assert require_provider(alias.upper()) == canonical
    assert require_provider(f"  {alias} ") == canonical


def test_provider_alias_targets_are_known_and_do_not_shadow_class_names() -> None:
    lowered = {name.lower() for name in PROVIDER_NAMES}
    for alias, target in PROVIDER_ALIASES.items():
        assert target in PROVIDER_NAMES
        assert alias.lower() not in lowered


@pytest.mark.parametrize("name", PROVIDER_NAMES)
def test_canonical_provider_keeps_class_names_and_ignores_case(name: str) -> None:
    assert canonical_provider(name) == name
    assert canonical_provider(name.lower()) == name
    assert require_provider(name) == name


def test_canonical_provider_passes_unknown_names_through_stripped() -> None:
    assert canonical_provider(" NotAProvider ") == "NotAProvider"


def test_require_provider_unknown_lists_names_and_aliases() -> None:
    with pytest.raises(UnknownProviderError) as exc:
        _ = require_provider("oaii")
    message = str(exc.value)
    assert "'oaii'" in message
    for name in PROVIDER_NAMES:
        assert name in message
    for alias in PROVIDER_ALIASES:
        assert alias in message
    assert isinstance(exc.value, ValueError)


def test_provider_class_resolves_aliases() -> None:
    assert provider_class("oai") is OpenAI
    assert provider_class("ANT") is Anthropic


@pytest.mark.parametrize(
    "name",
    ["SelfHostedModel", "build_provider", "PROVIDER_NAMES", "providers"],
)
def test_provider_class_rejects_non_provider_facade_attributes(name: str) -> None:
    assert provider_class(name) is None


def test_build_provider_unknown_lists_aliases() -> None:
    with pytest.raises(AttributeError, match="oaisub"):
        build_provider("DoesNotExist")


def test_build_provider_accepts_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-env-key")
    assert isinstance(build_provider("oai", "env"), OpenAI)


@pytest.mark.parametrize(
    ("alias", "auth"),
    [("ant", "env"), ("antsub", "credentials"), ("oaisub", "credentials")],
)
def test_default_auth_for_provider_accepts_aliases(alias: str, auth: str) -> None:
    assert providers.default_auth_for_provider(alias) == auth


def test_infer_provider_accepts_alias_for_current_provider() -> None:
    assert infer_provider("claude-haiku-4-5", current_provider="antsub") is None


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
