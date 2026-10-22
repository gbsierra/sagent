"""Subscription authentication (SIWC), request restrictions, credential I/O tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs, urlparse

import base64
import contextlib
import io
import json
import re
import threading
import time

from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import (
    RSAAlgorithm,  # ty: ignore[possibly-missing-import] -- pyjwt hides RSAAlgorithm behind its optional cryptography extra.
)

import httpx2
import jwt
import openai
import pytest

from sagent.catalog.openai import models, reasoning_effort, subscription_models
from sagent.providers import OpenAI
from sagent.providers.lib.errors import (
    PER_ITEM_STRING_CAP_BODY,
    StreamingResponseNotReadError,
    find_response_not_read,
)
from sagent.providers.lib.oauth import credentials_path
from sagent.providers.openai import siwc, sub
from sagent.providers.openai.sub import (
    OpenAISubscription,
    _jwt_exp,
    _jwt_payload,
)
from sagent.tools import tool
from sagent.types.capability import (
    ModelSettings,
    ThinkingEffort,
)
from sagent.types.exceptions import AuthRefreshError
from sagent.types.model import (
    ModelRequest,
)
from sagent.types.runtime import (
    UserMessage,
)


if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path

    from sagent.types.tools import Tool


@pytest.fixture(autouse=True)
def isolate_data_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Keep sagent's own credential store inside a temp XDG data home."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path_factory.mktemp("xdg")))


def test_subscription_context_clamps_request_tokens() -> None:
    clamped = subscription_models()["astra-6.0"].context[""]
    assert clamped.max_request_tokens == 272_000
    assert clamped.max_response_tokens == 32_000


def test_subscription_context_keeps_small_windows() -> None:
    clamped = subscription_models()["gpt-4.0"].context[""]
    assert clamped.max_request_tokens == 8_192
    assert clamped.max_response_tokens == 8_192


def test_subscription_context_drops_the_long_window_tag() -> None:
    """``+1m`` clamps to exactly the base id, so offering it would mislead."""
    rows = subscription_models()
    assert all(capability.context.keys() == {""} for capability in rows.values())


def test_subscription_context_inherits_size_caps_from_parent() -> None:
    # Only the token windows are subscription-specific; the image/wire byte
    # caps are a property of the underlying model and must flow through
    # unchanged, not be overwritten by a stale local constant. A divergent
    # parent capability proves inheritance rather than a hardcoded match.
    api = models()["astra-6.0"].context[""]
    clamped = subscription_models()["astra-6.0"].context[""]
    assert clamped.max_image_edge_px == api.max_image_edge_px
    assert clamped.max_image_bytes == api.max_image_bytes
    assert clamped.max_request_bytes == api.max_request_bytes


# 1x1 transparent PNG -- smallest valid PNG ``image_lib.resize`` will accept.


def _make_jwt(payload: dict[str, object]) -> str:
    """Build an unsigned JWT; the provider only reads expiry claims leniently."""
    header_b = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=")
    body_b = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=")
    return f"{header_b.decode()}.{body_b.decode()}.sig"


# -- SIWC ID-token fixtures -------------------------------------------------

_RSA_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _jwks() -> dict[str, object]:
    key = RSAAlgorithm.to_jwk(_RSA_KEY.public_key(), as_dict=True)
    key["kid"] = "test-key"
    key["alg"] = "RS256"
    key["use"] = "sig"
    return {"keys": [key]}


def _id_claims(**overrides: object) -> dict[str, object]:
    now = int(time.time())
    claims: dict[str, object] = {
        "iss": siwc.ISSUER,
        "aud": _CLIENT_ID,
        "sub": "user-test",
        "exp": now + 3600,
        "iat": now,
        "email": "agent@example.com",
        "nonce": "nonce-1",
    }
    claims.update(overrides)
    return claims


def _id_token(claims: dict[str, object]) -> str:
    return jwt.encode(claims, _RSA_KEY, algorithm="RS256", headers={"kid": "test-key"})


# -- Login ------------------------------------------------------------------


def _printed_authorize_params(out: io.StringIO) -> dict[str, list[str]]:
    """Return the query params of the authorize URL in the login output.

    The login preamble also prints a Terms of Use URL, so a bare
    ``https://`` search is not enough -- match the SIWC authorize URL.
    """
    for match in re.finditer(r"https://\S+", out.getvalue()):
        url = match.group(0)
        if url.startswith(siwc.AUTHORIZE_URL):
            return parse_qs(urlparse(url).query)
    msg = "no SIWC authorize URL in the login output"
    raise AssertionError(msg)


def _manual_login_input(out: io.StringIO) -> str:
    """Answer the paste prompt with a callback matching the printed URL."""
    params = _printed_authorize_params(out)
    return (
        f"http://127.0.0.1:{siwc.MANUAL_CALLBACK_PORT}{siwc.REDIRECT_PATH}"
        f"?code=the-code&state={params['state'][0]}&client_id={_CLIENT_ID}"
    )


def _login_http(out: io.StringIO, *, scope: str) -> httpx2.Client:
    """Mock token endpoint whose ID token matches the printed login nonce."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        if request.method == "GET":
            return httpx2.Response(200, json=_jwks())
        nonce = _printed_authorize_params(out)["nonce"][0]
        return httpx2.Response(
            200,
            json={
                "access_token": _make_jwt({"exp": time.time() + 3600}),
                "token_type": "Bearer",
                "expires_in": 3600,
                "refresh_token": _FRESH_REFRESH,
                "id_token": _id_token(_id_claims(nonce=nonce)),
                "scope": scope,
            },
        )

    return httpx2.Client(transport=httpx2.MockTransport(handler))


def test_login_manual_first_sign_in_saves_the_siwc_record() -> None:
    """Manual first sign-in: dynamic registration, verified ID token, record."""
    out = io.StringIO()
    http = _login_http(out, scope=siwc.SCOPES)

    def fake_input(_prompt: object = "") -> str:
        return _manual_login_input(out)

    with (
        patch(
            "sagent.providers.openai.sub.httpx2.Client",
            return_value=http,
        ),
        patch("builtins.input", fake_input),
    ):
        record = OpenAISubscription.login(out, manual=True)

    printed = out.getvalue()
    params = _printed_authorize_params(out)
    assert params["redirect_uri"] == [
        f"http://127.0.0.1:{siwc.MANUAL_CALLBACK_PORT}{siwc.REDIRECT_PATH}",
    ]
    assert siwc.DYNAMIC_CLIENT_ID in printed
    assert "Authenticated as agent@example.com (plan usage enabled)." in printed
    assert record["client_id"] == _CLIENT_ID
    assert record.get("subject") == "user-test"
    on_disk = siwc.load_record(sub._default_credentials_path())
    assert on_disk.get("refresh_token") == _FRESH_REFRESH
    assert on_disk["scopes"] == siwc.SCOPES.split()


def test_login_without_plan_usage_saves_but_refuses_inference() -> None:
    """A declined plan-usage grant is retained, then reported (re-consent)."""
    out = io.StringIO()
    http = _login_http(out, scope="openid profile email offline_access")

    def fake_input(_prompt: object = "") -> str:
        return _manual_login_input(out)

    with (
        patch(
            "sagent.providers.openai.sub.httpx2.Client",
            return_value=http,
        ),
        patch("builtins.input", fake_input),
        pytest.raises(siwc.PlanUsageNotGrantedError, match="plan usage"),
    ):
        OpenAISubscription.login(out, manual=True)

    retained = siwc.load_record(sub._default_credentials_path())
    assert not siwc.plan_usage_enabled(retained)
    assert retained["client_id"] == _CLIENT_ID


# -- JWT helpers (lenient, unverified) --------------------------------------


def test_jwt_payload_round_trip() -> None:
    token = _make_jwt({"exp": 1234, "https://api.openai.com/auth": {"x": "y"}})
    assert _jwt_payload(token).get("exp") == 1234


def test_jwt_payload_short_token_returns_empty() -> None:
    assert _jwt_payload("only_one_part") == {}


def test_jwt_payload_invalid_base64_returns_empty() -> None:
    # Crafted body that decodes to non-JSON.
    bad = "h." + base64.urlsafe_b64encode(b"not json").rstrip(b"=").decode() + ".s"
    assert _jwt_payload(bad) == {}


def test_jwt_exp_extracts_value() -> None:
    token = _make_jwt({"exp": 9001})
    assert _jwt_exp(token) == 9001.0


def test_jwt_exp_missing_returns_zero() -> None:
    token = _make_jwt({})
    assert _jwt_exp(token) == 0.0


def _jwt_with_raw_payload(payload: object) -> str:
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=")
    return f"h.{body.decode()}.s"


@pytest.mark.parametrize("payload", [[], None, 42, "text"])
def test_jwt_helpers_tolerate_a_non_object_payload(payload: object) -> None:
    token = _jwt_with_raw_payload(payload)
    assert _jwt_payload(token) == {}
    assert _jwt_exp(token) == 0.0


@pytest.mark.parametrize("payload", [{"exp": "soon"}, {"exp": [1]}])
def test_jwt_helpers_tolerate_mistyped_claims(payload: dict[str, object]) -> None:
    assert _jwt_exp(_make_jwt(payload)) == 0.0


def test_load_with_a_non_object_jwt_payload_keeps_the_value_error_contract(
    tmp_path: Path,
) -> None:
    """A junk access-token payload reads as "no expiry", never a crash."""
    target = tmp_path / "auth.json"
    record = _make_record(_jwt_with_raw_payload([]))
    # With neither a readable ``exp`` nor a ``saved_at`` + ``expires_in``
    # fallback, the record's expiry is "unknown" (0.0 -> refresh on use).
    record.pop("saved_at")
    record.pop("expires_in")
    OpenAISubscription.save(record, path=target)
    loaded = OpenAISubscription.load(path=target)
    assert sub._record_expiry(loaded) == 0.0


def test_default_credentials_path_lives_under_the_data_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """SIWC records live in sagent's own store, not a vendor's directory."""
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert (
        sub._default_credentials_path()
        == tmp_path / "xdg" / "rekursiv-ai" / "sagent" / "auth" / "openai.json"
    )


# -- Credential records ------------------------------------------------------

# Test-only token values; not credentials. Held in module variables so the
# bandit-style S105/S106 rules don't fire on inline literals at constructor
# call sites.
_FAKE_ACCESS = ""
_FAKE_REFRESH = "rt-test"
_FRESH_REFRESH = "rt-fresh"
_STALE_REFRESH = "rt-stale"
_CLIENT_ID = "oaiapp_test"
_HOST_ID = "urn:uuid:test-host-id"


def _scopes() -> list[str]:
    """Return the protocol scope string as an ordinary string list."""
    return [str(scope) for scope in siwc.SCOPES.split()]


def _make_record(
    access: str,
    **overrides: object,
) -> siwc.SIWCCredentials:
    record: siwc.SIWCCredentials = {
        "issuer": siwc.ISSUER,
        "client_id": _CLIENT_ID,
        "ext_agent_host_id": _HOST_ID,
        "scopes": _scopes(),
        "subject": "user-test",
        "access_token": access,
        "refresh_token": _FAKE_REFRESH,
        "expires_in": 3600.0,
        "saved_at": siwc.record_saved_at(),
    }
    record.update(cast(Any, overrides))
    return record


def _make_tombstone() -> siwc.SIWCCredentials:
    """Build the tokenless record used by sign-out tests."""
    return {
        "issuer": siwc.ISSUER,
        "client_id": _CLIENT_ID,
        "ext_agent_host_id": _HOST_ID,
        "scopes": _scopes(),
        "signed_out": True,
    }


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    access = _make_jwt({"exp": time.time() + 3600})
    target = tmp_path / "auth.json"
    OpenAISubscription.save(
        _make_record(access, email="agent@example.com"),
        path=target,
    )
    loaded = OpenAISubscription.load(path=target)
    assert loaded.get("access_token") == access
    assert loaded.get("refresh_token") == _FAKE_REFRESH
    assert loaded["client_id"] == _CLIENT_ID
    assert loaded["scopes"] == siwc.SCOPES.split()
    assert loaded.get("email") == "agent@example.com"


def test_default_credential_io_honors_xdg_data_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The unnamed slot lives under ``XDG_DATA_HOME``."""
    xdg = tmp_path / "custom-xdg"
    monkeypatch.setenv("XDG_DATA_HOME", str(xdg))
    access = _make_jwt({"exp": time.time() + 3600})

    OpenAISubscription.save(_make_record(access))
    loaded = OpenAISubscription.load()

    assert (xdg / "rekursiv-ai" / "sagent" / "auth" / "openai.json").exists()
    assert loaded.get("access_token") == access


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        OpenAISubscription.load(path=tmp_path / "nope.json")


def test_load_record_without_client_id_raises_value_error(tmp_path: Path) -> None:
    target = tmp_path / "auth.json"
    target.write_text(json.dumps({"scopes": ["openid"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="client_id"):
        OpenAISubscription.load(path=target)


@pytest.mark.parametrize("payload", [[], None, "not an object"])
def test_load_non_object_json_raises_value_error(
    tmp_path: Path,
    payload: object,
) -> None:
    target = tmp_path / "auth.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        OpenAISubscription.load(path=target)


def test_save_and_load_preserve_the_id_token(tmp_path: Path) -> None:
    target = tmp_path / "auth.json"
    record = _make_record(_make_jwt({"exp": time.time() + 3600}))
    record["id_token"] = _id_token(_id_claims())
    OpenAISubscription.save(record, path=target)
    assert OpenAISubscription.load(path=target).get("id_token") == record.get(
        "id_token"
    )


def test_load_signed_out_record_raises(tmp_path: Path) -> None:
    target = tmp_path / "auth.json"
    siwc.save_record(_make_tombstone(), target)
    with pytest.raises(ValueError, match="signed-out"):
        OpenAISubscription.load(path=target)


class TestFromCredentials:
    def test_missing_record_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            OpenAISubscription.from_credentials()

    def test_signed_out_record_is_rejected(self) -> None:
        siwc.save_record(_make_tombstone(), sub._default_credentials_path())
        with pytest.raises(ValueError, match="signed-out"):
            OpenAISubscription.from_credentials()

    def test_a_record_without_plan_usage_is_rejected(self) -> None:
        access = _make_jwt({"exp": time.time() + 3600})
        OpenAISubscription.save(
            _make_record(
                access,
                scopes=["openid", "profile", "email", "offline_access"],
            ),
        )
        with pytest.raises(siwc.PlanUsageNotGrantedError, match="plan usage"):
            OpenAISubscription.from_credentials()

    def test_a_complete_record_builds_a_ready_provider(self) -> None:
        access = _make_jwt({"exp": time.time() + 3600})
        OpenAISubscription.save(_make_record(access, email="agent@example.com"))
        provider = OpenAISubscription.from_credentials()
        assert provider._client_id == _CLIENT_ID
        assert provider._subject == "user-test"
        assert provider._email == "agent@example.com"
        assert not provider.expired


class TestLogout:
    @pytest.mark.anyio
    async def test_logout_revokes_and_leaves_a_tombstone(self) -> None:
        OpenAISubscription.save(_make_record(_make_jwt({"exp": time.time() + 3600})))

        def handler(request: httpx2.Request) -> httpx2.Response:
            if request.method == "GET":
                return httpx2.Response(
                    200,
                    json={
                        "issuer": siwc.ISSUER,
                        "revocation_endpoint": f"{siwc.ISSUER}/oauth/revoke",
                    },
                )
            return httpx2.Response(200)

        http = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
        out = io.StringIO()
        with patch(
            "sagent.providers.openai.sub.httpx2.AsyncClient",
            return_value=http,
        ):
            await OpenAISubscription.logout(output=out)

        tombstone = siwc.load_record(sub._default_credentials_path())
        assert siwc.is_signed_out(tombstone)
        assert tombstone["client_id"] == _CLIENT_ID
        assert "access_token" not in tombstone
        assert "Signed out" in out.getvalue()

    @pytest.mark.anyio
    async def test_logout_without_a_record_reports_nothing_to_do(self) -> None:
        out = io.StringIO()
        await OpenAISubscription.logout(output=out)
        assert "No saved sign-in" in out.getvalue()

    @pytest.mark.anyio
    async def test_an_unconfirmed_revocation_still_clears_local_tokens(
        self,
    ) -> None:
        OpenAISubscription.save(_make_record(_make_jwt({"exp": time.time() + 3600})))

        def handler(request: httpx2.Request) -> httpx2.Response:
            if request.method == "GET":
                return httpx2.Response(
                    200,
                    json={
                        "issuer": siwc.ISSUER,
                        "revocation_endpoint": f"{siwc.ISSUER}/oauth/revoke",
                    },
                )
            return httpx2.Response(400, text="bad request")

        http = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
        out = io.StringIO()
        with patch(
            "sagent.providers.openai.sub.httpx2.AsyncClient",
            return_value=http,
        ):
            await OpenAISubscription.logout(output=out)

        assert "Warning" in out.getvalue()
        assert siwc.is_signed_out(siwc.load_record(sub._default_credentials_path()))


# -- Provider fixtures -------------------------------------------------------


def _make_provider(*, expires_at: float = 9999.0) -> OpenAISubscription:
    return OpenAISubscription(
        access_token=_FAKE_ACCESS or _make_jwt({"exp": expires_at}),
        refresh_token=_FAKE_REFRESH,
        client_id=_CLIENT_ID,
        expires_at=expires_at,
        scopes=_scopes(),
        subject="user-test",
    )


def _patch_cred_path(path: Path) -> Any:  # noqa: ANN401 -- unittest.mock patchers are dynamically typed.
    """Route the unnamed credential slot at ``path``."""
    return patch.object(sub, "_default_credentials_path", lambda: path)


def test_subscription_from_key_returns_plain_openai() -> None:
    """``from_key`` delegates to the API-key ``OpenAI`` class."""
    p = OpenAISubscription.from_key("sk-test")
    assert not isinstance(p, OpenAISubscription)


@pytest.mark.parametrize("subscription", [False, True])
def test_model_accepts_keyword_model_id(subscription: bool) -> None:
    provider = _make_provider() if subscription else OpenAI.from_key("test-key")
    assert provider.model(model_id="gpt-5.5").capability.model_id == "gpt-5.5"


def test_subscription_model_unknown_id_raises() -> None:
    p = _make_provider()
    with pytest.raises(ValueError, match="Unknown model"):
        _ = p.model("not-a-model")


def test_subscription_model_uses_default_when_unset() -> None:
    m = _make_provider().model()
    assert (
        m.capability.model_id
        == OpenAISubscription.catalog.resolve("default")[0].model_id
    )


def test_subscription_default_model_is_openai_default_without_1m() -> None:
    """Sub default = the API default's base id (``+1m`` is not in the catalog)."""
    assert OpenAISubscription.catalog.resolve("default")[0].model_id == "sol-6.1"
    assert OpenAI.catalog.resolve("default")[0].model_id == "sol-6.1"
    # The narrowed default must resolve against the narrowed catalog.
    assert "default" in OpenAISubscription.catalog.models


def test_subscription_default_utility_model_inherits_from_openai() -> None:
    assert (
        OpenAISubscription.catalog.resolve("utility")[0].model_id
        == OpenAI.catalog.resolve("utility")[0].model_id
    )


def test_subscription_utility_model_uses_utility_default() -> None:
    m = _make_provider().model("utility")
    assert m.capability.model_id == "luna-6.0"


def test_subscription_model_clamps_against_wire_contract() -> None:
    m = _make_provider().model("gpt-5.5")
    # KNOWN_MODELS is clamped via ``_subscription_profile``.
    assert m.limits.max_request_tokens == 272_000
    assert m.limits.max_response_tokens == 32_000


def test_subscription_rejects_1m_ids() -> None:
    """``+1m`` buys nothing under the wire contract, so it is not a known id."""
    p = _make_provider()
    with pytest.raises(ValueError, match=r"has no \+1m context"):
        _ = p.model("sol-5.6+1m")
    assert not any(name.endswith("+1m") for name in OpenAISubscription.catalog.models)


def test_subscription_model_supports_thinking_via_reasoning_effort() -> None:
    m = _make_provider().model("gpt-5.5")
    assert m.capability.thinking.budget == frozenset({"none", "auto"})
    assert m.capability.account_auth is True


def test_subscription_effort_matches_api_key_path() -> None:
    """Both OpenAI transports must agree on which ids are reasoning models.

    They share one capability catalog, so a divergence is a contract bug.
    Prefix guessing (which over-matched a hypothetical ``omni-*``) is gone;
    an id absent from the catalog is not resolvable on either path.
    """
    sub_p = _make_provider()
    api_p = OpenAI.from_key("k")
    for model_id in ("o1", "o3-mini", "gpt-5.5"):
        assert (
            sub_p.model(model_id).capability.thinking.effort
            == api_p.model(model_id).capability.thinking.effort
        )
    for provider in (sub_p, api_p):
        with pytest.raises(ValueError, match="Unknown model"):
            _ = provider.model("omni-foo")


def test_subscription_valid_service_tiers_priority_only() -> None:
    # ``/fast`` sets service_tier="priority"; an unqualified request still
    # sends the vendor default, so ``auto`` stays selectable and nothing else.
    m = _make_provider().model("gpt-5.5")
    assert m.capability.service_tier == frozenset({"auto", "priority"})


@pytest.mark.anyio
async def test_subscription_provider_close_releases_sdk() -> None:
    """Model teardown preserves the SDK until its owning provider closes."""
    provider = _make_provider()
    model = provider.model("gpt-5.5")
    sdk = MagicMock()
    sdk.close = AsyncMock()
    provider._authed.set((cast(openai.AsyncOpenAI, sdk), "dummy-token"))

    await model.close()
    sdk.close.assert_not_awaited()
    assert provider._authed.peek() == (sdk, "dummy-token")

    await provider.close_sdk()
    sdk.close.assert_awaited_once()
    assert provider._authed.peek() is None


def test_subscription_priority_tier_reaches_the_wire() -> None:
    m = _make_provider().model("gpt-5.5")
    m._settings = ModelSettings(capability=m.capability, service_tier="priority")
    assert m._effective_service_tier() == "priority"


def test_subscription_omits_the_default_tier() -> None:
    """``auto`` is "let the vendor pick"; sending it pins nothing."""
    assert _make_provider().model("gpt-5.5")._effective_service_tier() is None


class _StatusError(Exception):
    """Error carrying a typed ``status_code`` (what ``error_status_code`` reads)."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


def test_subscription_context_overflow_detection() -> None:
    m = _make_provider().model("gpt-5.5")
    assert m.is_context_overflow(RuntimeError("context_length_exceeded")) is True
    assert m.is_context_overflow(RuntimeError("exceeds the context window")) is True
    assert m.is_context_overflow(RuntimeError("other failure")) is False


def test_subscription_per_item_string_cap_is_context_overflow() -> None:
    """The per-item string cap must route to the compactor's shrink path.

    Session ``190b6baec7ed`` wedged here: an 11 MB tool result tripped
    OpenAI's 10 MiB per-ITEM cap, the phrase matched no classifier, and
    ``stream`` re-raised the raw ``BadRequestError`` (``sub.py:1054``).
    That is neither ``RequestTooLargeError`` nor ``PromptTooLongError``,
    so ``_shrink_groups_for_compaction`` never ran -- and because the
    compactor's own request carries the same item, every retry (and every
    later turn, and ``/compact``) failed identically until the session died.

    Classifying it as token overflow reaches the shrink-and-retry that
    caps oversized tool results, which is what actually clears it.
    """
    m = _make_provider().model("gpt-5.5")
    err = _StatusError(PER_ITEM_STRING_CAP_BODY, 400)
    assert m.is_context_overflow(err) is True


def test_subscription_byte_limit_not_context_overflow() -> None:
    """A 413 byte limit must not classify as token-context overflow.

    Uniform with the compat/google models: byte overflow routes to
    byte-overflow recovery (shed attachment bytes), never to the
    ``/model`` larger-window remediation.
    """
    m = _make_provider().model("gpt-5.5")
    err = _StatusError("Request entity too large", 413)
    assert m.is_context_overflow(err) is False


def test_subscription_byte_413_with_window_phrase_is_not_context_overflow() -> None:
    """A 413 whose body also names the context window stays byte overflow.

    Discriminating case: the subscription classifier must consult the shared
    ``is_request_too_large`` guard (status 413 -> byte) rather than its own
    substring list, so it matches the compat/google siblings. Without the
    guard, the bare ``"exceeds the context window"`` substring would
    mis-route a 413 byte limit to token-overflow recovery.
    """
    m = _make_provider().model("gpt-5.5")
    err = _StatusError("413: request entity too large; exceeds the context window", 413)
    assert m.is_context_overflow(err) is False


def test_subscription_retryable_provider_error() -> None:
    m = _make_provider().model("gpt-5.5")
    assert m.is_retryable_provider_error(RuntimeError("You can retry your request"))
    assert not m.is_retryable_provider_error(RuntimeError("permanent failure"))


def test_subscription_expired_property_true_when_past() -> None:
    p = _make_provider(expires_at=0.0)
    assert p.expired is True


def test_subscription_expired_property_false_when_far_future() -> None:
    p = _make_provider(expires_at=9_999_999_999.0)
    assert p.expired is False


def _create_kwargs_for(
    *,
    model_id: str = "sol-5.6",
    effort: ThinkingEffort = "none",
) -> dict[str, object]:
    model = _make_provider().model(model_id)
    model.settings.thinking_effort = effort
    return dict(model._build_kwargs(ModelRequest(messages=[UserMessage(text="hi")])))


async def _look_up(*, term: str) -> str:
    """Look a term up."""
    return term


def _lookup_tool() -> Tool:
    return tool(_look_up, name="LookUp")


def test_a_request_with_tools_allows_parallel_tool_calls() -> None:
    model = _make_provider().model("sol-5.6")
    request = ModelRequest(
        messages=[UserMessage(text="hi")],
        tools=[_lookup_tool()],
    )

    body = dict(model._build_kwargs(request))

    assert body["parallel_tool_calls"] is True


def test_a_request_with_no_tools_says_nothing_about_parallel_calls() -> None:
    assert "parallel_tool_calls" not in _create_kwargs_for()


def _wire_effort_for(*, model_id: str, effort: ThinkingEffort) -> object:
    """Return the ``reasoning.effort`` value the request carried."""
    reasoning = (_create_kwargs_for(model_id=model_id, effort=effort))["reasoning"]
    assert isinstance(reasoning, dict)
    return cast(dict[str, object], reasoning)["effort"]


@pytest.mark.parametrize(
    ("effort", "wire_effort"),
    [("low", "low"), ("medium", "medium"), ("xhigh", "xhigh")],
)
def test_subscription_stream_maps_pre_56_effort_to_wire_vocabulary(
    effort: ThinkingEffort,
    wire_effort: str,
) -> None:
    """Subscription requests preserve the model's native effort vocabulary."""
    assert _wire_effort_for(model_id="gpt-5.5", effort=effort) == wire_effort


@pytest.mark.parametrize(
    ("effort", "wire_effort"),
    [("low", "low"), ("medium", "medium"), ("xhigh", "xhigh"), ("max", "max")],
)
def test_subscription_stream_preserves_gpt_56_effort(
    effort: ThinkingEffort,
    wire_effort: str,
) -> None:
    assert _wire_effort_for(model_id="sol-5.6", effort=effort) == wire_effort


def test_subscription_catalog_efforts_are_all_buildable() -> None:
    """Every effort the catalog offers must reach the wire."""
    model_id = "sol-5.6"
    capability = _make_provider().model(model_id).capability
    for effort in capability.thinking.effort - {"none"}:
        assert _wire_effort_for(
            model_id=model_id,
            effort=effort,
        ) == reasoning_effort(effort)


def test_subscription_stream_requests_reasoning_summary_when_thinking() -> None:
    """Thinking on must request a reasoning summary, else no text streams.

    The OpenAI Responses API only emits reasoning-text deltas when
    ``reasoning.summary`` is set; without it the model reasons silently
    and ``on_thinking`` never fires. Requesting thinking must therefore
    set ``summary`` so the reasoning surface is actually populated.
    """
    reasoning = (_create_kwargs_for(model_id="gpt-5.5", effort="medium"))["reasoning"]
    assert isinstance(reasoning, dict)
    assert cast(dict[str, object], reasoning)["summary"] == "auto"


def test_subscription_stream_requests_encrypted_reasoning_for_stateless_history() -> (
    None
):
    kwargs = _create_kwargs_for(effort="medium")
    assert kwargs["store"] is False
    assert kwargs["include"] == ["reasoning.encrypted_content"]
    reasoning = kwargs["reasoning"]
    assert isinstance(reasoning, dict)
    assert cast(dict[str, object], reasoning)["context"] == "all_turns"


def test_subscription_stream_omits_reasoning_object_without_thinking() -> None:
    """``none`` means "send no knob", not "send the wire's off value"."""
    kwargs = _create_kwargs_for(model_id="gpt-5.5")
    assert "reasoning" not in kwargs
    assert "include" not in kwargs


class TestHandleAuthError:
    """After ``/login`` writes fresh creds, the live provider must reload.

    Bug: ``do_login`` (``repl/run_repl.py``) calls
    ``provider.handle_auth_error()`` to hot-reload in-memory tokens
    from disk after a successful re-auth. Without this hook on
    :class:`OpenAISubscription`, the live instance keeps its stale
    ``_access_token`` / ``_refresh_token`` after ``/login``; the next
    model call either sends the stale access token (-> API 401) or
    posts the stale refresh token (-> 400 ->
    :class:`AuthRefreshError`). Either way ``/login`` is a no-op until
    process restart.
    """

    @pytest.mark.anyio
    async def test_reloads_from_disk(self, tmp_path: Path) -> None:
        """Fresh creds on disk replace stale in-memory creds."""
        cred_file = tmp_path / "auth.json"
        fresh_access = _make_jwt({"exp": time.time() + 3600})
        OpenAISubscription.save(
            _make_record(fresh_access, refresh_token=_FRESH_REFRESH),
            path=cred_file,
        )
        provider = OpenAISubscription(
            access_token=_make_jwt({"exp": 0.0}),
            refresh_token=_STALE_REFRESH,
            client_id=_CLIENT_ID,
            expires_at=0.0,
        )
        with _patch_cred_path(cred_file):
            await provider.handle_auth_error()
        assert provider._access_token == fresh_access
        assert provider._refresh_token == _FRESH_REFRESH

    @pytest.mark.anyio
    async def test_refreshes_when_adopted_disk_token_also_expired(
        self,
        tmp_path: Path,
    ) -> None:
        """A DIFFERENT but already-expired disk token must still force a refresh.

        On a 401, a sibling may have written a token whose own lifetime already
        lapsed. Adopting it and returning (no re-expiry check) hands the caller a
        stale token -> the retry 401s again. ``handle_auth_error`` must mirror
        ``_ensure_valid``: after adopting a fresher-looking disk token, refresh
        anyway if it is still expired.
        """
        cred_file = tmp_path / "auth.json"
        expired_disk = _make_jwt({"exp": time.time() - 50})
        OpenAISubscription.save(
            _make_record(expired_disk, refresh_token=_FRESH_REFRESH),
            path=cred_file,
        )
        provider = OpenAISubscription(
            access_token=_make_jwt({"exp": 0.0}),
            refresh_token=_STALE_REFRESH,
            client_id=_CLIENT_ID,
            expires_at=time.time() - 100,
        )
        refreshed: list[bool] = []

        async def _fake_refresh() -> None:
            refreshed.append(True)
            provider._access_token = _make_jwt({"exp": time.time() + 3600})
            provider._expires_at = time.time() + 3600

        with (
            patch.object(provider, "_refresh", _fake_refresh),
            _patch_cred_path(cred_file),
        ):
            await provider.handle_auth_error()

        assert refreshed == [True], "expired adopted disk token must trigger _refresh"
        assert not provider.expired

    @pytest.mark.anyio
    async def test_force_refresh_when_disk_same_even_if_clock_valid(
        self,
        tmp_path: Path,
    ) -> None:
        """A 401 on a clock-VALID token (same on disk) must STILL force a refresh.

        ``handle_auth_error`` runs because the server already rejected this exact
        token (revoked, or server/client clock skew). The local ``expired`` clock
        says "valid", but the server is authoritative -- trusting the clock and
        returning without refreshing leaves the caller retrying the same rejected
        token in an infinite 401 loop. Only a genuinely DIFFERENT disk token (a
        sibling's refresh) may be adopted without a network refresh.
        """
        cred_file = tmp_path / "auth.json"
        # `exp` far in the future means ``self.expired`` is False, yet the server 401'd.
        clock_valid = _make_jwt({"exp": time.time() + 3600})
        OpenAISubscription.save(
            _make_record(clock_valid),
            path=cred_file,
        )
        provider = OpenAISubscription(
            access_token=clock_valid,  # SAME as disk.
            refresh_token=_STALE_REFRESH,
            client_id=_CLIENT_ID,
            expires_at=time.time() + 3600,
        )
        refreshed: list[bool] = []

        async def _fake_refresh() -> None:
            refreshed.append(True)
            provider._access_token = _make_jwt({"exp": time.time() + 7200})

        with (
            patch.object(provider, "_refresh", _fake_refresh),
            _patch_cred_path(cred_file),
        ):
            await provider.handle_auth_error()

        assert refreshed == [True], (
            "a 401 on a clock-valid same-disk token must force _refresh, not return"
        )

    @pytest.mark.anyio
    async def test_force_refresh_when_disk_same(self, tmp_path: Path) -> None:
        """Disk matches in-memory -> fall through to network refresh."""
        in_memory_access = _make_jwt({"exp": 0.0})
        cred_file = tmp_path / "auth.json"
        OpenAISubscription.save(
            _make_record(in_memory_access),
            path=cred_file,
        )
        provider = OpenAISubscription(
            access_token=in_memory_access,
            refresh_token=_STALE_REFRESH,
            client_id=_CLIENT_ID,
            expires_at=0.0,
        )
        refreshed_access = _make_jwt({"exp": time.time() + 3600})
        http = _token_endpoint(refreshed_access)
        with (
            patch(
                "sagent.providers.openai.sub.httpx2.AsyncClient",
                return_value=http,
            ),
            _patch_cred_path(cred_file),
        ):
            await provider.handle_auth_error()
        assert provider._access_token == refreshed_access
        assert provider._refresh_token == _FRESH_REFRESH

    @pytest.mark.anyio
    async def test_force_refresh_when_disk_missing(self, tmp_path: Path) -> None:
        """Missing disk file -> caught -> fall through to network refresh."""
        provider = OpenAISubscription(
            access_token=_make_jwt({"exp": 0.0}),
            refresh_token=_STALE_REFRESH,
            client_id=_CLIENT_ID,
            expires_at=0.0,
        )
        refreshed_access = _make_jwt({"exp": time.time() + 3600})
        http = _token_endpoint(refreshed_access)
        with (
            patch(
                "sagent.providers.openai.sub.httpx2.AsyncClient",
                return_value=http,
            ),
            _patch_cred_path(tmp_path / "does_not_exist.json"),
        ):
            await provider.handle_auth_error()
        assert provider._access_token == refreshed_access
        assert provider._refresh_token == _FRESH_REFRESH


class TestEnsureValidRace:
    """``_ensure_valid`` must reload disk before refreshing.

    Multi-process race: process A's token expires; A refreshes,
    rotating the refresh_token and saving fresh creds to disk.
    Process B's in-memory refresh_token is now revoked. When B's
    token expires, ``_ensure_valid`` must check disk first; if disk
    has fresher creds, adopt them without posting B's stale
    refresh_token (which would 400). Mirrors ``handle_auth_error``'s
    disk-first pattern.
    """

    @pytest.mark.anyio
    async def test_reloads_from_disk_when_sibling_refreshed(
        self,
        tmp_path: Path,
    ) -> None:
        """Fresh disk creds preempt the network refresh entirely."""
        cred_file = tmp_path / "auth.json"
        fresh_access = _make_jwt({"exp": time.time() + 3600})
        OpenAISubscription.save(
            _make_record(fresh_access, refresh_token=_FRESH_REFRESH),
            path=cred_file,
        )
        provider = OpenAISubscription(
            access_token=_make_jwt({"exp": 0.0}),
            refresh_token=_STALE_REFRESH,
            client_id=_CLIENT_ID,
            expires_at=time.time() - 100,
        )

        # If ``_refresh`` is reached, the test fails fast -- the disk
        # reload should preempt it.
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(
            side_effect=AssertionError(
                "_ensure_valid called _refresh instead of reloading from disk",
            ),
        )
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        with (
            patch(
                "sagent.providers.openai.sub.httpx2.AsyncClient",
                return_value=mock_http,
            ),
            _patch_cred_path(cred_file),
        ):
            token = await provider._ensure_valid()

        assert token == fresh_access
        assert provider._access_token == fresh_access
        assert provider._refresh_token == _FRESH_REFRESH

    @pytest.mark.anyio
    async def test_refreshes_when_disk_token_also_expired(self, tmp_path: Path) -> None:
        """An adopted disk token that is ALSO expired must trigger a refresh.

        A sibling may have written a token whose own lifetime has already
        lapsed (its refresh aged out, or clock skew). Adopting it and returning
        it would 401 on the next call; ``_ensure_valid`` must fall through to
        ``_refresh`` instead of handing back the stale disk token.
        """
        cred_file = tmp_path / "auth.json"
        expired_disk = _make_jwt({"exp": time.time() - 50})
        OpenAISubscription.save(
            _make_record(expired_disk, refresh_token=_FRESH_REFRESH),
            path=cred_file,
        )
        provider = OpenAISubscription(
            access_token=_make_jwt({"exp": 0.0}),
            refresh_token=_STALE_REFRESH,
            client_id=_CLIENT_ID,
            expires_at=time.time() - 100,
        )
        refreshed: list[bool] = []

        async def _fake_refresh() -> None:
            refreshed.append(True)
            provider._access_token = _make_jwt({"exp": time.time() + 3600})
            provider._expires_at = time.time() + 3600

        with (
            patch.object(provider, "_refresh", _fake_refresh),
            _patch_cred_path(cred_file),
        ):
            token = await provider._ensure_valid()

        assert refreshed == [True], "expired disk token must trigger _refresh"
        assert token == provider._access_token
        assert not provider.expired

    @pytest.mark.anyio
    async def test_adopting_disk_token_closes_old_sdk(self, tmp_path: Path) -> None:
        """Adopting a sibling's fresher disk token must CLOSE the stale SDK.

        Dropping ``self._sdk`` without awaiting ``aclose`` orphans the old
        ``AsyncOpenAI`` HTTP client (its pooled connections leak until process
        exit). The credential-replacement path in ``get_sdk`` closes the old
        client; the disk-adopt path must do the same.
        """
        cred_file = tmp_path / "auth.json"
        fresh_access = _make_jwt({"exp": time.time() + 3600})
        OpenAISubscription.save(
            _make_record(fresh_access, refresh_token=_FRESH_REFRESH),
            path=cred_file,
        )
        provider = OpenAISubscription(
            access_token=_make_jwt({"exp": 0.0}),  # `expired` -> triggers adopt.
            refresh_token=_STALE_REFRESH,
            client_id=_CLIENT_ID,
            expires_at=0.0,
        )
        closed: list[bool] = []
        stale_sdk = MagicMock()
        stale_sdk.close = AsyncMock(side_effect=lambda: closed.append(True))
        provider._authed.set(
            (cast(openai.AsyncOpenAI, stale_sdk), _make_jwt({"exp": 0.0})),
        )

        with _patch_cred_path(cred_file):
            token = await provider._ensure_valid()

        assert token == fresh_access
        assert closed == [True], "adopting a disk token must close the old SDK client"
        assert provider._authed.peek() is None


def _token_endpoint(access_token: str, *, expires_in: int = 3600) -> AsyncMock:
    """Return a patched ``httpx2.AsyncClient`` whose POST issues ``access_token``."""
    response = MagicMock()
    response.status_code = 200
    response.json = MagicMock(
        return_value={
            "access_token": access_token,
            "token_type": "Bearer",
            "refresh_token": _FRESH_REFRESH,
            "expires_in": expires_in,
        },
    )
    http = AsyncMock()
    http.post = AsyncMock(return_value=response)
    http.__aenter__ = AsyncMock(return_value=http)
    http.__aexit__ = AsyncMock(return_value=False)
    return http


class TestRefreshLifecycle:
    """Every refresh runs under the cross-process lock, off-loop, with an expiry."""

    @pytest.mark.anyio
    async def test_get_sdk_never_refreshes_outside_the_file_lock(self) -> None:
        provider = _make_provider(expires_at=time.time() + 3600)
        held: list[bool] = []
        ensure_valid = provider._ensure_valid

        @contextlib.asynccontextmanager
        async def _recording_lock(path: Path) -> AsyncGenerator[None]:
            del path
            held.append(True)
            try:
                yield
            finally:
                held.pop()

        async def _expire_after_validation() -> str:
            token = await ensure_valid()
            # The deadline crosses the buffer while ``get_sdk`` waits on its lock.
            provider._expires_at = 0.0
            return token

        async def _refresh() -> None:
            assert held, "_refresh ran without the cross-process credential lock"

        with (
            patch.object(sub, "credential_file_lock", _recording_lock),
            patch.object(provider, "_ensure_valid", _expire_after_validation),
            patch.object(provider, "_refresh", _refresh),
        ):
            sdk = await provider.get_sdk()
        await sdk.close()

    @pytest.mark.anyio
    async def test_refresh_reads_and_writes_credentials_off_the_event_loop(
        self,
    ) -> None:
        provider = _make_provider(expires_at=0.0)
        loop_thread = threading.get_ident()
        io_threads: list[int] = []
        real_load = siwc.load_record
        real_save = siwc.save_record

        def _load(path: Path) -> siwc.SIWCCredentials:
            io_threads.append(threading.get_ident())
            return real_load(path)

        def _save(record: siwc.SIWCCredentials, path: Path) -> None:
            io_threads.append(threading.get_ident())
            real_save(record, path)

        with (
            patch(
                "sagent.providers.openai.sub.httpx2.AsyncClient",
                return_value=_token_endpoint(_make_jwt({"exp": time.time() + 3600})),
            ),
            patch.object(siwc, "load_record", _load),
            patch.object(siwc, "save_record", _save),
        ):
            await provider._refresh()

        assert io_threads, "the refresh should have touched the record on disk"
        assert loop_thread not in io_threads

    @pytest.mark.anyio
    async def test_refresh_posts_the_grant_and_persists_into_the_account_slot(
        self,
    ) -> None:
        """The grant is the SIWC wire contract; the write keeps disk-only fields."""
        provider = OpenAISubscription(
            access_token=_make_jwt({"exp": 0.0}),
            refresh_token=_STALE_REFRESH,
            client_id=_CLIENT_ID,
            expires_at=0.0,
            scopes=_scopes(),
            subject="user-test",
            account="work",
        )
        slot = credentials_path(sub._default_credentials_path(), "work")
        OpenAISubscription.save(
            _make_record(
                _make_jwt({"exp": 0.0}),
                refresh_token=_STALE_REFRESH,
                email="disk@example.com",
            ),
            path=slot,
        )
        fresh = _make_jwt({"exp": time.time() + 3600})
        http = _token_endpoint(fresh)
        with patch(
            "sagent.providers.openai.sub.httpx2.AsyncClient",
            return_value=http,
        ):
            await provider._refresh()

        http.post.assert_awaited_once_with(
            siwc.TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": _STALE_REFRESH,
                "client_id": _CLIENT_ID,
                "resource": siwc.RESOURCE,
            },
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
            timeout=15.0,
        )
        record = siwc.load_record(slot)
        assert record.get("access_token") == fresh
        assert record.get("refresh_token") == _FRESH_REFRESH
        # Identity fields only present on disk survive the merge.
        assert record.get("email") == "disk@example.com"
        assert record.get("subject") == "user-test"

    @pytest.mark.anyio
    @pytest.mark.parametrize("corrupt", [False, True])
    @pytest.mark.parametrize("account", [None, "work"])
    async def test_refresh_writes_a_fresh_record_when_disk_has_none(
        self,
        caplog: pytest.LogCaptureFixture,
        corrupt: bool,
        account: str | None,
    ) -> None:
        provider = OpenAISubscription(
            access_token=_make_jwt({"exp": 0.0}),
            refresh_token=_FAKE_REFRESH,
            client_id=_CLIENT_ID,
            expires_at=0.0,
            scopes=_scopes(),
            account=account,
        )
        slot = credentials_path(sub._default_credentials_path(), account)
        if corrupt:
            slot.parent.mkdir(parents=True, exist_ok=True)
            slot.write_text('{"tokens": {}}', encoding="utf-8")
        fresh = _make_jwt({"exp": time.time() + 3600})
        with (
            patch(
                "sagent.providers.openai.sub.httpx2.AsyncClient",
                return_value=_token_endpoint(fresh),
            ),
            caplog.at_level("WARNING"),
        ):
            await provider._refresh()

        record = siwc.load_record(slot)
        assert record.get("access_token") == fresh
        assert record.get("refresh_token") == _FRESH_REFRESH
        warned = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert warned == [
            f"SIWC record at {slot} was unreadable; rewriting from memory",
        ]

    @pytest.mark.anyio
    async def test_a_token_without_exp_takes_its_expiry_from_expires_in(
        self,
    ) -> None:
        provider = _make_provider(expires_at=0.0)
        http = _token_endpoint(_make_jwt({}), expires_in=3600)
        with patch(
            "sagent.providers.openai.sub.httpx2.AsyncClient",
            return_value=http,
        ):
            first = await provider.get_sdk()
            second = await provider.get_sdk()
        await first.close()

        assert http.post.await_count == 1
        assert first is second
        assert not provider.expired

    @pytest.mark.anyio
    async def test_a_token_with_no_expiry_at_all_is_rejected(self) -> None:
        provider = _make_provider(expires_at=0.0)
        http = _token_endpoint(_make_jwt({}), expires_in=0)
        with (
            patch(
                "sagent.providers.openai.sub.httpx2.AsyncClient",
                return_value=http,
            ),
            pytest.raises(AuthRefreshError, match="invalid token response"),
        ):
            await provider._refresh()


class TestPlanUsageErrors:
    """Documented SIWC plan-usage codes map to polished recovery text."""

    def _failing_sdk(self, error: Exception) -> MagicMock:
        sdk = MagicMock()
        sdk.responses = MagicMock()
        sdk.responses.with_raw_response.create = AsyncMock(side_effect=error)
        return sdk

    @pytest.mark.anyio
    async def test_documented_code_maps_to_subscription_usage_error(self) -> None:
        provider = _make_provider(expires_at=time.time() + 3600)
        request = httpx2.Request("POST", "https://api.openai.com/v1/responses")
        response = httpx2.Response(429, request=request)
        err = openai.RateLimitError(
            "Rate limited: subscription_sharing_usage_limit_exceeded",
            response=response,
            body=None,
        )
        with patch.object(
            provider,
            "get_sdk",
            AsyncMock(return_value=self._failing_sdk(err)),
        ):
            model = provider.model("gpt-5.5")
            with pytest.raises(siwc.SubscriptionUsageError, match="usage limit"):
                await model.stream(ModelRequest(messages=[UserMessage(text="hi")]))

    @pytest.mark.anyio
    async def test_an_unmapped_status_error_passes_through(self) -> None:
        provider = _make_provider(expires_at=time.time() + 3600)
        request = httpx2.Request("POST", "https://api.openai.com/v1/responses")
        response = httpx2.Response(400, request=request)
        err = openai.BadRequestError(
            "Some unrelated 400 failure",
            response=response,
            body=None,
        )
        with patch.object(
            provider,
            "get_sdk",
            AsyncMock(return_value=self._failing_sdk(err)),
        ):
            model = provider.model("gpt-5.5")
            with pytest.raises(openai.BadRequestError):
                await model.stream(ModelRequest(messages=[UserMessage(text="hi")]))


class TestSdkClient:
    @pytest.mark.anyio
    async def test_get_sdk_targets_the_siwc_inference_endpoint(self) -> None:
        """Bearer-only against the documented endpoint; Codex headers are gone."""
        provider = _make_provider(expires_at=time.time() + 3600)
        sdk = await provider.get_sdk()
        try:
            assert str(sdk.base_url) == siwc.RESOURCE + "/"
            assert sdk.api_key == provider._access_token
        finally:
            await provider.close_sdk()


class TestStreamResponseNotRead:
    """Unread SDK streaming errors must not leak raw httpx2 exceptions."""

    @pytest.mark.anyio
    async def test_create_response_not_read_is_user_facing(self) -> None:
        provider = _make_provider(expires_at=time.time() + 3600)
        sdk = MagicMock()
        sdk.responses = MagicMock()
        sdk.responses.with_raw_response.create = AsyncMock(
            side_effect=httpx2.ResponseNotRead(),
        )

        with patch.object(provider, "get_sdk", AsyncMock(return_value=sdk)):
            model = provider.model("gpt-5.5")
            with pytest.raises(StreamingResponseNotReadError) as raised:
                await model.stream(ModelRequest(messages=[UserMessage(text="hi")]))

        assert not isinstance(raised.value, httpx2.ResponseNotRead)
        assert "OpenAI streaming request failed" in str(raised.value)

    @pytest.mark.anyio
    async def test_wrapped_response_not_read_is_user_facing(self) -> None:
        provider = _make_provider(expires_at=time.time() + 3600)
        sdk = MagicMock()
        sdk.responses = MagicMock()
        err = RuntimeError("SDK failed")
        err.__cause__ = httpx2.ResponseNotRead()
        sdk.responses.with_raw_response.create = AsyncMock(side_effect=err)

        with patch.object(provider, "get_sdk", AsyncMock(return_value=sdk)):
            model = provider.model("gpt-5.5")
            with pytest.raises(StreamingResponseNotReadError) as raised:
                await model.stream(ModelRequest(messages=[UserMessage(text="hi")]))

        assert "OpenAI streaming request failed" in str(raised.value)

    def test_response_not_read_context_is_detected(self) -> None:
        err = RuntimeError("SDK failed")
        err.__context__ = httpx2.ResponseNotRead()

        assert find_response_not_read(err) is not None


class TestStreamAuthRetry:
    """Mid-call 401 must trigger ``handle_auth_error`` + one-shot retry.

    Bug: a stale in-memory bearer (rotated server-side between our
    local expiry check and the request arriving) surfaces as
    ``openai.AuthenticationError`` from ``sdk.responses.with_raw_response.create``.
    Without the catch + reload + retry wrapper, the error bubbles
    raw to the user; with it, the runtime reloads disk creds (or
    force-refreshes) and retries once before giving up.
    """

    @pytest.mark.anyio
    async def test_auth_error_triggers_reload_and_retries_once(self) -> None:
        provider = _make_provider(expires_at=time.time() + 3600)
        request = httpx2.Request("POST", "https://api.openai.com/v1/responses")
        response = httpx2.Response(401, request=request)
        auth_err = openai.AuthenticationError(
            "Unauthorized",
            response=response,
            body=None,
        )
        # Return DIFFERENT SDKs from get_sdk so the test can prove the
        # retry call landed on a freshly-built SDK (with a rotated
        # bearer baked into default_headers), not the cached one.
        sdk_stale = MagicMock()
        sdk_stale.responses = MagicMock()
        sdk_stale.responses.with_raw_response.create = AsyncMock(side_effect=auth_err)
        sdk_fresh = MagicMock()
        sdk_fresh.responses = MagicMock()
        sdk_fresh.responses.with_raw_response.create = AsyncMock(side_effect=auth_err)
        get_sdk = AsyncMock(side_effect=[sdk_stale, sdk_fresh])
        with (
            patch.object(provider, "get_sdk", get_sdk),
            patch.object(provider, "handle_auth_error", AsyncMock()) as ha,
        ):
            model = provider.model("gpt-5.5")
            req = ModelRequest(messages=[UserMessage(text="hi")])
            with pytest.raises(openai.AuthenticationError):
                await model.stream(req)
        ha.assert_awaited_once()
        # Each SDK saw exactly one create call: the original attempt on
        # the stale SDK, the retry on the freshly-fetched one.
        assert sdk_stale.responses.with_raw_response.create.await_count == 1
        assert sdk_fresh.responses.with_raw_response.create.await_count == 1
        assert get_sdk.await_count == 2


class TestRefreshErrors:
    """``_refresh`` must surface auth failures as :class:`AuthRefreshError`.

    SIWC's ``/api/accounts/oauth/token`` endpoint returns 400/401 when
    the refresh token has been rotated, revoked, or expired. The raw
    ``httpx2.HTTPStatusError`` is useless to the user -- it leaks the
    OAuth URL into the terminal. Convert it into a typed, user-facing
    error with actionable text so the renderer can present "Run /login"
    without dumping a traceback.
    """

    @pytest.mark.anyio
    @pytest.mark.parametrize("status", [400, 401])
    async def test_refresh_4xx_raises_auth_refresh_error(self, status: int) -> None:
        """400/401 on the token endpoint -> :class:`AuthRefreshError`."""
        provider = _make_provider(expires_at=0.0)
        request = httpx2.Request("POST", siwc.TOKEN_URL)
        response = httpx2.Response(status, request=request, text="invalid_grant")
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=response)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        with (
            patch(
                "sagent.providers.openai.sub.httpx2.AsyncClient",
                return_value=mock_http,
            ),
            pytest.raises(AuthRefreshError) as excinfo,
        ):
            await provider._refresh()

        assert str(excinfo.value) == (
            "OpenAI subscription session expired or revoked. Run /login to "
            "re-authenticate, or /model to switch providers."
        )

    @pytest.mark.anyio
    async def test_refresh_overwrites_missing_field_creds_file(
        self,
        tmp_path: Path,
    ) -> None:
        """A creds file with valid JSON but missing fields must not crash refresh.

        ``siwc.load_record`` raises ``ValueError`` on ``{"tokens": {}}``.
        ``_refresh`` already holds fresh tokens by then and is meant to
        rewrite the corrupt file from memory, so the load failure is
        caught and logged, not propagated.
        """
        cred_file = tmp_path / "auth.json"
        cred_file.write_text('{"tokens": {}}', encoding="utf-8")
        provider = _make_provider(expires_at=0.0)
        refreshed_access = _make_jwt({"exp": time.time() + 3600})
        http = _token_endpoint(refreshed_access)
        with (
            patch(
                "sagent.providers.openai.sub.httpx2.AsyncClient",
                return_value=http,
            ),
            _patch_cred_path(cred_file),
        ):
            await provider._refresh()  # Must not raise ValueError.
        assert provider._access_token == refreshed_access
        # The corrupt file was overwritten with a well-formed record.
        reloaded = OpenAISubscription.load(path=cred_file)
        assert reloaded.get("access_token") == refreshed_access

    @pytest.mark.anyio
    @pytest.mark.parametrize("status", [500, 502, 503])
    async def test_refresh_5xx_does_not_raise_auth_refresh_error(
        self,
        status: int,
    ) -> None:
        """Server-side failures bubble as plain ``httpx2.HTTPStatusError``."""
        provider = _make_provider(expires_at=0.0)
        request = httpx2.Request("POST", siwc.TOKEN_URL)
        response = httpx2.Response(status, request=request, text="upstream down")
        mock_http = AsyncMock()
        mock_http.post = AsyncMock(return_value=response)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)
        with (
            patch(
                "sagent.providers.openai.sub.httpx2.AsyncClient",
                return_value=mock_http,
            ),
            pytest.raises(httpx2.HTTPStatusError),
        ):
            await provider._refresh()


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
