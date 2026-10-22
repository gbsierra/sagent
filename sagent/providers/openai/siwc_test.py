"""Unit tests for the Sign in with ChatGPT (SIWC) protocol module.

All token-endpoint and JWKS traffic runs through ``httpx2.MockTransport``
handlers; ID tokens are signed with a locally generated RSA key so
``validate_id_token`` exercises real signature verification.
"""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qs, urlparse

import json
import time

from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import (
    RSAAlgorithm,  # ty: ignore[possibly-missing-import] -- pyjwt hides RSAAlgorithm behind its optional cryptography extra.
)

import httpx2
import jwt
import pytest

from sagent.providers.openai import siwc
from sagent.types.exceptions import AuthRefreshError


_RSA_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_KID = "test-key-1"
_AUDIENCE = "oaiapp_client123"
_HOST_ID = "urn:uuid:00000000-0000-4000-8000-000000000000"

type Handler = Callable[[httpx2.Request], httpx2.Response]


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    siwc._JWKS_CACHE.clear()
    yield
    siwc._JWKS_CACHE.clear()


def _jwk(kid: str = _KID) -> dict[str, object]:
    key = cast(
        dict[str, object],
        RSAAlgorithm.to_jwk(_RSA_KEY.public_key(), as_dict=True),
    )
    key["kid"] = kid
    key["alg"] = "RS256"
    key["use"] = "sig"
    return key


def _jwks(kid: str = _KID) -> dict[str, object]:
    return {"keys": [_jwk(kid)]}


def _claims(**overrides: object) -> dict[str, object]:
    now = int(time.time())
    claims: dict[str, object] = {
        "iss": siwc.ISSUER,
        "aud": _AUDIENCE,
        "sub": "user_abc",
        "exp": now + 3600,
        "iat": now,
        "email": "agent@example.com",
        "nonce": "nonce-1",
    }
    claims.update(overrides)
    return claims


def _scopes() -> list[str]:
    """Return the protocol scope string as an ordinary string list."""
    return [str(scope) for scope in siwc.SCOPES.split()]


def _id_token(claims: dict[str, object], kid: str = _KID) -> str:
    return jwt.encode(claims, _RSA_KEY, algorithm="RS256", headers={"kid": kid})


def _make_record(**overrides: object) -> siwc.SIWCCredentials:
    record: siwc.SIWCCredentials = {
        "issuer": siwc.ISSUER,
        "client_id": _AUDIENCE,
        "ext_agent_host_id": _HOST_ID,
        "scopes": _scopes(),
        "subject": "user_abc",
        "access_token": _id_token(_claims()),
        "refresh_token": "rt_live",
        "expires_in": 3600.0,
    }
    record.update(cast(Any, overrides))
    return record


def _client(handler: Handler) -> httpx2.Client:
    return httpx2.Client(transport=httpx2.MockTransport(handler))


def _async_client(handler: Handler) -> httpx2.AsyncClient:
    return httpx2.AsyncClient(transport=httpx2.MockTransport(handler))


def _form(request: httpx2.Request) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(request.content.decode("utf-8")).items()}


def _query(url: str) -> dict[str, list[str]]:
    return parse_qs(urlparse(url).query)


class TestAuthorizeUrl:
    def test_first_sign_in_registers_a_dynamic_client(self) -> None:
        url = siwc.authorize_url(
            state="st-1",
            nonce="nn-1",
            code_challenge="challenge",
            redirect_uri="http://127.0.0.1:50505/auth/callback",
            client_id=siwc.DYNAMIC_CLIENT_ID,
            agent_name_hint=siwc.AGENT_NAME_HINT,
            ext_agent_host_id=_HOST_ID,
        )
        assert url.startswith(siwc.AUTHORIZE_URL + "?")
        q = _query(url)
        assert q["client_id"] == [siwc.DYNAMIC_CLIENT_ID]
        assert q["agent_name_hint"] == [siwc.AGENT_NAME_HINT]
        assert q["redirect_uri"] == ["http://127.0.0.1:50505/auth/callback"]
        assert q["response_type"] == ["code"]
        assert q["scope"] == [siwc.SCOPES]
        assert q["resource"] == [siwc.RESOURCE]
        assert q["code_challenge"] == ["challenge"]
        assert q["code_challenge_method"] == ["S256"]
        assert q["state"] == ["st-1"]
        assert q["nonce"] == ["nn-1"]
        assert q["ext_agent_host_id"] == [_HOST_ID]
        assert "prompt" not in q
        assert "login_hint" not in q
        # Secrets never travel in URLs: no id_token_hint, ever.
        assert "id_token_hint" not in q

    def test_returning_sign_in_reuses_the_issued_client_id(self) -> None:
        url = siwc.authorize_url(
            state="st-2",
            nonce="nn-2",
            code_challenge="challenge",
            redirect_uri="http://127.0.0.1:50505/auth/callback",
            client_id="oaiapp_issued",
            login_hint="agent@example.com",
            ext_agent_host_id=_HOST_ID,
        )
        q = _query(url)
        assert q["client_id"] == ["oaiapp_issued"]
        assert q["login_hint"] == ["agent@example.com"]
        assert "agent_name_hint" not in q
        assert "prompt" not in q

    def test_reconsent_requests_the_consent_prompt(self) -> None:
        url = siwc.authorize_url(
            state="st-3",
            nonce="nn-3",
            code_challenge="challenge",
            redirect_uri="http://127.0.0.1:50505/auth/callback",
            client_id="oaiapp_issued",
            prompt="consent",
            ext_agent_host_id=_HOST_ID,
        )
        assert _query(url)["prompt"] == ["consent"]


class TestParseManualCallback:
    def test_full_url_extracts_code_and_client_id(self) -> None:
        code, client_id = siwc.parse_manual_callback(
            "http://localhost:1455/auth/callback"
            "?code=code-1&state=st-1&client_id=oaiapp_x",
            expected_state="st-1",
        )
        assert code == "code-1"
        assert client_id == "oaiapp_x"

    def test_returning_agents_callback_may_omit_the_client_id(self) -> None:
        code, client_id = siwc.parse_manual_callback(
            "http://localhost:1455/auth/callback?code=code-2&state=st-2",
            expected_state="st-2",
        )
        assert code == "code-2"
        assert client_id == ""

    def test_bare_code_fragment_is_accepted(self) -> None:
        code, client_id = siwc.parse_manual_callback(
            "code-3#st-3",
            expected_state="st-3",
        )
        assert code == "code-3"
        assert client_id == ""

    def test_state_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="State mismatch"):
            siwc.parse_manual_callback("code-4#wrong", expected_state="st-4")

    def test_a_url_without_state_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="State missing"):
            siwc.parse_manual_callback(
                "http://localhost:1455/auth/callback?code=code-5",
                expected_state="st-5",
            )

    def test_a_url_without_code_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Authorization code not found"):
            siwc.parse_manual_callback(
                "http://localhost:1455/auth/callback?state=st-6",
                expected_state="st-6",
            )


class TestResolveClientId:
    def test_new_registration_uses_the_callback_client_id(self) -> None:
        resolved = siwc.resolve_client_id(
            callback_client_id="oaiapp_new",
            saved_client_id=None,
        )
        assert resolved == "oaiapp_new"

    def test_returning_sign_in_keeps_the_saved_client_id(self) -> None:
        resolved = siwc.resolve_client_id(
            callback_client_id="",
            saved_client_id="oaiapp_saved",
        )
        assert resolved == "oaiapp_saved"

    def test_a_matching_callback_client_id_is_accepted(self) -> None:
        resolved = siwc.resolve_client_id(
            callback_client_id="oaiapp_saved",
            saved_client_id="oaiapp_saved",
        )
        assert resolved == "oaiapp_saved"

    def test_dynamic_placeholder_is_never_an_issued_id(self) -> None:
        with pytest.raises(siwc.OAuthTokenError, match="registration"):
            siwc.resolve_client_id(
                callback_client_id=siwc.DYNAMIC_CLIENT_ID,
                saved_client_id=None,
            )

    def test_malformed_callback_client_id_is_rejected(self) -> None:
        with pytest.raises(siwc.OAuthTokenError, match="registration"):
            siwc.resolve_client_id(
                callback_client_id="bogus client id with spaces",
                saved_client_id=None,
            )

    def test_a_different_client_id_than_saved_is_an_account_mismatch(self) -> None:
        with pytest.raises(siwc.OAuthTokenError, match="different ChatGPT"):
            siwc.resolve_client_id(
                callback_client_id="oaiapp_other",
                saved_client_id="oaiapp_saved",
            )

    def test_first_sign_in_without_callback_client_id_is_incomplete(self) -> None:
        with pytest.raises(siwc.OAuthTokenError, match="registration"):
            siwc.resolve_client_id(callback_client_id="", saved_client_id=None)


class TestRecords:
    def test_host_id_is_stable_and_urn_formatted(self, tmp_path: Path) -> None:
        auth_dir = tmp_path / "auth"
        first = siwc.host_id(auth_dir)
        second = siwc.host_id(auth_dir)
        assert first == second
        assert first.startswith("urn:uuid:")
        persisted = (auth_dir / siwc.HOST_ID_FILENAME).read_text(encoding="utf-8")
        assert persisted.strip() == first

    def test_record_round_trip_with_private_permissions(self, tmp_path: Path) -> None:
        path = tmp_path / "auth" / "openai.json"
        siwc.save_record(_make_record(email="agent@example.com"), path)
        assert (path.stat().st_mode & 0o777) == 0o600
        loaded = siwc.load_record(path)
        assert loaded["issuer"] == siwc.ISSUER
        assert loaded["client_id"] == _AUDIENCE
        assert loaded["ext_agent_host_id"] == _HOST_ID
        assert loaded.get("subject") == "user_abc"
        assert loaded.get("email") == "agent@example.com"
        assert loaded.get("refresh_token") == "rt_live"
        assert loaded["scopes"] == siwc.SCOPES.split()
        assert loaded.get("expires_in") == 3600.0

    def test_load_requires_a_client_id_and_scopes(self, tmp_path: Path) -> None:
        path = tmp_path / "auth" / "openai.json"
        siwc.save_record(_make_record(), path)
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw.pop("client_id")
        path.write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(ValueError, match="client_id"):
            siwc.load_record(path)

    def test_load_rejects_a_non_object_file(self, tmp_path: Path) -> None:
        path = tmp_path / "auth" / "openai.json"
        path.parent.mkdir(parents=True)
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON object"):
            siwc.load_record(path)

    def test_load_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            siwc.load_record(tmp_path / "nope.json")

    def test_signed_out_tombstone_round_trip(self, tmp_path: Path) -> None:
        path = tmp_path / "auth" / "openai.json"
        siwc.save_record(
            {
                "issuer": siwc.ISSUER,
                "client_id": _AUDIENCE,
                "ext_agent_host_id": _HOST_ID,
                "scopes": _scopes(),
                "signed_out": True,
            },
            path,
        )
        loaded = siwc.load_record(path)
        assert siwc.is_signed_out(loaded)
        assert "access_token" not in loaded

    def test_a_tokenless_record_counts_as_signed_out(self) -> None:
        record = _make_record()
        record.pop("access_token")
        assert siwc.is_signed_out(record)

    def test_is_signed_out_is_false_with_an_access_token(self) -> None:
        assert not siwc.is_signed_out(_make_record())

    def test_plan_usage_enabled_reflects_the_plan_scope(self) -> None:
        assert siwc.plan_usage_enabled(_make_record())
        assert not siwc.plan_usage_enabled(
            _make_record(scopes=["openid", "profile", "email", "offline_access"]),
        )


class TestGrants:
    def test_granted_scopes_parses_strings_and_lists(self) -> None:
        assert siwc.granted_scopes({"scope": "a b"}) == ["a", "b"]
        assert siwc.granted_scopes({"scopes": ["a", "b"]}) == ["a", "b"]
        assert siwc.granted_scopes({}) == []

    def test_exchange_posts_the_siwc_form(self) -> None:
        captured: list[dict[str, str]] = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            captured.append(_form(request))
            return httpx2.Response(
                200,
                json={
                    "access_token": "at-1",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "refresh_token": "rt-1",
                    "id_token": "it-1",
                    "scope": "openid profile",
                },
            )

        grant = siwc.exchange_code(
            _client(handler),
            code="code-1",
            redirect_uri="http://127.0.0.1:1/auth/callback",
            code_verifier="verifier-1",
            client_id=_AUDIENCE,
        )
        assert captured == [
            {
                "grant_type": "authorization_code",
                "code": "code-1",
                "redirect_uri": "http://127.0.0.1:1/auth/callback",
                "client_id": _AUDIENCE,
                "code_verifier": "verifier-1",
                "resource": siwc.RESOURCE,
            },
        ]
        assert grant["access_token"] == "at-1"  # noqa: S105 -- Fixture credential in a test; never a real secret.
        assert grant["refresh_token"] == "rt-1"  # noqa: S105 -- Fixture credential in a test; never a real secret.
        assert grant.get("id_token") == "it-1"
        assert grant["expires_in"] == 3600.0
        assert grant.get("scopes") == ["openid", "profile"]

    def test_exchange_without_refresh_token_is_an_invalid_response(self) -> None:
        def handler(_request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(
                200,
                json={
                    "access_token": "at-1",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                },
            )

        with pytest.raises(siwc.OAuthTokenError, match="incomplete credentials"):
            siwc.exchange_code(
                _client(handler),
                code="code-1",
                redirect_uri="http://127.0.0.1:1/auth/callback",
                code_verifier="verifier-1",
                client_id=_AUDIENCE,
            )

    @pytest.mark.parametrize(
        "body",
        [
            {"token_type": "Bearer", "expires_in": 3600, "refresh_token": "rt"},
            {
                "access_token": "at",
                "token_type": "basic",
                "expires_in": 3600,
                "refresh_token": "rt",
            },
            {
                "access_token": "at",
                "token_type": "Bearer",
                "expires_in": 0,
                "refresh_token": "rt",
            },
        ],
        ids=["no-access-token", "bad-token-type", "no-expiry"],
    )
    def test_exchange_rejects_malformed_grants(self, body: dict[str, object]) -> None:
        def handler(_request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(200, json=body)

        with pytest.raises(siwc.OAuthTokenError, match="invalid token response"):
            siwc.exchange_code(
                _client(handler),
                code="code-1",
                redirect_uri="http://127.0.0.1:1/auth/callback",
                code_verifier="verifier-1",
                client_id=_AUDIENCE,
            )

    def test_exchange_invalid_grant_gets_retryable_text(self) -> None:
        def handler(_request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(400, json={"error": "invalid_grant"})

        with pytest.raises(siwc.OAuthTokenError, match="Start the login again"):
            siwc.exchange_code(
                _client(handler),
                code="code-1",
                redirect_uri="http://127.0.0.1:1/auth/callback",
                code_verifier="verifier-1",
                client_id=_AUDIENCE,
            )

    def test_exchange_other_400s_keep_the_server_error_code(self) -> None:
        def handler(_request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(400, json={"error": "unauthorized_client"})

        with pytest.raises(siwc.OAuthTokenError, match="unauthorized_client"):
            siwc.exchange_code(
                _client(handler),
                code="code-1",
                redirect_uri="http://127.0.0.1:1/auth/callback",
                code_verifier="verifier-1",
                client_id=_AUDIENCE,
            )

    def test_exchange_5xx_is_a_polished_sign_in_failure(self) -> None:
        def handler(_request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(503, text="upstream down")

        with pytest.raises(siwc.OAuthTokenError, match="HTTP 503"):
            siwc.exchange_code(
                _client(handler),
                code="code-1",
                redirect_uri="http://127.0.0.1:1/auth/callback",
                code_verifier="verifier-1",
                client_id=_AUDIENCE,
            )


class TestRefreshGrant:
    @pytest.mark.anyio
    async def test_refresh_posts_the_siwc_form(self) -> None:
        captured: list[dict[str, str]] = []

        def handler(request: httpx2.Request) -> httpx2.Response:
            captured.append(_form(request))
            return httpx2.Response(
                200,
                json={
                    "access_token": "at-2",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "refresh_token": "rt-2",
                    "id_token": "it-2",
                },
            )

        grant = await siwc.async_refresh_grant(
            _async_client(handler),
            refresh_token="rt-1",  # noqa: S106 -- Fixture credential in a test; never a real secret.
            client_id=_AUDIENCE,
        )
        assert captured == [
            {
                "grant_type": "refresh_token",
                "refresh_token": "rt-1",
                "client_id": _AUDIENCE,
                "resource": siwc.RESOURCE,
            },
        ]
        assert grant["access_token"] == "at-2"  # noqa: S105 -- Fixture credential in a test; never a real secret.
        assert grant["refresh_token"] == "rt-2"  # noqa: S105 -- Fixture credential in a test; never a real secret.
        # The docs note refresh responses may omit scope; we then keep the
        # previously granted scopes ourselves.
        assert "scopes" not in grant

    @pytest.mark.anyio
    async def test_refresh_4xx_is_an_auth_refresh_error_with_login_guidance(
        self,
    ) -> None:
        def handler(_request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(400, json={"error": "invalid_grant"})

        with pytest.raises(AuthRefreshError, match="/login") as excinfo:
            await siwc.async_refresh_grant(
                _async_client(handler),
                refresh_token="rt-1",  # noqa: S106 -- Fixture credential in a test; never a real secret.
                client_id=_AUDIENCE,
            )
        assert isinstance(excinfo.value, siwc.OAuthTokenError)

    @pytest.mark.anyio
    async def test_refresh_5xx_bubbles_up_for_transient_retry(self) -> None:
        def handler(_request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(503, text="upstream down")

        with pytest.raises(httpx2.HTTPStatusError):
            await siwc.async_refresh_grant(
                _async_client(handler),
                refresh_token="rt-1",  # noqa: S106 -- Fixture credential in a test; never a real secret.
                client_id=_AUDIENCE,
            )


class TestValidateIdToken:
    def test_a_valid_token_returns_its_claims(self) -> None:
        claims = siwc.validate_id_token(
            _id_token(_claims()),
            client_id=_AUDIENCE,
            nonce="nonce-1",
            jwks=_jwks(),
        )
        assert claims["sub"] == "user_abc"
        assert claims["email"] == "agent@example.com"

    def test_a_tampered_signature_is_rejected(self) -> None:
        header, payload, signature = _id_token(_claims()).split(".")
        flipped = ("A" if signature[0] != "A" else "B") + signature[1:]
        with pytest.raises(
            siwc.IDTokenVerificationError,
            match="failed verification",
        ):
            siwc.validate_id_token(
                f"{header}.{payload}.{flipped}",
                client_id=_AUDIENCE,
                jwks=_jwks(),
            )

    def test_a_wrong_audience_is_rejected(self) -> None:
        with pytest.raises(
            siwc.IDTokenVerificationError,
            match="failed verification",
        ):
            siwc.validate_id_token(
                _id_token(_claims(aud="oaiapp_someone_else")),
                client_id=_AUDIENCE,
                jwks=_jwks(),
            )

    def test_a_wrong_nonce_is_rejected(self) -> None:
        with pytest.raises(siwc.IDTokenVerificationError, match="nonce mismatch"):
            siwc.validate_id_token(
                _id_token(_claims(nonce="other")),
                client_id=_AUDIENCE,
                nonce="nonce-1",
                jwks=_jwks(),
            )

    def test_nonce_is_only_checked_when_expected(self) -> None:
        claims = siwc.validate_id_token(
            _id_token(_claims(nonce="whatever")),
            client_id=_AUDIENCE,
            jwks=_jwks(),
        )
        assert claims["sub"] == "user_abc"

    def test_a_mismatched_azp_is_rejected(self) -> None:
        with pytest.raises(siwc.IDTokenVerificationError, match="different client"):
            siwc.validate_id_token(
                _id_token(_claims(azp="oaiapp_someone_else")),
                client_id=_AUDIENCE,
                jwks=_jwks(),
            )

    def test_a_non_rs256_algorithm_is_rejected(self) -> None:
        forged = jwt.encode(
            _claims(),
            "forged-secret-key-material-32-bytes!!",
            algorithm="HS256",
            headers={"kid": _KID},
        )
        with pytest.raises(siwc.IDTokenVerificationError, match="algorithm"):
            siwc.validate_id_token(forged, client_id=_AUDIENCE, jwks=_jwks())

    def test_an_unknown_kid_is_rejected_without_a_fetch(self) -> None:
        with pytest.raises(
            siwc.IDTokenVerificationError,
            match="not in the published JWKS",
        ):
            siwc.validate_id_token(
                _id_token(_claims(), kid="unknown-kid"),
                client_id=_AUDIENCE,
                jwks=_jwks(kid="other-kid"),
            )

    def test_a_missing_exp_is_rejected(self) -> None:
        claims = _claims()
        claims.pop("exp")
        with pytest.raises(siwc.IDTokenVerificationError, match="exp"):
            siwc.validate_id_token(
                _id_token(claims),
                client_id=_AUDIENCE,
                jwks=_jwks(),
            )

    def test_cold_cache_fetches_jwks_over_http(self) -> None:
        def handler(request: httpx2.Request) -> httpx2.Response:
            assert request.url == httpx2.URL(siwc.JWKS_URL)
            return httpx2.Response(200, json=_jwks())

        claims = siwc.validate_id_token(
            _id_token(_claims()),
            client_id=_AUDIENCE,
            http=_client(handler),
        )
        assert claims["sub"] == "user_abc"
        # The fetch populated the process cache.
        assert siwc._JWKS_CACHE

    def test_an_unknown_cached_kid_triggers_one_refetch(self) -> None:
        siwc._JWKS_CACHE[siwc.JWKS_URL] = {"stale-kid": _jwk(kid="stale-kid")}
        fetches = 0

        def handler(_request: httpx2.Request) -> httpx2.Response:
            nonlocal fetches
            fetches += 1
            return httpx2.Response(200, json=_jwks())

        claims = siwc.validate_id_token(
            _id_token(_claims()),
            client_id=_AUDIENCE,
            http=_client(handler),
        )
        assert claims["sub"] == "user_abc"
        assert fetches == 1

    def test_a_failed_jwks_fetch_is_unavailable_not_invalid(self) -> None:
        def handler(_request: httpx2.Request) -> httpx2.Response:
            return httpx2.Response(500, text="boom")

        with pytest.raises(siwc.JWKSUnavailableError):
            siwc.validate_id_token(
                _id_token(_claims()),
                client_id=_AUDIENCE,
                http=_client(handler),
            )

    def test_a_jwks_document_without_matching_keys_is_rejected(self) -> None:
        with pytest.raises(
            siwc.IDTokenVerificationError,
            match="not in the published JWKS",
        ):
            siwc.validate_id_token(
                _id_token(_claims()),
                client_id=_AUDIENCE,
                jwks={"keys": [{"kty": "EC"}]},
            )


class TestDiscoveryAndRevoke:
    def _revoke_handler(
        self,
        *,
        issuer: str = siwc.ISSUER,
        revoke_status: int = 200,
        captured: list[dict[str, str]] | None = None,
        calls: list[int] | None = None,
    ) -> Handler:
        def handler(request: httpx2.Request) -> httpx2.Response:
            if request.method == "GET":
                return httpx2.Response(
                    200,
                    json={
                        "issuer": issuer,
                        "revocation_endpoint": f"{siwc.ISSUER}/oauth/revoke",
                    },
                )
            if captured is not None:
                captured.append(_form(request))
            if calls is not None:
                calls.append(revoke_status)
            return httpx2.Response(revoke_status, text="boom")

        return handler

    @pytest.mark.anyio
    async def test_discovery_rejects_a_foreign_issuer(self) -> None:
        handler = self._revoke_handler(issuer="https://evil.example.com")
        with pytest.raises(siwc.RevocationUncertainError, match="issuer"):
            await siwc.async_discovery(_async_client(handler))

    @pytest.mark.anyio
    async def test_revoke_posts_the_token_and_client_id(self) -> None:
        captured: list[dict[str, str]] = []
        handler = self._revoke_handler(captured=captured)
        await siwc.async_revoke(
            _async_client(handler),
            refresh_token="rt-1",  # noqa: S106 -- Fixture credential in a test; never a real secret.
            client_id=_AUDIENCE,
        )
        assert captured == [
            {
                "token": "rt-1",
                "token_type_hint": "refresh_token",
                "client_id": _AUDIENCE,
            },
        ]

    @pytest.mark.anyio
    async def test_revoke_retries_a_5xx_once_then_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(siwc, "REVOKE_BACKOFF_SEC", 0.0)
        calls: list[int] = []
        handler = self._revoke_handler(revoke_status=500, calls=calls)
        with pytest.raises(siwc.RevocationUncertainError, match="HTTP 500"):
            await siwc.async_revoke(
                _async_client(handler),
                refresh_token="rt-1",  # noqa: S106 -- Fixture credential in a test; never a real secret.
                client_id=_AUDIENCE,
            )
        assert calls == [500, 500]

    @pytest.mark.anyio
    async def test_revoke_does_not_retry_a_4xx(self) -> None:
        calls: list[int] = []
        handler = self._revoke_handler(revoke_status=400, calls=calls)
        with pytest.raises(siwc.RevocationUncertainError, match="HTTP 400"):
            await siwc.async_revoke(
                _async_client(handler),
                refresh_token="rt-1",  # noqa: S106 -- Fixture credential in a test; never a real secret.
                client_id=_AUDIENCE,
            )
        assert calls == [400]

    @pytest.mark.anyio
    async def test_revoke_rejects_a_cross_origin_endpoint(self) -> None:
        discovery: Mapping[str, object] = {
            "issuer": siwc.ISSUER,
            "revocation_endpoint": "https://evil.example.com/revoke",
        }

        def handler(_request: httpx2.Request) -> httpx2.Response:
            raise AssertionError("no HTTP call should leave the client")

        with pytest.raises(
            siwc.RevocationUncertainError,
            match="revocation endpoint",
        ):
            await siwc.async_revoke(
                _async_client(handler),
                refresh_token="rt-1",  # noqa: S106 -- Fixture credential in a test; never a real secret.
                client_id=_AUDIENCE,
                discovery=discovery,
            )


if __name__ == "__main__":
    from sagent.lib.testing.main import test_main

    test_main(__file__)
