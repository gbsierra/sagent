"""Sign in with ChatGPT (SIWC) protocol for the OpenAI subscription provider.

Replaces the legacy subscription auth -- which re-used the Codex CLI's OAuth
client identity and credential file -- with OpenAI's official flow for
open-source clients. Sources of truth (fetched 2026-10-21):

  - Registration and sign-in:
    https://developers.openai.com/siwc/token-sharing-open-source/sign-in
  - Accounts and sessions:
    https://developers.openai.com/siwc/token-sharing-open-source/profiles-and-sessions
  - Token reference / errors and recovery:
    https://developers.openai.com/siwc/token-sharing-open-source/token-reference
  - On your website (the signature-validation example the OSS guide
    says also applies):
    https://developers.openai.com/siwc/website

The flow: a first sign-in sends the well-known ``dynamic_agent_client`` id,
``agent_name_hint=sagent``, and a stable per-host ``ext_agent_host_id``;
OpenAI issues sagent its own ``client_id`` in the loopback callback. The code
is exchanged for tokens (form-encoded, PKCE, no client secret, and
``resource=https://api.openai.com/v1``). The ID token is then verified the
way OpenAI's own DevKit verifies it: RS256 against the published JWKS, plus
issuer / audience / expiry / iat checks, the login ``nonce``, and ``azp``
when present. The documented credential record is stored locally per
account; later sign-ins reuse the issued client id. Inference runs against
``api.openai.com/v1/responses`` with the access token as a Bearer credential
-- owned by :mod:`sagent.providers.openai.sub`, not this module.

This module owns the protocol and the credential-record format;
:class:`sagent.providers.openai.sub.OpenAISubscription` owns paths, account
slots, the interactive login UX, and the SDK client.

Cross-checked against OpenAI's reference implementation
(openai/sign-in-with-chatgpt-devkit ``packages/local/src/oauth.ts``) and a
Python implementation of the same flow (Oaklight/llm-rosetta #1057).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    NotRequired,
    TypedDict,
    cast,
)
from urllib import parse
from urllib.parse import quote_plus

import asyncio
import contextlib
import json
import re
import uuid


if TYPE_CHECKING:
    from pathlib import Path

    import httpx2
    import jwt
else:
    from wrapt import lazy_import

    httpx2 = lazy_import("httpx2")  # 100ms cold.
    # Pulls in compiled ``cryptography``; paid only when a token is actually
    # verified, mirroring sub.py's lazy heavy imports.
    jwt = lazy_import("jwt")  # 250ms cold.


from sagent.lib.atomic_file import atomic_write_bytes
from sagent.types.exceptions import AuthRefreshError, UserFacingError


# -- Protocol constants (SIWC documentation, fetched 2026-10-21) ----------

ISSUER: Final = "https://auth.openai.com"
AUTHORIZE_URL: Final = f"{ISSUER}/api/accounts/authorize"
TOKEN_URL: Final = f"{ISSUER}/api/accounts/oauth/token"
DISCOVERY_URL: Final = f"{ISSUER}/.well-known/openid-configuration"
JWKS_URL: Final = f"{ISSUER}/.well-known/jwks.json"
#: The well-known client id an UNREGISTERED agent sends. OpenAI issues the
#: agent's real per-account client id in the loopback callback; never save
#: the placeholder as the issued id.
DYNAMIC_CLIENT_ID: Final = "dynamic_agent_client"
#: Agent name shown on OpenAI's consent screen during dynamic registration.
AGENT_NAME_HINT: Final = "sagent"
SCOPES: Final = (
    "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
)
#: The audience every token grant names; also the inference API base.
RESOURCE: Final = "https://api.openai.com/v1"
#: The scope that authorizes ChatGPT-plan inference. A sign-in without it
#: is retained but must not drive model calls (registration guide section 4).
PLAN_SCOPE: Final = "chatgpt.tokens.use.direct"
#: Loopback redirect path. ``127.0.0.1`` is required; ``localhost`` is not
#: an accepted substitute (registration guide section 2).
REDIRECT_PATH: Final = "/auth/callback"
#: Redirect port for MANUAL (headless) sign-ins only: no listener runs, so
#: the URI must be predictable before the browser opens. Interactive
#: sign-ins bind an ephemeral port instead.
MANUAL_CALLBACK_PORT: Final = 1455
#: Allowed clock skew for ID-token ``exp``/``iat`` checks. Both official
#: examples use 5 seconds.
CLOCK_LEEWAY_SEC: Final = 5.0
#: File holding the stable per-host agent id, shared by all account slots.
HOST_ID_FILENAME: Final = "openai-host-id"
#: Revocation retries (the DevKit retries once, 300 ms apart) before local
#: sign-out proceeds with an "unconfirmed" warning.
REVOKE_ATTEMPTS: Final = 2
REVOKE_BACKOFF_SEC: Final = 0.3

#: Issued client ids look like ``oaiapp_...``; bound the shape the way the
#: DevKit's callback handler does.
_CLIENT_ID_RE: Final = re.compile(r"[A-Za-z0-9_-]{1,200}\Z")


# -- Errors ---------------------------------------------------------------


class _CredentialRecordError(ValueError):
    """Stored credentials do not match the SIWC record schema.

    A subclass, like sub.py's ``_CredentialFileError``, so callers keep the
    plain ``ValueError`` contract while ruff's type-vs-value distinction
    (TRY004) stays satisfied.
    """


class OAuthTokenError(AuthRefreshError):
    """A token grant was rejected by OpenAI's OAuth endpoint.

    Subclasses :class:`~sagent.types.exceptions.AuthRefreshError` because
    every consumer of that type (the REPL renderer, the provider's retry
    logic, tests) treats "refresh failed, user must re-authenticate" as the
    contract, and an ``invalid_grant`` at sign-in carries the same meaning.

    Args:
        code: The endpoint's ``error`` code (e.g. ``invalid_grant``).
        message: Polished remediation text shown verbatim to the user.

    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class IDTokenVerificationError(UserFacingError):
    """The ID token failed signature or claim verification; sign in again."""


class JWKSUnavailableError(IDTokenVerificationError):
    """OpenAI's published keys could not be retrieved; retryable.

    The DevKit classifies failures at the key-retrieval boundary as an
    availability problem, not an invalid identity: an unreachable key
    service must not force a re-authentication. Callers decide the policy
    (login treats it as fatal; refresh may warn and keep the session).
    """


class PlanUsageNotGrantedError(UserFacingError):
    """The sign-in is retained but does not authorize ChatGPT-plan usage."""


class RevocationUncertainError(RuntimeError):
    """Sign-out revocation could not be confirmed with OpenAI.

    Not a :class:`UserFacingError`: the caller clears local tokens either
    way and turns this into a warning pointing at ChatGPT Settings, as the
    DevKit's ``revocation_failed`` message does.
    """


class SubscriptionUsageError(UserFacingError):
    """A ChatGPT-plan request failed with a documented SIWC error code.

    Args:
        code: The structured ``error.code`` from the Responses layer.
        message: Polished recovery text shown verbatim to the user.

    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


#: Documented structured Responses error codes mapped to recovery text
#: (errors and recovery, fetched 2026-10-21). Transient 503 availability
#: codes are deliberately absent: those pass through to the provider's
#: existing retry behavior, and ``subscription_sharing_invalid_user``
#: (401) rides the authentication-retry path instead.
PLAN_ERROR_RECOVERY: Final[dict[str, str]] = {
    "subscription_sharing_usage_limit_exceeded": (
        "ChatGPT plan usage limit reached. New plan-backed requests are "
        "paused; check or adjust limits at https://chatgpt.com/settings/"
        "usage, wait for the window, or /model to switch providers."
    ),
    "subscription_sharing_user_not_eligible": (
        "ChatGPT plan usage is unavailable for this account, workspace, "
        "or policy. Switch to an API-key provider with /model."
    ),
    "subscription_sharing_unsupported_capability": (
        "ChatGPT plan usage rejected an unsupported request feature; "
        "remove the unsupported option (the error names its param), or "
        "/model to switch providers."
    ),
    "subscription_sharing_route_not_supported": (
        "ChatGPT plan usage does not support this request route. "
        "Report this as a sagent bug."
    ),
    "chatpass_v2_scope_not_authorized": (
        "The signed permission context does not authorize this operation. "
        "Run /login again; if it persists, report it as a sagent bug."
    ),
    "chatpass_v2_invalid_authorization_context": (
        "The signed permission context does not authorize this operation. "
        "Run /login again; if it persists, report it as a sagent bug."
    ),
}


# -- Credential record -----------------------------------------------------


class SIWCCredentials(TypedDict):
    """Local SIWC credential record, one JSON file per account.

    The shape follows the documented record (registration guide section 5).
    Token fields are ``NotRequired`` so the same type covers a signed-out
    tombstone: registration retained for the next sign-in, tokens cleared.
    """

    email: NotRequired[str]
    issuer: str
    subject: NotRequired[str]
    #: Per-account client id OpenAI issued at registration. Later sign-ins
    #: present it (``agent_name_hint`` omitted) instead of re-registering.
    client_id: str
    ext_agent_host_id: str
    id_token: NotRequired[str]
    access_token: NotRequired[str]
    refresh_token: NotRequired[str]
    token_type: NotRequired[str]
    expires_in: NotRequired[float]
    scopes: list[str]
    saved_at: NotRequired[str]
    earliest_refresh_at: NotRequired[str]
    #: Present only after sign-out cleared the tokens while retaining the
    #: registration for a later sign-in.
    signed_out: NotRequired[bool]


class TokenGrant(TypedDict):
    """Normalized token-endpoint response.

    ``refresh_token`` is required: every grant this codebase requests carries
    ``offline_access`` (sign-in) or is itself a refresh, and the DevKit's
    ``tokenFields`` rejects a response without one. ``scopes`` is
    ``NotRequired`` because a refresh response may omit ``scope`` when the
    previously granted scopes are unchanged.
    """

    access_token: str
    token_type: str
    expires_in: float
    refresh_token: str
    id_token: NotRequired[str]
    scopes: NotRequired[list[str]]
    earliest_refresh_at: NotRequired[str]


def host_id(auth_dir: Path) -> str:
    """Return this host's stable ``ext_agent_host_id``, creating it once.

    The identifier must stay stable across sign-ins and accounts
    (registration guide section 1), so it lives in its own file shared by
    every account slot rather than inside any one account's record. It is
    an opaque identifier, not a credential.

    Args:
        auth_dir: Directory that holds sagent's own auth files.

    Returns:
        host_id: ``urn:uuid:<uuid4>`` identifier for this host.

    """
    path = auth_dir / HOST_ID_FILENAME
    if path.exists():
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    created = f"urn:uuid:{uuid.uuid4()}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{created}\n", encoding="utf-8")
    return created


def load_record(path: Path) -> SIWCCredentials:
    """Load one SIWC credential record from disk.

    Args:
        path: Record file path.

    Returns:
        record: Parsed credential record.

    Raises:
        FileNotFoundError: No record at ``path``.
        ValueError: The file is not a SIWC credential record.

    """
    decoded: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise _CredentialRecordError(f"{path} must contain a JSON object.")
    raw = cast(dict[str, object], decoded)
    client_id = raw.get("client_id")
    scopes = raw.get("scopes")
    if not isinstance(client_id, str) or not client_id:
        raise _CredentialRecordError(f"{path} is missing the SIWC client_id field.")
    if not isinstance(scopes, list) or not all(
        isinstance(scope, str) for scope in cast(list[object], scopes)
    ):
        raise _CredentialRecordError(f"{path} is missing the SIWC scopes field.")
    scopes_list = [str(scope) for scope in cast(list[object], scopes)]
    record: dict[str, object] = {
        "issuer": str(raw.get("issuer") or ISSUER),
        "client_id": client_id,
        "ext_agent_host_id": str(raw.get("ext_agent_host_id") or ""),
        "scopes": scopes_list,
    }
    for key in (
        "email",
        "subject",
        "id_token",
        "access_token",
        "refresh_token",
        "token_type",
        "saved_at",
        "earliest_refresh_at",
    ):
        value = raw.get(key)
        if isinstance(value, str):
            record[key] = value
    expires_in = raw.get("expires_in")
    if isinstance(expires_in, (int, float)):
        record["expires_in"] = float(expires_in)
    if raw.get("signed_out") is True:
        record["signed_out"] = True
    return cast(SIWCCredentials, record)


def save_record(record: SIWCCredentials, path: Path) -> None:
    """Persist one SIWC credential record atomically with owner-only mode.

    Args:
        record: Credential record to persist.
        path: Record file path.

    """
    payload = json.dumps(dict(record), indent=2, sort_keys=True).encode()
    atomic_write_bytes(path, payload, file_mode=0o600)


def is_signed_out(record: SIWCCredentials) -> bool:
    """Return True when the record is a signed-out tombstone.

    Args:
        record: Credential record to inspect.

    Returns:
        signed_out: True when the registration is retained but no tokens
            are present.

    """
    access_token = record.get("access_token")
    return (
        record.get("signed_out") is True or access_token is None or access_token == ""
    )


def plan_usage_enabled(record: SIWCCredentials) -> bool:
    """Return True when the record authorizes ChatGPT-plan inference.

    Args:
        record: Credential record to inspect.

    Returns:
        enabled: Whether the ``chatgpt.tokens.use.direct`` scope was granted.

    """
    return PLAN_SCOPE in record["scopes"]


# -- Authorization ---------------------------------------------------------


def authorize_url(
    *,
    client_id: str,
    redirect_uri: str,
    state: str,
    nonce: str,
    code_challenge: str,
    ext_agent_host_id: str,
    agent_name_hint: str | None = None,
    login_hint: str | None = None,
    prompt: str | None = None,
) -> str:
    """Build the SIWC authorization URL (registration guide section 2).

    ``id_token_hint`` is deliberately NOT sent. The DevKit keeps tokens out
    of browser URLs -- the same URLs sagent prints to the terminal and
    passes through the browser opener's argv -- and uses the saved email
    as ``login_hint`` instead. The retained ID token still lives in the
    stored record, per the documented record format.

    Args:
        client_id: ``DYNAMIC_CLIENT_ID`` for a first registration, or the
            previously issued client id for a returning agent.
        redirect_uri: Exact loopback URI this sign-in will exchange.
        state: Unpredictable state token, validated on callback.
        nonce: Unpredictable nonce, validated inside the ID token.
        code_challenge: S256 PKCE challenge.
        ext_agent_host_id: Stable per-host identifier.
        agent_name_hint: Agent display name; send ONLY on first
            registration, omit for returning agents.
        login_hint: Saved email for a returning agent's sign-in screen.
        prompt: OAuth prompt parameter. ``"consent"`` re-shows the consent
            screen to re-enable ChatGPT plan usage after an earlier
            decline (errors and recovery); never force it on ordinary
            sign-ins.

    Returns:
        url: URL to open in the user's browser.

    """
    parts: list[tuple[str, str]] = [
        ("client_id", client_id),
        ("response_type", "code"),
        ("redirect_uri", redirect_uri),
        ("scope", SCOPES),
        ("resource", RESOURCE),
        ("state", state),
        ("nonce", nonce),
        ("code_challenge", code_challenge),
        ("code_challenge_method", "S256"),
        ("ext_agent_host_id", ext_agent_host_id),
    ]
    if agent_name_hint is not None:
        parts.append(("agent_name_hint", agent_name_hint))
    if login_hint is not None:
        parts.append(("login_hint", login_hint))
    if prompt is not None:
        parts.append(("prompt", prompt))
    query = "&".join(f"{quote_plus(k)}={quote_plus(v)}" for k, v in parts)
    return f"{AUTHORIZE_URL}?{query}"


def parse_manual_callback(value: str, expected_state: str) -> tuple[str, str]:
    """Extract ``(code, client_id)`` from a pasted manual-login input.

    In headless mode no listener runs; the browser lands on the loopback
    redirect (or fails to connect), and the user copies the full URL back
    into the terminal. A first registration's callback URL carries the
    issued ``client_id`` -- without it the registration is incomplete.

    Args:
        value: Pasted redirect URL, or ``code#state`` shorthand.
        expected_state: State token originally sent in the authorize URL.

    Returns:
        pair: ``(code, client_id)``; ``client_id`` may be empty for a
            returning agent's callback or a bare-code paste.

    Raises:
        ValueError: No code present, or state missing/mismatched.

    """
    text = value.strip()
    parsed = parse.urlparse(text)
    code = ""
    state = ""
    client_id = ""
    if parsed.scheme and parsed.netloc:
        params = parse.parse_qs(parsed.query)
        code = (params.get("code") or [""])[0]
        state = (params.get("state") or [""])[0]
        client_id = (params.get("client_id") or [""])[0]
    else:
        code, sep, state = text.partition("#")
        if not sep:
            state = ""
    if not code:
        raise ValueError("Authorization code not found")
    if not state:
        raise ValueError("State missing in pasted authorization code")
    if state != expected_state:
        raise ValueError("State mismatch in pasted authorization code")
    return code, client_id


def resolve_client_id(
    *,
    callback_client_id: str,
    saved_client_id: str | None,
) -> str:
    """Resolve the effective issued client id for a sign-in attempt.

    Rules (registration guide section 3, mirrored from the DevKit callback
    handler): a new registration's callback must carry the issued id; the
    ``dynamic_agent_client`` placeholder is never accepted as issued; a
    reauthorization callback may omit it (the saved registration applies);
    and a callback id that differs from the saved registration is rejected
    rather than replacing the selected account's registration.

    Args:
        callback_client_id: Client id from the callback (may be empty).
        saved_client_id: Issued id retained from this account's previous
            sign-in, if any.

    Returns:
        client_id: The issued id to use for the token exchange.

    Raises:
        OAuthTokenError: Registration is incomplete or conflicts with the
            saved registration.

    """
    client_id = callback_client_id or (saved_client_id or "")
    if (
        not client_id
        or not _CLIENT_ID_RE.match(client_id)
        or client_id == DYNAMIC_CLIENT_ID
    ):
        raise OAuthTokenError(
            "registration_incomplete",
            "OpenAI did not complete app registration. Start the login again.",
        )
    if (
        saved_client_id is not None
        and saved_client_id != ""
        and callback_client_id not in ("", saved_client_id)
    ):
        raise OAuthTokenError(
            "account_mismatch",
            "This sign-in returned a different ChatGPT registration. "
            "Choose the original account or sign in under a new --account.",
        )
    return client_id


# -- Token grants ----------------------------------------------------------


def granted_scopes(payload: Mapping[str, object]) -> list[str]:
    """Return the scopes a token response actually granted.

    The endpoint returns a space-separated ``scope`` string (registration
    guide section 3); a list under ``scopes`` is tolerated for symmetry
    with the stored record.

    Args:
        payload: Decoded JSON body of the token response.

    Returns:
        scopes: Granted scope strings, empty when the response carried none
            (a refresh may omit ``scope`` when grants are unchanged).

    """
    scope = payload.get("scope")
    if isinstance(scope, str):
        return scope.split()
    scopes = payload.get("scopes")
    if isinstance(scopes, list):
        return [str(entry) for entry in cast(list[object], scopes)]
    return []


def _grant_from_response(payload: Mapping[str, object]) -> TokenGrant:
    """Validate a token-endpoint response and normalize it to a grant.

    Args:
        payload: Decoded JSON body of the token response.

    Returns:
        grant: Normalized grant fields.

    Raises:
        OAuthTokenError: The response is not a usable token grant.

    """
    access_token = payload.get("access_token")
    token_type = payload.get("token_type")
    expires_in = payload.get("expires_in")
    if (
        not isinstance(access_token, str)
        or not access_token
        or not isinstance(token_type, str)
        or token_type.lower() != "bearer"
        or not isinstance(expires_in, (int, float))
        or not expires_in > 0
    ):
        raise OAuthTokenError(
            "invalid_token_response",
            "OpenAI returned an invalid token response. Please sign in again.",
        )
    refresh_token = payload.get("refresh_token")
    if not isinstance(refresh_token, str) or not refresh_token:
        raise OAuthTokenError(
            "invalid_token_response",
            "OpenAI returned incomplete credentials. Please sign in again.",
        )
    grant: TokenGrant = {
        "access_token": access_token,
        "token_type": token_type,
        "expires_in": float(expires_in),
        "refresh_token": refresh_token,
    }
    id_token = payload.get("id_token")
    if isinstance(id_token, str) and id_token:
        grant["id_token"] = id_token
    scopes = granted_scopes(payload)
    if scopes:
        grant["scopes"] = scopes
    earliest_refresh_at = payload.get("earliest_refresh_at")
    if isinstance(earliest_refresh_at, (str, int, float)):
        grant["earliest_refresh_at"] = str(earliest_refresh_at)
    return grant


def _token_error(
    status_code: int,
    payload: Mapping[str, object],
    body_text: str,
    *,
    refresh: bool,
) -> OAuthTokenError:
    """Map a token-endpoint failure to a polished, actionable error."""
    code = payload.get("error")
    code_str = code if isinstance(code, str) else ""
    if refresh:
        # Any 400/401 on a refresh grant means the session is gone: the
        # rotating refresh token was consumed elsewhere, revoked, or aged
        # out. Retrying is pointless; the user must re-authenticate.
        return OAuthTokenError(
            code_str or str(status_code),
            "OpenAI subscription session expired or revoked. Run /login to "
            "re-authenticate, or /model to switch providers.",
        )
    if code_str == "invalid_grant":
        # Registration guide section 3: discard the code and start a fresh
        # authorization, retaining the issued client id.
        return OAuthTokenError(
            code_str,
            "OpenAI rejected the sign-in code (expired or already used). "
            "Start the login again.",
        )
    detail = f", error={code_str}" if code_str else ""
    return OAuthTokenError(
        code_str or str(status_code),
        f"OpenAI sign-in failed (HTTP {status_code}{detail}). Body: {body_text[:200]}",
    )


def _error_payload(response: httpx2.Response) -> dict[str, object]:
    """Best-effort decode of an error response body as a mapping."""
    payload: dict[str, object] = {}
    with contextlib.suppress(ValueError, TypeError):
        decoded: object = response.json()
        if isinstance(decoded, dict):
            payload = cast(dict[str, object], decoded)
    return payload


def exchange_code(
    http: httpx2.Client,
    *,
    code: str,
    redirect_uri: str,
    client_id: str,
    code_verifier: str,
) -> TokenGrant:
    """Exchange an authorization code for a token grant.

    Args:
        http: Synchronous HTTP client used for the POST.
        code: Authorization code from the callback.
        redirect_uri: The exact redirect URI sent in the authorize URL.
        client_id: The ISSUED client id (never the dynamic placeholder).
        code_verifier: PKCE verifier matching the authorize challenge.

    Returns:
        grant: Token response fields (access token, rotating refresh
            token, id token, granted scopes).

    Raises:
        OAuthTokenError: The endpoint rejected the grant.

    """
    r = http.post(
        TOKEN_URL,
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "code_verifier": code_verifier,
            "resource": RESOURCE,
        },
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        timeout=15.0,
    )
    if r.status_code >= 400:
        raise _token_error(
            r.status_code,
            _error_payload(r),
            r.text,
            refresh=False,
        )
    return _grant_from_response(cast(Mapping[str, object], r.json()))


async def async_refresh_grant(
    http: httpx2.AsyncClient,
    *,
    refresh_token: str,
    client_id: str,
) -> TokenGrant:
    """Exchange a refresh token for a new token grant.

    The refresh grant carries the same ``resource`` and the issued client
    id, and omits ``scope`` to retain the original grant. The response
    carries the rotating replacement refresh token (the DevKit's
    ``tokenFields`` requires one).

    Args:
        http: Async HTTP client used for the POST.
        refresh_token: The current (rotating) refresh token.
        client_id: The issued client id for this account.

    Returns:
        grant: Token response fields.

    Raises:
        OAuthTokenError: The endpoint rejected the grant; the caller must
            discard the stored tokens and ask the user to sign in again.

    """
    r = await http.post(
        TOKEN_URL,
        data={
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": client_id,
            "resource": RESOURCE,
        },
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
        timeout=15.0,
    )
    if r.status_code in (400, 401):
        # Terminal, not retryable: rotation consumed the shared token or
        # the session was revoked server-side.
        raise _token_error(
            r.status_code,
            _error_payload(r),
            r.text,
            refresh=True,
        )
    r.raise_for_status()
    return _grant_from_response(cast(Mapping[str, object], r.json()))


# -- ID-token verification -------------------------------------------------

_JWKS_CACHE: dict[str, dict[str, Mapping[str, object]]] = {}


def _map_jwks(doc: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    """Map a JWKS document to ``kid`` -> JWK."""
    raw = doc.get("keys")
    if not isinstance(raw, list):
        raise IDTokenVerificationError(
            "OpenAI's JWKS did not contain a keys list.",
        )
    mapped: dict[str, Mapping[str, object]] = {}
    for key in cast(list[object], raw):
        if isinstance(key, dict):
            key_map = cast(dict[str, object], key)
            kid = key_map.get("kid")
            if isinstance(kid, str):
                mapped[kid] = key_map
    return mapped


def _fetch_and_cache_jwks(
    http: httpx2.Client,
) -> dict[str, Mapping[str, object]]:
    """Fetch the JWKS document, map it by ``kid``, and cache the result.

    Raises:
        JWKSUnavailableError: The fetch or parse failed.

    """
    try:
        r = http.get(JWKS_URL, timeout=15.0)
        r.raise_for_status()
        doc = cast(dict[str, object], r.json())
    except (
        Exception
    ) as e:  # Retrieval failure is an availability problem, never an invalid identity.
        raise JWKSUnavailableError(f"OpenAI's JWKS could not be fetched: {e}") from e
    mapped = _map_jwks(doc)
    _JWKS_CACHE[JWKS_URL] = mapped
    return mapped


def _jwks_keys(
    *,
    jwks: Mapping[str, object] | None,
    http: httpx2.Client | None,
) -> dict[str, Mapping[str, object]]:
    """Return ``kid`` -> JWK from an injected doc, the cache, or a fetch.

    ``http=None`` with a cold cache fetches with a short-lived client, so
    refresh-path verification (which runs in a worker thread) is not
    blocked on the caller supplying a client.

    Raises:
        JWKSUnavailableError: The keys could not be retrieved.

    """
    if jwks is not None:
        return _map_jwks(jwks)
    cached = _JWKS_CACHE.get(JWKS_URL)
    if cached is not None:
        return cached
    if http is None:
        with httpx2.Client() as fallback:
            return _fetch_and_cache_jwks(fallback)
    return _fetch_and_cache_jwks(http)


def validate_id_token(
    id_token: str,
    *,
    client_id: str,
    nonce: str | None = None,
    http: httpx2.Client | None = None,
    jwks: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Verify an SIWC ID token and return its claims.

    Verification mirrors the DevKit: RS256 only, signature checked against
    OpenAI's published JWKS (cached per process, refetched once on an
    unfamiliar ``kid``), issuer and audience (the issued client id) and
    required claims checked by PyJWT, then the login ``nonce`` and ``azp``
    when present. A maintained JOSE library is used deliberately; the
    website guide says "Use a maintained JWT library" and warns against
    hand-rolling.

    Args:
        id_token: The ``id_token`` from a token grant.
        client_id: The issued client id; must be the token's ``aud``.
        nonce: Nonce sent in the authorize URL, for sign-in validation.
            ``None`` skips the nonce check (refresh-issued ID tokens carry
            none).
        http: HTTP client for a JWKS fetch when not already cached;
            ``None`` creates a short-lived client for a cold cache.
        jwks: Pre-fetched JWKS document; bypasses the cache entirely
            (tests and offline verification).

    Returns:
        claims: Verified ID-token claims (``sub``, ``email``, ...).

    Raises:
        IDTokenVerificationError: Verification failed.
        JWKSUnavailableError: OpenAI's keys could not be retrieved.

    """
    if not id_token:
        raise IDTokenVerificationError("OpenAI returned no ID token.")
    try:
        header = jwt.get_unverified_header(id_token)
    except jwt.PyJWTError as e:
        raise IDTokenVerificationError(
            f"OpenAI returned a malformed ID token: {e}",
        ) from e
    if header.get("alg") != "RS256":
        raise IDTokenVerificationError(
            f"OpenAI's ID token used unexpected algorithm "
            f"{header.get('alg')!r}; expected RS256.",
        )
    kid = header.get("kid")
    if not isinstance(kid, str) or not kid:
        raise IDTokenVerificationError(
            "OpenAI's ID token did not name its signature key.",
        )
    keys = _jwks_keys(jwks=jwks, http=http)
    jwk = keys.get(kid)
    if jwk is None and jwks is None:
        # Unfamiliar kid: the cached set may predate a key rotation. One
        # refetch, then give up loudly rather than skip checks.
        _JWKS_CACHE.pop(JWKS_URL, None)
        keys = _jwks_keys(jwks=None, http=http)
        jwk = keys.get(kid)
    if jwk is None:
        raise IDTokenVerificationError(
            "OpenAI's ID token signature key was not in the published JWKS.",
        )
    try:
        signing_key = jwt.PyJWK.from_dict(
            cast(dict[str, Any], jwk),
            algorithm="RS256",
        )
        claims = jwt.decode(
            id_token,
            signing_key.key,
            algorithms=["RS256"],
            audience=client_id,
            issuer=ISSUER,
            leeway=CLOCK_LEEWAY_SEC,
            options={"require": ["iss", "aud", "exp", "iat", "sub"]},
        )
    except jwt.PyJWTError as e:
        raise IDTokenVerificationError(
            f"OpenAI ID token failed verification: {e}",
        ) from e
    azp = claims.get("azp")
    if isinstance(azp, str) and azp != client_id:
        raise IDTokenVerificationError(
            "OpenAI ID token was issued for a different client.",
        )
    if nonce is not None and claims.get("nonce") != nonce:
        raise IDTokenVerificationError(
            "OpenAI ID token nonce mismatch; the token was not issued for "
            "this login attempt.",
        )
    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject:
        raise IDTokenVerificationError(
            "OpenAI ID token did not contain a subject.",
        )
    return cast(dict[str, object], claims)


# -- Sign-out ---------------------------------------------------------------


async def async_discovery(http: httpx2.AsyncClient) -> dict[str, object]:
    """Fetch and sanity-check OpenAI's OIDC discovery document.

    Args:
        http: Async HTTP client used for the fetch.

    Returns:
        document: The parsed discovery document.

    Raises:
        RevocationUncertainError: The document could not be fetched or its
            issuer does not match the pinned constant.

    """
    try:
        r = await http.get(DISCOVERY_URL, timeout=15.0)
        r.raise_for_status()
        document = cast(dict[str, object], r.json())
    except (
        Exception
    ) as e:  # Any fetch failure blocks revocation confirmation, not identity.
        raise RevocationUncertainError(
            "OpenAI's sign-in configuration could not be verified.",
        ) from e
    if document.get("issuer") != ISSUER:
        raise RevocationUncertainError(
            "OpenAI's discovery document has an unexpected issuer.",
        )
    return document


async def async_revoke(
    http: httpx2.AsyncClient,
    *,
    refresh_token: str,
    client_id: str,
    discovery: Mapping[str, object] | None = None,
) -> None:
    """Revoke a refresh token at the discovery document's endpoint.

    Follows the DevKit: success is HTTP 200; 5xx and network failures are
    retried once; anything else is reported as unconfirmed so the caller
    clears local tokens with an honest warning.

    Args:
        http: Async HTTP client used for the calls.
        refresh_token: The token to revoke.
        client_id: The issued client id the token belongs to.
        discovery: Pre-fetched discovery document; ``None`` fetches it.

    Raises:
        RevocationUncertainError: Revocation could not be confirmed.

    """
    document = discovery
    if document is None:
        document = await async_discovery(http)
    endpoint = document.get("revocation_endpoint")
    if not isinstance(endpoint, str) or not endpoint.startswith(f"{ISSUER}/"):
        raise RevocationUncertainError(
            "OpenAI's discovery document lists no usable revocation endpoint.",
        )
    last_failure = "the endpoint did not confirm revocation"
    for attempt in range(REVOKE_ATTEMPTS):
        try:
            r = await http.post(
                endpoint,
                data={
                    "token": refresh_token,
                    "token_type_hint": "refresh_token",
                    "client_id": client_id,
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                timeout=10.0,
            )
        except Exception as e:  # noqa: BLE001 -- Transient network failure; retried, then reported.
            last_failure = f"network error: {e}"
        else:
            if r.status_code == 200:
                return
            last_failure = f"HTTP {r.status_code}"
            if r.status_code < 500:
                break
        if attempt + 1 < REVOKE_ATTEMPTS:
            await asyncio.sleep(REVOKE_BACKOFF_SEC)
    raise RevocationUncertainError(
        f"OpenAI did not confirm disconnection ({last_failure}). Local "
        "credentials were removed; disconnect the app in ChatGPT Settings.",
    )


def record_saved_at() -> str:
    """Return the ``saved_at`` timestamp for a record, ISO-8601 UTC.

    Returns:
        stamp: Current UTC time in ISO-8601, as the documented record's
            ``saved_at`` field expects.

    """
    return datetime.now(UTC).isoformat()
