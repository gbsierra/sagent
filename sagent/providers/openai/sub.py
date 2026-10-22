"""OpenAI subscription provider (Sign in with ChatGPT + Responses API).

Authenticates through OpenAI's official ``Sign in with ChatGPT`` flow for
open-source clients: sagent registers as its own OAuth client, gets its
own issued ``client_id``, stores credentials in its own protected store
under the user data dir, and calls the public Responses API on the user's
ChatGPT plan. The protocol (registration, PKCE, token exchange, ID-token
verification, refresh, revocation, the credential-record format) lives in
:mod:`sagent.providers.openai.siwc`; this module owns paths, account
slots, the interactive login UX, and the SDK client.

Usage::

    from sagent.providers import OpenAISubscription

    provider = OpenAISubscription.from_credentials()
    model = provider.model("gpt-5.4-mini")
    response = await model.buffer(request)

NOTICE -- Provenance and Terms of Use
-------------------------------------

The protocol implemented here is OpenAI's documented "Sign in with
ChatGPT" flow for open-source clients (fetched 2026-10-21):

    https://developers.openai.com/siwc/token-sharing-open-source/sign-in

Cross-checked against OpenAI's reference implementation,
openai/sign-in-with-chatgpt-devkit (``packages/local/src/oauth.ts``).
The legacy flow this replaces re-used the Codex CLI's OAuth client and
credential file; that path is deliberately not imported or migrated --
existing Codex tokens do not carry SIWC grants, so users sign in once
with sagent's own client.

Use of the ChatGPT subscription is governed by OpenAI's Terms of Use:

    https://openai.com/policies/row-terms-of-use/  (rest of world)
    https://openai.com/policies/terms-of-use/      (US)

Users are responsible for ensuring their own usage complies with those
terms. Cost figures are computed from public per-token API pricing as a
"what would this session cost at API rates" metric -- the user is not
billed per-token for subscription auth.
"""

from __future__ import annotations

from datetime import datetime
from typing import (
    IO,
    TYPE_CHECKING,
    override,
)

import asyncio
import base64
import contextlib
import logging
import secrets
import sys
import time


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from openai.types.responses.response_create_params import (
        ResponseCreateParamsStreaming,
    )

    import httpx2
    import openai

    from sagent.types.model import (
        ModelRequest,
        ModelResponse,
    )
    from sagent.types.runtime import (
        RuntimeEvent,
    )
else:
    from wrapt import lazy_import

    httpx2 = lazy_import("httpx2")  # 100ms cold.
    openai = lazy_import("openai")  # 493ms cold.


from sagent.catalog.openai import subscription, subscription_models
from sagent.catalog.table import ModelCatalog
from sagent.lib.codec import ReadError, from_plain, loads
from sagent.lib.userdirs import data_dir
from sagent.providers.lib.oauth import (
    AuthCodeListener,
    credential_file_lock,
    credentials_path,
    pkce_pair,
)
from sagent.providers.lib.perloop import PerLoop
from sagent.providers.openai import siwc
from sagent.providers.openai.api import OpenAI
from sagent.providers.openai.responses import _OpenAIResponsesModel
from sagent.types.exceptions import AuthRefreshError


logger = logging.getLogger(__name__)

DEFAULT_REFRESH_BUFFER_SEC = (
    300.0  # house-ignore[globals] -- OAuth refresh lead time, user-retunable.
)


def _auth_dir() -> Path:
    """Return sagent's own auth directory under the user data dir.

    The SIWC credential records live here, one file per account slot,
    alongside the per-host ``ext_agent_host_id`` file shared by every
    account. Vendor-owned paths (the legacy flow read the Codex CLI's
    ``~/.codex/auth.json``) are no longer involved.
    """
    return data_dir() / "rekursiv-ai" / "sagent" / "auth"


def _default_credentials_path() -> Path:
    """Return the SIWC record path for the unnamed account slot."""
    return _auth_dir() / "openai.json"


class _CredentialFileError(ValueError):
    """Stored credentials do not match the SIWC record schema or state."""


class OpenAISubscription:
    """OpenAI provider -- SIWC OAuth + ChatGPT subscription billing.

    Uses the OpenAI catalog with token limits clamped to the subscription
    wire contract. Cost tracking uses standard API per-token pricing even
    though subscription users pay a flat fee. This is intentional: it gives
    a consistent "what would this session cost at API rates" metric
    regardless of auth mode.
    """

    catalog = ModelCatalog(models=subscription_models(), transport=subscription())

    #: The SIWC credential record (see :mod:`sagent.providers.openai.siwc`).
    Credentials = siwc.SIWCCredentials

    def __init__(
        self,
        *,
        access_token: str,
        refresh_token: str,
        client_id: str,
        expires_at: float,
        scopes: tuple[str, ...] | list[str] = (),
        subject: str = "",
        email: str = "",
        account: str | None = None,
        refresh_buffer_sec: float = DEFAULT_REFRESH_BUFFER_SEC,
    ) -> None:
        self._access_token = access_token
        self._refresh_token = refresh_token
        self._client_id = client_id  # SIWC registration (issued client id).
        self._scopes = list(scopes)
        self._subject = subject  # Verified ID-token identity, display only.
        self._email = email  # Verified ID-token email, display only.
        self._account = account  # Local credential slot name.
        self._expires_at = expires_at
        self._refresh_buffer_sec = refresh_buffer_sec
        # Client and its token cached together, per loop: the client's
        # pool belongs to the loop that opened it, and a token outliving
        # its client would authenticate a connection that no longer
        # exists. The guarding lock is per loop for the same reason -- it
        # binds to the loop that first contends on it.
        self._authed: PerLoop[tuple[openai.AsyncOpenAI, str] | None] = PerLoop(
            lambda: None,
        )
        self._lock: PerLoop[asyncio.Lock] = PerLoop(asyncio.Lock)

    def _cred_path(self) -> Path:
        """Return this account slot's SIWC record path."""
        return credentials_path(_default_credentials_path(), self._account)

    @classmethod
    def from_key(
        cls,
        api_key: str,
        *,
        base_url: str | None = None,
    ) -> OpenAI:
        """Create an API-key provider (delegates to ``OpenAI``).

        Subscription billing is incompatible with API keys, so this
        returns a plain ``OpenAI`` instance.

        Args:
          api_key: OpenAI API key.
          base_url: Override the default endpoint URL.

        Returns:
          provider: ``OpenAI`` provider instance.

        """
        return OpenAI.from_key(api_key, base_url=base_url)

    @classmethod
    def from_credentials(
        cls,
        creds: OpenAISubscription.Credentials | None = None,
        *,
        account: str | None = None,
        refresh_buffer_sec: float = DEFAULT_REFRESH_BUFFER_SEC,
    ) -> OpenAISubscription:
        """Create a provider from SIWC credentials.

        Args:
          creds: Pre-loaded SIWC record, or ``None`` to auto-load from
            sagent's own credential store.
          account: Named credential slot.
          refresh_buffer_sec: Seconds before token expiry to trigger
              proactive refresh.

        Returns:
          provider: Subscription provider instance.

        Raises:
          FileNotFoundError: No record on disk; the caller should point the
            user at the login command.
          _CredentialFileError: The record is a signed-out tombstone or
            lacks tokens.
          siwc.PlanUsageNotGrantedError: The retained sign-in was saved
            without the ChatGPT-plan-usage scope and must not drive
            inference.

        """
        if creds is None:
            creds = cls.load(account=account)
        access_token = creds.get("access_token")
        refresh_token = creds.get("refresh_token")
        if (
            access_token is None
            or access_token == ""
            or refresh_token is None
            or refresh_token == ""
        ):
            raise _CredentialFileError(
                "The saved sign-in is missing tokens. Run `sagent "
                "--provider OpenAISubscription login` to sign in again.",
            )
        if siwc.is_signed_out(creds):
            raise _CredentialFileError(
                "This account is signed out. Run `sagent --provider "
                "OpenAISubscription login` to sign in again.",
            )
        if not siwc.plan_usage_enabled(creds):
            raise siwc.PlanUsageNotGrantedError(
                "The saved sign-in does not include ChatGPT plan usage "
                "(its consent screen declined it). Run `sagent --provider "
                "OpenAISubscription login` again to re-consent, or switch "
                "to the API-key path: --provider OpenAI --auth env.",
            )
        return cls(
            access_token=access_token,
            refresh_token=refresh_token,
            client_id=creds["client_id"],
            expires_at=_record_expiry(creds),
            scopes=creds["scopes"],
            subject=creds.get("subject", ""),
            email=creds.get("email", ""),
            account=account,
            refresh_buffer_sec=refresh_buffer_sec,
        )

    @classmethod
    def login(
        cls,
        output: IO[str] | None = None,
        *,
        listener_timeout_sec: float = 300.0,
        account: str | None = None,
        manual: bool = False,
    ) -> OpenAISubscription.Credentials:
        """Run the interactive Sign in with ChatGPT flow (SIWC).

        A first sign-in registers sagent dynamically and saves the issued
        client id; later sign-ins reuse it. The callback's code is
        exchanged for tokens, the ID token is verified against OpenAI's
        JWKS (with this attempt's nonce), and the documented record is
        stored under sagent's own auth directory before returning.

        Args:
          output: Stream for user-facing messages. ``None`` uses stdout.
          listener_timeout_sec: Seconds to wait for the browser callback.
          account: Named credential slot.
          manual: Print a URL and prompt for a pasted callback without
            waiting for a browser callback (headless).

        Returns:
          creds: The stored SIWC credential record.

        Raises:
          RuntimeError: The callback listener failed to start or timed out.
          siwc.OAuthTokenError: OpenAI rejected the registration or the
            code exchange.
          siwc.IDTokenVerificationError: The ID token failed verification.
          siwc.PlanUsageNotGrantedError: The sign-in is retained but does
            not authorize ChatGPT plan usage.

        """
        out = output or sys.stdout
        cred_path = credentials_path(_default_credentials_path(), account)
        host = siwc.host_id(cred_path.parent)

        # An existing record makes this a RETURNING sign-in: reuse the
        # issued client id, omit agent_name_hint, and pre-select the
        # account by email. A record without plan usage re-shows the
        # consent screen (errors and recovery: prompt=consent).
        saved: OpenAISubscription.Credentials | None = None
        with contextlib.suppress(FileNotFoundError, ValueError):
            saved = cls.load(account=account)
        reconsent = saved is not None and not siwc.plan_usage_enabled(saved)

        verifier, challenge = pkce_pair()
        state = secrets.token_urlsafe(32)
        nonce = secrets.token_urlsafe(32)

        listener: AuthCodeListener | None = None
        if manual:
            # Headless: no listener runs, so the redirect URI must be
            # predictable before the browser opens. The user copies the
            # landed-on URL back into the terminal.
            redirect_uri = (
                f"http://127.0.0.1:{siwc.MANUAL_CALLBACK_PORT}{siwc.REDIRECT_PATH}"
            )
        else:
            listener = AuthCodeListener(
                state,
                port=0,
                callback_path=siwc.REDIRECT_PATH,
            )
            try:
                listener.start()
            except OSError as e:
                raise RuntimeError(f"Failed to start callback listener: {e}") from e
            redirect_uri = listener.redirect_uri

        url = siwc.authorize_url(
            client_id=saved["client_id"] if saved else siwc.DYNAMIC_CLIENT_ID,
            redirect_uri=redirect_uri,
            state=state,
            nonce=nonce,
            code_challenge=challenge,
            ext_agent_host_id=host,
            agent_name_hint=None if saved else siwc.AGENT_NAME_HINT,
            login_hint=saved.get("email") if saved else None,
            prompt="consent" if reconsent else None,
        )
        out.write(
            "Starting Sign in with ChatGPT for sagent.\n"
            "OpenAI registers sagent as its own client; subscription use "
            "is governed by OpenAI's Terms of Use:\n"
            "  https://openai.com/policies/row-terms-of-use/\n\n"
            f"Open this URL in any browser to authenticate:\n\n  {url}\n\n",
        )
        out.flush()

        if listener is None:
            code, callback_client_id = siwc.parse_manual_callback(
                input("Paste the redirected URL (or code#state) here: "),
                state,
            )
        else:
            try:
                code = listener.wait(listener_timeout_sec)
            finally:
                listener.stop()
            # A first registration returns the issued client id here.
            callback_client_id = listener.query_param("client_id")

        issued_client_id = siwc.resolve_client_id(
            callback_client_id=callback_client_id,
            saved_client_id=saved["client_id"] if saved else None,
        )

        with httpx2.Client() as http:
            grant = siwc.exchange_code(
                http,
                code=code,
                redirect_uri=redirect_uri,
                client_id=issued_client_id,
                code_verifier=verifier,
            )
            id_token = grant.get("id_token")
            if id_token is None or id_token == "":
                raise siwc.OAuthTokenError(
                    "invalid_token_response",
                    "OpenAI did not return a verifiable identity. "
                    "Please sign in again.",
                )
            claims = siwc.validate_id_token(
                id_token,
                client_id=issued_client_id,
                nonce=nonce,
                http=http,
            )

        subject = str(claims.get("sub") or "")
        saved_subject = saved.get("subject") if saved is not None else None
        if saved_subject is not None and saved_subject not in ("", subject):
            # Registration guide section 4: never replace the selected
            # account's registration with another account's identity.
            raise siwc.OAuthTokenError(
                "account_mismatch",
                "This sign-in returned a different ChatGPT account. "
                "Choose the original account or sign in under a new "
                "--account.",
            )

        scopes = grant.get("scopes")
        if scopes is None or len(scopes) == 0:
            raise siwc.OAuthTokenError(
                "invalid_token_response",
                "OpenAI did not confirm the granted permissions. "
                "Start the login again.",
            )
        record: OpenAISubscription.Credentials = {
            "issuer": siwc.ISSUER,
            "subject": subject,
            "client_id": issued_client_id,
            "ext_agent_host_id": host,
            "access_token": grant["access_token"],
            "refresh_token": grant["refresh_token"],
            "token_type": grant["token_type"],
            "expires_in": grant["expires_in"],
            "scopes": scopes,
            "saved_at": siwc.record_saved_at(),
        }
        email = claims.get("email")
        if isinstance(email, str) and email:
            record["email"] = email
        record["id_token"] = id_token
        earliest_refresh_at = grant.get("earliest_refresh_at")
        if earliest_refresh_at is not None and earliest_refresh_at != "":
            record["earliest_refresh_at"] = earliest_refresh_at

        siwc.save_record(record, cred_path)
        if not siwc.plan_usage_enabled(record):
            # Retain the sign-in but refuse inference (errors and
            # recovery): re-running login re-shows the consent screen.
            raise siwc.PlanUsageNotGrantedError(
                "Signed in, but ChatGPT plan usage was not granted. The "
                "sign-in is saved; run login again to approve plan usage, "
                "or switch to the API-key path: --provider OpenAI "
                "--auth env.",
            )
        as_user = record.get("email") or subject[:8] or "unknown account"
        out.write(f"Authenticated as {as_user} (plan usage enabled).\n")
        out.flush()
        return record

    @classmethod
    async def logout(
        cls,
        *,
        account: str | None = None,
        output: IO[str] | None = None,
    ) -> None:
        """Sign out: revoke the refresh token, then clear local tokens.

        Retains the registration (issued client id), the host id, and the
        verified identity in a signed-out tombstone, per the SIWC docs:
        a later sign-in reuses the registration instead of re-registering.

        Args:
          account: Named credential slot.
          output: Stream for user-facing messages. ``None`` uses stdout.

        """
        out = output or sys.stdout
        cred_path = credentials_path(_default_credentials_path(), account)
        try:
            record = await asyncio.to_thread(siwc.load_record, cred_path)
        except (FileNotFoundError, ValueError):
            out.write("No saved sign-in; nothing to sign out.\n")
            return
        if siwc.is_signed_out(record):
            out.write("Already signed out.\n")
            return
        refresh_token = record.get("refresh_token", "")
        if refresh_token:
            async with httpx2.AsyncClient() as http:
                try:
                    await siwc.async_revoke(
                        http,
                        refresh_token=refresh_token,
                        client_id=record["client_id"],
                    )
                except siwc.RevocationUncertainError as e:
                    out.write(f"Warning: {e}\n")
        record.pop("access_token", None)
        record.pop("refresh_token", None)
        record.pop("id_token", None)
        record["signed_out"] = True
        await asyncio.to_thread(siwc.save_record, record, cred_path)
        as_user = record.get("email") or record.get("subject") or "this account"
        out.write(f"Signed out ({as_user}).\n")
        out.flush()

    def model(
        self,
        model_id: str | None = None,
    ) -> _OpenAISubModel:
        """Create a Responses API model backend.

        Args:
          model_id: Catalog id with optional tags, or a role name.

        Returns:
          model: Responses API model backend.

        Raises:
          UnknownModelError: ``model_id`` is not in the catalog.
          UnsupportedTagError: The id asks for a tag this model or
              transport does not offer.

        """
        mid = model_id if model_id is not None else "default"
        # ``+1m`` is absent from the narrowed subscription catalog, so a
        # ``+1m`` id raises rather than silently clamping: the suffix buys
        # nothing under the subscription wire contract.
        capability, settings = self.catalog.resolve(mid)
        return _OpenAISubModel(
            provider=self,
            capability=capability,
            settings=settings,
        )

    # -- Token management ----------------------------------------------

    @property
    def expired(self) -> bool:
        """True if the access token is within ``refresh_buffer_sec`` of expiry."""
        return time.time() > self._expires_at - self._refresh_buffer_sec

    async def get_sdk(self) -> openai.AsyncOpenAI:
        """Return OAuth-authed SDK client, refreshing as needed.

        The base URL is the documented SIWC inference endpoint; the
        request carries only the Bearer access token -- the Codex-backend
        headers (``chatgpt-account-id``, ``originator``) are gone with the
        legacy flow.

        Returns:
          client: Authenticated ``AsyncOpenAI`` SDK client.

        """
        token = await self._ensure_valid()
        cached = self._authed.peek()
        if cached is not None and cached[1] == token:
            return cached[0]
        # No refresh here: ``_ensure_valid`` is the one path that holds the
        # cross-process credential lock, and a refresh outside it lets two
        # processes spend one rotating refresh token.
        async with self._lock.get():
            token = self._access_token
            cached = self._authed.peek()
            if cached is not None and cached[1] == token:
                return cached[0]
            sdk = openai.AsyncOpenAI(
                api_key=token,
                base_url=siwc.RESOURCE,
            )
            self._authed.set((sdk, token))
            if cached is not None:
                await cached[0].close()
            return sdk

    async def close_sdk(self) -> None:
        """Close and clear the shared OAuth SDK client.

        Idempotent: a no-op when no SDK has been created. This loop's
        client only -- a pool belongs to the loop that opened it, so
        closing another loop's from here would break a pool still in use.
        """
        async with self._lock.get():
            cached = self._authed.peek()
            self._authed.clear()
        if cached is not None:
            await cached[0].close()

    # A concurrent process holding the same account may have refreshed and written newer
    # creds to disk. Load them under the already-held file lock and adopt only when the
    # on-disk access token (a) DIFFERS from ours and (b) is itself non-expired. The
    # boolean answers exactly "did we just adopt a token we can use without a network
    # refresh?" -- never "is our current token valid?".
    #
    # Two failure modes this guards, both of which return False so the caller refreshes:
    # - disk token MATCHES ours: a sibling did not refresh; our token is the one that
    #   needs replacing (especially under ``handle_auth_error``, where the server already
    #   rejected it regardless of the local clock). - disk token DIFFERS but is itself
    #   already expired (its own refresh aged out, or clock skew): adopting it would 401
    #   again next call. A signed-out tombstone on disk (a sibling signed out) also lands
    #   here via the empty access token.
    #
    # ``siwc.load_record`` raises ``ValueError`` on malformed JSON or an incomplete
    # record; it means "no usable disk creds", so it returns False.
    #
    # Disk I/O runs in a worker thread so the event loop is not blocked on a slow/NFS
    # read while the credential lock is held. Adopting a new token invalidates the
    # cached SDK; the stale client is closed (not merely dropped) so its pooled HTTP
    # connections are released immediately rather than orphaned until GC.
    async def _adopt_fresher_disk_creds(self) -> bool:
        """Adopt a DIFFERENT, still-valid sibling-written disk token."""
        try:
            creds = await asyncio.to_thread(
                siwc.load_record,
                self._cred_path(),
            )
            disk_at = creds.get("access_token", "")
        except (FileNotFoundError, ValueError):
            return False
        if not disk_at or disk_at == self._access_token:
            return False
        self._access_token = disk_at
        self._refresh_token = creds.get("refresh_token", self._refresh_token)
        self._client_id = creds["client_id"]
        self._expires_at = _record_expiry(creds)
        if creds["scopes"]:
            self._scopes = list(creds["scopes"])
        await self._discard_sdk()
        return not self.expired

    # Idempotent. The caller must already hold ``self._lock``.
    async def _discard_sdk(self) -> None:
        """Drop and close the cached SDK so its pooled connections release."""
        cached = self._authed.peek()
        self._authed.clear()
        if cached is not None:
            await cached[0].close()

    # Holds a cross-process file lock around the read-disk -> maybe-POST -> write-disk
    # sequence so concurrent processes can't both consume the same refresh_token and
    # have one revoked by the OAuth endpoint's rotation rule.
    async def _ensure_valid(self) -> str:
        """Return a valid access token, reloading from disk or refreshing."""
        if not self.expired:
            return self._access_token
        async with self._lock.get(), credential_file_lock(self._cred_path()):
            if not self.expired:
                return self._access_token
            if await self._adopt_fresher_disk_creds():
                return self._access_token
            await self._refresh()
            return self._access_token

    async def handle_auth_error(self) -> None:
        """Reload credentials from disk or force-refresh on 401.

        Holds the cross-process credential lock so a concurrent
        sibling can't refresh between our disk-check and POST.
        """
        async with self._lock.get(), credential_file_lock(self._cred_path()):
            # Adopt a sibling's fresher creds only if they are actually valid;
            # an adopted-but-expired token would 401 again on retry, so fall
            # through to ``_refresh`` (the same rule ``_ensure_valid`` uses).
            if await self._adopt_fresher_disk_creds():
                return
            await self._refresh()
            await self._discard_sdk()

    async def _refresh(self) -> None:
        """Exchange the refresh token for a new grant and persist it.

        The caller must hold the cross-process credential lock. A
        refresh-issued ID token is verified too (no nonce; DevKit
        parity): an unreachable key service only warns -- availability is
        not invalidity -- while a hard verification failure re-raises as
        a re-authentication error.
        """
        async with httpx2.AsyncClient() as http:
            grant = await siwc.async_refresh_grant(
                http,
                refresh_token=self._refresh_token,
                client_id=self._client_id,
            )
        id_token = grant.get("id_token")
        if id_token is not None and id_token != "":
            try:
                claims = await asyncio.to_thread(
                    siwc.validate_id_token,
                    id_token,
                    client_id=self._client_id,
                )
            except siwc.JWKSUnavailableError as e:
                logger.warning(
                    "Skipping refreshed ID-token verification: %s",
                    e,
                )
            except siwc.IDTokenVerificationError as e:
                raise AuthRefreshError(
                    f"OpenAI rejected the refreshed session ({e}). "
                    "Run /login to re-authenticate, or /model to switch "
                    "providers.",
                ) from e
            else:
                subject = str(claims.get("sub") or "")
                if self._subject and subject and subject != self._subject:
                    raise AuthRefreshError(
                        "The refreshed ChatGPT identity does not match "
                        "this account. Run /login and choose the original "
                        "account.",
                    )
                if subject:
                    self._subject = subject
                email = claims.get("email")
                if isinstance(email, str) and email:
                    self._email = email
        self._access_token = grant["access_token"]
        self._refresh_token = grant["refresh_token"]
        self._expires_at = _grant_expiry(self._access_token, grant["expires_in"])
        self._scopes = list(grant.get("scopes") or self._scopes)
        await asyncio.to_thread(self._persist_refreshed_record)

    def _persist_refreshed_record(self) -> None:
        """Merge the refreshed tokens into the on-disk record.

        The caller holds the cross-process credential lock (via
        ``_ensure_valid``/``handle_auth_error``); this runs in a worker
        thread. An unreadable record is rebuilt from in-memory state
        rather than lost, logging first so the corruption is not silently
        erased (the same policy as the legacy flow).
        """
        cred_path = self._cred_path()
        record: siwc.SIWCCredentials
        try:
            record = siwc.load_record(cred_path)
        except (FileNotFoundError, ValueError):
            logger.warning(
                "SIWC record at %s was unreadable; rewriting from memory",
                cred_path,
            )
            record = {
                "issuer": siwc.ISSUER,
                "client_id": self._client_id,
                "ext_agent_host_id": siwc.host_id(cred_path.parent),
                "scopes": list(self._scopes),
            }
        record["access_token"] = self._access_token
        record["refresh_token"] = self._refresh_token
        expires_at = _jwt_exp(self._access_token)
        if expires_at > 0:
            record["expires_in"] = expires_at - time.time()
        record["scopes"] = list(self._scopes)
        record["saved_at"] = siwc.record_saved_at()
        if self._subject:
            record["subject"] = self._subject
        if self._email:
            record["email"] = self._email
        siwc.save_record(record, cred_path)

    # -- Credential I/O ------------------------------------------------

    @classmethod
    def load(
        cls,
        path: Path | None = None,
        *,
        account: str | None = None,
    ) -> OpenAISubscription.Credentials:
        """Load SIWC credentials from disk.

        Args:
          path: Explicit record path, or ``None`` for the account slot.
          account: Named credential slot. ``None`` reads the unnamed
            slot in sagent's own auth directory.

        Returns:
          creds: Loaded SIWC credential record.

        Raises:
          FileNotFoundError: If no record exists.
          ValueError: If the file is not a usable SIWC record.

        """
        p = path or credentials_path(_default_credentials_path(), account)
        record = siwc.load_record(p)
        if siwc.is_signed_out(record):
            raise _CredentialFileError(
                f"{p} is a signed-out record. Run `sagent --provider "
                "OpenAISubscription login` to sign in again.",
            )
        return record

    @classmethod
    def save(
        cls,
        creds: OpenAISubscription.Credentials,
        path: Path | None = None,
        *,
        account: str | None = None,
    ) -> None:
        """Persist SIWC credentials atomically with owner-only mode.

        Args:
          creds: SIWC credential record to persist.
          path: Explicit record path, or ``None`` for the account slot.
          account: Named credential slot.

        """
        p = path or credentials_path(_default_credentials_path(), account)
        siwc.save_record(creds, p)


class _OpenAISubModel(_OpenAIResponsesModel):
    """OAuth retry and plan-usage request restrictions."""

    _provider: OpenAISubscription

    @override
    def _build_kwargs(self, request: ModelRequest) -> ResponseCreateParamsStreaming:
        body = super()._build_kwargs(request)
        # SIWC preview limitations: temperature and max_output_tokens are
        # omitted from the plan-backed Responses wire contract.
        body.pop("temperature", None)
        body.pop("max_output_tokens", None)
        return body

    @override
    async def stream(
        self,
        request: ModelRequest,
        publish: Callable[[RuntimeEvent], None] | None = None,
    ) -> ModelResponse:
        try:
            return await super().stream(request, publish)
        except openai.AuthenticationError:
            await self._provider.handle_auth_error()
        except openai.APIStatusError as e:
            mapped = _plan_usage_error(e)
            if mapped is not None:
                raise mapped from e
            raise
        return await super().stream(request, publish)


def _plan_usage_error(
    error: openai.APIStatusError,
) -> siwc.SubscriptionUsageError | None:
    """Map a documented SIWC plan-usage error code to polished recovery text.

    Args:
      error: Status error raised by the Responses request.

    Returns:
      mapped: A user-facing error for a documented ``subscription_sharing``
        code, or ``None`` to let the original propagate (transient 503
        availability codes keep the provider's retry behavior).

    """
    text = str(error)
    for code, message in siwc.PLAN_ERROR_RECOVERY.items():
        if code in text:
            return siwc.SubscriptionUsageError(code, message)
    return None


def _grant_expiry(access_token: str, expires_in: float) -> float:
    """Return the access token's deadline: JWT ``exp``, else grant lifetime.

    ``exp`` wins because ``_record_expiry`` re-derives expiry from the
    token alone; a local-clock deadline would disagree with it under
    clock skew.
    """
    expires_at = _jwt_exp(access_token)
    if expires_at > 0:
        return expires_at
    # Defensive only: siwc grant parsing rejects non-positive lifetimes.
    return time.time() + expires_in if expires_in > 0 else 0.0


def _record_expiry(record: siwc.SIWCCredentials) -> float:
    """Derive the access token deadline from a stored record.

    ``exp`` from the token itself when present (robust to save-time clock
    skew); otherwise ``saved_at + expires_in``; otherwise ``0.0`` so the
    provider treats the token as expired and refreshes on first use.
    """
    expires_at = _jwt_exp(record.get("access_token", ""))
    if expires_at > 0:
        return expires_at
    saved_at = record.get("saved_at", "")
    expires_in = record.get("expires_in", 0.0)
    if saved_at and expires_in > 0:
        try:
            saved = datetime.fromisoformat(saved_at)
        except ValueError:
            return 0.0
        return saved.timestamp() + expires_in
    return 0.0


# Every JWT read below is lenient: a token the issuer shaped unexpectedly reads as
# "no claim", never as an exception, so callers keep their error contracts and the
# refresh paths stay reachable.
def _jwt_payload(token: str) -> dict[str, object]:
    """Decode JWT payload without verification."""
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    raw = parts[1]
    raw += "=" * (-len(raw) % 4)
    try:
        return from_plain(loads(base64.urlsafe_b64decode(raw)), dict[str, object])
    except (ValueError, ReadError):
        return {}


def _jwt_exp(token: str) -> float:
    """Extract ``exp`` claim from a JWT without verification."""
    try:
        return from_plain(_jwt_payload(token).get("exp"), float, default=0.0)
    except ReadError:
        return 0.0
