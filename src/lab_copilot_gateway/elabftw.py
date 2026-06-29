"""eLabFTW read adapter for the lab copilot gateway (C08).

Orchestrates the gateway-enforced path for reading eLabFTW experiment context
on behalf of a mapped user.  This module establishes the orchestration pattern
that C09 (eLabFTW bounded write), C16 (OpenCloning), and C18 (Wallac) will
follow:

    context token verification
        → identity resolution (mapped user required)
        → policy engine decision (tier 2 read)
        → audit record with context_refs and tool_args_hash
        → downstream client call (HTTP for production, stub for tests)

Invariants enforced before any downstream HTTP call:

    * Unmapped callers fail closed — the identity mapper returns None and the
      adapter raises ``UnmappedCaller`` without ever touching the network.
    * The context token must be signed by the gateway's own secret and must
      not be expired.  Tampered expiry, missing signature, or foreign-token
      reuse all raise ``InvalidContextToken``.  LibreChat cannot forge a
      token because it never sees the gateway secret.
    * The mapped_elabftw_user_id in the token must agree with the resolved
      identity.  A token issued to user A cannot be used by user B's session.
    * Every call — success or failure — persists an ``AuditRecord`` with the
      tool_name, tool_args_hash, context_refs (the experiment id and any
      resolved resource refs), and policy_decision.

The HTTP client follows the established lab pattern (see AGENTS.md →
"eLabFTW API — HTTP client patterns"): ``requests.Session()`` with an
``Authorization`` header set to the API key (no Bearer prefix) and
``verify=False`` for dev instances with self-signed certs.  The client is
injectable for tests via the ``ElabftwClient`` Protocol.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from lab_copilot_gateway.approval import compute_args_hash
from lab_copilot_gateway.audit import AuditRecord, AuditStore
from lab_copilot_gateway.identity import MappedIdentity
from lab_copilot_gateway.policy import Decision, PolicyEngine, PolicyRequest, Tier


# --- configuration --------------------------------------------------------


def _env_secret() -> bytes:
    """Return the gateway's context-token HMAC secret.

    Sourced from ``LAB_COPILOT_TOKEN_SECRET`` (must be set per-process for
    production stability).  If the env var is empty, a single ephemeral
    dev-only secret is generated *once per process* (cached at module scope)
    so tokens minted in the same process can verify.  Production deployments
    MUST set the env var to a stable high-entropy value.
    """
    raw = os.getenv("LAB_COPILOT_TOKEN_SECRET", "")
    if raw:
        return raw.encode("utf-8")
    return _dev_ephemeral_secret()


# Module-cached dev-only secret.  Generated once on first access when the
# env var is unset, then reused for the lifetime of the process.  DO NOT
# rely on this in production — set LAB_COPILOT_TOKEN_SECRET.
_dev_ephemeral_secret_cache: bytes | None = None


def _dev_ephemeral_secret() -> bytes:
    """Return one stable ephemeral secret per process (cached).

    Re-use across calls into this module so tokens minted in the same process
    can verify.  Tests should still monkeypatch ``LAB_COPILOT_TOKEN_SECRET``
    explicitly to avoid coupling to this path.
    """
    global _dev_ephemeral_secret_cache
    if _dev_ephemeral_secret_cache is None:
        import secrets

        _dev_ephemeral_secret_cache = secrets.token_bytes(32)
    return _dev_ephemeral_secret_cache


# --- exceptions -----------------------------------------------------------


class ElabftwAdapterError(Exception):
    """Base class for eLabFTW adapter errors.

    ``reason`` is a machine-readable code; ``message`` is human-readable.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message

    def to_dict(self) -> dict[str, str]:
        return {"reason": self.reason, "message": self.message}


class UnmappedCaller(ElabftwAdapterError):
    """Caller has no mapped eLabFTW identity — fail closed before any HTTP."""

    def __init__(self) -> None:
        super().__init__(
            reason="unmapped_caller",
            message="caller has no mapped eLabFTW identity",
        )


class InvalidContextToken(ElabftwAdapterError):
    """Context token signature, expiry, or claim is invalid.

    ``reason`` keeps the stable machine code; ``detail`` is the human-readable
    sub-cause (e.g. "expired", "bad signature", "user mismatch").
    """

    def __init__(self, detail: str) -> None:
        super().__init__(
            reason="invalid_context_token",
            message=f"context token invalid: {detail}",
        )
        self.detail = detail


class PolicyDenied(ElabftwAdapterError):
    """Policy engine refused the read — surfaced as an adapter error type so
    audit + caller can treat the deny path uniformly across adapters."""

    def __init__(self, decision: Decision) -> None:
        super().__init__(
            reason=f"policy_denied:{decision.reason}",
            message=f"policy engine denied read ({decision.reason})",
        )


# --- context token --------------------------------------------------------


@dataclass(frozen=True)
class ContextTokenClaims:
    """Claims bound into a signed context token.

    The eLabFTW launcher (C11) mints a short-lived token carrying these claims;
    the gateway verifies the signature and expiry before any read.
    """

    experiment_id: int
    mapped_elabftw_user_id: str
    keycloak_subject: str | None
    librechat_user_id: str | None
    issued_at: str  # ISO8601 UTC
    expires_at: str  # ISO8601 UTC


def _canonical_claims(claims: ContextTokenClaims) -> bytes:
    """Canonical JSON of the claim set, byte-stable for HMAC verification."""
    payload = {
        "experiment_id": claims.experiment_id,
        "mapped_elabftw_user_id": claims.mapped_elabftw_user_id,
        "keycloak_subject": claims.keycloak_subject,
        "librechat_user_id": claims.librechat_user_id,
        "issued_at": claims.issued_at,
        "expires_at": claims.expires_at,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def mint_context_token(
    claims: ContextTokenClaims, *, secret: bytes | None = None
) -> str:
    """Mint a signed context token.  Returns ``<payload_b64>.<sig_hex>``.

    The token is NOT a JWT — the gateway owns mint and verify, never the
    client.  Format is intentionally simple to keep the parsing surface tight.
    """
    sec = secret if secret is not None else _env_secret()
    payload_bytes = _canonical_claims(claims)
    payload_b64 = _b64url(payload_bytes)
    sig = hmac.new(sec, payload_b64.encode("ascii"), hashlib.sha256).hexdigest()
    return f"{payload_b64}.{sig}"


def verify_context_token(
    token: str, *, secret: bytes | None = None
) -> ContextTokenClaims:
    """Verify signature + expiry, return the claims.

    Raises ``InvalidContextToken`` on any failure (bad signature, expired,
    malformed encoding, missing field, malformed claim types).  Compares
    signatures with ``hmac.compare_digest`` to avoid timing leaks.
    """
    if not token or "." not in token:
        raise InvalidContextToken("malformed token")
    payload_b64, _, sig_hex = token.partition(".")
    if not payload_b64 or not sig_hex:
        raise InvalidContextToken("malformed token")

    sec = secret if secret is not None else _env_secret()
    expected_sig = hmac.new(
        sec, payload_b64.encode("ascii"), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected_sig, sig_hex):
        raise InvalidContextToken("bad signature")

    try:
        payload_bytes = _unb64url(payload_b64)
        payload = json.loads(payload_bytes.decode("utf-8"))
    except Exception as exc:
        raise InvalidContextToken(f"malformed payload: {exc}") from exc

    required = {
        "experiment_id",
        "mapped_elabftw_user_id",
        "issued_at",
        "expires_at",
    }
    missing = required - set(payload.keys())
    if missing:
        raise InvalidContextToken(f"missing claims: {sorted(missing)}")

    # Coerce claim types under a guard so a signed-but-malformed payload
    # surfaces InvalidContextToken instead of leaking TypeError/ValueError.
    try:
        experiment_id = int(payload["experiment_id"])
        mapped_elabftw_user_id = str(payload["mapped_elabftw_user_id"])
        issued_at = str(payload["issued_at"])
        expires_at = str(payload["expires_at"])
    except (TypeError, ValueError) as exc:
        raise InvalidContextToken(f"malformed claim types: {exc}") from exc

    # Parse expiry into an aware UTC datetime and compare against now.  Using
    # datetime comparison avoids ISO8601 string comparison pitfalls (e.g. non-UTC
    # offsets, fractional precision differences between issuers).
    try:
        expires_dt = _dt.datetime.fromisoformat(expires_at)
    except ValueError as exc:
        raise InvalidContextToken(f"malformed expires_at: {exc}") from exc
    now_dt = _dt.datetime.now(_dt.timezone.utc)
    if expires_dt.tzinfo is None:
        # Naive timestamps are interpreted as UTC.
        expires_dt = expires_dt.replace(tzinfo=_dt.timezone.utc)
    if now_dt >= expires_dt:
        raise InvalidContextToken("expired")

    return ContextTokenClaims(
        experiment_id=experiment_id,
        mapped_elabftw_user_id=mapped_elabftw_user_id,
        keycloak_subject=payload.get("keycloak_subject"),
        librechat_user_id=payload.get("librechat_user_id"),
        issued_at=issued_at,
        expires_at=expires_at,
    )


def _b64url(data: bytes) -> str:
    import base64

    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64url(s: str) -> bytes:
    import base64

    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _redact(token: str) -> str:
    """Return a short stable digest of a context token for audit hashing.

    We never store the raw token in the audit row.  A 16-hex-char prefix of
    the SHA-256 of the token is enough to correlate audit rows to a specific
    token issuance without leaking the token itself.
    """
    if not token:
        return "missing"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


def _normalize_metadata(value: Any) -> dict[str, Any] | None:
    """Normalize the eLabFTW ``metadata`` field as the repo's other tools do.

    The eLabFTW REST API may return ``metadata`` as:
        * a dict already (most modern responses)
        * a JSON-encoded string (older API responses)
        * a JSON-encoded JSON string (double-encoded — common drift)
        * None when the experiment has no metadata

    Returns ``None`` when the input is ``None`` so callers can distinguish.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return {"_raw": value}
        if isinstance(decoded, str):
            # Double-encoded JSON — decode one more level.
            try:
                decoded = json.loads(decoded)
            except json.JSONDecodeError:
                return {"_raw": decoded}
        if isinstance(decoded, dict):
            return decoded
        return {"_raw": decoded}
    return {"_raw": str(value)}


# --- downstream client ---------------------------------------------------


class ElabftwClient(Protocol):
    """eLabFTW REST client surface the read adapter requires."""

    def get_experiment(self, experiment_id: int) -> dict[str, Any]:
        """Return one experiment payload by id, or raise on error.

        Implementations: ``StubElabftwClient`` for tests; ``HttpElabftwClient``
        for live calls against an eLabFTW instance.
        """
        ...


@dataclass
class StubElabftwClient:
    """In-memory ``ElabftwClient`` for tests and offline dev.

    Pre-seed with ``seeds`` mapping experiment_id → payload dict.  Unknown
    experiment_ids raise ``KeyError`` (raised by dict access — callers can
    catch KeyError to test the missing-record path).
    """

    seeds: dict[int, dict[str, Any]] = field(default_factory=dict)

    def get_experiment(self, experiment_id: int) -> dict[str, Any]:
        if experiment_id not in self.seeds:
            raise KeyError(f"experiment {experiment_id} not found in stub")
        return dict(self.seeds[experiment_id])


@dataclass
class HttpElabftwClient:
    """Live eLabFTW REST client matching the established lab pattern.

    See AGENTS.md → "eLabFTW API — HTTP client patterns":
        * ``requests.Session`` with ``Authorization`` header set to the API
          key (no Bearer prefix).
        * ``verify=False`` for dev instances with self-signed certs.
    """

    base_url: str
    api_key: str
    verify_tls: bool = False
    timeout_seconds: float = 10.0
    _session: Any = None  # lazy-built requests.Session

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("HttpElabftwClient requires non-empty base_url")
        if not self.api_key:
            raise ValueError("HttpElabftwClient requires non-empty api_key")

    def _connect(self) -> Any:
        if self._session is None:
            import requests  # local import; not all deployments exercise HTTP

            s = requests.Session()
            s.headers.update({"Authorization": self.api_key})
            s.verify = self.verify_tls
            self._session = s
        return self._session

    def get_experiment(self, experiment_id: int) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/api/v2/experiments/{int(experiment_id)}"
        s = self._connect()
        resp = s.get(url, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()


def _default_client_from_env() -> ElabftwClient | None:
    """Construct the live HTTP client if env config is present.

    Returns None if base_url or api_key is unset — the adapter then refuses
    reads with ``ElabftwNotConfigured`` at call time.
    """
    base_url = os.getenv("LAB_COPILOT_ELABFTW_BASE_URL", "")
    api_key = os.getenv("LAB_COPILOT_ELABFTW_API_KEY", "")
    if not base_url or not api_key:
        return None
    verify_tls = os.getenv("LAB_COPILOT_ELABFTW_VERIFY_TLS", "0").lower() in {
        "1",
        "true",
        "yes",
    }
    return HttpElabftwClient(
        base_url=base_url,
        api_key=api_key,
        verify_tls=verify_tls,
        timeout_seconds=float(os.getenv("LAB_COPILOT_ELABFTW_TIMEOUT", "10.0")),
    )


# --- read result ---------------------------------------------------------


@dataclass(frozen=True)
class ReadResult:
    """Outcome of a successful ``ElabftwReadAdapter.read_current_experiment``."""

    experiment_id: int
    title: str
    body: str
    metadata: dict[str, Any] | None
    raw: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "title": self.title,
            "body": self.body,
            "metadata": self.metadata,
        }


# --- adapter -------------------------------------------------------------


TOOL_NAME = "elabftw.read_current_experiment"
TOOL_TIER = Tier.PERMISSIONED_ELABFTW_READ


@dataclass
class ElabftwReadAdapter:
    """Orchestrates eLabFTW reads through the gateway-enforced path.

    Dependencies are injected so tests can swap the policy engine, identity
    mapper, audit store, and downstream client individually.  In production,
    the module-level ``get_elabftw_read_adapter()`` wires all the
    singletons.
    """

    policy_engine: PolicyEngine
    audit_store: AuditStore
    client: ElabftwClient | None
    action_id_prefix: str = "elab-read"

    def read_current_experiment(
        self,
        *,
        context_token: str,
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None = None,
        request_id: str | None = None,
        keycloak_subject: str | None = None,
        librechat_user_id: str | None = None,
        provider: str | None = None,
        model_id: str | None = None,
    ) -> ReadResult:
        """Read the experiment referenced by ``context_token``.

        See module docstring for the orchestration order.  The adapter always
        persists an audit record (policy_decision allow OR deny) before
        returning or raising, so every read is observable.
        """
        # Compute a request-level args hash up front.  When the token has not
        # yet been verified we hash a redacted token digest so the audit row
        # still carries a stable hash without leaking the raw token.
        early_hash = (
            compute_args_hash(
                {"experiment_id": None, "token_digest": _redact(context_token)}
            )
            if context_token
            else compute_args_hash({"token_digest": "missing"})
        )

        # 1. Resolve token claims (signature + expiry verified inside).
        if not context_token:
            self._audit(
                policy_decision="deny",
                reason="missing_context_token",
                mapped_identity=mapped_identity,
                experiment_id=None,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=early_hash,
                error={"code": "MISSING_CONTEXT_TOKEN"},
            )
            raise InvalidContextToken("missing")

        try:
            claims = verify_context_token(context_token)
        except InvalidContextToken as exc:
            self._audit(
                policy_decision="deny",
                reason=f"invalid_context_token:{exc.detail}",
                mapped_identity=mapped_identity,
                experiment_id=None,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=early_hash,
                error={"code": "INVALID_CONTEXT_TOKEN", "detail": exc.detail},
            )
            raise

        # Once claims are verified, the args hash is deterministic on
        # the read experiment_id — replace the early redacted hash.
        tool_args_hash = compute_args_hash({"experiment_id": claims.experiment_id})

        # 2. Identity resolution.  Unmapped caller → deny before any HTTP.
        if mapped_identity is None:
            self._audit(
                policy_decision="deny",
                reason="unmapped_caller",
                mapped_identity=None,
                experiment_id=claims.experiment_id,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=tool_args_hash,
                error={"code": "UNMAPPED_CALLER"},
            )
            raise UnmappedCaller()

        # 3. Token-identity binding.  The token's mapped user MUST match the
        # resolved identity.  When the token also carries keycloak_subject or
        # librechat_user_id (issued alongside the identity at C11 launcher
        # time), cross-check those too — prevents cross-user token replay.
        if claims.mapped_elabftw_user_id != mapped_identity.elabftw_user_id:
            self._audit(
                policy_decision="deny",
                reason="context_token_user_mismatch",
                mapped_identity=mapped_identity,
                experiment_id=claims.experiment_id,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=tool_args_hash,
                error={"code": "CONTEXT_TOKEN_USER_MISMATCH"},
            )
            raise InvalidContextToken("token user does not match resolved identity")

        if (
            claims.keycloak_subject is not None
            and mapped_identity.keycloak_subject is not None
        ):
            if claims.keycloak_subject != mapped_identity.keycloak_subject:
                self._audit(
                    policy_decision="deny",
                    reason="context_token_kc_subject_mismatch",
                    mapped_identity=mapped_identity,
                    experiment_id=claims.experiment_id,
                    conversation_id=conversation_id,
                    request_id=request_id,
                    keycloak_subject=keycloak_subject,
                    librechat_user_id=librechat_user_id,
                    provider=provider,
                    model_id=model_id,
                    tool_args_hash=tool_args_hash,
                    error={"code": "CONTEXT_TOKEN_KC_SUBJECT_MISMATCH"},
                )
                raise InvalidContextToken(
                    "token keycloak_subject does not match resolved identity"
                )

        if (
            claims.librechat_user_id is not None
            and mapped_identity.librechat_user_id is not None
        ):
            if claims.librechat_user_id != mapped_identity.librechat_user_id:
                self._audit(
                    policy_decision="deny",
                    reason="context_token_lc_user_mismatch",
                    mapped_identity=mapped_identity,
                    experiment_id=claims.experiment_id,
                    conversation_id=conversation_id,
                    request_id=request_id,
                    keycloak_subject=keycloak_subject,
                    librechat_user_id=librechat_user_id,
                    provider=provider,
                    model_id=model_id,
                    tool_args_hash=tool_args_hash,
                    error={"code": "CONTEXT_TOKEN_LC_USER_MISMATCH"},
                )
                raise InvalidContextToken(
                    "token librechat_user_id does not match resolved identity"
                )

        # 4. Policy decision.  Tier 2 read with mapped caller.
        policy_req = PolicyRequest(
            tool_name=TOOL_NAME,
            tier=TOOL_TIER,
            adapter="elabftw",
            user_id=mapped_identity.elabftw_user_id,
            team_id=mapped_identity.elabftw_team_id,
            has_approval=False,
            approval_id=None,
        )
        decision = self.policy_engine.decide(policy_req)

        if decision.decision != "allow":
            self._audit(
                policy_decision="deny",
                reason=decision.reason,
                mapped_identity=mapped_identity,
                experiment_id=claims.experiment_id,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=tool_args_hash,
                error={
                    "code": "POLICY_DENIED",
                    "reason": decision.reason,
                    "tier": decision.tier,
                },
            )
            raise PolicyDenied(decision)

        # 5. Downstream client.  If the env didn't configure a client, raise
        # after audit so a misconfigured gateway fails observably.
        if self.client is None:
            self._audit(
                policy_decision="deny",
                reason="elabftw_not_configured",
                mapped_identity=mapped_identity,
                experiment_id=claims.experiment_id,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=tool_args_hash,
                error={"code": "ELABFTW_NOT_CONFIGURED"},
            )
            raise ElabftwAdapterError(
                reason="elabftw_not_configured",
                message="eLabFTW client not configured (set LAB_COPILOT_ELABFTW_BASE_URL and LAB_COPILOT_ELABFTW_API_KEY)",
            )

        # 6. Make the call.  Wrap any downstream failure in an
        # ElabftwAdapterError after the audit record persists so the HTTP
        # endpoint returns the documented {ok:false, reason, message} shape.
        try:
            payload = self.client.get_experiment(claims.experiment_id)
        except Exception as exc:
            self._audit(
                policy_decision="allow",
                reason="client_error",
                mapped_identity=mapped_identity,
                experiment_id=claims.experiment_id,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=tool_args_hash,
                error={
                    "code": "CLIENT_ERROR",
                    "exception": type(exc).__name__,
                    "message": str(exc),
                },
            )
            raise ElabftwAdapterError(
                reason="client_error",
                message=f"eLabFTW client raised {type(exc).__name__}: {exc}",
            ) from exc

        # 7. Build the read result + successful audit record.  Normalize
        # metadata the way the repo's other eLabFTW tools do — the API may
        # return metadata as a JSON-encoded string under the "metadata" key,
        # and that string may itself be a JSON-encoded JSON string.
        result = ReadResult(
            experiment_id=claims.experiment_id,
            title=str(payload.get("title", "")),
            body=str(payload.get("body", "")),
            metadata=_normalize_metadata(payload.get("metadata")),
            raw=payload,
        )

        self._audit(
            policy_decision="allow",
            reason="read_succeeded",
            mapped_identity=mapped_identity,
            experiment_id=claims.experiment_id,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            tool_args_hash=tool_args_hash,
            api_call_summary={
                "method": "GET",
                "path": f"/api/v2/experiments/{claims.experiment_id}",
            },
            result_summary={
                "title_length": len(result.title),
                "body_length": len(result.body),
            },
            require_persistence=True,  # success path: audit must persist
        )
        return result

    # --- audit writer ----------------------------------------------------

    def _audit(
        self,
        *,
        policy_decision: str,
        reason: str,
        mapped_identity: MappedIdentity | None,
        experiment_id: int | None,
        conversation_id: str | None,
        request_id: str | None,
        keycloak_subject: str | None,
        librechat_user_id: str | None,
        provider: str | None,
        model_id: str | None,
        tool_args_hash: str | None = None,
        api_call_summary: dict[str, Any] | None = None,
        result_summary: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        require_persistence: bool = False,
    ) -> None:
        """Persist an audit record.

        On deny / error paths (``require_persistence=False``, the default),
        an audit-write failure is swallowed — those are observability rows
        for already-failing calls, not the source of truth for the deny.

        On the success path (``require_persistence=True``), an audit-write
        failure raises ``ElabftwAdapterError`` — a read that succeeds without
        an audit trail violates the gateway's safety contract and must not
        return data to the caller.
        """
        import secrets as _secrets
        import time as _time

        action_id = f"{self.action_id_prefix}-{int(_time.time() * 1000)}-{_secrets.token_hex(4)}"
        context_refs: list[dict[str, Any]] = []
        if experiment_id is not None:
            context_refs.append({"kind": "experiment", "id": experiment_id})

        record = AuditRecord(
            action_id=action_id,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            mapped_elabftw_user_id=(
                mapped_identity.elabftw_user_id if mapped_identity else None
            ),
            mapped_elabftw_team_id=(
                mapped_identity.elabftw_team_id if mapped_identity else None
            ),
            provider=provider,
            model_id=model_id,
            tool_name=TOOL_NAME,
            tool_args_hash=tool_args_hash,
            context_refs=context_refs,
            policy_decision=policy_decision,
            api_call_summary=api_call_summary or {},
            result_summary=result_summary or {},
            error=error,
        )
        try:
            self.audit_store.append(record)
        except Exception as exc:
            if require_persistence:
                raise ElabftwAdapterError(
                    reason="audit_persistence_failed",
                    message=f"audit append failed on success path: {exc}",
                ) from exc
            # Reason: audit failures must not mask a deny / client-error
            # outcome — those are already failing calls; the audit row is
            # observability, not enforcement on those paths.
            pass


# --- module-level singleton for dependency injection ----------------------
_default_adapter: ElabftwReadAdapter | None = None


def get_elabftw_read_adapter() -> ElabftwReadAdapter:
    """Return the process-wide eLabFTW read adapter (created lazily)."""
    global _default_adapter
    if _default_adapter is None:
        from lab_copilot_gateway.audit import get_audit_store
        from lab_copilot_gateway.policy import get_policy_engine

        _default_adapter = ElabftwReadAdapter(
            policy_engine=get_policy_engine(),
            audit_store=get_audit_store(),
            client=_default_client_from_env(),
        )
    return _default_adapter


def reset_elabftw_read_adapter(adapter: ElabftwReadAdapter | None = None) -> None:
    """Test helper: replace or clear the singleton."""
    global _default_adapter
    _default_adapter = adapter
