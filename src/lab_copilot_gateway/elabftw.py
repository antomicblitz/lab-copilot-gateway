"""eLabFTW adapters for the lab copilot gateway (C08 read, C09 bounded write).

Orchestrates the gateway-enforced path for reading eLabFTW experiment context
and appending approved amendments to it on behalf of a mapped user.  This
module establishes the orchestration pattern that C16 (OpenCloning) and C18
(Wallac) will follow:

    context token verification
        → identity resolution (mapped user required)
        → policy engine decision (tier 2 read, tier 4 bounded write)
        → approval token consume (tier >= 4 mutating path only)
        → append-only enforcement (writes reject locked / archived)
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

# TTL caps for the C11 launcher's mint endpoint.  The launcher requests a
# short-lived token scoped to one experiment; the gateway clamps the
# requested TTL to [1, _MAX_MINT_TTL_SECONDS] and defaults to
# _NORMAL_TTL_SECONDS when no TTL is supplied.  These are module-level so
# tests can monkeypatch if needed; production tuning should be via env
# vars only (added when a need is demonstrated).
_NORMAL_TTL_SECONDS = 300  # 5 minutes — ample for a copilot read+amend.
_MAX_MINT_TTL_SECONDS = 600  # 10 minutes hard cap; launchers may not exceed.


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
    record_type: str = "experiment"  # "experiment" or "resource" (database item)


def _canonical_claims(claims: ContextTokenClaims) -> bytes:
    """Canonical JSON of the claim set, byte-stable for HMAC verification."""
    payload = {
        "experiment_id": claims.experiment_id,
        "mapped_elabftw_user_id": claims.mapped_elabftw_user_id,
        "keycloak_subject": claims.keycloak_subject,
        "librechat_user_id": claims.librechat_user_id,
        "issued_at": claims.issued_at,
        "expires_at": claims.expires_at,
        "record_type": claims.record_type,
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


def mint_token_for_identity(
    *,
    experiment_id: int,
    mapped_elabftw_user_id: str,
    keycloak_subject: str | None,
    librechat_user_id: str | None,
    ttl_seconds: int | None = None,
    secret: bytes | None = None,
    record_type: str = "experiment",
) -> tuple[str, str]:
    """Mint a context token bound to one experiment + one mapped user.

    Returns ``(token, expires_at_iso)``.  TTL is clamped to
    ``[1, _MAX_MINT_TTL_SECONDS]`` and defaults to
    ``_NORMAL_TTL_SECONDS`` when ``None`` or non-positive.  ``expires_at``
    is returned alongside the token so the HTTP layer does not have to
    re-derive it from the signed payload.

    This is the entrypoint the C11 launcher's ``POST /elabftw/mint_context_token``
    endpoint calls; it deliberately takes resolved identity fields
    (``mapped_elabftw_user_id``, ``keycloak_subject``, ``librechat_user_id``)
    rather than an untrusted ``IdentityMapper.map()`` result — the
    caller must first authenticate via the identity mapper (Keycloak
    session in C14; dev header in C11) and pass the resolved fields in.

    The token is single-use for *policy* (the gateway re-authenticates
    the token signature on every read/write), but TTL-bounded: an
    attacker who exfiltrates a token from the browser has at most
    ``_MAX_MINT_TTL_SECONDS`` to replay it.
    """
    if ttl_seconds is None or ttl_seconds <= 0:
        ttl_seconds = _NORMAL_TTL_SECONDS
    if ttl_seconds > _MAX_MINT_TTL_SECONDS:
        ttl_seconds = _MAX_MINT_TTL_SECONDS

    now = _dt.datetime.now(_dt.timezone.utc)
    expires = now + _dt.timedelta(seconds=ttl_seconds)
    issued_at = now.isoformat(timespec="seconds")
    expires_at = expires.isoformat(timespec="seconds")

    claims = ContextTokenClaims(
        experiment_id=experiment_id,
        mapped_elabftw_user_id=mapped_elabftw_user_id,
        keycloak_subject=keycloak_subject,
        librechat_user_id=librechat_user_id,
        issued_at=issued_at,
        expires_at=expires_at,
        record_type=record_type,
    )
    token = mint_context_token(claims, secret=secret)
    return token, expires_at


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
        record_type=str(payload.get("record_type", "experiment")),
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


def _merge_provenance_into_metadata(
    metadata: dict[str, Any] | None,
    audit_action_id: str,
    extra_values: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Merge a provenance audit_action_id into the experiment metadata dict.

    Normalizes the metadata, then appends (or creates) a provenance entry
    under the ``lab_copilot_audit_action_id`` extra field key.  Optional
    ``extra_values`` are stored as additional text extra fields (e.g.
    tool name, upload id, job id).  Returns the normalized, updated
    metadata dict.
    """
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {"_raw": metadata}

    extra_fields = metadata.get("extra_fields") or {}
    if not isinstance(extra_fields, dict):
        extra_fields = {}

    provenance_key = "lab_copilot_audit_action_id"
    if provenance_key not in extra_fields:
        extra_fields[provenance_key] = {
            "type": "text",
            "value": audit_action_id,
            "description": "gateway audit action id",
        }
    else:
        current_entry = extra_fields[provenance_key]
        current_value = (
            current_entry.get("value", "")
            if isinstance(current_entry, dict)
            else str(current_entry)
        )
        if audit_action_id not in current_value:
            new_value = (
                f"{current_value}; {audit_action_id}"
                if current_value
                else audit_action_id
            )
            current_entry["value"] = new_value  # type: ignore[index]
            current_entry.setdefault("description", "gateway audit action id")
            extra_fields[provenance_key] = current_entry

    for key, value in (extra_values or {}).items():
        extra_fields[key] = {
            "type": "text",
            "value": value,
            "description": f"lab copilot {key}",
        }

    metadata["extra_fields"] = extra_fields
    return metadata


# --- downstream client ---------------------------------------------------


class ElabftwClient(Protocol):
    """eLabFTW REST client surface the gateway adapters require.

    Read methods support C08; write methods support C09.  Implementations:
    ``StubElabftwClient`` for tests; ``HttpElabftwClient`` for live calls.
    """

    # --- read (C08 / C36) -----------------------------------------------

    def get_experiment(self, experiment_id: int) -> dict[str, Any]:
        """Return one experiment payload by id, or raise on error.

        Implementations: ``StubElabftwClient`` for tests; ``HttpElabftwClient``
        for live calls against an eLabFTW instance.
        """
        ...

    def get_item(self, item_id: int) -> dict[str, Any]:
        """Return one database resource item by id, or raise on error."""
        ...

    def get_uploads(self, record_type: str, record_id: int) -> list[dict[str, Any]]:
        """Return the uploads (attached files) for an experiment or item.

        ``record_type`` is ``"experiments"`` or ``"items"``.  Each upload
        dict includes at least ``id``, ``real_name``, and ``file_extension``.
        """
        ...

    def get_upload_content(
        self, record_type: str, record_id: int, upload_id: int
    ) -> str:
        """Download the raw content of an attached file.

        Returns the file body as a string (text files: GenBank, FASTA, etc.).
        """
        ...

    def search_items(
        self, query: str, limit: int = 20, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Search items (database resources) by free-text query.

        Same shape as ``search_experiments`` but against the items endpoint.
        Used to find plasmids, primers, antibodies, etc.
        """
        ...

    def search_experiments(
        self, query: str, limit: int = 20, offset: int = 0
    ) -> list[dict[str, Any]]:
        """Search experiments by free-text query.

        Returns a list of experiment payload dicts (same shape as
        ``get_experiment`` returns, but potentially truncated by the
        upstream API).  Per-record permissions are enforced server-side
        by the eLabFTW API key; the gateway additionally maps the caller
        identity before invoking this.
        """
        ...

    # --- write (C09) -----------------------------------------------------

    def create_experiment(self, title: str | None = None) -> int:
        """POST a new experiment; return the new experiment id.

        The body is left empty at creation; the caller patches title/body
        afterwards (matches eLabFTW REST v2: POST /experiments returns
        201 with ``location`` header pointing at the new id).
        """
        ...

    def patch_experiment_body(self, experiment_id: int, body_html: str) -> None:
        """PATCH the experiment body to ``body_html``.

        Note: this is a *replace* between eLabFTW's API semantics — callers
        that need append-only must read-then-concatenate first.  The
        :class:`ElabftwWriteAdapter` does that for ``append_amendment``.
        """
        ...

    def upload_attachment(
        self,
        experiment_id: int,
        filename: str,
        data: bytes,
        comment: str = "",
        record_type: str = "experiments",
    ) -> int:
        """POST an attachment to ``/{record_type}/{id}/uploads``.

        ``record_type`` is ``"experiments"`` or ``"items"``.
        Returns the new upload id.
        """
        ...

    def patch_experiment_metadata(
        self,
        experiment_id: int,
        metadata: dict[str, Any],
        record_type: str = "experiments",
    ) -> None:
        """PATCH the record ``metadata`` field (a JSON-encoded string
        per eLabFTW REST v2).  Used by :meth:`ElabftwWriteAdapter.write_provenance`
        to attach the gateway audit id alongside any existing extra_fields."""
        ...


@dataclass
class StubElabftwClient:
    """In-memory ``ElabftwClient`` for tests and offline dev.

    Pre-seed with ``seeds`` mapping experiment_id → payload dict.  Unknown
    experiment_ids raise ``KeyError`` on reads (raised by dict access — callers
    can catch KeyError to test the missing-record path).  Writes mutate the
    in-memory seeds dict in place so tests can assert state change.
    """

    seeds: dict[int, dict[str, Any]] = field(default_factory=dict)
    _next_id: int = 1000  # auto-increment for created experiments

    def get_experiment(self, experiment_id: int) -> dict[str, Any]:
        if experiment_id not in self.seeds:
            raise KeyError(f"experiment {experiment_id} not found in stub")
        return dict(self.seeds[experiment_id])

    def get_item(self, item_id: int) -> dict[str, Any]:
        if item_id not in self.seeds:
            raise KeyError(f"item {item_id} not found in stub")
        return dict(self.seeds[item_id])

    def get_uploads(self, record_type: str, record_id: int) -> list[dict[str, Any]]:
        record = self.seeds.get(record_id, {})
        return list(record.get("uploads", []))

    def get_upload_content(
        self, record_type: str, record_id: int, upload_id: int
    ) -> str:
        for u in self.get_uploads(record_type, record_id):
            if u.get("id") == upload_id:
                return u.get("_content", "")
        raise KeyError(f"upload {upload_id} not found")

    def search_experiments(
        self, query: str, limit: int = 20, offset: int = 0
    ) -> list[dict[str, Any]]:
        q = query.lower()
        matches = [
            dict(v)
            for v in self.seeds.values()
            if q in str(v.get("title", "")).lower()
            or q in str(v.get("body", "")).lower()
        ]
        return matches[offset : offset + limit]

    def search_items(
        self, query: str, limit: int = 20, offset: int = 0
    ) -> list[dict[str, Any]]:
        # Stub: search the same seeds (items and experiments share the
        # same shape in tests).
        return self.search_experiments(query, limit, offset)

    def create_experiment(self, title: str | None = None) -> int:
        new_id = self._next_id
        self._next_id += 1
        body = {
            "id": new_id,
            "title": title or "",
            "body": "",
            "metadata": None,
            "state": 1,  # normal
            "locked": 0,
            "uploads": [],
        }
        self.seeds[new_id] = body
        return new_id

    def patch_experiment_body(self, experiment_id: int, body_html: str) -> None:
        if experiment_id not in self.seeds:
            raise KeyError(f"experiment {experiment_id} not found in stub")
        self.seeds[experiment_id]["body"] = body_html
        # Append a changelog entry so tests can assert order.
        changelog = self.seeds[experiment_id].setdefault("changelog", [])
        changelog.append({"action": "patch_body", "len": len(body_html)})

    def upload_attachment(
        self,
        experiment_id: int,
        filename: str,
        data: bytes,
        comment: str = "",
        record_type: str = "experiments",
    ) -> int:
        if experiment_id not in self.seeds:
            raise KeyError(f"experiment {experiment_id} not found in stub")
        uploads = self.seeds[experiment_id].setdefault("uploads", [])
        upload_id = len(uploads) + 1
        uploads.append(
            {
                "id": upload_id,
                "real_name": filename,
                "comment": comment,
                "size": len(data),
            }
        )
        return upload_id

    def patch_experiment_metadata(
        self,
        experiment_id: int,
        metadata: dict[str, Any],
        record_type: str = "experiments",
    ) -> None:
        if experiment_id not in self.seeds:
            raise KeyError(f"experiment {experiment_id} not found in stub")
        self.seeds[experiment_id]["metadata"] = metadata


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

    def get_item(self, item_id: int) -> dict[str, Any]:
        """GET /api/v2/items/{id} — read a database resource item."""
        url = f"{self.base_url.rstrip('/')}/api/v2/items/{int(item_id)}"
        s = self._connect()
        resp = s.get(url, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    def get_uploads(self, record_type: str, record_id: int) -> list[dict[str, Any]]:
        """GET /api/v2/{record_type}/{id}/uploads — list attached files."""
        url = (
            f"{self.base_url.rstrip('/')}/api/v2/{record_type}/{int(record_id)}/uploads"
        )
        s = self._connect()
        resp = s.get(url, timeout=self.timeout_seconds)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return data["data"]
        return []

    def get_upload_content(
        self, record_type: str, record_id: int, upload_id: int
    ) -> str:
        """GET /api/v2/{record_type}/{id}/uploads/{uid}?format=binary — raw file content."""
        url = (
            f"{self.base_url.rstrip('/')}/api/v2/{record_type}"
            f"/{int(record_id)}/uploads/{int(upload_id)}"
        )
        s = self._connect()
        resp = s.get(url, params={"format": "binary"}, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.text

    def search_experiments(
        self, query: str, limit: int = 20, offset: int = 0
    ) -> list[dict[str, Any]]:
        """GET /api/v2/experiments?q=...&limit=...&offset=...

        eLabFTW REST v2 supports the ``q`` query parameter for free-text
        search across experiment title and body.  The response is a JSON
        array of experiment objects (same shape as ``get_experiment``).
        """
        url = f"{self.base_url.rstrip('/')}/api/v2/experiments"
        s = self._connect()
        resp = s.get(
            url,
            params={"q": query, "limit": limit, "offset": offset},
            timeout=self.timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()
        # Reason: eLabFTW may return a dict with a "data" key or a bare
        # list depending on version; normalise to list[dict].
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return data["data"]
        return []

    def search_items(
        self, query: str, limit: int = 20, offset: int = 0
    ) -> list[dict[str, Any]]:
        """GET /api/v2/items?q=...&limit=...&offset=...

        Free-text search across database resources (items).  Same ``q``
        parameter as ``search_experiments`` but hits the items endpoint.
        Used to find plasmids, primers, antibodies, etc. stored as items.
        """
        url = f"{self.base_url.rstrip('/')}/api/v2/items"
        s = self._connect()
        resp = s.get(
            url,
            params={"q": query, "limit": limit, "offset": offset},
            timeout=self.timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("data"), list):
            return data["data"]
        return []

    # --- write (C09) ------------------------------------------------------

    def create_experiment(self, title: str | None = None) -> int:
        """POST /experiments → 201 with Location header → parse new id.

        Matches the established lab pattern (see AGENTS.md → "eLabFTW API —
        HTTP client patterns"): new id comes from the Location header suffix.
        The ``title`` (if provided) goes into the POST body per eLabFTW REST
        v2 (the API accepts title on create).
        """
        url = f"{self.base_url.rstrip('/')}/api/v2/experiments"
        s = self._connect()
        post_body: dict[str, Any] = {}
        if title:
            post_body["title"] = title
        resp = s.post(url, json=post_body, timeout=self.timeout_seconds)
        resp.raise_for_status()
        loc = resp.headers.get("location") or resp.headers.get("Location")
        if not loc:
            raise RuntimeError("eLabFTW create_experiment: no Location header")
        try:
            return int(loc.rstrip("/").rsplit("/", 1)[-1])
        except ValueError as exc:
            raise RuntimeError(
                f"eLabFTW create_experiment: unparseable Location {loc!r}"
            ) from exc

    def patch_experiment_body(self, experiment_id: int, body_html: str) -> None:
        """PATCH /experiments/{id} with body field.

        Note: eLabFTW treats this as a body REPLACE — callers needing append
        semantics must read-then-concatenate first (the write adapter does).
        """
        url = f"{self.base_url.rstrip('/')}/api/v2/experiments/{int(experiment_id)}"
        s = self._connect()
        resp = s.patch(url, json={"body": body_html}, timeout=self.timeout_seconds)
        resp.raise_for_status()

    def upload_attachment(
        self,
        experiment_id: int,
        filename: str,
        data: bytes,
        comment: str = "",
        record_type: str = "experiments",
    ) -> int:
        """POST /{record_type}/{id}/uploads with multipart file form.

        ``record_type`` is ``"experiments"`` or ``"items"``.
        Returns the new upload id parsed from the Location header.
        """
        rt = "items" if record_type == "items" else "experiments"
        url = f"{self.base_url.rstrip('/')}/api/v2/{rt}/{int(experiment_id)}/uploads"
        s = self._connect()
        files = {"file": (filename, data)}
        form = {"comment": comment} if comment else {}
        resp = s.post(url, data=form, files=files, timeout=self.timeout_seconds)
        resp.raise_for_status()
        loc = resp.headers.get("location") or resp.headers.get("Location")
        if not loc:
            return 0  # eLabFTW <4.x may not return a Location; treat as success
        try:
            return int(loc.rstrip("/").rsplit("/", 1)[-1])
        except ValueError:
            return 0

    def patch_experiment_metadata(
        self,
        experiment_id: int,
        metadata: dict[str, Any],
        record_type: str = "experiments",
    ) -> None:
        """PATCH /{record_type}/{id} with metadata field.

        ``record_type`` is ``"experiments"`` or ``"items"``.
        Per AGENTS.md: metadata must be ``json.dumps()``'d in the body, never
        a raw dict — passing a dict causes HTTP 500.
        """
        rt = "items" if record_type == "items" else "experiments"
        url = f"{self.base_url.rstrip('/')}/api/v2/{rt}/{int(experiment_id)}"
        s = self._connect()
        resp = s.patch(
            url,
            json={"metadata": json.dumps(metadata)},
            timeout=self.timeout_seconds,
        )
        resp.raise_for_status()


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
    uploads: list[dict[str, Any]]
    raw: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "title": self.title,
            "body": self.body,
            "metadata": self.metadata,
            "uploads": self.uploads,
        }


@dataclass(frozen=True)
class ExperimentSummary:
    """Compact summary of one experiment in a search result (C36).

    Deliberately excludes ``body`` — the full content is available via
    ``read_experiment_by_id`` if the LLM needs it.  Returning only
    summaries keeps the context window manageable for multi-record
    searches.
    """

    experiment_id: int
    title: str
    created_at: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class SearchResult:
    """Outcome of a successful ``search_my_experiments`` call (C36)."""

    query: str
    total_returned: int
    experiments: list[ExperimentSummary]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "total_returned": self.total_returned,
            "experiments": [e.to_dict() for e in self.experiments],
        }


@dataclass(frozen=True)
class ItemSummary:
    """Compact summary of one item (resource) in a search result (C36b).

    Mirrors ``ExperimentSummary`` but uses ``item_id`` so the LLM knows
    to call ``read_item_by_id`` (not ``read_experiment_by_id``) with it.
    """

    item_id: int
    title: str
    created_at: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class ItemSearchResult:
    """Outcome of a successful ``search_items`` call (C36b)."""

    query: str
    total_returned: int
    items: list[ItemSummary]

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "total_returned": self.total_returned,
            "items": [i.to_dict() for i in self.items],
        }


@dataclass(frozen=True)
class ItemReadResult:
    """Outcome of a successful ``read_item_by_id`` call (C36b).

    Mirrors ``ReadResult`` but uses ``item_id`` so the LLM can chain
    the id into ``download_upload`` with ``record_type=\"items\"``.
    """

    item_id: int
    title: str
    body: str
    metadata: dict[str, Any] | None
    uploads: list[dict[str, Any]]
    raw: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "title": self.title,
            "body": self.body,
            "metadata": self.metadata,
            "uploads": self.uploads,
        }


# --- adapter -------------------------------------------------------------


TOOL_NAME = "elabftw.read_current_experiment"
TOOL_TIER = Tier.PERMISSIONED_ELABFTW_READ

# C36 — multi-record retrieval tool names (same tier, no approval).
TOOL_NAME_SEARCH = "elabftw.search_my_experiments"
TOOL_NAME_READ_BY_ID = "elabftw.read_experiment_by_id"
TOOL_NAME_SEARCH_ITEMS = "elabftw.search_items"
TOOL_NAME_READ_ITEM = "elabftw.read_item_by_id"

# Maximum results returned by search_my_experiments.  Keeps context window
# manageable and bounds the audit record size.
_MAX_SEARCH_RESULTS = 50


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
        self._verify_identity_binding(
            claims=claims,
            mapped_identity=mapped_identity,
            tool_name=TOOL_NAME,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            tool_args_hash=tool_args_hash,
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

        # 6. Make the call.  Dispatch to the right endpoint based on
        # record_type: "resource" → /api/v2/items/{id}, default → /api/v2/experiments/{id}.
        try:
            if claims.record_type == "resource":
                payload = self.client.get_item(claims.experiment_id)
            else:
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

        # 7. Fetch uploads (attached files) so the LLM can see what
        # plasmid/sequence files are attached to the experiment.  This is
        # critical for cloning workflows where the user says "use this
        # plasmid" — the plasmid is an attached .gb/.fasta file.
        uploads: list[dict[str, Any]] = []
        try:
            record_type = "items" if claims.record_type == "resource" else "experiments"
            uploads_payload = self.client.get_uploads(record_type, claims.experiment_id)
            if isinstance(uploads_payload, list):
                for u in uploads_payload:
                    uploads.append(
                        {
                            "id": u.get("id"),
                            "real_name": u.get("real_name", ""),
                            "type": u.get("type", ""),
                            "file_extension": u.get("file_extension", ""),
                            "filesize": u.get("filesize"),
                        }
                    )
        except Exception:
            pass  # uploads are best-effort; don't fail the read

        # 8. Build the read result + successful audit record.  Normalize
        # metadata the way the repo's other eLabFTW tools do — the API may
        # return metadata as a JSON-encoded string under the "metadata" key,
        # and that string may itself be a JSON-encoded JSON string.
        result = ReadResult(
            experiment_id=claims.experiment_id,
            title=str(payload.get("title", "")),
            body=str(payload.get("body", "")),
            metadata=_normalize_metadata(payload.get("metadata")),
            uploads=uploads,
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

    # --- C36: multi-record retrieval -------------------------------------

    def search_my_experiments(
        self,
        *,
        query: str,
        limit: int = 20,
        offset: int = 0,
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None = None,
        request_id: str | None = None,
        keycloak_subject: str | None = None,
        librechat_user_id: str | None = None,
        provider: str | None = None,
        model_id: str | None = None,
    ) -> SearchResult:
        """Search the caller's accessible experiments by free-text query.

        C36 orchestration (simpler than ``read_current_experiment`` — no
        context-token binding because the search is not scoped to a single
        experiment):

            1. Identity resolution (mapped user required).
            2. Policy decision (tier 2 read, no approval).
            3. Downstream client search.
            4. Audit: query envelope + returned record IDs.

        Per-record permissions are enforced server-side by the eLabFTW API
        key; the gateway additionally maps the caller identity before
        invoking the search.
        """
        # Clamp limit to the maximum to prevent context-blowing queries.
        clamped_limit = min(max(1, limit), _MAX_SEARCH_RESULTS)
        clamped_offset = max(0, offset)

        tool_args_hash = compute_args_hash(
            {"query": query, "limit": clamped_limit, "offset": clamped_offset}
        )

        # 1. Identity resolution — unmapped caller → deny before any HTTP.
        if mapped_identity is None:
            self._audit(
                policy_decision="deny",
                reason="unmapped_caller",
                mapped_identity=None,
                experiment_id=None,
                tool_name=TOOL_NAME_SEARCH,
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

        # 2. Policy decision — tier 2 read with mapped caller.
        policy_req = PolicyRequest(
            tool_name=TOOL_NAME_SEARCH,
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
                experiment_id=None,
                tool_name=TOOL_NAME_SEARCH,
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

        # 3. Downstream client.
        if self.client is None:
            self._audit(
                policy_decision="deny",
                reason="elabftw_not_configured",
                mapped_identity=mapped_identity,
                experiment_id=None,
                tool_name=TOOL_NAME_SEARCH,
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

        # 4. Make the call.
        try:
            payloads = self.client.search_experiments(
                query, limit=clamped_limit, offset=clamped_offset
            )
        except Exception as exc:
            self._audit(
                policy_decision="allow",
                reason="client_error",
                mapped_identity=mapped_identity,
                experiment_id=None,
                tool_name=TOOL_NAME_SEARCH,
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

        # 5. Build summaries + successful audit record.  Only id/title/
        # timestamps are returned — bodies are available via
        # read_experiment_by_id if the LLM needs them.
        summaries = [
            ExperimentSummary(
                experiment_id=int(p.get("id", 0)),
                title=str(p.get("title", "")),
                created_at=p.get("created_at") or p.get("createdat"),
                updated_at=p.get("updated_at") or p.get("updatedat"),
            )
            for p in payloads
        ]
        result = SearchResult(
            query=query,
            total_returned=len(summaries),
            experiments=summaries,
        )

        record_ids = [s.experiment_id for s in summaries]
        self._audit(
            policy_decision="allow",
            reason="search_succeeded",
            mapped_identity=mapped_identity,
            experiment_id=None,
            tool_name=TOOL_NAME_SEARCH,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            tool_args_hash=tool_args_hash,
            api_call_summary={
                "method": "GET",
                "path": "/api/v2/experiments",
                "query": query,
                "limit": clamped_limit,
                "offset": clamped_offset,
            },
            result_summary={
                "record_count": result.total_returned,
                "record_ids": record_ids,
            },
            extra_context_refs=[
                {"kind": "experiment", "id": rid} for rid in record_ids
            ],
            require_persistence=True,
        )
        return result

    def read_experiment_by_id(
        self,
        *,
        experiment_id: int,
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None = None,
        request_id: str | None = None,
        keycloak_subject: str | None = None,
        librechat_user_id: str | None = None,
        provider: str | None = None,
        model_id: str | None = None,
    ) -> ReadResult:
        """Read a specific experiment by id (C36).

        Unlike ``read_current_experiment``, the experiment id comes from
        the tool args (the LLM specifies which record to read), not from
        a context token.  No context-token binding is required — the
        identity mapper + policy engine + eLabFTW API key permissions
        are sufficient.

        Orchestration:
            1. Identity resolution (mapped user required).
            2. Policy decision (tier 2 read, no approval).
            3. Downstream client read.
            4. Audit with experiment_id in context_refs.
        """
        tool_args_hash = compute_args_hash({"experiment_id": experiment_id})

        # 1. Identity resolution.
        if mapped_identity is None:
            self._audit(
                policy_decision="deny",
                reason="unmapped_caller",
                mapped_identity=None,
                experiment_id=experiment_id,
                tool_name=TOOL_NAME_READ_BY_ID,
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

        # 2. Policy decision.
        policy_req = PolicyRequest(
            tool_name=TOOL_NAME_READ_BY_ID,
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
                experiment_id=experiment_id,
                tool_name=TOOL_NAME_READ_BY_ID,
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

        # 3. Downstream client.
        if self.client is None:
            self._audit(
                policy_decision="deny",
                reason="elabftw_not_configured",
                mapped_identity=mapped_identity,
                experiment_id=experiment_id,
                tool_name=TOOL_NAME_READ_BY_ID,
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

        # 4. Make the call.
        try:
            payload = self.client.get_experiment(experiment_id)
        except Exception as exc:
            self._audit(
                policy_decision="allow",
                reason="client_error",
                mapped_identity=mapped_identity,
                experiment_id=experiment_id,
                tool_name=TOOL_NAME_READ_BY_ID,
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

        # 5. Fetch uploads (same pattern as read_current_experiment).
        uploads: list[dict[str, Any]] = []
        try:
            uploads_payload = self.client.get_uploads("experiments", experiment_id)
            if isinstance(uploads_payload, list):
                for u in uploads_payload:
                    uploads.append(
                        {
                            "id": u.get("id"),
                            "real_name": u.get("real_name", ""),
                            "type": u.get("type", ""),
                            "file_extension": u.get("file_extension", ""),
                            "filesize": u.get("filesize"),
                        }
                    )
        except Exception:
            pass

        # 6. Build the read result + successful audit record.
        result = ReadResult(
            experiment_id=experiment_id,
            title=str(payload.get("title", "")),
            body=str(payload.get("body", "")),
            metadata=_normalize_metadata(payload.get("metadata")),
            uploads=uploads,
            raw=payload,
        )

        self._audit(
            policy_decision="allow",
            reason="read_succeeded",
            mapped_identity=mapped_identity,
            experiment_id=experiment_id,
            tool_name=TOOL_NAME_READ_BY_ID,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            tool_args_hash=tool_args_hash,
            api_call_summary={
                "method": "GET",
                "path": f"/api/v2/experiments/{experiment_id}",
            },
            result_summary={
                "title_length": len(result.title),
                "body_length": len(result.body),
            },
            require_persistence=True,
        )
        return result

    # --- C36b: item (resource) search + read -----------------------------

    def search_items(
        self,
        *,
        query: str,
        limit: int = 20,
        offset: int = 0,
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None = None,
        request_id: str | None = None,
        keycloak_subject: str | None = None,
        librechat_user_id: str | None = None,
        provider: str | None = None,
        model_id: str | None = None,
    ) -> ItemSearchResult:
        """Search the caller's accessible items (resources) by free-text query.

        Same orchestration as ``search_my_experiments`` but hits the items
        endpoint.  Used to find plasmids, primers, antibodies, etc. stored
        as database resources.
        """
        clamped_limit = min(max(1, limit), _MAX_SEARCH_RESULTS)
        clamped_offset = max(0, offset)

        tool_args_hash = compute_args_hash(
            {"query": query, "limit": clamped_limit, "offset": clamped_offset}
        )

        if mapped_identity is None:
            self._audit(
                policy_decision="deny",
                reason="unmapped_caller",
                mapped_identity=None,
                experiment_id=None,
                tool_name=TOOL_NAME_SEARCH_ITEMS,
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

        policy_req = PolicyRequest(
            tool_name=TOOL_NAME_SEARCH_ITEMS,
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
                experiment_id=None,
                tool_name=TOOL_NAME_SEARCH_ITEMS,
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

        if self.client is None:
            self._audit(
                policy_decision="deny",
                reason="elabftw_not_configured",
                mapped_identity=mapped_identity,
                experiment_id=None,
                tool_name=TOOL_NAME_SEARCH_ITEMS,
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

        try:
            payloads = self.client.search_items(
                query, limit=clamped_limit, offset=clamped_offset
            )
        except Exception as exc:
            self._audit(
                policy_decision="allow",
                reason="client_error",
                mapped_identity=mapped_identity,
                experiment_id=None,
                tool_name=TOOL_NAME_SEARCH_ITEMS,
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

        summaries = [
            ItemSummary(
                item_id=int(p.get("id", 0)),
                title=str(p.get("title", "")),
                created_at=p.get("created_at") or p.get("createdat"),
                updated_at=p.get("updated_at") or p.get("updatedat"),
            )
            for p in payloads
        ]
        result = ItemSearchResult(
            query=query,
            total_returned=len(summaries),
            items=summaries,
        )

        record_ids = [s.item_id for s in summaries]
        self._audit(
            policy_decision="allow",
            reason="search_succeeded",
            mapped_identity=mapped_identity,
            experiment_id=None,
            tool_name=TOOL_NAME_SEARCH_ITEMS,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            tool_args_hash=tool_args_hash,
            api_call_summary={
                "method": "GET",
                "path": "/api/v2/items",
                "query": query,
                "limit": clamped_limit,
                "offset": clamped_offset,
            },
            result_summary={
                "record_count": result.total_returned,
                "record_ids": record_ids,
            },
            extra_context_refs=[{"kind": "item", "id": rid} for rid in record_ids],
            require_persistence=True,
        )
        return result

    def read_item_by_id(
        self,
        *,
        item_id: int,
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None = None,
        request_id: str | None = None,
        keycloak_subject: str | None = None,
        librechat_user_id: str | None = None,
        provider: str | None = None,
        model_id: str | None = None,
    ) -> ItemReadResult:
        """Read a specific item (resource) by id.

        Same orchestration as ``read_experiment_by_id`` but hits the items
        endpoint.  Used to read plasmids, primers, etc. stored as database
        resources.
        """
        tool_args_hash = compute_args_hash({"item_id": item_id})

        if mapped_identity is None:
            self._audit(
                policy_decision="deny",
                reason="unmapped_caller",
                mapped_identity=None,
                experiment_id=item_id,
                tool_name=TOOL_NAME_READ_ITEM,
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

        policy_req = PolicyRequest(
            tool_name=TOOL_NAME_READ_ITEM,
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
                experiment_id=item_id,
                tool_name=TOOL_NAME_READ_ITEM,
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

        if self.client is None:
            self._audit(
                policy_decision="deny",
                reason="elabftw_not_configured",
                mapped_identity=mapped_identity,
                experiment_id=item_id,
                tool_name=TOOL_NAME_READ_ITEM,
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

        try:
            payload = self.client.get_item(item_id)
        except Exception as exc:
            self._audit(
                policy_decision="allow",
                reason="client_error",
                mapped_identity=mapped_identity,
                experiment_id=item_id,
                tool_name=TOOL_NAME_READ_ITEM,
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

        uploads: list[dict[str, Any]] = []
        try:
            uploads_payload = self.client.get_uploads("items", item_id)
            if isinstance(uploads_payload, list):
                for u in uploads_payload:
                    uploads.append(
                        {
                            "id": u.get("id"),
                            "real_name": u.get("real_name", ""),
                            "type": u.get("type", ""),
                            "file_extension": u.get("file_extension", ""),
                            "filesize": u.get("filesize"),
                        }
                    )
        except Exception:
            pass

        result = ItemReadResult(
            item_id=item_id,
            title=str(payload.get("title", "")),
            body=str(payload.get("body", "")),
            metadata=_normalize_metadata(payload.get("metadata")),
            uploads=uploads,
            raw=payload,
        )

        self._audit(
            policy_decision="allow",
            reason="read_succeeded",
            mapped_identity=mapped_identity,
            experiment_id=item_id,
            tool_name=TOOL_NAME_READ_ITEM,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            tool_args_hash=tool_args_hash,
            api_call_summary={
                "method": "GET",
                "path": f"/api/v2/items/{item_id}",
            },
            result_summary={
                "title_length": len(result.title),
                "body_length": len(result.body),
            },
            require_persistence=True,
        )
        return result

    def _verify_identity_binding(
        self,
        *,
        claims: Any,
        mapped_identity: MappedIdentity,
        tool_name: str,
        conversation_id: str | None,
        request_id: str | None,
        keycloak_subject: str | None,
        librechat_user_id: str | None,
        provider: str | None,
        model_id: str | None,
        tool_args_hash: str,
    ) -> None:
        """Verify token-identity binding (three-way check).

        Raises InvalidContextToken on mismatch (after audit).
        """
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
        tool_name: str = TOOL_NAME,
        api_call_summary: dict[str, Any] | None = None,
        result_summary: dict[str, Any] | None = None,
        extra_context_refs: list[dict[str, Any]] | None = None,
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

        ``tool_name`` defaults to ``TOOL_NAME`` (read_current_experiment)
        for backward compatibility; C36 methods pass their own tool name.

        ``extra_context_refs`` appends additional context references beyond
        the single ``experiment_id`` — used by C36 search to record every
        returned record id in the audit trail.
        """
        import secrets as _secrets
        import time as _time

        action_id = f"{self.action_id_prefix}-{int(_time.time() * 1000)}-{_secrets.token_hex(4)}"
        context_refs: list[dict[str, Any]] = []
        if experiment_id is not None:
            context_refs.append({"kind": "experiment", "id": experiment_id})
        if extra_context_refs:
            context_refs.extend(extra_context_refs)

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
            tool_name=tool_name,
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


# ============================================================================
# C09 — eLabFTW bounded write adapter
# ============================================================================
#
# The write adapter extends the C08 orchestration template:
#
#     context token verification (signed, unexpired, three-way identity binding)
#         → identity resolution (mapped user required)
#         → policy engine decision (tier 4 BOUNDED_WRITES)
#         → approval token consume (single-use, hash-bound, target-record-bound)
#         → append-only enforcement (locked / archived records rejected)
#         → downstream client write (POST, PATCH-append, uploads)
#         → write provenance back to eLabFTW (audit_id in metadata.extra_fields)
#         → audit record with context_refs, tool_args_hash, approval_id
#
# Invariants specific to writes (in addition to the read invariants):
#
#     * An approval token MUST be presented and consumed atomically before any
#       downstream write call.  Without approval, ``PolicyDenied`` is raised
#       (the policy engine returns ``approval_required`` for tier >= 4 without
#       approval).  Even with policy allow, the adapter calls
#       ``ApprovalStore.consume`` to enforce single-use, args-hash-bound,
#       target-record-bound approval.  This is server-side enforcement;
#       LibreChat's approval card is untrusted UI.
#
#     * Append-only enforcement: writes are refused if the target experiment
#       is locked or in archived / deleted state.  eLabFTW's REST API will
#       happily PATCH a locked experiment (GitHub issue #3433) — the gateway
#       must enforce this itself.  The adapter reads the experiment state
#       (``state`` integer: 1=normal, 2=archived, 3=deleted) and ``locked``
#       boolean, and refuses if state != 1 or locked is truthy.
#
#     * The four V1 write operations map onto two C06 tool registry entries:
#         - ``elabftw.draft_experiment_update``           → create_experiment
#                                                            + patch body
#         - ``elabftw.amend_my_experiment_after_approval`` → append_amendment
#                                                            + upload_attachment
#                                                            (optional)
#                                                            + write_provenance
#       The adapter exposes all four primitives but the two tool names above
#       are how they are invoked.


# --- write exceptions ----------------------------------------------------


class AppendOnlyViolation(ElabftwAdapterError):
    """Target experiment is locked or in a non-editable state."""

    def __init__(self, experiment_id: int, state: int, locked: bool) -> None:
        detail = (
            f"experiment {experiment_id} is not appendable "
            f"(state={state}, locked={locked})"
        )
        super().__init__(reason="append_only_violation", message=detail)
        self.experiment_id = experiment_id
        self.state = state
        self.locked = locked


class ApprovalRequired(ElabftwAdapterError):
    """Tier 4 write requires an approval token; none provided or consumed."""

    def __init__(self, detail: str) -> None:
        super().__init__(
            reason="approval_required",
            message=f"write requires approval: {detail}",
        )
        self.detail = detail


# --- write result ---------------------------------------------------------


@dataclass(frozen=True)
class WriteResult:
    """Outcome of a successful ``ElabftwWriteAdapter`` operation.

    ``operation`` is one of ``create_draft``, ``append_amendment``,
    ``upload_attachment``, ``write_provenance``.  The downstream id is the
    new experiment / upload / metadata row id when applicable.
    """

    operation: str
    experiment_id: int
    downstream_id: int | None
    audit_action_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "experiment_id": self.experiment_id,
            "downstream_id": self.downstream_id,
            "audit_action_id": self.audit_action_id,
        }


# --- write adapter --------------------------------------------------------


# Tool names from the C06 registry.  ``draft_experiment_update`` is the
# "create + populate" path; ``amend_my_experiment_after_approval`` is the
# "append amendment + attachments + provenance" path.  Both map onto tier 4
# bounded writes and require approval tokens.
TOOL_NAME_DRAFT = "elabftw.draft_experiment_update"
TOOL_NAME_AMEND = "elabftw.amend_my_experiment_after_approval"
TOOL_NAME_EDIT_SECTION = "elabftw.edit_experiment_section"
TOOL_TIER_WRITE = Tier.BOUNDED_WRITES


# eLabFTW experiment state integers (matching the upstream schema).
_STATE_NORMAL = 1
_STATE_ARCHIVED = 2
_STATE_DELETED = 3


@dataclass
class ElabftwWriteAdapter:
    """Orchestrates eLabFTW bounded writes through the gateway-enforced path.

    Dependencies are injected so tests can swap the policy engine, identity
    mapper, audit store, approval store, and downstream client individually.
    In production, the module-level ``get_elabftw_write_adapter()`` wires all
    the singletons (mirrors the read adapter pattern from C08).
    """

    policy_engine: PolicyEngine
    audit_store: AuditStore
    approval_store: Any  # ApprovalStore — typed as Any to avoid import cycle
    client: ElabftwClient | None
    action_id_prefix: str = "elab-write"

    # --- public API: two C09 tool entrypoints ----------------------------

    def draft_experiment_update(
        self,
        *,
        context_token: str,
        approval_id: str,
        approval_args: dict[str, Any],
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None = None,
        request_id: str | None = None,
        keycloak_subject: str | None = None,
        librechat_user_id: str | None = None,
        provider: str | None = None,
        model_id: str | None = None,
    ) -> WriteResult:
        """Create a new draft experiment and patch its title.

        This is the ``elabftw.draft_experiment_update`` tool from C06.
        Returns the new experiment id with an audit row.

        Append-only enforcement does NOT apply here — the operation creates a
        new experiment, no existing record is mutated.  The approval token
        bind still applies: ``approval_args`` (the exact body the LLM drafted)
        must hash-match the approval at consume time.

        Security note: the only operation parameter (`title`) is sourced
        directly from ``approval_args['title']`` — NOT passed as a separate
        HTTP field — so the approval-token hash binds it.  An attacker cannot
        swap the title without breaking the args hash (and thus the consume).
        See C09 review blocker 2 — originally a separate ``target_title``
        HTTP field was accepted, which made the binding incomplete.
        """
        # Pull the title from the approval_args — it is the hash-bound source
        # of truth.  If approval_args did not include a title, the caller
        # approved an empty title and we honour that.  This removes the attack
        # surface where a separate HTTP body field could differ from the
        # approved args dict.
        target_title = (
            approval_args.get("title") if isinstance(approval_args, dict) else None
        )
        return self._execute(
            operation="create_draft",
            tool_name=TOOL_NAME_DRAFT,
            context_token=context_token,
            approval_id=approval_id,
            approval_args=approval_args,
            mapped_identity=mapped_identity,
            target_experiment_id=None,  # new experiment, no append-only check
            apply_append_only=False,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            exec_fn=lambda _client, _claims: self._exec_create_draft(
                _client, target_title=target_title
            ),
        )

    def amend_my_experiment_after_approval(
        self,
        *,
        context_token: str,
        approval_id: str,
        approval_args: dict[str, Any],
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None = None,
        request_id: str | None = None,
        keycloak_subject: str | None = None,
        librechat_user_id: str | None = None,
        provider: str | None = None,
        model_id: str | None = None,
        amendment_html: str = "",
        attachment_filename: str | None = None,
        attachment_data: bytes | None = None,
        attachment_comment: str = "",
    ) -> WriteResult:
        """Append an approved amendment section to the user's experiment.

        This is the ``elabftw.amend_my_experiment_after_approval`` tool from
        C06.  Reads the current body, concatenates the amendment HTML, PATCHes
        the experiment body back, optionally uploads an attachment, then
        writes provenance (the gateway audit_action_id) into the experiment's
        metadata.

        Append-only enforcement DOES apply: the target experiment must be in
        state=normal (1) and not locked.  Locked / archived / deleted records
        are refused with ``AppendOnlyViolation`` after the audit row is
        written.
        """
        # Reason: capture inputs at call time so the caller can correlate the
        # audit row back to the exact amendment text.  The approval_args hash
        # is checked at consume() against whatever args the LLM proposed.
        amendment_html = amendment_html or ""
        # Pre-resolve the target experiment id from the context token; the
        # append-only check needs it to read state from downstream.
        verified = self._verify_token_and_identity(
            context_token=context_token,
            mapped_identity=mapped_identity,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            tool_name=TOOL_NAME_AMEND,
            early_args_for_hash=approval_args,
        )
        # Failure path returns a 1-tuple carrying the exception to re-raise
        # (the audit row was already written inside the helper).
        if len(verified) == 1:
            raise verified[0]  # type: ignore[misc]
        claims, tool_args_hash = verified  # type: ignore[assignment]
        # Reason: when the LLM passes target_experiment_id in approval_args,
        # write to THAT experiment instead of the context token's experiment.
        # This enables cross-experiment workflows (copy results from #78 to
        # #42, create new experiment then amend it, etc.).  The approval
        # hash already binds target_experiment_id because it's part of
        # approval_args — the user approves writing to that specific target.
        # Fall back to claims.experiment_id for backward compatibility.
        target_experiment_id = (
            approval_args.get("target_experiment_id") or claims.experiment_id
        )
        if not isinstance(target_experiment_id, int):
            target_experiment_id = int(target_experiment_id)
        return self._execute(
            operation="append_amendment",
            tool_name=TOOL_NAME_AMEND,
            context_token=context_token,
            approval_id=approval_id,
            approval_args=approval_args,
            mapped_identity=mapped_identity,
            target_experiment_id=target_experiment_id,
            apply_append_only=True,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            precomputed_claims=claims,
            precomputed_tool_args_hash=tool_args_hash,
            exec_fn=lambda client, _claims: self._exec_amend(
                client,
                target_experiment_id=target_experiment_id,
                amendment_html=amendment_html,
                attachment_filename=attachment_filename,
                attachment_data=attachment_data,
                attachment_comment=attachment_comment,
            ),
        )

    def edit_experiment_section(
        self,
        *,
        context_token: str,
        approval_id: str,
        approval_args: dict[str, Any],
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None = None,
        request_id: str | None = None,
        keycloak_subject: str | None = None,
        librechat_user_id: str | None = None,
        provider: str | None = None,
        model_id: str | None = None,
        plan_approval_id: str | None = None,
    ) -> WriteResult:
        """Replace the body of the user's experiment with approved content (C35).

        This is the ``elabftw.edit_experiment_section`` tool. Unlike the
        append-only amend (C09), this is a DIRECT edit: the approved
        ``new_body`` REPLACES the experiment body wholesale. Rollback is
        provided by eLabFTW's own revision history (each PATCH creates a
        revision snapshot upstream); the gateway records audit/provenance.

        Safety model:
          - The approval args MUST include ``old_body_hash`` (SHA-256 hex of
            the body as it was when the user approved the edit) and
            ``new_body`` (the replacement content). The approval token's
            args_hash binds both (tamper-detection).
          - At execution, the adapter re-reads the current body and recomputes
            its hash. If it no longer matches ``old_body_hash`` (the record
            changed between approval and execution), the edit is refused with
            ``ElabftwAdapterError(reason="body_drift_since_approval")`` — a
            stale edit must never silently clobber a newer state.
          - Locked / archived / deleted records are refused (same state/locked
            guard as the amend path) — you cannot direct-edit a non-editable
            record.
        """
        verified = self._verify_token_and_identity(
            context_token=context_token,
            mapped_identity=mapped_identity,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            tool_name=TOOL_NAME_EDIT_SECTION,
            early_args_for_hash=approval_args,
        )
        if len(verified) == 1:
            raise verified[0]  # type: ignore[misc]
        claims, tool_args_hash = verified  # type: ignore[assignment]
        target_experiment_id = claims.experiment_id
        return self._execute(
            operation="edit_section",
            tool_name=TOOL_NAME_EDIT_SECTION,
            context_token=context_token,
            approval_id=approval_id,
            approval_args=approval_args,
            mapped_identity=mapped_identity,
            target_experiment_id=target_experiment_id,
            apply_append_only=True,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            precomputed_claims=claims,
            precomputed_tool_args_hash=tool_args_hash,
            plan_approval_id=plan_approval_id,
            exec_fn=lambda client, _claims: self._exec_edit_section(
                client,
                target_experiment_id=target_experiment_id,
                approval_args=approval_args,
            ),
        )

    # --- core executor (shared by both tool entrypoints) ------------------

    def _execute(
        self,
        *,
        operation: str,
        tool_name: str,
        context_token: str,
        approval_id: str,
        approval_args: dict[str, Any],
        mapped_identity: MappedIdentity | None,
        target_experiment_id: int | None,
        apply_append_only: bool,
        conversation_id: str | None,
        request_id: str | None,
        keycloak_subject: str | None,
        librechat_user_id: str | None,
        provider: str | None,
        model_id: str | None,
        exec_fn: Any,  # Callable[[ElabftwClient, ContextTokenClaims], int]
        precomputed_claims: ContextTokenClaims | None = None,
        precomputed_tool_args_hash: str | None = None,
        plan_approval_id: str | None = None,
    ) -> WriteResult:
        """Run the C09 orchestration template.

        Steps ( denying at any step still writes the audit row first ):

            1. Verify context token + bind to mapped identity (unless the
               caller pre-computed these from a prior verify — saves a second
               verify in the amend path).
            2. Policy decision (tier 4 bounded write).
            3. Append-only enforcement (only when ``apply_append_only``).
            4. Approval token consume (single-use, args-hash-bound).
               SKIPPED when ``plan_approval_id`` is set — the plan executor
               has already consumed the plan-level approval (bound to the
               plan hash, which covers all step args).
            5. Downstream write.
            6. Provenance writeback (audit action_id in metadata).
            7. Success audit row (require_persistence=True).
        """
        # 1. Verify + identity bind.
        if precomputed_claims is not None and precomputed_tool_args_hash is not None:
            claims = precomputed_claims
            tool_args_hash = precomputed_tool_args_hash
        else:
            verified = self._verify_token_and_identity(
                context_token=context_token,
                mapped_identity=mapped_identity,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_name=tool_name,
                early_args_for_hash=approval_args,
            )
            if len(verified) == 1:
                raise verified[0]  # type: ignore[misc]
            claims, tool_args_hash = verified  # type: ignore[assignment]

        # 2. Policy decision — tier 4 write requires approval.
        self._check_write_policy(
            tool_name=tool_name,
            approval_id=approval_id,
            mapped_identity=mapped_identity,
            target_experiment_id=target_experiment_id,
            claims=claims,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            tool_args_hash=tool_args_hash,
        )

        # 3. Append-only enforcement (when applicable).
        if (
            apply_append_only
            and self.client is not None
            and target_experiment_id is not None
        ):
            self._enforce_append_only(
                tool_name=tool_name,
                approval_id=approval_id,
                mapped_identity=mapped_identity,
                target_experiment_id=target_experiment_id,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=tool_args_hash,
            )

        # 4. Approval token consume.
        effective_approval_id = self._consume_write_approval(
            plan_approval_id=plan_approval_id,
            approval_id=approval_id,
            approval_args=approval_args,
            claims=claims,
            mapped_identity=mapped_identity,
            target_experiment_id=target_experiment_id,
            tool_name=tool_name,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            tool_args_hash=tool_args_hash,
        )

        # 5. Client must be present.
        self._require_write_client(
            tool_name=tool_name,
            effective_approval_id=effective_approval_id,
            mapped_identity=mapped_identity,
            target_experiment_id=target_experiment_id,
            claims=claims,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            tool_args_hash=tool_args_hash,
        )

        # 6. Downstream write.  Wrap any failure in an ElabftwAdapterError
        # after the audit row persists.  The approval was already consumed —
        # a downstream failure does NOT "refund" the approval.  Callers retry
        # with a fresh approval; the audit row explains what happened.
        #
        # C35: an exec_fn may raise a SEMANTIC ElabftwAdapterError (e.g.
        # body_drift_since_approval) that already carries the correct reason
        # code + message. Re-raise those directly (after the audit row) so
        # the reason isn't clobbered to "client_error"; only wrap unexpected
        # non-adapter exceptions as client_error.
        try:
            downstream_id = exec_fn(self.client, claims)
        except ElabftwAdapterError as exc:
            self._audit(
                policy_decision="allow",
                reason=exc.reason,
                mapped_identity=mapped_identity,
                experiment_id=target_experiment_id or claims.experiment_id,
                tool_name=tool_name,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=tool_args_hash,
                approval_id=effective_approval_id,
                error={
                    "code": exc.reason.upper(),
                    "exception": type(exc).__name__,
                    "message": str(exc),
                },
            )
            raise
        except Exception as exc:
            self._audit(
                policy_decision="allow",
                reason="client_error",
                mapped_identity=mapped_identity,
                experiment_id=target_experiment_id or claims.experiment_id,
                tool_name=tool_name,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=tool_args_hash,
                approval_id=effective_approval_id,
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

        # On the create_draft path downstream_id IS the new experiment id;
        # rewrite target_experiment_id for the provenance writeback below.
        effective_experiment_id = target_experiment_id or downstream_id

        # 7. Provenance writeback — embed the audit action_id into the
        # experiment metadata so eLabFTW-facing viewers see the gateway's
        # trail.  Best-effort; failures are logged but not fatal.
        assert self.client is not None  # guarded by _require_write_client above
        audit_action_id = self._write_provenance(
            client=self.client,
            experiment_id=effective_experiment_id,
            existing_audit_action_id=None,  # set below via _audit
            tool_args_hash=tool_args_hash,
            approval_id=effective_approval_id,
            mapped_identity=mapped_identity,
            tool_name=tool_name,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
        )

        # 8. Success audit row.  require_persistence=True — a write that
        # succeeded without an audit trail is a safety violation.
        self._audit(
            policy_decision="allow",
            reason=f"{operation}_succeeded",
            mapped_identity=mapped_identity,
            experiment_id=effective_experiment_id,
            tool_name=tool_name,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            tool_args_hash=tool_args_hash,
            approval_id=effective_approval_id,
            api_call_summary={
                "operation": operation,
                "downstream_id": downstream_id,
            },
            result_summary={
                "effective_experiment_id": effective_experiment_id,
                "audit_action_id_provenance": audit_action_id,
            },
            require_persistence=True,
        )

        return WriteResult(
            operation=operation,
            experiment_id=effective_experiment_id,
            downstream_id=downstream_id
            if downstream_id != effective_experiment_id
            else None,
            audit_action_id=audit_action_id,
        )

    def _check_write_policy(
        self,
        *,
        tool_name: str,
        approval_id: str,
        mapped_identity: MappedIdentity | None,
        target_experiment_id: int | None,
        claims: ContextTokenClaims,
        conversation_id: str | None,
        request_id: str | None,
        keycloak_subject: str | None,
        librechat_user_id: str | None,
        provider: str | None,
        model_id: str | None,
        tool_args_hash: str,
    ) -> None:
        """Check tier 4 write policy and raise on denial."""
        policy_req = PolicyRequest(
            tool_name=tool_name,
            tier=TOOL_TIER_WRITE,
            adapter="elabftw",
            user_id=mapped_identity.elabftw_user_id if mapped_identity else None,
            team_id=mapped_identity.elabftw_team_id if mapped_identity else None,
            has_approval=bool(approval_id),
            approval_id=approval_id,
        )
        decision = self.policy_engine.decide(policy_req)
        if decision.decision != "allow":
            self._audit(
                policy_decision="deny",
                reason=decision.reason,
                mapped_identity=mapped_identity,
                experiment_id=target_experiment_id or claims.experiment_id,
                tool_name=tool_name,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=tool_args_hash,
                approval_id=approval_id,
                error={
                    "code": "POLICY_DENIED",
                    "reason": decision.reason,
                    "tier": int(decision.tier),
                },
            )
            if decision.requires_approval:
                raise ApprovalRequired(decision.reason)
            raise PolicyDenied(decision)

    def _enforce_append_only(
        self,
        *,
        tool_name: str,
        approval_id: str,
        mapped_identity: MappedIdentity | None,
        target_experiment_id: int,
        conversation_id: str | None,
        request_id: str | None,
        keycloak_subject: str | None,
        librechat_user_id: str | None,
        provider: str | None,
        model_id: str | None,
        tool_args_hash: str,
    ) -> None:
        """Check append-only state and raise on violation."""
        assert self.client is not None  # caller guards before invoking
        try:
            target = self.client.get_experiment(target_experiment_id)
        except Exception as exc:
            self._audit(
                policy_decision="allow",
                reason="state_check_failed",
                mapped_identity=mapped_identity,
                experiment_id=target_experiment_id,
                tool_name=tool_name,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=tool_args_hash,
                approval_id=approval_id,
                error={
                    "code": "STATE_CHECK_FAILED",
                    "exception": type(exc).__name__,
                    "message": str(exc),
                },
            )
            raise ElabftwAdapterError(
                reason="client_error",
                message=(
                    f"eLabFTW client raised {type(exc).__name__} during "
                    f"append-only state check: {exc}"
                ),
            ) from exc
        state = int(target.get("state", _STATE_NORMAL) or _STATE_NORMAL)
        locked = bool(target.get("locked", 0))
        if state != _STATE_NORMAL or locked:
            self._audit(
                policy_decision="deny",
                reason="append_only_violation",
                mapped_identity=mapped_identity,
                experiment_id=target_experiment_id,
                tool_name=tool_name,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=tool_args_hash,
                approval_id=approval_id,
                error={
                    "code": "APPEND_ONLY_VIOLATION",
                    "experiment_id": target_experiment_id,
                    "state": state,
                    "locked": locked,
                },
            )
            raise AppendOnlyViolation(target_experiment_id, state, locked)  # type: ignore[arg-type]

    def _consume_write_approval(
        self,
        *,
        plan_approval_id: str | None,
        approval_id: str,
        approval_args: dict[str, Any],
        claims: ContextTokenClaims,
        mapped_identity: MappedIdentity | None,
        target_experiment_id: int | None,
        tool_name: str,
        conversation_id: str | None,
        request_id: str | None,
        keycloak_subject: str | None,
        librechat_user_id: str | None,
        provider: str | None,
        model_id: str | None,
        tool_args_hash: str,
    ) -> str:
        """Consume write approval token (or use plan-level bypass)."""
        if plan_approval_id:
            return plan_approval_id
        args_hash_now = compute_args_hash(approval_args)
        target_record_str = (
            f"elabftw:experiment:{target_experiment_id}"
            if target_experiment_id is not None
            else None
        )
        try:
            self.approval_store.consume(
                approval_id,
                tool_name=tool_name,
                args_hash=args_hash_now,
                target_record=target_record_str,
            )
            return approval_id
        except Exception as exc:
            from lab_copilot_gateway.approval import ApprovalError

            reason_code = "approval_consume_failed"
            error_payload: dict[str, Any] = {
                "code": "APPROVAL_CONSUME_FAILED",
                "exception": type(exc).__name__,
            }
            if isinstance(exc, ApprovalError):
                reason_code = f"approval_denied:{exc.reason}"
                error_payload["approval_reason"] = exc.reason
                error_payload["approval_id"] = exc.approval_id
            self._audit(
                policy_decision="deny",
                reason=reason_code,
                mapped_identity=mapped_identity,
                experiment_id=target_experiment_id or claims.experiment_id,
                tool_name=tool_name,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=tool_args_hash,
                approval_id=approval_id,
                error=error_payload,
            )
            raise ElabftwAdapterError(
                reason=reason_code,
                message=f"approval token consume failed: {exc}",
            ) from exc

    def _require_write_client(
        self,
        *,
        tool_name: str,
        effective_approval_id: str,
        mapped_identity: MappedIdentity | None,
        target_experiment_id: int | None,
        claims: ContextTokenClaims,
        conversation_id: str | None,
        request_id: str | None,
        keycloak_subject: str | None,
        librechat_user_id: str | None,
        provider: str | None,
        model_id: str | None,
        tool_args_hash: str,
    ) -> None:
        """Raise ElabftwAdapterError if the write client is not configured."""
        if self.client is not None:
            return
        self._audit(
            policy_decision="allow",
            reason="elabftw_not_configured",
            mapped_identity=mapped_identity,
            experiment_id=target_experiment_id or claims.experiment_id,
            tool_name=tool_name,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            tool_args_hash=tool_args_hash,
            approval_id=effective_approval_id,
            error={"code": "ELABFTW_NOT_CONFIGURED"},
        )
        raise ElabftwAdapterError(
            reason="elabftw_not_configured",
            message=(
                "eLabFTW client not configured "
                "(set LAB_COPILOT_ELABFTW_BASE_URL and "
                "LAB_COPILOT_ELABFTW_API_KEY)"
            ),
        )

    # --- token + identity verify (shared with read adapter) --------------

    def _verify_token_and_identity(
        self,
        *,
        context_token: str,
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None,
        request_id: str | None,
        keycloak_subject: str | None,
        librechat_user_id: str | None,
        provider: str | None,
        model_id: str | None,
        tool_name: str,
        early_args_for_hash: dict[str, Any],
    ) -> tuple[ContextTokenClaims, str] | tuple[Exception]:
        """Verify the context token and bind it to the mapped identity.

        Returns ``(claims, tool_args_hash)`` on success, ``(exception,)`` on
        failure (the audit row has already been written — callers re-raise).
        """
        early_hash = compute_args_hash(
            {"tool_name": tool_name, "args_digest": _redact(context_token)}
            if context_token
            else {"tool_name": tool_name, "token_digest": "missing"}
        )

        # 1. Token presence + signature + expiry.
        if not context_token:
            self._audit(
                policy_decision="deny",
                reason="missing_context_token",
                mapped_identity=mapped_identity,
                experiment_id=None,
                tool_name=tool_name,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=early_hash,
                error={"code": "MISSING_CONTEXT_TOKEN"},
            )
            return (InvalidContextToken("missing"),)

        try:
            claims = verify_context_token(context_token)
        except InvalidContextToken as exc:
            self._audit(
                policy_decision="deny",
                reason=f"invalid_context_token:{exc.detail}",
                mapped_identity=mapped_identity,
                experiment_id=None,
                tool_name=tool_name,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=early_hash,
                error={"code": "INVALID_CONTEXT_TOKEN", "detail": exc.detail},
            )
            return (exc,)

        tool_args_hash = compute_args_hash(
            {"experiment_id": claims.experiment_id, "args": early_args_for_hash}
        )

        # 2. Identity resolution — unmapped caller → deny.
        if mapped_identity is None:
            self._audit(
                policy_decision="deny",
                reason="unmapped_caller",
                mapped_identity=None,
                experiment_id=claims.experiment_id,
                tool_name=tool_name,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=tool_args_hash,
                error={"code": "UNMAPPED_CALLER"},
            )
            return (UnmappedCaller(),)

        # 3. Three-way cross-user binding.
        if claims.mapped_elabftw_user_id != mapped_identity.elabftw_user_id:
            self._audit(
                policy_decision="deny",
                reason="context_token_user_mismatch",
                mapped_identity=mapped_identity,
                experiment_id=claims.experiment_id,
                tool_name=tool_name,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=tool_args_hash,
                error={"code": "CONTEXT_TOKEN_USER_MISMATCH"},
            )
            return (InvalidContextToken("token user does not match resolved identity"),)

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
                    tool_name=tool_name,
                    conversation_id=conversation_id,
                    request_id=request_id,
                    keycloak_subject=keycloak_subject,
                    librechat_user_id=librechat_user_id,
                    provider=provider,
                    model_id=model_id,
                    tool_args_hash=tool_args_hash,
                    error={"code": "CONTEXT_TOKEN_KC_SUBJECT_MISMATCH"},
                )
                return (
                    InvalidContextToken(
                        "token keycloak_subject does not match resolved identity"
                    ),
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
                    tool_name=tool_name,
                    conversation_id=conversation_id,
                    request_id=request_id,
                    keycloak_subject=keycloak_subject,
                    librechat_user_id=librechat_user_id,
                    provider=provider,
                    model_id=model_id,
                    tool_args_hash=tool_args_hash,
                    error={"code": "CONTEXT_TOKEN_LC_USER_MISMATCH"},
                )
                return (
                    InvalidContextToken(
                        "token librechat_user_id does not match resolved identity"
                    ),
                )

        return claims, tool_args_hash

    # --- operation implementations --------------------------------------

    def _exec_create_draft(
        self, client: ElabftwClient, *, target_title: str | None
    ) -> int:
        """Create experiment + patch title (body can be left empty)."""
        new_id = client.create_experiment(title=target_title)
        return new_id

    def _exec_amend(
        self,
        client: ElabftwClient,
        *,
        target_experiment_id: int,
        amendment_html: str,
        attachment_filename: str | None,
        attachment_data: bytes | None,
        attachment_comment: str,
    ) -> int:
        """Read, append amendment, PATCH back; upload attachment if provided.

        Returns the new upload id (when an attachment was uploaded) or 0.
        """
        current = client.get_experiment(target_experiment_id)
        existing_body = str(current.get("body", "") or "")
        # Append the amendment as a dedicated HTML section.  The boundary is
        # explicit so eLabFTW viewers can see where the AI-generated section
        # begins and the original experiment body ended.
        amended_body = self._wrap_amendment(existing_body, amendment_html)
        client.patch_experiment_body(target_experiment_id, amended_body)
        if attachment_filename and attachment_data is not None:
            return client.upload_attachment(
                target_experiment_id,
                attachment_filename,
                attachment_data,
                comment=attachment_comment,
            )
        return 0

    @staticmethod
    def _wrap_amendment(existing_body: str, amendment_html: str) -> str:
        """Wrap the amendment in an HTML section marker.

        The wrapper is a styled ``<section>`` so eLabFTW viewers can visually
        identify AI-generated amendments and so the gateway can later parse
        them back out if needed.
        """
        if not amendment_html:
            return existing_body
        section = (
            '<section data-source="lab-copilot-gateway" '
            'data-amendment="1">'
            f"{amendment_html}"
            "</section>"
        )
        if not existing_body:
            return section
        return existing_body.rstrip() + "\n\n" + section

    def _exec_edit_section(
        self,
        client: ElabftwClient,
        *,
        target_experiment_id: int,
        approval_args: dict[str, Any],
    ) -> int:
        """Direct-edit executor (C35): verify body hasn't drifted, then PATCH.

        Reads the current body, recomputes its SHA-256, and compares to the
        ``old_body_hash`` captured in the approval args. If they differ, the
        record changed between approval and execution → refuse (a stale edit
        must not clobber a newer state). On match, PATCH the body to the
        approved ``new_body``. Returns 0 (no upload id for a direct edit).
        """
        expected_old_hash = str(approval_args.get("old_body_hash") or "")
        new_body = str(approval_args.get("new_body") or "")
        current = client.get_experiment(target_experiment_id)
        current_body = str(current.get("body", "") or "")
        current_hash = hashlib.sha256(current_body.encode("utf-8")).hexdigest()
        # C35: old_body_hash is REQUIRED, not optional. The orchestrator
        # always reads the body before proposing an edit (to generate the
        # preview), so it always has the hash. Refusing on a missing hash
        # closes the safety hole where a stale edit could clobber a newer
        # state with zero drift protection.
        if not expected_old_hash:
            raise ElabftwAdapterError(
                reason="missing_old_body_hash",
                message=f"edit_experiment_section requires old_body_hash in approval_args for experiment {target_experiment_id}",
            )
        if current_hash != expected_old_hash:
            # Body drifted since the approval was issued. Do NOT patch —
            # the user approved an edit against a now-stale snapshot. The
            # caller must re-read, re-preview, and re-approve.
            # NOTE: TOCTOU window exists between this read and the patch
            # below; eLabFTW revision history mitigates (both versions
            # are preserved upstream).
            raise ElabftwAdapterError(
                reason="body_drift_since_approval",
                message=f"body drift detected for experiment {target_experiment_id}",
            )
        client.patch_experiment_body(target_experiment_id, new_body)
        return 0

    # --- provenance writeback --------------------------------------------

    def _write_provenance(
        self,
        *,
        client: ElabftwClient,
        experiment_id: int,
        existing_audit_action_id: str | None,
        tool_args_hash: str,
        approval_id: str,
        mapped_identity: MappedIdentity | None,
        tool_name: str = TOOL_NAME_AMEND,
        conversation_id: str | None = None,
        request_id: str | None = None,
        keycloak_subject: str | None = None,
        librechat_user_id: str | None = None,
        provider: str | None = None,
        model_id: str | None = None,
    ) -> str:
        """Embed the gateway audit action_id into the experiment metadata.

        Idempotent: if ``existing_audit_action_id`` is provided, append a new
        entry to a list rather than overwriting.

        Best-effort — failures to PATCH metadata are logged in the audit row
        but do not block the write from succeeding (the write already landed
        in the experiment body).  Provenance is observability, not enforcement.
        """
        import secrets as _secrets
        import time as _time

        audit_action_id = (
            existing_audit_action_id
            or f"{self.action_id_prefix}-{int(_time.time() * 1000)}-{_secrets.token_hex(4)}"
        )

        # Read current metadata so we merge rather than overwrite.  The
        # eLabFTW metadata field is a JSON-encoded string of extra_fields.
        try:
            current = client.get_experiment(experiment_id)
        except Exception as exc:
            # Read failed — log but do not fail the write.  The body
            # amendment is already applied; provenance is best-effort.
            self._audit(
                policy_decision="allow",
                reason="provenance_readback_failed",
                mapped_identity=mapped_identity,
                experiment_id=experiment_id,
                tool_name=tool_name,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=tool_args_hash,
                approval_id=approval_id,
                error={
                    "code": "PROVENANCE_READBACK_FAILED",
                    "exception": type(exc).__name__,
                    "message": str(exc),
                },
            )
            return audit_action_id

        existing_metadata = _merge_provenance_into_metadata(
            _normalize_metadata(current.get("metadata")),
            audit_action_id,
        )

        try:
            client.patch_experiment_metadata(experiment_id, existing_metadata)
        except Exception as exc:
            self._audit(
                policy_decision="allow",
                reason="provenance_writeback_failed",
                mapped_identity=mapped_identity,
                experiment_id=experiment_id,
                tool_name=tool_name,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=tool_args_hash,
                approval_id=approval_id,
                error={
                    "code": "PROVENANCE_WRITEBACK_FAILED",
                    "exception": type(exc).__name__,
                    "message": str(exc),
                },
            )
            return audit_action_id

        return audit_action_id

    # --- audit writer (mirror C08 read adapter) --------------------------

    def _audit(
        self,
        *,
        policy_decision: str,
        reason: str,
        mapped_identity: MappedIdentity | None,
        experiment_id: int | None,
        tool_name: str = TOOL_NAME_AMEND,
        conversation_id: str | None = None,
        request_id: str | None = None,
        keycloak_subject: str | None = None,
        librechat_user_id: str | None = None,
        provider: str | None = None,
        model_id: str | None = None,
        tool_args_hash: str | None = None,
        approval_id: str | None = None,
        api_call_summary: dict[str, Any] | None = None,
        result_summary: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        require_persistence: bool = False,
    ) -> None:
        """Persist an audit record — same semantics as the read adapter's.

        ``tool_name`` defaults to the amend tool because most write paths go
        through it; the draft path explicitly passes ``TOOL_NAME_DRAFT``.
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
            tool_name=tool_name,
            tool_args_hash=tool_args_hash,
            context_refs=context_refs,
            policy_decision=policy_decision,
            approval_id=approval_id,
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
            pass


# --- module-level singleton for C09 write adapter -------------------------
_default_write_adapter: ElabftwWriteAdapter | None = None


def get_elabftw_write_adapter() -> ElabftwWriteAdapter:
    """Return the process-wide eLabFTW write adapter (created lazily)."""
    global _default_write_adapter
    if _default_write_adapter is None:
        from lab_copilot_gateway.approval import get_approval_store
        from lab_copilot_gateway.audit import get_audit_store
        from lab_copilot_gateway.policy import get_policy_engine

        _default_write_adapter = ElabftwWriteAdapter(
            policy_engine=get_policy_engine(),
            audit_store=get_audit_store(),
            approval_store=get_approval_store(),
            client=_default_client_from_env(),
        )
    return _default_write_adapter


def reset_elabftw_write_adapter(
    adapter: ElabftwWriteAdapter | None = None,
) -> None:
    """Test helper: replace or clear the singleton."""
    global _default_write_adapter
    _default_write_adapter = adapter
