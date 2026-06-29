"""Single-use approval token flow for the lab copilot gateway (C07).

The approval store enforces that mutating tool calls (Tier >= BOUNDED_WRITES
in V1) only execute when the caller presents a real, unconsumed, hash-bound,
unexpired approval that the gateway itself issued.

Threat model and guarantees:

    * LibreChat displays approval UI but is NOT trusted as the enforcement
      authority.  All approval-token validation happens server-side in this
      module.  A tampered client request carrying a forged approval_id, a
      replayed token, or a token bound to different args is rejected.

    * Approvals are single-use.  ``consume()`` is atomic: it verifies
      (tool_name, args_hash, target_record) match, expiry has not elapsed,
      and the row is not already consumed; if all checks pass it marks
      ``consumed_at``.  Concurrent calls race on the SQLite row update; the
      row-level ``consumed_at IS NULL`` predicate guarantees only one call
      wins, the rest see ``already_consumed``.

    * Args are hashed with SHA-256 over canonical JSON
      (``sort_keys=True, separators=(",", ":")``).  The hash is stored at
      request time; ``consume()`` recomputes the hash from the args the
      caller is actually trying to use and compares.  Any field reorder,
      extra field, or value change yields a different hash and is denied.

Schema (SQLite, Postgres-compatible types):

    approval_id              TEXT PRIMARY KEY        opaque token
    tool_name                TEXT NOT NULL           exact registry name
    args_hash                TEXT NOT NULL           sha256 hex of canonical args JSON
    target_record            TEXT                    opaque e.g. elabftw:experiment:123
    tier                     INTEGER NOT NULL        risk tier at request time
    keycloak_subject         TEXT                    requesting identity
    librechat_user_id        TEXT
    mapped_elabftw_user_id   TEXT                    resolved at request time
    provider                 TEXT
    model_id                 TEXT
    expires_at               TEXT NOT NULL           ISO8601 UTC
    consumed_at              TEXT                    NULL until single-use consume
    created_at               TEXT NOT NULL           ISO8601 UTC
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import secrets
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS approvals (
    approval_id              TEXT PRIMARY KEY,
    tool_name                TEXT NOT NULL,
    args_hash                TEXT NOT NULL,
    target_record            TEXT,
    tier                     INTEGER NOT NULL,
    keycloak_subject         TEXT,
    librechat_user_id        TEXT,
    mapped_elabftw_user_id   TEXT,
    provider                 TEXT,
    model_id                 TEXT,
    expires_at               TEXT NOT NULL,
    consumed_at              TEXT,
    created_at               TEXT NOT NULL
);
"""

_INSERT_SQL = """
INSERT INTO approvals (
    approval_id, tool_name, args_hash, target_record, tier,
    keycloak_subject, librechat_user_id, mapped_elabftw_user_id,
    provider, model_id, expires_at, consumed_at, created_at
) VALUES (
    :approval_id, :tool_name, :args_hash, :target_record, :tier,
    :keycloak_subject, :librechat_user_id, :mapped_elabftw_user_id,
    :provider, :model_id, :expires_at, :consumed_at, :created_at
);
"""

# Atomic single-use consume with hard expiry enforcement.  Both conditions
# are part of the UPDATE predicate so a token cannot race past expiry or be
# replayed; rowcount==1 means this call won the race and consumed the token.
# rowcount==0 => either already consumed, expired at the moment of the
# UPDATE, or no such approval_id.
_CONSUME_SQL = """
UPDATE approvals
   SET consumed_at = :now
 WHERE approval_id = :approval_id
   AND consumed_at IS NULL
   AND datetime(expires_at) > datetime(:now);
"""

_SELECT_SQL = """
SELECT approval_id, tool_name, args_hash, target_record, tier,
       keycloak_subject, librechat_user_id, mapped_elabftw_user_id,
       provider, model_id, expires_at, consumed_at, created_at
  FROM approvals
 WHERE approval_id = ?;
"""

# Default TTL for an approval token in seconds (10 minutes).  Caller can
# override per-request; short enough to bound replay windows, long enough
# for a human to act on the LibreChat approval card.
DEFAULT_APPROVAL_TTL_SECONDS = 600


# --- exceptions ------------------------------------------------------------


class ApprovalError(Exception):
    """Base class for approval-consumption failures.

    ``reason`` is a stable machine-readable code; ``message`` is human-readable.
    Tests and the API layer switch on ``reason``.
    """

    def __init__(
        self, reason: str, message: str, approval_id: str | None = None
    ) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message
        self.approval_id = approval_id

    def to_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            "reason": self.reason,
            "message": self.message,
        }
        if self.approval_id is not None:
            out["approval_id"] = self.approval_id
        return out


class ApprovalNotFound(ApprovalError):
    """approval_id does not exist in the store."""

    def __init__(self, approval_id: str) -> None:
        super().__init__(
            reason="not_found",
            message=f"approval_id {approval_id!r} does not exist",
            approval_id=approval_id,
        )


class ApprovalAlreadyConsumed(ApprovalError):
    """approval_id was already consumed (single-use replay attempt)."""

    def __init__(self, approval_id: str) -> None:
        super().__init__(
            reason="already_consumed",
            message=f"approval_id {approval_id!r} has already been consumed",
            approval_id=approval_id,
        )


class ApprovalExpired(ApprovalError):
    """approval_id's expires_at has elapsed."""

    def __init__(self, approval_id: str, expires_at: str) -> None:
        super().__init__(
            reason="expired",
            message=f"approval_id {approval_id!r} expired at {expires_at}",
            approval_id=approval_id,
        )


class ApprovalMismatch(ApprovalError):
    """Caller supplied a token whose bound fields do not match the request."""

    def __init__(self, approval_id: str | None, field_name: str) -> None:
        super().__init__(
            reason="mismatch",
            message=f"approval {approval_id!r} does not match request field {field_name!r}",
            approval_id=approval_id,
        )
        self.field_name = field_name


# --- args hashing ----------------------------------------------------------


def compute_args_hash(args: dict[str, Any] | None) -> str:
    """SHA-256 hex of canonical JSON encoding of ``args``.

    Canonical form: ``json.dumps(sort_keys=True, separators=(",", ":"))``
    produces a byte-stable encoding independent of dict insertion order.  An
    absent ``args`` is hashed as the empty canonical object ``{}``.

    Raises ``TypeError`` if ``args`` contains any non-JSON-native type —
    callers must only pass JSON primitives (dict, list, str, int, float,
    bool, None).  Silent fallback to ``str()`` would produce hash instabilities
    across Python versions and contexts, so we explicitly reject it.
    """
    canonical = json.dumps(args or {}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# --- data classes ----------------------------------------------------------


@dataclass(frozen=True)
class ApprovalRequest:
    """Inputs needed to issue a single-use approval token.

    Captured at request time so ``consume()`` can verify the caller is using
    the token for exactly the request that was approved — no field may differ.
    """

    tool_name: str
    args_hash: str
    target_record: str | None
    tier: int
    keycloak_subject: str | None = None
    librechat_user_id: str | None = None
    mapped_elabftw_user_id: str | None = None
    provider: str | None = None
    model_id: str | None = None
    ttl_seconds: int = DEFAULT_APPROVAL_TTL_SECONDS


@dataclass(frozen=True)
class ApprovalRecord:
    """All fields of an approval row, in the shape returned by ``get()``."""

    approval_id: str
    tool_name: str
    args_hash: str
    target_record: str | None
    tier: int
    keycloak_subject: str | None
    librechat_user_id: str | None
    mapped_elabftw_user_id: str | None
    provider: str | None
    model_id: str | None
    expires_at: str
    consumed_at: str | None
    created_at: str

    def is_consumed(self) -> bool:
        return self.consumed_at is not None

    def is_expired(self, *, now: str | None = None) -> bool:
        stamp = now or _now_iso()
        return stamp >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "approval_id": self.approval_id,
            "tool_name": self.tool_name,
            "args_hash": self.args_hash,
            "target_record": self.target_record,
            "tier": self.tier,
            "keycloak_subject": self.keycloak_subject,
            "librechat_user_id": self.librechat_user_id,
            "mapped_elabftw_user_id": self.mapped_elabftw_user_id,
            "provider": self.provider,
            "model_id": self.model_id,
            "expires_at": self.expires_at,
            "consumed_at": self.consumed_at,
            "created_at": self.created_at,
        }


@dataclass
class ConsumeResult:
    """Return value of ``ApprovalStore.consume()`` on success."""

    approval_id: str
    tool_name: str
    args_hash: str
    target_record: str | None
    tier: int
    consumed_at: str


# --- store -----------------------------------------------------------------


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _new_approval_id() -> str:
    # 16 bytes (~128 bits) of URL-safe entropy.  Not a JWT — the gateway owns
    # the issuance and verification, never the client.
    return secrets.token_urlsafe(16)


class ApprovalStore:
    """Single-use approval token store backed by SQLite.

    Thread-safe through a per-instance lock; one instance per process is the
    intended usage pattern (mirrors ``AuditStore``).
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        env_path = os.getenv("LAB_COPILOT_APPROVAL_DB")
        self.db_path: str = str(db_path or env_path or ":memory:")
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            # check_same_thread=False because we guard with our own lock.
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.executescript(SCHEMA_SQL)
            self._conn.commit()
        return self._conn

    def request(self, req: ApprovalRequest) -> tuple[str, str]:
        """Issue a new single-use approval token.

        Returns ``(approval_id, expires_at)``.  ``approval_id`` is opaque to
        the caller; the gateway hands it to LibreChat for display only.
        """
        if not req.tool_name:
            raise ValueError("approval request requires non-empty tool_name")
        if not req.args_hash:
            raise ValueError("approval request requires args_hash")
        if req.ttl_seconds <= 0:
            raise ValueError("approval request requires ttl_seconds > 0")

        created_at = _now_iso()
        # Reason: compute expiry server-side so a tampered client cannot extend
        # the window.  Use UTC ISO8601 throughout for stable comparisons.
        expires_dt = _dt.datetime.fromisoformat(created_at) + _dt.timedelta(
            seconds=req.ttl_seconds
        )
        expires_at = expires_dt.isoformat()

        approval_id = _new_approval_id()
        row = {
            "approval_id": approval_id,
            "tool_name": req.tool_name,
            "args_hash": req.args_hash,
            "target_record": req.target_record,
            "tier": req.tier,
            "keycloak_subject": req.keycloak_subject,
            "librechat_user_id": req.librechat_user_id,
            "mapped_elabftw_user_id": req.mapped_elabftw_user_id,
            "provider": req.provider,
            "model_id": req.model_id,
            "expires_at": expires_at,
            "consumed_at": None,
            "created_at": created_at,
        }
        conn = self._connect()
        with self._lock:
            conn.execute(_INSERT_SQL, row)
            conn.commit()
        return approval_id, expires_at

    def get(self, approval_id: str) -> ApprovalRecord | None:
        """Fetch one approval row by id (admin / test reads)."""
        conn = self._connect()
        with self._lock:
            row = conn.execute(_SELECT_SQL, (approval_id,)).fetchone()
        if row is None:
            return None
        cols = [
            d[0] for d in conn.execute("SELECT * FROM approvals LIMIT 0").description
        ]
        data = dict(zip(cols, row))
        return ApprovalRecord(
            approval_id=data["approval_id"],
            tool_name=data["tool_name"],
            args_hash=data["args_hash"],
            target_record=data["target_record"],
            tier=int(data["tier"]),
            keycloak_subject=data["keycloak_subject"],
            librechat_user_id=data["librechat_user_id"],
            mapped_elabftw_user_id=data["mapped_elabftw_user_id"],
            provider=data["provider"],
            model_id=data["model_id"],
            expires_at=data["expires_at"],
            consumed_at=data["consumed_at"],
            created_at=data["created_at"],
        )

    def consume(
        self,
        approval_id: str,
        *,
        tool_name: str,
        args_hash: str,
        target_record: str | None = None,
    ) -> ConsumeResult:
        """Atomically verify and consume a single-use approval token.

        Verifications, in order (cheapest first):

            1. approval_id exists in the store.
            2. ``tool_name`` matches the bound tool.
            3. ``args_hash`` matches the bound hash  (catches modified args).
            4. ``target_record`` matches if the caller provides one.
            5. Token has not expired (advisory check; see step 6 for hard
               enforcement).
            6. Token has not already been consumed (advisory check; see step
               7 for hard enforcement).

        On success, marks ``consumed_at`` and returns ``ConsumeResult``.
        On any failure, raises one of the ``ApprovalError`` subclasses.

        Concurrency / TOCTOU hardening: the final UPDATE is the source of
        truth.  Its WHERE clause requires BOTH ``consumed_at IS NULL`` AND
        ``datetime(expires_at) > datetime(:now)`` at the moment of the
        UPDATE.  A token cannot race past expiry or be replayed concurrently:
        only one concurrent UPDATE will see rowcount == 1, the rest will see
        rowcount == 0.

        When rowcount == 0 there are three possible causes (in priority
        order): already consumed (we re-read the record to distinguish),
        expired between the soft check and UPDATE, or never existed (already
        caught at step 1).
        """
        record = self.get(approval_id)
        if record is None:
            raise ApprovalNotFound(approval_id)
        # Field-level mismatches (the gateway should never hand these to
        # LibreChat in the first place — defence in depth).
        if record.tool_name != tool_name:
            raise ApprovalMismatch(approval_id, "tool_name")
        if record.args_hash != args_hash:
            raise ApprovalMismatch(approval_id, "args_hash")
        if target_record is not None and record.target_record != target_record:
            raise ApprovalMismatch(approval_id, "target_record")

        # Advisory checks — give callers a fast, readable error before the
        # atomic UPDATE.  The authoritative checks are in _CONSUME_SQL below.
        if record.is_expired():
            raise ApprovalExpired(approval_id, record.expires_at)
        if record.is_consumed():
            raise ApprovalAlreadyConsumed(approval_id)

        # Atomic single-use + hard expiry marker.  Two concurrent calls may
        # both pass the advisory checks above, but only one UPDATE will hit a
        # row that is both unconsumed AND still within its expiry window at
        # the moment of the UPDATE.
        now = _now_iso()
        conn = self._connect()
        with self._lock:
            cur = conn.execute(_CONSUME_SQL, {"approval_id": approval_id, "now": now})
            conn.commit()
            if cur.rowcount == 0:
                # Lost the race.  Re-read to distinguish already-consumed vs
                # expired-at-the-moment-of-update.
                post = self.get(approval_id)
                if post is not None and post.is_consumed():
                    raise ApprovalAlreadyConsumed(approval_id)
                # Reason: not consumed but UPDATE still missed → expired
                # between the soft check and the atomic UPDATE.
                raise ApprovalExpired(approval_id, record.expires_at)

        return ConsumeResult(
            approval_id=approval_id,
            tool_name=tool_name,
            args_hash=args_hash,
            target_record=record.target_record,
            tier=record.tier,
            consumed_at=now,
        )

    def count(self) -> int:
        conn = self._connect()
        with self._lock:
            row = conn.execute("SELECT COUNT(*) FROM approvals").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


# --- module-level singleton for FastAPI dependency injection ---------------
_default_store: ApprovalStore | None = None


def get_approval_store() -> ApprovalStore:
    """Return the process-wide approval store (created lazily)."""
    global _default_store
    if _default_store is None:
        _default_store = ApprovalStore()
    return _default_store


def reset_approval_store(store: ApprovalStore | None = None) -> None:
    """Test helper: replace or clear the singleton."""
    global _default_store
    _default_store = store
