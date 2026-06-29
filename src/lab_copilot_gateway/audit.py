"""Append-only audit record store for lab copilot gateway.

Storage: SQLite for local development  (Postgres-compatible type choices).
Semantics: records are append-only.  No UPDATE or DELETE is exposed.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS audit_records (
    action_id              TEXT PRIMARY KEY,
    conversation_id        TEXT,
    request_id            TEXT,
    keycloak_subject      TEXT,
    librechat_user_id     TEXT,
    mapped_elabftw_user_id  TEXT,
    mapped_elabftw_team_id  TEXT,
    provider              TEXT,
    model_id              TEXT,
    tool_name             TEXT,
    tool_args_hash        TEXT,
    context_refs_json      TEXT,
    policy_decision        TEXT,
    approval_id           TEXT,
    api_call_summary_json  TEXT,
    result_summary_json    TEXT,
    error_json            TEXT,
    artifact_manifest_json TEXT,
    created_at            TEXT NOT NULL
);
"""

# Append-only guard: reject any UPDATE/DELETE attempted through this module.
_INSERT_SQL = """
INSERT INTO audit_records (
    action_id, conversation_id, request_id, keycloak_subject,
    librechat_user_id, mapped_elabftw_user_id, mapped_elabftw_team_id,
    provider, model_id, tool_name, tool_args_hash, context_refs_json,
    policy_decision, approval_id, api_call_summary_json,
    result_summary_json, error_json, artifact_manifest_json, created_at
) VALUES (
    :action_id, :conversation_id, :request_id, :keycloak_subject,
    :librechat_user_id, :mapped_elabftw_user_id, :mapped_elabftw_team_id,
    :provider, :model_id, :tool_name, :tool_args_hash, :context_refs_json,
    :policy_decision, :approval_id, :api_call_summary_json,
    :result_summary_json, :error_json, :artifact_manifest_json, :created_at
);
"""


@dataclass
class AuditRecord:
    """A single audit-worthy action attempted through the gateway."""

    action_id: str
    conversation_id: str | None = None
    request_id: str | None = None
    keycloak_subject: str | None = None
    librechat_user_id: str | None = None
    mapped_elabftw_user_id: str | None = None
    mapped_elabftw_team_id: str | None = None
    provider: str | None = None
    model_id: str | None = None
    tool_name: str | None = None
    tool_args_hash: str | None = None
    context_refs: list[dict[str, Any]] = field(default_factory=list)
    policy_decision: str | None = None
    approval_id: str | None = None
    api_call_summary: dict[str, Any] = field(default_factory=dict)
    result_summary: dict[str, Any] = field(default_factory=dict)
    error: dict[str, Any] | None = None
    artifact_manifest: list[dict[str, Any]] = field(default_factory=list)
    created_at: str | None = None  # set by store

    def to_row(self) -> dict[str, Any]:
        """Flatten to a row suitable for parameterized INSERT."""
        return {
            "action_id": self.action_id,
            "conversation_id": self.conversation_id,
            "request_id": self.request_id,
            "keycloak_subject": self.keycloak_subject,
            "librechat_user_id": self.librechat_user_id,
            "mapped_elabftw_user_id": self.mapped_elabftw_user_id,
            "mapped_elabftw_team_id": self.mapped_elabftw_team_id,
            "provider": self.provider,
            "model_id": self.model_id,
            "tool_name": self.tool_name,
            "tool_args_hash": self.tool_args_hash,
            "context_refs_json": _dumps(self.context_refs),
            "policy_decision": self.policy_decision,
            "approval_id": self.approval_id,
            "api_call_summary_json": _dumps(self.api_call_summary),
            "result_summary_json": _dumps(self.result_summary),
            "error_json": _dumps(self.error),
            "artifact_manifest_json": _dumps(self.artifact_manifest),
            "created_at": self.created_at,
        }


def _dumps(value: Any) -> str | None:
    """Serialize to JSON, preserving None as NULL (not the string 'null')."""
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, default=str)


class AuditStore:
    """Append-only audit store backed by SQLite.

    Thread-safe through a per-instance lock; SQLite connections are not
    shared across threads by default.  One store instance per process is
    the intended usage pattern.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        env_path = os.getenv("LAB_COPILOT_AUDIT_DB")
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

    def append(self, record: AuditRecord) -> str:
        """Persist a single audit record.  Returns the created_at timestamp."""
        import datetime as _dt

        if record.created_at is None:
            record.created_at = _dt.datetime.now(_dt.timezone.utc).isoformat()

        if not record.action_id:
            raise ValueError("audit record requires non-empty action_id")

        conn = self._connect()
        with self._lock:
            conn.execute(_INSERT_SQL, record.to_row())
            conn.commit()
        return record.created_at

    def count(self) -> int:
        """Total audit rows.  Used by tests; no general read API is exposed."""
        conn = self._connect()
        with self._lock:
            row = conn.execute("SELECT COUNT(*) FROM audit_records").fetchone()
        return int(row[0]) if row else 0

    def get(self, action_id: str) -> dict[str, Any] | None:
        """Fetch one row by action_id for test assertions and admin reads."""
        conn = self._connect()
        with self._lock:
            row = conn.execute(
                "SELECT * FROM audit_records WHERE action_id = ?",
                (action_id,),
            ).fetchone()
        if row is None:
            return None
        cols = [
            d[0]
            for d in conn.execute("SELECT * FROM audit_records LIMIT 0").description
        ]
        return dict(zip(cols, row))

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


# --- module-level singleton for FastAPI dependency injection ---------------
_default_store: AuditStore | None = None


def get_audit_store() -> AuditStore:
    """Return the process-wide audit store (created lazily)."""
    global _default_store
    if _default_store is None:
        _default_store = AuditStore()
    return _default_store
