"""Durable asynchronous strategy run store (Slice 7).

Persists run metadata for strategy executions — operation IDs, dependencies,
opaque sequence refs, statuses — but **no sequence content**.

The run store enables:

* Browser disconnect/reconnect — the UI can poll status after a refresh.
* Gateway restart recovery — active runs are marked ``interrupted`` on
  startup; they are not auto-resumed (sequence references are process-local).
* Cancellation at safe boundaries — between operations, not mid-call.

Schema (SQLite, Postgres-compatible types):

    run_id              TEXT PRIMARY KEY     opaque token
    strategy_id         TEXT NOT NULL        strategy identifier
    plan_hash           TEXT NOT NULL        SHA-256 of canonical JSON
    approval_id         TEXT NOT NULL        consumed approval token
    target_record       TEXT                 experiment + conversation binding
    status              TEXT NOT NULL        pending/running/completed/failed/blocked/ambiguous/interrupted/cancelled
    operation_results   TEXT NOT NULL        JSON array of per-op results
    product_sequence_ids TEXT                JSON object: molecule_id → seq_id
    created_at          TEXT NOT NULL        ISO8601 UTC
    updated_at          TEXT NOT NULL        ISO8601 UTC
    completed_at        TEXT                 NULL until terminal
    error               TEXT                 error message if failed
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import secrets
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Any

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS strategy_runs (
    run_id               TEXT PRIMARY KEY,
    strategy_id          TEXT NOT NULL,
    plan_hash            TEXT NOT NULL,
    approval_id          TEXT NOT NULL,
    target_record        TEXT,
    status               TEXT NOT NULL,
    operation_results    TEXT NOT NULL DEFAULT '[]',
    product_sequence_ids TEXT NOT NULL DEFAULT '{}',
    created_at           TEXT NOT NULL,
    updated_at           TEXT NOT NULL,
    completed_at         TEXT,
    error                TEXT
);
"""

_INSERT_SQL = """
INSERT INTO strategy_runs (
    run_id, strategy_id, plan_hash, approval_id, target_record,
    status, operation_results, product_sequence_ids,
    created_at, updated_at, completed_at, error
) VALUES (
    :run_id, :strategy_id, :plan_hash, :approval_id, :target_record,
    :status, :operation_results, :product_sequence_ids,
    :created_at, :updated_at, :completed_at, :error
);
"""

_UPDATE_STATUS_SQL = """
UPDATE strategy_runs
   SET status = :status,
       updated_at = :updated_at,
       completed_at = :completed_at,
       error = :error,
       operation_results = :operation_results,
       product_sequence_ids = :product_sequence_ids
 WHERE run_id = :run_id;
"""

_SELECT_SQL = """
SELECT run_id, strategy_id, plan_hash, approval_id, target_record,
       status, operation_results, product_sequence_ids,
       created_at, updated_at, completed_at, error
  FROM strategy_runs
 WHERE run_id = ?;
"""

_SELECT_ACTIVE_SQL = """
SELECT run_id FROM strategy_runs
 WHERE status IN ('pending', 'running');
"""

# --- Run status constants ---------------------------------------------------

PENDING = "pending"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
BLOCKED = "blocked"
AMBIGUOUS = "ambiguous"
INTERRUPTED = "interrupted"
CANCELLED = "cancelled"

TERMINAL_STATUSES = frozenset(
    {COMPLETED, FAILED, BLOCKED, AMBIGUOUS, INTERRUPTED, CANCELLED}
)
ACTIVE_STATUSES = frozenset({PENDING, RUNNING})


# --- Data classes ------------------------------------------------------------


@dataclass(frozen=True)
class StrategyRun:
    """A strategy execution run record."""

    run_id: str
    strategy_id: str
    plan_hash: str
    approval_id: str
    target_record: str | None
    status: str
    operation_results: list[dict[str, Any]] = field(default_factory=list)
    product_sequence_ids: dict[str, int] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""
    completed_at: str | None = None
    error: str | None = None

    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "strategy_id": self.strategy_id,
            "plan_hash": self.plan_hash,
            "approval_id": self.approval_id,
            "target_record": self.target_record,
            "status": self.status,
            "operation_results": self.operation_results,
            "product_sequence_ids": self.product_sequence_ids,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }


# --- Store -------------------------------------------------------------------


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _new_run_id() -> str:
    return secrets.token_urlsafe(16)


class StrategyRunStore:
    """SQLite-backed store for strategy execution runs.

    Thread-safe through a per-instance lock; one instance per process.
    """

    def __init__(self, db_path: str | None = None) -> None:
        env_path = os.getenv("LAB_COPILOT_STRATEGY_RUN_DB")
        self.db_path: str = str(db_path or env_path or ":memory:")
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.executescript(SCHEMA_SQL)
            self._conn.commit()
        return self._conn

    def create(
        self,
        *,
        strategy_id: str,
        plan_hash: str,
        approval_id: str,
        target_record: str | None = None,
    ) -> StrategyRun:
        """Create a new run record with status 'pending'."""
        now = _now_iso()
        run_id = _new_run_id()
        row = {
            "run_id": run_id,
            "strategy_id": strategy_id,
            "plan_hash": plan_hash,
            "approval_id": approval_id,
            "target_record": target_record,
            "status": PENDING,
            "operation_results": "[]",
            "product_sequence_ids": "{}",
            "created_at": now,
            "updated_at": now,
            "completed_at": None,
            "error": None,
        }
        conn = self._connect()
        with self._lock:
            conn.execute(_INSERT_SQL, row)
            conn.commit()
        return self.get(run_id)  # type: ignore[return-value]

    def get(self, run_id: str) -> StrategyRun | None:
        """Fetch a run by ID."""
        conn = self._connect()
        with self._lock:
            row = conn.execute(_SELECT_SQL, (run_id,)).fetchone()
        if row is None:
            return None
        return _row_to_run(row)

    def update_status(
        self,
        run_id: str,
        *,
        status: str,
        operation_results: list[dict[str, Any]] | None = None,
        product_sequence_ids: dict[str, int] | None = None,
        error: str | None = None,
    ) -> StrategyRun | None:
        """Update a run's status. Sets completed_at if terminal."""
        now = _now_iso()
        completed_at = now if status in TERMINAL_STATUSES else None

        conn = self._connect()
        with self._lock:
            conn.execute(
                _UPDATE_STATUS_SQL,
                {
                    "run_id": run_id,
                    "status": status,
                    "updated_at": now,
                    "completed_at": completed_at,
                    "error": error,
                    "operation_results": json.dumps(operation_results or []),
                    "product_sequence_ids": json.dumps(product_sequence_ids or {}),
                },
            )
            conn.commit()
            row = conn.execute(_SELECT_SQL, (run_id,)).fetchone()
        return _row_to_run(row) if row else None

    def list_active(self) -> list[str]:
        """Return run_ids of all pending or running runs."""
        conn = self._connect()
        with self._lock:
            rows = conn.execute(_SELECT_ACTIVE_SQL).fetchall()
        return [row[0] for row in rows]

    def mark_active_interrupted(self) -> int:
        """Mark all pending/running runs as interrupted.

        Called on gateway startup. Returns the count of interrupted runs.
        """
        active_ids = self.list_active()
        for run_id in active_ids:
            self.update_status(run_id, status=INTERRUPTED, error="gateway restart")
        return len(active_ids)

    def count(self) -> int:
        conn = self._connect()
        with self._lock:
            row = conn.execute("SELECT COUNT(*) FROM strategy_runs").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


def _row_to_run(row: sqlite3.Row | tuple) -> StrategyRun:
    """Convert a database row to a StrategyRun dataclass."""
    if isinstance(row, sqlite3.Row):
        cols = [d[0] for d in row.keys()]
        data = dict(zip(cols, row))
    else:
        cols = [
            "run_id",
            "strategy_id",
            "plan_hash",
            "approval_id",
            "target_record",
            "status",
            "operation_results",
            "product_sequence_ids",
            "created_at",
            "updated_at",
            "completed_at",
            "error",
        ]
        data = dict(zip(cols, row))

    return StrategyRun(
        run_id=data["run_id"],
        strategy_id=data["strategy_id"],
        plan_hash=data["plan_hash"],
        approval_id=data["approval_id"],
        target_record=data["target_record"],
        status=data["status"],
        operation_results=json.loads(data["operation_results"] or "[]"),
        product_sequence_ids=json.loads(data["product_sequence_ids"] or "{}"),
        created_at=data["created_at"],
        updated_at=data["updated_at"],
        completed_at=data["completed_at"],
        error=data["error"],
    )


# --- Singleton ---------------------------------------------------------------

_store: StrategyRunStore | None = None


def get_strategy_run_store() -> StrategyRunStore:
    """Return the singleton run store."""
    global _store
    if _store is None:
        _store = StrategyRunStore()
    return _store


def reset_strategy_run_store(store: StrategyRunStore | None = None) -> None:
    """Test helper: replace or clear the singleton."""
    global _store
    _store = store
