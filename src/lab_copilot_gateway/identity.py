"""Identity mapping for lab copilot gateway (C05).

Maps LibreChat / Keycloak principals to eLabFTW (user, team) identities.

The policy engine (C04) refuses unauthenticated callers (``user_id is None``).
This module turns the untrusted Keycloak subject / LibreChat user id arriving
on the wire into the trusted mapped identity that the policy engine and audit
store consume.

Two backends are provided:

    * ``StaticIdentityMapper``  - reads a YAML or JSON file.  Dev only.
    * ``DbIdentityMapper``      - SQLite-backed.  Stands in for the production
                                  Postgres backend; ``LAB_COPILOT_IDENTITY_DB``
                                  points at the database file.

Both backends implement the :class:`IdentityMapper` protocol.  Selection is via
``LAB_COPILOT_IDENTITY_MAP_BACKEND`` (``static`` or ``db``).  Missing mappings
return ``None`` — callers must treat that as fail-closed.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class MappedIdentity:
    """A trusted identity produced by resolving a wire-side principal.

    ``elabftw_team_ids`` is the full set of teams the user belongs to; the
    primary team is repeated at ``elabftw_team_id`` for ergonomics and to
    match the audit record schema (which has a single ``team_id`` column).
    """

    keycloak_subject: str
    librechat_user_id: str
    elabftw_user_id: str
    elabftw_team_ids: list[str] = field(default_factory=list)

    @property
    def elabftw_team_id(self) -> str | None:
        """Primary team, or None if the user has no team assignments."""
        return self.elabftw_team_ids[0] if self.elabftw_team_ids else None


class IdentityMapper(Protocol):
    """Map an untrusted principal to a trusted eLabFTW identity.

    Returns None when no mapping exists — callers MUST deny such requests
    (the policy engine already rejects ``user_id=None``).
    """

    def map(
        self,
        *,
        keycloak_subject: str | None = None,
        librechat_user_id: str | None = None,
    ) -> MappedIdentity | None: ...


# --- Static (file-backed) mapper -----------------------------------------


class StaticIdentityMapper:
    """File-backed identity mapping for local development.

    Source file may be JSON or YAML.  YAML support requires ``pyyaml`` to be
    installed; JSON is always supported via the stdlib.

    File shape::

        mappings:
          - keycloak_subject: kc-subj-1
            librechat_user_id: lc-user-1
            elabftw_user_id: elab-user-1
            elabftw_team_ids: [team-1, team-2]
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._mappings = self._load(self.path)

    @staticmethod
    def _load(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"identity map file not found: {path}")
        text = path.read_text()
        if path.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError as exc:  # pragma: no cover - env-dependent
                raise RuntimeError(
                    "YAML identity map requires pyyaml; install it or use JSON"
                ) from exc
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text) if text.strip() else {}
        entries = data.get("mappings", []) if isinstance(data, dict) else []
        if not isinstance(entries, list):
            raise ValueError(f"identity map {path}: 'mappings' must be a list")
        return entries

    def map(
        self,
        *,
        keycloak_subject: str | None = None,
        librechat_user_id: str | None = None,
    ) -> MappedIdentity | None:
        for entry in self._mappings:
            if not isinstance(entry, dict):
                continue
            ks_match = entry.get("keycloak_subject") == keycloak_subject and (
                keycloak_subject is not None
            )
            lc_match = entry.get("librechat_user_id") == librechat_user_id and (
                librechat_user_id is not None
            )
            # A row matches if it matches EITHER identifier we were given, as
            # long as that identifier was actually provided.  A row with both
            # identifiers None would match nothing because both conditions
            # guard on the input being non-None.
            if ks_match or lc_match:
                user_id = entry.get("elabftw_user_id")
                if not user_id:
                    continue
                team_ids = list(entry.get("elabftw_team_ids") or [])
                return MappedIdentity(
                    keycloak_subject=keycloak_subject
                    or entry.get("keycloak_subject")
                    or "",
                    librechat_user_id=librechat_user_id
                    or entry.get("librechat_user_id")
                    or "",
                    elabftw_user_id=str(user_id),
                    elabftw_team_ids=[str(t) for t in team_ids],
                )
        return None


# --- Db (SQLite-backed) mapper -------------------------------------------


_DB_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS identity_mappings (
    keycloak_subject      TEXT,
    librechat_user_id     TEXT,
    elabftw_user_id       TEXT NOT NULL,
    elabftw_team_ids_json TEXT NOT NULL,
    PRIMARY KEY (keycloak_subject, librechat_user_id)
);
"""


class DbIdentityMapper:
    """SQLite-backed identity mapping.

    Stands in for the production Postgres backend.  The schema mirrors the
    upstream eLabFTW ``users`` table shape sufficiently for tests; C08/C14
    will swap in a real adapter against the live database.

    Team membership is stored as a JSON-encoded list to support multiple teams
    per user (acceptance: "Multiple eLabFTW teams can be represented").
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        env_path = os.getenv("LAB_COPILOT_IDENTITY_DB")
        self.db_path: str = str(db_path or env_path or ":memory:")
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            # check_same_thread=False because we guard with our own lock.
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.executescript(_DB_SCHEMA_SQL)
            self._conn.commit()
        return self._conn

    def upsert(
        self,
        *,
        keycloak_subject: str,
        librechat_user_id: str,
        elabftw_user_id: str,
        elabftw_team_ids: list[str],
    ) -> None:
        """Insert or replace a mapping row.  Used by tests and admin tooling."""
        conn = self._connect()
        with self._lock:
            conn.execute(
                """
                INSERT INTO identity_mappings (
                    keycloak_subject, librechat_user_id,
                    elabftw_user_id, elabftw_team_ids_json
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(keycloak_subject, librechat_user_id) DO UPDATE SET
                    elabftw_user_id       = excluded.elabftw_user_id,
                    elabftw_team_ids_json = excluded.elabftw_team_ids_json
                """,
                (
                    keycloak_subject,
                    librechat_user_id,
                    elabftw_user_id,
                    json.dumps(elabftw_team_ids, sort_keys=True),
                ),
            )
            conn.commit()

    def map(
        self,
        *,
        keycloak_subject: str | None = None,
        librechat_user_id: str | None = None,
    ) -> MappedIdentity | None:
        conn = self._connect()
        with self._lock:
            if keycloak_subject is not None:
                row = conn.execute(
                    "SELECT librechat_user_id, elabftw_user_id, elabftw_team_ids_json "
                    "FROM identity_mappings WHERE keycloak_subject = ?",
                    (keycloak_subject,),
                ).fetchone()
            elif librechat_user_id is not None:
                row = conn.execute(
                    "SELECT librechat_user_id, elabftw_user_id, elabftw_team_ids_json "
                    "FROM identity_mappings WHERE librechat_user_id = ?",
                    (librechat_user_id,),
                ).fetchone()
            else:
                row = None
        if row is None:
            return None
        lc_user_id, elab_user_id, team_ids_json = row
        team_ids: list[str] = json.loads(team_ids_json) if team_ids_json else []
        return MappedIdentity(
            keycloak_subject=keycloak_subject or "",
            librechat_user_id=lc_user_id or librechat_user_id or "",
            elabftw_user_id=elab_user_id,
            elabftw_team_ids=team_ids,
        )

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


# --- module-level singleton for dependency injection ----------------------

#: Process-wide identity mapper.  Set lazily by :func:`get_identity_mapper`.
#: Tests can override this via direct assignment (see ``reset_identity_mapper``).
_default_mapper: IdentityMapper | None = None


def get_identity_mapper() -> IdentityMapper:
    """Return the process-wide identity mapper (created lazily).

    Backend selection:

        LAB_COPILOT_IDENTITY_MAP_BACKEND=db (default)
        LAB_COPILOT_IDENTITY_DB=/path/to/identity.db  (default :memory:)

        LAB_COPILOT_IDENTITY_MAP_BACKEND=static
        LAB_COPILOT_IDENTITY_MAP_PATH=/path/to/map.json  (or .yaml)

    The default (``db`` with ``:memory:``) has no mappings: every call to
    :meth:`IdentityMapper.map` returns ``None``, callers reject — fail closed.
    """
    global _default_mapper
    if _default_mapper is None:
        _default_mapper = _construct_mapper()
    return _default_mapper


def reset_identity_mapper(mapper: IdentityMapper | None = None) -> None:
    """Test helper: replace or clear the singleton."""
    global _default_mapper
    _default_mapper = mapper


def _construct_mapper() -> IdentityMapper:
    backend = os.getenv("LAB_COPILOT_IDENTITY_MAP_BACKEND", "db").strip().lower()
    if backend == "db":
        return DbIdentityMapper()
    if backend == "static":
        path = os.getenv("LAB_COPILOT_IDENTITY_MAP_PATH")
        if not path:
            raise RuntimeError(
                "LAB_COPILOT_IDENTITY_MAP_PATH not set; cannot construct "
                "static identity mapper"
            )
        return StaticIdentityMapper(path)
    raise RuntimeError(f"unknown identity backend: {backend!r}")
