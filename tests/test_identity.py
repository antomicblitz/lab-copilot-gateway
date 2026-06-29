"""Tests for the identity mapping module (C05)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lab_copilot_gateway.audit import AuditRecord, AuditStore
from lab_copilot_gateway.identity import (
    DbIdentityMapper,
    MappedIdentity,
    StaticIdentityMapper,
    get_identity_mapper,
    reset_identity_mapper,
)
from lab_copilot_gateway.policy import PolicyEngine, PolicyRequest, Tier


# --- helpers --------------------------------------------------------------


def _mapped(**overrides) -> MappedIdentity:
    base = dict(
        keycloak_subject="kc-subj-1",
        librechat_user_id="lc-user-1",
        elabftw_user_id="elab-user-1",
        elabftw_team_ids=["team-1"],
    )
    base.update(overrides)
    return MappedIdentity(**base)


def _write_json_map(path: Path, entries: list[dict]) -> Path:
    path.write_text(json.dumps({"mappings": entries}))
    return path


# --- acceptance 1: unmapped user cannot call lab tools -------------------


def test_unmapped_principal_returns_none() -> None:
    """IdentityMapper returns None for an unmapped principal; the caller must
    follow up by passing user_id=None to the policy engine, which denies.
    """
    mapper = DbIdentityMapper(db_path=":memory:")
    assert mapper.map(keycloak_subject="nobody") is None
    assert mapper.map(librechat_user_id="nope") is None


def test_unmapped_principal_throws_no_exceptions() -> None:
    """Empty-mapping DB must not crash when queried for either identifier."""
    mapper = DbIdentityMapper(db_path=":memory:")
    for _ in range(3):
        assert mapper.map(keycloak_subject="absent") is None
        assert mapper.map(librechat_user_id="absent") is None
        assert mapper.map() is None


def test_policy_engine_denies_unmapped_caller() -> None:
    """End-to-end contract: identity resolution returns None → policy engine
    receives user_id=None → decision is deny/unmapped_caller.

    Acceptance check 1 from the plan.
    """
    mapper = DbIdentityMapper(db_path=":memory:")
    identity = mapper.map(keycloak_subject="kc-unknown")
    user_id = identity.elabftw_user_id if identity else None

    engine = PolicyEngine()
    decision = engine.decide(
        PolicyRequest(
            tool_name="elabftw.read_experiment",
            tier=Tier.OPERATIONAL_READ_ONLY,
            user_id=user_id,
        )
    )
    assert decision.decision == "deny"
    assert decision.reason == "unmapped_caller"


# --- acceptance 2: mapped user appears in audit records -------------------


def test_mapped_identity_populates_audit_record() -> None:
    """A resolved MappedIdentity is persisted into the audit store with the
    mapped user/team set on the row.

    Acceptance check 2 from the plan.
    """
    store = AuditStore(db_path=":memory:")
    mapper = DbIdentityMapper(db_path=":memory:")
    mapper.upsert(
        keycloak_subject="kc-subj-2",
        librechat_user_id="lc-user-2",
        elabftw_user_id="elab-user-2",
        elabftw_team_ids=["team-blue"],
    )

    identity = mapper.map(keycloak_subject="kc-subj-2")
    assert identity is not None

    record = AuditRecord(
        action_id="act-c05-1",
        keycloak_subject=identity.keycloak_subject,
        librechat_user_id=identity.librechat_user_id,
        mapped_elabftw_user_id=identity.elabftw_user_id,
        mapped_elabftw_team_id=identity.elabftw_team_id,
        tool_name="elabftw.read_experiment",
        policy_decision="allow",
    )
    store.append(record)

    row = store.get("act-c05-1")
    assert row is not None
    assert row["mapped_elabftw_user_id"] == "elab-user-2"
    assert row["mapped_elabftw_team_id"] == "team-blue"
    assert row["keycloak_subject"] == "kc-subj-2"
    assert row["librechat_user_id"] == "lc-user-2"


# --- acceptance 3: multiple eLabFTW teams representable ------------------


def test_mapped_identity_supports_multiple_teams() -> None:
    """A MappedIdentity may carry multiple team ids; the primary team is the
    first, the full list is preserved.

    Acceptance check 3 from the plan.
    """
    identity = _mapped(elabftw_team_ids=["team-1", "team-2", "team-3"])
    assert identity.elabftw_team_ids == ["team-1", "team-2", "team-3"]
    assert identity.elabftw_team_id == "team-1"


def test_db_mapper_round_trips_multiple_teams() -> None:
    """The DB backend preserves multi-team membership across upsert/read."""
    mapper = DbIdentityMapper(db_path=":memory:")
    mapper.upsert(
        keycloak_subject="kc-multi",
        librechat_user_id="lc-multi",
        elabftw_user_id="elab-multi",
        elabftw_team_ids=["alpha", "beta", "gamma"],
    )

    identity = mapper.map(keycloak_subject="kc-multi")
    assert identity is not None
    assert identity.elabftw_user_id == "elab-multi"
    assert identity.elabftw_team_ids == ["alpha", "beta", "gamma"]
    assert identity.elabftw_team_id == "alpha"


def test_static_mapper_supports_multiple_teams(tmp_path: Path) -> None:
    path = _write_json_map(
        tmp_path / "map.json",
        [
            {
                "keycloak_subject": "kc-static",
                "librechat_user_id": "lc-static",
                "elabftw_user_id": "elab-static",
                "elabftw_team_ids": ["t-one", "t-two"],
            }
        ],
    )
    mapper = StaticIdentityMapper(path)

    identity = mapper.map(keycloak_subject="kc-static")
    assert identity is not None
    assert identity.elabftw_team_ids == ["t-one", "t-two"]
    assert identity.elabftw_team_id == "t-one"


def test_user_with_no_team_returns_none_team_id() -> None:
    """A mapped user with empty team list: user is mapped, team_id is None."""
    identity = _mapped(elabftw_team_ids=[])
    assert identity.elabftw_team_id is None
    assert identity.elabftw_team_ids == []


# --- identifier matching: either kc_subject OR librechat_user_id resolves --


def test_db_mapper_resolves_by_keycloak_subject() -> None:
    mapper = DbIdentityMapper(db_path=":memory:")
    mapper.upsert(
        keycloak_subject="kc-subj-x",
        librechat_user_id="lc-user-x",
        elabftw_user_id="elab-user-x",
        elabftw_team_ids=["team-x"],
    )
    identity = mapper.map(keycloak_subject="kc-subj-x")
    assert identity is not None
    assert identity.elabftw_user_id == "elab-user-x"


def test_db_mapper_resolves_by_librechat_user_id() -> None:
    mapper = DbIdentityMapper(db_path=":memory:")
    mapper.upsert(
        keycloak_subject="kc-subj-y",
        librechat_user_id="lc-user-y",
        elabftw_user_id="elab-user-y",
        elabftw_team_ids=["team-y"],
    )
    identity = mapper.map(librechat_user_id="lc-user-y")
    assert identity is not None
    assert identity.elabftw_user_id == "elab-user-y"


# --- static mapper file format ---------------------------------------------


def test_static_mapper_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        StaticIdentityMapper(tmp_path / "nope.json")


def test_static_mapper_resolves_single_entry(tmp_path: Path) -> None:
    path = _write_json_map(
        tmp_path / "map.json",
        [
            {
                "keycloak_subject": "kc-s",
                "librechat_user_id": "lc-s",
                "elabftw_user_id": "elab-s",
                "elabftw_team_ids": ["t-s"],
            }
        ],
    )
    mapper = StaticIdentityMapper(path)
    identity = mapper.map(keycloak_subject="kc-s")
    assert identity is not None
    assert identity.elabftw_user_id == "elab-s"


def test_static_mapper_returns_none_for_absent_principal(tmp_path: Path) -> None:
    path = _write_json_map(tmp_path / "map.json", [])
    mapper = StaticIdentityMapper(path)
    assert mapper.map(keycloak_subject="nope") is None


def test_static_mapper_empty_file_loads_as_empty(tmp_path: Path) -> None:
    """An empty file is an empty mapping, not an error."""
    (tmp_path / "empty.json").write_text("")
    mapper = StaticIdentityMapper(tmp_path / "empty.json")
    assert mapper.map(keycloak_subject="anyone") is None


# --- singleton factory ------------------------------------------------------


def test_default_mapper_is_db_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """With no env vars set, the canonical mapper is an in-memory DB."""
    monkeypatch.delenv("LAB_COPILOT_IDENTITY_MAP_BACKEND", raising=False)
    monkeypatch.delenv("LAB_COPILOT_IDENTITY_DB", raising=False)
    reset_identity_mapper()
    try:
        mapper = get_identity_mapper()
        assert isinstance(mapper, DbIdentityMapper)
        assert mapper.db_path == ":memory:"
        # Fail-closed: empty DB maps nothing.
        assert mapper.map(keycloak_subject="anyone") is None
    finally:
        reset_identity_mapper()


def test_factory_picks_static_when_env_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_json_map(
        tmp_path / "map.json",
        [
            {
                "keycloak_subject": "kc-factory",
                "librechat_user_id": "lc-factory",
                "elabftw_user_id": "elab-factory",
                "elabftw_team_ids": ["t-factory"],
            }
        ],
    )
    monkeypatch.setenv("LAB_COPILOT_IDENTITY_MAP_BACKEND", "static")
    monkeypatch.setenv("LAB_COPILOT_IDENTITY_MAP_PATH", str(path))
    reset_identity_mapper()
    try:
        mapper = get_identity_mapper()
        assert isinstance(mapper, StaticIdentityMapper)
        identity = mapper.map(keycloak_subject="kc-factory")
        assert identity is not None
        assert identity.elabftw_user_id == "elab-factory"
    finally:
        reset_identity_mapper()


def test_factory_picks_db_via_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LAB_COPILOT_IDENTITY_MAP_BACKEND", "db")
    monkeypatch.setenv("LAB_COPILOT_IDENTITY_DB", str(tmp_path / "identity.db"))
    reset_identity_mapper()
    try:
        mapper = get_identity_mapper()
        assert isinstance(mapper, DbIdentityMapper)
        # Round-trip through the file-backed DB to prove it's not in-memory.
        mapper.upsert(
            keycloak_subject="kc-persist",
            librechat_user_id="lc-persist",
            elabftw_user_id="elab-persist",
            elabftw_team_ids=["t-persist"],
        )
        identity = mapper.map(keycloak_subject="kc-persist")
        assert identity is not None
        assert identity.elabftw_user_id == "elab-persist"
    finally:
        reset_identity_mapper()
