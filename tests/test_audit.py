"""Tests for the append-only audit store."""

from __future__ import annotations

import sqlite3

import pytest

from lab_copilot_gateway.audit import AuditRecord, AuditStore


@pytest.fixture
def store(tmp_path) -> AuditStore:
    """Fresh file-backed audit store per test."""
    s = AuditStore(db_path=tmp_path / "audit.db")
    yield s
    s.close()


def _make_record(action_id: str = "act-1", **overrides) -> AuditRecord:
    defaults = dict(
        action_id=action_id,
        conversation_id="conv-1",
        request_id="req-1",
        keycloak_subject="kc-subj-1",
        librechat_user_id="lc-user-1",
        mapped_elabftw_user_id="elab-user-1",
        mapped_elabftw_team_id="elab-team-1",
        provider="librechat",
        model_id="gpt-4o",
        tool_name="elabftw.read_experiment",
        tool_args_hash="sha256:abc",
        context_refs=[{"type": "experiment", "id": 42}],
        policy_decision="allow",
        approval_id=None,
        api_call_summary={"method": "GET", "path": "/experiments/42"},
        result_summary={"status": "ok", "bytes": 1024},
        error=None,
        artifact_manifest=[{"kind": "experiment", "id": 42}],
    )
    defaults.update(overrides)
    return AuditRecord(**defaults)


def test_append_persists_success_record(store: AuditStore) -> None:
    record = _make_record(action_id="success-1")

    created_at = store.append(record)

    assert created_at is not None
    assert store.count() == 1

    row = store.get("success-1")
    assert row is not None
    assert row["action_id"] == "success-1"
    assert row["tool_name"] == "elabftw.read_experiment"
    assert row["policy_decision"] == "allow"
    assert row["error_json"] is None
    assert row["created_at"] == created_at


def test_append_persists_failure_record_with_error(store: AuditStore) -> None:
    record = _make_record(
        action_id="failure-1",
        policy_decision="deny",
        error={"code": "POLICY_DENIED", "message": "tier too high for user"},
        result_summary={},
    )

    store.append(record)

    row = store.get("failure-1")
    assert row is not None
    assert row["policy_decision"] == "deny"
    assert row["error_json"] is not None
    assert "POLICY_DENIED" in row["error_json"]


def test_records_are_append_only_no_update(store: AuditStore) -> None:
    """A second append with the same action_id must fail, not overwrite."""
    store.append(_make_record(action_id="dup-1", tool_name="tool.v1"))
    with pytest.raises(sqlite3.IntegrityError):
        store.append(_make_record(action_id="dup-1", tool_name="tool.v2"))

    row = store.get("dup-1")
    assert row is not None
    assert row["tool_name"] == "tool.v1"


def test_no_delete_api_exists() -> None:
    """AuditStore exposes no delete method — verify by attribute lookup."""
    assert not hasattr(AuditStore, "delete")
    assert not hasattr(AuditStore, "remove")
    assert not hasattr(AuditStore, "update")


def test_context_refs_and_manifest_are_json_serialized(store: AuditStore) -> None:
    record = _make_record(
        action_id="json-1",
        context_refs=[{"type": "experiment", "id": 1}, {"type": "experiment", "id": 2}],
        artifact_manifest=[{"kind": "file", "path": "/tmp/x.csv"}],
    )
    store.append(record)

    import json

    row = store.get("json-1")
    assert row is not None
    assert json.loads(row["context_refs_json"]) == [
        {"type": "experiment", "id": 1},
        {"type": "experiment", "id": 2},
    ]
    assert json.loads(row["artifact_manifest_json"]) == [
        {"kind": "file", "path": "/tmp/x.csv"}
    ]


def test_empty_action_id_rejected(store: AuditStore) -> None:
    record = _make_record(action_id="")
    with pytest.raises(ValueError, match="action_id"):
        store.append(record)


def test_multiple_records_distinct_action_ids(store: AuditStore) -> None:
    for i in range(5):
        store.append(_make_record(action_id=f"act-{i}"))

    assert store.count() == 5
    assert store.get("act-3") is not None
    assert store.get("act-99") is None
