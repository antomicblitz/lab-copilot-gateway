"""Smoke tests for the gateway scaffold endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from lab_copilot_gateway import __version__
from lab_copilot_gateway.app import create_app
from lab_copilot_gateway.audit import AuditStore
from lab_copilot_gateway.policy import PolicyEngine


@pytest.fixture(autouse=True)
def _reset_audit_store() -> None:
    """Use in-memory audit store per test to avoid cross-test bleed."""
    import lab_copilot_gateway.app as appmod
    import lab_copilot_gateway.audit as auditmod
    import lab_copilot_gateway.policy as policymod

    original_audit = appmod.get_audit_store()
    store = AuditStore(db_path=":memory:")
    original_policy = policymod._default_engine
    policymod._default_engine = PolicyEngine()  # clean default, no kill switches
    # Inject into the module-level singleton before create_app runs.
    auditmod._default_store = store
    yield
    store.close()
    auditmod._default_store = original_audit
    policymod._default_engine = original_policy


def make_client() -> TestClient:
    return TestClient(create_app())


def test_health_returns_version_and_dependency_status() -> None:
    response = make_client().get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "lab-copilot-gateway"
    assert body["version"] == __version__
    assert body["status"] == "ok"
    assert body["dependencies"]["audit_db"] == "memory"
    assert body["dependencies"]["policy_engine"] == "ready"
    assert body["dependencies"]["policy_max_tier"] == 6
    assert body["dependencies"]["kill_switches"] == []
    assert body["dependencies"]["elabftw"] == "not_configured"


def test_public_config_is_deterministic_and_non_secret() -> None:
    response = make_client().get("/config/public")

    assert response.status_code == 200
    assert response.json() == {
        "service": "lab-copilot-gateway",
        "version": __version__,
        "environment": "local",
        "approvals_required_for_mutations": True,
        "hardware_execution_enabled": False,
        "autonomous_execution_enabled": False,
    }


def test_tools_registry_starts_empty() -> None:
    response = make_client().get("/tools")

    assert response.status_code == 200
    assert response.json() == {"tools": []}


def test_audit_roundtrip_via_http() -> None:
    """POST /audit persists and GET /audit/{id} reads back the record."""
    client = make_client()
    body = {
        "action_id": "http-1",
        "tool_name": "elabftw.read_experiment",
        "policy_decision": "allow",
        "result_summary": {"status": "ok"},
    }

    post = client.post("/audit", json=body)
    assert post.status_code == 200
    assert post.json()["action_id"] == "http-1"
    assert post.json()["created_at"]

    got = client.get("/audit/http-1")
    assert got.status_code == 200
    row = got.json()
    assert row["action_id"] == "http-1"
    assert row["tool_name"] == "elabftw.read_experiment"
    assert row["policy_decision"] == "allow"


def test_audit_failure_record_with_error_via_http() -> None:
    client = make_client()
    body = {
        "action_id": "http-fail-1",
        "tool_name": "wallac.run_measurement",
        "policy_decision": "deny",
        "error": {"code": "POLICY_DENIED", "message": "tier too high"},
    }

    post = client.post("/audit", json=body)
    assert post.status_code == 200

    row = client.get("/audit/http-fail-1").json()
    assert row["policy_decision"] == "deny"
    assert "POLICY_DENIED" in row["error_json"]


def test_policy_evaluate_allows_mapped_read() -> None:
    client = make_client()
    body = {
        "tool_name": "elabftw.read_experiment",
        "tier": 1,
        "user_id": "elab-user-1",
    }
    r = client.post("/policy/evaluate", json=body)
    assert r.status_code == 200
    out = r.json()
    assert out["decision"] == "allow"
    assert out["tier"] == 1


def test_policy_evaluate_denies_write_without_approval() -> None:
    client = make_client()
    body = {
        "tool_name": "elabftw.amend_experiment_after_approval",
        "tier": 4,
        "user_id": "elab-user-1",
    }
    r = client.post("/policy/evaluate", json=body)
    assert r.status_code == 200
    out = r.json()
    assert out["decision"] == "deny"
    assert out["reason"] == "approval_required"
    assert out["requires_approval"] is True


def test_policy_evaluate_denies_hardware_even_with_approval() -> None:
    client = make_client()
    body = {
        "tool_name": "wallac.run_measurement",
        "tier": 5,
        "user_id": "elab-user-1",
        "has_approval": True,
    }
    r = client.post("/policy/evaluate", json=body)
    assert r.status_code == 200
    out = r.json()
    assert out["decision"] == "deny"
    assert out["reason"] == "hardware_blocked"


def test_policy_evaluate_denies_unmapped_user() -> None:
    client = make_client()
    body = {
        "tool_name": "help.faq",
        "tier": 0,
        "user_id": None,
    }
    r = client.post("/policy/evaluate", json=body)
    assert r.status_code == 200
    out = r.json()
    assert out["decision"] == "deny"
    assert out["reason"] == "unmapped_caller"
