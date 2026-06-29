"""Smoke tests for the gateway scaffold endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from lab_copilot_gateway import __version__
from lab_copilot_gateway.app import create_app
from lab_copilot_gateway.audit import AuditStore


@pytest.fixture(autouse=True)
def _reset_audit_store() -> None:
    """Use in-memory audit store per test to avoid cross-test bleed."""
    import lab_copilot_gateway.app as appmod

    original = appmod.get_audit_store()
    store = AuditStore(db_path=":memory:")
    # Inject into the module-level singleton before create_app runs.
    import lab_copilot_gateway.audit as auditmod

    auditmod._default_store = store
    yield
    store.close()
    auditmod._default_store = original


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
