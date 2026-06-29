"""Smoke tests for the gateway scaffold endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from lab_copilot_gateway import __version__
from lab_copilot_gateway.app import create_app
from lab_copilot_gateway.audit import AuditStore
from lab_copilot_gateway.identity import DbIdentityMapper
from lab_copilot_gateway.policy import PolicyEngine


@pytest.fixture(autouse=True)
def _reset_audit_store() -> None:
    """Use in-memory audit store per test to avoid cross-test bleed."""
    import lab_copilot_gateway.app as appmod
    import lab_copilot_gateway.audit as auditmod
    import lab_copilot_gateway.identity as identitymod
    import lab_copilot_gateway.policy as policymod
    import lab_copilot_gateway.tools as toolsmod

    original_audit = appmod.get_audit_store()
    store = AuditStore(db_path=":memory:")
    original_policy = policymod._default_engine
    policymod._default_engine = PolicyEngine()  # clean default, no kill switches
    # Inject into the module-level singleton before create_app runs.
    auditmod._default_store = store
    identitymod._default_mapper = DbIdentityMapper(db_path=":memory:")
    original_tools = toolsmod._default_registry
    toolsmod._default_registry = None  # fall back to the curated catalog
    yield
    store.close()
    auditmod._default_store = original_audit
    policymod._default_engine = original_policy
    identitymod._default_mapper = None
    toolsmod._default_registry = original_tools


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
    # Identity mapper defaults to in-memory db backend (fail-closed).
    assert body["dependencies"]["identity_backend"] == "db"
    assert body["dependencies"]["identity_db"] == "memory"
    # C06 curated tool catalog.
    assert body["dependencies"]["tool_count"] == 13  # noqa: PLR2004 — catalog size


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


def test_tools_registry_returns_curated_catalog() -> None:
    """GET /tools returns the curated V1 catalog (C06).

    Each entry must contain the required fields and must NOT expose raw
    low-level endpoint URLs (acceptance check: registry does not surface
    callable endpoint strings).
    """
    response = make_client().get("/tools")

    assert response.status_code == 200
    tools = response.json()["tools"]
    assert len(tools) == 13  # noqa: PLR2004 — V1 catalog size is a contract

    required_fields = {
        "name",
        "tier",
        "tier_name",
        "adapter",
        "requires_approval",
        "mutability",
        "description",
    }
    forbidden_fields = {
        "url",
        "endpoint",
        "endpoint_url",
        "raw_endpoint",
        "raw_url",
        "http_method",
        "http_path",
    }
    names = set()
    for entry in tools:
        assert required_fields.issubset(entry.keys())
        # Reason: registry must never surface raw downstream endpoint URLs as
        # callable strings to the LLM.
        assert not (forbidden_fields & set(entry.keys()))
        names.add(entry["name"])

    # All 13 tools from the C06 plan must be present.
    expected_names = {
        "elabftw.read_current_experiment",
        "elabftw.draft_experiment_update",
        "elabftw.amend_my_experiment_after_approval",
        "opencloning.parse_sequence_file",
        "opencloning.manual_sequence",
        "opencloning.oligo_hybridization",
        "opencloning.simulate_assembly",
        "wallac.get_status",
        "wallac.propose_generated_protocol",
        "wallac.validate_generated_protocol",
        "wallac.prepare_submission_package",
        "bentolab.get_status",
        "bentolab.validate_pcr_profile",
    }
    assert names == expected_names


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


# --- /identity/resolve HTTP roundtrips -----------------------------------


def test_identity_resolve_unmapped_returns_mapped_false() -> None:
    """Default in-memory DB has no mappings → fail-closed mapped=false."""
    client = make_client()
    r = client.post("/identity/resolve", json={"keycloak_subject": "absent"})
    assert r.status_code == 200
    assert r.json() == {"mapped": False}


def test_identity_resolve_mapped_user_returns_team_ids() -> None:
    """Seed the in-memory mapper; resolver returns the full identity."""
    import lab_copilot_gateway.identity as identitymod

    mapper = identitymod._default_mapper
    assert isinstance(mapper, identitymod.DbIdentityMapper)
    mapper.upsert(
        keycloak_subject="kc-app-1",
        librechat_user_id="lc-app-1",
        elabftw_user_id="elab-app-1",
        elabftw_team_ids=["team-a", "team-b"],
    )

    client = make_client()
    r = client.post("/identity/resolve", json={"keycloak_subject": "kc-app-1"})
    assert r.status_code == 200
    body = r.json()
    assert body["mapped"] is True
    assert body["elabftw_user_id"] == "elab-app-1"
    assert body["elabftw_team_id"] == "team-a"
    assert body["elabftw_team_ids"] == ["team-a", "team-b"]
    assert body["keycloak_subject"] == "kc-app-1"
    assert body["librechat_user_id"] == "lc-app-1"


def test_identity_resolve_by_librechat_user_id_works() -> None:
    """Either identifier resolves the identity."""
    import lab_copilot_gateway.identity as identitymod

    mapper = identitymod._default_mapper
    assert isinstance(mapper, identitymod.DbIdentityMapper)
    mapper.upsert(
        keycloak_subject="kc-app-2",
        librechat_user_id="lc-app-2",
        elabftw_user_id="elab-app-2",
        elabftw_team_ids=["only-team"],
    )

    client = make_client()
    r = client.post("/identity/resolve", json={"librechat_user_id": "lc-app-2"})
    assert r.status_code == 200
    body = r.json()
    assert body["mapped"] is True
    assert body["elabftw_user_id"] == "elab-app-2"
    assert body["elabftw_team_ids"] == ["only-team"]
