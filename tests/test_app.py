"""Smoke tests for the gateway scaffold endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from lab_copilot_gateway import __version__
from lab_copilot_gateway.app import create_app
from lab_copilot_gateway.approval import ApprovalStore
from lab_copilot_gateway.audit import AuditStore
from lab_copilot_gateway.elabftw import (
    ContextTokenClaims,
    ElabftwReadAdapter,
    StubElabftwClient,
    mint_context_token,
)
from lab_copilot_gateway.identity import DbIdentityMapper
from lab_copilot_gateway.policy import PolicyEngine


@pytest.fixture(autouse=True)
def _reset_audit_store(monkeypatch) -> None:
    """Use in-memory audit store per test to avoid cross-test bleed."""
    import lab_copilot_gateway.app as appmod
    import lab_copilot_gateway.approval as approvalmod
    import lab_copilot_gateway.audit as auditmod
    import lab_copilot_gateway.elabftw as elabmod
    import lab_copilot_gateway.identity as identitymod
    import lab_copilot_gateway.policy as policymod
    import lab_copilot_gateway.tools as toolsmod

    # Pin the context-token secret for HTTP-level tests that mint tokens.
    monkeypatch.setenv("LAB_COPILOT_TOKEN_SECRET", "http-test-secret")

    original_audit = appmod.get_audit_store()
    store = AuditStore(db_path=":memory:")
    approval_store = ApprovalStore(db_path=":memory:")
    original_policy = policymod._default_engine
    policymod._default_engine = PolicyEngine()  # clean default, no kill switches
    # Inject into the module-level singleton before create_app runs.
    auditmod._default_store = store
    identitymod._default_mapper = DbIdentityMapper(db_path=":memory:")
    approvalmod._default_store = approval_store
    # Wire the eLabFTW adapter with a stub client so the HTTP path is testable
    # without a live eLabFTW instance.
    stub_client = StubElabftwClient(
        seeds={
            42: {
                "id": 42,
                "title": "HTTP test experiment",
                "body": "<p>body</p>",
                "metadata": {"extra_fields": {}},
            }
        }
    )
    elabmod._default_adapter = ElabftwReadAdapter(
        policy_engine=policymod._default_engine,
        audit_store=store,
        client=stub_client,
    )
    original_tools = toolsmod._default_registry
    toolsmod._default_registry = None  # fall back to the curated catalog
    yield
    store.close()
    approval_store.close()
    auditmod._default_store = original_audit
    policymod._default_engine = original_policy
    identitymod._default_mapper = None
    approvalmod._default_store = None
    elabmod._default_adapter = None
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
    # eLabFTW adapter is wired with a stub client by the test fixture.
    assert body["dependencies"]["elabftw"] == "configured"
    # Identity mapper defaults to in-memory db backend (fail-closed).
    assert body["dependencies"]["identity_backend"] == "db"
    assert body["dependencies"]["identity_db"] == "memory"
    # C07 approval token store defaults to in-memory db backend.
    assert body["dependencies"]["approval_backend"] == "db"
    assert body["dependencies"]["approval_db"] == "memory"
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


# --- /approval/* HTTP roundtrips -----------------------------------------


_APPROVAL_REQUEST_BODY = {
    "tool_name": "elabftw.amend_my_experiment_after_approval",
    "args": {"experiment_id": 42, "amendment": "add AI note"},
    "target_record": "elabftw:experiment:42",
    "tier": 4,
    "keycloak_subject": "kc-1",
    "librechat_user_id": "lc-1",
    "mapped_elabftw_user_id": "elab-1",
    "provider": "openai",
    "model_id": "gpt-4o-mini",
    "ttl_seconds": 300,
}


def test_approval_request_returns_id_and_expiry() -> None:
    client = make_client()
    r = client.post("/approval/request", json=_APPROVAL_REQUEST_BODY)
    assert r.status_code == 200
    body = r.json()
    assert body["approval_id"]
    assert body["expires_at"]
    assert body["tool_name"] == "elabftw.amend_my_experiment_after_approval"
    assert body["args_hash"]


def test_approval_consume_roundtrip_succeeds() -> None:
    """Happy path: request → consume with same args → consumed: true."""
    client = make_client()
    req = client.post("/approval/request", json=_APPROVAL_REQUEST_BODY)
    approval_id = req.json()["approval_id"]

    consume_body = {
        "approval_id": approval_id,
        "tool_name": "elabftw.amend_my_experiment_after_approval",
        "args": {"experiment_id": 42, "amendment": "add AI note"},
        "target_record": "elabftw:experiment:42",
    }
    r = client.post("/approval/consume", json=consume_body)
    assert r.status_code == 200
    out = r.json()
    assert out["consumed"] is True
    assert out["approval_id"] == approval_id
    assert out["consumed_at"]
    assert out["tool_name"] == "elabftw.amend_my_experiment_after_approval"


def test_approval_consume_rejects_mismatched_args() -> None:
    """Acceptance check: modified args → consumed: false, reason: mismatch."""
    client = make_client()
    req = client.post("/approval/request", json=_APPROVAL_REQUEST_BODY)
    approval_id = req.json()["approval_id"]

    consume_body = {
        "approval_id": approval_id,
        "tool_name": "elabftw.amend_my_experiment_after_approval",
        "args": {"experiment_id": 999, "amendment": "tampered"},  # differs
        "target_record": "elabftw:experiment:42",
    }
    r = client.post("/approval/consume", json=consume_body)
    assert r.status_code == 200
    out = r.json()
    assert out["consumed"] is False
    assert out["reason"] == "mismatch"


def test_approval_consume_rejects_replay() -> None:
    """Acceptance check: replayed (already-consumed) token → consumed: false."""
    client = make_client()
    approval_id = client.post("/approval/request", json=_APPROVAL_REQUEST_BODY).json()[
        "approval_id"
    ]

    consume_body = {
        "approval_id": approval_id,
        "tool_name": "elabftw.amend_my_experiment_after_approval",
        "args": {"experiment_id": 42, "amendment": "add AI note"},
        "target_record": "elabftw:experiment:42",
    }
    first = client.post("/approval/consume", json=consume_body).json()
    assert first["consumed"] is True

    second = client.post("/approval/consume", json=consume_body).json()
    assert second["consumed"] is False
    assert second["reason"] == "already_consumed"


def test_approval_consume_rejects_unknown_id() -> None:
    client = make_client()
    consume_body = {
        "approval_id": "never-issued",
        "tool_name": "elabftw.amend_my_experiment_after_approval",
        "args": {"a": 1},
    }
    r = client.post("/approval/consume", json=consume_body)
    assert r.status_code == 200
    out = r.json()
    assert out["consumed"] is False
    assert out["reason"] == "not_found"


def test_approval_get_returns_record() -> None:
    client = make_client()
    approval_id = client.post("/approval/request", json=_APPROVAL_REQUEST_BODY).json()[
        "approval_id"
    ]

    r = client.get(f"/approval/{approval_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["approval_id"] == approval_id
    assert body["tool_name"] == "elabftw.amend_my_experiment_after_approval"
    assert body["consumed_at"] is None


def test_approval_get_returns_null_for_unknown() -> None:
    """Unknown approval_id returns 200 with null body (FastAPI encodes Optional
    return as JSON null).  An explicit 404 would require a response_model; the
    null body is intentional and consistent with GET /audit/{action_id}."""
    client = make_client()
    r = client.get("/approval/does-not-exist")
    assert r.status_code == 200
    assert r.json() is None


# --- /elabftw/read_current_experiment HTTP roundtrips (C08) ---------------


def _http_claims(experiment_id: int = 42) -> ContextTokenClaims:
    import datetime as _dt

    issued = _dt.datetime.now(_dt.timezone.utc)
    return ContextTokenClaims(
        experiment_id=experiment_id,
        mapped_elabftw_user_id="elab-http-1",
        keycloak_subject="kc-http-1",
        librechat_user_id="lc-http-1",
        issued_at=issued.isoformat(),
        expires_at=(issued + _dt.timedelta(seconds=600)).isoformat(),
    )


def _http_token(claims: ContextTokenClaims | None = None) -> str:
    return mint_context_token(claims or _http_claims())


def test_elabftw_read_current_experiment_succeeds() -> None:
    """Happy path: mapped user with valid token → adapter returns the
    experiment, audit record persisted."""
    import lab_copilot_gateway.identity as identitymod

    mapper = identitymod._default_mapper
    assert isinstance(mapper, identitymod.DbIdentityMapper)
    mapper.upsert(
        keycloak_subject="kc-http-1",
        librechat_user_id="lc-http-1",
        elabftw_user_id="elab-http-1",
        elabftw_team_ids=["team-http-1"],
    )

    client = make_client()
    body = {
        "context_token": _http_token(),
        "keycloak_subject": "kc-http-1",
        "conversation_id": "conv-http-1",
        "request_id": "req-http-1",
        "provider": "openai",
        "model_id": "gpt-4o-mini",
    }
    r = client.post("/elabftw/read_current_experiment", json=body)
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is True
    exp = out["experiment"]
    assert exp["experiment_id"] == 42
    assert exp["title"] == "HTTP test experiment"


def test_elabftw_read_current_experiment_unmapped_user_fails() -> None:
    """No identity mapping in DB → mapped_identity=None → ok:false."""
    client = make_client()
    body = {
        "context_token": _http_token(),
        "keycloak_subject": "absent",
    }
    r = client.post("/elabftw/read_current_experiment", json=body)
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is False
    assert out["reason"] == "unmapped_caller"


def test_elabftw_read_current_experiment_invalid_token_fails() -> None:
    """Invalid (garbage) token → ok:false with reason invalid_context_token."""
    client = make_client()
    body = {
        "context_token": "garbage.payload",
        "keycloak_subject": "absent",
    }
    r = client.post("/elabftw/read_current_experiment", json=body)
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is False
    assert out["reason"] == "invalid_context_token"
