"""Smoke tests for the gateway scaffold endpoints."""

from __future__ import annotations

import datetime as _dt

import pytest
from fastapi.testclient import TestClient

from lab_copilot_gateway import __version__
from lab_copilot_gateway.app import create_app
from lab_copilot_gateway.approval import ApprovalStore
from lab_copilot_gateway.audit import AuditStore
from lab_copilot_gateway.elabftw import (
    ContextTokenClaims,
    ElabftwReadAdapter,
    ElabftwWriteAdapter,
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

    # Set up dev auth for C14 — all tests use dev mode (verify=false).
    monkeypatch.setenv("LAB_COPILOT_KEYCLOAK_VERIFY", "false")
    monkeypatch.setenv("LAB_COPILOT_ENV", "local")
    monkeypatch.setenv("LAB_COPILOT_DEV_AUTH_SECRET", "test-dev-secret")
    import lab_copilot_gateway.auth as authmod

    authmod.reset_auth_config()  # force re-creation with new env vars

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
                "state": 1,
                "locked": 0,
                "uploads": [],
            }
        }
    )
    elabmod._default_adapter = ElabftwReadAdapter(
        policy_engine=policymod._default_engine,
        audit_store=store,
        client=stub_client,
    )
    # Wire the C09 write adapter with the same stub client and approval store
    # so the write HTTP path is testable without a live eLabFTW instance.
    elabmod._default_write_adapter = ElabftwWriteAdapter(
        policy_engine=policymod._default_engine,
        audit_store=store,
        approval_store=approval_store,
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
    elabmod._default_write_adapter = None
    toolsmod._default_registry = original_tools
    authmod.reset_auth_config()  # clear auth singleton for next test


TEST_DEV_SECRET = "test-dev-secret"


class _DevAuthClient:
    """TestClient wrapper that auto-injects dev-auth headers.

    Extracts keycloak_subject from JSON body to set X-Lab-Copilot-Dev-Sub,
    so existing tests don't need to change their request bodies.
    """

    def __init__(self, app):
        self._client = TestClient(app)

    def _headers(self, kwargs):
        headers = dict(kwargs.pop("headers", {}) or {})
        headers["X-Lab-Copilot-Dev-Auth"] = TEST_DEV_SECRET
        json_body = kwargs.get("json")
        if isinstance(json_body, dict) and "keycloak_subject" in json_body:
            ks = json_body["keycloak_subject"]
            if ks is not None:
                headers["X-Lab-Copilot-Dev-Sub"] = str(ks)
        return headers

    def get(self, url, **kwargs):
        headers = self._headers(kwargs)
        return self._client.get(url, headers=headers, **kwargs)

    def post(self, url, **kwargs):
        headers = self._headers(kwargs)
        return self._client.post(url, headers=headers, **kwargs)

    @property
    def headers(self):
        return self._client.headers


def make_client() -> _DevAuthClient:
    return _DevAuthClient(create_app())


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
    assert body["dependencies"]["tool_count"] == 14  # noqa: PLR2004 — V1 catalog size is a contract


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
    assert len(tools) == 14  # noqa: PLR2004 — V1 catalog size is a contract is a contract

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
        "opencloning.writeback_artifact",
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
    r = client.post(
        "/identity/resolve",
        json={"keycloak_subject": "kc-app-2", "librechat_user_id": "lc-app-2"},
    )
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


# --- /elabftw/draft_experiment_update and /elabftw/amend_my_experiment_after_approval
#     HTTP roundtrips (C09)


def _http_write_claims(experiment_id: int = 42) -> ContextTokenClaims:
    issued = _dt.datetime.now(_dt.timezone.utc)
    return ContextTokenClaims(
        experiment_id=experiment_id,
        mapped_elabftw_user_id="elab-http-1",
        keycloak_subject="kc-http-1",
        librechat_user_id="lc-http-1",
        issued_at=issued.isoformat(),
        expires_at=(issued + _dt.timedelta(seconds=600)).isoformat(),
    )


def _http_amend_body(
    *,
    context_token: str,
    approval_id: str,
    args: dict,
    amendment_html: str = "<p>AI note</p>",
) -> dict[str, object]:
    return {
        "context_token": context_token,
        "approval_id": approval_id,
        "approval_args": args,
        "keycloak_subject": "kc-http-1",
        "conversation_id": "conv-write",
        "request_id": "req-write",
        "provider": "openai",
        "model_id": "gpt-4o-mini",
        "amendment_html": amendment_html,
    }


def test_elabftw_amend_my_experiment_after_approval_succeeds() -> None:
    """Happy path: mapped user with valid token + approval → amend succeeds.

    Verifies:
        * POST /approval/request issues an approval_id bound to args.
        * POST /elabftw/amend_my_experiment_after_approval consumes the
          approval and writes the amendment downstream.
        * Response includes ok:true with operation, experiment_id,
          downstream_id, and audit_action_id.
    """
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
    amend_args = {"experiment_id": 42, "amendment_html": "<p>AI note</p>"}
    # 1. Request an approval token.
    approval_resp = client.post(
        "/approval/request",
        json={
            "tool_name": "elabftw.amend_my_experiment_after_approval",
            "args": amend_args,
            "target_record": "elabftw:experiment:42",
            "tier": 4,
            "keycloak_subject": "kc-http-1",
            "librechat_user_id": "lc-http-1",
            "mapped_elabftw_user_id": "elab-http-1",
            "provider": "openai",
            "model_id": "gpt-4o-mini",
            "ttl_seconds": 300,
        },
    )
    assert approval_resp.status_code == 200
    approval_id = approval_resp.json()["approval_id"]

    # 2. Amend with the approval.
    body = _http_amend_body(
        context_token=mint_context_token(_http_write_claims()),
        approval_id=approval_id,
        args=amend_args,
    )
    r = client.post("/elabftw/amend_my_experiment_after_approval", json=body)
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is True
    write = out["write"]
    assert write["operation"] == "append_amendment"
    assert write["experiment_id"] == 42
    assert write["audit_action_id"]


def test_elabftw_amend_denies_without_approval_token() -> None:
    """Approval token is required for tier-4 writes — passing a nonexistent
    approval_id → ok:false with an approval_denied reason."""
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
    body = _http_amend_body(
        context_token=mint_context_token(_http_write_claims()),
        approval_id="nonexistent-approval",
        args={"experiment_id": 42, "amendment_html": "<p>x</p>"},
    )
    r = client.post("/elabftw/amend_my_experiment_after_approval", json=body)
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is False
    assert "approval_denied" in out["reason"] or "approval_consume" in out["reason"]


def test_elabftw_amend_rejects_invalid_context_token() -> None:
    """Invalid context token → ok:false with reason invalid_context_token,
    before any approval token is consumed."""
    client = make_client()
    body = _http_amend_body(
        context_token="garbage.payload",
        approval_id="any",
        args={"experiment_id": 42, "amendment_html": "<p>x</p>"},
    )
    r = client.post("/elabftw/amend_my_experiment_after_approval", json=body)
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is False
    assert out["reason"] == "invalid_context_token"


def test_elabftw_draft_experiment_update_succeeds() -> None:
    """Happy path: draft_experiment_update creates a new experiment."""
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
    draft_args = {"title": "AI-proposed draft"}
    approval_resp = client.post(
        "/approval/request",
        json={
            "tool_name": "elabftw.draft_experiment_update",
            "args": draft_args,
            "target_record": None,
            "tier": 4,
            "keycloak_subject": "kc-http-1",
            "librechat_user_id": "lc-http-1",
            "mapped_elabftw_user_id": "elab-http-1",
            "provider": "openai",
            "model_id": "gpt-4o-mini",
            "ttl_seconds": 300,
        },
    )
    approval_id = approval_resp.json()["approval_id"]

    body = {
        "context_token": mint_context_token(_http_write_claims()),
        "approval_id": approval_id,
        "approval_args": draft_args,
        "keycloak_subject": "kc-http-1",
    }
    r = client.post("/elabftw/draft_experiment_update", json=body)
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is True
    write = out["write"]
    assert write["operation"] == "create_draft"
    assert write["audit_action_id"]


def test_elabftw_health_reports_elabftw_write_status() -> None:
    """The /health endpoint should still report `elabftw: configured` when
    the stub client is wired (the C09 adapter shares the same client)."""
    r = make_client().get("/health")
    assert r.status_code == 200
    assert r.json()["dependencies"]["elabftw"] == "configured"


# ============================================================================
# C11 — POST /elabftw/mint_context_token launcher endpoint
# ============================================================================
#
# The launcher (browser JS injected by Caddy) needs to obtain a short-lived
# context token scoped to the experiment the user is viewing.  It cannot
# read the HMAC secret (server-side only), so it calls this endpoint.
# Trust boundary in C11 is the identity mapper: the caller self-attests
# its `keycloak_subject`/`librechat_user_id` and the mapper must resolve
# to a registered eLabFTW user.  C14 wraps this with Keycloak session
# verification layered on top.


def _register_mapped_user(
    keycloak_subject: str = "kc-mint-1",
    librechat_user_id: str = "lc-mint-1",
    elabftw_user_id: str = "elab-mint-1",
) -> None:
    """Register a mapping in the test identity mapper; used by mint tests."""
    import lab_copilot_gateway.identity as identitymod

    mapper = identitymod._default_mapper
    assert isinstance(mapper, identitymod.DbIdentityMapper)
    mapper.upsert(
        keycloak_subject=keycloak_subject,
        librechat_user_id=librechat_user_id,
        elabftw_user_id=elabftw_user_id,
        elabftw_team_ids=["team-mint-1"],
    )


def test_elabftw_mint_context_token_succeeds_for_mapped_user() -> None:
    """Mapped user → 200 ok with a signed token bound to the experiment.

    The token returned must verify against the gateway secret and must
    be consumable by the C08 read endpoint (full launcher roundtrip).
    """
    from lab_copilot_gateway.elabftw import verify_context_token

    _register_mapped_user()

    client = make_client()
    r = client.post(
        "/elabftw/mint_context_token",
        json={
            "experiment_id": 42,
            "keycloak_subject": "kc-mint-1",
            "librechat_user_id": "lc-mint-1",
        },
    )
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is True
    assert out["experiment_id"] == 42
    assert out["mapped_elabftw_user_id"] == "elab-mint-1"
    assert "expires_at" in out and out["expires_at"]
    token = out["context_token"]
    assert token and "." in token

    # Token is verifiable and carries the expected claims.
    claims = verify_context_token(token)
    assert claims.experiment_id == 42
    assert claims.mapped_elabftw_user_id == "elab-mint-1"
    assert claims.keycloak_subject == "kc-mint-1"
    assert claims.librechat_user_id == "lc-mint-1"


def test_elabftw_mint_then_read_current_experiment_roundtrip() -> None:
    """End-to-end launcher flow: mint → read with the minted token.

    This is the C11 acceptance test: 'Context token is created and
    consumed by gateway.'  The token minted here must be accepted by
    the C08 read endpoint without further ceremony.
    """
    _register_mapped_user()
    stub_seed_id = 42  # seeded by the autouse fixture

    client = make_client()
    mint_r = client.post(
        "/elabftw/mint_context_token",
        json={
            "experiment_id": stub_seed_id,
            "keycloak_subject": "kc-mint-1",
        },
    )
    assert mint_r.status_code == 200
    token = mint_r.json()["context_token"]

    read_r = client.post(
        "/elabftw/read_current_experiment",
        json={
            "context_token": token,
            "keycloak_subject": "kc-mint-1",
        },
    )
    assert read_r.status_code == 200
    out = read_r.json()
    assert out["ok"] is True
    assert out["experiment"]["experiment_id"] == stub_seed_id
    assert out["experiment"]["title"] == "HTTP test experiment"


def test_elabftw_mint_denies_unmapped_caller() -> None:
    """Caller not in the identity mapper → ok:false with reason
    unmapped_caller.  This is the C11 fail-closed auth gate."""
    client = make_client()
    r = client.post(
        "/elabftw/mint_context_token",
        json={
            "experiment_id": 42,
            "keycloak_subject": "not-registered",
        },
    )
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is False
    assert out["reason"] == "unmapped_caller"


def test_elabftw_mint_denies_caller_with_no_identity_fields() -> None:
    """Neither keycloak_subject nor librechat_user_id → unmapped.
    Defends against a launcher that drops its identity headers."""
    client = make_client()
    r = client.post(
        "/elabftw/mint_context_token",
        json={"experiment_id": 42},
    )
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is False
    assert out["reason"] == "unmapped_caller"


def test_elabftw_mint_rejects_nonpositive_experiment_id() -> None:
    """experiment_id <= 0 is invalid even for a mapped user."""
    _register_mapped_user()
    client = make_client()
    r = client.post(
        "/elabftw/mint_context_token",
        json={
            "experiment_id": 0,
            "keycloak_subject": "kc-mint-1",
        },
    )
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is False
    assert out["reason"] == "invalid_experiment_id"


def test_elabftw_mint_ttl_clamped_to_max() -> None:
    """requested_ttl_seconds > _MAX_MINT_TTL_SECONDS is clamped silently.

    The launcher should never be able to mint a token that outlives the
    hard cap.  We assert the returned expires_at is at most
    _NORMAL_TTL_SECONDS + a few seconds of slack — actually capped to
    _MAX_MINT_TTL_SECONDS, so we verify the gap is bounded by that cap
    plus a small clock slack.
    """
    from lab_copilot_gateway.elabftw import _MAX_MINT_TTL_SECONDS

    _register_mapped_user()
    client = make_client()
    r = client.post(
        "/elabftw/mint_context_token",
        json={
            "experiment_id": 42,
            "keycloak_subject": "kc-mint-1",
            "requested_ttl_seconds": 3_600,  # 1 hour — over cap
        },
    )
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is True
    expires = _dt.datetime.fromisoformat(out["expires_at"])
    now = _dt.datetime.now(_dt.timezone.utc)
    delta = (expires - now).total_seconds()
    # Cap is 600s; allow 5s clock slack.
    assert delta <= _MAX_MINT_TTL_SECONDS + 5
    # And not absurdly short.
    assert delta > 60


def test_elabftw_mint_default_ttl_when_none_supplied() -> None:
    """No requested_ttl_seconds → uses _NORMAL_TTL_SECONDS (300s)."""
    from lab_copilot_gateway.elabftw import _NORMAL_TTL_SECONDS

    _register_mapped_user()
    client = make_client()
    r = client.post(
        "/elabftw/mint_context_token",
        json={
            "experiment_id": 42,
            "keycloak_subject": "kc-mint-1",
        },
    )
    assert r.status_code == 200
    out = r.json()
    expires = _dt.datetime.fromisoformat(out["expires_at"])
    now = _dt.datetime.now(_dt.timezone.utc)
    delta = (expires - now).total_seconds()
    # Default 300s; allow 5s slack.
    assert _NORMAL_TTL_SECONDS - 5 <= delta <= _NORMAL_TTL_SECONDS + 5


def test_elabftw_mint_token_binds_to_resolved_user_not_request() -> None:
    """The token carries the *resolved* eLabFTW user id, not what the
    caller claims.  A caller mapped to elab-user-A cannot obtain a token
    bound to elab-user-B by passing a different keycloak_subject — the
    mapper is the source of truth."""
    _register_mapped_user(
        keycloak_subject="kc-mint-2",
        librechat_user_id="lc-mint-2",
        elabftw_user_id="elab-mint-2",
    )

    client = make_client()
    r = client.post(
        "/elabftw/mint_context_token",
        json={
            "experiment_id": 42,
            # Correctly registered subject.
            "keycloak_subject": "kc-mint-2",
            # Caller lies about its librechat id — must not influence the
            # bound user; the mapper resolves from keycloak_subject.
            "librechat_user_id": "lc-impostor",
        },
    )
    # The mapper is keyed by both — if it doesn't match both, it returns
    # None.  Either way, the bound user is NOT lc-impostor.
    out = r.json()
    if out["ok"]:
        # If the mapper resolved by keycloak_subject alone, the bound
        # user must still be elab-mint-2, never a value derived from
        # the librechat_user_id field.
        assert out["mapped_elabftw_user_id"] == "elab-mint-2"
    else:
        assert out["reason"] == "unmapped_caller"


# --- POST /invoke (C13 — LibreChat tool surface) --------------------------
#
# /invoke is the single dispatch entry point used by the LibreChat
# custom-endpoint service.  It enforces the C06 tool registry and
# routes to the right adapter by tool name.  These tests cover:
#
#   * Acceptance #4: tool not in registry → tool_not_registered.
#   * Happy-path dispatch to elabftw.read_current_experiment.
#   * Happy-path dispatch to elabftw.amend_my_experiment_after_approval
#     (with an approval token issued via /approval/request).
#   * A registered tool whose adapter isn't implemented yet →
#     adapter_not_implemented (so LibreChat sees a structured error,
#     not a transport 404).
#   * Invalid token / unmapped caller propagation: /invoke must not
#     silently swallow adapter errors — the same reason codes the
#     per-tool endpoints emit must come back through /invoke.
#   * Registry unchanged after /invoke (dispatch is read-only on
#     the registry; no tool added or removed).


def test_invoke_rejects_tool_not_in_registry() -> None:
    """Acceptance #4: gateway rejects direct privileged calls not in
    tool registry.

    A request with a tool_name that isn't in the C06 catalog must
    return ok:false with reason tool_not_registered.  This is the
    core LibreChat-facing guardrail: an attacker who crafts an
    arbitrary tool_name (e.g. 'elabftw.delete_everything' or a raw
    privileged endpoint name) gets nothing back.
    """
    client = make_client()
    r = client.post(
        "/invoke",
        json={
            "tool_name": "elabftw.delete_everything",
            "context_token": "doesn't-matter",
            "keycloak_subject": "kc-invoke-1",
        },
    )
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is False
    assert out["tool_name"] == "elabftw.delete_everything"
    assert out["reason"] == "tool_not_registered"
    assert "not in the gateway registry" in out["message"]


def test_invoke_rejects_empty_tool_name() -> None:
    """Empty tool_name must fail closed (tool_not_registered), not
    dispatch to anything."""
    client = make_client()
    r = client.post(
        "/invoke",
        json={
            "tool_name": "",
            "context_token": "doesn't-matter",
        },
    )
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is False
    assert out["reason"] == "tool_not_registered"


def test_invoke_rejects_raw_endpoint_name_as_tool() -> None:
    """An attacker trying to pass a raw privileged endpoint path as
    tool_name (e.g. '/elabftw/experiments/42' or 'DELETE /elabftw/123')
    must hit the registry guard, not route to anything."""
    client = make_client()
    r = client.post(
        "/invoke",
        json={
            "tool_name": "DELETE /elabftw/experiments/42",
            "context_token": "doesn't-matter",
        },
    )
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is False
    assert out["reason"] == "tool_not_registered"


def test_invoke_dispatches_elabftw_read() -> None:
    """Happy path: tool 'elabftw.read_current_experiment' in registry,
    mapped user, valid token → adapter returns the experiment via the
    /invoke dispatcher (same result as POST /elabftw/read_current_experiment).

    Uses kc-http-1/lc-http-1/elab-http-1 because _http_claims() (called by
    _http_token()) binds the token to that identity — the cross-user
    binding check requires the request identity to match the token claims.
    """
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
    r = client.post(
        "/invoke",
        json={
            "tool_name": "elabftw.read_current_experiment",
            "context_token": _http_token(),
            "keycloak_subject": "kc-http-1",
            "librechat_user_id": "lc-http-1",
            "conversation_id": "conv-invoke-read",
            "request_id": "req-invoke-read",
            "provider": "openai",
            "model_id": "gpt-4o-mini",
        },
    )
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is True
    assert out["tool_name"] == "elabftw.read_current_experiment"
    result = out["result"]
    assert result["experiment_id"] == 42
    assert result["title"] == "HTTP test experiment"


def test_invoke_dispatches_elabftw_amend_with_approval() -> None:
    """Happy path: tool 'elabftw.amend_my_experiment_after_approval'
    with an approval token issued via /approval/request → /invoke
    dispatches to the write adapter and returns the amendment result.

    Uses kc-http-1/lc-http-1/elab-http-1 to match _http_write_claims()."""
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
    amend_args = {"experiment_id": 42, "amendment_html": "<p>AI note via invoke</p>"}
    # Issue the approval bound to amend_args.
    approval_resp = client.post(
        "/approval/request",
        json={
            "tool_name": "elabftw.amend_my_experiment_after_approval",
            "args": amend_args,
            "target_record": "elabftw:experiment:42",
            "tier": 4,
            "keycloak_subject": "kc-http-1",
            "librechat_user_id": "lc-http-1",
            "mapped_elabftw_user_id": "elab-http-1",
            "provider": "openai",
            "model_id": "gpt-4o-mini",
            "ttl_seconds": 300,
        },
    )
    assert approval_resp.status_code == 200
    approval_id = approval_resp.json()["approval_id"]

    r = client.post(
        "/invoke",
        json={
            "tool_name": "elabftw.amend_my_experiment_after_approval",
            "context_token": mint_context_token(_http_write_claims()),
            "approval_id": approval_id,
            "args": amend_args,
            "keycloak_subject": "kc-http-1",
            "librechat_user_id": "lc-http-1",
            "conversation_id": "conv-invoke-amend",
            "request_id": "req-invoke-amend",
            "provider": "openai",
            "model_id": "gpt-4o-mini",
        },
    )
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is True
    assert out["tool_name"] == "elabftw.amend_my_experiment_after_approval"
    write = out["result"]
    assert write["operation"] == "append_amendment"
    assert write["experiment_id"] == 42
    assert write["audit_action_id"]


def test_invoke_propagates_invalid_context_token() -> None:
    """Invalid context_token → ok:false with reason invalid_context_token
    (propagated from the underlying adapter, not swallowed)."""
    import lab_copilot_gateway.identity as identitymod

    mapper = identitymod._default_mapper
    assert isinstance(mapper, identitymod.DbIdentityMapper)
    mapper.upsert(
        keycloak_subject="kc-invoke-invalid",
        librechat_user_id="lc-invoke-invalid",
        elabftw_user_id="elab-http-1",
        elabftw_team_ids=["team-http-1"],
    )

    client = make_client()
    r = client.post(
        "/invoke",
        json={
            "tool_name": "elabftw.read_current_experiment",
            "context_token": "garbage.payload",
            "keycloak_subject": "kc-invoke-invalid",
        },
    )
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is False
    assert out["tool_name"] == "elabftw.read_current_experiment"
    assert out["reason"] == "invalid_context_token"


def test_invoke_propagates_unmapped_caller() -> None:
    """Unmapped caller identity → ok:false with reason unmapped_caller
    (same as the per-tool endpoint)."""
    client = make_client()
    r = client.post(
        "/invoke",
        json={
            "tool_name": "elabftw.read_current_experiment",
            "context_token": _http_token(),
            "keycloak_subject": "absent",
        },
    )
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is False
    assert out["tool_name"] == "elabftw.read_current_experiment"
    assert out["reason"] == "unmapped_caller"


def test_invoke_returns_adapter_not_implemented_for_future_tools() -> None:
    """Tools whose adapter is not yet implemented (wallac.*,
    bentolab.* — C18+) must return ok:false with reason
    adapter_not_implemented, not 404 or a stack trace.  LibreChat can
    surface this to the user as 'this tool is not wired up yet'."""
    client = make_client()
    r = client.post(
        "/invoke",
        json={
            "tool_name": "wallac.get_status",
            "context_token": "doesn't-matter",
            "args": {},
        },
    )
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is False
    assert out["tool_name"] == "wallac.get_status"
    assert out["reason"] == "adapter_not_implemented"


def test_invoke_does_not_modify_registry() -> None:
    """Dispatch is read-only on the registry: invoking a tool neither
    adds nor removes registry entries.  /health still reports
    tool_count: 13 and /tools still returns the same catalog."""
    client = make_client()
    before = client.get("/tools").json()["tools"]
    # Hit a couple of paths that exercise different code branches.
    client.post(
        "/invoke",
        json={
            "tool_name": "elabftw.delete_everything",
            "context_token": "x",
        },
    )
    client.post(
        "/invoke",
        json={
            "tool_name": "wallac.get_status",
            "context_token": "x",
        },
    )
    after = client.get("/tools").json()["tools"]
    assert before == after
    health = client.get("/health").json()
    assert health["dependencies"]["tool_count"] == 14  # noqa: PLR2004


def test_invoke_amend_missing_approval_id_denies() -> None:
    """amend requires approval; passing a nonexistent approval_id still
    hits the adapter's approval gate (same as the per-tool endpoint),
    returning approval_denied rather than /invoke short-circuiting.

    Uses kc-http-1/lc-http-1 to match _http_write_claims()."""
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
    r = client.post(
        "/invoke",
        json={
            "tool_name": "elabftw.amend_my_experiment_after_approval",
            "context_token": mint_context_token(_http_write_claims()),
            "approval_id": "nonexistent-approval",
            "args": {"experiment_id": 42, "amendment_html": "<p>x</p>"},
            "keycloak_subject": "kc-http-1",
            "librechat_user_id": "lc-http-1",
        },
    )
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is False
    assert out["tool_name"] == "elabftw.amend_my_experiment_after_approval"
    # Adapter emits approval_denied / approval_consume_* (mirrors the
    # per-tool endpoint's behaviour — /invoke doesn't short-circuit).
    assert "approval" in out["reason"] or "consum" in out["reason"]


def test_invoke_health_reflects_tool_count() -> None:
    """/health still exposes the tool_count surface so operators and
    the custom-endpoint service can discover whether /invoke is wired
    to a non-empty registry."""
    body = make_client().get("/health").json()
    assert body["dependencies"]["tool_count"] == 14  # noqa: PLR2004
