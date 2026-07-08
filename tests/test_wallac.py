"""Tests for the Wallac adapter (C18 + C19).

Acceptance checks (from build plan):

C18:
    1. Gateway calls Wallac health/status.
    2. Tool is read-only and allowed for mapped users.
    3. Wallac offline state is handled gracefully.

C19:
    4. Copilot can prepare generated-protocol package.
    5. Package can be validated.
    6. Submission remains blocked in v1.

Additional coverage (mirrors the OpenCloning adapter test patterns):

    * Unmapped caller denies before any downstream call.
    * Invalid/expired context token denies with audit.
    * Kill switch denies via policy engine; audit deny record preserved.
    * Client error propagates after audit record persisted.
    * Token-identity mismatch denies with audit.
    * Client-not-configured raises structured error after audit.
    * Audit records carry tool_args_hash on every path.
    * Module-level singleton get/reset behaviour.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from lab_copilot_gateway.approval import (
    ApprovalStore,
    ApprovalRequest,
    compute_args_hash,
)
from lab_copilot_gateway.audit import AuditStore
from lab_copilot_gateway.elabftw import (
    ContextTokenClaims,
    InvalidContextToken,
    StubElabftwClient,
    UnmappedCaller,
    mint_context_token,
)
from lab_copilot_gateway.identity import MappedIdentity
from lab_copilot_gateway.policy import PolicyEngine, Tier
from lab_copilot_gateway.wallac import (
    HttpWallacBridgeClient,
    HttpWallacClient,
    PolicyDenied,
    StubWallacBridgeClient,
    StubWallacClient,
    WallacAdapter,
    WallacAdapterError,
    WallacResult,
    TOOL_SUBMIT,
    _default_bridge_client_from_env,
    _default_client_from_env,
    get_wallac_adapter,
    reset_wallac_adapter,
)


SECRET = b"test-secret-do-not-use-in-prod"


@pytest.fixture(autouse=True)
def _stable_token_secret(monkeypatch) -> None:
    """Pin the HMAC secret so tokens minted in tests verify in the adapter."""
    monkeypatch.setenv("LAB_COPILOT_TOKEN_SECRET", SECRET.decode("latin-1"))


# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def audit() -> AuditStore:
    s = AuditStore(db_path=":memory:")
    yield s
    s.close()


@pytest.fixture
def approval() -> ApprovalStore:
    s = ApprovalStore(db_path=":memory:")
    yield s
    s.close()


@pytest.fixture
def policy() -> PolicyEngine:
    return PolicyEngine()  # default: max_tier=6, no kill switches


@pytest.fixture
def stub_client() -> StubWallacClient:
    return StubWallacClient()


@pytest.fixture
def stub_elabftw() -> StubElabftwClient:
    return StubElabftwClient(seeds={42: {"id": 42, "title": "test", "metadata": None}})


@pytest.fixture
def adapter(
    policy: PolicyEngine,
    audit: AuditStore,
    stub_client: StubWallacClient,
) -> WallacAdapter:
    return WallacAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=stub_client,
    )


@pytest.fixture
def stub_bridge_client() -> StubWallacBridgeClient:
    return StubWallacBridgeClient()


@pytest.fixture
def adapter_with_bridge(
    policy: PolicyEngine,
    audit: AuditStore,
    stub_client: StubWallacClient,
    stub_bridge_client: StubWallacBridgeClient,
) -> WallacAdapter:
    return WallacAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=stub_client,
        bridge_client=stub_bridge_client,
    )  # --- helpers ----------------------------------------------------------------


def _claims(
    *,
    experiment_id: int = 42,
    mapped_elabftw_user_id: str = "elab-human-1",
    keycloak_subject: str | None = "kc-human-1",
    librechat_user_id: str | None = "lc-human-1",
    ttl_seconds: int = 600,
) -> ContextTokenClaims:
    issued_at = _dt.datetime.now(_dt.timezone.utc)
    expires_at = issued_at + _dt.timedelta(seconds=ttl_seconds)
    return ContextTokenClaims(
        experiment_id=experiment_id,
        mapped_elabftw_user_id=mapped_elabftw_user_id,
        keycloak_subject=keycloak_subject,
        librechat_user_id=librechat_user_id,
        issued_at=issued_at.isoformat(),
        expires_at=expires_at.isoformat(),
    )


def _token(claims: ContextTokenClaims | None = None) -> str:
    return mint_context_token(claims or _claims())


def _identity(
    *,
    elab_user_id: str = "elab-human-1",
    team_ids: list[str] | None = None,
    kc_subject: str = "kc-human-1",
    lc_user_id: str = "lc-human-1",
) -> MappedIdentity:
    return MappedIdentity(
        keycloak_subject=kc_subject,
        librechat_user_id=lc_user_id,
        elabftw_user_id=elab_user_id,
        elabftw_team_ids=team_ids or ["team-1"],
    )


def _audit_rows(audit: AuditStore) -> list[dict]:
    """Read all audit rows as dicts for assertions."""
    import json

    conn = audit._connect()
    with audit._lock:
        rows = conn.execute(
            "SELECT * FROM audit_records ORDER BY created_at"
        ).fetchall()
        cols = [
            d[0]
            for d in conn.execute("SELECT * FROM audit_records LIMIT 0").description
        ]
    result = []
    for row in rows:
        d = dict(zip(cols, row))
        for key in list(d.keys()):
            if key.endswith("_json") and d[key]:
                d[key.removesuffix("_json")] = json.loads(d[key])
        result.append(d)
    return result


# --- get_status: happy path -------------------------------------------------


def test_get_status_succeeds(
    adapter: WallacAdapter,
    audit: AuditStore,
    stub_client: StubWallacClient,
) -> None:
    """Acceptance #1: gateway calls Wallac status endpoint."""
    result = adapter.get_status(
        context_token=_token(),
        mapped_identity=_identity(),
    )
    assert isinstance(result, WallacResult)
    assert result.tool_name == "wallac.get_status"
    assert result.result["state"] == "Idle"
    assert result.result["connected"] is True
    assert result.result["is_running"] is False
    # Stub recorded the call
    assert len(stub_client.calls) == 1
    assert stub_client.calls[0]["method"] == "get_status"
    # Audit row written
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["policy_decision"] == "allow"
    assert rows[0]["tool_name"] == "wallac.get_status"
    assert rows[0]["tool_args_hash"] is not None
    assert rows[0]["context_refs"] == [{"kind": "experiment", "id": 42}]


def test_get_status_returns_instrument_state(
    adapter: WallacAdapter,
    stub_client: StubWallacClient,
) -> None:
    """Status includes instrument state, connection, and running flag."""
    stub_client.status_response = {
        "seq": 42,
        "ts": "2026-06-30T00:00:00Z",
        "connected": True,
        "state": "Measuring assay",
        "state_code": 4004,
        "is_running": True,
        "is_error": False,
        "is_idle": False,
        "target_temperature": 37.0,
        "error": None,
    }
    result = adapter.get_status(
        context_token=_token(),
        mapped_identity=_identity(),
    )
    assert result.result["state"] == "Measuring assay"
    assert result.result["is_running"] is True
    assert result.result["target_temperature"] == 37.0


# --- get_status: Wallac offline ---------------------------------------------


def test_get_status_wallac_offline_handled_gracefully(
    audit: AuditStore,
    policy: PolicyEngine,
) -> None:
    """Acceptance #3: Wallac offline state is handled gracefully."""
    stub = StubWallacClient(error=ConnectionError("connection refused"))
    adapter = WallacAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=stub,
    )
    with pytest.raises(WallacAdapterError) as exc_info:
        adapter.get_status(
            context_token=_token(),
            mapped_identity=_identity(),
        )
    assert exc_info.value.reason == "client_error"
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["policy_decision"] == "allow"  # policy allowed, client failed
    assert rows[0]["error"]["code"] == "CLIENT_ERROR"
    assert rows[0]["error"]["exception"] == "ConnectionError"


# --- unmapped caller --------------------------------------------------------


def test_unmapped_caller_denies_before_http(
    adapter: WallacAdapter,
    audit: AuditStore,
    stub_client: StubWallacClient,
) -> None:
    """Unmapped caller denies before any downstream call."""
    with pytest.raises(UnmappedCaller):
        adapter.get_status(
            context_token=_token(),
            mapped_identity=None,
        )
    assert len(stub_client.calls) == 0
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["policy_decision"] == "deny"
    assert rows[0]["error"]["code"] == "UNMAPPED_CALLER"
    assert rows[0]["tool_args_hash"] is not None


# --- invalid context token --------------------------------------------------


def test_missing_context_token_denies(
    adapter: WallacAdapter,
    audit: AuditStore,
) -> None:
    """Missing context token denies with audit."""
    with pytest.raises(InvalidContextToken, match="missing"):
        adapter.get_status(
            context_token="",
            mapped_identity=_identity(),
        )
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["error"]["code"] == "MISSING_CONTEXT_TOKEN"


def test_invalid_context_token_denies(
    adapter: WallacAdapter,
    audit: AuditStore,
) -> None:
    """Malformed context token denies with audit."""
    with pytest.raises(InvalidContextToken):
        adapter.get_status(
            context_token="garbage.payload",
            mapped_identity=_identity(),
        )
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["error"]["code"] == "INVALID_CONTEXT_TOKEN"


def test_expired_context_token_denies(
    adapter: WallacAdapter,
    audit: AuditStore,
) -> None:
    """Expired context token denies with audit."""
    expired_claims = _claims(ttl_seconds=-10)
    with pytest.raises(InvalidContextToken, match="expired"):
        adapter.get_status(
            context_token=_token(expired_claims),
            mapped_identity=_identity(),
        )
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert "expired" in rows[0]["error"]["detail"]


# --- token-identity mismatch ------------------------------------------------


def test_token_user_mismatch_denies(
    adapter: WallacAdapter,
    audit: AuditStore,
    stub_client: StubWallacClient,
) -> None:
    """Token bound to user A cannot be used by user B."""
    with pytest.raises(InvalidContextToken, match="user does not match"):
        adapter.get_status(
            context_token=_token(),
            mapped_identity=_identity(elab_user_id="different-user"),
        )
    assert len(stub_client.calls) == 0
    rows = _audit_rows(audit)
    assert rows[0]["error"]["code"] == "CONTEXT_TOKEN_USER_MISMATCH"


# --- policy deny ------------------------------------------------------------


def test_kill_switch_denies(
    audit: AuditStore,
    stub_client: StubWallacClient,
) -> None:
    """Kill switch denies via policy engine; audit deny record preserved."""
    policy = PolicyEngine(kill_switches=("wallac.*",))
    adapter = WallacAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=stub_client,
    )
    with pytest.raises(PolicyDenied):
        adapter.get_status(
            context_token=_token(),
            mapped_identity=_identity(),
        )
    assert len(stub_client.calls) == 0
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["policy_decision"] == "deny"
    assert "kill_switch" in rows[0]["error"]["reason"]


# --- client not configured --------------------------------------------------


def test_client_not_configured_raises(
    audit: AuditStore,
    policy: PolicyEngine,
) -> None:
    """Client=None raises structured error after audit."""
    adapter = WallacAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=None,
    )
    with pytest.raises(WallacAdapterError) as exc_info:
        adapter.get_status(
            context_token=_token(),
            mapped_identity=_identity(),
        )
    assert exc_info.value.reason == "wallac_not_configured"
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["error"]["code"] == "WALLAC_NOT_CONFIGURED"


# --- audit hash on all paths ------------------------------------------------


def test_audit_hash_present_on_success(
    adapter: WallacAdapter,
    audit: AuditStore,
) -> None:
    """tool_args_hash is present on success path."""
    adapter.get_status(
        context_token=_token(),
        mapped_identity=_identity(),
    )
    rows = _audit_rows(audit)
    assert rows[0]["tool_args_hash"] is not None
    assert len(rows[0]["tool_args_hash"]) == 64  # SHA-256 hex


def test_audit_hash_present_on_deny(
    adapter: WallacAdapter,
    audit: AuditStore,
) -> None:
    """tool_args_hash is present on deny path."""
    with pytest.raises(UnmappedCaller):
        adapter.get_status(
            context_token=_token(),
            mapped_identity=None,
        )
    rows = _audit_rows(audit)
    assert rows[0]["tool_args_hash"] is not None


# --- audit context_refs -----------------------------------------------------


def test_audit_context_refs_include_experiment_id(
    adapter: WallacAdapter,
    audit: AuditStore,
) -> None:
    """context_refs include the experiment_id from the token."""
    adapter.get_status(
        context_token=_token(_claims(experiment_id=99)),
        mapped_identity=_identity(),
    )
    rows = _audit_rows(audit)
    assert rows[0]["context_refs"] == [{"kind": "experiment", "id": 99}]


# --- api_call_summary -------------------------------------------------------


def test_audit_api_call_summary_on_success(
    adapter: WallacAdapter,
    audit: AuditStore,
) -> None:
    """api_call_summary is populated on success."""
    adapter.get_status(
        context_token=_token(),
        mapped_identity=_identity(),
    )
    rows = _audit_rows(audit)
    assert rows[0]["api_call_summary"]["method"] == "GET"
    assert rows[0]["api_call_summary"]["path"] == "/status"


# --- singleton --------------------------------------------------------------


def test_singleton_get_and_reset() -> None:
    """get_wallac_adapter returns a singleton; reset clears it."""
    reset_wallac_adapter()
    a1 = get_wallac_adapter()
    a2 = get_wallac_adapter()
    assert a1 is a2
    reset_wallac_adapter()
    a3 = get_wallac_adapter()
    assert a3 is not a1


# --- _default_client_from_env -----------------------------------------------


def test_default_client_from_env_returns_none_when_unset(monkeypatch) -> None:
    """No env vars → None (adapter will refuse calls)."""
    monkeypatch.delenv("LAB_COPILOT_WALLAC_BASE_URL", raising=False)
    assert _default_client_from_env() is None


def test_default_client_from_env_returns_client_when_set(monkeypatch) -> None:
    """Env vars set → HttpWallacClient."""
    monkeypatch.setenv("LAB_COPILOT_WALLAC_BASE_URL", "http://192.168.122.203:8420")
    monkeypatch.setenv("LAB_COPILOT_WALLAC_BEARER_TOKEN", "secret-token")
    client = _default_client_from_env()
    assert client is not None
    assert isinstance(client, HttpWallacClient)
    assert client.base_url == "http://192.168.122.203:8420"
    assert client.bearer_token == "secret-token"


# --- HttpWallacClient construction ------------------------------------------


def test_http_client_requires_base_url() -> None:
    """HttpWallacClient rejects empty base_url."""
    with pytest.raises(ValueError, match="non-empty base_url"):
        HttpWallacClient(base_url="")


# --- StubWallacClient default response --------------------------------------


def test_stub_client_default_response() -> None:
    """Stub returns default idle status when no pre-seeded data."""
    stub = StubWallacClient()
    result = stub.get_status()
    assert result["state"] == "Idle"
    assert result["connected"] is True
    assert result["is_idle"] is True
    assert len(stub.calls) == 1


# ============================================================================
# C19: propose_generated_protocol
# ============================================================================


def test_propose_happy_path(adapter: WallacAdapter) -> None:
    """Copilot can prepare a generated-protocol package (acceptance #4)."""
    spec = {
        "execution_mode": "generated_protocol",
        "method": {"name": "fluorescence", "wavelength": 485},
        "layout": {"rows": 8, "cols": 12},
    }
    result = adapter.propose_generated_protocol(
        context_token=_token(),
        mapped_identity=_identity(),
        protocol_spec=spec,
    )
    assert result.tool_name == "wallac.propose_generated_protocol"
    proposal = result.result["proposal"]
    assert proposal == spec
    assert "spec_hash" in result.result
    assert len(result.result["spec_hash"]) == 64  # SHA-256 hex
    assert result.result["execution_blocked"] is True


def test_propose_execution_blocked_in_v1(adapter: WallacAdapter) -> None:
    """Submission remains blocked in v1 (acceptance #6)."""
    result = adapter.propose_generated_protocol(
        context_token=_token(),
        mapped_identity=_identity(),
        protocol_spec={"execution_mode": "generated_protocol"},
    )
    assert result.result["execution_blocked"] is True
    assert "blocked" in result.result["message"].lower()


def test_propose_unmapped_caller_denies(adapter: WallacAdapter) -> None:
    """Unmapped caller denies before any computation."""
    with pytest.raises(UnmappedCaller):
        adapter.propose_generated_protocol(
            context_token=_token(),
            mapped_identity=None,
            protocol_spec={"execution_mode": "generated_protocol"},
        )
    rows = _audit_rows(adapter.audit_store)
    assert len(rows) == 1
    assert rows[0]["policy_decision"] == "deny"
    assert rows[0]["error"]["code"] == "UNMAPPED_CALLER"


def test_propose_invalid_token_denies(adapter: WallacAdapter) -> None:
    """Invalid token denies with audit."""
    with pytest.raises(InvalidContextToken):
        adapter.propose_generated_protocol(
            context_token="garbage",
            mapped_identity=_identity(),
            protocol_spec={"execution_mode": "generated_protocol"},
        )
    rows = _audit_rows(adapter.audit_store)
    assert len(rows) == 1
    assert rows[0]["error"]["code"] == "INVALID_CONTEXT_TOKEN"


def test_propose_kill_switch_denies(
    policy: PolicyEngine, audit: AuditStore, stub_client: StubWallacClient
) -> None:
    """Kill switch denies wallac.propose_generated_protocol."""
    p = PolicyEngine(kill_switches=("wallac.propose_*",))
    adapter = WallacAdapter(policy_engine=p, audit_store=audit, client=stub_client)
    with pytest.raises(PolicyDenied):
        adapter.propose_generated_protocol(
            context_token=_token(),
            mapped_identity=_identity(),
            protocol_spec={"execution_mode": "generated_protocol"},
        )
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert "kill_switch" in rows[0]["error"]["reason"]


def test_propose_audit_hash_persisted(adapter: WallacAdapter) -> None:
    """Audit records carry tool_args_hash on success path."""
    spec = {"execution_mode": "generated_protocol", "method": {"name": "absorbance"}}
    adapter.propose_generated_protocol(
        context_token=_token(),
        mapped_identity=_identity(),
        protocol_spec=spec,
    )
    rows = _audit_rows(adapter.audit_store)
    assert len(rows) == 1
    assert rows[0]["tool_args_hash"] is not None
    assert len(rows[0]["tool_args_hash"]) == 64
    assert rows[0]["tool_name"] == "wallac.propose_generated_protocol"
    assert rows[0]["policy_decision"] == "allow"


def test_propose_token_user_mismatch_denies(adapter: WallacAdapter) -> None:
    """Token-identity mismatch denies with audit."""
    claims = _claims(mapped_elabftw_user_id="elab-other-user")
    with pytest.raises(InvalidContextToken):
        adapter.propose_generated_protocol(
            context_token=_token(claims),
            mapped_identity=_identity(elab_user_id="elab-human-1"),
            protocol_spec={"execution_mode": "generated_protocol"},
        )
    rows = _audit_rows(adapter.audit_store)
    assert len(rows) == 1
    assert rows[0]["error"]["code"] == "CONTEXT_TOKEN_USER_MISMATCH"


def test_propose_empty_spec(adapter: WallacAdapter) -> None:
    """Empty spec is still processed (gateway-side validation, not schema enforcement)."""
    result = adapter.propose_generated_protocol(
        context_token=_token(),
        mapped_identity=_identity(),
        protocol_spec={},
    )
    assert result.result["proposal"] == {}
    assert result.result["execution_blocked"] is True


def test_propose_spec_hash_is_deterministic(adapter: WallacAdapter) -> None:
    """Same spec produces same hash."""
    spec = {"execution_mode": "generated_protocol", "method": {"name": "fluorescence"}}
    r1 = adapter.propose_generated_protocol(
        context_token=_token(),
        mapped_identity=_identity(),
        protocol_spec=spec,
    )
    r2 = adapter.propose_generated_protocol(
        context_token=_token(),
        mapped_identity=_identity(),
        protocol_spec=spec,
    )
    assert r1.result["spec_hash"] == r2.result["spec_hash"]


def test_propose_different_specs_produce_different_hashes(
    adapter: WallacAdapter,
) -> None:
    """Different specs produce different hashes."""
    r1 = adapter.propose_generated_protocol(
        context_token=_token(),
        mapped_identity=_identity(),
        protocol_spec={"method": {"name": "absorbance"}},
    )
    r2 = adapter.propose_generated_protocol(
        context_token=_token(),
        mapped_identity=_identity(),
        protocol_spec={"method": {"name": "fluorescence"}},
    )
    assert r1.result["spec_hash"] != r2.result["spec_hash"]


# ============================================================================
# C19: validate_generated_protocol
# ============================================================================


def test_validate_happy_path(adapter_with_bridge: WallacAdapter) -> None:
    """Package can be validated (acceptance #5)."""
    result = adapter_with_bridge.validate_generated_protocol(
        context_token=_token(),
        mapped_identity=_identity(),
        job_item_id=99,
    )
    assert result.tool_name == "wallac.validate_generated_protocol"
    assert result.result["valid"] is True
    assert len(result.result["checks"]) == 3
    assert result.result["errors"] == []


def test_validate_bridge_not_configured(adapter: WallacAdapter) -> None:
    """Bridge not configured → structured error."""
    with pytest.raises(WallacAdapterError) as exc_info:
        adapter.validate_generated_protocol(
            context_token=_token(),
            mapped_identity=_identity(),
            job_item_id=99,
        )
    assert exc_info.value.reason == "wallac_bridge_not_configured"
    rows = _audit_rows(adapter.audit_store)
    assert len(rows) == 1
    assert rows[0]["error"]["code"] == "WALLAC_BRIDGE_NOT_CONFIGURED"


def test_validate_bridge_error_propagates(
    policy: PolicyEngine, audit: AuditStore, stub_client: StubWallacClient
) -> None:
    """Bridge client error propagates after audit."""
    bridge = StubWallacBridgeClient(error=ConnectionError("bridge offline"))
    adapter = WallacAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=stub_client,
        bridge_client=bridge,
    )
    with pytest.raises(WallacAdapterError) as exc_info:
        adapter.validate_generated_protocol(
            context_token=_token(),
            mapped_identity=_identity(),
            job_item_id=99,
        )
    assert exc_info.value.reason == "client_error"
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["error"]["code"] == "CLIENT_ERROR"
    assert rows[0]["policy_decision"] == "allow"


def test_validate_invalid_token_denies(adapter_with_bridge: WallacAdapter) -> None:
    """Invalid token denies before bridge call."""
    with pytest.raises(InvalidContextToken):
        adapter_with_bridge.validate_generated_protocol(
            context_token="bad",
            mapped_identity=_identity(),
            job_item_id=99,
        )
    rows = _audit_rows(adapter_with_bridge.audit_store)
    assert len(rows) == 1
    assert rows[0]["error"]["code"] == "INVALID_CONTEXT_TOKEN"


def test_validate_unmapped_caller_denies(adapter_with_bridge: WallacAdapter) -> None:
    """Unmapped caller denies before bridge call."""
    with pytest.raises(UnmappedCaller):
        adapter_with_bridge.validate_generated_protocol(
            context_token=_token(),
            mapped_identity=None,
            job_item_id=99,
        )
    rows = _audit_rows(adapter_with_bridge.audit_store)
    assert len(rows) == 1
    assert rows[0]["error"]["code"] == "UNMAPPED_CALLER"


def test_validate_kill_switch_denies(
    policy: PolicyEngine,
    audit: AuditStore,
    stub_client: StubWallacClient,
    stub_bridge_client: StubWallacBridgeClient,
) -> None:
    """Kill switch denies wallac.validate_generated_protocol."""
    p = PolicyEngine(kill_switches=("wallac.validate_*",))
    adapter = WallacAdapter(
        policy_engine=p,
        audit_store=audit,
        client=stub_client,
        bridge_client=stub_bridge_client,
    )
    with pytest.raises(PolicyDenied):
        adapter.validate_generated_protocol(
            context_token=_token(),
            mapped_identity=_identity(),
            job_item_id=99,
        )
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert "kill_switch" in rows[0]["error"]["reason"]


def test_validate_audit_hash_persisted(adapter_with_bridge: WallacAdapter) -> None:
    """Audit records carry tool_args_hash on success path."""
    adapter_with_bridge.validate_generated_protocol(
        context_token=_token(),
        mapped_identity=_identity(),
        job_item_id=42,
    )
    rows = _audit_rows(adapter_with_bridge.audit_store)
    assert len(rows) == 1
    assert rows[0]["tool_args_hash"] is not None
    assert len(rows[0]["tool_args_hash"]) == 64
    assert rows[0]["tool_name"] == "wallac.validate_generated_protocol"
    assert rows[0]["policy_decision"] == "allow"


def test_validate_bridge_called_with_correct_job_id(
    adapter_with_bridge: WallacAdapter, stub_bridge_client: StubWallacBridgeClient
) -> None:
    """Bridge client receives the correct job_item_id."""
    adapter_with_bridge.validate_generated_protocol(
        context_token=_token(),
        mapped_identity=_identity(),
        job_item_id=123,
    )
    assert len(stub_bridge_client.calls) == 1
    assert stub_bridge_client.calls[0]["job_item_id"] == 123


def test_validate_invalid_report_propagates(
    policy: PolicyEngine, audit: AuditStore, stub_client: StubWallacClient
) -> None:
    """Bridge returns invalid validation report — it propagates as a result."""
    bridge = StubWallacBridgeClient(
        validation_response={"valid": False, "errors": ["hash mismatch"]}
    )
    adapter = WallacAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=stub_client,
        bridge_client=bridge,
    )
    result = adapter.validate_generated_protocol(
        context_token=_token(),
        mapped_identity=_identity(),
        job_item_id=99,
    )
    assert result.result["valid"] is False
    assert "hash mismatch" in result.result["errors"]


# ============================================================================
# C19: prepare_submission_package
# ============================================================================


def test_prepare_submission_happy_path(adapter: WallacAdapter) -> None:
    """Prepare an approval-ready submission package."""
    spec = {"execution_mode": "generated_protocol", "method": {"name": "absorbance"}}
    validation = {"valid": True, "checks": [{"name": "signature", "passed": True}]}
    result = adapter.prepare_submission_package(
        context_token=_token(),
        mapped_identity=_identity(),
        protocol_spec=spec,
        validation_report=validation,
    )
    assert result.tool_name == "wallac.prepare_submission_package"
    pkg = result.result["submission_package"]
    assert pkg["spec"] == spec
    assert pkg["validation"] == validation
    assert pkg["execution_blocked"] is True
    assert "spec_hash" in pkg
    assert "v1_notice" in pkg


def test_prepare_submission_execution_blocked(adapter: WallacAdapter) -> None:
    """Submission package explicitly states execution_blocked (acceptance #6)."""
    result = adapter.prepare_submission_package(
        context_token=_token(),
        mapped_identity=_identity(),
        protocol_spec={"execution_mode": "generated_protocol"},
    )
    assert result.result["submission_package"]["execution_blocked"] is True
    assert "v1.1" in result.result["submission_package"]["v1_notice"]


def test_prepare_submission_without_validation(adapter: WallacAdapter) -> None:
    """Submission package without validation report marks it not_validated."""
    result = adapter.prepare_submission_package(
        context_token=_token(),
        mapped_identity=_identity(),
        protocol_spec={"execution_mode": "generated_protocol"},
    )
    pkg = result.result["submission_package"]
    assert pkg["validation"]["valid"] is False
    assert pkg["validation"]["reason"] == "not_validated"


def test_prepare_submission_unmapped_caller_denies(adapter: WallacAdapter) -> None:
    """Unmapped caller denies before computation."""
    with pytest.raises(UnmappedCaller):
        adapter.prepare_submission_package(
            context_token=_token(),
            mapped_identity=None,
            protocol_spec={"execution_mode": "generated_protocol"},
        )
    rows = _audit_rows(adapter.audit_store)
    assert len(rows) == 1
    assert rows[0]["error"]["code"] == "UNMAPPED_CALLER"


def test_prepare_submission_invalid_token_denies(adapter: WallacAdapter) -> None:
    """Invalid token denies with audit."""
    with pytest.raises(InvalidContextToken):
        adapter.prepare_submission_package(
            context_token="garbage",
            mapped_identity=_identity(),
            protocol_spec={"execution_mode": "generated_protocol"},
        )
    rows = _audit_rows(adapter.audit_store)
    assert len(rows) == 1
    assert rows[0]["error"]["code"] == "INVALID_CONTEXT_TOKEN"


def test_prepare_submission_kill_switch_denies(
    policy: PolicyEngine, audit: AuditStore, stub_client: StubWallacClient
) -> None:
    """Kill switch denies wallac.prepare_submission_package."""
    p = PolicyEngine(kill_switches=("wallac.prepare_*",))
    adapter = WallacAdapter(policy_engine=p, audit_store=audit, client=stub_client)
    with pytest.raises(PolicyDenied):
        adapter.prepare_submission_package(
            context_token=_token(),
            mapped_identity=_identity(),
            protocol_spec={"execution_mode": "generated_protocol"},
        )
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert "kill_switch" in rows[0]["error"]["reason"]


def test_prepare_submission_audit_hash_persisted(adapter: WallacAdapter) -> None:
    """Audit records carry tool_args_hash on success path."""
    adapter.prepare_submission_package(
        context_token=_token(),
        mapped_identity=_identity(),
        protocol_spec={"execution_mode": "generated_protocol"},
    )
    rows = _audit_rows(adapter.audit_store)
    assert len(rows) == 1
    assert rows[0]["tool_args_hash"] is not None
    assert len(rows[0]["tool_args_hash"]) == 64
    assert rows[0]["tool_name"] == "wallac.prepare_submission_package"
    assert rows[0]["policy_decision"] == "allow"


def test_prepare_submission_spec_hash_deterministic(adapter: WallacAdapter) -> None:
    """Same spec produces same hash in submission package."""
    spec = {"execution_mode": "generated_protocol", "method": {"name": "absorbance"}}
    r1 = adapter.prepare_submission_package(
        context_token=_token(),
        mapped_identity=_identity(),
        protocol_spec=spec,
    )
    r2 = adapter.prepare_submission_package(
        context_token=_token(),
        mapped_identity=_identity(),
        protocol_spec=spec,
    )
    assert (
        r1.result["submission_package"]["spec_hash"]
        == r2.result["submission_package"]["spec_hash"]
    )


# ============================================================================
# C19: bridge client construction
# ============================================================================


def test_default_bridge_client_from_env_returns_none_when_unset(monkeypatch) -> None:
    """No env vars → None (adapter will refuse validation calls)."""
    monkeypatch.delenv("LAB_COPILOT_WALLAC_BRIDGE_URL", raising=False)
    assert _default_bridge_client_from_env() is None


def test_default_bridge_client_from_env_returns_client_when_set(monkeypatch) -> None:
    """Env vars set → HttpWallacBridgeClient."""
    monkeypatch.setenv("LAB_COPILOT_WALLAC_BRIDGE_URL", "http://bridge.local:8080")
    monkeypatch.setenv("LAB_COPILOT_WALLAC_BRIDGE_TOKEN", "bridge-secret")
    client = _default_bridge_client_from_env()
    assert client is not None
    assert isinstance(client, HttpWallacBridgeClient)
    assert client.base_url == "http://bridge.local:8080"
    assert client.bearer_token == "bridge-secret"


def test_http_bridge_client_requires_base_url() -> None:
    """HttpWallacBridgeClient rejects empty base_url."""
    with pytest.raises(ValueError, match="non-empty base_url"):
        HttpWallacBridgeClient(base_url="")


def test_stub_bridge_client_default_response() -> None:
    """Stub bridge returns default valid response."""
    stub = StubWallacBridgeClient()
    result = stub.validate_protocol(job_item_id=1)
    assert result["valid"] is True
    assert len(stub.calls) == 1
    health = stub.get_health()
    assert health["status"] == "ok"


# ============================================================================
# C19: singleton with bridge client
# ============================================================================


def test_singleton_wires_bridge_client(monkeypatch) -> None:
    """get_wallac_adapter() wires bridge_client from env."""
    reset_wallac_adapter()
    monkeypatch.setenv("LAB_COPILOT_WALLAC_BRIDGE_URL", "http://bridge.local:8080")
    monkeypatch.setenv("LAB_COPILOT_WALLAC_BRIDGE_TOKEN", "tok")
    try:
        adapter = get_wallac_adapter()
        assert adapter.bridge_client is not None
        assert isinstance(adapter.bridge_client, HttpWallacBridgeClient)
    finally:
        reset_wallac_adapter()


def test_singleton_bridge_client_none_when_unset(monkeypatch) -> None:
    """get_wallac_adapter() leaves bridge_client None when env unset."""
    reset_wallac_adapter()
    monkeypatch.delenv("LAB_COPILOT_WALLAC_BRIDGE_URL", raising=False)
    try:
        adapter = get_wallac_adapter()
        assert adapter.bridge_client is None
    finally:
        reset_wallac_adapter()


# ============================================================================
# C21: submit_generated_protocol
# ============================================================================


def _approval_token(
    approval_store: ApprovalStore,
    *,
    tool_name: str = TOOL_SUBMIT,
    args: dict | None = None,
    target_record: str | None = None,
    tier: int = 5,
) -> str:
    """Helper: issue an approval token for submit tests."""
    req = ApprovalRequest(
        tool_name=tool_name,
        args_hash=compute_args_hash(args or {"job_item_id": 99}),
        target_record=target_record or "wallac:job:99",
        tier=tier,
    )
    approval_id, _ = approval_store.request(req)
    return approval_id


# --- policy blocks in V1 ---------------------------------------------------


def test_submit_hardware_blocked_in_v1(
    policy: PolicyEngine,
    audit: AuditStore,
    stub_client: StubWallacClient,
    stub_bridge_client: StubWallacBridgeClient,
    approval: ApprovalStore,
) -> None:
    """Default policy engine blocks hardware execution (V1 acceptance)."""
    adapter = WallacAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=stub_client,
        bridge_client=stub_bridge_client,
        approval_store=approval,
    )
    approval_id = _approval_token(approval)
    with pytest.raises(PolicyDenied) as exc_info:
        adapter.submit_generated_protocol(
            context_token=_token(),
            mapped_identity=_identity(),
            approval_id=approval_id,
            approval_args={"job_item_id": 99},
            job_item_id=99,
        )
    assert "hardware_blocked" in exc_info.value.reason
    # Audit records the deny
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["policy_decision"] == "deny"
    assert rows[0]["error"]["reason"] == "hardware_blocked"
    # No bridge call was made
    assert len(stub_bridge_client.calls) == 0


# --- approval gated happy path (with policy override) ----------------------


def test_submit_happy_path(
    policy: PolicyEngine,
    audit: AuditStore,
    stub_client: StubWallacClient,
    stub_bridge_client: StubWallacBridgeClient,
    approval: ApprovalStore,
) -> None:
    """Approved generated-protocol run executes (acceptance check)."""
    # Allow tier 5 by raising block_hardware_above past HARDWARE_EXECUTION.
    policy = PolicyEngine(
        max_tier=Tier.CLOSED_LOOP_AUTONOMY,
        block_hardware_above=Tier.CLOSED_LOOP_AUTONOMY,
    )

    adapter = WallacAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=stub_client,
        bridge_client=stub_bridge_client,
        approval_store=approval,
    )
    approval_id = _approval_token(approval)

    result = adapter.submit_generated_protocol(
        context_token=_token(),
        mapped_identity=_identity(),
        approval_id=approval_id,
        approval_args={"job_item_id": 99},
        job_item_id=99,
    )

    assert isinstance(result, WallacResult)
    assert result.tool_name == TOOL_SUBMIT
    assert result.result["job_id"] == 1
    assert result.result["status"] == "submitted"
    assert result.result["bridge_audit_action_id"] == "bridge-audit-1"
    assert result.audit_action_id is not None
    assert result.audit_action_id.startswith("wallac-submit-")

    # Bridge was called
    assert len(stub_bridge_client.calls) == 1
    assert stub_bridge_client.calls[0]["method"] == "submit_job"
    assert stub_bridge_client.calls[0]["job_item_id"] == 99

    # Approval was consumed
    record = approval.get(approval_id)
    assert record is not None
    assert record.is_consumed()

    # Audit row written
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["policy_decision"] == "allow"
    assert rows[0]["tool_name"] == TOOL_SUBMIT
    assert rows[0]["tool_args_hash"] is not None
    assert rows[0]["result_summary"]["job_id"] == 1
    assert rows[0]["result_summary"]["status"] == "submitted"


def test_submit_elabftw_provenance_written(
    policy: PolicyEngine,
    audit: AuditStore,
    stub_client: StubWallacClient,
    stub_bridge_client: StubWallacBridgeClient,
    approval: ApprovalStore,
    stub_elabftw: StubElabftwClient,
) -> None:
    """Results write back to eLabFTW (provenance)."""
    policy = PolicyEngine(
        max_tier=Tier.CLOSED_LOOP_AUTONOMY,
        block_hardware_above=Tier.CLOSED_LOOP_AUTONOMY,
    )

    adapter = WallacAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=stub_client,
        bridge_client=stub_bridge_client,
        approval_store=approval,
        elabftw_client=stub_elabftw,
    )
    approval_id = _approval_token(approval)

    result = adapter.submit_generated_protocol(
        context_token=_token(),
        mapped_identity=_identity(),
        approval_id=approval_id,
        approval_args={"job_item_id": 99},
        job_item_id=99,
    )

    # eLabFTW experiment 42 should have provenance metadata in the validated
    # {type, value, description} object format.
    exp = stub_elabftw.get_experiment(42)
    assert exp["metadata"] is not None
    meta = exp["metadata"]
    extra = meta.get("extra_fields", {})
    assert extra.get("wallac_job_id", {}).get("value") == "1"
    assert extra.get("wallac_job_status", {}).get("value") == "submitted"
    assert (
        extra.get("wallac_bridge_audit_action_id", {}).get("value") == "bridge-audit-1"
    )
    assert extra.get("wallac_submit_tool", {}).get("value") == TOOL_SUBMIT
    assert extra.get("wallac_gateway_audit_action_id", {}).get("value") is not None
    assert result.audit_action_id == extra["wallac_gateway_audit_action_id"]["value"]


# --- result package includes audit ID (acceptance check) --------------------


def test_submit_result_includes_audit_id(
    policy: PolicyEngine,
    audit: AuditStore,
    stub_client: StubWallacClient,
    stub_bridge_client: StubWallacBridgeClient,
    approval: ApprovalStore,
) -> None:
    """Result package includes audit ID (acceptance check)."""
    policy = PolicyEngine(
        max_tier=Tier.CLOSED_LOOP_AUTONOMY,
        block_hardware_above=Tier.CLOSED_LOOP_AUTONOMY,
    )

    adapter = WallacAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=stub_client,
        bridge_client=stub_bridge_client,
        approval_store=approval,
    )
    approval_id = _approval_token(approval)

    result = adapter.submit_generated_protocol(
        context_token=_token(),
        mapped_identity=_identity(),
        approval_id=approval_id,
        approval_args={"job_item_id": 99},
        job_item_id=99,
    )
    # Result includes audit_action_id
    assert result.audit_action_id is not None
    assert result.to_dict()["audit_action_id"] == result.audit_action_id


# --- no approval denies ----------------------------------------------------


def test_submit_no_approval_denies(
    policy: PolicyEngine,
    audit: AuditStore,
    stub_client: StubWallacClient,
    stub_bridge_client: StubWallacBridgeClient,
) -> None:
    """Missing approval_id is denied by policy (approval_required)."""
    policy = PolicyEngine(
        max_tier=Tier.CLOSED_LOOP_AUTONOMY,
        block_hardware_above=Tier.CLOSED_LOOP_AUTONOMY,
    )

    adapter = WallacAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=stub_client,
        bridge_client=stub_bridge_client,
    )
    with pytest.raises(PolicyDenied) as exc_info:
        adapter.submit_generated_protocol(
            context_token=_token(),
            mapped_identity=_identity(),
            approval_id="",  # no approval
            approval_args={"job_item_id": 99},
            job_item_id=99,
        )
    assert "approval_required" in exc_info.value.reason
    # No bridge call
    assert len(stub_bridge_client.calls) == 0
    # Audit records the deny
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["policy_decision"] == "deny"


# --- mismatched approval args ----------------------------------------------


def test_submit_mismatched_approval_args_denies(
    policy: PolicyEngine,
    audit: AuditStore,
    stub_client: StubWallacClient,
    stub_bridge_client: StubWallacBridgeClient,
    approval: ApprovalStore,
) -> None:
    """Approval with wrong args hash is denied at consume time."""
    policy = PolicyEngine(
        max_tier=Tier.CLOSED_LOOP_AUTONOMY,
        block_hardware_above=Tier.CLOSED_LOOP_AUTONOMY,
    )

    adapter = WallacAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=stub_client,
        bridge_client=stub_bridge_client,
        approval_store=approval,
    )
    # Issue approval bound to args={"job_item_id": 99}
    approval_id = _approval_token(approval)

    # Call with different args (different hash)
    with pytest.raises(WallacAdapterError) as exc_info:
        adapter.submit_generated_protocol(
            context_token=_token(),
            mapped_identity=_identity(),
            approval_id=approval_id,
            approval_args={"job_item_id": 999},  # different from what was approved
            job_item_id=999,
        )
    assert "approval" in exc_info.value.reason.lower()
    # Approval was NOT consumed (mismatch prevents consume)
    record = approval.get(approval_id)
    assert record is not None
    assert not record.is_consumed()
    # No bridge call
    assert len(stub_bridge_client.calls) == 0


# --- bridge not configured -------------------------------------------------


def test_submit_bridge_not_configured_denies(
    policy: PolicyEngine,
    audit: AuditStore,
    stub_client: StubWallacClient,
    approval: ApprovalStore,
) -> None:
    """Bridge not configured raises structured error."""
    policy = PolicyEngine(
        max_tier=Tier.CLOSED_LOOP_AUTONOMY,
        block_hardware_above=Tier.CLOSED_LOOP_AUTONOMY,
    )

    adapter = WallacAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=stub_client,
        bridge_client=None,
        approval_store=approval,
    )
    approval_id = _approval_token(approval)
    with pytest.raises(WallacAdapterError) as exc_info:
        adapter.submit_generated_protocol(
            context_token=_token(),
            mapped_identity=_identity(),
            approval_id=approval_id,
            approval_args={"job_item_id": 99},
            job_item_id=99,
        )
    assert exc_info.value.reason == "wallac_bridge_not_configured"
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["error"]["code"] == "WALLAC_BRIDGE_NOT_CONFIGURED"


# --- bridge error propagates -----------------------------------------------


def test_submit_bridge_error_propagates(
    policy: PolicyEngine,
    audit: AuditStore,
    stub_client: StubWallacClient,
    approval: ApprovalStore,
) -> None:
    """Bridge client error propagates after audit."""
    policy = PolicyEngine(
        max_tier=Tier.CLOSED_LOOP_AUTONOMY,
        block_hardware_above=Tier.CLOSED_LOOP_AUTONOMY,
    )

    bridge = StubWallacBridgeClient(error=ConnectionError("bridge offline"))
    adapter = WallacAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=stub_client,
        bridge_client=bridge,
        approval_store=approval,
    )
    approval_id = _approval_token(approval)
    with pytest.raises(WallacAdapterError) as exc_info:
        adapter.submit_generated_protocol(
            context_token=_token(),
            mapped_identity=_identity(),
            approval_id=approval_id,
            approval_args={"job_item_id": 99},
            job_item_id=99,
        )
    assert exc_info.value.reason == "client_error"
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["error"]["code"] == "CLIENT_ERROR"
    assert rows[0]["policy_decision"] == "allow"  # policy allowed, bridge failed
    # Approval was consumed before bridge call
    record = approval.get(approval_id)
    assert record is not None
    assert record.is_consumed()


# --- abort path ------------------------------------------------------------


def test_submit_abort_path(
    policy: PolicyEngine,
    audit: AuditStore,
    stub_client: StubWallacClient,
    stub_bridge_client: StubWallacBridgeClient,
    approval: ApprovalStore,
) -> None:
    """Abort path produces operator-review state (acceptance check)."""
    policy = PolicyEngine(
        max_tier=Tier.CLOSED_LOOP_AUTONOMY,
        block_hardware_above=Tier.CLOSED_LOOP_AUTONOMY,
    )

    # Simulate bridge returning aborted status
    stub_bridge_client.submit_response = {
        "job_id": 1,
        "status": "aborted",
        "audit_action_id": "bridge-audit-1",
    }

    adapter = WallacAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=stub_client,
        bridge_client=stub_bridge_client,
        approval_store=approval,
    )
    approval_id = _approval_token(approval)

    result = adapter.submit_generated_protocol(
        context_token=_token(),
        mapped_identity=_identity(),
        approval_id=approval_id,
        approval_args={"job_item_id": 99},
        job_item_id=99,
    )
    # Non-submitted status is returned in result (not raised)
    assert result.result["status"] == "aborted"
    assert result.result["job_id"] == 1
    assert result.result["bridge_audit_action_id"] == "bridge-audit-1"
    # Audit records the status
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["result_summary"]["status"] == "aborted"


# --- audit hash on all paths -----------------------------------------------


def test_submit_audit_hash_present_on_success(
    policy: PolicyEngine,
    audit: AuditStore,
    stub_client: StubWallacClient,
    stub_bridge_client: StubWallacBridgeClient,
    approval: ApprovalStore,
) -> None:
    """tool_args_hash is present on success path."""
    policy = PolicyEngine(
        max_tier=Tier.CLOSED_LOOP_AUTONOMY,
        block_hardware_above=Tier.CLOSED_LOOP_AUTONOMY,
    )

    adapter = WallacAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=stub_client,
        bridge_client=stub_bridge_client,
        approval_store=approval,
    )
    approval_id = _approval_token(approval)
    adapter.submit_generated_protocol(
        context_token=_token(),
        mapped_identity=_identity(),
        approval_id=approval_id,
        approval_args={"job_item_id": 99},
        job_item_id=99,
    )
    rows = _audit_rows(audit)
    assert rows[0]["tool_args_hash"] is not None
    assert len(rows[0]["tool_args_hash"]) == 64


def test_submit_audit_hash_present_on_deny(
    policy: PolicyEngine,
    audit: AuditStore,
    stub_client: StubWallacClient,
    stub_bridge_client: StubWallacBridgeClient,
) -> None:
    """tool_args_hash is present on deny path (no approval)."""
    policy = PolicyEngine(
        max_tier=Tier.CLOSED_LOOP_AUTONOMY,
        block_hardware_above=Tier.CLOSED_LOOP_AUTONOMY,
    )

    adapter = WallacAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=stub_client,
        bridge_client=stub_bridge_client,
    )
    with pytest.raises(PolicyDenied):
        adapter.submit_generated_protocol(
            context_token=_token(),
            mapped_identity=_identity(),
            approval_id="",
            approval_args={"job_item_id": 99},
            job_item_id=99,
        )
    rows = _audit_rows(audit)
    assert rows[0]["tool_args_hash"] is not None


def test_submit_audit_hash_present_on_unknown(
    policy: PolicyEngine,
    audit: AuditStore,
    stub_client: StubWallacClient,
    stub_bridge_client: StubWallacBridgeClient,
) -> None:
    """tool_args_hash present when unmapped caller denies."""
    policy = PolicyEngine(
        max_tier=Tier.CLOSED_LOOP_AUTONOMY,
        block_hardware_above=Tier.CLOSED_LOOP_AUTONOMY,
    )

    adapter = WallacAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=stub_client,
        bridge_client=stub_bridge_client,
    )
    with pytest.raises(UnmappedCaller):
        adapter.submit_generated_protocol(
            context_token=_token(),
            mapped_identity=None,
            approval_id="test",
            approval_args={"job_item_id": 99},
            job_item_id=99,
        )
    rows = _audit_rows(audit)
    assert rows[0]["tool_args_hash"] is not None


# --- context token checks --------------------------------------------------


def test_submit_missing_context_token(
    policy: PolicyEngine,
    audit: AuditStore,
    stub_client: StubWallacClient,
    stub_bridge_client: StubWallacBridgeClient,
) -> None:
    """Missing context token denies."""
    adapter = WallacAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=stub_client,
        bridge_client=stub_bridge_client,
    )
    with pytest.raises(InvalidContextToken, match="missing"):
        adapter.submit_generated_protocol(
            context_token="",
            mapped_identity=_identity(),
            approval_id="test",
            approval_args={"job_item_id": 99},
            job_item_id=99,
        )
    rows = _audit_rows(audit)
    assert rows[0]["error"]["code"] == "MISSING_CONTEXT_TOKEN"


def test_submit_invalid_context_token(
    policy: PolicyEngine,
    audit: AuditStore,
    stub_client: StubWallacClient,
    stub_bridge_client: StubWallacBridgeClient,
) -> None:
    """Invalid context token denies."""
    adapter = WallacAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=stub_client,
        bridge_client=stub_bridge_client,
    )
    with pytest.raises(InvalidContextToken):
        adapter.submit_generated_protocol(
            context_token="garbage",
            mapped_identity=_identity(),
            approval_id="test",
            approval_args={"job_item_id": 99},
            job_item_id=99,
        )
    rows = _audit_rows(audit)
    assert rows[0]["error"]["code"] == "INVALID_CONTEXT_TOKEN"


def test_submit_unmapped_caller_denies(
    policy: PolicyEngine,
    audit: AuditStore,
    stub_client: StubWallacClient,
    stub_bridge_client: StubWallacBridgeClient,
) -> None:
    """Unmapped caller denies before any downstream call."""
    adapter = WallacAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=stub_client,
        bridge_client=stub_bridge_client,
    )
    with pytest.raises(UnmappedCaller):
        adapter.submit_generated_protocol(
            context_token=_token(),
            mapped_identity=None,
            approval_id="test",
            approval_args={"job_item_id": 99},
            job_item_id=99,
        )
    rows = _audit_rows(audit)
    assert rows[0]["error"]["code"] == "UNMAPPED_CALLER"


def test_submit_token_user_mismatch_denies(
    policy: PolicyEngine,
    audit: AuditStore,
    stub_client: StubWallacClient,
    stub_bridge_client: StubWallacBridgeClient,
) -> None:
    """Token-identity mismatch denies."""
    adapter = WallacAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=stub_client,
        bridge_client=stub_bridge_client,
    )
    with pytest.raises(InvalidContextToken):
        adapter.submit_generated_protocol(
            context_token=_token(),
            mapped_identity=_identity(elab_user_id="other-user"),
            approval_id="test",
            approval_args={"job_item_id": 99},
            job_item_id=99,
        )
    rows = _audit_rows(audit)
    assert rows[0]["error"]["code"] == "CONTEXT_TOKEN_USER_MISMATCH"


# --- kill switch deny ------------------------------------------------------


def test_submit_kill_switch_denies(
    audit: AuditStore,
    stub_client: StubWallacClient,
    stub_bridge_client: StubWallacBridgeClient,
) -> None:
    """Kill switch denies wallac.submit_generated_protocol."""
    policy = PolicyEngine(kill_switches=("wallac.submit_*",))
    adapter = WallacAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=stub_client,
        bridge_client=stub_bridge_client,
    )
    with pytest.raises(PolicyDenied):
        adapter.submit_generated_protocol(
            context_token=_token(),
            mapped_identity=_identity(),
            approval_id="test",
            approval_args={"job_item_id": 99},
            job_item_id=99,
        )
    rows = _audit_rows(audit)
    assert "kill_switch" in rows[0]["error"]["reason"]
