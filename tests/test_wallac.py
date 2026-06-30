"""Tests for the Wallac adapter (C18).

Acceptance checks (from build plan):

    1. Gateway calls Wallac health/status.
    2. Tool is read-only and allowed for mapped users.
    3. Wallac offline state is handled gracefully.

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

from lab_copilot_gateway.audit import AuditStore
from lab_copilot_gateway.elabftw import (
    ContextTokenClaims,
    InvalidContextToken,
    UnmappedCaller,
    mint_context_token,
)
from lab_copilot_gateway.identity import MappedIdentity
from lab_copilot_gateway.policy import PolicyEngine
from lab_copilot_gateway.wallac import (
    HttpWallacClient,
    PolicyDenied,
    StubWallacClient,
    WallacAdapter,
    WallacAdapterError,
    WallacResult,
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
def policy() -> PolicyEngine:
    return PolicyEngine()  # default: max_tier=6, no kill switches


@pytest.fixture
def stub_client() -> StubWallacClient:
    return StubWallacClient()


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


# --- helpers ----------------------------------------------------------------


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
