"""Tests for the BentoLab adapter (C43).

Covers:
    * Read-only tools succeed without context_token (C60 no-context mode).
    * Mutate tools still require context_token.
    * Invalid token denies.
    * Unmapped caller denies.
    * Identity binding checks.
    * Policy deny.
    * Client not configured.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from lab_copilot_gateway.audit import AuditStore
from lab_copilot_gateway.bentolab import (
    BentoLabAdapter,
    BentoLabAdapterError,
    BentoLabResult,
    PolicyDenied,
    StubBentoLabClient,
)
from lab_copilot_gateway.elabftw import (
    ContextTokenClaims,
    InvalidContextToken,
    UnmappedCaller,
    mint_context_token,
)
from lab_copilot_gateway.identity import MappedIdentity
from lab_copilot_gateway.policy import PolicyEngine


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
    return PolicyEngine()


@pytest.fixture
def stub_client() -> StubBentoLabClient:
    return StubBentoLabClient()


@pytest.fixture
def adapter(
    policy: PolicyEngine,
    audit: AuditStore,
    stub_client: StubBentoLabClient,
) -> BentoLabAdapter:
    return BentoLabAdapter(
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


def test_get_status_succeeds(adapter: BentoLabAdapter, audit: AuditStore) -> None:
    """BentoLab get_status returns device status."""
    result = adapter.get_status(
        context_token=_token(),
        mapped_identity=_identity(),
    )
    assert isinstance(result, BentoLabResult)
    assert result.tool_name == "bentolab.get_status"
    assert result.result["state"] == "idle"
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["policy_decision"] == "allow"


# --- get_status without context_token (C60 no-context mode) -----------------


def test_get_status_without_context_token_succeeds(
    adapter: BentoLabAdapter, audit: AuditStore
) -> None:
    """Read-only tools succeed without a context_token."""
    result = adapter.get_status(
        context_token="",
        mapped_identity=_identity(),
    )
    assert isinstance(result, BentoLabResult)
    assert result.tool_name == "bentolab.get_status"
    assert result.result["state"] == "idle"
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["policy_decision"] == "allow"
    # context_refs should be empty (experiment_id is None)
    assert rows[0]["context_refs"] == []


# --- validate_pcr_profile without context_token ------------------------------


def test_validate_pcr_profile_without_context_token(
    adapter: BentoLabAdapter, audit: AuditStore
) -> None:
    """validate_pcr_profile succeeds without a context_token."""
    profile = {"stages": [{"temperature": 95, "duration_s": 30}]}
    result = adapter.validate_pcr_profile(
        context_token="",
        mapped_identity=_identity(),
        profile=profile,
    )
    assert isinstance(result, BentoLabResult)
    assert result.tool_name == "bentolab.validate_pcr_profile"
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["policy_decision"] == "allow"


# --- dry_run_pcr_profile without context_token ------------------------------


def test_dry_run_pcr_profile_without_context_token(
    adapter: BentoLabAdapter, audit: AuditStore
) -> None:
    """dry_run_pcr_profile succeeds without a context_token."""
    profile = {"stages": [{"temperature": 95, "duration_s": 30}]}
    result = adapter.dry_run_pcr_profile(
        context_token="",
        mapped_identity=_identity(),
        profile=profile,
    )
    assert isinstance(result, BentoLabResult)
    assert result.tool_name == "bentolab.dry_run_pcr_profile"
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["policy_decision"] == "allow"


# --- invalid context token still denies -------------------------------------


def test_invalid_context_token_denies(
    adapter: BentoLabAdapter, audit: AuditStore
) -> None:
    """Malformed context token still denies with audit."""
    with pytest.raises(InvalidContextToken):
        adapter.get_status(
            context_token="garbage.payload",
            mapped_identity=_identity(),
        )
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["error"]["code"] == "INVALID_CONTEXT_TOKEN"


# --- unmapped caller denies -------------------------------------------------


def test_unmapped_caller_denies(adapter: BentoLabAdapter, audit: AuditStore) -> None:
    """Unmapped caller denies before any downstream call."""
    with pytest.raises(UnmappedCaller):
        adapter.get_status(
            context_token="",
            mapped_identity=None,
        )
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["error"]["code"] == "UNMAPPED_CALLER"


# --- token-identity mismatch denies ------------------------------------------


def test_token_user_mismatch_denies(
    adapter: BentoLabAdapter, audit: AuditStore
) -> None:
    """Token bound to user A cannot be used by user B."""
    with pytest.raises(InvalidContextToken, match="user does not match"):
        adapter.get_status(
            context_token=_token(),
            mapped_identity=_identity(elab_user_id="different-user"),
        )
    rows = _audit_rows(audit)
    assert rows[0]["error"]["code"] == "CONTEXT_TOKEN_USER_MISMATCH"


# --- policy deny ------------------------------------------------------------


def test_kill_switch_denies(audit: AuditStore, stub_client: StubBentoLabClient) -> None:
    """Kill switch denies via policy engine."""
    policy = PolicyEngine(kill_switches=("bentolab.*",))
    adapter = BentoLabAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=stub_client,
    )
    with pytest.raises(PolicyDenied):
        adapter.get_status(
            context_token=_token(),
            mapped_identity=_identity(),
        )
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["policy_decision"] == "deny"
    assert "kill_switch" in rows[0]["error"]["reason"]


# --- client not configured --------------------------------------------------


def test_client_not_configured_raises(audit: AuditStore, policy: PolicyEngine) -> None:
    """Client=None raises structured error after audit."""
    adapter = BentoLabAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=None,
    )
    with pytest.raises(BentoLabAdapterError) as exc_info:
        adapter.get_status(
            context_token=_token(),
            mapped_identity=_identity(),
        )
    assert exc_info.value.reason == "bentolab_not_configured"
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["error"]["code"] == "BENTOLAB_NOT_CONFIGURED"
