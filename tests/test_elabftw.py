"""Tests for the eLabFTW read adapter (C08).

Acceptance checks:

    1. Reads require mapped user — unmapped caller denies before any HTTP.
    2. Reads fail if context token is expired or invalid.
    3. Audit records context refs and hashes (on both success and failure).

Additional coverage:

    * Token-user mismatch (token bound to user A cannot be used by user B).
    * Kill switch denies via policy engine; audit deny record preserved.
    * Client error propagates after audit record persisted.
    * Stub client returns a populated ``ReadResult``.
    * Context token mint and verify roundtrip (signature + expiry).
    * Bad signature, malformed token, missing claims.
    * Module-level singleton get/reset behaviour.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from lab_copilot_gateway.approval import compute_args_hash
from lab_copilot_gateway.audit import AuditStore
from lab_copilot_gateway.elabftw import (
    ContextTokenClaims,
    ElabftwAdapterError,
    ElabftwReadAdapter,
    HttpElabftwClient,
    InvalidContextToken,
    PolicyDenied,
    StubElabftwClient,
    UnmappedCaller,
    _default_client_from_env,
    get_elabftw_read_adapter,
    mint_context_token,
    reset_elabftw_read_adapter,
    verify_context_token,
)
from lab_copilot_gateway.identity import MappedIdentity
from lab_copilot_gateway.policy import PolicyEngine


SECRET = b"test-secret-do-not-use-in-prod"


@pytest.fixture(autouse=True)
def _stable_token_secret(monkeypatch) -> None:
    """Force the adapter to use our test HMAC secret for context tokens.

    Without this the module-level ``_env_secret()`` would generate a fresh
    ephemeral secret per call, so tokens minted in tests would fail to verify
    in the adapter (different secret).  We pin the env var for every test.
    """
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
def stub_client() -> StubElabftwClient:
    return StubElabftwClient(
        seeds={
            42: {
                "id": 42,
                "title": "PCR optimization for EF1a promoter",
                "body": "<p>Test experiment body</p>",
                "metadata": {"extra_fields": {}},
            },
        }
    )


@pytest.fixture
def adapter(
    policy: PolicyEngine,
    audit: AuditStore,
    stub_client: StubElabftwClient,
) -> ElabftwReadAdapter:
    return ElabftwReadAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=stub_client,
    )


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


def _token(
    claims: ContextTokenClaims | None = None, *, secret: bytes | None = None
) -> str:
    """Mint a token.  Pass ``secret=`` only when testing secret mismatch — the
    adapter will use the env-pinned secret via _env_secret()."""
    return mint_context_token(claims or _claims(), secret=secret)


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


# --- Token mint / verify roundtrips -----------------------------------------


def test_mint_and_verify_token_roundtrip() -> None:
    """Acceptance: a freshly minted token verifies under the same secret."""
    claims = _claims()
    token = mint_context_token(claims, secret=SECRET)
    parsed = verify_context_token(token, secret=SECRET)
    assert parsed.experiment_id == 42
    assert parsed.mapped_elabftw_user_id == "elab-human-1"
    assert parsed.keycloak_subject == "kc-human-1"


def test_verify_rejects_token_minted_with_different_secret() -> None:
    """Bad signature rejected — bearer cannot forge a token without the secret.

    Use the explicit SECRET here (not the env-pinned one) so the verification
    with a DIFFERENT secret fails.  Other tests mint with secret=None which
    falls back to the env-pinned secret the adapter also uses.
    """
    token = _token(secret=SECRET)
    with pytest.raises(InvalidContextToken, match="bad signature"):
        verify_context_token(token, secret=b"different-secret")


def test_verify_rejects_tampered_token() -> None:
    """Malformed token rejected."""
    with pytest.raises(InvalidContextToken, match="malformed"):
        verify_context_token("not-a-token", secret=SECRET)
    with pytest.raises(InvalidContextToken, match="malformed"):
        verify_context_token("nopdot", secret=SECRET)


def test_verify_rejects_expired_token() -> None:
    """Acceptance check: expired token rejected with reason 'expired'."""
    claims = _claims(ttl_seconds=-10)  # already expired at mint time
    token = mint_context_token(claims, secret=SECRET)
    with pytest.raises(InvalidContextToken, match="expired"):
        verify_context_token(token, secret=SECRET)


def test_verify_rejects_missing_claims() -> None:
    """Token payload missing required claims rejected."""
    # Manually craft a payload missing experiment_id.
    import base64
    import hashlib
    import hmac
    import json

    now_iso = _dt.datetime.now(_dt.timezone.utc).isoformat()
    expires_iso = (
        _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(seconds=600)
    ).isoformat()
    payload = {
        "mapped_elabftw_user_id": "elab-human-1",
        "issued_at": now_iso,
        "expires_at": expires_iso,
    }
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload_b64 = base64.urlsafe_b64encode(payload_bytes).rstrip(b"=").decode()
    sig = hmac.new(SECRET, payload_b64.encode("ascii"), hashlib.sha256).hexdigest()
    token = f"{payload_b64}.{sig}"
    with pytest.raises(InvalidContextToken, match="missing claims"):
        verify_context_token(token, secret=SECRET)


# --- Acceptance check 1: reads require mapped user -------------------------


def test_unmapped_caller_denies_before_http(
    adapter: ElabftwReadAdapter,
    audit: AuditStore,
) -> None:
    """Unmapped caller (mapped_identity=None) is denied before any HTTP call
    to the downstream client.  Audit record persists the deny reason."""
    token = _token()
    with pytest.raises(UnmappedCaller):
        adapter.read_current_experiment(
            context_token=token,
            mapped_identity=None,
            keycloak_subject="absent",
        )

    # Audit deny record must have been persisted.
    assert audit.count() == 1
    # We can't fetch directly by action_id (it's auto-generated), so ask the
    # store for SOMETHING present.  In practice downstream code should look up
    # by request_id; here we just assert the record exists.
    row = audit.get(_last_action_id(audit))
    assert row is not None
    assert row["policy_decision"] == "deny"
    assert row["error_json"] is not None
    assert "UNMAPPED_CALLER" in row["error_json"]


def _last_action_id(store: AuditStore) -> str:
    """Helper to fetch the action_id of the most recently appended record."""
    conn = store._connect()
    with store._lock:
        row = conn.execute(
            "SELECT action_id FROM audit_records ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    return str(row[0])


# --- Security regression: audit hash on every path -------------------------


def test_audit_persists_args_hash_on_early_failure_path(
    adapter: ElabftwReadAdapter,
    audit: AuditStore,
) -> None:
    """Security regression: every deny path (including the very first one —
    missing/invalid token, before claims are resolved) must persist a
    tool_args_hash so the audit row is hashable and correlated."""
    # Missing token — earliest deny path.
    with pytest.raises(InvalidContextToken):
        adapter.read_current_experiment(
            context_token="",
            mapped_identity=_identity(),
        )
    assert audit.count() == 1
    row = audit.get(_last_action_id(audit))
    assert row is not None
    assert row["tool_args_hash"] is not None
    # Redacted digests are sha256-hex; either the missing-token or the
    # post-verify deterministic hash are 64 chars.
    assert len(row["tool_args_hash"]) == 64  # noqa: PLR2004 — sha256 hex


def test_audit_persists_args_hash_on_invalid_token_path(
    adapter: ElabftwReadAdapter,
    audit: AuditStore,
) -> None:
    """Invalid-token deny path also persists a redacted digest hash."""
    with pytest.raises(InvalidContextToken):
        adapter.read_current_experiment(
            context_token="garbage.payload",
            mapped_identity=_identity(),
        )
    row = audit.get(_last_action_id(audit))
    assert row is not None
    assert row["tool_args_hash"] is not None


def test_audit_persists_args_hash_on_policy_deny_path(
    audit: AuditStore,
    stub_client: StubElabftwClient,
) -> None:
    """Policy deny path persists the deterministic experiment_id hash."""
    engine = PolicyEngine(kill_switches=("elabftw.read_current_experiment",))
    adapter = ElabftwReadAdapter(
        policy_engine=engine,
        audit_store=audit,
        client=stub_client,
    )
    with pytest.raises(PolicyDenied):
        adapter.read_current_experiment(
            context_token=_token(),
            mapped_identity=_identity(),
        )
    row = audit.get(_last_action_id(audit))
    assert row is not None
    expected_hash = compute_args_hash({"experiment_id": 42})
    assert row["tool_args_hash"] == expected_hash


# --- Cross-user token binding defence --------------------------------------


def test_token_keycloak_subject_mismatch_denies(
    adapter: ElabftwReadAdapter,
    audit: AuditStore,
) -> None:
    """Token bound to kc_subj=Alice cannot be replayed with the resolved
    identity kc_subj=Bob — cross-user replay defence extends to the
    keycloak_subject claim, not just elabftw_user_id."""
    # Token claims kc_subj=alice.
    token = _token(
        _claims(
            mapped_elabftw_user_id="elab-alice",
            keycloak_subject="kc-alice",
        )
    )
    # Resolved identity says kc_subj=bob BUT the elabftw_user_id matches the
    # token.  Without the cross-user check on kc_subj this would succeed.
    # The cross-check should catch the mismatch.
    alice_but_kc_bob = _identity(
        elab_user_id="elab-alice",
        kc_subject="kc-bob",  # different from token's kc-alice
    )
    with pytest.raises(InvalidContextToken, match="keycloak_subject does not match"):
        adapter.read_current_experiment(
            context_token=token,
            mapped_identity=alice_but_kc_bob,
        )
    row = audit.get(_last_action_id(audit))
    assert row is not None
    assert "CONTEXT_TOKEN_KC_SUBJECT_MISMATCH" in (row["error_json"] or "")


def test_token_librechat_user_id_mismatch_denies(
    adapter: ElabftwReadAdapter,
    audit: AuditStore,
) -> None:
    """Cross-user replay defence extends to librechat_user_id claim too."""
    token = _token(
        _claims(
            mapped_elabftw_user_id="elab-alice",
            librechat_user_id="lc-alice",
        )
    )
    alice_but_lc_bob = _identity(
        elab_user_id="elab-alice",
        lc_user_id="lc-bob",
    )
    with pytest.raises(InvalidContextToken, match="librechat_user_id does not match"):
        adapter.read_current_experiment(
            context_token=token,
            mapped_identity=alice_but_lc_bob,
        )
    row = audit.get(_last_action_id(audit))
    assert row is not None
    assert "CONTEXT_TOKEN_LC_USER_MISMATCH" in (row["error_json"] or "")


# --- Audit persistence hard requirement on success path --------------------


def test_audit_persistence_failure_blocks_successful_read(
    audit: AuditStore,
    stub_client: StubElabftwClient,
    policy: PolicyEngine,
) -> None:
    """If the audit store fails on the success path, the adapter MUST raise
    instead of returning read data — a read without an audit trail violates
    the gateway's safety contract."""

    class _FailingAuditStore(AuditStore):
        """Audit store whose append() always raises on success-path calls."""

        def append(self, record):  # type: ignore[override]
            if record.policy_decision == "allow" and record.api_call_summary:
                # Reason: simulate SQLite connection lost mid-call after the
                # downstream read succeeded.  The adapter must surface this as
                # an ElabftwAdapterError, not silently return data.
                raise RuntimeError("simulated audit DB outage")
            return super().append(record)

    failing_store = _FailingAuditStore(db_path=":memory:")
    adapter = ElabftwReadAdapter(
        policy_engine=policy,
        audit_store=failing_store,
        client=stub_client,
    )
    with pytest.raises(ElabftwAdapterError) as exc:
        adapter.read_current_experiment(
            context_token=_token(),
            mapped_identity=_identity(),
        )
    assert exc.value.reason == "audit_persistence_failed"
    failing_store.close()


def test_experiment_id_zero_denies_with_no_context_reason(
    audit: AuditStore,
    stub_client: StubElabftwClient,
    policy: PolicyEngine,
) -> None:
    """Regression: experiment_id 0 means "no experiment context" (see
    elabftw_mint_context_token comment).  eLabFTW's REST API treats
    /experiments/0 as an unfiltered list and returns a JSON array, not a
    single record — calling .get("title") on that crashes the gateway
    with AttributeError.  The adapter must fail fast with a structured
    no_experiment_context error before reaching the eLabFTW client,
    and must NOT call the downstream client (which would 500).
    """
    adapter = ElabftwReadAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=stub_client,
    )
    token = _token(claims=_claims(experiment_id=0))
    with pytest.raises(ElabftwAdapterError) as exc:
        adapter.read_current_experiment(
            context_token=token,
            mapped_identity=_identity(),
        )
    assert exc.value.reason == "no_experiment_context"
    assert "experiment_id=0" in exc.value.message
    # Stub client must NOT have been called.
    assert stub_client.get_experiment_calls == 0
    assert stub_client.get_item_calls == 0


def test_experiment_id_zero_audit_records_deny_with_no_context(
    audit: AuditStore,
    stub_client: StubElabftwClient,
    policy: PolicyEngine,
) -> None:
    """Regression: experiment_id 0 must persist a deny audit record
    (code NO_EXPERIMENT_CONTEXT) so the failure is observable in the
    audit log even though the downstream client was never called."""
    adapter = ElabftwReadAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=stub_client,
    )
    token = _token(claims=_claims(experiment_id=0))
    with pytest.raises(ElabftwAdapterError):
        adapter.read_current_experiment(
            context_token=token,
            mapped_identity=_identity(),
        )
    assert audit.count() == 1
    row = audit.get(_last_action_id(audit))
    assert row is not None
    assert row["policy_decision"] == "deny"
    assert "NO_EXPERIMENT_CONTEXT" in (row["error_json"] or "")


def test_read_experiment_by_id_zero_denies_with_no_context(
    audit: AuditStore,
    stub_client: StubElabftwClient,
    policy: PolicyEngine,
) -> None:
    """Regression: read_experiment_by_id with id=0 (the dispatcher default
    when the LLM omits the arg) must also deny with no_experiment_context.
    Same eLabFTW 'id=0 returns a list' trap as read_current_experiment."""
    adapter = ElabftwReadAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=stub_client,
    )
    with pytest.raises(ElabftwAdapterError) as exc:
        adapter.read_experiment_by_id(
            experiment_id=0,
            mapped_identity=_identity(),
        )
    assert exc.value.reason == "no_experiment_context"
    assert stub_client.get_experiment_calls == 0


def test_read_item_by_id_zero_denies_with_no_context(
    audit: AuditStore,
    stub_client: StubElabftwClient,
    policy: PolicyEngine,
) -> None:
    """Regression: read_item_by_id with id=0 (the dispatcher default when
    the LLM omits item_id) must also deny with no_experiment_context."""
    adapter = ElabftwReadAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=stub_client,
    )
    with pytest.raises(ElabftwAdapterError) as exc:
        adapter.read_item_by_id(
            item_id=0,
            mapped_identity=_identity(),
        )
    assert exc.value.reason == "no_experiment_context"
    assert stub_client.get_item_calls == 0


# --- Metadata normalization -----------------------------------------------


def test_normalize_metadata_handles_dict_passthrough() -> None:
    from lab_copilot_gateway.elabftw import _normalize_metadata

    assert _normalize_metadata({"a": 1}) == {"a": 1}


def test_normalize_metadata_handles_none() -> None:
    from lab_copilot_gateway.elabftw import _normalize_metadata

    assert _normalize_metadata(None) is None


def test_normalize_metadata_handles_json_string() -> None:
    from lab_copilot_gateway.elabftw import _normalize_metadata

    assert _normalize_metadata('{"a": 1}') == {"a": 1}


def test_normalize_metadata_handles_double_encoded_json_string() -> None:
    from lab_copilot_gateway.elabftw import _normalize_metadata

    # Double-encoded: json.dumps(json.dumps({"a":1})) → '"{\\"a\\": 1}"'
    import json

    double = json.dumps(json.dumps({"a": 1}))
    assert _normalize_metadata(double) == {"a": 1}


def test_normalize_metadata_handles_invalid_json_string() -> None:
    from lab_copilot_gateway.elabftw import _normalize_metadata

    out = _normalize_metadata("not json")
    assert out == {"_raw": "not json"}


# --- Env-secret per-process caching ---------------------------------------


def test_env_secret_returns_stable_ephemeral_when_unset(monkeypatch) -> None:
    """When LAB_COPILOT_TOKEN_SECRET is unset, the dev-only fallback should be
    stable across calls within the same process (not regenerated per call)."""
    import lab_copilot_gateway.elabftw as elabmod

    monkeypatch.delenv("LAB_COPILOT_TOKEN_SECRET", raising=False)
    # Reset the cached dev secret so we exercise the generation path.
    elabmod._dev_ephemeral_secret_cache = None
    s1 = elabmod._env_secret()
    s2 = elabmod._env_secret()
    assert s1 == s2 and len(s1) == 32  # noqa: PLR2004 — secret length
    # Restore so we don't poison other tests.
    elabmod._dev_ephemeral_secret_cache = None


def test_env_secret_returns_env_value_when_set(monkeypatch) -> None:
    monkeypatch.setenv("LAB_COPILOT_TOKEN_SECRET", "explicit-stable-secret")
    from lab_copilot_gateway.elabftw import _env_secret

    assert _env_secret() == b"explicit-stable-secret"


# --- Expiry parse as datetime, robust to offsets --------------------------


def test_verify_token_rejects_malformed_expires_at() -> None:
    """A signed token with a malformed expires_at string raises
    InvalidContextToken (not ValueError) — claim coercion is guarded."""
    import base64
    import hashlib
    import hmac
    import json

    payload = {
        "experiment_id": 42,
        "mapped_elabftw_user_id": "elab-human-1",
        "issued_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "expires_at": "not-a-date",
    }
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload_b64 = base64.urlsafe_b64encode(payload_bytes).rstrip(b"=").decode()
    sig = hmac.new(SECRET, payload_b64.encode("ascii"), hashlib.sha256).hexdigest()
    token = f"{payload_b64}.{sig}"
    with pytest.raises(InvalidContextToken, match="malformed expires_at"):
        verify_context_token(token, secret=SECRET)


def test_verify_token_rejects_non_int_experiment_id() -> None:
    """Signed token with non-integer experiment_id raises InvalidContextToken
    (TypeError caught) instead of leaking up to the caller."""
    import base64
    import hashlib
    import hmac
    import json

    payload = {
        "experiment_id": "not-an-int",
        "mapped_elabftw_user_id": "elab-human-1",
        "issued_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "expires_at": (
            _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(seconds=600)
        ).isoformat(),
    }
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload_b64 = base64.urlsafe_b64encode(payload_bytes).rstrip(b"=").decode()
    sig = hmac.new(SECRET, payload_b64.encode("ascii"), hashlib.sha256).hexdigest()
    token = f"{payload_b64}.{sig}"
    with pytest.raises(InvalidContextToken, match="malformed claim types"):
        verify_context_token(token, secret=SECRET)


# --- Acceptance check 2: invalid / expired context token denied -------------


def test_invalid_context_token_denies_before_http(
    adapter: ElabftwReadAdapter,
    audit: AuditStore,
) -> None:
    """Invalid token (bad signature) is denied before any HTTP call."""
    with pytest.raises(InvalidContextToken):
        adapter.read_current_experiment(
            context_token="garbage.payload",
            mapped_identity=_identity(),
        )
    assert audit.count() == 1
    row = audit.get(_last_action_id(audit))
    assert row is not None
    assert row["policy_decision"] == "deny"
    assert "INVALID_CONTEXT_TOKEN" in (row["error_json"] or "")


def test_expired_context_token_denies_before_http(
    adapter: ElabftwReadAdapter,
    audit: AuditStore,
) -> None:
    """Acceptance check: expired token denied with audit record persisted."""
    token = _token(_claims(ttl_seconds=-10))
    with pytest.raises(InvalidContextToken, match="expired"):
        adapter.read_current_experiment(
            context_token=token,
            mapped_identity=_identity(),
        )
    assert audit.count() == 1
    row = audit.get(_last_action_id(audit))
    assert row is not None
    assert "expired" in (row["error_json"] or "")


def test_missing_context_token_denies_before_http(
    adapter: ElabftwReadAdapter,
    audit: AuditStore,
) -> None:
    """Empty context token rejected at the adapter boundary."""
    with pytest.raises(InvalidContextToken, match="missing"):
        adapter.read_current_experiment(
            context_token="",
            mapped_identity=_identity(),
        )
    assert audit.count() == 1


def test_token_user_mismatch_denies(
    adapter: ElabftwReadAdapter,
    audit: AuditStore,
) -> None:
    """Token bound to user A cannot be used by user B's resolved identity.

    This is the cross-user replay defence — without it, a stolen token from
    user A could be paired with user B's mapped identity.
    """
    token = _token(_claims(mapped_elabftw_user_id="elab-alice"))
    bob_identity = _identity(elab_user_id="elab-bob")
    with pytest.raises(InvalidContextToken, match="token user does not match"):
        adapter.read_current_experiment(
            context_token=token,
            mapped_identity=bob_identity,
        )
    assert audit.count() == 1
    row = audit.get(_last_action_id(audit))
    assert row is not None
    assert "CONTEXT_TOKEN_USER_MISMATCH" in (row["error_json"] or "")


# --- Acceptance check 3: audit records context refs and hashes -----------


def test_audit_persists_context_refs_and_args_hash_on_success(
    adapter: ElabftwReadAdapter,
    audit: AuditStore,
) -> None:
    """Acceptance: successful read writes an audit record containing
    context_refs with experiment_id and a tool_args_hash."""
    token = _token()
    result = adapter.read_current_experiment(
        context_token=token,
        mapped_identity=_identity(),
        conversation_id="conv-1",
        request_id="req-1",
        provider="openai",
        model_id="gpt-4o-mini",
    )

    assert result.experiment_id == 42
    assert result.title == "PCR optimization for EF1a promoter"
    assert "<p>Test experiment body</p>" in result.body
    assert result.metadata == {"extra_fields": {}}

    assert audit.count() == 1
    row = audit.get(_last_action_id(audit))
    assert row is not None
    assert row["tool_name"] == "elabftw.read_current_experiment"
    assert row["tool_args_hash"] is not None
    assert len(row["tool_args_hash"]) == 64  # sha256 hex
    assert "experiment" in (row["context_refs_json"] or "")
    assert "42" in (row["context_refs_json"] or "")
    assert row["policy_decision"] == "allow"
    assert row["conversation_id"] == "conv-1"
    assert row["request_id"] == "req-1"
    assert row["provider"] == "openai"
    assert row["model_id"] == "gpt-4o-mini"
    assert row["mapped_elabftw_user_id"] == "elab-human-1"
    assert "GET" in (row["api_call_summary_json"] or "")


def test_audit_persists_on_failure_path(
    adapter: ElabftwReadAdapter,
    audit: AuditStore,
) -> None:
    """Audit record persists even when the read fails downstream.  Client
    exceptions are wrapped in ``ElabftwAdapterError(reason='client_error')``
    after the audit row is written so the HTTP endpoint can return the
    documented {ok:false, ...} shape instead of an unhandled 500."""
    token = _token()
    # Stub client with no seed for experiment 42 → KeyError.
    empty_client = StubElabftwClient(seeds={})
    adapter.client = empty_client
    with pytest.raises(ElabftwAdapterError) as exc:
        adapter.read_current_experiment(
            context_token=token,
            mapped_identity=_identity(),
        )
    assert exc.value.reason == "client_error"
    assert audit.count() == 1
    row = audit.get(_last_action_id(audit))
    assert row is not None
    assert row["policy_decision"] == "allow"  # policy allowed, client errored
    assert "CLIENT_ERROR" in (row["error_json"] or "")


# --- Policy / kill switch integration -------------------------------------


def test_kill_switch_denies_via_policy_engine(
    audit: AuditStore,
    stub_client: StubElabftwClient,
) -> None:
    """Kill switch on the tool name denies via the policy engine; audit
    records the deny reason."""
    engine = PolicyEngine(kill_switches=("elabftw.read_current_experiment",))
    adapter = ElabftwReadAdapter(
        policy_engine=engine,
        audit_store=audit,
        client=stub_client,
    )
    token = _token()
    with pytest.raises(PolicyDenied):
        adapter.read_current_experiment(
            context_token=token,
            mapped_identity=_identity(),
        )
    assert audit.count() == 1
    row = audit.get(_last_action_id(audit))
    assert row is not None
    assert row["policy_decision"] == "deny"
    assert "kill_switch" in (row["error_json"] or "")


# --- Client not configured -------------------------------------------------


def test_adapter_without_client_denies_after_policy(
    audit: AuditStore,
    policy: PolicyEngine,
) -> None:
    """When no downstream client is configured (env missing), the adapter
    fails observably after the policy decision."""
    adapter = ElabftwReadAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=None,
    )
    token = _token()
    with pytest.raises(ElabftwAdapterError) as exc:
        adapter.read_current_experiment(
            context_token=token,
            mapped_identity=_identity(),
        )
    assert exc.value.reason == "elabftw_not_configured"
    assert audit.count() == 1
    row = audit.get(_last_action_id(audit))
    assert row is not None
    assert "ELABFTW_NOT_CONFIGURED" in (row["error_json"] or "")


# --- Stub client behaviour ------------------------------------------------


def test_stub_client_unknown_experiment_raises() -> None:
    c = StubElabftwClient(seeds={1: {}})
    with pytest.raises(KeyError):
        c.get_experiment(999)


def test_stub_client_returns_copy_of_seed() -> None:
    """Stub returns a copy so callers can't mutate the seed."""
    seed = {"id": 42, "title": "original"}
    c = StubElabftwClient(seeds={42: seed})
    out = c.get_experiment(42)
    out["title"] = "mutated"
    assert seed["title"] == "original"  # seed untouched


# --- HTTP client surface --------------------------------------------------


def test_http_client_requires_base_url_and_api_key() -> None:
    with pytest.raises(ValueError, match="base_url"):
        HttpElabftwClient(base_url="", api_key="k")
    with pytest.raises(ValueError, match="api_key"):
        HttpElabftwClient(base_url="https://example.org", api_key="")


def test_default_client_from_env_returns_none_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("LAB_COPILOT_ELABFTW_BASE_URL", raising=False)
    monkeypatch.delenv("LAB_COPILOT_ELABFTW_API_KEY", raising=False)
    assert _default_client_from_env() is None


def test_default_client_from_env_builds_when_set(monkeypatch) -> None:
    monkeypatch.setenv("LAB_COPILOT_ELABFTW_BASE_URL", "https://elab.example.org")
    monkeypatch.setenv("LAB_COPILOT_ELABFTW_API_KEY", "2-abcdef")
    monkeypatch.setenv("LAB_COPILOT_ELABFTW_VERIFY_TLS", "0")
    client = _default_client_from_env()
    assert client is not None
    assert isinstance(client, HttpElabftwClient)
    assert client.base_url == "https://elab.example.org"
    assert client.api_key == "2-abcdef"
    assert client.verify_tls is False


def test_default_client_from_env_verify_tls_truthy(monkeypatch) -> None:
    monkeypatch.setenv("LAB_COPILOT_ELABFTW_BASE_URL", "https://elab.example.org")
    monkeypatch.setenv("LAB_COPILOT_ELABFTW_API_KEY", "2-abcdef")
    monkeypatch.setenv("LAB_COPILOT_ELABFTW_VERIFY_TLS", "true")
    client = _default_client_from_env()
    assert client is not None
    assert client.verify_tls is True


# --- Singleton behaviour --------------------------------------------------


def test_get_elabftw_read_adapter_returns_singleton(
    monkeypatch,
    audit: AuditStore,
) -> None:
    """Singleton construction uses env-config + module singletons."""
    # Make sure adapter construction doesn't fail because of missing env —
    # the default client is allowed to be None.
    monkeypatch.delenv("LAB_COPILOT_ELABFTW_BASE_URL", raising=False)
    monkeypatch.delenv("LAB_COPILOT_ELABFTW_API_KEY", raising=False)

    import lab_copilot_gateway.audit as auditmod
    import lab_copilot_gateway.policy as policymod

    auditmod._default_store = audit
    policymod._default_engine = PolicyEngine()
    reset_elabftw_read_adapter()
    first = get_elabftw_read_adapter()
    second = get_elabftw_read_adapter()
    assert first is second
    reset_elabftw_read_adapter()


def test_reset_elabftw_read_adapter_clears_singleton() -> None:
    import lab_copilot_gateway.elabftw as elabmod

    reset_elabftw_read_adapter()
    assert elabmod._default_adapter is None


# --- Error class to_dict ---------------------------------------------------


def test_unmapped_caller_to_dict() -> None:
    err = UnmappedCaller()
    out = err.to_dict()
    assert out["reason"] == "unmapped_caller"
    assert "no mapped" in out["message"]


def test_adapter_error_to_dict() -> None:
    err = ElabftwAdapterError(reason="custom", message="some message")
    out = err.to_dict()
    assert out == {"reason": "custom", "message": "some message"}


# --- C36: multi-record retrieval --------------------------------------------


# -- StubElabftwClient.search_experiments --


def test_stub_search_finds_by_title() -> None:
    """Stub search matches by title (case-insensitive substring)."""
    c = StubElabftwClient(
        seeds={
            1: {"id": 1, "title": "Gel electrophoresis", "body": ""},
            2: {"id": 2, "title": "PCR optimization", "body": ""},
        }
    )
    results = c.search_experiments("pcr")
    assert len(results) == 1
    assert results[0]["id"] == 2


def test_stub_search_finds_by_body() -> None:
    """Stub search matches by body keyword (case-insensitive)."""
    c = StubElabftwClient(
        seeds={
            1: {"id": 1, "title": "Run A", "body": "<p>western blot</p>"},
            2: {"id": 2, "title": "Run B", "body": "<p>agarose gel</p>"},
        }
    )
    results = c.search_experiments("western")
    assert len(results) == 1
    assert results[0]["id"] == 1


def test_stub_search_no_matches_returns_empty() -> None:
    """Stub search with no matching term returns empty list."""
    c = StubElabftwClient(seeds={1: {"id": 1, "title": "A", "body": "B"}})
    assert c.search_experiments("zzz_no_match") == []


def test_stub_search_respects_limit_and_offset() -> None:
    """Stub search applies offset and limit correctly."""
    c = StubElabftwClient(
        seeds={
            1: {"id": 1, "title": "PCR test A", "body": ""},
            2: {"id": 2, "title": "PCR test B", "body": ""},
            3: {"id": 3, "title": "PCR test C", "body": ""},
        }
    )
    results = c.search_experiments("pcr", limit=2, offset=1)
    assert len(results) == 2
    assert {r["id"] for r in results} == {2, 3}


# -- ElabftwReadAdapter.search_my_experiments --


def test_search_success_returns_summaries_and_audits(
    audit: AuditStore,
    policy: PolicyEngine,
) -> None:
    """Happy path: mapped identity, multiple seeds, search returns
    SearchResult with summaries and full audit trail."""
    client = StubElabftwClient(
        seeds={
            42: {
                "id": 42,
                "title": "PCR optimization for EF1a promoter",
                "body": "<p>Body 42</p>",
                "metadata": {"extra_fields": {}},
            },
            43: {
                "id": 43,
                "title": "PCR for CRISPR guide",
                "body": "<p>Body 43</p>",
                "metadata": {"extra_fields": {}},
            },
        }
    )
    adapter = ElabftwReadAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=client,
    )
    result = adapter.search_my_experiments(
        query="PCR",
        limit=10,
        mapped_identity=_identity(),
        conversation_id="conv-1",
        request_id="req-1",
        provider="openai",
        model_id="gpt-4o-mini",
    )

    # Result structure.
    assert result.query == "PCR"
    assert result.total_returned == 2
    assert len(result.experiments) == 2
    ids = {e.experiment_id for e in result.experiments}
    assert ids == {42, 43}
    titles = {e.title for e in result.experiments}
    assert titles == {
        "PCR optimization for EF1a promoter",
        "PCR for CRISPR guide",
    }

    # Audit record.
    assert audit.count() == 1
    row = audit.get(_last_action_id(audit))
    assert row is not None
    assert row["tool_name"] == "elabftw.search_my_experiments"
    assert row["policy_decision"] == "allow"

    # api_call_summary has query/limit/offset.
    import json as _json

    api = _json.loads(row["api_call_summary_json"])
    assert api["query"] == "PCR"
    assert api["limit"] == 10
    assert api["offset"] == 0

    # result_summary has record_count + record_ids.
    res_meta = _json.loads(row["result_summary_json"])
    assert res_meta["record_count"] == 2
    assert set(res_meta["record_ids"]) == {42, 43}

    # context_refs has one entry per returned record.
    ctx = _json.loads(row["context_refs_json"])
    assert len(ctx) == 2
    ctx_ids = {r["id"] for r in ctx if r["kind"] == "experiment"}
    assert ctx_ids == {42, 43}

    # to_dict roundtrip.
    d = result.to_dict()
    assert d["query"] == "PCR"
    assert d["total_returned"] == 2
    assert len(d["experiments"]) == 2


def test_search_unmapped_caller_denies(
    adapter: ElabftwReadAdapter,
    audit: AuditStore,
) -> None:
    """Unmapped caller (mapped_identity=None) → UnmappedCaller, audit deny."""
    with pytest.raises(UnmappedCaller):
        adapter.search_my_experiments(
            query="PCR",
            mapped_identity=None,
        )
    assert audit.count() == 1
    row = audit.get(_last_action_id(audit))
    assert row is not None
    assert row["policy_decision"] == "deny"
    assert row["tool_name"] == "elabftw.search_my_experiments"
    assert "UNMAPPED_CALLER" in (row["error_json"] or "")


def test_search_policy_denied(
    audit: AuditStore,
    policy: PolicyEngine,
) -> None:
    """Kill switch on search tool → PolicyDenied, audit deny."""
    engine = PolicyEngine(kill_switches=("elabftw.search_my_experiments",))
    client = StubElabftwClient(seeds={42: {"id": 42, "title": "A", "body": ""}})
    adapter = ElabftwReadAdapter(
        policy_engine=engine,
        audit_store=audit,
        client=client,
    )
    with pytest.raises(PolicyDenied):
        adapter.search_my_experiments(
            query="A",
            mapped_identity=_identity(),
        )
    assert audit.count() == 1
    row = audit.get(_last_action_id(audit))
    assert row is not None
    assert row["policy_decision"] == "deny"
    assert "kill_switch" in (row["error_json"] or "")


def test_search_client_not_configured(
    audit: AuditStore,
    policy: PolicyEngine,
) -> None:
    """client=None → ElabftwAdapterError, audit deny."""
    adapter = ElabftwReadAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=None,
    )
    with pytest.raises(ElabftwAdapterError) as exc:
        adapter.search_my_experiments(
            query="PCR",
            mapped_identity=_identity(),
        )
    assert exc.value.reason == "elabftw_not_configured"
    assert audit.count() == 1
    row = audit.get(_last_action_id(audit))
    assert row is not None
    assert "ELABFTW_NOT_CONFIGURED" in (row["error_json"] or "")


def test_search_client_error(
    audit: AuditStore,
    policy: PolicyEngine,
) -> None:
    """Client raises → ElabftwAdapterError(reason='client_error'), audit allow."""

    # Custom client whose search_experiments always raises.
    class _FailingClient:
        def search_experiments(self, query, limit=20, offset=0):
            raise ConnectionError("downstream unreachable")

    adapter = ElabftwReadAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=_FailingClient(),  # type: ignore[arg-type]
    )
    with pytest.raises(ElabftwAdapterError) as exc:
        adapter.search_my_experiments(
            query="PCR",
            mapped_identity=_identity(),
        )
    assert exc.value.reason == "client_error"
    assert audit.count() == 1
    row = audit.get(_last_action_id(audit))
    assert row is not None
    assert row["policy_decision"] == "allow"
    assert "CLIENT_ERROR" in (row["error_json"] or "")


def test_search_limit_clamped_to_max(
    audit: AuditStore,
    policy: PolicyEngine,
) -> None:
    """limit=100 clamped to _MAX_SEARCH_RESULTS (50) in audit record."""
    from lab_copilot_gateway.elabftw import _MAX_SEARCH_RESULTS

    client = StubElabftwClient(seeds={1: {"id": 1, "title": "PCR", "body": ""}})
    adapter = ElabftwReadAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=client,
    )
    adapter.search_my_experiments(
        query="PCR",
        limit=100,
        offset=5,
        mapped_identity=_identity(),
    )
    row = audit.get(_last_action_id(audit))
    assert row is not None
    import json as _json

    api = _json.loads(row["api_call_summary_json"])
    assert api["limit"] == _MAX_SEARCH_RESULTS
    # offset is clamped to >= 0, so 5 stays 5.
    assert api["offset"] == 5


def test_search_audit_persistence_failure_blocks_result(
    audit: AuditStore,
    policy: PolicyEngine,
) -> None:
    """Failing audit store on success path → ElabftwAdapterError."""

    class _FailingAuditStore(AuditStore):
        def append(self, record):
            # Fail on the success (allow + api_call_summary) path only.
            if record.policy_decision == "allow" and record.api_call_summary:
                raise RuntimeError("simulated audit DB outage")
            return super().append(record)

    failing_store = _FailingAuditStore(db_path=":memory:")
    client = StubElabftwClient(seeds={42: {"id": 42, "title": "PCR", "body": ""}})
    adapter = ElabftwReadAdapter(
        policy_engine=policy,
        audit_store=failing_store,
        client=client,
    )
    with pytest.raises(ElabftwAdapterError) as exc:
        adapter.search_my_experiments(
            query="PCR",
            mapped_identity=_identity(),
        )
    assert exc.value.reason == "audit_persistence_failed"
    failing_store.close()


# -- ElabftwReadAdapter.read_experiment_by_id --


def test_read_by_id_success(
    adapter: ElabftwReadAdapter,
    audit: AuditStore,
) -> None:
    """Happy path: reads experiment 42 by id, returns ReadResult, full audit."""
    result = adapter.read_experiment_by_id(
        experiment_id=42,
        mapped_identity=_identity(),
        conversation_id="conv-1",
        request_id="req-1",
        provider="openai",
        model_id="gpt-4o-mini",
    )

    assert result.experiment_id == 42
    assert result.title == "PCR optimization for EF1a promoter"
    assert "<p>Test experiment body</p>" in result.body
    assert result.metadata == {"extra_fields": {}}

    assert audit.count() == 1
    row = audit.get(_last_action_id(audit))
    assert row is not None
    assert row["tool_name"] == "elabftw.read_experiment_by_id"
    assert row["policy_decision"] == "allow"
    assert "GET" in (row["api_call_summary_json"] or "")
    assert "42" in (row["context_refs_json"] or "")
    assert row["conversation_id"] == "conv-1"
    assert row["provider"] == "openai"


def test_read_by_id_unmapped_caller_denies(
    adapter: ElabftwReadAdapter,
    audit: AuditStore,
) -> None:
    """mapped_identity=None → UnmappedCaller, audit deny."""
    with pytest.raises(UnmappedCaller):
        adapter.read_experiment_by_id(
            experiment_id=42,
            mapped_identity=None,
        )
    assert audit.count() == 1
    row = audit.get(_last_action_id(audit))
    assert row is not None
    assert row["policy_decision"] == "deny"
    assert row["tool_name"] == "elabftw.read_experiment_by_id"
    assert "UNMAPPED_CALLER" in (row["error_json"] or "")


def test_read_by_id_policy_denied(
    audit: AuditStore,
    policy: PolicyEngine,
) -> None:
    """Kill switch on read_by_id → PolicyDenied, audit deny."""
    engine = PolicyEngine(kill_switches=("elabftw.read_experiment_by_id",))
    adapter = ElabftwReadAdapter(
        policy_engine=engine,
        audit_store=audit,
        client=StubElabftwClient(
            seeds={42: {"id": 42, "title": "A", "body": "", "metadata": None}}
        ),
    )
    with pytest.raises(PolicyDenied):
        adapter.read_experiment_by_id(
            experiment_id=42,
            mapped_identity=_identity(),
        )
    assert audit.count() == 1
    row = audit.get(_last_action_id(audit))
    assert row is not None
    assert row["policy_decision"] == "deny"
    assert "kill_switch" in (row["error_json"] or "")


def test_read_by_id_client_not_configured(
    audit: AuditStore,
    policy: PolicyEngine,
) -> None:
    """client=None → ElabftwAdapterError(reason='elabftw_not_configured')."""
    adapter = ElabftwReadAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=None,
    )
    with pytest.raises(ElabftwAdapterError) as exc:
        adapter.read_experiment_by_id(
            experiment_id=42,
            mapped_identity=_identity(),
        )
    assert exc.value.reason == "elabftw_not_configured"
    assert audit.count() == 1
    row = audit.get(_last_action_id(audit))
    assert row is not None
    assert "ELABFTW_NOT_CONFIGURED" in (row["error_json"] or "")


def test_read_by_id_client_error(
    adapter: ElabftwReadAdapter,
    audit: AuditStore,
) -> None:
    """Experiment id not in stub seeds → KeyError → ElabftwAdapterError."""
    # Experiment 999 is not in the stub client's seeds.
    with pytest.raises(ElabftwAdapterError) as exc:
        adapter.read_experiment_by_id(
            experiment_id=999,
            mapped_identity=_identity(),
        )
    assert exc.value.reason == "client_error"
    assert audit.count() == 1
    row = audit.get(_last_action_id(audit))
    assert row is not None
    assert row["policy_decision"] == "allow"
    assert "CLIENT_ERROR" in (row["error_json"] or "")


# --- HttpElabftwClient.get_upload_content binary regression -------------


def test_get_upload_content_must_return_raw_bytes_not_text():
    """Binary uploads remain bytes until the JSON response boundary."""
    from unittest.mock import MagicMock

    client = HttpElabftwClient(base_url="https://elab.example.org", api_key="k")
    original = bytes(range(256))  # full byte range incl. non-UTF-8

    mock_sess = MagicMock()
    mock_resp = MagicMock()
    mock_resp.content = original
    mock_resp.raise_for_status = MagicMock()
    mock_sess.get.return_value = mock_resp
    client._session = mock_sess

    result = client.get_upload_content("experiments", 42, 1)

    assert isinstance(result, bytes)
    assert result == original
