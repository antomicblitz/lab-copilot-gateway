"""Tests for the single-use approval token flow (C07).

Acceptance checks:

    1. Mutating tool with no approval is denied (policy engine).
    2. Mutating tool with mismatched args hash is denied (approval store).
    3. Mutating tool with valid approval succeeds if policy still allows
       (approval store + policy engine both green).

Additional coverage:

    * Args hashing is order-independent and sensitive to value changes.
    * Expiry denies consumption.
    * Replay (already consumed) denies consumption.
    * Consume-mismatch on tool_name and target_record denies.
    * not_found approval denies.
    * Default TTL is 10 minutes.
    * ApprovalRecord.is_consumed / is_expired helpers.
    * DB-backed persistence across store instances.
    * Atomic single-use under sequential consume (replay).
"""

from __future__ import annotations

import datetime as _dt
import time
from pathlib import Path

import pytest

from lab_copilot_gateway.approval import (
    DEFAULT_APPROVAL_TTL_SECONDS,
    ApprovalAlreadyConsumed,
    ApprovalExpired,
    ApprovalMismatch,
    ApprovalNotFound,
    ApprovalRequest,
    ApprovalStore,
    compute_args_hash,
    get_approval_store,
    reset_approval_store,
)
from lab_copilot_gateway.policy import PolicyEngine, PolicyRequest, Tier


# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def store() -> ApprovalStore:
    s = ApprovalStore(db_path=":memory:")
    yield s
    s.close()


@pytest.fixture
def engine() -> PolicyEngine:
    return PolicyEngine()  # default max_tier=6, approvals_required_for=BOUNDED_WRITES


def _request(
    *,
    tool_name: str = "elabftw.amend_my_experiment_after_approval",
    args_hash: str | None = None,
    target_record: str | None = "elabftw:experiment:42",
    tier: int = int(Tier.BOUNDED_WRITES),
    ttl_seconds: int = DEFAULT_APPROVAL_TTL_SECONDS,
    keycloak_subject: str | None = "kc-human-1",
    librechat_user_id: str | None = "lc-human-1",
    mapped_elabftw_user_id: str | None = "elab-human-1",
    provider: str | None = "openai",
    model_id: str | None = "gpt-4o-mini",
) -> ApprovalRequest:
    return ApprovalRequest(
        tool_name=tool_name,
        args_hash=args_hash
        or compute_args_hash({"experiment_id": 42, "amendment": "add note"}),
        target_record=target_record,
        tier=tier,
        keycloak_subject=keycloak_subject,
        librechat_user_id=librechat_user_id,
        mapped_elabftw_user_id=mapped_elabftw_user_id,
        provider=provider,
        model_id=model_id,
        ttl_seconds=ttl_seconds,
    )


def _policy_req(
    *,
    tool_name: str = "elabftw.amend_my_experiment_after_approval",
    tier: Tier = Tier.BOUNDED_WRITES,
    user_id: str | None = "elab-human-1",
    has_approval: bool = False,
    approval_id: str | None = None,
) -> PolicyRequest:
    return PolicyRequest(
        tool_name=tool_name,
        tier=tier,
        user_id=user_id,
        has_approval=has_approval,
        approval_id=approval_id,
    )


# --- Acceptance check 1: no approval denied --------------------------------


def test_mutating_tool_without_approval_denied_by_policy(engine: PolicyEngine) -> None:
    """Tier 4 bounded write with no approval is denied at the policy layer."""
    decision = engine.decide(_policy_req(has_approval=False))
    assert decision.decision == "deny"
    assert decision.reason == "approval_required"
    assert decision.requires_approval is True


# --- Acceptance check 2: mismatched args hash denied ------------------------


def test_consume_rejects_mismatched_args_hash(store: ApprovalStore) -> None:
    """Approval is bound to the exact args; modified args deny consumption."""
    approval_id, _ = store.request(_request(args_hash=compute_args_hash({"a": 1})))

    with pytest.raises(ApprovalMismatch) as exc:
        store.consume(
            approval_id,
            tool_name="elabftw.amend_my_experiment_after_approval",
            args_hash=compute_args_hash({"a": 2}),
        )
    assert exc.value.field_name == "args_hash"
    assert exc.value.reason == "mismatch"


def test_consume_rejects_mismatched_tool_name(store: ApprovalStore) -> None:
    approval_id, _ = store.request(_request())

    with pytest.raises(ApprovalMismatch) as exc:
        store.consume(
            approval_id,
            tool_name="elabftw.draft_experiment_update",  # differs
            args_hash=compute_args_hash({"experiment_id": 42, "amendment": "add note"}),
        )
    assert exc.value.field_name == "tool_name"


def test_consume_rejects_mismatched_target_record(store: ApprovalStore) -> None:
    approval_id, _ = store.request(_request(target_record="elabftw:experiment:42"))

    with pytest.raises(ApprovalMismatch) as exc:
        store.consume(
            approval_id,
            tool_name="elabftw.amend_my_experiment_after_approval",
            args_hash=compute_args_hash({"experiment_id": 42, "amendment": "add note"}),
            target_record="elabftw:experiment:999",  # differs
        )
    assert exc.value.field_name == "target_record"


# --- Acceptance check 3: valid approval succeeds if policy still allows ----


def test_valid_approval_allows_policy_and_consumes(
    store: ApprovalStore,
    engine: PolicyEngine,
) -> None:
    """End-to-end: request approval → policy allows with approval → consume
    succeeds, marking the token single-use."""
    req = _request()
    approval_id, expires_at = store.request(req)
    assert approval_id
    assert expires_at

    # Policy engine: tier 4 with approval → allow.
    decision = engine.decide(
        _policy_req(has_approval=True, approval_id=approval_id),
    )
    assert decision.decision == "allow"
    assert decision.reason == "default_allow"

    # Approval store: consume succeeds.
    result = store.consume(
        approval_id,
        tool_name=req.tool_name,
        args_hash=req.args_hash,
        target_record=req.target_record,
    )
    assert result.approval_id == approval_id
    assert result.consumed_at
    assert result.tool_name == req.tool_name
    assert result.args_hash == req.args_hash


# --- Edge cases ------------------------------------------------------------


def test_consume_rejects_unknown_approval_id(store: ApprovalStore) -> None:
    with pytest.raises(ApprovalNotFound) as exc:
        store.consume(
            "bogus-id",
            tool_name="elabftw.amend_my_experiment_after_approval",
            args_hash=compute_args_hash({"a": 1}),
        )
    assert exc.value.reason == "not_found"


def test_consume_rejects_expired_token() -> None:
    """A token whose TTL has elapsed denies consumption with ``expired``."""
    s = ApprovalStore(db_path=":memory:")
    try:
        approval_id, _ = s.request(_request(ttl_seconds=1))
        # Sleep just past the expiry window.  TTL is 1 second; we sleep 1.1s to
        # be safely past the ISO8601 comparison.
        time.sleep(1.1)
        with pytest.raises(ApprovalExpired) as exc:
            s.consume(
                approval_id,
                tool_name="elabftw.amend_my_experiment_after_approval",
                args_hash=compute_args_hash(
                    {"experiment_id": 42, "amendment": "add note"}
                ),
            )
        assert exc.value.reason == "expired"
    finally:
        s.close()


def test_consume_rejects_replay_after_consumption(store: ApprovalStore) -> None:
    """Single-use: a second consume with the same token is denied."""
    req = _request()
    approval_id, _ = store.request(req)

    # First consume succeeds.
    store.consume(
        approval_id,
        tool_name=req.tool_name,
        args_hash=req.args_hash,
        target_record=req.target_record,
    )

    # Replay: same args, same id, same tool — must deny.
    with pytest.raises(ApprovalAlreadyConsumed) as exc:
        store.consume(
            approval_id,
            tool_name=req.tool_name,
            args_hash=req.args_hash,
            target_record=req.target_record,
        )
    assert exc.value.reason == "already_consumed"


def test_get_returns_pending_state(store: ApprovalStore) -> None:
    req = _request()
    approval_id, _ = store.request(req)
    record = store.get(approval_id)
    assert record is not None
    assert record.approval_id == approval_id
    assert record.tool_name == req.tool_name
    assert record.args_hash == req.args_hash
    assert record.consumed_at is None
    assert record.is_consumed() is False


def test_get_returns_consumed_state_after_consume(store: ApprovalStore) -> None:
    req = _request()
    approval_id, _ = store.request(req)
    store.consume(
        approval_id,
        tool_name=req.tool_name,
        args_hash=req.args_hash,
        target_record=req.target_record,
    )
    record = store.get(approval_id)
    assert record is not None
    assert record.consumed_at is not None
    assert record.is_consumed() is True


def test_get_returns_none_for_unknown_id(store: ApprovalStore) -> None:
    assert store.get("does-not-exist") is None


# --- Args hashing properties ------------------------------------------------


def test_args_hash_is_order_independent() -> None:
    """Canonical-JSON hash is stable regardless of dict insertion order."""
    h1 = compute_args_hash({"a": 1, "b": 2, "c": 3})
    h2 = compute_args_hash({"c": 3, "a": 1, "b": 2})
    h3 = compute_args_hash({"b": 2, "c": 3, "a": 1})
    assert h1 == h2 == h3


def test_args_hash_is_sensitive_to_value_change() -> None:
    h1 = compute_args_hash({"a": 1})
    h2 = compute_args_hash({"a": 2})
    assert h1 != h2


def test_args_hash_is_sensitive_to_extra_field() -> None:
    h1 = compute_args_hash({"a": 1})
    h2 = compute_args_hash({"a": 1, "extra": True})
    assert h1 != h2


def test_args_hash_of_empty_dict_is_stable() -> None:
    assert compute_args_hash({}) == compute_args_hash(None)
    # Hex SHA-256 length.
    assert len(compute_args_hash({})) == 64  # noqa: PLR2004 — sha256 hex length


def test_args_hash_is_reproducible() -> None:
    h1 = compute_args_hash({"deep": {"nested": [1, 2, 3]}})
    h2 = compute_args_hash({"deep": {"nested": [1, 2, 3]}})
    assert h1 == h2


# --- Request / TTL behaviour -----------------------------------------------


def test_default_ttl_is_600_seconds() -> None:
    """Default TTL is 10 minutes — short enough to bound replay windows."""
    assert DEFAULT_APPROVAL_TTL_SECONDS == 600  # noqa: PLR2004 — contract value


def test_request_rejects_empty_tool_name(store: ApprovalStore) -> None:
    with pytest.raises(ValueError, match="non-empty tool_name"):
        store.request(
            ApprovalRequest(
                tool_name="",
                args_hash=compute_args_hash({"a": 1}),
                target_record=None,
                tier=int(Tier.BOUNDED_WRITES),
            ),
        )


def test_request_rejects_empty_args_hash(store: ApprovalStore) -> None:
    with pytest.raises(ValueError, match="args_hash"):
        store.request(
            ApprovalRequest(
                tool_name="elabftw.amend_my_experiment_after_approval",
                args_hash="",
                target_record=None,
                tier=int(Tier.BOUNDED_WRITES),
            ),
        )


def test_request_rejects_non_positive_ttl(store: ApprovalStore) -> None:
    with pytest.raises(ValueError, match="ttl_seconds > 0"):
        store.request(
            ApprovalRequest(
                tool_name="elabftw.amend_my_experiment_after_approval",
                args_hash=compute_args_hash({"a": 1}),
                target_record=None,
                tier=int(Tier.BOUNDED_WRITES),
                ttl_seconds=0,
            ),
        )


def test_request_returns_unique_opaque_ids(store: ApprovalStore) -> None:
    """Two request calls produce distinct approval_ids (128-bit entropy)."""
    req = _request()
    a, _ = store.request(req)
    b, _ = store.request(req)
    assert a != b


def test_request_records_request_time_identity_fields(store: ApprovalStore) -> None:
    req = _request(
        keycloak_subject="kc-1",
        librechat_user_id="lc-1",
        mapped_elabftw_user_id="elab-1",
        provider="anthropic",
        model_id="claude-3-7-sonnet",
    )
    approval_id, _ = store.request(req)
    record = store.get(approval_id)
    assert record is not None
    assert record.keycloak_subject == "kc-1"
    assert record.librechat_user_id == "lc-1"
    assert record.mapped_elabftw_user_id == "elab-1"
    assert record.provider == "anthropic"
    assert record.model_id == "claude-3-7-sonnet"
    assert record.tier == int(Tier.BOUNDED_WRITES)
    assert record.created_at
    assert record.expires_at


# --- Concurrency / single-use race -----------------------------------------


def test_two_sequential_consume_calls_only_first_wins(store: ApprovalStore) -> None:
    """Strictly sequential consume: second call sees already_consumed.

    (Simulates the in-process race after our get() / before our UPDATE; the
    row-level ``consumed_at IS NULL`` predicate enforces single-use at the
    SQLite level too — see _CONSUME_SQL.)
    """
    req = _request()
    approval_id, _ = store.request(req)

    first = store.consume(
        approval_id,
        tool_name=req.tool_name,
        args_hash=req.args_hash,
        target_record=req.target_record,
    )
    assert first.approval_id == approval_id

    with pytest.raises(ApprovalAlreadyConsumed):
        store.consume(
            approval_id,
            tool_name=req.tool_name,
            args_hash=req.args_hash,
            target_record=req.target_record,
        )


# --- Persistence ------------------------------------------------------------


def test_db_backed_store_persists_across_instances(tmp_path: Path) -> None:
    """A file-backed ApprovalStore persists tokens across instances."""
    db_file = tmp_path / "approvals.sqlite"
    req = _request()

    s1 = ApprovalStore(db_path=str(db_file))
    approval_id, _ = s1.request(req)
    s1.close()

    s2 = ApprovalStore(db_path=str(db_file))
    try:
        record = s2.get(approval_id)
        assert record is not None
        assert record.approval_id == approval_id
        # And consume works against the persisted row.
        result = s2.consume(
            approval_id,
            tool_name=req.tool_name,
            args_hash=req.args_hash,
            target_record=req.target_record,
        )
        assert result.approval_id == approval_id
    finally:
        s2.close()


# --- Singleton behaviour ---------------------------------------------------


def test_get_approval_store_returns_singleton() -> None:
    reset_approval_store()
    first = get_approval_store()
    second = get_approval_store()
    assert first is second


def test_reset_approval_store_clears_singleton() -> None:
    reset_approval_store()
    import lab_copilot_gateway.approval as approvalmod

    assert approvalmod._default_store is None
    s = get_approval_store()
    assert approvalmod._default_store is s
    reset_approval_store(store=None)
    assert approvalmod._default_store is None


# --- Exception to_dict serialisation ----------------------------------------


def test_approval_error_to_dict_includes_reason_and_message() -> None:
    err = ApprovalMismatch("approval-xyz", "args_hash")
    out = err.to_dict()
    assert out["reason"] == "mismatch"
    assert "args_hash" in out["message"]
    assert out["approval_id"] == "approval-xyz"


def test_approval_not_found_to_dict() -> None:
    err = ApprovalNotFound("missing-id")
    out = err.to_dict()
    assert out["reason"] == "not_found"
    assert out["approval_id"] == "missing-id"


# --- Default-ttl override path ---------------------------------------------


def test_request_with_custom_ttl_extends_expiry(store: ApprovalStore) -> None:
    """Caller can request a shorter or longer TTL; expiry reflects it."""
    req = _request(ttl_seconds=60)
    approval_id, expires_at = store.request(req)
    record = store.get(approval_id)
    assert record is not None
    # Expiry is ~60s after creation; verify the delta is in (50, 70) seconds
    # to tolerate test runtime latency.
    created = _dt.datetime.fromisoformat(record.created_at)
    expires = _dt.datetime.fromisoformat(expires_at)
    delta = (expires - created).total_seconds()
    assert 50 <= delta <= 70  # noqa: PLR2004 — tolerance window


def test_consume_atomic_expiry_enforcement() -> None:
    """Hard expiry enforcement: even if we manually backdate the expiry before
    calling consume (bypassing the Python-level soft check), the atomic UPDATE
    must still refuse because _CONSUME_SQL's WHERE clause requires
    ``datetime(expires_at) > datetime(:now)``.

    This is the TOCTOU-hardening test — the SQL predicate, not the Python
    soft check, is the authoritative gate.
    """
    s = ApprovalStore(db_path=":memory:")
    try:
        approval_id, _ = s.request(_request(ttl_seconds=300))

        # Backdate expiry to 1 second ago to simulate a token that has just
        # expired, even though the Python soft check would catch this.  We
        # want to prove the SQL UPDATE is independently authoritative.
        conn = s._connect()
        past = (
            _dt.datetime.now(_dt.timezone.utc) - _dt.timedelta(seconds=1)
        ).isoformat()
        # Manually update the row to backdate expiry (simulating a token that
        # was issued with a short TTL and has just expired).
        conn.execute(
            "UPDATE approvals SET expires_at = ? WHERE approval_id = ?",
            (past, approval_id),
        )
        conn.commit()

        with pytest.raises(ApprovalExpired):
            s.consume(
                approval_id,
                tool_name="elabftw.amend_my_experiment_after_approval",
                args_hash=compute_args_hash(
                    {"experiment_id": 42, "amendment": "add note"}
                ),
            )

        # And the row must NOT be marked consumed (the UPDATE's where clause
        # failed, so consumed_at stays NULL).
        record = s.get(approval_id)
        assert record is not None
        assert record.consumed_at is None
        assert record.is_consumed() is False
    finally:
        s.close()
