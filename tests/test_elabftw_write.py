"""Tests for the eLabFTW bounded write adapter (C09).

Acceptance checks (from the build plan):

    1. Approved amendment appears in eLabFTW (stub mutates the experiment body).
    2. Audit ID appears in eLabFTW-facing provenance (metadata.extra_fields).
    3. Attempted write to disallowed record (locked / archived) is rejected.

Additional coverage:

    * Approval token is consumed (single-use) — replay after amend is rejected.
    * Mismatched approval_args (different hash) is rejected.
    * Unmapped caller is rejected before any downstream write or approval consume.
    * Invalid / expired / missing context token is rejected before any write.
    * Policy denial (kill switch, tier cap) is rejected before approval consume.
    * Identity binding: token bound to user A cannot write as user B.
    * create_draft path: a new experiment is created with the approved args.
    * Attachment upload: amend with attachment writes both the amendment and
      the upload.
    * Audit row persists on every path (success AND deny).
    * Audit row requires persistence on the success path.
    * Provenance writeback fails-soft (does NOT block successful write).
"""

from __future__ import annotations

import datetime as _dt
import hashlib

import pytest

from lab_copilot_gateway.approval import (
    ApprovalRequest,
    ApprovalStore,
    compute_args_hash,
)
from lab_copilot_gateway.audit import AuditStore
from lab_copilot_gateway.elabftw import (
    AppendOnlyViolation,
    ContextTokenClaims,
    ElabftwAdapterError,
    ElabftwWriteAdapter,
    InvalidContextToken,
    StubElabftwClient,
    TOOL_NAME_AMEND,
    TOOL_NAME_DRAFT,
    TOOL_NAME_EDIT_SECTION,
    mint_context_token,
)
from lab_copilot_gateway.identity import MappedIdentity
from lab_copilot_gateway.policy import PolicyEngine, Tier


SECRET = b"test-secret-do-not-use-in-prod"


@pytest.fixture(autouse=True)
def _stable_token_secret(monkeypatch) -> None:
    """Pin the context-token HMAC secret for tests."""
    monkeypatch.setenv("LAB_COPILOT_TOKEN_SECRET", SECRET.decode("latin-1"))


# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def audit() -> AuditStore:
    s = AuditStore(db_path=":memory:")
    yield s
    s.close()


@pytest.fixture
def approval_store() -> ApprovalStore:
    s = ApprovalStore(db_path=":memory:")
    yield s
    s.close()


@pytest.fixture
def policy() -> PolicyEngine:
    return PolicyEngine()  # default: max_tier=6, no kill switches


@pytest.fixture
def stub_client() -> StubElabftwClient:
    """Pre-seed experiment 42 in normal (appendable) state."""
    return StubElabftwClient(
        seeds={
            42: {
                "id": 42,
                "title": "PCR optimization",
                "body": "<p>Original body</p>",
                "metadata": {"extra_fields": {}},
                "state": 1,  # normal
                "locked": 0,
                "uploads": [],
            },
            99: {  # not appendable: locked
                "id": 99,
                "title": "Signed-off experiment",
                "body": "<p>final</p>",
                "metadata": {"extra_fields": {}},
                "state": 1,
                "locked": 1,
                "uploads": [],
            },
            100: {  # not appendable: archived
                "id": 100,
                "title": "Archived",
                "body": "<p>archived</p>",
                "metadata": {"extra_fields": {}},
                "state": 2,  # archived
                "locked": 0,
                "uploads": [],
            },
        }
    )


@pytest.fixture
def write_adapter(
    policy: PolicyEngine,
    audit: AuditStore,
    approval_store: ApprovalStore,
    stub_client: StubElabftwClient,
) -> ElabftwWriteAdapter:
    return ElabftwWriteAdapter(
        policy_engine=policy,
        audit_store=audit,
        approval_store=approval_store,
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


def _issue_approval(
    store: ApprovalStore,
    *,
    tool_name: str,
    args: dict,
    target_record: str | None,
    mapped_elabftw_user_id: str = "elab-human-1",
    keycloak_subject: str = "kc-human-1",
    librechat_user_id: str = "lc-human-1",
) -> str:
    """Issue and return an approval_id bound to the given args."""
    req = ApprovalRequest(
        tool_name=tool_name,
        args_hash=compute_args_hash(args),
        target_record=target_record,
        tier=int(Tier.BOUNDED_WRITES),
        keycloak_subject=keycloak_subject,
        librechat_user_id=librechat_user_id,
        mapped_elabftw_user_id=mapped_elabftw_user_id,
        provider="openai",
        model_id="gpt-4o-mini",
        ttl_seconds=600,
    )
    approval_id, _expires = store.request(req)
    return approval_id


# --- acceptance: approved amendment appears in eLabFTW ----------------------


def test_amend_appends_section_to_experiment_body(
    write_adapter: ElabftwWriteAdapter,
    stub_client: StubElabftwClient,
    approval_store: ApprovalStore,
    audit: AuditStore,
) -> None:
    """Acceptance 1: approved amendment section appears in eLabFTW body.

    After the amend call, experiment 42's body must contain the amendment
    HTML wrapped in the section marker with `data-amendment="1"`.
    """
    approval_args = {"experiment_id": 42, "amendment_html": "<p>AI note</p>"}
    approval_id = _issue_approval(
        approval_store,
        tool_name=TOOL_NAME_AMEND,
        args=approval_args,
        target_record="elabftw:experiment:42",
    )
    result = write_adapter.amend_my_experiment_after_approval(
        context_token=_token(),
        approval_id=approval_id,
        approval_args=approval_args,
        mapped_identity=_identity(),
        amendment_html="<p>AI note</p>",
    )

    assert result.operation == "append_amendment"
    assert result.experiment_id == 42
    # Confirm the downstream record was actually mutated.
    body = stub_client.seeds[42]["body"]
    assert 'data-amendment="1"' in body
    assert "<p>AI note</p>" in body
    # Original content preserved.
    assert "<p>Original body</p>" in body
    # Audit row was persisted.
    assert audit.count() >= 1


# --- acceptance: audit ID appears in eLabFTW provenance --------------------


def test_amend_writes_provenance_into_metadata(
    write_adapter: ElabftwWriteAdapter,
    stub_client: StubElabftwClient,
    approval_store: ApprovalStore,
) -> None:
    """Acceptance 2: audit_action_id appears in experiment metadata.

    After the amend, the experiment's metadata.extra_fields must contain
    a `lab_copilot_audit_action_id` entry whose value matches what the
    adapter returned.
    """
    approval_args = {"experiment_id": 42, "amendment_html": "<p>X</p>"}
    approval_id = _issue_approval(
        approval_store,
        tool_name=TOOL_NAME_AMEND,
        args=approval_args,
        target_record="elabftw:experiment:42",
    )
    result = write_adapter.amend_my_experiment_after_approval(
        context_token=_token(),
        approval_id=approval_id,
        approval_args=approval_args,
        mapped_identity=_identity(),
        amendment_html="<p>X</p>",
    )

    metadata = stub_client.seeds[42]["metadata"] or {}
    extra_fields = metadata.get("extra_fields", {})
    assert "lab_copilot_audit_action_id" in extra_fields
    provenance_value = extra_fields["lab_copilot_audit_action_id"]["value"]
    assert result.audit_action_id in provenance_value


# --- acceptance: write to disallowed (locked/archived) record is rejected --


def test_amend_rejects_locked_experiment(
    write_adapter: ElabftwWriteAdapter,
    approval_store: ApprovalStore,
    audit: AuditStore,
) -> None:
    """Acceptance 3a: target experiment is locked → AppendOnlyViolation,
    audit row records the deny."""
    approval_args = {"experiment_id": 99, "amendment_html": "<p>late</p>"}
    approval_id = _issue_approval(
        approval_store,
        tool_name=TOOL_NAME_AMEND,
        args=approval_args,
        target_record="elabftw:experiment:99",
    )
    # Token bound to experiment 99 (locked).
    token = _token(_claims(experiment_id=99))

    with pytest.raises(AppendOnlyViolation) as exc_info:
        write_adapter.amend_my_experiment_after_approval(
            context_token=token,
            approval_id=approval_id,
            approval_args=approval_args,
            mapped_identity=_identity(),
            amendment_html="<p>late</p>",
        )
    assert exc_info.value.experiment_id == 99
    assert exc_info.value.locked is True
    # Audit row recorded the deny reason.
    rows = audit.count()
    assert rows >= 1


def test_amend_rejects_archived_experiment(
    write_adapter: ElabftwWriteAdapter,
    approval_store: ApprovalStore,
) -> None:
    """Acceptance 3b: target experiment state=archived → AppendOnlyViolation."""
    approval_args = {"experiment_id": 100, "amendment_html": "<p>late</p>"}
    approval_id = _issue_approval(
        approval_store,
        tool_name=TOOL_NAME_AMEND,
        args=approval_args,
        target_record="elabftw:experiment:100",
    )
    token = _token(_claims(experiment_id=100))

    with pytest.raises(AppendOnlyViolation) as exc_info:
        write_adapter.amend_my_experiment_after_approval(
            context_token=token,
            approval_id=approval_id,
            approval_args=approval_args,
            mapped_identity=_identity(),
            amendment_html="<p>late</p>",
        )
    assert exc_info.value.state == 2  # archived


def test_amend_append_only_violation_does_not_consume_approval(
    write_adapter: ElabftwWriteAdapter,
    approval_store: ApprovalStore,
) -> None:
    """When the append-only check rejects, the approval token must remain
    consumable — the gateway did not perform the write."""
    approval_args = {"experiment_id": 99, "amendment_html": "<p>late</p>"}
    approval_id = _issue_approval(
        approval_store,
        tool_name=TOOL_NAME_AMEND,
        args=approval_args,
        target_record="elabftw:experiment:99",
    )
    token = _token(_claims(experiment_id=99))

    with pytest.raises(AppendOnlyViolation):
        write_adapter.amend_my_experiment_after_approval(
            context_token=token,
            approval_id=approval_id,
            approval_args=approval_args,
            mapped_identity=_identity(),
            amendment_html="<p>late</p>",
        )

    # Approval row still has consumed_at=None.
    approval_record = approval_store.get(approval_id)
    assert approval_record is not None
    assert approval_record.consumed_at is None


# --- approval flow: single-use replay defense --------------------------------


def test_amend_consumes_approval_first_call_then_replay_rejected(
    write_adapter: ElabftwWriteAdapter,
    approval_store: ApprovalStore,
) -> None:
    """The first amend consumes the approval; a replay raises already_consumed."""
    approval_args = {"experiment_id": 42, "amendment_html": "<p>note</p>"}
    approval_id = _issue_approval(
        approval_store,
        tool_name=TOOL_NAME_AMEND,
        args=approval_args,
        target_record="elabftw:experiment:42",
    )

    # First amend consumes the approval.
    write_adapter.amend_my_experiment_after_approval(
        context_token=_token(),
        approval_id=approval_id,
        approval_args=approval_args,
        mapped_identity=_identity(),
        amendment_html="<p>note</p>",
    )

    # Replay → approval_store raises ApprovalAlreadyConsumed.
    with pytest.raises(ElabftwAdapterError) as exc_info:
        write_adapter.amend_my_experiment_after_approval(
            context_token=_token(),
            approval_id=approval_id,
            approval_args=approval_args,
            mapped_identity=_identity(),
            amendment_html="<p>note</p>",
        )
    assert "already_consumed" in exc_info.value.reason


# --- approval flow: mismatched args denied ----------------------------------


def test_amend_rejects_mismatched_approval_args(
    write_adapter: ElabftwWriteAdapter,
    approval_store: ApprovalStore,
) -> None:
    """Approval was issued for args A; caller presents args B → denied."""
    # Issue approval for arg A.
    original_args = {"experiment_id": 42, "amendment_html": "<p>original</p>"}
    approval_id = _issue_approval(
        approval_store,
        tool_name=TOOL_NAME_AMEND,
        args=original_args,
        target_record="elabftw:experiment:42",
    )

    # Try to consume with tampered args.
    tampered_args = {"experiment_id": 42, "amendment_html": "<p>TAMPERED</p>"}
    with pytest.raises(ElabftwAdapterError) as exc_info:
        write_adapter.amend_my_experiment_after_approval(
            context_token=_token(),
            approval_id=approval_id,
            approval_args=tampered_args,
            mapped_identity=_identity(),
            amendment_html="<p>TAMPERED</p>",
        )
    # ApprovalError surfaces as a mismatch reason in the adapter.
    assert (
        "mismatch" in exc_info.value.reason
        or "approval_denied" in exc_info.value.reason
    )


# --- deny paths: invalid token / unmapped caller / policy deny --------------


def test_amend_rejects_invalid_context_token_before_approval_consume(
    write_adapter: ElabftwWriteAdapter,
    approval_store: ApprovalStore,
) -> None:
    """Invalid context token must deny BEFORE consuming the approval token.

    Without this ordering, an attacker could present a fake identity to burn
    a legitimate user's approval.
    """
    approval_args = {"experiment_id": 42, "amendment_html": "<p>x</p>"}
    approval_id = _issue_approval(
        approval_store,
        tool_name=TOOL_NAME_AMEND,
        args=approval_args,
        target_record="elabftw:experiment:42",
    )

    with pytest.raises(InvalidContextToken):
        write_adapter.amend_my_experiment_after_approval(
            context_token="garbage.payload",
            approval_id=approval_id,
            approval_args=approval_args,
            mapped_identity=_identity(),
            amendment_html="<p>x</p>",
        )

    # Approval token was NOT consumed — token verify happens first.
    record = approval_store.get(approval_id)
    assert record is not None
    assert record.consumed_at is None


def test_amend_rejects_missing_context_token_before_approval_consume(
    write_adapter: ElabftwWriteAdapter,
    approval_store: ApprovalStore,
) -> None:
    """Missing context token must deny BEFORE consuming the approval token."""
    approval_args = {"experiment_id": 42, "amendment_html": "<p>x</p>"}
    approval_id = _issue_approval(
        approval_store,
        tool_name=TOOL_NAME_AMEND,
        args=approval_args,
        target_record="elabftw:experiment:42",
    )

    with pytest.raises(InvalidContextToken):
        write_adapter.amend_my_experiment_after_approval(
            context_token="",
            approval_id=approval_id,
            approval_args=approval_args,
            mapped_identity=_identity(),
            amendment_html="<p>x</p>",
        )

    record = approval_store.get(approval_id)
    assert record is not None
    assert record.consumed_at is None


def test_amend_rejects_unmapped_caller_before_approval_consume(
    write_adapter: ElabftwWriteAdapter,
    approval_store: ApprovalStore,
) -> None:
    """Unmapped caller (mapped_identity=None) must deny BEFORE consuming the
    approval token.  Same protection as invalid token."""
    approval_args = {"experiment_id": 42, "amendment_html": "<p>x</p>"}
    approval_id = _issue_approval(
        approval_store,
        tool_name=TOOL_NAME_AMEND,
        args=approval_args,
        target_record="elabftw:experiment:42",
    )

    with pytest.raises(ElabftwAdapterError) as exc_info:
        write_adapter.amend_my_experiment_after_approval(
            context_token=_token(),
            approval_id=approval_id,
            approval_args=approval_args,
            mapped_identity=None,  # unmapped
            amendment_html="<p>x</p>",
        )
    assert exc_info.value.reason == "unmapped_caller"

    # Approval token NOT consumed.
    record = approval_store.get(approval_id)
    assert record is not None
    assert record.consumed_at is None


def test_amend_rejects_token_user_mismatch_before_approval_consume(
    write_adapter: ElabftwWriteAdapter,
    approval_store: ApprovalStore,
) -> None:
    """Token bound to user A refused when caller resolves as user B."""
    approval_args = {"experiment_id": 42, "amendment_html": "<p>x</p>"}
    approval_id = _issue_approval(
        approval_store,
        tool_name=TOOL_NAME_AMEND,
        args=approval_args,
        target_record="elabftw:experiment:42",
    )

    # Token says elab-human-1; caller's identity says elab-someone-else.
    with pytest.raises(InvalidContextToken):
        write_adapter.amend_my_experiment_after_approval(
            context_token=_token(_claims(mapped_elabftw_user_id="elab-human-1")),
            approval_id=approval_id,
            approval_args=approval_args,
            mapped_identity=_identity(elab_user_id="attacker"),
            amendment_html="<p>x</p>",
        )

    record = approval_store.get(approval_id)
    assert record is not None
    assert record.consumed_at is None


def test_amend_rejects_with_kill_switch(
    stub_client: StubElabftwClient,
    audit: AuditStore,
    approval_store: ApprovalStore,
) -> None:
    """Kill switch on the amend tool denies via policy engine before
    any downstream HTTP or approval consume."""
    policy = PolicyEngine(kill_switches=(TOOL_NAME_AMEND,))
    adapter = ElabftwWriteAdapter(
        policy_engine=policy,
        audit_store=audit,
        approval_store=approval_store,
        client=stub_client,
    )

    approval_args = {"experiment_id": 42, "amendment_html": "<p>kill</p>"}
    approval_id = _issue_approval(
        approval_store,
        tool_name=TOOL_NAME_AMEND,
        args=approval_args,
        target_record="elabftw:experiment:42",
    )

    with pytest.raises(ElabftwAdapterError) as exc_info:
        adapter.amend_my_experiment_after_approval(
            context_token=_token(),
            approval_id=approval_id,
            approval_args=approval_args,
            mapped_identity=_identity(),
            amendment_html="<p>kill</p>",
        )
    assert "policy_denied" in exc_info.value.reason

    # Kill switch denied before approval consume.
    record = approval_store.get(approval_id)
    assert record is not None
    assert record.consumed_at is None


# --- audit row invariants ---------------------------------------------------


def test_amend_persists_audit_row_on_success(
    write_adapter: ElabftwWriteAdapter,
    approval_store: ApprovalStore,
    audit: AuditStore,
) -> None:
    """Success path persists an audit row with audit_action_id and approval_id."""
    approval_args = {"experiment_id": 42, "amendment_html": "<p>note</p>"}
    approval_id = _issue_approval(
        approval_store,
        tool_name=TOOL_NAME_AMEND,
        args=approval_args,
        target_record="elabftw:experiment:42",
    )

    before = audit.count()
    write_adapter.amend_my_experiment_after_approval(
        context_token=_token(),
        approval_id=approval_id,
        approval_args=approval_args,
        mapped_identity=_identity(),
        amendment_html="<p>note</p>",
    )
    after = audit.count()
    assert after == before + 1

    # Audit row's approval_id column must match the consumed approval.
    # AuditStore.get() returns 'elab-write-…' action_ids; scan for ours.
    # The audit row for the success path has tool_name=TOOL_NAME_AMEND.
    # We rely on count() since AuditStore.get() needs an action_id.


def test_amend_audit_persistence_required_on_success(
    stub_client: StubElabftwClient,
    approval_store: ApprovalStore,
) -> None:
    """If audit persistence fails on the SUCCESS path, raise — a write
    without an audit trail violates the gateway's safety contract."""

    class _FailingAudit(AuditStore):
        def append(self, record):  # type: ignore[override]
            raise RuntimeError("disk full")

    failing = _FailingAudit(db_path=":memory:")
    policy = PolicyEngine()
    adapter = ElabftwWriteAdapter(
        policy_engine=policy,
        audit_store=failing,
        approval_store=approval_store,
        client=stub_client,
    )

    approval_args = {"experiment_id": 42, "amendment_html": "<p>note</p>"}
    approval_id = _issue_approval(
        approval_store,
        tool_name=TOOL_NAME_AMEND,
        args=approval_args,
        target_record="elabftw:experiment:42",
    )

    with pytest.raises(ElabftwAdapterError) as exc_info:
        adapter.amend_my_experiment_after_approval(
            context_token=_token(),
            approval_id=approval_id,
            approval_args=approval_args,
            mapped_identity=_identity(),
            amendment_html="<p>note</p>",
        )
    assert exc_info.value.reason == "audit_persistence_failed"


# --- create_draft path ------------------------------------------------------


def test_draft_experiment_update_creates_new_experiment(
    write_adapter: ElabftwWriteAdapter,
    stub_client: StubElabftwClient,
    approval_store: ApprovalStore,
) -> None:
    """The draft_experiment_update tool creates a new experiment row.

    Append-only enforcement does NOT apply on create (no existing record is
    mutated).  The approval_args bind still applies.
    """
    approval_args = {"title": "AI-proposed draft"}
    approval_id = _issue_approval(
        approval_store,
        tool_name=TOOL_NAME_DRAFT,
        args=approval_args,
        target_record=None,  # no existing target on create
    )
    result = write_adapter.draft_experiment_update(
        context_token=_token(),  # any valid token; no target_experiment_id
        approval_id=approval_id,
        approval_args=approval_args,
        mapped_identity=_identity(),
    )

    assert result.operation == "create_draft"
    # Downstream created a NEW experiment (id from stub's auto-increment).
    new_id = result.experiment_id
    assert new_id in stub_client.seeds
    # The title is sourced FROM approval_args['title'], so it matches.
    assert stub_client.seeds[new_id]["title"] == "AI-proposed draft"


def test_draft_without_approval_args_hash_match_denies(
    write_adapter: ElabftwWriteAdapter,
    approval_store: ApprovalStore,
) -> None:
    """draft_experiment_update also enforces approval_args hash binding."""
    approval_id = _issue_approval(
        approval_store,
        tool_name=TOOL_NAME_DRAFT,
        args={"title": "approved title"},
        target_record=None,
    )
    with pytest.raises(ElabftwAdapterError) as exc_info:
        write_adapter.draft_experiment_update(
            context_token=_token(),
            approval_id=approval_id,
            approval_args={"title": "TAMPERED"},
            mapped_identity=_identity(),
        )
    assert (
        "mismatch" in exc_info.value.reason
        or "approval_denied" in exc_info.value.reason
    )


def test_draft_title_is_sourced_from_approval_args_not_a_separate_field(
    write_adapter: ElabftwWriteAdapter,
    stub_client: StubElabftwClient,
    approval_store: ApprovalStore,
) -> None:
    """C09 review blocker 2: the title that lands downstream must be the
    one inside the approval_args (which is hash-bound), NOT an override
    HTTP body field.

    Without this guarantee an attacker with a valid approval token for
    `{"title": "benign"}` could pass `target_title="malicious"` and the
    adapter would write the malicious title while the args hash still
    matched.  Resolved by removing the separate ``target_title`` parameter
    entirely — the title is fetched from approval_args inside the adapter.
    """
    approval_args = {"title": "approved-title"}
    approval_id = _issue_approval(
        approval_store,
        tool_name=TOOL_NAME_DRAFT,
        args=approval_args,
        target_record=None,
    )

    result = write_adapter.draft_experiment_update(
        context_token=_token(),
        approval_id=approval_id,
        approval_args=approval_args,  # title comes from here
        mapped_identity=_identity(),
        # Note: no target_title kwarg exists anymore — if the adapter
        # tried to use one it would be a TypeError.
    )

    new_id = result.experiment_id
    assert stub_client.seeds[new_id]["title"] == "approved-title"


def test_draft_without_title_in_approval_args_creates_untitled(
    write_adapter: ElabftwWriteAdapter,
    stub_client: StubElabftwClient,
    approval_store: ApprovalStore,
) -> None:
    """If approval_args doesn't carry a title, the adapter creates an
    experiment with an empty title — that's what was approved."""
    approval_args: dict = {}
    approval_id = _issue_approval(
        approval_store,
        tool_name=TOOL_NAME_DRAFT,
        args=approval_args,
        target_record=None,
    )
    result = write_adapter.draft_experiment_update(
        context_token=_token(),
        approval_id=approval_id,
        approval_args=approval_args,
        mapped_identity=_identity(),
    )
    new_id = result.experiment_id
    assert stub_client.seeds[new_id]["title"] == ""


# --- attachment upload ------------------------------------------------------


def test_amend_with_attachment_uploads_file(
    write_adapter: ElabftwWriteAdapter,
    stub_client: StubElabftwClient,
    approval_store: ApprovalStore,
) -> None:
    """Amend path with attachment_filename + data uploads the attachment
    to the same experiment as the amendment."""
    approval_args = {
        "experiment_id": 42,
        "amendment_html": "<p>amend</p>",
        "attachment_filename": "plot.png",
    }
    approval_id = _issue_approval(
        approval_store,
        tool_name=TOOL_NAME_AMEND,
        args=approval_args,
        target_record="elabftw:experiment:42",
    )

    result = write_adapter.amend_my_experiment_after_approval(
        context_token=_token(),
        approval_id=approval_id,
        approval_args=approval_args,
        mapped_identity=_identity(),
        amendment_html="<p>amend</p>",
        attachment_filename="plot.png",
        attachment_data=b"\x89PNG fake bytes",
        attachment_comment="AI-generated plot",
    )

    # The new upload landed on experiment 42.
    uploads = stub_client.seeds[42]["uploads"]
    assert len(uploads) == 1
    assert uploads[0]["real_name"] == "plot.png"
    assert result.downstream_id == 1  # first upload id


# --- provenance writeback is best-effort ------------------------------------


def test_provenance_writeback_failure_does_not_block_write(
    stub_client: StubElabftwClient,
    audit: AuditStore,
    approval_store: ApprovalStore,
) -> None:
    """If PATCH metadata fails after the body amend succeeded, the write
    must still return success (provenance is observability, not enforcement)."""

    class _BadMetadataClient(StubElabftwClient):
        def patch_experiment_metadata(self, experiment_id, metadata):  # type: ignore[override]
            raise RuntimeError("metadata endpoint down")

    bad_stub = _BadMetadataClient(
        seeds={
            42: {
                "id": 42,
                "title": "x",
                "body": "<p>y</p>",
                "metadata": {"extra_fields": {}},
                "state": 1,
                "locked": 0,
                "uploads": [],
            }
        }
    )
    policy = PolicyEngine()
    adapter = ElabftwWriteAdapter(
        policy_engine=policy,
        audit_store=audit,
        approval_store=approval_store,
        client=bad_stub,
    )

    approval_args = {"experiment_id": 42, "amendment_html": "<p>note</p>"}
    approval_id = _issue_approval(
        approval_store,
        tool_name=TOOL_NAME_AMEND,
        args=approval_args,
        target_record="elabftw:experiment:42",
    )

    # Body amend should still succeed, despite metadata PATCH failing.
    result = adapter.amend_my_experiment_after_approval(
        context_token=_token(),
        approval_id=approval_id,
        approval_args=approval_args,
        mapped_identity=_identity(),
        amendment_html="<p>note</p>",
    )

    # Body amendment was applied.
    assert "data-amendment" in bad_stub.seeds[42]["body"]
    # Return value reflects the audit_action_id (provenance was best-effort).
    assert result.audit_action_id


# --- client-error wrapping --------------------------------------------------


def test_amend_client_error_after_approval_consume_raises(
    stub_client: StubElabftwClient,
    audit: AuditStore,
    approval_store: ApprovalStore,
) -> None:
    """A downstream client error during amend is wrapped in
    ElabftwAdapterError(reason='client_error') and the approval has
    already been consumed (no refund)."""

    class _ExplodingClient(StubElabftwClient):
        def patch_experiment_body(self, experiment_id, body_html):  # type: ignore[override]
            raise RuntimeError("network down")

    exploding_stub = _ExplodingClient(
        seeds={
            42: {
                "id": 42,
                "title": "x",
                "body": "<p>y</p>",
                "metadata": {"extra_fields": {}},
                "state": 1,
                "locked": 0,
                "uploads": [],
            }
        }
    )
    policy = PolicyEngine()
    adapter = ElabftwWriteAdapter(
        policy_engine=policy,
        audit_store=audit,
        approval_store=approval_store,
        client=exploding_stub,
    )

    approval_args = {"experiment_id": 42, "amendment_html": "<p>note</p>"}
    approval_id = _issue_approval(
        approval_store,
        tool_name=TOOL_NAME_AMEND,
        args=approval_args,
        target_record="elabftw:experiment:42",
    )

    with pytest.raises(ElabftwAdapterError) as exc_info:
        adapter.amend_my_experiment_after_approval(
            context_token=_token(),
            approval_id=approval_id,
            approval_args=approval_args,
            mapped_identity=_identity(),
            amendment_html="<p>note</p>",
        )
    assert exc_info.value.reason == "client_error"

    # Approval was already consumed when the downstream call blew up.
    record = approval_store.get(approval_id)
    assert record is not None
    assert record.consumed_at is not None


# --- not configured ---------------------------------------------------------


def test_amend_without_client_denies_after_approval_consume(
    audit: AuditStore,
    approval_store: ApprovalStore,
) -> None:
    """A gateway with no LAB_COPILOT_ELABFTW_* env vars denies at the
    "not_configured" step.  The approval MUST be consumed first because the
    gateway evaluated the request as legitimate up to that point — there is
    no refund policy for a misconfigured gateway."""
    policy = PolicyEngine()
    adapter = ElabftwWriteAdapter(
        policy_engine=policy,
        audit_store=audit,
        approval_store=approval_store,
        client=None,
    )

    approval_args = {"experiment_id": 42, "amendment_html": "<p>note</p>"}
    approval_id = _issue_approval(
        approval_store,
        tool_name=TOOL_NAME_AMEND,
        args=approval_args,
        target_record="elabftw:experiment:42",
    )

    with pytest.raises(ElabftwAdapterError) as exc_info:
        adapter.amend_my_experiment_after_approval(
            context_token=_token(),
            approval_id=approval_id,
            approval_args=approval_args,
            mapped_identity=_identity(),
            amendment_html="<p>note</p>",
        )
    assert exc_info.value.reason == "elabftw_not_configured"


# --- singleton get/reset ----------------------------------------------------


def test_get_elabftw_write_adapter_returns_singleton() -> None:
    from lab_copilot_gateway.elabftw import (
        get_elabftw_write_adapter,
        reset_elabftw_write_adapter,
    )

    reset_elabftw_write_adapter()
    a1 = get_elabftw_write_adapter()
    a2 = get_elabftw_write_adapter()
    assert a1 is a2

    reset_elabftw_write_adapter(None)
    a3 = get_elabftw_write_adapter()
    assert a3 is not a1

    reset_elabftw_write_adapter(None)


# --- exception to_dict ------------------------------------------------------


def test_append_only_violation_carries_state_and_locked() -> None:
    """AppendOnlyViolation exposes experiment_id, state, locked for callers
    that want to surface a clean error message."""
    err = AppendOnlyViolation(experiment_id=42, state=2, locked=True)
    d = err.to_dict()
    assert d["reason"] == "append_only_violation"
    assert "state=2" in d["message"]
    assert "locked=True" in d["message"]
    assert err.experiment_id == 42
    assert err.state == 2
    assert err.locked is True


# --- amendment wrapping ----------------------------------------------------


def test_wrap_amendment_adds_section_marker() -> None:
    """The amendment is wrapped in an HTML section marker for downstream
    parsing and to visually distinguish AI-generated content."""
    wrapped = ElabftwWriteAdapter._wrap_amendment(
        existing_body="<p>original</p>", amendment_html="<p>note</p>"
    )
    assert 'data-source="lab-copilot-gateway"' in wrapped
    assert 'data-amendment="1"' in wrapped
    assert "<p>original</p>" in wrapped
    assert "<p>note</p>" in wrapped


def test_wrap_amendment_empty_amendment_returns_body_unchanged() -> None:
    """Empty amendment_html is a no-op (don't insert empty sections)."""
    body = "<p>preserve me</p>"
    assert ElabftwWriteAdapter._wrap_amendment(body, "") == body


def test_wrap_amendment_on_empty_body_creates_section_only() -> None:
    """When experiment has no body, amendment becomes the body."""
    wrapped = ElabftwWriteAdapter._wrap_amendment("", "<p>only amendment</p>")
    assert wrapped == (
        '<section data-source="lab-copilot-gateway" data-amendment="1">'
        "<p>only amendment</p>"
        "</section>"
    )


# --- C35 direct-edit tests --------------------------------------------------


def test_edit_section_replaces_body(
    write_adapter: ElabftwWriteAdapter,
    stub_client: StubElabftwClient,
    approval_store: ApprovalStore,
    audit: AuditStore,
) -> None:
    """C35 acceptance: approved edit_section replaces experiment body wholesale."""
    old_body = "OLD BODY"
    new_body = "NEW BODY"
    old_hash = hashlib.sha256(old_body.encode()).hexdigest()
    # Seed experiment 42 with the old body.
    stub_client.seeds[42]["body"] = old_body

    approval_args = {
        "experiment_id": 42,
        "old_body_hash": old_hash,
        "new_body": new_body,
    }
    approval_id = _issue_approval(
        approval_store,
        tool_name=TOOL_NAME_EDIT_SECTION,
        args=approval_args,
        target_record="elabftw:experiment:42",
    )
    result = write_adapter.edit_experiment_section(
        context_token=_token(),
        approval_id=approval_id,
        approval_args=approval_args,
        mapped_identity=_identity(),
    )

    assert result.operation == "edit_section"
    assert result.experiment_id == 42
    assert stub_client.seeds[42]["body"] == new_body
    assert audit.count() >= 1


def test_edit_section_rejects_body_drift(
    write_adapter: ElabftwWriteAdapter,
    stub_client: StubElabftwClient,
    approval_store: ApprovalStore,
    audit: AuditStore,
) -> None:
    """C35: if body changed between approval and execution, reject with drift reason."""
    original = "ORIGINAL"
    old_hash = hashlib.sha256(original.encode()).hexdigest()
    stub_client.seeds[42]["body"] = original

    approval_args = {
        "experiment_id": 42,
        "old_body_hash": old_hash,
        "new_body": "ATTEMPTED EDIT",
    }
    approval_id = _issue_approval(
        approval_store,
        tool_name=TOOL_NAME_EDIT_SECTION,
        args=approval_args,
        target_record="elabftw:experiment:42",
    )

    # Mutate the body after approval was issued — simulates a concurrent edit.
    stub_client.seeds[42]["body"] = "CHANGED BY SOMEONE ELSE"

    with pytest.raises(ElabftwAdapterError) as exc_info:
        write_adapter.edit_experiment_section(
            context_token=_token(),
            approval_id=approval_id,
            approval_args=approval_args,
            mapped_identity=_identity(),
        )
    assert exc_info.value.reason == "body_drift_since_approval"
    # Body was NOT patched — the concurrent edit is preserved.
    assert stub_client.seeds[42]["body"] == "CHANGED BY SOMEONE ELSE"
    # Approval WAS consumed (step 4, BEFORE drift check at step 6).
    record = approval_store.get(approval_id)
    assert record is not None
    assert record.consumed_at is not None
    # Audit row recorded the deny.
    assert audit.count() >= 1


def test_edit_section_rejects_locked_experiment(
    write_adapter: ElabftwWriteAdapter,
    approval_store: ApprovalStore,
    audit: AuditStore,
) -> None:
    """C35: direct-edit to a locked experiment raises AppendOnlyViolation."""
    approval_args = {
        "experiment_id": 99,
        "old_body_hash": "dummy",
        "new_body": "X",
    }
    approval_id = _issue_approval(
        approval_store,
        tool_name=TOOL_NAME_EDIT_SECTION,
        args=approval_args,
        target_record="elabftw:experiment:99",
    )
    token = _token(_claims(experiment_id=99))

    with pytest.raises(AppendOnlyViolation) as exc_info:
        write_adapter.edit_experiment_section(
            context_token=token,
            approval_id=approval_id,
            approval_args=approval_args,
            mapped_identity=_identity(),
        )
    assert exc_info.value.experiment_id == 99
    assert exc_info.value.locked is True
    # Approval NOT consumed (deny-before-consume invariant).
    record = approval_store.get(approval_id)
    assert record is not None
    assert record.consumed_at is None
    assert audit.count() >= 1


def test_edit_section_consumes_approval_first_call_then_replay_rejected(
    write_adapter: ElabftwWriteAdapter,
    stub_client: StubElabftwClient,
    approval_store: ApprovalStore,
) -> None:
    """C35: first edit_section consumes approval; replay with same args raises already_consumed."""
    old_body = "BODY"
    new_body = "AFTER EDIT"
    old_hash = hashlib.sha256(old_body.encode()).hexdigest()
    stub_client.seeds[42]["body"] = old_body

    approval_args = {
        "experiment_id": 42,
        "old_body_hash": old_hash,
        "new_body": new_body,
    }
    approval_id = _issue_approval(
        approval_store,
        tool_name=TOOL_NAME_EDIT_SECTION,
        args=approval_args,
        target_record="elabftw:experiment:42",
    )

    # First call — succeeds.
    write_adapter.edit_experiment_section(
        context_token=_token(),
        approval_id=approval_id,
        approval_args=approval_args,
        mapped_identity=_identity(),
    )
    assert stub_client.seeds[42]["body"] == new_body

    # Replay with the SAME approval_args — the approval is already consumed,
    # so the consume step rejects before the drift check runs.
    with pytest.raises(ElabftwAdapterError) as exc_info:
        write_adapter.edit_experiment_section(
            context_token=_token(),
            approval_id=approval_id,
            approval_args=approval_args,
            mapped_identity=_identity(),
        )
    assert "consum" in exc_info.value.reason


def test_edit_section_writes_provenance(
    write_adapter: ElabftwWriteAdapter,
    stub_client: StubElabftwClient,
    approval_store: ApprovalStore,
) -> None:
    """C35: after successful edit, metadata.extra_fields carries the audit id."""
    old_body = "ORIG"
    new_body = "EDITED"
    old_hash = hashlib.sha256(old_body.encode()).hexdigest()
    stub_client.seeds[42]["body"] = old_body

    approval_args = {
        "experiment_id": 42,
        "old_body_hash": old_hash,
        "new_body": new_body,
    }
    approval_id = _issue_approval(
        approval_store,
        tool_name=TOOL_NAME_EDIT_SECTION,
        args=approval_args,
        target_record="elabftw:experiment:42",
    )
    result = write_adapter.edit_experiment_section(
        context_token=_token(),
        approval_id=approval_id,
        approval_args=approval_args,
        mapped_identity=_identity(),
    )

    metadata = stub_client.seeds[42]["metadata"] or {}
    extra_fields = metadata.get("extra_fields", {})
    assert "lab_copilot_audit_action_id" in extra_fields
    provenance_value = extra_fields["lab_copilot_audit_action_id"]["value"]
    assert result.audit_action_id in provenance_value


def test_edit_section_missing_old_body_hash_rejects(
    write_adapter: ElabftwWriteAdapter,
    stub_client: StubElabftwClient,
    approval_store: ApprovalStore,
    audit: AuditStore,
) -> None:
    """C35: old_body_hash is REQUIRED — missing/empty raises missing_old_body_hash."""
    stub_client.seeds[42]["body"] = "WHATEVER"

    approval_args = {
        "experiment_id": 42,
        "new_body": "X",
    }
    approval_id = _issue_approval(
        approval_store,
        tool_name=TOOL_NAME_EDIT_SECTION,
        args=approval_args,
        target_record="elabftw:experiment:42",
    )

    with pytest.raises(ElabftwAdapterError) as exc_info:
        write_adapter.edit_experiment_section(
            context_token=_token(),
            approval_id=approval_id,
            approval_args=approval_args,
            mapped_identity=_identity(),
        )
    assert exc_info.value.reason == "missing_old_body_hash"
    # Body was NOT patched — the original seed body is preserved.
    assert stub_client.seeds[42]["body"] == "WHATEVER"
    # Audit row recorded the deny.
    assert audit.count() >= 1
