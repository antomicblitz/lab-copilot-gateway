"""Tests for the deterministic validation bundle (T9).

Acceptance checks:

    1. All blockers pass → ``is_approved=True``.
    2. Final construct missing → blocker, ``is_approved=False``.
    3. GenBank missing → blocker.
    4. Annotation not attempted → warning, still approved.
    5. Protocol fallback → warning, still approved.
    6. Primer heterodimer not checked → warning.
    7. ``to_dict()`` produces correct JSON shape.
    8. Empty cloning run → multiple blockers.
    9. Full happy-path cloning run → all checks pass, approved.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from lab_copilot_gateway.approval import (
    ApprovalRequest,
    ApprovalStore,
    compute_args_hash,
)
from lab_copilot_gateway.audit import AuditStore
from lab_copilot_gateway.elabftw import (
    ContextTokenClaims,
    StubElabftwClient,
    mint_context_token,
)
from lab_copilot_gateway.identity import MappedIdentity
from lab_copilot_gateway.opencloning import OpenCloningAdapter
from lab_copilot_gateway.opencloning_validation import (
    ValidationBundle,
    ValidationBundleBuilder,
    ValidationCheck,
    ValidationLevel,
    build_validation_bundle,
)
from lab_copilot_gateway.policy import PolicyEngine, Tier

SECRET = b"test-secret-do-not-use-in-prod"

GENBANK = """LOCUS       pDemo                   12 bp    DNA     circular SYN 01-JUL-2026
FEATURES             Location/Qualifiers
     source          1..12
     CDS             1..9
                     /label="demo"
ORIGIN
        1 atgcgatcgatc
//
"""

GENBANK_NO_FEATURES = """LOCUS       plain                   8 bp    DNA     linear SYN 01-JUL-2026
FEATURES             Location/Qualifiers
ORIGIN
        1 atgcatgc
//
"""


# --- fixtures ----------------------------------------------------------------


@pytest.fixture(autouse=True)
def _stable_token_secret(monkeypatch) -> None:
    """Pin the HMAC secret so tokens minted in tests verify in the adapter."""
    monkeypatch.setenv("LAB_COPILOT_TOKEN_SECRET", SECRET.decode("latin-1"))


@pytest.fixture
def audit() -> AuditStore:
    s = AuditStore(db_path=":memory:")
    yield s
    s.close()


@pytest.fixture
def policy() -> PolicyEngine:
    return PolicyEngine()


@pytest.fixture
def approval_store() -> ApprovalStore:
    s = ApprovalStore(db_path=":memory:")
    yield s
    s.close()


@pytest.fixture
def elabftw_stub() -> StubElabftwClient:
    return StubElabftwClient(
        seeds={
            42: {
                "id": 42,
                "title": "Cloning experiment",
                "body": "<p>Original body</p>",
                "metadata": {"extra_fields": {}},
                "state": 1,
                "locked": 0,
                "uploads": [],
            },
        }
    )


@pytest.fixture
def writeback_adapter(
    policy: PolicyEngine,
    audit: AuditStore,
    approval_store: ApprovalStore,
    elabftw_stub: StubElabftwClient,
) -> OpenCloningAdapter:
    return OpenCloningAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=None,
        elabftw_client=elabftw_stub,
        approval_store=approval_store,
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


def _token(claims: ContextTokenClaims | None = None) -> str:
    return mint_context_token(claims or _claims())


def _identity(
    *,
    elab_user_id: str = "elab-human-1",
    kc_subject: str = "kc-human-1",
    lc_user_id: str = "lc-human-1",
) -> MappedIdentity:
    return MappedIdentity(
        keycloak_subject=kc_subject,
        librechat_user_id=lc_user_id,
        elabftw_user_id=elab_user_id,
        elabftw_team_ids=["team-1"],
    )


def _issue_writeback_approval(
    store: ApprovalStore,
    *,
    args: dict,
    target_record: str | None = "42",
) -> str:
    req = ApprovalRequest(
        tool_name="opencloning.writeback_artifact",
        args_hash=compute_args_hash(args),
        target_record=target_record,
        tier=int(Tier.BOUNDED_WRITES),
        keycloak_subject="kc-human-1",
        librechat_user_id="lc-human-1",
        mapped_elabftw_user_id="elab-human-1",
        provider="openai",
        model_id="gpt-4o-mini",
        ttl_seconds=600,
    )
    approval_id, _ = store.request(req)
    return approval_id


# --- a happy-path cloning run fixture ----------------------------------------


def _happy_path_sequences() -> list[dict]:
    """Sequences from a typical Gibson assembly run."""
    return [
        {
            "id": 1,
            "type": "TextFileSequence",
            "name": "insert",
            "sequence_file_format": "genbank",
            "file_content": GENBANK,
            "length": 12,
            "circular": False,
            "feature_count": 2,
        },
        {
            "id": 2,
            "type": "TextFileSequence",
            "name": "vector",
            "sequence_file_format": "genbank",
            "file_content": GENBANK,
            "length": 12,
            "circular": True,
            "feature_count": 2,
        },
        {
            "id": 3,
            "type": "TextFileSequence",
            "name": "final_construct",
            "sequence_file_format": "genbank",
            "file_content": GENBANK,
            "length": 24,
            "circular": True,
            "feature_count": 4,
        },
    ]


def _happy_path_sources() -> list[dict]:
    """Sources for a Gibson assembly — seq 3 is the leaf (final construct)."""
    return [
        {"id": 1, "type": "UploadedFileSource", "input": []},
        {"id": 2, "type": "UploadedFileSource", "input": []},
        {
            "id": 3,
            "type": "GibsonAssemblySource",
            "input": [
                {"sequence": 1},
                {"sequence": 2},
            ],
        },
    ]


def _happy_path_primers() -> list[dict]:
    return [
        {"id": 1, "name": "fwd", "sequence": "ATGCATGCAT", "tm": 58.5},
        {"id": 2, "name": "rev", "sequence": "GCATGCATGC", "tm": 60.2},
    ]


def _happy_path_artifacts() -> dict:
    """Normalized artifacts payload with one GenBank artifact."""
    return {
        "summary": "Produced 1 circular final construct (24 bp).",
        "artifacts": [
            {
                "artifact_id": "oc-artifact-1",
                "kind": "genbank",
                "filename": "final_construct.gb",
                "size_bytes": len(GENBANK.encode()),
            }
        ],
        "warnings": [],
        "blocking_errors": [],
        "provenance": {},
    }


def _happy_path_step_manifests() -> list[dict]:
    """Step manifests including annotation and primer analysis."""
    return [
        {
            "schema": "opencloning_step_manifest_v1",
            "run_id": "run-1",
            "step_id": "step-1",
            "operation_type": "sequence_import",
            "endpoint": "/read_from_file",
            "status": "succeeded",
            "preview_ref": "preview-aaa",
        },
        {
            "schema": "opencloning_step_manifest_v1",
            "run_id": "run-1",
            "step_id": "step-2",
            "operation_type": "assembly",
            "endpoint": "/gibson_assembly",
            "status": "succeeded",
            "preview_ref": "preview-bbb",
        },
        {
            "schema": "opencloning_step_manifest_v1",
            "run_id": "run-1",
            "step_id": "step-3",
            "operation_type": "annotation",
            "endpoint": "/annotate/plannotate",
            "status": "succeeded",
            "preview_ref": None,
        },
        {
            "schema": "opencloning_step_manifest_v1",
            "run_id": "run-1",
            "step_id": "step-4",
            "operation_type": "primer_design",
            "endpoint": "/primer_details",
            "status": "succeeded",
            "preview_ref": None,
        },
    ]


def _happy_path_protocol_result() -> dict:
    return {
        "query_method": "gibson_assembly",
        "query_reagent": "NEBuilder HiFi DNA Assembly Master Mix",
        "matches": [{"item_id": 10, "title": "NEBuilder Gibson"}],
        "best_match": {"item_id": 10, "title": "NEBuilder Gibson"},
        "fallback_used": False,
        "warnings": [],
    }


# --- ValidationBundle / ValidationCheck unit tests --------------------------


def test_validation_level_values() -> None:
    """ValidationLevel has blocker and warning values."""
    assert ValidationLevel.BLOCKER.value == "blocker"
    assert ValidationLevel.WARNING.value == "warning"


def test_validation_check_dataclass() -> None:
    """ValidationCheck stores all fields correctly."""
    check = ValidationCheck(
        check_id="test_check",
        label="Test Check",
        level=ValidationLevel.BLOCKER,
        status="pass",
        detail="All good",
    )
    assert check.check_id == "test_check"
    assert check.label == "Test Check"
    assert check.level == ValidationLevel.BLOCKER
    assert check.status == "pass"
    assert check.detail == "All good"
    assert check.data is None


def test_bundle_add_check_routes_blockers_and_warnings() -> None:
    """add_check routes failed blockers to blockers, non-pass warnings to warnings."""
    bundle = ValidationBundle(run_id="run-1")
    bundle.add_check(
        ValidationCheck(
            check_id="b1",
            label="Blocker 1",
            level=ValidationLevel.BLOCKER,
            status="fail",
            detail="failed",
        )
    )
    bundle.add_check(
        ValidationCheck(
            check_id="b2",
            label="Blocker 2",
            level=ValidationLevel.BLOCKER,
            status="pass",
            detail="ok",
        )
    )
    bundle.add_check(
        ValidationCheck(
            check_id="w1",
            label="Warning 1",
            level=ValidationLevel.WARNING,
            status="not_checked",
            detail="not checked",
        )
    )
    bundle.add_check(
        ValidationCheck(
            check_id="w2",
            label="Warning 2",
            level=ValidationLevel.WARNING,
            status="pass",
            detail="ok",
        )
    )
    bundle.add_check(
        ValidationCheck(
            check_id="s1",
            label="Skipped",
            level=ValidationLevel.WARNING,
            status="skipped",
            detail="not applicable",
        )
    )
    bundle.finalize()

    assert len(bundle.checks) == 5
    assert len(bundle.blockers) == 1
    assert bundle.blockers[0].check_id == "b1"
    assert len(bundle.warnings) == 1
    assert bundle.warnings[0].check_id == "w1"
    assert bundle.is_approved is False


def test_bundle_finalize_approved_when_no_blockers() -> None:
    """finalize sets is_approved=True when no blockers."""
    bundle = ValidationBundle(run_id="run-1")
    bundle.add_check(
        ValidationCheck(
            check_id="b1",
            label="Blocker",
            level=ValidationLevel.BLOCKER,
            status="pass",
            detail="ok",
        )
    )
    bundle.finalize()
    assert bundle.is_approved is True
    assert len(bundle.blockers) == 0


# --- ValidationBundleBuilder check methods -----------------------------------


def test_builder_check_final_construct_pass() -> None:
    """check_final_construct passes when has_final_sequence is True."""
    builder = ValidationBundleBuilder(run_id="run-1")
    builder.check_final_construct(True, sequence_id=3, sequence_length=5421)
    bundle = builder.build()
    assert bundle.is_approved is True
    check = bundle.checks[0]
    assert check.check_id == "final_construct_exists"
    assert check.status == "pass"
    assert "id=3" in check.detail
    assert "5421 bp" in check.detail


def test_builder_check_final_construct_fail() -> None:
    """check_final_construct fails (blocker) when has_final_sequence is False."""
    builder = ValidationBundleBuilder(run_id="run-1")
    builder.check_final_construct(False)
    bundle = builder.build()
    assert bundle.is_approved is False
    assert len(bundle.blockers) == 1
    assert bundle.blockers[0].check_id == "final_construct_exists"


def test_builder_check_genbank_artifact_pass_and_fail() -> None:
    """check_genbank_artifact passes/fails as a blocker."""
    builder = ValidationBundleBuilder(run_id="run-1")
    builder.check_genbank_artifact(True)
    builder.check_genbank_artifact(False)
    bundle = builder.build()
    assert len(bundle.checks) == 2
    assert bundle.checks[0].status == "pass"
    assert bundle.checks[1].status == "fail"
    assert bundle.is_approved is False


def test_builder_check_annotation_states() -> None:
    """check_annotation: pass / fail / not_checked."""
    # Attempted + completed → pass
    b = ValidationBundleBuilder(run_id="r")
    b.check_annotation(True, True)
    bundle = b.build()
    assert bundle.checks[0].status == "pass"
    assert len(bundle.warnings) == 0

    # Attempted + not completed → fail (warning)
    b = ValidationBundleBuilder(run_id="r")
    b.check_annotation(True, False)
    bundle = b.build()
    assert bundle.checks[0].status == "fail"
    assert len(bundle.warnings) == 1
    assert bundle.is_approved is True  # warnings don't block

    # Not attempted → not_checked (warning)
    b = ValidationBundleBuilder(run_id="r")
    b.check_annotation(False, False)
    bundle = b.build()
    assert bundle.checks[0].status == "not_checked"
    assert len(bundle.warnings) == 1


def test_builder_check_history_json() -> None:
    """check_history_json: pass when generated, warning when not."""
    b = ValidationBundleBuilder(run_id="r")
    b.check_history_json(True)
    bundle = b.build()
    assert bundle.checks[0].status == "pass"

    b = ValidationBundleBuilder(run_id="r")
    b.check_history_json(False)
    bundle = b.build()
    assert bundle.checks[0].status == "not_checked"
    assert len(bundle.warnings) == 1


def test_builder_check_primer_details() -> None:
    """check_primer_details: pass / fail / skipped."""
    # All details available
    b = ValidationBundleBuilder(run_id="r")
    b.check_primer_details(2, 2)
    bundle = b.build()
    assert bundle.checks[0].status == "pass"

    # Some missing
    b = ValidationBundleBuilder(run_id="r")
    b.check_primer_details(3, 1)
    bundle = b.build()
    assert bundle.checks[0].status == "fail"
    assert "2 of 3" in bundle.checks[0].detail

    # No primers → skipped
    b = ValidationBundleBuilder(run_id="r")
    b.check_primer_details(0, 0)
    bundle = b.build()
    assert bundle.checks[0].status == "skipped"
    assert len(bundle.warnings) == 0  # skipped is neither blocker nor warning


def test_builder_check_primer_heterodimer() -> None:
    """check_primer_heterodimer: pass / fail / skipped."""
    # All pairs checked
    b = ValidationBundleBuilder(run_id="r")
    b.check_primer_heterodimer(1, 1)
    bundle = b.build()
    assert bundle.checks[0].status == "pass"

    # Not all checked
    b = ValidationBundleBuilder(run_id="r")
    b.check_primer_heterodimer(3, 1)
    bundle = b.build()
    assert bundle.checks[0].status == "fail"
    assert "2 of 3" in bundle.checks[0].detail

    # No pairs → skipped
    b = ValidationBundleBuilder(run_id="r")
    b.check_primer_heterodimer(0, 0)
    bundle = b.build()
    assert bundle.checks[0].status == "skipped"


def test_builder_check_protocol_lookup() -> None:
    """check_protocol_lookup: pass / fail (fallback) / not_checked."""
    # Found, no fallback
    b = ValidationBundleBuilder(run_id="r")
    b.check_protocol_lookup(True, False)
    bundle = b.build()
    assert bundle.checks[0].status == "pass"

    # Fallback used
    b = ValidationBundleBuilder(run_id="r")
    b.check_protocol_lookup(False, True)
    bundle = b.build()
    assert bundle.checks[0].status == "fail"
    assert len(bundle.warnings) == 1

    # Not performed
    b = ValidationBundleBuilder(run_id="r")
    b.check_protocol_lookup(False, False)
    bundle = b.build()
    assert bundle.checks[0].status == "not_checked"


def test_builder_check_preview_refs() -> None:
    """check_preview_refs: pass when previews exist, not_checked when none."""
    b = ValidationBundleBuilder(run_id="r")
    b.check_preview_refs(3, 5)
    bundle = b.build()
    assert bundle.checks[0].status == "pass"
    assert "3 of 5" in bundle.checks[0].detail

    b = ValidationBundleBuilder(run_id="r")
    b.check_preview_refs(0, 5)
    bundle = b.build()
    assert bundle.checks[0].status == "not_checked"
    assert len(bundle.warnings) == 1


def test_builder_check_notebook_and_audit_drafts() -> None:
    """check_notebook_summary and check_audit_report warn when not generated."""
    b = ValidationBundleBuilder(run_id="r")
    b.check_notebook_summary(False)
    b.check_audit_report(False)
    bundle = b.build()
    assert len(bundle.warnings) == 2
    assert bundle.is_approved is True  # warnings don't block

    b = ValidationBundleBuilder(run_id="r")
    b.check_notebook_summary(True)
    b.check_audit_report(True)
    bundle = b.build()
    assert len(bundle.warnings) == 0


# --- to_dict() shape ---------------------------------------------------------


def test_to_dict_produces_correct_json_shape() -> None:
    """to_dict() returns the opencloning_validation_bundle_v1 schema."""
    builder = ValidationBundleBuilder(run_id="run-abc")
    builder.check_final_construct(True, sequence_id=5, sequence_length=1000)
    builder.check_genbank_artifact(True)
    builder.check_annotation(False, False)
    bundle = builder.build()
    d = bundle.to_dict()

    assert d["schema"] == "opencloning_validation_bundle_v1"
    assert d["run_id"] == "run-abc"
    assert d["is_approved"] is True
    assert d["blocker_count"] == 0
    assert d["warning_count"] == 1
    assert len(d["checks"]) == 3
    assert len(d["blockers"]) == 0
    assert len(d["warnings"]) == 1

    check = d["checks"][0]
    assert check["check_id"] == "final_construct_exists"
    assert check["label"] == "Final construct exists"
    assert check["level"] == "blocker"
    assert check["status"] == "pass"
    assert isinstance(check["detail"], str)

    warning = d["warnings"][0]
    assert warning["check_id"] == "post_assembly_annotation"
    assert isinstance(warning["label"], str)
    assert isinstance(warning["detail"], str)


# --- build_validation_bundle() standalone function --------------------------


def test_build_bundle_empty_run_multiple_blockers() -> None:
    """An empty cloning run produces multiple blockers (no construct, no artifact)."""
    bundle = build_validation_bundle(
        run_id="empty",
        sequences=[],
        sources=[],
        primers=[],
        artifacts=None,
        step_manifests=[],
        protocol_result=None,
        preview_refs_count=0,
    )
    assert bundle.is_approved is False
    assert len(bundle.blockers) == 2  # final_construct + genbank_artifact
    blocker_ids = {b.check_id for b in bundle.blockers}
    assert "final_construct_exists" in blocker_ids
    assert "genbank_artifact_exists" in blocker_ids


def test_build_bundle_happy_path_all_pass() -> None:
    """A full happy-path cloning run passes all checks and is approved."""
    bundle = build_validation_bundle(
        run_id="happy",
        sequences=_happy_path_sequences(),
        sources=_happy_path_sources(),
        primers=_happy_path_primers(),
        artifacts=_happy_path_artifacts(),
        step_manifests=_happy_path_step_manifests(),
        protocol_result=_happy_path_protocol_result(),
        preview_refs_count=2,
    )
    assert bundle.is_approved is True
    assert len(bundle.blockers) == 0
    # All 10 checks present.
    check_ids = {c.check_id for c in bundle.checks}
    assert check_ids == {
        "final_construct_exists",
        "genbank_artifact_exists",
        "post_assembly_annotation",
        "history_json_attempted",
        "primer_details",
        "primer_heterodimer",
        "protocol_lookup",
        "preview_refs",
        "notebook_summary_draft",
        "audit_report_draft",
    }


def test_build_bundle_final_construct_missing_blocks() -> None:
    """Missing final construct → blocker, is_approved=False."""
    bundle = build_validation_bundle(
        run_id="no-construct",
        sequences=[],  # no sequences → no final construct
        sources=_happy_path_sources(),
        primers=_happy_path_primers(),
        artifacts=_happy_path_artifacts(),
        step_manifests=_happy_path_step_manifests(),
        protocol_result=_happy_path_protocol_result(),
        preview_refs_count=2,
    )
    assert bundle.is_approved is False
    assert any(b.check_id == "final_construct_exists" for b in bundle.blockers)


def test_build_bundle_genbank_missing_blocks() -> None:
    """Missing GenBank artifact → blocker, is_approved=False."""
    bundle = build_validation_bundle(
        run_id="no-genbank",
        sequences=_happy_path_sequences(),
        sources=_happy_path_sources(),
        primers=_happy_path_primers(),
        artifacts=None,  # no artifact
        step_manifests=_happy_path_step_manifests(),
        protocol_result=_happy_path_protocol_result(),
        preview_refs_count=2,
    )
    assert bundle.is_approved is False
    assert any(b.check_id == "genbank_artifact_exists" for b in bundle.blockers)


def test_build_bundle_annotation_not_attempted_warning() -> None:
    """Annotation not attempted → warning, still approved (blockers pass)."""
    # Use sequences without features and no annotation step manifest.
    sequences = [
        {
            "id": 1,
            "file_content": GENBANK_NO_FEATURES,
            "length": 8,
            "feature_count": 0,
        }
    ]
    bundle = build_validation_bundle(
        run_id="no-annotation",
        sequences=sequences,
        sources=[],
        primers=[],
        artifacts=_happy_path_artifacts(),
        step_manifests=[],  # no annotation manifest
        protocol_result=None,
        preview_refs_count=0,
    )
    # Blockers pass (construct + artifact exist).
    assert bundle.is_approved is True
    # Annotation warning present.
    annotation_check = next(
        c for c in bundle.checks if c.check_id == "post_assembly_annotation"
    )
    assert annotation_check.status == "not_checked"
    assert any(w.check_id == "post_assembly_annotation" for w in bundle.warnings)


def test_build_bundle_annotation_detected_from_features() -> None:
    """Annotation is detected from sequence features when no step manifests."""
    bundle = build_validation_bundle(
        run_id="annotated",
        sequences=_happy_path_sequences(),  # final construct has feature_count=4
        sources=_happy_path_sources(),
        primers=[],
        artifacts=_happy_path_artifacts(),
        step_manifests=[],  # no annotation manifest, but features exist
        protocol_result=None,
        preview_refs_count=0,
    )
    annotation_check = next(
        c for c in bundle.checks if c.check_id == "post_assembly_annotation"
    )
    assert annotation_check.status == "pass"


def test_build_bundle_protocol_fallback_warning() -> None:
    """Protocol fallback → warning, still approved."""
    bundle = build_validation_bundle(
        run_id="fallback",
        sequences=_happy_path_sequences(),
        sources=_happy_path_sources(),
        primers=_happy_path_primers(),
        artifacts=_happy_path_artifacts(),
        step_manifests=_happy_path_step_manifests(),
        protocol_result={
            "query_method": "gibson_assembly",
            "query_reagent": None,
            "matches": [],
            "best_match": None,
            "fallback_used": True,
            "warnings": ["no approved protocol"],
        },
        preview_refs_count=2,
    )
    assert bundle.is_approved is True
    protocol_check = next(c for c in bundle.checks if c.check_id == "protocol_lookup")
    assert protocol_check.status == "fail"
    assert any(w.check_id == "protocol_lookup" for w in bundle.warnings)


def test_build_bundle_primer_heterodimer_not_checked_warning() -> None:
    """Primer heterodimer not checked → warning."""
    bundle = build_validation_bundle(
        run_id="no-hetero",
        sequences=_happy_path_sequences(),
        sources=_happy_path_sources(),
        primers=_happy_path_primers(),  # 2 primers → 1 pair
        artifacts=_happy_path_artifacts(),
        step_manifests=[],  # no primer analysis manifest
        protocol_result=None,
        preview_refs_count=0,
    )
    hetero_check = next(c for c in bundle.checks if c.check_id == "primer_heterodimer")
    assert hetero_check.status == "fail"
    assert any(w.check_id == "primer_heterodimer" for w in bundle.warnings)


def test_build_bundle_primer_heterodimer_detected_from_manifest() -> None:
    """Heterodimer is detected from step manifests with primer_details endpoint."""
    bundle = build_validation_bundle(
        run_id="hetero-ok",
        sequences=_happy_path_sequences(),
        sources=_happy_path_sources(),
        primers=_happy_path_primers(),
        artifacts=_happy_path_artifacts(),
        step_manifests=_happy_path_step_manifests(),  # has primer_design manifest
        protocol_result=None,
        preview_refs_count=2,
    )
    hetero_check = next(c for c in bundle.checks if c.check_id == "primer_heterodimer")
    assert hetero_check.status == "pass"


def test_build_bundle_finds_leaf_as_final_construct() -> None:
    """The final construct is the leaf sequence (not an input to any source)."""
    # seq 1 and 2 are inputs to source 3; seq 3 is the leaf.
    bundle = build_validation_bundle(
        run_id="leaf",
        sequences=_happy_path_sequences(),
        sources=_happy_path_sources(),
        primers=[],
        artifacts=_happy_path_artifacts(),
        step_manifests=[],
        protocol_result=None,
        preview_refs_count=0,
    )
    final_check = next(
        c for c in bundle.checks if c.check_id == "final_construct_exists"
    )
    assert final_check.status == "pass"
    assert "id=3" in final_check.detail


def test_build_bundle_artifact_from_approval_args_content() -> None:
    """GenBank artifact detected from approval_args with artifact_content."""
    bundle = build_validation_bundle(
        run_id="inline",
        sequences=_happy_path_sequences(),
        sources=_happy_path_sources(),
        primers=[],
        artifacts={"artifact_content": GENBANK, "artifact_filename": "x.gb"},
        step_manifests=[],
        protocol_result=None,
        preview_refs_count=0,
    )
    genbank_check = next(
        c for c in bundle.checks if c.check_id == "genbank_artifact_exists"
    )
    assert genbank_check.status == "pass"


def test_build_bundle_artifact_from_approval_args_id() -> None:
    """GenBank artifact detected from approval_args with artifact_id (bundle ref)."""
    bundle = build_validation_bundle(
        run_id="bundle-ref",
        sequences=_happy_path_sequences(),
        sources=_happy_path_sources(),
        primers=[],
        artifacts={"artifact_id": "oc-artifact-1", "plan_id": "plan-1"},
        step_manifests=[],
        protocol_result=None,
        preview_refs_count=0,
    )
    genbank_check = next(
        c for c in bundle.checks if c.check_id == "genbank_artifact_exists"
    )
    assert genbank_check.status == "pass"


def test_build_bundle_primer_details_partial() -> None:
    """Primer details: some primers lack Tm → warning."""
    primers = [
        {"id": 1, "name": "fwd", "sequence": "ATGC", "tm": 58.0},
        {"id": 2, "name": "rev", "sequence": "GCAT"},  # no tm
    ]
    bundle = build_validation_bundle(
        run_id="partial-primers",
        sequences=_happy_path_sequences(),
        sources=_happy_path_sources(),
        primers=primers,
        artifacts=_happy_path_artifacts(),
        step_manifests=_happy_path_step_manifests(),
        protocol_result=None,
        preview_refs_count=2,
    )
    details_check = next(c for c in bundle.checks if c.check_id == "primer_details")
    assert details_check.status == "fail"
    assert "1 of 2" in details_check.detail


def test_build_bundle_notebook_and_audit_drafts_always_warn() -> None:
    """Notebook summary and audit report drafts always warn (T10 not done)."""
    bundle = build_validation_bundle(
        run_id="drafts",
        sequences=_happy_path_sequences(),
        sources=_happy_path_sources(),
        primers=_happy_path_primers(),
        artifacts=_happy_path_artifacts(),
        step_manifests=_happy_path_step_manifests(),
        protocol_result=_happy_path_protocol_result(),
        preview_refs_count=2,
    )
    notebook_check = next(
        c for c in bundle.checks if c.check_id == "notebook_summary_draft"
    )
    audit_check = next(c for c in bundle.checks if c.check_id == "audit_report_draft")
    assert notebook_check.status == "not_checked"
    assert audit_check.status == "not_checked"
    assert bundle.is_approved is True  # warnings don't block


# --- adapter integration -----------------------------------------------------


def test_adapter_build_validation_bundle_from_strategy_state(
    writeback_adapter: OpenCloningAdapter,
) -> None:
    """The adapter's build_validation_bundle reads from _strategy_* state."""
    writeback_adapter._strategy_sequences = _happy_path_sequences()
    writeback_adapter._strategy_sources = _happy_path_sources()
    writeback_adapter._strategy_primers = _happy_path_primers()

    bundle_dict = writeback_adapter.build_validation_bundle(
        run_id="adapter-run",
        approval_args={"artifact_content": GENBANK, "artifact_filename": "x.gb"},
        step_manifests=_happy_path_step_manifests(),
        protocol_result=_happy_path_protocol_result(),
        preview_refs_count=2,
    )
    assert bundle_dict["schema"] == "opencloning_validation_bundle_v1"
    assert bundle_dict["run_id"] == "adapter-run"
    assert bundle_dict["is_approved"] is True
    assert bundle_dict["blocker_count"] == 0


def test_adapter_build_validation_bundle_empty_state_blocks(
    writeback_adapter: OpenCloningAdapter,
) -> None:
    """Empty adapter state → blockers (no construct, no artifact)."""
    writeback_adapter._strategy_sequences = []
    writeback_adapter._strategy_sources = []
    writeback_adapter._strategy_primers = []

    bundle_dict = writeback_adapter.build_validation_bundle(
        run_id="empty-adapter",
        approval_args={},
    )
    assert bundle_dict["is_approved"] is False
    assert bundle_dict["blocker_count"] == 2


# --- writeback integration ---------------------------------------------------


def test_writeback_includes_validation_bundle_in_result(
    writeback_adapter: OpenCloningAdapter,
    approval_store: ApprovalStore,
) -> None:
    """writeback_artifact includes validation_bundle in the result dict."""
    # Populate strategy state so the bundle has content.
    writeback_adapter._strategy_sequences = _happy_path_sequences()
    writeback_adapter._strategy_sources = _happy_path_sources()
    writeback_adapter._strategy_primers = _happy_path_primers()

    approval_args = {
        "artifact_filename": "construct.gb",
        "artifact_content": GENBANK,
        "artifact_comment": "OpenCloning design artifact",
    }
    approval_id = _issue_writeback_approval(approval_store, args=approval_args)

    result = writeback_adapter.writeback_artifact(
        context_token=_token(),
        approval_id=approval_id,
        approval_args=approval_args,
        mapped_identity=_identity(),
    )
    assert "validation_bundle" in result.result
    bundle = result.result["validation_bundle"]
    assert bundle["schema"] == "opencloning_validation_bundle_v1"
    assert bundle["is_approved"] is True
    assert bundle["blocker_count"] == 0


def test_writeback_validation_bundle_blocks_when_no_construct(
    writeback_adapter: OpenCloningAdapter,
    approval_store: ApprovalStore,
) -> None:
    """When no final construct exists, the bundle reports blockers."""
    # No strategy state → no final construct.
    writeback_adapter._strategy_sequences = []
    writeback_adapter._strategy_sources = []
    writeback_adapter._strategy_primers = []

    approval_args = {
        "artifact_filename": "construct.gb",
        "artifact_content": GENBANK,
        "artifact_comment": "OpenCloning design artifact",
    }
    approval_id = _issue_writeback_approval(approval_store, args=approval_args)

    result = writeback_adapter.writeback_artifact(
        context_token=_token(),
        approval_id=approval_id,
        approval_args=approval_args,
        mapped_identity=_identity(),
    )
    bundle = result.result["validation_bundle"]
    assert bundle["is_approved"] is False
    assert bundle["blocker_count"] >= 1
    blocker_ids = {b["check_id"] for b in bundle["blockers"]}
    assert "final_construct_exists" in blocker_ids
