"""Tests for the execution-plan schema v0 module (C38)."""

from __future__ import annotations

import json

import pytest

from lab_copilot_gateway.plan import (
    Plan,
    PlanRiskTier,
    PlanStep,
    PlanValidationError,
    USER_VISIBLE_FIELDS,
    compute_plan_hash,
    fixture_autonomous,
    fixture_batch_edit,
    fixture_hardware,
    fixture_opencloning,
    fixture_simple_edit,
    validate_plan,
    validate_or_raise,
)


# =============================================================================
# PlanRiskTier
# =============================================================================


def test_risk_tier_values() -> None:
    """All 5 tiers have correct integer values."""
    assert int(PlanRiskTier.SINGLE_RECORD_EDIT) == 1
    assert int(PlanRiskTier.BATCH_EDIT) == 2
    assert int(PlanRiskTier.COMPUTATIONAL) == 3
    assert int(PlanRiskTier.HARDWARE) == 4
    assert int(PlanRiskTier.AUTONOMOUS) == 5


def test_risk_tier_labels() -> None:
    """Label property returns lowercase name."""
    assert PlanRiskTier.SINGLE_RECORD_EDIT.label == "single_record_edit"
    assert PlanRiskTier.BATCH_EDIT.label == "batch_edit"
    assert PlanRiskTier.COMPUTATIONAL.label == "computational"
    assert PlanRiskTier.HARDWARE.label == "hardware"
    assert PlanRiskTier.AUTONOMOUS.label == "autonomous"


def test_risk_tier_approval_modes() -> None:
    """Approval mode mapping matches the tier boundaries."""
    # Tier 1 → lightweight_preview
    assert PlanRiskTier.SINGLE_RECORD_EDIT.approval_mode == "lightweight_preview"
    # Tiers 2–3 → formal_change_set
    assert PlanRiskTier.BATCH_EDIT.approval_mode == "formal_change_set"
    assert PlanRiskTier.COMPUTATIONAL.approval_mode == "formal_change_set"
    # Tiers 4–5 → strong_execution
    assert PlanRiskTier.HARDWARE.approval_mode == "strong_execution"
    assert PlanRiskTier.AUTONOMOUS.approval_mode == "strong_execution"


# =============================================================================
# PlanStep
# =============================================================================


def test_plan_step_to_dict_minimal() -> None:
    """Only tool_name, no optional fields → dict has only tool_name."""
    step = PlanStep(tool_name="elabftw.read_current_experiment")
    result = step.to_dict()
    assert result == {"tool_name": "elabftw.read_current_experiment"}


def test_plan_step_to_dict_full() -> None:
    """All fields populated → dict has all keys."""
    step = PlanStep(
        tool_name="elabftw.edit_experiment_section",
        record_id="elabftw:experiment:42",
        args={"old_body_hash": "abc", "new_body": "<p>hello</p>"},
        preview_type="diff",
    )
    result = step.to_dict()
    assert set(result.keys()) == {"tool_name", "record_id", "args", "preview_type"}
    assert result["tool_name"] == "elabftw.edit_experiment_section"
    assert result["record_id"] == "elabftw:experiment:42"
    assert result["args"] == {"old_body_hash": "abc", "new_body": "<p>hello</p>"}
    assert result["preview_type"] == "diff"


def test_plan_step_from_dict_roundtrip() -> None:
    """to_dict → from_dict → equal."""
    step = PlanStep(
        tool_name="elabftw.edit_experiment_section",
        record_id="elabftw:experiment:42",
        args={"old_body_hash": "abc", "new_body": "<p>hello</p>"},
        preview_type="diff",
    )
    result = PlanStep.from_dict(step.to_dict())
    assert result == step


# =============================================================================
# Plan serialization + hashing
# =============================================================================


def test_plan_to_dict_has_approval_subdict() -> None:
    """to_dict includes "approval" with "mode" and "required"."""
    plan = fixture_simple_edit()
    d = plan.to_dict()
    assert "approval" in d
    assert isinstance(d["approval"], dict)
    assert d["approval"]["mode"] == PlanRiskTier.SINGLE_RECORD_EDIT.approval_mode
    assert d["approval"]["required"] is True


def test_plan_from_dict_roundtrip_preserves_hash() -> None:
    """to_dict → from_dict → hash matches original."""
    plan = fixture_simple_edit()
    d = plan.to_dict()
    rebuilt = Plan.from_dict(d)
    assert rebuilt.hash() == plan.hash()


def test_plan_hash_is_stable() -> None:
    """Same plan constructed twice → same hash."""
    a = fixture_simple_edit()
    b = fixture_simple_edit()
    assert a.hash() == b.hash()


def test_plan_hash_changes_on_content_change() -> None:
    """Change intent → different hash."""
    plan = fixture_simple_edit()
    # Create an otherwise-identical plan with a different intent
    modified = Plan(
        plan_id=plan.plan_id,
        intent="Different intent string",
        anchor=plan.anchor,
        risk_tier=plan.risk_tier,
        summary=plan.summary,
        reads=plan.reads,
        writes=plan.writes,
        artifacts=plan.artifacts,
        approval_required=plan.approval_required,
        rollback=plan.rollback,
    )
    assert modified.hash() != plan.hash()


def test_plan_canonical_json_is_sorted() -> None:
    """canonical_json has sorted keys (verify by checking the string)."""
    plan = fixture_simple_edit()
    cjson = plan.canonical_json()
    d = json.loads(cjson)
    # Build expected canonical JSON with sort_keys
    expected = json.dumps(d, sort_keys=True, separators=(",", ":"))
    assert cjson == expected


def test_compute_plan_hash_matches_method() -> None:
    """compute_plan_hash(plan) == plan.hash()"""
    plan = fixture_batch_edit()
    assert compute_plan_hash(plan) == plan.hash()


# =============================================================================
# Fixtures validate
# =============================================================================


def test_fixture_simple_edit_validates() -> None:
    """fixture_simple_edit passes validation."""
    assert validate_plan(fixture_simple_edit()) == []


def test_fixture_batch_edit_validates() -> None:
    """fixture_batch_edit passes validation."""
    assert validate_plan(fixture_batch_edit()) == []


def test_fixture_opencloning_validates() -> None:
    """fixture_opencloning passes validation."""
    assert validate_plan(fixture_opencloning()) == []


def test_fixture_hardware_validates() -> None:
    """fixture_hardware passes validation."""
    assert validate_plan(fixture_hardware()) == []


def test_fixture_autonomous_validates() -> None:
    """fixture_autonomous passes validation (C29)."""
    assert validate_plan(fixture_autonomous()) == []


def test_all_fixtures_have_distinct_hashes() -> None:
    """No two fixtures share the same hash."""
    hashes = {
        fixture_simple_edit().hash(),
        fixture_batch_edit().hash(),
        fixture_opencloning().hash(),
        fixture_hardware().hash(),
        fixture_autonomous().hash(),
    }
    assert len(hashes) == 5


# =============================================================================
# Validation rules
# =============================================================================


def _make_plan(**overrides: object) -> Plan:
    """Construct a valid plan from fixture_simple_edit with overrides."""
    base = fixture_simple_edit()
    kwargs: dict[str, object] = {
        "plan_id": base.plan_id,
        "intent": base.intent,
        "anchor": base.anchor,
        "risk_tier": base.risk_tier,
        "summary": base.summary,
        "reads": base.reads,
        "writes": base.writes,
        "artifacts": base.artifacts,
        "approval_required": base.approval_required,
        "rollback": base.rollback,
    }
    kwargs.update(overrides)
    return Plan(**kwargs)  # type: ignore[arg-type]


def test_validate_empty_plan_id() -> None:
    """plan_id="" → error."""
    plan = _make_plan(plan_id="")
    errors = validate_plan(plan)
    assert any("plan_id" in e for e in errors)


def test_validate_empty_intent() -> None:
    """intent="" → error."""
    plan = _make_plan(intent="")
    errors = validate_plan(plan)
    assert any("intent" in e for e in errors)


def test_validate_empty_summary() -> None:
    """summary="" → error."""
    plan = _make_plan(summary="")
    errors = validate_plan(plan)
    assert any("summary" in e for e in errors)


def test_validate_missing_anchor_fields() -> None:
    """anchor missing system/record_type/record_id → 3 errors."""
    plan = _make_plan(anchor={})
    errors = validate_plan(plan)
    assert len([e for e in errors if e.startswith("anchor.")]) == 3


def test_validate_write_step_missing_tool_name() -> None:
    """write step with empty tool_name → error."""
    plan = _make_plan(
        writes=[PlanStep(tool_name="", record_id="elabftw:experiment:123")],
    )
    errors = validate_plan(plan)
    assert any("writes[0].tool_name" in e for e in errors)


def test_validate_write_step_missing_record_id() -> None:
    """write step with record_id=None → error."""
    plan = _make_plan(
        writes=[PlanStep(tool_name="elabftw.amend_experiment")],
    )
    errors = validate_plan(plan)
    assert any("writes[0].record_id" in e for e in errors)


def test_validate_read_step_missing_tool_name() -> None:
    """read step with empty tool_name → error."""
    plan = _make_plan(
        reads=[PlanStep(tool_name="")],
    )
    errors = validate_plan(plan)
    assert any("reads[0].tool_name" in e for e in errors)


def test_validate_forbidden_raw_endpoint_in_args() -> None:
    """step args containing "url" → error."""
    plan = _make_plan(
        writes=[
            PlanStep(
                tool_name="elabftw.edit_experiment_section",
                record_id="elabftw:experiment:123",
                args={"url": "https://evil.com"},
            ),
        ],
    )
    errors = validate_plan(plan)
    assert any("forbidden" in e.lower() for e in errors)


def test_validate_hardware_without_rollback() -> None:
    """HARDWARE tier, rollback=None → error."""
    plan = _make_plan(risk_tier=PlanRiskTier.HARDWARE, rollback=None)
    errors = validate_plan(plan)
    assert any("rollback" in e.lower() for e in errors)


def test_validate_autonomous_without_rollback() -> None:
    """AUTONOMOUS tier, rollback=None → error."""
    plan = _make_plan(risk_tier=PlanRiskTier.AUTONOMOUS, rollback=None)
    errors = validate_plan(plan)
    assert any("rollback" in e.lower() for e in errors)


def test_validate_writes_without_approval() -> None:
    """writes present, approval_required=False → error."""
    plan = _make_plan(approval_required=False)
    errors = validate_plan(plan)
    assert any("approval" in e.lower() for e in errors)


def test_validate_or_raise_raises_on_invalid() -> None:
    """invalid plan → PlanValidationError with .errors."""
    plan = _make_plan(plan_id="")
    with pytest.raises(PlanValidationError) as exc_info:
        validate_or_raise(plan)
    assert len(exc_info.value.errors) >= 1
    assert any("plan_id" in e for e in exc_info.value.errors)


def test_validate_or_raise_passes_on_valid() -> None:
    """valid plan → no exception."""
    validate_or_raise(fixture_simple_edit())  # should not raise


# =============================================================================
# from_dict error handling
# =============================================================================


def test_from_dict_invalid_risk_tier() -> None:
    """risk_tier="bogus" → PlanValidationError."""
    d = fixture_simple_edit().to_dict()
    d["risk_tier"] = "bogus"
    with pytest.raises(PlanValidationError) as exc_info:
        Plan.from_dict(d)
    assert "risk_tier" in str(exc_info.value)


def test_from_dict_handles_missing_fields() -> None:
    """Minimal dict with only required fields → constructs with defaults for optionals."""
    minimal = {"plan_id": "plan-minimal", "risk_tier": "single_record_edit"}
    plan = Plan.from_dict(minimal)
    assert plan.plan_id == "plan-minimal"
    assert plan.intent == ""
    assert plan.summary == ""
    assert plan.anchor == {}
    assert plan.reads == []
    assert plan.writes == []
    assert plan.artifacts == []
    assert plan.approval_required is True
    assert plan.rollback is None


# =============================================================================
# USER_VISIBLE_FIELDS
# =============================================================================


def test_user_visible_fields_complete() -> None:
    """USER_VISIBLE_FIELDS contains all expected field names."""
    expected = (
        "plan_id",
        "intent",
        "anchor",
        "risk_tier",
        "summary",
        "reads",
        "writes",
        "artifacts",
        "approval",
        "rollback",
    )
    assert USER_VISIBLE_FIELDS == expected
