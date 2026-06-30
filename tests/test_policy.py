"""Tests for the deterministic policy engine (C04)."""

from __future__ import annotations

import pytest

from lab_copilot_gateway.policy import (
    PolicyEngine,
    PolicyRequest,
    Tier,
)


def _req(
    *,
    tool_name: str = "elabftw.read_experiment",
    tier: Tier = Tier.OPERATIONAL_READ_ONLY,
    user_id: str | None = "elab-user-1",
    has_approval: bool = False,
    **overrides,
) -> PolicyRequest:
    base = dict(
        tool_name=tool_name,
        tier=tier,
        user_id=user_id,
        has_approval=has_approval,
    )
    base.update(overrides)
    return PolicyRequest(**base)


@pytest.fixture
def engine() -> PolicyEngine:
    """Default policy engine: max_tier=BOUNDED_WRITES(4), no kill switches."""
    return PolicyEngine()


# --- Acceptance check 1: Default policy allows read/status/validate for mapped users.


def test_allows_tier0_static_help_for_mapped_user(engine: PolicyEngine) -> None:
    d = engine.decide(_req(tier=Tier.STATIC_HELP, tool_name="help.faq"))
    assert d.decision == "allow"
    assert d.reason == "default_allow"
    assert d.tier == 0


def test_allows_tier1_operational_read_only(engine: PolicyEngine) -> None:
    d = engine.decide(
        _req(tier=Tier.OPERATIONAL_READ_ONLY, tool_name="elabftw.read_experiment")
    )
    assert d.decision == "allow"
    assert d.tier == 1


def test_allows_tier2_permissioned_read_for_mapped_user(engine: PolicyEngine) -> None:
    d = engine.decide(
        _req(
            tier=Tier.PERMISSIONED_ELABFTW_READ,
            tool_name="elabftw.read_current_experiment",
        )
    )
    assert d.decision == "allow"
    assert d.tier == 2


def test_allows_tier3_validation_dry_run_for_mapped_user(engine: PolicyEngine) -> None:
    d = engine.decide(
        _req(tier=Tier.VALIDATION_DRY_RUN, tool_name="validate.bentolab_pcr_profile")
    )
    assert d.decision == "allow"
    assert d.tier == 3


# --- Acceptance check 2: Default policy requires approval for bounded writes (tier 4).


def test_tier4_bounded_write_without_approval_denies(engine: PolicyEngine) -> None:
    d = engine.decide(
        _req(
            tier=Tier.BOUNDED_WRITES,
            tool_name="elabftw.amend_experiment_after_approval",
        )
    )
    assert d.decision == "deny"
    assert d.reason == "approval_required"
    assert d.requires_approval is True
    assert d.tier == 4


def test_tier4_bounded_write_with_approval_allows(engine: PolicyEngine) -> None:
    d = engine.decide(
        _req(
            tier=Tier.BOUNDED_WRITES,
            tool_name="elabftw.amend_experiment_after_approval",
            has_approval=True,
            approval_id="appr-1",
        )
    )
    assert d.decision == "allow"
    assert d.tier == 4


# --- Acceptance check 3: Default policy blocks hardware execution (tier >= 5).


def test_tier5_hardware_execution_denies_even_with_approval(
    engine: PolicyEngine,
) -> None:
    d = engine.decide(
        _req(
            tier=Tier.HARDWARE_EXECUTION,
            tool_name="wallac.run_measurement",
            has_approval=True,
        )
    )
    assert d.decision == "deny"
    assert d.reason == "hardware_blocked"
    assert d.tier == 5


def test_tier6_closed_loop_autonomy_denies(engine: PolicyEngine) -> None:
    d = engine.decide(
        _req(tier=Tier.CLOSED_LOOP_AUTONOMY, tool_name="bentolab.run_autonomous")
    )
    assert d.decision == "deny"
    assert d.tier == 6


# --- Acceptance check 4: Kill switch denies matching tools.


def test_kill_switch_exact_match_denies(engine: PolicyEngine) -> None:
    eng = PolicyEngine(kill_switches=("validate.bentolab_pcr_profile",))
    d = eng.decide(
        _req(tier=Tier.VALIDATION_DRY_RUN, tool_name="validate.bentolab_pcr_profile")
    )
    assert d.decision == "deny"
    assert d.reason == "kill_switch_match"
    assert "validate.bentolab_pcr_profile" in d.matched_kill_switches


def test_kill_switch_fnmatch_pattern_denies(engine: PolicyEngine) -> None:
    eng = PolicyEngine(kill_switches=("wallac.*",))
    d = eng.decide(
        _req(
            tier=Tier.OPERATIONAL_READ_ONLY,
            tool_name="wallac.get_status",
            user_id="elab-user-1",
        )
    )
    assert d.decision == "deny"
    assert d.reason == "kill_switch_match"
    assert "wallac.*" in d.matched_kill_switches


def test_kill_switch_takes_precedence_over_allow(engine: PolicyEngine) -> None:
    """Even a tier-0 tool that would normally allow must deny if kill switch matches."""
    eng = PolicyEngine(kill_switches=("help.faq",))
    d = eng.decide(_req(tier=Tier.STATIC_HELP, tool_name="help.faq"))
    assert d.decision == "deny"
    assert d.reason == "kill_switch_match"


def test_kill_switch_non_matching_does_not_block(engine: PolicyEngine) -> None:
    eng = PolicyEngine(kill_switches=("validate.bentolab_pcr_profile",))
    d = eng.decide(
        _req(tier=Tier.OPERATIONAL_READ_ONLY, tool_name="elabftw.read_experiment")
    )
    assert d.decision == "allow"


# --- Fail-closed behavior (not in acceptance list, but implied).


def test_unmapped_user_denies_even_for_tier0(engine: PolicyEngine) -> None:
    d = engine.decide(_req(user_id=None, tier=Tier.STATIC_HELP))
    assert d.decision == "deny"
    assert d.reason == "unmapped_caller"


def test_tier_exceeds_global_max_denies() -> None:
    # max_tier lowered to 2; tier 3 is above cap.
    eng = PolicyEngine(max_tier=Tier.PERMISSIONED_ELABFTW_READ)
    d = eng.decide(_req(tier=Tier.VALIDATION_DRY_RUN, user_id="elab-user-1"))
    assert d.decision == "deny"
    assert d.reason == "tier_exceeds_max"


# --- Determinism (verified by repeating the same input).


def test_decide_is_deterministic(engine: PolicyEngine) -> None:
    req = _req(tier=Tier.OPERATIONAL_READ_ONLY, tool_name="elabftw.read_experiment")
    decisions = [engine.decide(req) for _ in range(20)]
    assert all(d.decision == "allow" for d in decisions)
    assert all(d.reason == "default_allow" for d in decisions)


# --- Kill switch categories (C26) -----------------------------------------


class TestKillSwitchCategories:
    """Named kill switch category tests."""

    def test_kill_all_denies_every_tool(self) -> None:
        eng = PolicyEngine(kill_categories={"all": True})
        d = eng.decide(_req(tier=Tier.STATIC_HELP, tool_name="help.faq"))
        assert d.decision == "deny"
        assert d.reason == "kill_switch_match"
        assert d.matched_kill_switches == ["all"]

    def test_kill_all_denies_tier6_tool(self) -> None:
        eng = PolicyEngine(kill_categories={"all": True})
        d = eng.decide(
            _req(
                tier=Tier.CLOSED_LOOP_AUTONOMY,
                tool_name="bentolab.run_autonomous",
                has_approval=True,
            )
        )
        assert d.decision == "deny"
        assert "all" in d.matched_kill_switches

    def test_kill_mutating_denies_tier4_but_allows_tier3(self) -> None:
        eng = PolicyEngine(kill_categories={"mutating": True})
        # Tier 3 (validation / dry-run) allowed.
        d = eng.decide(
            _req(
                tier=Tier.VALIDATION_DRY_RUN,
                tool_name="opencloning.parse_sequence_file",
            )
        )
        assert d.decision == "allow"
        # Tier 4 (bounded write) denied.
        d = eng.decide(
            _req(
                tier=Tier.BOUNDED_WRITES,
                tool_name="elabftw.amend_experiment",
                has_approval=True,
            )
        )
        assert d.decision == "deny"
        assert "mutating" in d.matched_kill_switches

    def test_kill_hardware_denies_tier5_but_allows_tier4(self) -> None:
        """Hardware kill switch is checked before the default hardware block."""
        eng = PolicyEngine(kill_categories={"hardware": True})
        # Tier 4 with approval allowed (hardware kill switch does not match).
        d = eng.decide(
            _req(
                tier=Tier.BOUNDED_WRITES,
                tool_name="elabftw.draft_experiment_update",
                has_approval=True,
            )
        )
        assert d.decision == "allow"
        # Tier 5 denied by hardware kill switch (checked at step 0, before
        # the default ``hardware_blocked`` check).
        d = eng.decide(
            _req(
                tier=Tier.HARDWARE_EXECUTION,
                tool_name="wallac.run_measurement",
                has_approval=True,
            )
        )
        assert d.decision == "deny"
        assert "hardware" in d.matched_kill_switches

    def test_kill_autonomy_denies_tier6_but_allows_tier5(self) -> None:
        """Autonomy kill switch is checked before the default hardware block."""
        eng = PolicyEngine(kill_categories={"autonomy": True})
        # Tier 5 is NOT matched by autonomy (tier 5 < 6).  It is eventually
        # denied by the default ``hardware_blocked`` check — that is standard
        # behaviour unrelated to the kill switch.
        matched = eng._matched_kill_categories(
            _req(
                tier=Tier.HARDWARE_EXECUTION,
                tool_name="wallac.run_measurement",
            )
        )
        assert matched == []
        # Tier 6 is matched by autonomy kill switch (step 0, before
        # ``hardware_blocked`` would trigger).
        d = eng.decide(
            _req(
                tier=Tier.CLOSED_LOOP_AUTONOMY,
                tool_name="autonomous.run",
            )
        )
        assert d.decision == "deny"
        assert "autonomy" in d.matched_kill_switches

    def test_kill_adapter_elabftw_denies_elabftw_but_allows_opencloning(
        self,
    ) -> None:
        eng = PolicyEngine(kill_categories={"adapter_elabftw": True})
        d = eng.decide(
            _req(
                tier=Tier.PERMISSIONED_ELABFTW_READ,
                tool_name="elabftw.read_current_experiment",
                adapter="elabftw",
            )
        )
        assert d.decision == "deny"
        assert "adapter_elabftw" in d.matched_kill_switches
        d = eng.decide(
            _req(
                tier=Tier.VALIDATION_DRY_RUN,
                tool_name="opencloning.parse_sequence_file",
                adapter="opencloning",
            )
        )
        assert d.decision == "allow"

    def test_kill_adapter_wallac_denies_wallac_but_allows_elabftw(
        self,
    ) -> None:
        eng = PolicyEngine(kill_categories={"adapter_wallac": True})
        d = eng.decide(
            _req(
                tier=Tier.OPERATIONAL_READ_ONLY,
                tool_name="wallac.get_status",
                adapter="wallac",
            )
        )
        assert d.decision == "deny"
        assert "adapter_wallac" in d.matched_kill_switches
        d = eng.decide(
            _req(
                tier=Tier.PERMISSIONED_ELABFTW_READ,
                tool_name="elabftw.read_current_experiment",
                adapter="elabftw",
            )
        )
        assert d.decision == "allow"

    def test_adapter_detected_from_tool_name_when_adapter_not_set(
        self,
    ) -> None:
        """Adapter-specific kill switches work even when adapter field is None."""
        eng = PolicyEngine(kill_categories={"adapter_bentolab": True})
        d = eng.decide(
            _req(
                tier=Tier.OPERATIONAL_READ_ONLY,
                tool_name="bentolab.get_status",
            )
        )
        assert d.decision == "deny"
        assert "adapter_bentolab" in d.matched_kill_switches

    def test_adapter_no_match_when_tool_has_no_adapter_prefix(
        self,
    ) -> None:
        """Tool without '.' in name does not match adapter categories."""
        eng = PolicyEngine(kill_categories={"adapter_elabftw": True})
        d = eng.decide(_req(tier=Tier.STATIC_HELP, tool_name="help"))
        assert d.decision == "allow"

    def test_named_categories_take_precedence_over_fnmatch(self) -> None:
        """Named categories are evaluated before fnmatch patterns."""
        eng = PolicyEngine(
            kill_categories={"adapter_elabftw": True},
            kill_switches=("opencloning.*",),
        )
        # Named category denies elabftw tools.
        d = eng.decide(
            _req(
                tier=Tier.PERMISSIONED_ELABFTW_READ,
                tool_name="elabftw.read_experiment",
                adapter="elabftw",
            )
        )
        assert d.decision == "deny"
        assert "adapter_elabftw" in d.matched_kill_switches
        # fnmatch pattern still works for opencloning tools.
        d = eng.decide(
            _req(
                tier=Tier.VALIDATION_DRY_RUN,
                tool_name="opencloning.parse_sequence_file",
                adapter="opencloning",
            )
        )
        assert d.decision == "deny"
        assert "opencloning.*" in d.matched_kill_switches

    def test_fnmatch_still_works_alongside_named_categories(self) -> None:
        """Both named categories and fnmatch patterns are enforced."""
        eng = PolicyEngine(
            kill_categories={"adapter_elabftw": True},
            kill_switches=("wallac.*",),
        )
        # elabftw denied by named category.
        d = eng.decide(
            _req(
                tier=Tier.PERMISSIONED_ELABFTW_READ,
                tool_name="elabftw.read_experiment",
                adapter="elabftw",
            )
        )
        assert d.decision == "deny"
        assert "adapter_elabftw" in d.matched_kill_switches
        # wallac denied by fnmatch pattern.
        d = eng.decide(
            _req(
                tier=Tier.OPERATIONAL_READ_ONLY,
                tool_name="wallac.get_status",
                adapter="wallac",
            )
        )
        assert d.decision == "deny"
        assert "wallac.*" in d.matched_kill_switches
        # opencloning allowed (no matching switch).
        d = eng.decide(
            _req(
                tier=Tier.VALIDATION_DRY_RUN,
                tool_name="opencloning.parse_sequence_file",
                adapter="opencloning",
            )
        )
        assert d.decision == "allow"

    def test_kill_all_bypasses_other_categories(self) -> None:
        """When ``all`` is active, no other category checks are needed."""
        eng = PolicyEngine(
            kill_categories={"all": True, "adapter_elabftw": True},
        )
        d = eng.decide(
            _req(
                tier=Tier.STATIC_HELP,
                tool_name="help.faq",
            )
        )
        assert d.decision == "deny"
        # Only "all" is in matched list (short-circuit).
        assert d.matched_kill_switches == ["all"]

    def test_empty_categories_no_effect(self) -> None:
        """Engine with empty kill_categories behaves like no categories."""
        eng = PolicyEngine()
        d = eng.decide(
            _req(
                tier=Tier.STATIC_HELP,
                tool_name="help.faq",
            )
        )
        assert d.decision == "allow"
