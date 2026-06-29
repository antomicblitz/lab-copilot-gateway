"""Deterministic policy engine for lab copilot gateway.

Policy is independent of LLM prompts. All decisions are deterministic functions
of (user/team identity, tool tier, adapter, autonomy flag, approval state,
kill-switch set).

Tiers (0-6, lower = less privileged):

    0  static/help
    1  operational read-only
    2  permissioned eLabFTW read
    3  validation/dry-run/computational design
    4  bounded writes
    5  hardware execution
    6  closed-loop autonomy

Default policy stance (fail-closed, least privilege):

    * Kill-switch matches always deny (highest priority).
    * Unmapped callers (user_id is None) deny.
    * Tier above the configured max deny.
    * Tier >= HARDWARE_EXECUTION (5) deny by default.
    * Tier >= BOUNDED_WRITES (4) require a valid approval; absent approval denies.
    * Tier <= 3 allow for mapped callers (read/status/validate).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Tier(IntEnum):
    """Action tiers. Higher = more privilege required."""

    STATIC_HELP = 0
    OPERATIONAL_READ_ONLY = 1
    PERMISSIONED_ELABFTW_READ = 2
    VALIDATION_DRY_RUN = 3
    BOUNDED_WRITES = 4
    HARDWARE_EXECUTION = 5
    CLOSED_LOOP_AUTONOMY = 6


# Sentinel tiers used in default rules.
HARDWARE_EXECUTION_TIER = Tier.HARDWARE_EXECUTION
BOUNDED_WRITES_TIER = Tier.BOUNDED_WRITES


@dataclass(frozen=True)
class PolicyRequest:
    """Inputs to a policy decision. None for identity fields => unmapped."""

    tool_name: str
    tier: Tier
    adapter: str | None = None
    user_id: str | None = None
    team_id: str | None = None
    autonomy_enabled: bool = False
    has_approval: bool = False
    approval_id: str | None = None


@dataclass(frozen=True)
class Decision:
    """Outcome of a policy evaluation."""

    decision: str  # "allow" | "deny"
    reason: str
    tier: int
    requires_approval: bool = False
    matched_kill_switches: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PolicyEngine:
    """Deterministic policy engine.

    Configurable knobs:
        max_tier:              highest tier permitted at all (default 6, the
                               full set; admins lower this to lock everything
                               down without rewriting rules).
        kill_switches:         tool-name patterns (exact or fnmatch) to deny
                               unconditionally.  Injected at runtime so kill
                               switch can be toggled without redeploy.
        approvals_required_for: tier threshold at/above which approval becomes
                               mandatory (default BOUNDED_WRITES).
        block_hardware_above:  tier threshold at/above which execution is blocked
                               regardless of approval (default HARDWARE_EXECUTION).
    """

    max_tier: Tier = Tier.CLOSED_LOOP_AUTONOMY
    kill_switches: tuple[str, ...] = ()
    approvals_required_for: Tier = Tier.BOUNDED_WRITES
    block_hardware_above: Tier = Tier.HARDWARE_EXECUTION

    def decide(self, req: PolicyRequest) -> Decision:
        # 1. Kill switch (highest precedence).
        matched = self._matched_kill_switches(req.tool_name)
        if matched:
            return Decision(
                decision="deny",
                reason="kill_switch_match",
                tier=int(req.tier),
                matched_kill_switches=matched,
            )

        # 2. Unmapped caller.
        if not req.user_id:
            return Decision(
                decision="deny",
                reason="unmapped_caller",
                tier=int(req.tier),
            )

        # 3. Tier exceeds global cap.
        if req.tier > self.max_tier:
            return Decision(
                decision="deny",
                reason="tier_exceeds_max",
                tier=int(req.tier),
            )

        # 4. Hardware / above-block threshold denies by default.
        if req.tier >= int(self.block_hardware_above):
            return Decision(
                decision="deny",
                reason="hardware_blocked",
                tier=int(req.tier),
            )

        # 5. Bounded writes & higher require approval.
        if req.tier >= int(self.approvals_required_for) and not req.has_approval:
            return Decision(
                decision="deny",
                reason="approval_required",
                tier=int(req.tier),
                requires_approval=True,
            )

        # 6. Read / status / validate (tiers 0-3 with approval if required) allow.
        return Decision(decision="allow", reason="default_allow", tier=int(req.tier))

    def _matched_kill_switches(self, tool_name: str) -> list[str]:
        import fnmatch

        return [
            pat
            for pat in self.kill_switches
            if pat == tool_name or fnmatch.fnmatch(tool_name, pat)
        ]


# --- module-level singleton for dependency injection ----------------------
_default_engine: PolicyEngine | None = None


def get_policy_engine() -> PolicyEngine:
    """Return the process-wide policy engine (created lazily).

    Kill switches are read from the LAB_COPILOT_KILL_SWITCHES env var on first
    construction; updates require a process restart or explicit reset_policy_engine()
    call.
    """
    global _default_engine
    if _default_engine is None:
        from lab_copilot_gateway.config import get_kill_switches

        _default_engine = PolicyEngine(
            kill_switches=tuple(get_kill_switches()),
        )
    return _default_engine


def reset_policy_engine(engine: PolicyEngine | None = None) -> None:
    """Test helper: replace or clear the singleton."""
    global _default_engine
    _default_engine = engine
