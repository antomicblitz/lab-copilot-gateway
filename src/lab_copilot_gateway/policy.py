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
        kill_categories:       named kill switch categories (all, mutating,
                               hardware, autonomy, adapter_*) that deny
                               groups of tools by tier or adapter.
        approvals_required_for: tier threshold at/above which approval becomes
                               mandatory (default BOUNDED_WRITES).
        block_hardware_above:  tier threshold at/above which execution is blocked
                               regardless of approval (default HARDWARE_EXECUTION).
    """

    max_tier: Tier = Tier.CLOSED_LOOP_AUTONOMY
    kill_switches: tuple[str, ...] = ()
    kill_categories: dict[str, bool] = field(default_factory=dict)
    approvals_required_for: Tier = Tier.BOUNDED_WRITES
    block_hardware_above: Tier = Tier.HARDWARE_EXECUTION

    def decide(self, req: PolicyRequest) -> Decision:
        # 0. Named kill switch categories (highest precedence).
        matched_categories = self._matched_kill_categories(req)
        if matched_categories:
            return Decision(
                decision="deny",
                reason="kill_switch_match",
                tier=int(req.tier),
                matched_kill_switches=matched_categories,
            )

        # 1. Kill switch fnmatch patterns.
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

        # 4. C29: Autonomy-enabled callers bypass the hardware block and
        # approval requirement for the autonomous tier (6) only.
        # Kill switches (steps 0-1) and unmapped caller (step 2) still deny.
        if req.autonomy_enabled and req.tier == Tier.CLOSED_LOOP_AUTONOMY:
            return Decision(
                decision="allow",
                reason="autonomy_enabled",
                tier=int(req.tier),
            )

        # 5. Hardware / above-block threshold denies by default.
        if req.tier >= int(self.block_hardware_above):
            return Decision(
                decision="deny",
                reason="hardware_blocked",
                tier=int(req.tier),
            )

        # 6. Bounded writes & higher require approval.
        if req.tier >= int(self.approvals_required_for) and not req.has_approval:
            return Decision(
                decision="deny",
                reason="approval_required",
                tier=int(req.tier),
                requires_approval=True,
            )

        # 7. Read / status / validate (tiers 0-3 with approval if required) allow.
        return Decision(decision="allow", reason="default_allow", tier=int(req.tier))

    def _matched_kill_categories(self, req: PolicyRequest) -> list[str]:
        """Check named kill switch categories against a request.

        Returns the list of matching category names, or an empty list if
        no category applies.  ``all`` short-circuits -- it denies every
        tool unconditionally.
        """
        if not self.kill_categories:
            return []

        matched: list[str] = []

        # ``all`` denies every tool unconditionally.
        if self.kill_categories.get("all", False):
            matched.append("all")
            return matched

        matched.extend(self._matched_tier_categories(req))
        matched.extend(self._matched_adapter_categories(req))
        return matched

    def _matched_tier_categories(self, req: PolicyRequest) -> list[str]:
        """Check tier-based kill switch categories."""
        matched: list[str] = []
        if self.kill_categories.get("mutating", False) and req.tier >= int(
            Tier.BOUNDED_WRITES
        ):
            matched.append("mutating")
        if self.kill_categories.get("hardware", False) and req.tier >= int(
            Tier.HARDWARE_EXECUTION
        ):
            matched.append("hardware")
        if self.kill_categories.get("autonomy", False) and req.tier >= int(
            Tier.CLOSED_LOOP_AUTONOMY
        ):
            matched.append("autonomy")
        return matched

    def _matched_adapter_categories(self, req: PolicyRequest) -> list[str]:
        """Check adapter-specific kill switch categories."""
        matched: list[str] = []
        adapter = req.adapter
        if adapter is None and "." in req.tool_name:
            adapter = req.tool_name.split(".", 1)[0]

        if adapter:
            for name in ("elabftw", "opencloning", "wallac", "bentolab", "mcp"):
                key = f"adapter_{name}"
                if self.kill_categories.get(key, False) and adapter == name:
                    matched.append(key)

        return matched

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

    Kill switches are read from the ``LAB_COPILOT_KILL_SWITCHES`` env var
    and kill switch categories from the ``LAB_COPILOT_KILL_*`` env vars on
    first construction; updates require a process restart or explicit
    ``reset_policy_engine()`` or ``set_kill_category()`` call.
    """
    global _default_engine
    if _default_engine is None:
        from lab_copilot_gateway.config import (
            get_kill_switch_categories,
            get_kill_switches,
        )

        _default_engine = PolicyEngine(
            kill_switches=tuple(get_kill_switches()),
            kill_categories=get_kill_switch_categories(),
        )
    return _default_engine


def reset_policy_engine(engine: PolicyEngine | None = None) -> None:
    """Test helper: replace or clear the singleton."""
    global _default_engine
    _default_engine = engine


def set_kill_category(switch: str, enabled: bool) -> None:
    """Set a named kill switch category on the singleton engine at runtime.

    Mutates the process-wide engine's ``kill_categories`` dict in-place so
    all references (closures, adapter references) see the change immediately
    without needing a restart.
    """
    global _default_engine
    if _default_engine is None:
        _default_engine = get_policy_engine()
    _default_engine.kill_categories[switch] = enabled
