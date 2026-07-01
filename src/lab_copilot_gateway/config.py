"""Non-secret gateway configuration helpers."""

from __future__ import annotations

import os


def _csv_env(name: str) -> list[str]:
    """Parse a comma-separated env var into a stripped list, dropping blanks."""
    raw = os.getenv(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


def get_kill_switches() -> list[str]:
    """Tool-name patterns the policy engine denies unconditionally."""
    return _csv_env("LAB_COPILOT_KILL_SWITCHES")


# Mapping from named kill switch category -> env var name.
_KILL_SWITCH_CATEGORY_ENV_VARS: dict[str, str] = {
    "all": "LAB_COPILOT_KILL_ALL",
    "mutating": "LAB_COPILOT_KILL_MUTATING",
    "hardware": "LAB_COPILOT_KILL_HARDWARE",
    "autonomy": "LAB_COPILOT_KILL_AUTONOMY",
    "adapter_elabftw": "LAB_COPILOT_KILL_ADAPTER_ELABFTW",
    "adapter_opencloning": "LAB_COPILOT_KILL_ADAPTER_OPENCLONING",
    "adapter_wallac": "LAB_COPILOT_KILL_ADAPTER_WALLAC",
    "adapter_bentolab": "LAB_COPILOT_KILL_ADAPTER_BENTOLAB",
}

#: The set of valid named kill switch category names.
KILL_SWITCH_CATEGORY_NAMES: frozenset[str] = frozenset(
    _KILL_SWITCH_CATEGORY_ENV_VARS.keys()
)


def get_kill_switch_categories() -> dict[str, bool]:
    """Read named kill switch categories from env vars.

    Returns a dict mapping category name to bool.  Only categories whose
    env var is ``"1"``, ``"true"``, or ``"yes"`` (case-insensitive) are
    considered enabled.
    """
    _true_values = {"1", "true", "yes"}
    return {
        name: os.getenv(env_var, "0").strip().lower() in _true_values
        for name, env_var in _KILL_SWITCH_CATEGORY_ENV_VARS.items()
    }


def is_autonomy_enabled() -> bool:
    """Whether autonomous execution is enabled (admin-controlled, C29).

    When True, the policy engine allows tier-6 (CLOSED_LOOP_AUTONOMY)
    requests for mapped callers without per-action approval, subject to
    kill switches and budget limits.
    """
    return os.getenv("LAB_COPILOT_AUTONOMY_ENABLED", "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def get_autonomy_max_steps() -> int:
    """Maximum total steps (reads + writes) in an autonomous plan (C29).

    Budget enforcement prevents runaway autonomous plans from executing
    unbounded work.  Default is 10 steps.
    """
    try:
        return max(1, int(os.getenv("LAB_COPILOT_AUTONOMY_MAX_STEPS", "10")))
    except ValueError:
        return 10


def get_public_config(*, service_name: str, version: str) -> dict[str, object]:
    """Return config safe to expose to authenticated clients."""
    return {
        "service": service_name,
        "version": version,
        "environment": os.getenv("LAB_COPILOT_ENV", "local"),
        "approvals_required_for_mutations": True,
        "hardware_execution_enabled": False,
        "autonomous_execution_enabled": is_autonomy_enabled(),
    }
