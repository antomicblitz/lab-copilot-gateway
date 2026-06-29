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


def get_public_config(*, service_name: str, version: str) -> dict[str, object]:
    """Return config safe to expose to authenticated clients."""
    return {
        "service": service_name,
        "version": version,
        "environment": os.getenv("LAB_COPILOT_ENV", "local"),
        "approvals_required_for_mutations": True,
        "hardware_execution_enabled": False,
        "autonomous_execution_enabled": False,
    }
