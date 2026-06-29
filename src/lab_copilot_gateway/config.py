"""Non-secret gateway configuration helpers."""

from __future__ import annotations

import os


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
