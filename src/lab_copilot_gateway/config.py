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


# --- MCP (Model Context Protocol) outbound adapter settings -------------------
#
# MCP servers connect behind the gateway as downstream adapters.  These
# settings govern how the gateway connects to them, enforces size/timeout
# limits, and surfaces health diagnostics.


def get_mcp_server_specs() -> list[dict[str, object]]:
    """Return the configured MCP server specs from env.

    Reads ``LAB_COPILOT_MCP_SERVERS`` — a JSON list of server objects, each
    with ``id`` (str), ``url`` (str), ``timeout`` (float, seconds, optional),
    and ``max_result_bytes`` (int, optional).  Missing is empty list.

    Example env value::

        [{"id": "pubmed", "url": "https://pubmed-mcp.example.org/mcp",
          "timeout": 30, "max_result_bytes": 1048576}]
    """
    import json as _json

    raw = os.getenv("LAB_COPILOT_MCP_SERVERS", "")
    if not raw.strip():
        return []
    try:
        specs = _json.loads(raw)
    except _json.JSONDecodeError:
        return []
    if not isinstance(specs, list):
        return []
    result: list[dict[str, object]] = []
    for spec in specs:
        if not isinstance(spec, dict) or not isinstance(spec.get("id"), str):
            continue
        result.append(
            {
                "id": spec["id"],
                "url": spec.get("url", ""),
                "timeout": float(spec.get("timeout", 30.0)),
                "max_result_bytes": int(spec.get("max_result_bytes", 1048576)),
            }
        )
    return result


def get_mcp_default_timeout() -> float:
    """Default timeout in seconds for MCP server connections."""
    try:
        return float(os.getenv("LAB_COPILOT_MCP_TIMEOUT", "30"))
    except ValueError:
        return 30.0


def get_mcp_default_max_result_bytes() -> int:
    """Default maximum result size in bytes for MCP tool calls."""
    try:
        return int(os.getenv("LAB_COPILOT_MCP_MAX_RESULT_BYTES", "1048576"))
    except ValueError:
        return 1048576


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
