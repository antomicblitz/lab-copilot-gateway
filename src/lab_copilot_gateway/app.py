"""FastAPI application for the lab copilot gateway scaffold."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from lab_copilot_gateway import __version__
from lab_copilot_gateway.audit import AuditRecord, get_audit_store
from lab_copilot_gateway.config import get_public_config
from lab_copilot_gateway.identity import (
    DbIdentityMapper,
    IdentityMapper,
    MappedIdentity,
    get_identity_mapper,
)
from lab_copilot_gateway.policy import PolicyRequest, Tier, get_policy_engine
from lab_copilot_gateway.tools import list_tools


class AuditBody(BaseModel):
    """Request body for POST /audit."""

    action_id: str
    conversation_id: str | None = None
    request_id: str | None = None
    keycloak_subject: str | None = None
    librechat_user_id: str | None = None
    mapped_elabftw_user_id: str | None = None
    mapped_elabftw_team_id: str | None = None
    provider: str | None = None
    model_id: str | None = None
    tool_name: str | None = None
    tool_args_hash: str | None = None
    context_refs: list[dict[str, object]] = Field(default_factory=list)
    policy_decision: str | None = None
    approval_id: str | None = None
    api_call_summary: dict[str, object] = Field(default_factory=dict)
    result_summary: dict[str, object] = Field(default_factory=dict)
    error: dict[str, object] | None = None
    artifact_manifest: list[dict[str, object]] = Field(default_factory=list)


class PolicyRequestBody(BaseModel):
    """Request body for POST /policy/evaluate."""

    tool_name: str
    tier: int
    adapter: str | None = None
    user_id: str | None = None
    team_id: str | None = None
    autonomy_enabled: bool = False
    has_approval: bool = False
    approval_id: str | None = None


class IdentityResolveBody(BaseModel):
    """Request body for POST /identity/resolve.

    Either identifier alone is sufficient — a row matching either one resolves
    the identity.  Both ``None`` resolves nothing.
    """

    keycloak_subject: str | None = None
    librechat_user_id: str | None = None


def _identity_to_dict(identity: MappedIdentity) -> dict[str, Any]:
    return {
        "mapped": True,
        "keycloak_subject": identity.keycloak_subject,
        "librechat_user_id": identity.librechat_user_id,
        "elabftw_user_id": identity.elabftw_user_id,
        "elabftw_team_id": identity.elabftw_team_id,
        "elabftw_team_ids": list(identity.elabftw_team_ids),
    }


def _identity_backend_status(mapper: IdentityMapper) -> dict[str, str]:
    """Health-facing summary of the identity mapper backend."""
    if isinstance(mapper, DbIdentityMapper):
        return {
            "identity_backend": "db",
            "identity_db": "configured" if mapper.db_path != ":memory:" else "memory",
        }
    return {
        "identity_backend": "static",
        "identity_path": os.getenv("LAB_COPILOT_IDENTITY_MAP_PATH", ""),
    }


def create_app() -> FastAPI:
    """Create the ASGI application."""
    service_name = "lab-copilot-gateway"
    api = FastAPI(title="Lab Copilot Gateway", version=__version__)

    # Eagerly initialize audit store, policy engine, and identity mapper so
    # /health reflects state.
    audit_store = get_audit_store()
    policy_engine = get_policy_engine()
    identity_mapper = get_identity_mapper()

    # CORS is intentionally closed in the scaffold. Configure allowed LibreChat
    # origins when browser-based calls are introduced.

    @api.get("/health")
    def health() -> dict[str, object]:
        deps: dict[str, object] = {
            "audit_db": "configured" if audit_store.db_path != ":memory:" else "memory",
            "policy_engine": "ready",
            "policy_max_tier": int(policy_engine.max_tier),
            "kill_switches": list(policy_engine.kill_switches),
            "elabftw": "not_configured",
            "opencloning": "not_configured",
            "wallac": "not_configured",
            "bentolab": "not_configured",
        }
        deps.update(_identity_backend_status(identity_mapper))
        return {
            "service": service_name,
            "version": __version__,
            "status": "ok",
            "dependencies": deps,
        }

    @api.get("/config/public")
    def public_config() -> dict[str, object]:
        return get_public_config(service_name=service_name, version=__version__)

    @api.get("/tools")
    def tools() -> dict[str, list[dict[str, object]]]:
        return {"tools": list_tools()}

    @api.post("/audit")
    def append_audit(body: AuditBody) -> dict[str, object]:
        record = AuditRecord(**body.model_dump())
        created_at = audit_store.append(record)
        return {"action_id": record.action_id, "created_at": created_at}

    @api.get("/audit/{action_id}")
    def get_audit(action_id: str) -> dict[str, object] | None:
        return audit_store.get(action_id)

    @api.post("/policy/evaluate")
    def evaluate_policy(body: PolicyRequestBody) -> dict[str, object]:
        req = PolicyRequest(
            tool_name=body.tool_name,
            tier=Tier(body.tier),
            adapter=body.adapter,
            user_id=body.user_id,
            team_id=body.team_id,
            autonomy_enabled=body.autonomy_enabled,
            has_approval=body.has_approval,
            approval_id=body.approval_id,
        )
        decision = policy_engine.decide(req)
        return {
            "decision": decision.decision,
            "reason": decision.reason,
            "tier": decision.tier,
            "requires_approval": decision.requires_approval,
            "matched_kill_switches": decision.matched_kill_switches,
        }

    @api.post("/identity/resolve")
    def resolve_identity(body: IdentityResolveBody) -> dict[str, object]:
        identity = identity_mapper.map(
            keycloak_subject=body.keycloak_subject,
            librechat_user_id=body.librechat_user_id,
        )
        if identity is None:
            # Missing mapping fails closed — caller (LibreChat) must treat
            # this as "no lab access" and the policy engine separately denies
            # any subsequent /policy/evaluate call (user_id would be None).
            return {"mapped": False}
        return _identity_to_dict(identity)

    return api


app = create_app()
