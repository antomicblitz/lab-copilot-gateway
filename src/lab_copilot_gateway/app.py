"""FastAPI application for the lab copilot gateway scaffold."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from lab_copilot_gateway import __version__
from lab_copilot_gateway.audit import AuditRecord, get_audit_store
from lab_copilot_gateway.config import get_public_config
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


def create_app() -> FastAPI:
    """Create the ASGI application."""
    service_name = "lab-copilot-gateway"
    api = FastAPI(title="Lab Copilot Gateway", version=__version__)

    # Eagerly initialize audit store so /health reflects real state.
    audit_store = get_audit_store()

    # CORS is intentionally closed in the scaffold. Configure allowed LibreChat
    # origins when browser-based calls are introduced.

    @api.get("/health")
    def health() -> dict[str, object]:
        return {
            "service": service_name,
            "version": __version__,
            "status": "ok",
            "dependencies": {
                "audit_db": "configured"
                if audit_store.db_path != ":memory:"
                else "memory",
                "elabftw": "not_configured",
                "opencloning": "not_configured",
                "wallac": "not_configured",
                "bentolab": "not_configured",
            },
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

    return api


app = create_app()
