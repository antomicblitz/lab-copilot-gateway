"""FastAPI application for the lab copilot gateway scaffold."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from lab_copilot_gateway import __version__
from lab_copilot_gateway.approval import (
    ApprovalError,
    ApprovalRequest,
    ApprovalStore,
    compute_args_hash,
    get_approval_store,
)
from lab_copilot_gateway.audit import AuditRecord, get_audit_store
from lab_copilot_gateway.config import get_public_config
from lab_copilot_gateway.elabftw import (
    ElabftwAdapterError,
    get_elabftw_read_adapter,
)
from lab_copilot_gateway.identity import (
    DbIdentityMapper,
    IdentityMapper,
    MappedIdentity,
    get_identity_mapper,
)
from lab_copilot_gateway.policy import PolicyRequest, Tier, get_policy_engine
from lab_copilot_gateway.tools import get_tool_registry, list_tools


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


class ApprovalRequestBody(BaseModel):
    """Request body for POST /approval/request.

    Captures the exact tool name, args (the gateway computes the hash), target
    record, tier, requesting identity, provider/model, and TTL.  The gateway
    stores the hash of canonical-JSON args; the caller never sees the hash.
    """

    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    target_record: str | None = None
    tier: int
    keycloak_subject: str | None = None
    librechat_user_id: str | None = None
    mapped_elabftw_user_id: str | None = None
    provider: str | None = None
    model_id: str | None = None
    ttl_seconds: int | None = None


class ApprovalConsumeBody(BaseModel):
    """Request body for POST /approval/consume.

    The gateway recomputes the args hash from the args the caller is actually
    using, then verifies the token is bound to the same hash.  An attacker
    cannot replay an approval with modified args — the hash will not match.
    """

    approval_id: str
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)
    target_record: str | None = None


class ElabftwReadBody(BaseModel):
    """Request body for POST /elabftw/read_current_experiment (C08).

    Caller supplies the signed context token (from the eLabFTW launcher, C11)
    and the identity fields the gateway uses to resolve the mapped user via
    the identity mapper.  The adapter does the rest: token verify → identity
    resolution → policy decision → downstream HTTP → audit record.
    """

    context_token: str
    keycloak_subject: str | None = None
    librechat_user_id: str | None = None
    conversation_id: str | None = None
    request_id: str | None = None
    provider: str | None = None
    model_id: str | None = None


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


def _approval_backend_status(store: ApprovalStore) -> dict[str, str]:
    """Health-facing summary of the approval token store."""
    return {
        "approval_backend": "db",
        "approval_db": "configured" if store.db_path != ":memory:" else "memory",
    }


def create_app() -> FastAPI:
    """Create the ASGI application."""
    service_name = "lab-copilot-gateway"
    api = FastAPI(title="Lab Copilot Gateway", version=__version__)

    # Eagerly initialize audit store, policy engine, identity mapper,
    # approval store, and eLabFTW read adapter so /health reflects state.
    audit_store = get_audit_store()
    policy_engine = get_policy_engine()
    identity_mapper = get_identity_mapper()
    approval_store = get_approval_store()
    elabftw_adapter = get_elabftw_read_adapter()

    # CORS is intentionally closed in the scaffold. Configure allowed LibreChat
    # origins when browser-based calls are introduced.

    @api.get("/health")
    def health() -> dict[str, object]:
        deps: dict[str, object] = {
            "audit_db": "configured" if audit_store.db_path != ":memory:" else "memory",
            "policy_engine": "ready",
            "policy_max_tier": int(policy_engine.max_tier),
            "kill_switches": list(policy_engine.kill_switches),
            "elabftw": (
                "configured" if elabftw_adapter.client is not None else "not_configured"
            ),
            "opencloning": "not_configured",
            "wallac": "not_configured",
            "bentolab": "not_configured",
        }
        deps.update(_identity_backend_status(identity_mapper))
        deps.update(_approval_backend_status(approval_store))
        deps["tool_count"] = len(get_tool_registry().list())
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

    @api.post("/approval/request")
    def request_approval(body: ApprovalRequestBody) -> dict[str, object]:
        """Issue a single-use approval token bound to (tool, args hash, target).

        The caller (typically the gateway's tool-execution path or the
        LibreChat approval card) supplies the EXACT args the tool will run
        with; the gateway hashes them and stores the hash.  ``approval_id``
        is opaque to the client and may be displayed but not modified.
        """
        from lab_copilot_gateway.approval import DEFAULT_APPROVAL_TTL_SECONDS

        req = ApprovalRequest(
            tool_name=body.tool_name,
            args_hash=compute_args_hash(body.args),
            target_record=body.target_record,
            tier=body.tier,
            keycloak_subject=body.keycloak_subject,
            librechat_user_id=body.librechat_user_id,
            mapped_elabftw_user_id=body.mapped_elabftw_user_id,
            provider=body.provider,
            model_id=body.model_id,
            ttl_seconds=body.ttl_seconds or DEFAULT_APPROVAL_TTL_SECONDS,
        )
        approval_id, expires_at = approval_store.request(req)
        return {
            "approval_id": approval_id,
            "expires_at": expires_at,
            "tool_name": req.tool_name,
            "args_hash": req.args_hash,
        }

    @api.post("/approval/consume")
    def consume_approval(body: ApprovalConsumeBody) -> dict[str, object]:
        """Verify + consume a single-use approval token.

        Returns ``{"consumed": true, ...}`` on success or
        ``{"consumed": false, "reason": ..., "message": ...}`` on any failure
        (not_found / already_consumed / expired / mismatch).  The reason code
        is stable for callers; the message is human-readable.
        """
        args_hash = compute_args_hash(body.args)
        try:
            result = approval_store.consume(
                body.approval_id,
                tool_name=body.tool_name,
                args_hash=args_hash,
                target_record=body.target_record,
            )
        except ApprovalError as exc:
            return {"consumed": False, **exc.to_dict()}
        return {
            "consumed": True,
            "approval_id": result.approval_id,
            "tool_name": result.tool_name,
            "args_hash": result.args_hash,
            "target_record": result.target_record,
            "tier": result.tier,
            "consumed_at": result.consumed_at,
        }

    @api.get("/approval/{approval_id}")
    def get_approval(approval_id: str) -> dict[str, object] | None:
        """Fetch one approval row by id (admin / test reads)."""
        record = approval_store.get(approval_id)
        if record is None:
            return None
        return record.to_dict()

    @api.post("/elabftw/read_current_experiment")
    def elabftw_read_current_experiment(body: ElabftwReadBody) -> dict[str, object]:
        """Read the experiment referenced by the caller's context token.

        Orchestrates token verification → identity resolution → policy decision
        → downstream eLabFTW HTTP call → audit record.  Returns the read
        result on success or an ``error`` payload on any adapter failure
        (invalid token, unmapped caller, policy denied, client error).
        """
        adapter = get_elabftw_read_adapter()
        # Resolve the mapped identity before invoking the adapter so the
        # adapter receives a `MappedIdentity | None` to fail closed on.
        mapped_identity = identity_mapper.map(
            keycloak_subject=body.keycloak_subject,
            librechat_user_id=body.librechat_user_id,
        )
        try:
            result = adapter.read_current_experiment(
                context_token=body.context_token,
                mapped_identity=mapped_identity,
                conversation_id=body.conversation_id,
                request_id=body.request_id,
                keycloak_subject=body.keycloak_subject,
                librechat_user_id=body.librechat_user_id,
                provider=body.provider,
                model_id=body.model_id,
            )
        except ElabftwAdapterError as exc:
            return {"ok": False, **exc.to_dict()}
        return {"ok": True, "experiment": result.to_dict()}

    return api


app = create_app()
