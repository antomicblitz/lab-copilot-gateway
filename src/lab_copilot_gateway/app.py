"""FastAPI application for the lab copilot gateway scaffold."""

from __future__ import annotations

import os
from typing import Any

from fastapi import Depends, FastAPI
from pydantic import BaseModel, Field

from lab_copilot_gateway import __version__
from lab_copilot_gateway.approval import (
    ApprovalError,
    ApprovalRequest,
    ApprovalStore,
    compute_args_hash,
    get_approval_store,
)
from lab_copilot_gateway.auth import (
    AuthenticatedPrincipal,
    get_auth_config,
    get_jwks_cache,
    register_auth_exception_handler,
    verified_principal,
)
from lab_copilot_gateway.audit import AuditRecord, get_audit_store
from lab_copilot_gateway.config import get_public_config
from lab_copilot_gateway.elabftw import (
    ElabftwAdapterError,
    mint_token_for_identity,
    get_elabftw_read_adapter,
    get_elabftw_write_adapter,
)
from lab_copilot_gateway.opencloning import (
    OpenCloningAdapterError,
    get_opencloning_adapter,
)
from lab_copilot_gateway.wallac import (
    WallacAdapterError,
    get_wallac_adapter,
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


class ElabftwMintBody(BaseModel):
    """Request body for POST /elabftw/mint_context_token (C11 launcher).

    Caller identifies itself via ``keycloak_subject`` and/or
    ``librechat_user_id`` and requests a short-lived token scoped to one
    experiment.  The gateway resolves the caller to a mapped eLabFTW
    identity via the identity mapper (fails closed if unmapped), then mints
    a token bound to ``experiment_id`` + the resolved identity.

    Authentication: in C11 (Phase 1 dev / Tailscale) the caller
    self-attests its ``keycloak_subject``/``librechat_user_id`` and the
    identity mapper is the trust boundary (only pre-registered mappings
    resolve).  C14 wraps the endpoint with Keycloak session cookie
    verification; until then, the mapper is the auth gate.

    ``requested_ttl_seconds`` is clamped server-side; callers cannot
    exceed ``_MAX_MINT_TTL_SECONDS``.
    """

    experiment_id: int
    keycloak_subject: str | None = None
    librechat_user_id: str | None = None
    requested_ttl_seconds: int | None = None


class ElabftwDraftBody(BaseModel):
    """Request body for POST /elabftw/draft_experiment_update (C09).

    Caller supplies the signed context token, an approval_id issued via
    POST /approval/request bound to the exact args the LLM is about to
    draft, the identity fields used to resolve the mapped user, and the
    proposed draft args (title/body) the approval was issued for.  The
    adapter verifies the approval token, consumes it, and writes the
    draft downstream.

    Security: there is NO separate ``target_title`` field.  The title is
    sourced from ``approval_args['title']`` so the approval-token hash
    binds it; callers cannot override the title without breaking the hash.
    """

    context_token: str
    approval_id: str
    approval_args: dict[str, Any] = Field(default_factory=dict)
    keycloak_subject: str | None = None
    librechat_user_id: str | None = None
    conversation_id: str | None = None
    request_id: str | None = None
    provider: str | None = None
    model_id: str | None = None


class ElabftwAmendBody(BaseModel):
    """Request body for POST /elabftw/amend_my_experiment_after_approval (C09).

    Caller supplies the signed context token, an approval_id issued via
    POST /approval/request bound to the exact amendment args, the
    identity fields used to resolve the mapped user, the amendment HTML,
    and (optionally) an attachment to upload alongside the amendment.  The
    adapter verifies the approval, consumes it, checks the target experiment's
    state (append-only enforcement), appends the amendment, optionally
    uploads the attachment, then writes provenance (audit_action_id) back
    into the experiment metadata.
    """

    context_token: str
    approval_id: str
    approval_args: dict[str, Any] = Field(default_factory=dict)
    keycloak_subject: str | None = None
    librechat_user_id: str | None = None
    conversation_id: str | None = None
    request_id: str | None = None
    provider: str | None = None
    model_id: str | None = None
    amendment_html: str = ""
    attachment_filename: str | None = None
    attachment_b64: str | None = None  # base64-encoded bytes
    attachment_comment: str = ""


class InvokeBody(BaseModel):
    """Request body for POST /invoke (C13 — LibreChat tool surface).

    Single dispatch entry point used by the LibreChat custom-endpoint
    service (see copilot/librechat-custom-endpoint/) to invoke a cur-
    ated gateway tool by name.  The body carries:

        * ``tool_name``    — must match an entry in the C06 registry;
          anything else is rejected with ``tool_not_registered``
          (C13 acceptance check #4: gateway rejects direct privileged
          calls not in tool registry).
        * ``context_token`` — signed token from the eLabFTW launcher
          (C11 mint endpoint); adapter verifies + binds identity.
        * ``args``          — tool-specific args (title, amendment_html,
          experiment_id carryover, etc.).
        * ``approval_id``   — required for tier-4 (mutating) tools; the
          adapter consumes it after hashing ``args``.
        * identity / provenance fields — mirror the per-tool endpoints
          so audit records carry the same metadata.

    The endpoint is a thin dispatcher: it looks up the tool in the
    registry, refuses anything that is not registered, then routes to
    the appropriate adapter by ``tool.adapter`` + ``tool.name``.  Tools
    whose adapter is not yet implemented (opencloning.*, wallac.*,
    bentolab.* — C16+) return ``adapter_not_implemented`` rather than
    404 so the LibreChat side gets a structured error it can surface.
    """

    tool_name: str
    context_token: str
    args: dict[str, Any] = Field(default_factory=dict)
    approval_id: str | None = None
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
    register_auth_exception_handler(api)

    # Eagerly initialize audit store, policy engine, identity mapper,
    # approval store, eLabFTW read adapter (C08), eLabFTW write adapter
    # (C09), and OpenCloning adapter (C16) so /health reflects state.
    audit_store = get_audit_store()
    policy_engine = get_policy_engine()
    identity_mapper = get_identity_mapper()
    approval_store = get_approval_store()
    elabftw_adapter = get_elabftw_read_adapter()
    # Trigger construction of the write adapter singleton (no binding needed
    # here, but the call ensures /health shows the configured state).
    get_elabftw_write_adapter()
    opencloning_adapter = get_opencloning_adapter()
    wallac_adapter = get_wallac_adapter()

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
            "opencloning": (
                "configured"
                if opencloning_adapter.client is not None
                else "not_configured"
            ),
            "wallac": (
                "configured" if wallac_adapter.client is not None else "not_configured"
            ),
            "bentolab": "not_configured",
        }
        deps.update(_identity_backend_status(identity_mapper))
        deps.update(_approval_backend_status(approval_store))
        deps["tool_count"] = len(get_tool_registry().list())

        # C14 auth status
        auth_cfg = get_auth_config()
        auth_status: dict[str, object] = {
            "auth_mode": "dev" if not auth_cfg.verify_enabled else "jwt",
            "verify_enabled": auth_cfg.verify_enabled,
        }
        if auth_cfg.verify_enabled:
            try:
                jwks_status = get_jwks_cache().status()
                auth_status["jwks"] = jwks_status
            except Exception:
                auth_status["jwks"] = {"error": "unavailable"}
        deps["auth"] = auth_status

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
    def append_audit(
        body: AuditBody,
        principal: AuthenticatedPrincipal = Depends(verified_principal),  # noqa: ARG001
    ) -> dict[str, object]:
        record = AuditRecord(**body.model_dump())
        created_at = audit_store.append(record)
        return {"action_id": record.action_id, "created_at": created_at}

    @api.get("/audit/{action_id}")
    def get_audit(
        action_id: str,
        principal: AuthenticatedPrincipal = Depends(verified_principal),  # noqa: ARG001
    ) -> dict[str, object] | None:
        return audit_store.get(action_id)

    @api.post("/policy/evaluate")
    def evaluate_policy(
        body: PolicyRequestBody,
        principal: AuthenticatedPrincipal = Depends(verified_principal),  # noqa: ARG001
    ) -> dict[str, object]:
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
    def resolve_identity(
        body: IdentityResolveBody,
        principal: AuthenticatedPrincipal = Depends(verified_principal),
    ) -> dict[str, object]:
        identity = identity_mapper.map(
            keycloak_subject=principal.keycloak_subject,
            librechat_user_id=body.librechat_user_id,
        )
        if identity is None:
            # Missing mapping fails closed — caller (LibreChat) must treat
            # this as "no lab access" and the policy engine separately denies
            # any subsequent /policy/evaluate call (user_id would be None).
            return {"mapped": False}
        return _identity_to_dict(identity)

    @api.post("/approval/request")
    def request_approval(
        body: ApprovalRequestBody,
        principal: AuthenticatedPrincipal = Depends(verified_principal),  # noqa: ARG001
    ) -> dict[str, object]:
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
    def consume_approval(
        body: ApprovalConsumeBody,
        principal: AuthenticatedPrincipal = Depends(verified_principal),  # noqa: ARG001
    ) -> dict[str, object]:
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
    def get_approval(
        approval_id: str,
        principal: AuthenticatedPrincipal = Depends(verified_principal),  # noqa: ARG001
    ) -> dict[str, object] | None:
        """Fetch one approval row by id (admin / test reads)."""
        record = approval_store.get(approval_id)
        if record is None:
            return None
        return record.to_dict()

    @api.post("/elabftw/read_current_experiment")
    def elabftw_read_current_experiment(
        body: ElabftwReadBody,
        principal: AuthenticatedPrincipal = Depends(verified_principal),
    ) -> dict[str, object]:
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
            keycloak_subject=principal.keycloak_subject,
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

    @api.post("/elabftw/mint_context_token")
    def elabftw_mint_context_token(
        body: ElabftwMintBody,
        principal: AuthenticatedPrincipal = Depends(verified_principal),
    ) -> dict[str, object]:
        """Mint a short-lived context token for the C11 launcher.

        Resolves the caller to a mapped eLabFTW identity via the identity
        mapper (fail-closed if unmapped), then mints a token bound to
        ``experiment_id`` + the resolved identity.  Returns the token and
        its ``expires_at`` ISO timestamp.  TTL is clamped server-side.

        The token authorizes downstream reads/writes (C08/C09) on this
        experiment only, scoped to the resolved user.  The launcher opens
        LibreChat with the token in the URL.

        In C11, trust is the identity mapper (callers self-attest their
        ``keycloak_subject``/``librechat_user_id``; only pre-registered
        mappings resolve).  C14 wraps this endpoint with Keycloak
        session cookie verification layered on top.
        """
        # Fail-closed auth gate: identity mapper must resolve the caller.
        # Either keycloak_subject or librechat_user_id may be the lookup key
        # depending on the mapper backend; both-None is always unmapped.
        mapped = identity_mapper.map(
            keycloak_subject=principal.keycloak_subject,
            librechat_user_id=body.librechat_user_id,
        )
        if mapped is None:
            return {
                "ok": False,
                "reason": "unmapped_caller",
                "message": ("caller did not resolve to a mapped eLabFTW identity"),
            }

        if body.experiment_id <= 0:
            return {
                "ok": False,
                "reason": "invalid_experiment_id",
                "message": "experiment_id must be a positive integer",
            }

        token, expires_at = mint_token_for_identity(
            experiment_id=body.experiment_id,
            mapped_elabftw_user_id=mapped.elabftw_user_id,
            keycloak_subject=principal.keycloak_subject,
            librechat_user_id=body.librechat_user_id,
            ttl_seconds=body.requested_ttl_seconds,
        )
        return {
            "ok": True,
            "context_token": token,
            "expires_at": expires_at,
            "experiment_id": body.experiment_id,
            "mapped_elabftw_user_id": mapped.elabftw_user_id,
        }

    @api.post("/elabftw/draft_experiment_update")
    def elabftw_draft_experiment_update(
        body: ElabftwDraftBody,
        principal: AuthenticatedPrincipal = Depends(verified_principal),
    ) -> dict[str, object]:
        """Create a new draft experiment through the C09 write adapter.

        Orchestrates token verification → identity resolution → policy
        decision (tier 4 bounded write) → approval token consume →
        downstream ``create_experiment`` → provenance writeback → audit
        record.  Returns the new experiment id on success or an ``error``
        payload on any failure (invalid token, unmapped caller, policy
        denied, approval_denied, append_only_violation, client_error).
        """
        adapter = get_elabftw_write_adapter()
        mapped_identity = identity_mapper.map(
            keycloak_subject=principal.keycloak_subject,
            librechat_user_id=body.librechat_user_id,
        )
        try:
            result = adapter.draft_experiment_update(
                context_token=body.context_token,
                approval_id=body.approval_id,
                approval_args=body.approval_args,
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
        return {"ok": True, "write": result.to_dict()}

    @api.post("/elabftw/amend_my_experiment_after_approval")
    def elabftw_amend_my_experiment_after_approval(
        body: ElabftwAmendBody,
        principal: AuthenticatedPrincipal = Depends(verified_principal),
    ) -> dict[str, object]:
        """Append an approved amendment section to the user's experiment.

        Orchestrates token verification → identity resolution → policy
        decision (tier 4 bounded write) → append-only state check →
        approval token consume → append amendment + optional attachment →
        provenance writeback → audit record.  Returns the amendment
        outcome on success or an ``error`` payload on any failure.
        """
        import base64

        # Decode attachment (if provided) from base64 to bytes.  Malformed
        # base64 returns a clean error rather than a 500.
        attachment_data: bytes | None = None
        if body.attachment_b64 is not None and body.attachment_filename:
            try:
                attachment_data = base64.b64decode(body.attachment_b64)
            except Exception as exc:
                return {
                    "ok": False,
                    "reason": "client_error",
                    "message": f"attachment_b64 is not valid base64: {exc}",
                }

        adapter = get_elabftw_write_adapter()
        mapped_identity = identity_mapper.map(
            keycloak_subject=principal.keycloak_subject,
            librechat_user_id=body.librechat_user_id,
        )
        try:
            result = adapter.amend_my_experiment_after_approval(
                context_token=body.context_token,
                approval_id=body.approval_id,
                approval_args=body.approval_args,
                mapped_identity=mapped_identity,
                conversation_id=body.conversation_id,
                request_id=body.request_id,
                keycloak_subject=body.keycloak_subject,
                librechat_user_id=body.librechat_user_id,
                provider=body.provider,
                model_id=body.model_id,
                amendment_html=body.amendment_html,
                attachment_filename=body.attachment_filename,
                attachment_data=attachment_data,
                attachment_comment=body.attachment_comment,
            )
        except ElabftwAdapterError as exc:
            return {"ok": False, **exc.to_dict()}
        return {"ok": True, "write": result.to_dict()}

    # --- C13: LibreChat tool-surface dispatcher -------------------------
    #
    # POST /invoke is the single entry point used by the LibreChat
    # custom-endpoint service (copilot/librechat-custom-endpoint/).  It
    # enforces the C06 tool registry — anything not in the registry is
    # rejected with tool_not_registered (C13 acceptance #4).  Routing
    # then fans out by adapter:
    #
    #   elabftw.read_current_experiment
    #     → ElabftwReadAdapter.read_current_experiment
    #   elabftw.draft_experiment_update
    #     → ElabftwWriteAdapter.draft_experiment_update
    #   elabftw.amend_my_experiment_after_approval
    #     → ElabftwWriteAdapter.amend_my_experiment_after_approval
    #   opencloning.*, wallac.*, bentolab.*
    #     → adapter_not_implemented (those adapters land in C16+)
    #
    # Returns a uniform {ok, tool_name, result? | reason, message?}
    # shape so the custom-endpoint translator can map it into OpenAI's
    # chat-completions tool-call response without per-tool branching.

    @api.post("/invoke")
    def invoke(
        body: InvokeBody,
        principal: AuthenticatedPrincipal = Depends(verified_principal),
    ) -> dict[str, object]:
        """Invoke a registered tool by name (C13 LibreChat surface).

        Acceptance #4 (gateway rejects direct privileged calls not in
        tool registry): an unregistered ``tool_name`` returns
        ``ok:false`` with reason ``tool_not_registered`` and HTTP 200
        (the custom-endpoint service treats non-200 as transport
        failure; structured errors stay in-band).
        """
        registry = get_tool_registry()
        tool = registry.find(body.tool_name)
        if tool is None:
            return {
                "ok": False,
                "tool_name": body.tool_name,
                "reason": "tool_not_registered",
                "message": (
                    f"tool {body.tool_name!r} is not in the gateway registry; "
                    "LibreChat may only invoke curated C06 tools"
                ),
            }

        # Dispatch by adapter.  Each branch reuses the same adapter
        # call the per-tool endpoint makes so the C08/C09 invariants
        # (token verify, identity binding, policy, approval consume,
        # audit) are preserved exactly.
        mapped_identity = identity_mapper.map(
            keycloak_subject=principal.keycloak_subject,
            librechat_user_id=body.librechat_user_id,
        )

        if tool.adapter == "elabftw":
            # Reuse the per-tool endpoint logic by calling the
            # underlying adapter directly.  This keeps /invoke and the
            # per-tool endpoints in lockstep — no separate code path
            # for the same tool.
            try:
                if tool.name == "elabftw.read_current_experiment":
                    adapter = get_elabftw_read_adapter()
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
                    return {
                        "ok": True,
                        "tool_name": tool.name,
                        "result": result.to_dict(),
                    }
                elif tool.name == "elabftw.draft_experiment_update":
                    adapter = get_elabftw_write_adapter()
                    result = adapter.draft_experiment_update(
                        context_token=body.context_token,
                        approval_id=body.approval_id or "",
                        approval_args=body.args,
                        mapped_identity=mapped_identity,
                        conversation_id=body.conversation_id,
                        request_id=body.request_id,
                        keycloak_subject=body.keycloak_subject,
                        librechat_user_id=body.librechat_user_id,
                        provider=body.provider,
                        model_id=body.model_id,
                    )
                    return {
                        "ok": True,
                        "tool_name": tool.name,
                        "result": result.to_dict(),
                    }
                elif tool.name == "elabftw.amend_my_experiment_after_approval":
                    import base64

                    attachment_data: bytes | None = None
                    attachment_filename = body.args.get("attachment_filename")
                    attachment_b64 = body.args.get("attachment_b64")
                    if attachment_b64 and attachment_filename:
                        try:
                            attachment_data = base64.b64decode(attachment_b64)
                        except Exception as exc:
                            return {
                                "ok": False,
                                "tool_name": tool.name,
                                "reason": "client_error",
                                "message": f"attachment_b64 is not valid base64: {exc}",
                            }
                    adapter = get_elabftw_write_adapter()
                    result = adapter.amend_my_experiment_after_approval(
                        context_token=body.context_token,
                        approval_id=body.approval_id or "",
                        approval_args=body.args,
                        mapped_identity=mapped_identity,
                        conversation_id=body.conversation_id,
                        request_id=body.request_id,
                        keycloak_subject=body.keycloak_subject,
                        librechat_user_id=body.librechat_user_id,
                        provider=body.provider,
                        model_id=body.model_id,
                        amendment_html=body.args.get("amendment_html", ""),
                        attachment_filename=attachment_filename,
                        attachment_data=attachment_data,
                        attachment_comment=body.args.get("attachment_comment", ""),
                    )
                    return {
                        "ok": True,
                        "tool_name": tool.name,
                        "result": result.to_dict(),
                    }
                else:
                    # elabftw adapter but no known tool mapping —
                    # shouldn't happen (registry only contains the
                    # three above), but fail closed just in case.
                    return {
                        "ok": False,
                        "tool_name": tool.name,
                        "reason": "tool_not_dispatched",
                        "message": (
                            f"tool {tool.name!r} is in the elabftw adapter but "
                            "has no /invoke dispatch path"
                        ),
                    }
            except ElabftwAdapterError as exc:
                return {"ok": False, "tool_name": tool.name, **exc.to_dict()}

        if tool.adapter == "opencloning":
            try:
                adapter = get_opencloning_adapter()
                if tool.name == "opencloning.parse_sequence_file":
                    result = adapter.parse_sequence_file(
                        context_token=body.context_token,
                        file_content=body.args.get("file_content", ""),
                        file_format=body.args.get("file_format", "genbank"),
                        mapped_identity=mapped_identity,
                        conversation_id=body.conversation_id,
                        request_id=body.request_id,
                        keycloak_subject=body.keycloak_subject,
                        librechat_user_id=body.librechat_user_id,
                        provider=body.provider,
                        model_id=body.model_id,
                    )
                    return {
                        "ok": True,
                        "tool_name": tool.name,
                        "result": result.to_dict(),
                    }
                elif tool.name == "opencloning.manual_sequence":
                    result = adapter.manual_sequence(
                        context_token=body.context_token,
                        sequence=body.args.get("sequence", ""),
                        circular=body.args.get("circular", False),
                        mapped_identity=mapped_identity,
                        conversation_id=body.conversation_id,
                        request_id=body.request_id,
                        keycloak_subject=body.keycloak_subject,
                        librechat_user_id=body.librechat_user_id,
                        provider=body.provider,
                        model_id=body.model_id,
                    )
                    return {
                        "ok": True,
                        "tool_name": tool.name,
                        "result": result.to_dict(),
                    }
                elif tool.name == "opencloning.oligo_hybridization":
                    result = adapter.oligo_hybridization(
                        context_token=body.context_token,
                        forward_oligo=body.args.get("forward_oligo", ""),
                        reverse_oligo=body.args.get("reverse_oligo", ""),
                        minimal_annealing=body.args.get("minimal_annealing", 20),
                        mapped_identity=mapped_identity,
                        conversation_id=body.conversation_id,
                        request_id=body.request_id,
                        keycloak_subject=body.keycloak_subject,
                        librechat_user_id=body.librechat_user_id,
                        provider=body.provider,
                        model_id=body.model_id,
                    )
                    return {
                        "ok": True,
                        "tool_name": tool.name,
                        "result": result.to_dict(),
                    }
                elif tool.name == "opencloning.simulate_assembly":
                    result = adapter.simulate_assembly(
                        context_token=body.context_token,
                        fragments=body.args.get("fragments", []),
                        assembly_config=body.args.get("assembly_config", {}),
                        mapped_identity=mapped_identity,
                        conversation_id=body.conversation_id,
                        request_id=body.request_id,
                        keycloak_subject=body.keycloak_subject,
                        librechat_user_id=body.librechat_user_id,
                        provider=body.provider,
                        model_id=body.model_id,
                    )
                    return {
                        "ok": True,
                        "tool_name": tool.name,
                        "result": result.to_dict(),
                    }
                elif tool.name == "opencloning.writeback_artifact":
                    result = adapter.writeback_artifact(
                        context_token=body.context_token,
                        approval_id=body.approval_id or "",
                        approval_args=body.args,
                        mapped_identity=mapped_identity,
                        conversation_id=body.conversation_id,
                        request_id=body.request_id,
                        keycloak_subject=body.keycloak_subject,
                        librechat_user_id=body.librechat_user_id,
                        provider=body.provider,
                        model_id=body.model_id,
                    )
                    return {
                        "ok": True,
                        "tool_name": tool.name,
                        "result": result.to_dict(),
                    }
                else:
                    return {
                        "ok": False,
                        "tool_name": tool.name,
                        "reason": "tool_not_dispatched",
                        "message": (
                            f"tool {tool.name!r} is in the opencloning adapter "
                            "but has no /invoke dispatch path"
                        ),
                    }
            except OpenCloningAdapterError as exc:
                return {"ok": False, "tool_name": tool.name, **exc.to_dict()}
            except ElabftwAdapterError as exc:
                # OpenCloning adapter reuses eLabFTW's InvalidContextToken and
                # UnmappedCaller, so those errors surface as ElabftwAdapterError
                # subclasses. Map them to the same {ok:false, reason, message} shape.
                return {"ok": False, "tool_name": tool.name, **exc.to_dict()}

        if tool.adapter == "wallac":
            try:
                adapter = get_wallac_adapter()
                if tool.name == "wallac.get_status":
                    result = adapter.get_status(
                        context_token=body.context_token,
                        mapped_identity=mapped_identity,
                        conversation_id=body.conversation_id,
                        request_id=body.request_id,
                        keycloak_subject=body.keycloak_subject,
                        librechat_user_id=body.librechat_user_id,
                        provider=body.provider,
                        model_id=body.model_id,
                    )
                    return {
                        "ok": True,
                        "tool_name": tool.name,
                        "result": result.to_dict(),
                    }
                else:
                    return {
                        "ok": False,
                        "tool_name": tool.name,
                        "reason": "tool_not_dispatched",
                        "message": (
                            f"tool {tool.name!r} is in the wallac adapter "
                            "but has no /invoke dispatch path"
                        ),
                    }
            except WallacAdapterError as exc:
                return {"ok": False, "tool_name": tool.name, **exc.to_dict()}
            except ElabftwAdapterError as exc:
                # Wallac adapter reuses eLabFTW's InvalidContextToken and
                # UnmappedCaller, so those errors surface as ElabftwAdapterError
                # subclasses. Map them to the same {ok:false, reason, message} shape.
                return {"ok": False, "tool_name": tool.name, **exc.to_dict()}

        # Adapters that don't exist yet (C19+).  Return a structured
        # error so the LibreChat side can surface a clear "this tool
        # is not wired up yet" message instead of a transport 404.
        return {
            "ok": False,
            "tool_name": tool.name,
            "reason": "adapter_not_implemented",
            "message": (
                f"tool {tool.name!r} is in the registry but its "
                f"{tool.adapter!r} adapter is not implemented yet "
                "(lands in C19+)"
            ),
        }

    return api


app = create_app()
