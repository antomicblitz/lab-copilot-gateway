"""FastAPI application for the lab copilot gateway scaffold."""

from __future__ import annotations

import os
import uuid as _uuid
import hashlib
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Response
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
from lab_copilot_gateway.audit import AuditRecord, AuditStore, get_audit_store
from lab_copilot_gateway.artifact_bundle import (
    ArtifactBundleStore,
    ArtifactContextMismatch,
    ArtifactExpired,
    ArtifactHashMismatch,
    ArtifactMissing,
    ArtifactSizeMismatch,
    get_artifact_bundle_store,
)
from lab_copilot_gateway.config import (
    KILL_SWITCH_CATEGORY_NAMES,
    get_public_config,
)
from lab_copilot_gateway.elabftw import (
    ElabftwAdapterError,
    ElabftwReadAdapter,
    InvalidContextToken,
    mint_token_for_identity,
    get_elabftw_read_adapter,
    get_elabftw_write_adapter,
    verify_context_token,
)
from lab_copilot_gateway.bentolab import (
    BentoLabAdapterError,
    get_bentolab_adapter,
)
from lab_copilot_gateway.opencloning import (
    OpenCloningAdapter,
    OpenCloningAdapterError,
    OpenCloningResult,
    get_opencloning_adapter,
)
from lab_copilot_gateway.opencloning_artifacts import normalize_opencloning_artifacts
from lab_copilot_gateway.opencloning_previews import get_preview_store
from lab_copilot_gateway.protocol_lookup import ProtocolLookupService
from lab_copilot_gateway.wallac import (
    WallacAdapter,
    WallacAdapterError,
    get_wallac_adapter,
)
from lab_copilot_gateway.identity import (
    DbIdentityMapper,
    IdentityMapper,
    MappedIdentity,
    get_identity_mapper,
)
from lab_copilot_gateway.policy import (
    PolicyEngine,
    PolicyRequest,
    Tier,
    get_policy_engine,
    set_kill_category,
)
from lab_copilot_gateway.tools import get_tool_registry, list_tools
from lab_copilot_gateway.plan import Plan, PlanValidationError
from lab_copilot_gateway.plan_executor import get_plan_executor


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


class KillSwitchBody(BaseModel):
    """Request body for POST /policy/kill_switch."""

    switch: str
    enabled: bool


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
    record_type: str = "experiment"  # "experiment" or "resource" (database item)


class ElabftwCreateBody(BaseModel):
    """Request body for POST /elabftw/create_experiment (C__FIX2__).

    Caller identifies itself via ``keycloak_subject``/``librechat_user_id``.
    The gateway resolves the identity via the mapper, then creates a new
    eLabFTW experiment with the given ``title`` (defaults to empty string
    so the user can name it later).
    """

    keycloak_subject: str | None = None
    librechat_user_id: str | None = None
    title: str | None = None


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


class ElabftwEditBody(BaseModel):
    """Request body for POST /elabftw/edit_experiment_section (C35).

    Direct edit (not append-only): the approved ``new_body`` REPLACES the
    experiment body. The approval_args MUST carry ``old_body_hash`` (SHA-256
    hex of the body at approval time) + ``new_body``; the adapter re-reads
    the current body at execution and refuses if the hash drifted (stale-edit
    guard). Rollback is via eLabFTW revision history.
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


class PlanExecuteBody(BaseModel):
    """Request body for POST /plan/execute (C39)."""

    plan: dict[str, Any]
    approval_id: str | None = None  # C29: optional for autonomous plans
    context_token: str = ""
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


def _download_filename(filename: str) -> str:
    """Return a conservative Content-Disposition filename."""
    safe = filename.replace("/", "_").replace("\\", "_").replace('"', "_")
    return safe or "artifact.bin"


def _redact_opencloning_file_content(value: Any) -> Any:
    """Replace OpenCloning file_content payloads with hash/size metadata."""
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if key == "file_content" and isinstance(item, str):
                raw = item.encode("utf-8")
                redacted[key] = {
                    "redacted": True,
                    "size_bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            else:
                redacted[key] = _redact_opencloning_file_content(item)
        return redacted
    if isinstance(value, list):
        return [_redact_opencloning_file_content(item) for item in value]
    return value


_ARTIFACT_MANIFEST_RESERVED_KEYS = {
    "artifact_id",
    "plan_id",
    "plan_hash",
    "filename",
    "mime_type",
    "size_bytes",
    "sha256",
    "created_at",
    "expires_at",
}


def _opencloning_invoke_success(
    *,
    tool_name: str,
    result: OpenCloningResult,
    context_token: str,
    request_id: str | None,
    artifact_store: ArtifactBundleStore,
) -> dict[str, object]:
    """Return OpenCloning /invoke output without raw sequence bytes."""
    raw = result.to_dict()
    raw_result = raw.get("result", {}) if isinstance(raw.get("result"), dict) else {}
    plan_id = f"invoke-{request_id or result.audit_action_id}"
    plan_hash = compute_args_hash(
        {"tool_name": tool_name, "audit_action_id": result.audit_action_id}
    )

    # Capture preview refs for OVE: store sequence GenBank/Fasta payloads
    # in the short-lived preview store so the widget can fetch them on-demand.
    token_hash = hashlib.sha256(context_token.encode()).hexdigest()
    preview_refs = _capture_opencloning_preview_refs(
        raw_result,
        run_id=plan_id,
        step_id=result.audit_action_id,
        context_token_hash=token_hash,
    )

    normalized = normalize_opencloning_artifacts(
        raw_result,
        plan_id=plan_id,
        plan_hash=plan_hash,
        operation_label=tool_name,
    )

    binding: dict[str, object] = {}
    try:
        claims = verify_context_token(context_token)
        binding = {
            "record_type": claims.record_type,
            "record_id": str(claims.experiment_id),
            "mapped_elabftw_user_id": claims.mapped_elabftw_user_id,
        }
    except InvalidContextToken:
        # The adapter already verified the token before returning success.
        # If this defensive re-verify fails, keep manifests but skip storing
        # downloadable bytes rather than creating an unbound artifact.
        binding = {}

    manifests: list[dict[str, object]] = []
    for artifact in normalized.artifacts:
        artifact_id = str(artifact.get("artifact_id") or "")
        data = normalized.artifact_bytes.get(artifact_id)
        if not artifact_id or data is None or not binding:
            manifests.append(artifact)
            continue
        extra = {
            key: value
            for key, value in artifact.items()
            if key not in _ARTIFACT_MANIFEST_RESERVED_KEYS
        }
        manifests.append(
            artifact_store.put(
                plan_id=plan_id,
                plan_hash=plan_hash,
                artifact_id=artifact_id,
                filename=str(artifact.get("filename") or "artifact.bin"),
                mime_type=str(artifact.get("mime_type") or "application/octet-stream"),
                data=data,
                binding=binding,
                manifest_extra=extra,
            )
        )

    artifact_payload = normalized.to_event_payload()
    artifact_payload.update(
        {
            "type": "lab_copilot_opencloning_artifacts",
            "plan_id": plan_id,
            "plan_hash": plan_hash,
            "artifacts": manifests,
        }
    )

    redacted_result = _redact_opencloning_file_content(raw)
    # Annotate each sequence in the redacted result with its preview_ref
    # so the widget can pass it to the fetch endpoint.
    _inject_preview_refs_into_result(redacted_result, preview_refs)

    return {
        "ok": True,
        "tool_name": tool_name,
        "result": redacted_result,
        "opencloning_artifacts": artifact_payload,
    }


def _capture_opencloning_preview_refs(
    raw_result: dict[str, Any],
    run_id: str,
    step_id: str,
    context_token_hash: str,
) -> dict[int, str]:
    """Store OpenCloning sequence payloads in the preview store.

    Delegates to the adapter's ``capture_preview_refs`` method to
    extract file_content from each sequence and store it in the
    short-lived preview store.
    """
    adapter = get_opencloning_adapter()
    return adapter.capture_preview_refs(
        result=raw_result,
        run_id=run_id,
        step_id=step_id,
        context_token_hash=context_token_hash,
    )


def _inject_preview_refs_into_result(value: Any, preview_refs: dict[int, str]) -> None:
    """Mutate a result dict to carry preview_ref on each sequence."""
    if isinstance(value, dict):
        sequences = value.get("sequences")
        if isinstance(sequences, list):
            for seq in sequences:
                if not isinstance(seq, dict):
                    continue
                seq_id = seq.get("id")
                if seq_id is not None and seq_id in preview_refs:
                    seq["preview_ref"] = preview_refs[seq_id]
        for v in value.values():
            _inject_preview_refs_into_result(v, preview_refs)
    elif isinstance(value, list):
        for item in value:
            _inject_preview_refs_into_result(item, preview_refs)


def _invoke_elabftw_tool(
    tool: Any,
    body: InvokeBody,
    mapped_identity: MappedIdentity | None,
) -> dict[str, object]:
    """Dispatch an elabftw adapter tool invocation."""
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
        elif tool.name == "elabftw.search_my_experiments":
            adapter = get_elabftw_read_adapter()
            result = adapter.search_my_experiments(
                query=body.args.get("query", ""),
                limit=body.args.get("limit", 20),
                offset=body.args.get("offset", 0),
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
        elif tool.name == "elabftw.read_experiment_by_id":
            adapter = get_elabftw_read_adapter()
            result = adapter.read_experiment_by_id(
                experiment_id=int(
                    body.args.get("experiment_id", body.args.get("id", 0))
                ),
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
        elif tool.name == "elabftw.search_items":
            adapter = get_elabftw_read_adapter()
            result = adapter.search_items(
                query=body.args.get("query", ""),
                limit=body.args.get("limit", 20),
                offset=body.args.get("offset", 0),
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
        elif tool.name == "elabftw.read_item_by_id":
            adapter = get_elabftw_read_adapter()
            result = adapter.read_item_by_id(
                item_id=int(body.args.get("item_id", body.args.get("id", 0))),
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
        elif tool.name == "elabftw.download_upload":
            return _invoke_elabftw_download_upload(tool, body)
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
            return _invoke_elabftw_amend(tool, body, mapped_identity)
        elif tool.name == "elabftw.edit_experiment_section":
            adapter = get_elabftw_write_adapter()
            result = adapter.edit_experiment_section(
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
                    f"tool {tool.name!r} is in the elabftw adapter but "
                    "has no /invoke dispatch path"
                ),
            }
    except ElabftwAdapterError as exc:
        return {"ok": False, "tool_name": tool.name, **exc.to_dict()}


def _resolve_download_record_id(
    body: InvokeBody,
) -> tuple[int, str, dict[str, object] | None]:
    """Resolve record_id and record_type for download_upload.

    Returns (record_id, record_type, error_dict).  If error_dict is not None,
    the caller should return it as the error response.
    """
    record_id = int(body.args.get("record_id", body.args.get("experiment_id", 0)))
    record_type = body.args.get("record_type", "experiments")
    if record_id == 0:
        if not body.context_token:
            return (
                0,
                "",
                {
                    "ok": False,
                    "reason": "missing_record_id",
                    "message": "Provide record_id (experiment or item id), or call read_current_experiment first.",
                },
            )
        try:
            claims = verify_context_token(body.context_token)
            record_id = claims.experiment_id
            record_type = "items" if claims.record_type == "resource" else "experiments"
        except Exception:
            return (
                0,
                "",
                {
                    "ok": False,
                    "reason": "invalid_context_token",
                    "message": "Could not resolve experiment id from context token. Provide record_id explicitly.",
                },
            )
    return record_id, record_type, None


def _invoke_elabftw_download_upload(
    tool: Any,
    body: InvokeBody,
) -> dict[str, object]:
    """Handle elabftw.download_upload tool invocation."""
    upload_id = int(body.args.get("upload_id", 0))
    record_id, record_type, error = _resolve_download_record_id(body)
    if error:
        error["tool_name"] = tool.name
        return error

    adapter = get_elabftw_read_adapter()
    if adapter.client is None:
        return {
            "ok": False,
            "tool_name": tool.name,
            "reason": "download_failed",
        }
    try_record_types = [record_type]
    if record_type not in ("items",):
        try_record_types.append("items")
    if record_type not in ("experiments",):
        try_record_types.append("experiments")

    content: str | None = None
    last_exc: Exception | None = None
    for rt in try_record_types:
        try:
            content = adapter.client.get_upload_content(rt, record_id, upload_id)
            break
        except Exception as exc:
            last_exc = exc
    if content is None:
        return {
            "ok": False,
            "tool_name": tool.name,
            "reason": "download_failed",
            "message": str(last_exc) if last_exc else "upload not found",
        }
    return {
        "ok": True,
        "tool_name": tool.name,
        "result": {
            "upload_id": upload_id,
            "content": content,
            "length": len(content),
        },
    }


def _invoke_elabftw_amend(
    tool: Any,
    body: InvokeBody,
    mapped_identity: MappedIdentity | None,
) -> dict[str, object]:
    """Handle elabftw.amend_my_experiment_after_approval tool invocation."""
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


def _invoke_opencloning_tool(
    tool: Any,
    body: InvokeBody,
    mapped_identity: MappedIdentity | None,
    artifact_store: ArtifactBundleStore,
) -> dict[str, object]:
    """Dispatch an opencloning adapter tool invocation."""
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
            return _opencloning_invoke_success(
                tool_name=tool.name,
                result=result,
                context_token=body.context_token,
                request_id=body.request_id,
                artifact_store=artifact_store,
            )
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
            return _opencloning_invoke_success(
                tool_name=tool.name,
                result=result,
                context_token=body.context_token,
                request_id=body.request_id,
                artifact_store=artifact_store,
            )
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
            return _opencloning_invoke_success(
                tool_name=tool.name,
                result=result,
                context_token=body.context_token,
                request_id=body.request_id,
                artifact_store=artifact_store,
            )
        elif tool.name == "opencloning.simulate_assembly":
            result = adapter.simulate_assembly(
                context_token=body.context_token,
                sequences=body.args.get("sequences", []),
                source=body.args.get(
                    "source", {"id": 0, "type": "GibsonAssemblySource"}
                ),
                mapped_identity=mapped_identity,
                conversation_id=body.conversation_id,
                request_id=body.request_id,
                keycloak_subject=body.keycloak_subject,
                librechat_user_id=body.librechat_user_id,
                provider=body.provider,
                model_id=body.model_id,
            )
            return _opencloning_invoke_success(
                tool_name=tool.name,
                result=result,
                context_token=body.context_token,
                request_id=body.request_id,
                artifact_store=artifact_store,
            )
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
            response: dict[str, object] = {
                "ok": True,
                "tool_name": tool.name,
                "result": result.to_dict(),
            }
            # Surface the validation bundle at the top level so the
            # orchestrator can emit it as an SSE event
            # (lab_copilot_validation_bundle) and the widget can display
            # validation status on the approval card.
            bundle = result.result.get("validation_bundle")
            if isinstance(bundle, dict):
                response["validation_bundle"] = bundle
            return response
        elif tool.name == "opencloning.call":
            result = adapter.call_endpoint(
                context_token=body.context_token,
                endpoint=body.args.get("endpoint", "/"),
                body=body.args.get("body", {}),
                mapped_identity=mapped_identity,
                conversation_id=body.conversation_id,
                request_id=body.request_id,
                keycloak_subject=body.keycloak_subject,
                librechat_user_id=body.librechat_user_id,
                provider=body.provider,
                model_id=body.model_id,
            )
            return _opencloning_invoke_success(
                tool_name=tool.name,
                result=result,
                context_token=body.context_token,
                request_id=body.request_id,
                artifact_store=artifact_store,
            )
        elif tool.name == "opencloning.search_parts":
            return _invoke_opencloning_search_parts(tool, body)
        elif tool.name == "opencloning.fetch_igem_part":
            return _invoke_opencloning_fetch_igem(tool, body)
        elif tool.name == "opencloning.lookup_protocol":
            return _invoke_opencloning_lookup_protocol(tool, body, adapter)
        elif tool.name == "protocols.validate_corpus":
            return _invoke_opencloning_validate_corpus(tool, body, adapter)
        else:
            return {
                "ok": False,
                "tool_name": tool.name,
                "reason": "tool_not_dispatched",
                "message": (
                    f"tool {tool.name!r} is in the opencloning adapter but "
                    "has no /invoke dispatch path"
                ),
            }
    except OpenCloningAdapterError as exc:
        return {"ok": False, "tool_name": tool.name, **exc.to_dict()}
    except WallacAdapterError as exc:
        return {"ok": False, "tool_name": tool.name, **exc.to_dict()}
    except ElabftwAdapterError as exc:
        return {"ok": False, "tool_name": tool.name, **exc.to_dict()}


def _invoke_opencloning_search_parts(
    tool: Any,
    body: InvokeBody,
) -> dict[str, object]:
    """Handle opencloning.search_parts — SynVectorDB semantic search."""
    from urllib.request import urlopen as _urlopen, Request as _Request
    import json as _json

    query = body.args.get("query", "")
    retmax = int(body.args.get("retmax", 5))

    svdb_url = "https://testsdb.sjtu.bio/tools/semantic_search_cf"
    payload = _json.dumps({"query": query, "limit": retmax}).encode()
    req = _Request(svdb_url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", "LabCopilot/1.0")
    with _urlopen(req, timeout=15) as resp:
        data = _json.loads(resp.read())

    results = []
    for match in data.get("matches", []):
        m = match.get("metadata", {})
        results.append(
            {
                "uid": match.get("id", ""),
                "name": m.get("name", ""),
                "source_collection": m.get("source_collection", ""),
                "source_name": m.get("source_name", ""),
                "type": m.get("type_level_1", ""),
                "subtype": m.get("type_level_2", ""),
                "score": match.get("score", 0),
            }
        )
    return {
        "ok": True,
        "tool_name": tool.name,
        "result": {"results": results, "count": len(results)},
    }


def _invoke_opencloning_fetch_igem(
    tool: Any,
    body: InvokeBody,
) -> dict[str, object]:
    """Handle opencloning.fetch_igem_part — iGEM Registry fetch."""
    from lab_copilot_gateway.igem_registry import (
        fetch_igem_part_as_genbank,
    )

    part_name = body.args.get("part_name", "")
    if not part_name:
        return {
            "ok": False,
            "tool_name": tool.name,
            "reason": "missing_arg",
            "message": "part_name is required",
        }
    try:
        genbank_str = fetch_igem_part_as_genbank(part_name)
    except ValueError as exc:
        return {
            "ok": False,
            "tool_name": tool.name,
            "reason": "client_error",
            "message": str(exc),
        }
    return {
        "ok": True,
        "tool_name": tool.name,
        "result": {
            "part_name": part_name,
            "genbank": genbank_str,
            "file_format": "genbank",
        },
    }


def _invoke_wallac_tool(
    tool: Any,
    body: InvokeBody,
    mapped_identity: MappedIdentity | None,
) -> dict[str, object]:
    """Dispatch a wallac adapter tool invocation."""
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
        elif tool.name == "wallac.call":
            result = adapter.call(
                context_token=body.context_token,
                mapped_identity=mapped_identity,
                method=body.args.get("method", "GET"),
                endpoint=body.args.get("endpoint", "/"),
                body=body.args.get("body"),
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
        elif tool.name == "wallac.propose_generated_protocol":
            result = adapter.propose_generated_protocol(
                context_token=body.context_token,
                mapped_identity=mapped_identity,
                protocol_spec=body.args.get("protocol_spec", {}),
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
        elif tool.name == "wallac.validate_generated_protocol":
            result = adapter.validate_generated_protocol(
                context_token=body.context_token,
                mapped_identity=mapped_identity,
                job_item_id=body.args.get("job_item_id") or 0,
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
        elif tool.name == "wallac.prepare_submission_package":
            result = adapter.prepare_submission_package(
                context_token=body.context_token,
                mapped_identity=mapped_identity,
                protocol_spec=body.args.get("protocol_spec", {}),
                validation_report=body.args.get("validation_report"),
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
        elif tool.name == "wallac.run":
            result = adapter.run(
                context_token=body.context_token,
                mapped_identity=mapped_identity,
                approval_id=body.approval_id or "",
                protocol_id=body.args.get("protocol_id") or 0,
                plate_id=body.args.get("plate_id"),
                plate_layout=body.args.get("plate_layout"),
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
        elif tool.name == "wallac.submit_generated_protocol":
            result = adapter.submit_generated_protocol(
                context_token=body.context_token,
                mapped_identity=mapped_identity,
                approval_id=body.approval_id or "",
                approval_args=body.args,
                job_item_id=body.args.get("job_item_id") or 0,
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
        elif tool.name == "wallac.bridge_status":
            result = adapter.bridge_status(
                job_id=body.args.get("job_id", ""),
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
        return {"ok": False, "tool_name": tool.name, **exc.to_dict()}


def _invoke_bentolab_tool(
    tool: Any,
    body: InvokeBody,
    mapped_identity: MappedIdentity | None,
) -> dict[str, object]:
    """Dispatch a bentolab adapter tool invocation."""
    try:
        adapter = get_bentolab_adapter()
        if tool.name == "bentolab.get_status":
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
        elif tool.name == "bentolab.validate_pcr_profile":
            result = adapter.validate_pcr_profile(
                context_token=body.context_token,
                mapped_identity=mapped_identity,
                profile=body.args.get("profile", {}),
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
        elif tool.name == "bentolab.dry_run_pcr_profile":
            result = adapter.dry_run_pcr_profile(
                context_token=body.context_token,
                mapped_identity=mapped_identity,
                profile=body.args.get("profile", {}),
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
        elif tool.name == "bentolab.submit_pcr_run":
            result = adapter.submit_pcr_run(
                context_token=body.context_token,
                mapped_identity=mapped_identity,
                approval_id=body.approval_id or "",
                approval_args=body.args,
                profile=body.args.get("profile", {}),
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
                    f"tool {tool.name!r} is in the bentolab adapter "
                    "but has no /invoke dispatch path"
                ),
            }
    except BentoLabAdapterError as exc:
        return {"ok": False, "tool_name": tool.name, **exc.to_dict()}
    except ElabftwAdapterError as exc:
        return {"ok": False, "tool_name": tool.name, **exc.to_dict()}


# Maps adapter name to its dispatch function for POST /invoke.
_INVOKE_DISPATCHERS: dict[str, Any] = {
    "elabftw": _invoke_elabftw_tool,
    "opencloning": _invoke_opencloning_tool,
    "wallac": _invoke_wallac_tool,
    "bentolab": _invoke_bentolab_tool,
}


def create_app() -> FastAPI:
    """Create the ASGI application."""
    service_name = "lab-copilot-gateway"
    api = FastAPI(title="Lab Copilot Gateway", version=__version__)
    register_auth_exception_handler(api)

    # Eagerly initialize all singleton dependencies.
    audit_store = get_audit_store()
    policy_engine = get_policy_engine()
    identity_mapper = get_identity_mapper()
    approval_store = get_approval_store()
    elabftw_adapter = get_elabftw_read_adapter()
    get_elabftw_write_adapter()
    opencloning_adapter = get_opencloning_adapter()
    wallac_adapter = get_wallac_adapter()
    artifact_store = get_artifact_bundle_store()

    # Register routes in dependency groups.
    _register_health_routes(
        api,
        service_name,
        audit_store,
        policy_engine,
        identity_mapper,
        approval_store,
        elabftw_adapter,
        opencloning_adapter,
        wallac_adapter,
    )
    _register_config_routes(api, service_name)
    _register_artifact_routes(api, artifact_store)
    _register_audit_routes(api, audit_store)
    _register_policy_routes(api, policy_engine, audit_store)
    _register_identity_routes(api, identity_mapper)
    _register_approval_routes(api, approval_store, identity_mapper)
    _register_elabftw_routes(api, identity_mapper)
    _register_elabftw_amend_route(api, identity_mapper)
    _register_invoke_route(api, identity_mapper, artifact_store)
    _register_plan_route(api, identity_mapper)
    _register_bentolab_route(api, identity_mapper)
    _register_opencloning_preview_routes(api)

    return api


def _register_opencloning_preview_routes(api: FastAPI) -> None:
    """Register /v1/opencloning/previews/{preview_ref} route."""

    @api.get("/v1/opencloning/previews/{preview_ref}")
    def fetch_opencloning_preview(
        preview_ref: str,
        x_lab_copilot_context_token: str = Header(...),
    ) -> dict[str, object]:
        store = get_preview_store()
        token_hash = hashlib.sha256(x_lab_copilot_context_token.encode()).hexdigest()
        payload = store.fetch(preview_ref, token_hash)
        if payload is None:
            raise HTTPException(
                status_code=404,
                detail="Preview not found or expired",
            )
        return {
            "preview_ref": payload.preview_ref,
            "run_id": payload.run_id,
            "step_id": payload.step_id,
            "sequence_id": payload.sequence_id,
            "file_format": payload.file_format,
            "file_content": payload.file_content,
            "sequence_length": payload.sequence_length,
            "is_circular": payload.is_circular,
        }


def _invoke_opencloning_lookup_protocol(
    tool: Any,
    body: InvokeBody,
    adapter: OpenCloningAdapter,
) -> dict[str, object]:
    """Handle opencloning.lookup_protocol — protocol lookup from eLabFTW."""
    elabftw_client = adapter.elabftw_client
    if elabftw_client is None:
        return {
            "ok": False,
            "tool_name": tool.name,
            "reason": "no_elabftw_client",
            "message": "eLabFTW client is not configured for protocol lookup",
        }
    service = ProtocolLookupService(elabftw_client)
    result = service.lookup(
        method_type=body.args.get("method_type", ""),
        reagent_name=body.args.get("reagent_name"),
        aliases=body.args.get("aliases"),
    )
    return {
        "ok": True,
        "tool_name": tool.name,
        "result": result.to_dict(),
    }


def _invoke_opencloning_validate_corpus(
    tool: Any,
    body: InvokeBody,
    adapter: OpenCloningAdapter,
) -> dict[str, object]:
    """Handle protocols.validate_corpus — validate protocol entries."""
    elabftw_client = adapter.elabftw_client
    if elabftw_client is None:
        return {
            "ok": False,
            "tool_name": tool.name,
            "reason": "no_elabftw_client",
            "message": "eLabFTW client is not configured for protocol validation",
        }
    service = ProtocolLookupService(elabftw_client)
    issues = service.validate_corpus()
    return {
        "ok": True,
        "tool_name": tool.name,
        "result": {"issues": issues, "count": len(issues)},
    }


def _register_config_routes(api: FastAPI, service_name: str) -> None:
    """Register /config/public and /tools routes."""

    @api.get("/config/public")
    def public_config() -> dict[str, object]:
        return get_public_config(service_name=service_name, version=__version__)

    @api.get("/tools")
    def tools() -> dict[str, list[dict[str, object]]]:
        return {"tools": list_tools()}


def _register_artifact_routes(
    api: FastAPI, artifact_store: ArtifactBundleStore
) -> None:
    """Register /v1/artifacts/{id}/download route."""

    @api.get("/v1/artifacts/{artifact_id}/download")
    def download_artifact(
        artifact_id: str,
        plan_id: str,
        plan_hash: str,
        artifact_sha256: str | None = None,
        artifact_size_bytes: int | None = None,
        context_token: str = Header("", alias="X-Lab-Copilot-Context-Token"),
        principal: AuthenticatedPrincipal = Depends(verified_principal),  # noqa: ARG001
    ) -> Response:
        try:
            claims = verify_context_token(context_token)
        except InvalidContextToken as exc:
            raise HTTPException(
                status_code=401,
                detail={"reason": "invalid_context_token", "message": exc.detail},
            ) from exc
        binding = {
            "record_type": claims.record_type,
            "record_id": str(claims.experiment_id),
            "mapped_elabftw_user_id": claims.mapped_elabftw_user_id,
        }
        if not artifact_sha256 or artifact_size_bytes is None:
            raise HTTPException(
                status_code=400,
                detail={
                    "reason": "missing_artifact_identity",
                    "message": "artifact_sha256 and artifact_size_bytes are required",
                },
            )
        try:
            artifact = artifact_store.get(
                plan_id=plan_id,
                plan_hash=plan_hash,
                artifact_id=artifact_id,
                expected_sha256=artifact_sha256,
                expected_size_bytes=artifact_size_bytes,
                binding=binding,
            )
        except ArtifactExpired as exc:
            raise HTTPException(
                status_code=410,
                detail={"reason": "artifact_expired", "message": str(exc)},
            ) from exc
        except ArtifactMissing as exc:
            raise HTTPException(
                status_code=404,
                detail={"reason": "artifact_missing", "message": str(exc)},
            ) from exc
        except ArtifactContextMismatch as exc:
            raise HTTPException(
                status_code=403,
                detail={"reason": "artifact_context_mismatch", "message": str(exc)},
            ) from exc
        except (ArtifactHashMismatch, ArtifactSizeMismatch) as exc:
            raise HTTPException(
                status_code=409,
                detail={"reason": "artifact_identity_mismatch", "message": str(exc)},
            ) from exc
        filename = _download_filename(artifact.filename)
        return Response(
            content=artifact.bytes_data,
            media_type=artifact.mime_type,
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Cache-Control": "no-store",
                "X-Lab-Copilot-Artifact-Sha256": artifact.sha256,
            },
        )


def _register_audit_routes(api: FastAPI, audit_store: AuditStore) -> None:
    """Register audit POST/GET routes."""

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


def _register_policy_routes(
    api: FastAPI, policy_engine: PolicyEngine, audit_store: AuditStore
) -> None:
    """Register /policy/evaluate and /policy/kill_switch routes."""

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

    @api.post("/policy/kill_switch")
    def set_kill_switch(
        body: KillSwitchBody,
        principal: AuthenticatedPrincipal = Depends(verified_principal),  # noqa: ARG001
    ) -> dict[str, object]:
        if body.switch not in KILL_SWITCH_CATEGORY_NAMES:
            return {
                "ok": False,
                "reason": "unknown_switch",
                "message": f"unknown kill switch category {body.switch!r}; valid switches: {sorted(KILL_SWITCH_CATEGORY_NAMES)}",
            }
        action_id = str(_uuid.uuid4())
        record = AuditRecord(
            action_id=action_id,
            tool_name="__kill_switch__",
            policy_decision="deny" if body.enabled else "allow",
            api_call_summary={"switch": body.switch, "enabled": body.enabled},
        )
        audit_store.append(record)
        set_kill_category(body.switch, body.enabled)
        return {
            "ok": True,
            "switch": body.switch,
            "enabled": body.enabled,
            "audit_action_id": action_id,
        }


def _register_identity_routes(api: FastAPI, identity_mapper: IdentityMapper) -> None:
    """Register /identity/resolve route."""

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
            return {"mapped": False}
        return _identity_to_dict(identity)


def _register_approval_routes(
    api: FastAPI, approval_store: ApprovalStore, identity_mapper: IdentityMapper
) -> None:
    """Register /approval/* routes."""

    @api.post("/approval/request")
    def request_approval(
        body: ApprovalRequestBody,
        principal: AuthenticatedPrincipal = Depends(verified_principal),  # noqa: ARG001
    ) -> dict[str, object]:
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
        record = approval_store.get(approval_id)
        if record is None:
            return None
        return record.to_dict()


def _register_elabftw_routes(api: FastAPI, identity_mapper: IdentityMapper) -> None:
    """Register all /elabftw/* routes."""

    @api.post("/elabftw/read_current_experiment")
    def elabftw_read_current_experiment(
        body: ElabftwReadBody,
        principal: AuthenticatedPrincipal = Depends(verified_principal),
    ) -> dict[str, object]:
        adapter = get_elabftw_read_adapter()
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
        mapped = identity_mapper.map(
            keycloak_subject=principal.keycloak_subject,
            librechat_user_id=body.librechat_user_id,
        )
        if mapped is None:
            return {
                "ok": False,
                "reason": "unmapped_caller",
                "message": "caller did not resolve to a mapped eLabFTW identity",
            }
        if body.experiment_id < 0:
            return {
                "ok": False,
                "reason": "invalid_experiment_id",
                "message": "experiment_id must be a non-negative integer "
                "(0 means no experiment context)",
            }
        token, expires_at = mint_token_for_identity(
            experiment_id=body.experiment_id,
            mapped_elabftw_user_id=mapped.elabftw_user_id,
            keycloak_subject=principal.keycloak_subject,
            librechat_user_id=body.librechat_user_id,
            ttl_seconds=body.requested_ttl_seconds,
            record_type=body.record_type,
        )
        return {
            "ok": True,
            "context_token": token,
            "expires_at": expires_at,
            "experiment_id": body.experiment_id,
            "mapped_elabftw_user_id": mapped.elabftw_user_id,
        }

    # --- C__FIX2__: create experiment endpoint (widget-initiated) ---------
    @api.post("/elabftw/create_experiment")
    def elabftw_create_experiment(
        body: ElabftwCreateBody,
        principal: AuthenticatedPrincipal = Depends(verified_principal),
    ) -> dict[str, object]:
        """Create a new eLabFTW experiment and return its ID.

        Called by the Lab Copilot widget when the user wants to save
        an OpenCloning result but no experiment is currently open (the
        writeback tool was requested with experiment_id=0).  The widget
        creates a fresh experiment, re-mints the context token for it,
        then re-triggers the writeback tool in the orchestrator so the
        approval goes through with a valid target.

        Uses the write adapter's eLabFTW client to POST to the upstream
        API.  The caller's identity is resolved via the identity mapper.
        """
        mapped = identity_mapper.map(
            keycloak_subject=principal.keycloak_subject,
            librechat_user_id=body.librechat_user_id,
        )
        if mapped is None:
            return {
                "ok": False,
                "reason": "unmapped_caller",
                "message": "caller did not resolve to a mapped eLabFTW identity",
            }
        adapter = get_elabftw_write_adapter()
        if adapter.client is None:
            return {
                "ok": False,
                "reason": "no_elabftw_client",
                "message": "eLabFTW client is not configured",
            }
        try:
            new_id = adapter.client.create_experiment(title=body.title or None)
        except Exception as exc:
            return {
                "ok": False,
                "reason": "create_failed",
                "message": f"eLabFTW experiment creation failed: {exc}",
            }
        return {
            "ok": True,
            "experiment_id": new_id,
        }

    @api.post("/elabftw/draft_experiment_update")
    def elabftw_draft_experiment_update(
        body: ElabftwDraftBody,
        principal: AuthenticatedPrincipal = Depends(verified_principal),
    ) -> dict[str, object]:
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

    @api.post("/elabftw/edit_experiment_section")
    def elabftw_edit_experiment_section(
        body: ElabftwEditBody,
        principal: AuthenticatedPrincipal = Depends(verified_principal),
    ) -> dict[str, object]:
        adapter = get_elabftw_write_adapter()
        mapped_identity = identity_mapper.map(
            keycloak_subject=principal.keycloak_subject,
            librechat_user_id=body.librechat_user_id,
        )
        try:
            result = adapter.edit_experiment_section(
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


def _register_elabftw_amend_route(
    api: FastAPI, identity_mapper: IdentityMapper
) -> None:
    """Register the amend route (separated to reduce complexity)."""

    @api.post("/elabftw/amend_my_experiment_after_approval")
    def elabftw_amend_my_experiment_after_approval(
        body: ElabftwAmendBody,
        principal: AuthenticatedPrincipal = Depends(verified_principal),
    ) -> dict[str, object]:
        import base64

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


def _register_invoke_route(
    api: FastAPI,
    identity_mapper: IdentityMapper,
    artifact_store: ArtifactBundleStore,
) -> None:
    """Register POST /invoke route."""

    @api.post("/invoke")
    def invoke(
        body: InvokeBody,
        principal: AuthenticatedPrincipal = Depends(verified_principal),
    ) -> dict[str, object]:
        registry = get_tool_registry()
        tool = registry.find(body.tool_name)
        if tool is None:
            return {
                "ok": False,
                "tool_name": body.tool_name,
                "reason": "tool_not_registered",
                "message": f"tool {body.tool_name!r} is not in the gateway registry; "
                "LibreChat may only invoke curated C06 tools",
            }
        mapped_identity = identity_mapper.map(
            keycloak_subject=principal.keycloak_subject,
            librechat_user_id=body.librechat_user_id,
        )
        dispatcher = _INVOKE_DISPATCHERS.get(tool.adapter)
        if dispatcher is not None:
            if tool.adapter == "opencloning":
                return dispatcher(tool, body, mapped_identity, artifact_store)
            return dispatcher(tool, body, mapped_identity)
        return {
            "ok": False,
            "tool_name": tool.name,
            "reason": "adapter_not_implemented",
            "message": f"tool {tool.name!r} is in the registry but its "
            f"{tool.adapter!r} adapter is not implemented yet (lands in C19+)",
        }


def _register_plan_route(api: FastAPI, identity_mapper: IdentityMapper) -> None:
    """Register POST /plan/execute route."""

    @api.post("/plan/execute")
    def plan_execute(
        body: PlanExecuteBody,
        principal: AuthenticatedPrincipal = Depends(verified_principal),
    ) -> dict[str, object]:
        try:
            plan = Plan.from_dict(body.plan)
        except PlanValidationError as exc:
            return {
                "ok": False,
                "plan_id": body.plan.get("plan_id", ""),
                "reason": "plan_validation_failed",
                "errors": exc.errors,
            }
        mapped_identity = identity_mapper.map(
            keycloak_subject=principal.keycloak_subject,
            librechat_user_id=body.librechat_user_id,
        )
        executor = get_plan_executor()
        result = executor.execute(
            plan,
            approval_id=body.approval_id,
            context_token=body.context_token,
            mapped_identity=mapped_identity,
            conversation_id=body.conversation_id,
            request_id=body.request_id,
            keycloak_subject=body.keycloak_subject,
            librechat_user_id=body.librechat_user_id,
            provider=body.provider,
            model_id=body.model_id,
        )
        return {"ok": result.status == "completed", **result.to_dict()}


def _build_health_deps(
    *,
    audit_store: AuditStore,
    policy_engine: PolicyEngine,
    identity_mapper: IdentityMapper,
    approval_store: ApprovalStore,
    elabftw_adapter: ElabftwReadAdapter,
    opencloning_adapter: OpenCloningAdapter,
    wallac_adapter: WallacAdapter,
) -> dict[str, object]:
    """Build the /health dependencies dict."""
    deps: dict[str, object] = {
        "audit_db": "configured" if audit_store.db_path != ":memory:" else "memory",
        "policy_engine": "ready",
        "policy_max_tier": int(policy_engine.max_tier),
        "kill_switches": list(policy_engine.kill_switches),
        "kill_switch_categories": {
            k: v for k, v in policy_engine.kill_categories.items() if v
        },
        "elabftw": (
            "configured" if elabftw_adapter.client is not None else "not_configured"
        ),
        "opencloning": (
            "configured" if opencloning_adapter.client is not None else "not_configured"
        ),
        "wallac": (
            "configured" if wallac_adapter.client is not None else "not_configured"
        ),
        "wallac_bridge": (
            "configured"
            if wallac_adapter.bridge_client is not None
            else "not_configured"
        ),
        "bentolab": (
            "configured"
            if get_bentolab_adapter().client is not None
            else "not_configured"
        ),
    }
    deps.update(_identity_backend_status(identity_mapper))
    deps.update(_approval_backend_status(approval_store))
    deps["tool_count"] = len(get_tool_registry().list())
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
    return deps


def _register_health_routes(
    api: FastAPI,
    service_name: str,
    audit_store: AuditStore,
    policy_engine: PolicyEngine,
    identity_mapper: IdentityMapper,
    approval_store: ApprovalStore,
    elabftw_adapter: ElabftwReadAdapter,
    opencloning_adapter: OpenCloningAdapter,
    wallac_adapter: WallacAdapter,
) -> None:
    """Register GET /health route."""

    @api.get("/health")
    def health() -> dict[str, object]:
        deps = _build_health_deps(
            audit_store=audit_store,
            policy_engine=policy_engine,
            identity_mapper=identity_mapper,
            approval_store=approval_store,
            elabftw_adapter=elabftw_adapter,
            opencloning_adapter=opencloning_adapter,
            wallac_adapter=wallac_adapter,
        )
        return {
            "service": service_name,
            "version": __version__,
            "status": "ok",
            "dependencies": deps,
        }


def _register_bentolab_route(api: FastAPI, identity_mapper: IdentityMapper) -> None:
    """Register POST /bentolab route."""

    @api.post("/bentolab")
    def bentolab_invoke(
        body: InvokeBody,
        principal: AuthenticatedPrincipal = Depends(verified_principal),
    ) -> dict[str, object]:
        import base64

        registry = get_tool_registry()
        tool = registry.find(body.tool_name)
        if tool is None:
            return {
                "ok": False,
                "tool_name": body.tool_name,
                "reason": "tool_not_registered",
                "message": f"tool {body.tool_name!r} is not in the gateway registry",
            }
        mapped_identity = identity_mapper.map(
            keycloak_subject=principal.keycloak_subject,
            librechat_user_id=body.librechat_user_id,
        )
        if body.args.get("attachment_b64") and body.args.get("attachment_filename"):
            try:
                base64.b64decode(body.args["attachment_b64"])
            except Exception as exc:
                return {
                    "ok": False,
                    "tool_name": tool.name,
                    "reason": "client_error",
                    "message": f"attachment_b64 is not valid base64: {exc}",
                }
        return _invoke_bentolab_tool(tool, body, mapped_identity)


# Module-level ASGI app for uvicorn (Dockerfile CMD expects
# lab_copilot_gateway.app:app).
app = create_app()
