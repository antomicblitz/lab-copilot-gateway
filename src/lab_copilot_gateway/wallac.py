"""Wallac adapter for the lab copilot gateway (C18 + C19).

Orchestrates gateway-enforced calls to the Wallac Victor2 vm-agent service
for instrument status reads (C18) and protocol proposal/validation (C19).
The vm-agent is a Python HTTP service running on a Windows 7 VM
(192.168.122.203:8420) that controls the Wallac Victor2 plate reader via
COM.  The Wallac bridge (designer/validation) is a separate FastAPI service
that handles protocol authoring and signed-spec validation.

This module follows the orchestration template established by the eLabFTW
adapter (C08/C09) and OpenCloning adapter (C16):

    context token verification
        → identity resolution (mapped user required)
        → policy engine decision (tier 2 OPERATIONAL_READ_ONLY for status,
          tier 3 VALIDATION_DRY_RUN for proposal/validation)
        → downstream client call (HTTP for production, stub for tests)
        → audit record with tool_args_hash

Key differences from the eLabFTW/OpenCloning adapters:

    * The vm-agent uses optional Bearer token auth (not API key).
    * Wallac status is not experiment-scoped — the context token's
      experiment_id is carried as a context_ref for audit correlation
      but is not used in the downstream call.
    * The vm-agent may be offline (VM down, network issue) — the adapter
      handles this gracefully with a structured error.
    * C19 proposal and submission-package tools are gateway-side
      computations (no downstream call) — they validate and structure
      the protocol spec without calling the bridge.
    * C19 validation calls the Wallac bridge's validation endpoint
      (separate service from the vm-agent).

Tools exposed (already in the C06 tool registry):

    wallac.get_status — read instrument status and current job (C18)
    wallac.propose_generated_protocol — propose a protocol design (C19)
    wallac.validate_generated_protocol — validate a signed bundle (C19)
    wallac.prepare_submission_package — prepare approval-ready package (C19)

Future chunks (C20/C21) will add:
    wallac.submit_generated_protocol — approval-gated hardware execution
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from lab_copilot_gateway.approval import ApprovalStore, compute_args_hash
from lab_copilot_gateway.audit import AuditRecord, AuditStore
from lab_copilot_gateway.elabftw import (
    ElabftwClient,
    InvalidContextToken,
    UnmappedCaller,
    verify_context_token,
)
from lab_copilot_gateway.identity import MappedIdentity
from lab_copilot_gateway.policy import Decision, PolicyEngine, PolicyRequest, Tier


# --- configuration --------------------------------------------------------

# Request timeout for downstream vm-agent calls (seconds).  The vm-agent's
# COM calls have a 20s timeout; 10s is enough for the cached /status
# endpoint (no live COM call needed).
DEFAULT_TIMEOUT_SECONDS = 10.0


# --- exceptions -----------------------------------------------------------


class WallacAdapterError(Exception):
    """Base class for Wallac adapter errors.

    ``reason`` is a machine-readable code; ``message`` is human-readable.
    Mirrors ``ElabftwAdapterError`` / ``OpenCloningAdapterError`` so the
    /invoke endpoint can handle all three uniformly.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message

    def to_dict(self) -> dict[str, str]:
        return {"reason": self.reason, "message": self.message}


class PolicyDenied(WallacAdapterError):
    """Policy engine refused the call."""

    def __init__(self, decision: Decision) -> None:
        super().__init__(
            reason=f"policy_denied:{decision.reason}",
            message=f"policy engine denied call ({decision.reason})",
        )


# --- downstream client ---------------------------------------------------


class WallacClient(Protocol):
    """Wallac vm-agent client surface the gateway adapter requires.

    Implementations: ``StubWallacClient`` for tests;
    ``HttpWallacClient`` for live calls.
    """

    def get_status(self) -> dict[str, Any]:
        """GET /status — cached instrument snapshot from the monitor thread.

        Returns the status dict with fields: seq, ts, connected, state,
        state_code, is_running, is_error, is_idle, target_temperature, error.
        """
        ...


@dataclass
class StubWallacClient:
    """In-memory ``WallacClient`` for tests and offline dev.

    Pre-seed ``status_response`` with the desired status dict.  Set
    ``error`` to simulate a downstream failure (e.g. vm-agent offline).
    """

    status_response: dict[str, Any] = field(
        default_factory=lambda: {
            "seq": 1,
            "ts": "2026-01-01T00:00:00Z",
            "connected": True,
            "state": "Idle",
            "state_code": 4003,
            "is_running": False,
            "is_error": False,
            "is_idle": True,
            "target_temperature": None,
            "error": None,
        }
    )
    error: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def get_status(self) -> dict[str, Any]:
        self.calls.append({"method": "get_status"})
        if self.error is not None:
            raise self.error
        return dict(self.status_response)


@dataclass
class HttpWallacClient:
    """Live Wallac vm-agent client.

    The vm-agent uses optional Bearer token auth.  Uses
    ``requests.Session`` for connection reuse, matching the lab HTTP
    pattern (see AGENTS.md → "eLabFTW API — HTTP client patterns").
    """

    base_url: str
    bearer_token: str | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    _session: Any = None

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("HttpWallacClient requires non-empty base_url")

    def _connect(self) -> Any:
        if self._session is None:
            import requests  # local import; not all deployments exercise HTTP

            s = requests.Session()
            if self.bearer_token:
                s.headers.update({"Authorization": f"Bearer {self.bearer_token}"})
            s.verify = False  # internal service, no TLS
            self._session = s
        return self._session

    def get_status(self) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/status"
        s = self._connect()
        resp = s.get(url, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()


# --- C19: Wallac bridge client (designer/validation) -----------------------


class WallacBridgeClient(Protocol):
    """Wallac bridge client surface for protocol validation (C19).

    The bridge is a separate FastAPI service (bridge_app.py +
    designer_app.py in wallac-victor2-api).  It handles signed-spec
    validation, protocol authoring, and job execution.

    Implementations: ``StubWallacBridgeClient`` for tests;
    ``HttpWallacBridgeClient`` for live calls.
    """

    def validate_protocol(self, job_item_id: int) -> dict[str, Any]:
        """POST /api/jobs/{id}/validate — validate a signed bundle.

        Returns a validation report dict with ``valid`` (bool),
        ``checks`` (list), and ``errors`` (list).
        """
        ...

    def submit_job(self, job_item_id: int) -> dict[str, Any]:
        """POST /api/jobs/{id}/submit — submit a job for hardware execution.

        Returns a dict with ``job_id``, ``status`` (e.g. ``submitted``,
        ``aborted``, ``failed``), and ``audit_action_id``.
        """
        ...

    def get_health(self) -> dict[str, Any]:
        """GET /health — bridge health check."""
        ...


@dataclass
class StubWallacBridgeClient:
    """In-memory ``WallacBridgeClient`` for tests and offline dev.

    Pre-seed ``validation_response`` with the desired report.  Set
    ``error`` to simulate a downstream failure.
    """

    validation_response: dict[str, Any] = field(
        default_factory=lambda: {
            "valid": True,
            "checks": [
                {"name": "signature", "passed": True},
                {"name": "schema", "passed": True},
                {"name": "lifecycle", "passed": True},
            ],
            "errors": [],
        }
    )
    submit_response: dict[str, Any] = field(
        default_factory=lambda: {
            "job_id": 1,
            "status": "submitted",
            "audit_action_id": "bridge-audit-1",
        }
    )
    health_response: dict[str, Any] = field(default_factory=lambda: {"status": "ok"})
    error: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def validate_protocol(self, job_item_id: int) -> dict[str, Any]:
        self.calls.append({"method": "validate_protocol", "job_item_id": job_item_id})
        if self.error is not None:
            raise self.error
        return dict(self.validation_response)

    def submit_job(self, job_item_id: int) -> dict[str, Any]:
        self.calls.append({"method": "submit_job", "job_item_id": job_item_id})
        if self.error is not None:
            raise self.error
        return dict(self.submit_response)

    def get_health(self) -> dict[str, Any]:
        if self.error is not None:
            raise self.error
        return dict(self.health_response)


@dataclass
class HttpWallacBridgeClient:
    """Live Wallac bridge client.

    The bridge uses optional Bearer token auth
    (``WALLAC_DESIGNER_TOKEN`` / ``WALLAC_BRIDGE_TOKEN``).
    """

    base_url: str
    bearer_token: str | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    _session: Any = None

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("HttpWallacBridgeClient requires non-empty base_url")

    def _connect(self) -> Any:
        if self._session is None:
            import requests

            s = requests.Session()
            if self.bearer_token:
                s.headers.update({"Authorization": f"Bearer {self.bearer_token}"})
            s.verify = False  # internal service, no TLS
            self._session = s
        return self._session

    def validate_protocol(self, job_item_id: int) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/api/jobs/{job_item_id}/validate"
        s = self._connect()
        resp = s.post(url, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    def submit_job(self, job_item_id: int) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/api/jobs/{job_item_id}/submit"
        s = self._connect()
        resp = s.post(url, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    def get_health(self) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/health"
        s = self._connect()
        resp = s.get(url, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()


def _default_bridge_client_from_env() -> WallacBridgeClient | None:
    """Construct the live bridge client if env config is present.

    Returns None if base_url is unset — the adapter then refuses
    validation calls with ``wallac_bridge_not_configured`` at call time.
    """
    base_url = os.getenv("LAB_COPILOT_WALLAC_BRIDGE_URL", "")
    if not base_url:
        return None
    bearer_token = os.getenv("LAB_COPILOT_WALLAC_BRIDGE_TOKEN", "") or None
    timeout = float(
        os.getenv("LAB_COPILOT_WALLAC_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS))
    )
    return HttpWallacBridgeClient(
        base_url=base_url,
        bearer_token=bearer_token,
        timeout_seconds=timeout,
    )


def _default_client_from_env() -> WallacClient | None:
    """Construct the live HTTP client if env config is present.

    Returns None if base_url is unset — the adapter then refuses calls
    with ``wallac_not_configured`` at call time.
    """
    base_url = os.getenv("LAB_COPILOT_WALLAC_BASE_URL", "")
    if not base_url:
        return None
    bearer_token = os.getenv("LAB_COPILOT_WALLAC_BEARER_TOKEN", "") or None
    timeout = float(
        os.getenv("LAB_COPILOT_WALLAC_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS))
    )
    return HttpWallacClient(
        base_url=base_url,
        bearer_token=bearer_token,
        timeout_seconds=timeout,
    )


# --- result ---------------------------------------------------------------


@dataclass(frozen=True)
class WallacResult:
    """Outcome of a successful Wallac adapter call."""

    tool_name: str
    result: dict[str, Any]
    audit_action_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "result": self.result,
            "audit_action_id": self.audit_action_id,
        }


# --- adapter --------------------------------------------------------------


# Tool name from the C06 registry.
TOOL_GET_STATUS = "wallac.get_status"
TOOL_PROPOSE = "wallac.propose_generated_protocol"
TOOL_VALIDATE = "wallac.validate_generated_protocol"
TOOL_PREPARE_SUBMISSION = "wallac.prepare_submission_package"
TOOL_SUBMIT = "wallac.submit_generated_protocol"

# C18 status is tier 2 (OPERATIONAL_READ_ONLY).
TOOL_TIER = Tier.OPERATIONAL_READ_ONLY

# C19 proposal/validation is tier 3 (VALIDATION_DRY_RUN).
TOOL_TIER_C19 = Tier.VALIDATION_DRY_RUN

# C21 hardware execution is tier 5 (HARDWARE_EXECUTION).
TOOL_TIER_SUBMIT = Tier.HARDWARE_EXECUTION


@dataclass
class WallacAdapter:
    """Orchestrates Wallac vm-agent and bridge calls through the gateway-enforced path.

    Dependencies are injected so tests can swap the policy engine, audit
    store, and downstream clients individually.  In production, the
    module-level ``get_wallac_adapter()`` wires all the singletons.

    Two clients:
        * ``client`` — vm-agent for status reads (C18).
        * ``bridge_client`` — Wallac bridge for protocol validation (C19).
          May be None when the bridge is not deployed; proposal and
          submission-package tools still work (gateway-side computation).
    """

    policy_engine: PolicyEngine
    audit_store: AuditStore
    client: WallacClient | None
    bridge_client: WallacBridgeClient | None = None
    approval_store: ApprovalStore | None = None
    elabftw_client: ElabftwClient | None = None
    action_id_prefix: str = "wallac"

    # --- public API: C18 tool entrypoint --------------------------------

    def get_status(
        self,
        *,
        context_token: str,
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None = None,
        request_id: str | None = None,
        keycloak_subject: str | None = None,
        librechat_user_id: str | None = None,
        provider: str | None = None,
        model_id: str | None = None,
    ) -> WallacResult:
        """Read Wallac Victor2 instrument status and current job.

        Calls the vm-agent's ``GET /status`` endpoint, which returns a
        cached snapshot from the 1 Hz monitor thread (no live COM call).
        Returns the status dict with instrument state, connection status,
        and current job info.
        """
        return self._execute(
            tool_name=TOOL_GET_STATUS,
            context_token=context_token,
            mapped_identity=mapped_identity,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            args_for_hash={"tool": "get_status"},
            api_call_summary={"method": "GET", "path": "/status"},
            exec_fn=lambda _client: _client.get_status(),
        )

    # --- C19 tools: proposal / validation / submission package -----------

    def propose_generated_protocol(
        self,
        *,
        context_token: str,
        mapped_identity: MappedIdentity | None,
        protocol_spec: dict[str, Any],
        conversation_id: str | None = None,
        request_id: str | None = None,
        keycloak_subject: str | None = None,
        librechat_user_id: str | None = None,
        provider: str | None = None,
        model_id: str | None = None,
    ) -> WallacResult:
        """Propose a generated-protocol package without execution.

        Takes a protocol design spec (method, layout, analysis, job
        parameters) and returns a structured proposal with a content hash.
        This is a gateway-side computation — no downstream call is made.

        The proposal includes:
            - the validated spec
            - a SHA-256 content hash (for approval binding)
            - execution_blocked: True (v1 blocks hardware execution)
        """
        spec_hash = compute_args_hash(protocol_spec)

        def _propose(_client: Any) -> dict[str, Any]:
            return {
                "proposal": protocol_spec,
                "spec_hash": spec_hash,
                "execution_blocked": True,
                "message": (
                    "Protocol proposal prepared. Execution remains blocked in v1. "
                    "Use prepare_submission_package to create an approval-ready package."
                ),
            }

        return self._execute(
            tool_name=TOOL_PROPOSE,
            context_token=context_token,
            mapped_identity=mapped_identity,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            args_for_hash={"tool": "propose_generated_protocol", "spec": protocol_spec},
            api_call_summary={"method": "none", "path": "gateway-side"},
            exec_fn=_propose,
            tier_override=TOOL_TIER_C19,
        )

    def validate_generated_protocol(
        self,
        *,
        context_token: str,
        mapped_identity: MappedIdentity | None,
        job_item_id: int,
        conversation_id: str | None = None,
        request_id: str | None = None,
        keycloak_subject: str | None = None,
        librechat_user_id: str | None = None,
        provider: str | None = None,
        model_id: str | None = None,
    ) -> WallacResult:
        """Validate a generated-protocol package against the signed-spec schema.

        Calls the Wallac bridge's validation endpoint to verify the
        signed bundle (signature, schema, lifecycle, capabilities).
        The bridge is a separate service — if not configured, returns
        ``wallac_bridge_not_configured``.
        """
        return self._execute(
            tool_name=TOOL_VALIDATE,
            context_token=context_token,
            mapped_identity=mapped_identity,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            args_for_hash={
                "tool": "validate_generated_protocol",
                "job_item_id": job_item_id,
            },
            api_call_summary={
                "method": "POST",
                "path": f"/api/jobs/{job_item_id}/validate",
            },
            exec_fn=lambda _client: self._call_bridge_validate(job_item_id),
            tier_override=TOOL_TIER_C19,
            use_bridge=True,
        )

    def prepare_submission_package(
        self,
        *,
        context_token: str,
        mapped_identity: MappedIdentity | None,
        protocol_spec: dict[str, Any],
        validation_report: dict[str, Any] | None = None,
        conversation_id: str | None = None,
        request_id: str | None = None,
        keycloak_subject: str | None = None,
        librechat_user_id: str | None = None,
        provider: str | None = None,
        model_id: str | None = None,
    ) -> WallacResult:
        """Prepare an approval-ready Wallac submission package.

        Assembles the protocol spec and validation report into a
        structured submission package.  Execution remains blocked in v1 —
        the package explicitly states ``execution_blocked: True``.

        This is a gateway-side computation — no downstream call is made.
        """
        spec_hash = compute_args_hash(protocol_spec)

        def _prepare(_client: Any) -> dict[str, Any]:
            return {
                "submission_package": {
                    "spec": protocol_spec,
                    "spec_hash": spec_hash,
                    "validation": validation_report
                    or {"valid": False, "reason": "not_validated"},
                    "execution_blocked": True,
                    "v1_notice": (
                        "Hardware execution is blocked in v1. "
                        "This package is approval-ready but cannot be submitted "
                        "until v1.1 (C20/C21)."
                    ),
                }
            }

        return self._execute(
            tool_name=TOOL_PREPARE_SUBMISSION,
            context_token=context_token,
            mapped_identity=mapped_identity,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            args_for_hash={"tool": "prepare_submission_package", "spec": protocol_spec},
            api_call_summary={"method": "none", "path": "gateway-side"},
            exec_fn=_prepare,
            tier_override=TOOL_TIER_C19,
        )

    # --- C21: approval-gated submit --------------------------------------

    def submit_generated_protocol(
        self,
        *,
        context_token: str,
        mapped_identity: MappedIdentity | None,
        approval_id: str,
        approval_args: dict[str, Any],
        job_item_id: int,
        conversation_id: str | None = None,
        request_id: str | None = None,
        keycloak_subject: str | None = None,
        librechat_user_id: str | None = None,
        provider: str | None = None,
        model_id: str | None = None,
        plan_approval_id: str | None = None,
    ) -> WallacResult:
        """Submit an approved Wallac generated protocol for hardware execution.

        Orchestrates:
            token verify -> identity -> policy (tier 5) -> approval consume
            -> bridge submit -> eLabFTW provenance writeback -> audit

        The approval consume happens AFTER policy allow but BEFORE the
        bridge call, so a consumed-but-failed submission still has an
        audit trail (the approval is NOT refunded).

        ``job_item_id`` identifies the job in the Wallac bridge that was
        prepared by the C19 prepare+validate flow.  ``approval_id`` and
        ``approval_args`` are the single-use approval token and its
        bound args (issued via POST /approval/request).

        C42: when ``plan_approval_id`` is set, the plan executor has
        already consumed the plan-level approval (bound to the plan hash).
        Skip the per-step consume and use the plan approval_id in audit
        records (same pattern as OpenCloning C41).
        """
        return self._execute_submit(
            context_token=context_token,
            mapped_identity=mapped_identity,
            approval_id=approval_id,
            approval_args=approval_args,
            job_item_id=job_item_id,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            plan_approval_id=plan_approval_id,
        )

    # --- shared orchestration --------------------------------------------

    def _call_bridge_validate(self, job_item_id: int) -> dict[str, Any]:
        """Call the Wallac bridge to validate a signed protocol bundle.

        Separated from the lambda so the bridge-not-configured check
        happens at the right point in the orchestration (after policy
        allow, before the downstream call).
        """
        if self.bridge_client is None:
            raise WallacAdapterError(
                reason="wallac_bridge_not_configured",
                message=(
                    "Wallac bridge client not configured "
                    "(set LAB_COPILOT_WALLAC_BRIDGE_URL)"
                ),
            )
        return self.bridge_client.validate_protocol(job_item_id)

    def _call_bridge_submit(self, job_item_id: int) -> dict[str, Any]:
        """Call the Wallac bridge to submit a job for execution.

        Separated from the lambda so the bridge-not-configured check
        happens at the right point in the orchestration (after approval
        consume, before the downstream call).
        """
        if self.bridge_client is None:
            raise WallacAdapterError(
                reason="wallac_bridge_not_configured",
                message=(
                    "Wallac bridge client not configured "
                    "(set LAB_COPILOT_WALLAC_BRIDGE_URL)"
                ),
            )
        return self.bridge_client.submit_job(job_item_id)

    def _execute_submit(
        self,
        *,
        context_token: str,
        mapped_identity: MappedIdentity | None,
        approval_id: str,
        approval_args: dict[str, Any],
        job_item_id: int,
        conversation_id: str | None,
        request_id: str | None,
        keycloak_subject: str | None,
        librechat_user_id: str | None,
        provider: str | None,
        model_id: str | None,
        plan_approval_id: str | None = None,
    ) -> WallacResult:
        """Submit orchestration: token -> identity -> policy -> approval -> bridge -> provenance -> audit.

        Mirrors the OpenCloning adapter's ``_execute_writeback`` pattern.
        The approval is consumed AFTER the policy allow but BEFORE the
        bridge call — the token is single-use and NOT refunded on bridge
        failure, preventing replay.

        C42: when ``plan_approval_id`` is set, the plan executor has
        already consumed the plan-level approval.  Skip the per-step
        consume and use the plan approval_id in audit records.

        Unknown/abort statuses from the bridge are returned as results
        (not exceptions) so the caller sees the actual execution outcome.
        """
        import secrets as _secrets
        import time as _time

        # Compute early redacted args hash for audit on failure paths.
        early_hash = compute_args_hash(
            {
                "tool": "submit_generated_protocol",
                "job_item_id": job_item_id,
                "token_present": bool(context_token),
            }
        )

        # 1. Verify context token (signature + expiry).
        if not context_token:
            self._audit(
                tool_name=TOOL_SUBMIT,
                policy_decision="deny",
                reason="missing_context_token",
                mapped_identity=mapped_identity,
                experiment_id=None,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=early_hash,
                error={"code": "MISSING_CONTEXT_TOKEN"},
            )
            raise InvalidContextToken("missing")

        try:
            claims = verify_context_token(context_token)
        except InvalidContextToken as exc:
            self._audit(
                tool_name=TOOL_SUBMIT,
                policy_decision="deny",
                reason=f"invalid_context_token:{exc.detail}",
                mapped_identity=mapped_identity,
                experiment_id=None,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=early_hash,
                error={"code": "INVALID_CONTEXT_TOKEN", "detail": exc.detail},
            )
            raise

        tool_args_hash = compute_args_hash(
            {
                "tool": "submit_generated_protocol",
                "job_item_id": job_item_id,
                "experiment_id": claims.experiment_id,
            }
        )

        # 2. Identity resolution — unmapped caller denies.
        if mapped_identity is None:
            self._audit(
                tool_name=TOOL_SUBMIT,
                policy_decision="deny",
                reason="unmapped_caller",
                mapped_identity=None,
                experiment_id=claims.experiment_id,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=tool_args_hash,
                error={"code": "UNMAPPED_CALLER"},
            )
            raise UnmappedCaller()

        # 3. Token-identity binding (three-way check).
        if claims.mapped_elabftw_user_id != mapped_identity.elabftw_user_id:
            self._audit(
                tool_name=TOOL_SUBMIT,
                policy_decision="deny",
                reason="context_token_user_mismatch",
                mapped_identity=mapped_identity,
                experiment_id=claims.experiment_id,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=tool_args_hash,
                error={"code": "CONTEXT_TOKEN_USER_MISMATCH"},
            )
            raise InvalidContextToken("token user does not match resolved identity")

        if (
            claims.keycloak_subject is not None
            and mapped_identity.keycloak_subject is not None
        ):
            if claims.keycloak_subject != mapped_identity.keycloak_subject:
                self._audit(
                    tool_name=TOOL_SUBMIT,
                    policy_decision="deny",
                    reason="context_token_kc_subject_mismatch",
                    mapped_identity=mapped_identity,
                    experiment_id=claims.experiment_id,
                    conversation_id=conversation_id,
                    request_id=request_id,
                    keycloak_subject=keycloak_subject,
                    librechat_user_id=librechat_user_id,
                    provider=provider,
                    model_id=model_id,
                    tool_args_hash=tool_args_hash,
                    error={"code": "CONTEXT_TOKEN_KC_SUBJECT_MISMATCH"},
                )
                raise InvalidContextToken(
                    "token keycloak_subject does not match resolved identity"
                )

        if (
            claims.librechat_user_id is not None
            and mapped_identity.librechat_user_id is not None
        ):
            if claims.librechat_user_id != mapped_identity.librechat_user_id:
                self._audit(
                    tool_name=TOOL_SUBMIT,
                    policy_decision="deny",
                    reason="context_token_lc_user_mismatch",
                    mapped_identity=mapped_identity,
                    experiment_id=claims.experiment_id,
                    conversation_id=conversation_id,
                    request_id=request_id,
                    keycloak_subject=keycloak_subject,
                    librechat_user_id=librechat_user_id,
                    provider=provider,
                    model_id=model_id,
                    tool_args_hash=tool_args_hash,
                    error={"code": "CONTEXT_TOKEN_LC_USER_MISMATCH"},
                )
                raise InvalidContextToken(
                    "token librechat_user_id does not match resolved identity"
                )

        # 4. Policy decision (tier 5 HARDWARE_EXECUTION, requires approval).
        policy_req = PolicyRequest(
            tool_name=TOOL_SUBMIT,
            tier=TOOL_TIER_SUBMIT,
            adapter="wallac",
            user_id=mapped_identity.elabftw_user_id,
            team_id=mapped_identity.elabftw_team_id,
            has_approval=bool(approval_id),
            approval_id=approval_id,
        )
        decision = self.policy_engine.decide(policy_req)

        if decision.decision != "allow":
            self._audit(
                tool_name=TOOL_SUBMIT,
                policy_decision="deny",
                reason=decision.reason,
                mapped_identity=mapped_identity,
                experiment_id=claims.experiment_id,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=tool_args_hash,
                approval_id=approval_id if approval_id else None,
                error={
                    "code": "POLICY_DENIED",
                    "reason": decision.reason,
                    "tier": decision.tier,
                },
            )
            raise PolicyDenied(decision)

        # 5. Approval token consume (single-use, args-hash-bound).
        #    Happens AFTER policy allow, BEFORE the bridge call.
        # C42: when ``plan_approval_id`` is set, the plan executor has
        # already consumed the plan-level approval (bound to the plan hash,
        # which covers all step args).  Skip the per-step consume and use
        # the plan approval_id in audit records.
        if plan_approval_id:
            effective_approval_id = plan_approval_id
        elif self.approval_store is not None:
            try:
                self.approval_store.consume(
                    approval_id,
                    tool_name=TOOL_SUBMIT,
                    args_hash=compute_args_hash(approval_args),
                    target_record=f"wallac:job:{job_item_id}",
                )
                effective_approval_id = approval_id
            except Exception as exc:
                from lab_copilot_gateway.approval import ApprovalError

                reason_code = "approval_consume_failed"
                error_payload: dict[str, Any] = {
                    "code": "APPROVAL_CONSUME_FAILED",
                    "exception": type(exc).__name__,
                }
                if isinstance(exc, ApprovalError):
                    reason_code = f"approval_denied:{exc.reason}"
                    error_payload["approval_reason"] = exc.reason
                    error_payload["approval_id"] = exc.approval_id
                self._audit(
                    tool_name=TOOL_SUBMIT,
                    policy_decision="deny",
                    reason=reason_code,
                    mapped_identity=mapped_identity,
                    experiment_id=claims.experiment_id,
                    conversation_id=conversation_id,
                    request_id=request_id,
                    keycloak_subject=keycloak_subject,
                    librechat_user_id=librechat_user_id,
                    provider=provider,
                    model_id=model_id,
                    tool_args_hash=tool_args_hash,
                    approval_id=approval_id,
                    error=error_payload,
                )
                raise WallacAdapterError(
                    reason=reason_code,
                    message=f"approval token consume failed: {exc}",
                ) from exc
        else:
            effective_approval_id = approval_id

        # 6. Bridge client must be configured.
        if self.bridge_client is None:
            self._audit(
                tool_name=TOOL_SUBMIT,
                policy_decision="allow",
                reason="wallac_bridge_not_configured",
                mapped_identity=mapped_identity,
                experiment_id=claims.experiment_id,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=tool_args_hash,
                approval_id=effective_approval_id,
                error={"code": "WALLAC_BRIDGE_NOT_CONFIGURED"},
            )
            raise WallacAdapterError(
                reason="wallac_bridge_not_configured",
                message=(
                    "Wallac bridge client not configured "
                    "(set LAB_COPILOT_WALLAC_BRIDGE_URL)"
                ),
            )

        # 7. Bridge submit call.
        try:
            bridge_result = self._call_bridge_submit(job_item_id)
        except Exception as exc:
            self._audit(
                tool_name=TOOL_SUBMIT,
                policy_decision="allow",
                reason="client_error",
                mapped_identity=mapped_identity,
                experiment_id=claims.experiment_id,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=tool_args_hash,
                approval_id=effective_approval_id,
                api_call_summary={
                    "method": "POST",
                    "path": f"/api/jobs/{job_item_id}/submit",
                },
                error={
                    "code": "CLIENT_ERROR",
                    "exception": type(exc).__name__,
                    "message": str(exc),
                },
            )
            raise WallacAdapterError(
                reason="client_error",
                message=f"Wallac bridge submitted raised {type(exc).__name__}: {exc}",
            ) from exc

        # 8. Pre-compute the action_id for audit + provenance writeback.
        action_id = f"{self.action_id_prefix}-submit-{int(_time.time() * 1000)}-{_secrets.token_hex(4)}"

        # 9. eLabFTW provenance writeback (best-effort, non-fatal).
        if self.elabftw_client is not None:
            try:
                self.elabftw_client.patch_experiment_metadata(
                    experiment_id=claims.experiment_id,
                    metadata={
                        "extra_fields": {
                            "wallac_job_id": bridge_result.get("job_id"),
                            "wallac_job_status": bridge_result.get("status"),
                            "wallac_bridge_audit_action_id": bridge_result.get(
                                "audit_action_id"
                            ),
                            "wallac_submit_tool": TOOL_SUBMIT,
                            "wallac_gateway_audit_action_id": action_id,
                        }
                    },
                )
            except Exception:
                # Provenance write failure is not fatal — the job was
                # already submitted to the bridge.
                pass

        # 10. Success audit.

        self._audit(
            tool_name=TOOL_SUBMIT,
            policy_decision="allow",
            reason="call_succeeded",
            mapped_identity=mapped_identity,
            experiment_id=claims.experiment_id,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            tool_args_hash=tool_args_hash,
            approval_id=effective_approval_id,
            api_call_summary={
                "method": "POST",
                "path": f"/api/jobs/{job_item_id}/submit",
            },
            result_summary={
                "job_id": bridge_result.get("job_id"),
                "status": bridge_result.get("status"),
                "bridge_audit_action_id": bridge_result.get("audit_action_id"),
            },
            require_persistence=True,
            action_id=action_id,
        )

        return WallacResult(
            tool_name=TOOL_SUBMIT,
            result={
                "job_id": bridge_result.get("job_id"),
                "status": bridge_result.get("status"),
                "bridge_audit_action_id": bridge_result.get("audit_action_id"),
            },
            audit_action_id=action_id,
        )

    def _execute(
        self,
        *,
        tool_name: str,
        context_token: str,
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None,
        request_id: str | None,
        keycloak_subject: str | None,
        librechat_user_id: str | None,
        provider: str | None,
        model_id: str | None,
        args_for_hash: dict[str, Any],
        api_call_summary: dict[str, Any],
        exec_fn: Any,  # callable(client) -> dict
        tier_override: Tier | None = None,
        use_bridge: bool = False,
    ) -> WallacResult:
        """Shared orchestration: token → identity → policy → client → audit.

        Mirrors the OpenCloning adapter's _execute method.

        ``tier_override`` lets C19 tools use tier 3 (VALIDATION_DRY_RUN)
        instead of the default tier 2 (OPERATIONAL_READ_ONLY).

        ``use_bridge`` selects which client is checked for the
        not-configured path: False (default) checks ``self.client``
        (vm-agent), True checks ``self.bridge_client`` (Wallac bridge).
        For gateway-side computations (propose, prepare_submission), the
        exec_fn doesn't use the client at all, but we still require
        ``self.client`` to be configured for the audit path to work.
        """
        # Compute early redacted args hash for audit on failure paths.
        early_hash = compute_args_hash(
            {**args_for_hash, "token_present": bool(context_token)}
        )

        # 1. Verify context token (signature + expiry).
        if not context_token:
            self._audit(
                tool_name=tool_name,
                policy_decision="deny",
                reason="missing_context_token",
                mapped_identity=mapped_identity,
                experiment_id=None,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=early_hash,
                error={"code": "MISSING_CONTEXT_TOKEN"},
            )
            raise InvalidContextToken("missing")

        try:
            claims = verify_context_token(context_token)
        except InvalidContextToken as exc:
            self._audit(
                tool_name=tool_name,
                policy_decision="deny",
                reason=f"invalid_context_token:{exc.detail}",
                mapped_identity=mapped_identity,
                experiment_id=None,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=early_hash,
                error={"code": "INVALID_CONTEXT_TOKEN", "detail": exc.detail},
            )
            raise

        tool_args_hash = compute_args_hash(
            {**args_for_hash, "experiment_id": claims.experiment_id}
        )

        # 2. Identity resolution — unmapped caller denies before any HTTP.
        if mapped_identity is None:
            self._audit(
                tool_name=tool_name,
                policy_decision="deny",
                reason="unmapped_caller",
                mapped_identity=None,
                experiment_id=claims.experiment_id,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=tool_args_hash,
                error={"code": "UNMAPPED_CALLER"},
            )
            raise UnmappedCaller()

        # 3. Token-identity binding (three-way check).
        if claims.mapped_elabftw_user_id != mapped_identity.elabftw_user_id:
            self._audit(
                tool_name=tool_name,
                policy_decision="deny",
                reason="context_token_user_mismatch",
                mapped_identity=mapped_identity,
                experiment_id=claims.experiment_id,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=tool_args_hash,
                error={"code": "CONTEXT_TOKEN_USER_MISMATCH"},
            )
            raise InvalidContextToken("token user does not match resolved identity")

        if (
            claims.keycloak_subject is not None
            and mapped_identity.keycloak_subject is not None
        ):
            if claims.keycloak_subject != mapped_identity.keycloak_subject:
                self._audit(
                    tool_name=tool_name,
                    policy_decision="deny",
                    reason="context_token_kc_subject_mismatch",
                    mapped_identity=mapped_identity,
                    experiment_id=claims.experiment_id,
                    conversation_id=conversation_id,
                    request_id=request_id,
                    keycloak_subject=keycloak_subject,
                    librechat_user_id=librechat_user_id,
                    provider=provider,
                    model_id=model_id,
                    tool_args_hash=tool_args_hash,
                    error={"code": "CONTEXT_TOKEN_KC_SUBJECT_MISMATCH"},
                )
                raise InvalidContextToken(
                    "token keycloak_subject does not match resolved identity"
                )

        if (
            claims.librechat_user_id is not None
            and mapped_identity.librechat_user_id is not None
        ):
            if claims.librechat_user_id != mapped_identity.librechat_user_id:
                self._audit(
                    tool_name=tool_name,
                    policy_decision="deny",
                    reason="context_token_lc_user_mismatch",
                    mapped_identity=mapped_identity,
                    experiment_id=claims.experiment_id,
                    conversation_id=conversation_id,
                    request_id=request_id,
                    keycloak_subject=keycloak_subject,
                    librechat_user_id=librechat_user_id,
                    provider=provider,
                    model_id=model_id,
                    tool_args_hash=tool_args_hash,
                    error={"code": "CONTEXT_TOKEN_LC_USER_MISMATCH"},
                )
                raise InvalidContextToken(
                    "token librechat_user_id does not match resolved identity"
                )

        # 4. Policy decision (tier from override or default).
        effective_tier = tier_override if tier_override is not None else TOOL_TIER
        policy_req = PolicyRequest(
            tool_name=tool_name,
            tier=effective_tier,
            adapter="wallac",
            user_id=mapped_identity.elabftw_user_id,
            team_id=mapped_identity.elabftw_team_id,
            has_approval=False,
            approval_id=None,
        )
        decision = self.policy_engine.decide(policy_req)

        if decision.decision != "allow":
            self._audit(
                tool_name=tool_name,
                policy_decision="deny",
                reason=decision.reason,
                mapped_identity=mapped_identity,
                experiment_id=claims.experiment_id,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=tool_args_hash,
                error={
                    "code": "POLICY_DENIED",
                    "reason": decision.reason,
                    "tier": decision.tier,
                },
            )
            raise PolicyDenied(decision)

        # 5. Client-not-configured check.
        #    For bridge tools (validate_generated_protocol), check bridge_client.
        #    For gateway-side tools (propose, prepare_submission), the exec_fn
        #    doesn't use the client, but we still require self.client for
        #    consistency (the adapter is wired).  For vm-agent tools
        #    (get_status), check self.client.
        if use_bridge:
            if self.bridge_client is None:
                self._audit(
                    tool_name=tool_name,
                    policy_decision="deny",
                    reason="wallac_bridge_not_configured",
                    mapped_identity=mapped_identity,
                    experiment_id=claims.experiment_id,
                    conversation_id=conversation_id,
                    request_id=request_id,
                    keycloak_subject=keycloak_subject,
                    librechat_user_id=librechat_user_id,
                    provider=provider,
                    model_id=model_id,
                    tool_args_hash=tool_args_hash,
                    error={"code": "WALLAC_BRIDGE_NOT_CONFIGURED"},
                )
                raise WallacAdapterError(
                    reason="wallac_bridge_not_configured",
                    message=(
                        "Wallac bridge client not configured "
                        "(set LAB_COPILOT_WALLAC_BRIDGE_URL)"
                    ),
                )
        elif self.client is None:
            self._audit(
                tool_name=tool_name,
                policy_decision="deny",
                reason="wallac_not_configured",
                mapped_identity=mapped_identity,
                experiment_id=claims.experiment_id,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=tool_args_hash,
                error={"code": "WALLAC_NOT_CONFIGURED"},
            )
            raise WallacAdapterError(
                reason="wallac_not_configured",
                message=(
                    "Wallac client not configured (set LAB_COPILOT_WALLAC_BASE_URL)"
                ),
            )

        # 6. Downstream call.
        #    For gateway-side tools (propose, prepare_submission), exec_fn
        #    ignores the client argument.  For bridge tools (validate),
        #    exec_fn calls self._call_bridge_validate() which uses
        #    self.bridge_client.  For vm-agent tools (get_status), exec_fn
        #    calls client.get_status().  Pass the appropriate client.
        downstream_client = self.bridge_client if use_bridge else self.client
        try:
            result = exec_fn(downstream_client)
        except Exception as exc:
            self._audit(
                tool_name=tool_name,
                policy_decision="allow",
                reason="client_error",
                mapped_identity=mapped_identity,
                experiment_id=claims.experiment_id,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=tool_args_hash,
                api_call_summary=api_call_summary,
                error={
                    "code": "CLIENT_ERROR",
                    "exception": type(exc).__name__,
                    "message": str(exc),
                },
            )
            raise WallacAdapterError(
                reason="client_error",
                message=f"Wallac client raised {type(exc).__name__}: {exc}",
            ) from exc

        # 7. Success — audit + return.
        import secrets as _secrets
        import time as _time

        action_id = f"{self.action_id_prefix}-{int(_time.time() * 1000)}-{_secrets.token_hex(4)}"

        self._audit(
            tool_name=tool_name,
            policy_decision="allow",
            reason="call_succeeded",
            mapped_identity=mapped_identity,
            experiment_id=claims.experiment_id,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            tool_args_hash=tool_args_hash,
            api_call_summary=api_call_summary,
            result_summary={
                "connected": result.get("connected"),
                "state": result.get("state"),
                "is_running": result.get("is_running"),
            },
            require_persistence=True,
            action_id=action_id,
        )

        return WallacResult(
            tool_name=tool_name,
            result=result,
            audit_action_id=action_id,
        )

    # --- audit writer ----------------------------------------------------

    def _audit(
        self,
        *,
        tool_name: str,
        policy_decision: str,
        reason: str,
        mapped_identity: MappedIdentity | None,
        experiment_id: int | None,
        conversation_id: str | None,
        request_id: str | None,
        keycloak_subject: str | None,
        librechat_user_id: str | None,
        provider: str | None,
        model_id: str | None,
        tool_args_hash: str | None = None,
        api_call_summary: dict[str, Any] | None = None,
        result_summary: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        require_persistence: bool = False,
        action_id: str | None = None,
        approval_id: str | None = None,
    ) -> None:
        """Persist an audit record.

        Mirrors the eLabFTW/OpenCloning adapter's audit pattern: deny/error
        paths swallow audit-write failures; success paths raise if audit
        persistence fails.
        """
        import secrets as _secrets
        import time as _time

        if action_id is None:
            action_id = f"{self.action_id_prefix}-{int(_time.time() * 1000)}-{_secrets.token_hex(4)}"

        context_refs: list[dict[str, Any]] = []
        if experiment_id is not None:
            context_refs.append({"kind": "experiment", "id": experiment_id})

        record = AuditRecord(
            action_id=action_id,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            mapped_elabftw_user_id=(
                mapped_identity.elabftw_user_id if mapped_identity else None
            ),
            mapped_elabftw_team_id=(
                mapped_identity.elabftw_team_id if mapped_identity else None
            ),
            provider=provider,
            model_id=model_id,
            tool_name=tool_name,
            tool_args_hash=tool_args_hash,
            approval_id=approval_id,
            context_refs=context_refs,
            policy_decision=policy_decision,
            api_call_summary=api_call_summary or {},
            result_summary=result_summary or {},
            error=error,
        )
        try:
            self.audit_store.append(record)
        except Exception as exc:
            if require_persistence:
                raise WallacAdapterError(
                    reason="audit_persistence_failed",
                    message=f"audit append failed on success path: {exc}",
                ) from exc
            pass


# --- module-level singleton for dependency injection ----------------------

_default_adapter: WallacAdapter | None = None


def get_wallac_adapter() -> WallacAdapter:
    """Return the process-wide Wallac adapter (created lazily)."""
    global _default_adapter
    if _default_adapter is None:
        from lab_copilot_gateway.approval import get_approval_store
        from lab_copilot_gateway.audit import get_audit_store
        from lab_copilot_gateway.elabftw import (
            _default_client_from_env as _default_elabftw_client_from_env,
        )
        from lab_copilot_gateway.policy import get_policy_engine

        _default_adapter = WallacAdapter(
            policy_engine=get_policy_engine(),
            audit_store=get_audit_store(),
            client=_default_client_from_env(),
            bridge_client=_default_bridge_client_from_env(),
            approval_store=get_approval_store(),
            elabftw_client=_default_elabftw_client_from_env(),
        )
    return _default_adapter


def reset_wallac_adapter(adapter: WallacAdapter | None = None) -> None:
    """Test helper: replace or clear the singleton."""
    global _default_adapter
    _default_adapter = adapter
