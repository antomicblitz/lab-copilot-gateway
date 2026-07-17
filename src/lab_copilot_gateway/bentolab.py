"""BentoLab adapter for the lab copilot gateway (C43).

Provides gateway-mediated access to the BentoLab HTTP wrapper API (C23/C24).
The adapter enforces identity, policy, audit, and approval before any
downstream call — the browser never talks to the BentoLab wrapper directly.

Tools exposed:
    - ``bentolab.get_status`` (tier 1, read-only) — device + service status
    - ``bentolab.validate_pcr_profile`` (tier 3, read-only) — profile validation
    - ``bentolab.dry_run_pcr_profile`` (tier 3, read-only) — simulate a run
    - ``bentolab.submit_pcr_run`` (tier 5, write) — start a real PCR run

The ``submit_pcr_run`` method supports ``plan_approval_id`` (C42 pattern)
so the plan executor can skip per-step approval consume when the plan-level
approval has already been consumed.

See ``docs/plans/lab-copilot-nlux-execution-plan-architecture.md`` for the
architectural rationale.
"""

from __future__ import annotations

import os
import secrets as _secrets
import time as _time
from dataclasses import dataclass, field
from typing import Any, Protocol

from lab_copilot_gateway.audit import AuditRecord, AuditStore
from lab_copilot_gateway.elabftw import (
    InvalidContextToken,
    MappedIdentity,
    UnmappedCaller,
    compute_args_hash,
    verify_context_token,
)
from lab_copilot_gateway.policy import Decision, PolicyEngine, PolicyRequest, Tier


# --- exceptions --------------------------------------------------------------


class BentoLabAdapterError(Exception):
    """Base class for BentoLab adapter errors.

    ``reason`` is a machine-readable code; ``message`` is human-readable.
    Mirrors ``ElabftwAdapterError`` so the /invoke endpoint can handle
    both uniformly.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message

    def to_dict(self) -> dict[str, str]:
        return {"reason": self.reason, "message": self.message}


class PolicyDenied(BentoLabAdapterError):
    """Policy engine refused the call."""

    def __init__(self, decision: Decision) -> None:
        super().__init__(
            reason=f"policy_denied:{decision.reason}",
            message=f"policy engine denied call ({decision.reason})",
        )


# --- downstream client -------------------------------------------------------


class BentoLabClient(Protocol):
    """BentoLab HTTP wrapper client surface the gateway adapter requires.

    Implementations: ``StubBentoLabClient`` for tests;
    ``HttpBentoLabClient`` for live calls.
    """

    def get_status(self) -> dict[str, Any]:
        """GET /status — device + temperature + run state."""
        ...

    def validate_profile(self, profile: dict[str, Any]) -> dict[str, Any]:
        """POST /profiles/validate — validate a PCR profile."""
        ...

    def dry_run(self, profile: dict[str, Any]) -> dict[str, Any]:
        """POST /runs/dry-run — simulate a run without hardware."""
        ...

    def start_run(
        self,
        profile: dict[str, Any],
        approval_id: str | None = None,
        operator: str | None = None,
    ) -> dict[str, Any]:
        """POST /runs — start a real PCR run (requires preflight + lock)."""
        ...


@dataclass
class StubBentoLabClient:
    """In-memory stub for testing — no network calls."""

    status_response: dict[str, Any] = field(
        default_factory=lambda: {
            "state": "idle",
            "device": "AA:BB:CC:DD:EE:FF",
            "temperature": {"current": 25.0, "lid": None, "block": 25.0},
            "run": None,
            "errors": [],
        }
    )
    validate_response: dict[str, Any] = field(
        default_factory=lambda: {"ok": True, "errors": [], "warnings": []}
    )
    dry_run_response: dict[str, Any] = field(
        default_factory=lambda: {
            "ok": True,
            "simulation": {
                "duration_s": 3600,
                "steps": [
                    {
                        "phase": "initial_denaturation",
                        "temperature": 95.0,
                        "duration_s": 30,
                    },
                ],
                "warnings": [],
            },
            "errors": [],
        }
    )
    start_run_response: dict[str, Any] = field(
        default_factory=lambda: {
            "ok": True,
            "run_id": "run-stub-1",
            "state": "accepted",
            "started_at": "2026-01-01T00:00:00Z",
        }
    )
    calls: list[dict[str, Any]] = field(default_factory=list)

    def get_status(self) -> dict[str, Any]:
        self.calls.append({"method": "get_status"})
        return self.status_response

    def validate_profile(self, profile: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"method": "validate_profile", "profile": profile})
        return self.validate_response

    def dry_run(self, profile: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"method": "dry_run", "profile": profile})
        return self.dry_run_response

    def start_run(
        self,
        profile: dict[str, Any],
        approval_id: str | None = None,
        operator: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(
            {"method": "start_run", "profile": profile, "approval_id": approval_id}
        )
        return self.start_run_response


class HttpBentoLabClient:
    """Live HTTP client for the BentoLab wrapper API.

    Talks to the FastAPI wrapper from C23/C24.  The base URL is read from
    ``LAB_COPILOT_BENTOLAB_BASE_URL`` at construction time.
    """

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def _request(
        self, method: str, path: str, json: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        import requests

        url = f"{self.base_url}{path}"
        resp = requests.request(method, url, json=json, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_status(self) -> dict[str, Any]:
        return self._request("GET", "/status")

    def validate_profile(self, profile: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/profiles/validate", json={"profile": profile})

    def dry_run(self, profile: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/runs/dry-run", json={"profile": profile})

    def start_run(
        self,
        profile: dict[str, Any],
        approval_id: str | None = None,
        operator: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"profile": profile}
        if approval_id:
            body["approval_id"] = approval_id
        if operator:
            body["operator"] = operator
        return self._request("POST", "/runs", json=body)


def _default_client_from_env() -> BentoLabClient | None:
    """Construct an HttpBentoLabClient from env, or None if not configured."""
    base_url = os.getenv("LAB_COPILOT_BENTOLAB_BASE_URL", "")
    if not base_url:
        return None
    return HttpBentoLabClient(base_url)


# --- result envelope ---------------------------------------------------------


@dataclass(frozen=True)
class BentoLabResult:
    """Result envelope returned by all BentoLab adapter methods."""

    tool_name: str
    result: dict[str, Any]
    audit_action_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"tool_name": self.tool_name, "result": self.result}
        if self.audit_action_id is not None:
            d["audit_action_id"] = self.audit_action_id
        return d


# --- tool name constants -----------------------------------------------------

TOOL_GET_STATUS = "bentolab.get_status"
TOOL_VALIDATE = "bentolab.validate_pcr_profile"
TOOL_DRY_RUN = "bentolab.dry_run_pcr_profile"
TOOL_SUBMIT = "bentolab.submit_pcr_run"

TOOL_TIER_STATUS = Tier.OPERATIONAL_READ_ONLY
TOOL_TIER_VALIDATE = Tier.VALIDATION_DRY_RUN
TOOL_TIER_DRY_RUN = Tier.VALIDATION_DRY_RUN
TOOL_TIER_SUBMIT = Tier.HARDWARE_EXECUTION


# --- adapter -----------------------------------------------------------------


class BentoLabAdapter:
    """Orchestrates BentoLab calls through the gateway-enforced path.

    Dependencies are injected so tests can swap the policy engine, audit
    store, and downstream client individually.  In production, the
    module-level ``get_bentolab_adapter()`` wires all the singletons.
    """

    policy_engine: PolicyEngine
    audit_store: AuditStore
    client: BentoLabClient | None
    approval_store: Any | None = None  # ApprovalStore | None
    action_id_prefix: str = "bentolab"

    def __init__(
        self,
        *,
        policy_engine: PolicyEngine,
        audit_store: AuditStore,
        client: BentoLabClient | None = None,
        approval_store: Any | None = None,
    ) -> None:
        self.policy_engine = policy_engine
        self.audit_store = audit_store
        self.client = client
        self.approval_store = approval_store

    # --- public API: tool entrypoints ----------------------------------

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
    ) -> BentoLabResult:
        """Read BentoLab device + service status (tier 1, read-only)."""
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
            exec_fn=lambda client: client.get_status(),
            tier_override=TOOL_TIER_STATUS,
        )

    def validate_pcr_profile(
        self,
        *,
        context_token: str,
        mapped_identity: MappedIdentity | None,
        profile: dict[str, Any],
        conversation_id: str | None = None,
        request_id: str | None = None,
        keycloak_subject: str | None = None,
        librechat_user_id: str | None = None,
        provider: str | None = None,
        model_id: str | None = None,
    ) -> BentoLabResult:
        """Validate a PCR profile against device limits (tier 3, read-only)."""
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
            args_for_hash={"tool": "validate_pcr_profile", "profile": profile},
            api_call_summary={"method": "POST", "path": "/profiles/validate"},
            exec_fn=lambda client: client.validate_profile(profile),
            tier_override=TOOL_TIER_VALIDATE,
        )

    def dry_run_pcr_profile(
        self,
        *,
        context_token: str,
        mapped_identity: MappedIdentity | None,
        profile: dict[str, Any],
        conversation_id: str | None = None,
        request_id: str | None = None,
        keycloak_subject: str | None = None,
        librechat_user_id: str | None = None,
        provider: str | None = None,
        model_id: str | None = None,
    ) -> BentoLabResult:
        """Simulate a PCR run without hardware side effects (tier 3, read-only)."""
        return self._execute(
            tool_name=TOOL_DRY_RUN,
            context_token=context_token,
            mapped_identity=mapped_identity,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            args_for_hash={"tool": "dry_run_pcr_profile", "profile": profile},
            api_call_summary={"method": "POST", "path": "/runs/dry-run"},
            exec_fn=lambda client: client.dry_run(profile),
            tier_override=TOOL_TIER_DRY_RUN,
        )

    def submit_pcr_run(
        self,
        *,
        context_token: str,
        mapped_identity: MappedIdentity | None,
        approval_id: str,
        approval_args: dict[str, Any],
        profile: dict[str, Any],
        conversation_id: str | None = None,
        request_id: str | None = None,
        keycloak_subject: str | None = None,
        librechat_user_id: str | None = None,
        provider: str | None = None,
        model_id: str | None = None,
        plan_approval_id: str | None = None,
    ) -> BentoLabResult:
        """Submit an approved PCR profile for hardware execution (tier 5).

        C43: when ``plan_approval_id`` is set, the plan executor has
        already consumed the plan-level approval.  Skip the per-step
        consume and use the plan approval_id in audit records (same C42
        pattern as Wallac).
        """
        return self._execute_submit(
            context_token=context_token,
            mapped_identity=mapped_identity,
            approval_id=approval_id,
            approval_args=approval_args,
            profile=profile,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            plan_approval_id=plan_approval_id,
        )

    # --- shared orchestration ------------------------------------------

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
    ) -> BentoLabResult:
        """Shared orchestration: token -> identity -> policy -> client -> audit.

        Mirrors the Wallac/OpenCloning adapter's _execute method.
        """
        early_hash = compute_args_hash(
            {**args_for_hash, "token_present": bool(context_token)}
        )

        # 0. Determine tier early (needed for read-only bypass).
        tier = tier_override or TOOL_TIER_STATUS

        # 1. Identity resolution (before token check so unmapped callers
        #    are denied even without a token).
        if mapped_identity is None:
            self._audit(
                tool_name=tool_name,
                policy_decision="deny",
                reason="unmapped_caller",
                mapped_identity=None,
                experiment_id=None,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=early_hash,
                error={"code": "UNMAPPED_CALLER"},
            )
            raise UnmappedCaller()

        # 2. Verify context token (skip for read-only tiers without a token).
        experiment_id: int | None = None
        if context_token:
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
            experiment_id = claims.experiment_id

            # Token-identity binding.
            if claims.mapped_elabftw_user_id != mapped_identity.elabftw_user_id:
                self._audit(
                    tool_name=tool_name,
                    policy_decision="deny",
                    reason="context_token_user_mismatch",
                    mapped_identity=mapped_identity,
                    experiment_id=experiment_id,
                    conversation_id=conversation_id,
                    request_id=request_id,
                    keycloak_subject=keycloak_subject,
                    librechat_user_id=librechat_user_id,
                    provider=provider,
                    model_id=model_id,
                    tool_args_hash=early_hash,
                    error={"code": "CONTEXT_TOKEN_USER_MISMATCH"},
                )
                raise InvalidContextToken("token user does not match resolved identity")
        elif tier >= Tier.HARDWARE_EXECUTION:
            # Hardware execution requires a context token.
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

        tool_args_hash = compute_args_hash(
            {**args_for_hash, "experiment_id": experiment_id}
        )

        # 4. Policy decision.
        policy_req = PolicyRequest(
            tool_name=tool_name,
            tier=tier,
            adapter="bentolab",
            user_id=mapped_identity.elabftw_user_id,
            team_id=mapped_identity.elabftw_team_id,
        )
        decision = self.policy_engine.decide(policy_req)

        if decision.decision != "allow":
            self._audit(
                tool_name=tool_name,
                policy_decision="deny",
                reason=decision.reason,
                mapped_identity=mapped_identity,
                experiment_id=experiment_id,
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

        # 5. Client must be configured.
        if self.client is None:
            self._audit(
                tool_name=tool_name,
                policy_decision="allow",
                reason="bentolab_not_configured",
                mapped_identity=mapped_identity,
                experiment_id=experiment_id,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=tool_args_hash,
                error={"code": "BENTOLAB_NOT_CONFIGURED"},
            )
            raise BentoLabAdapterError(
                reason="bentolab_not_configured",
                message=(
                    "BentoLab client not configured (set LAB_COPILOT_BENTOLAB_BASE_URL)"
                ),
            )

        # 6. Execute downstream call.
        try:
            result_data = exec_fn(self.client)
        except Exception as exc:
            self._audit(
                tool_name=tool_name,
                policy_decision="allow",
                reason="client_error",
                mapped_identity=mapped_identity,
                experiment_id=experiment_id,
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
            raise BentoLabAdapterError(
                reason="client_error",
                message=f"BentoLab client raised {type(exc).__name__}: {exc}",
            ) from exc

        # 7. Success audit.
        action_id = f"{self.action_id_prefix}-{tool_name.split('.')[-1]}-{int(_time.time() * 1000)}-{_secrets.token_hex(4)}"

        self._audit(
            tool_name=tool_name,
            policy_decision="allow",
            reason="call_succeeded",
            mapped_identity=mapped_identity,
            experiment_id=experiment_id,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            tool_args_hash=tool_args_hash,
            api_call_summary=api_call_summary,
            result_summary=result_data,
            require_persistence=True,
            action_id=action_id,
        )

        return BentoLabResult(
            tool_name=tool_name,
            result=result_data,
            audit_action_id=action_id,
        )

    def _execute_submit(
        self,
        *,
        context_token: str,
        mapped_identity: MappedIdentity | None,
        approval_id: str,
        approval_args: dict[str, Any],
        profile: dict[str, Any],
        conversation_id: str | None,
        request_id: str | None,
        keycloak_subject: str | None,
        librechat_user_id: str | None,
        provider: str | None,
        model_id: str | None,
        plan_approval_id: str | None = None,
    ) -> BentoLabResult:
        """Submit orchestration: token -> identity -> policy -> approval -> client -> audit.

        C43: when ``plan_approval_id`` is set, skip per-step approval consume.
        """
        early_hash = compute_args_hash(
            {
                "tool": "submit_pcr_run",
                "profile": profile,
                "token_present": bool(context_token),
            }
        )

        # 1. Verify context token.
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
                "tool": "submit_pcr_run",
                "profile": profile,
                "experiment_id": claims.experiment_id,
            }
        )

        # 2. Identity resolution.
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

        # 3. Token-identity binding.
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

        # 4. Policy decision (tier 5 HARDWARE_EXECUTION).
        policy_req = PolicyRequest(
            tool_name=TOOL_SUBMIT,
            tier=TOOL_TIER_SUBMIT,
            adapter="bentolab",
            user_id=mapped_identity.elabftw_user_id,
            team_id=mapped_identity.elabftw_team_id,
            has_approval=bool(approval_id or plan_approval_id),
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

        # 5. Approval token consume.
        # C43: when plan_approval_id is set, skip per-step consume.
        if plan_approval_id:
            effective_approval_id = plan_approval_id
        elif self.approval_store is not None:
            try:
                self.approval_store.consume(
                    approval_id,
                    tool_name=TOOL_SUBMIT,
                    args_hash=compute_args_hash(approval_args),
                    target_record=f"bentolab:run:{claims.experiment_id}",
                )
                effective_approval_id = approval_id
            except Exception as exc:
                self._audit(
                    tool_name=TOOL_SUBMIT,
                    policy_decision="deny",
                    reason=f"approval_consume_failed:{exc}",
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
                    error={
                        "code": "APPROVAL_CONSUME_FAILED",
                        "exception": type(exc).__name__,
                    },
                )
                raise BentoLabAdapterError(
                    reason="approval_consume_failed",
                    message=f"approval token consume failed: {exc}",
                ) from exc
        else:
            effective_approval_id = approval_id

        # 6. Client must be configured.
        if self.client is None:
            self._audit(
                tool_name=TOOL_SUBMIT,
                policy_decision="allow",
                reason="bentolab_not_configured",
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
                error={"code": "BENTOLAB_NOT_CONFIGURED"},
            )
            raise BentoLabAdapterError(
                reason="bentolab_not_configured",
                message=(
                    "BentoLab client not configured (set LAB_COPILOT_BENTOLAB_BASE_URL)"
                ),
            )

        # 7. Execute downstream call.
        try:
            result_data = self.client.start_run(
                profile=profile,
                approval_id=effective_approval_id,
                operator=mapped_identity.elabftw_user_id,
            )
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
                api_call_summary={"method": "POST", "path": "/runs"},
                error={
                    "code": "CLIENT_ERROR",
                    "exception": type(exc).__name__,
                    "message": str(exc),
                },
            )
            raise BentoLabAdapterError(
                reason="client_error",
                message=f"BentoLab client raised {type(exc).__name__}: {exc}",
            ) from exc

        # 8. Success audit.
        action_id = f"{self.action_id_prefix}-submit-{int(_time.time() * 1000)}-{_secrets.token_hex(4)}"

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
            api_call_summary={"method": "POST", "path": "/runs"},
            result_summary=result_data,
            require_persistence=True,
            action_id=action_id,
        )

        return BentoLabResult(
            tool_name=TOOL_SUBMIT,
            result=result_data,
            audit_action_id=action_id,
        )

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
        """Persist an audit record (mirrors Wallac/OpenCloning pattern)."""
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
                raise BentoLabAdapterError(
                    reason="audit_persistence_failed",
                    message=f"audit append failed on success path: {exc}",
                ) from exc


# --- module-level singleton --------------------------------------------------

_default_adapter: BentoLabAdapter | None = None


def get_bentolab_adapter() -> BentoLabAdapter:
    """Return the process-wide BentoLab adapter (created lazily)."""
    global _default_adapter
    if _default_adapter is None:
        from lab_copilot_gateway.approval import get_approval_store
        from lab_copilot_gateway.audit import get_audit_store
        from lab_copilot_gateway.policy import get_policy_engine

        _default_adapter = BentoLabAdapter(
            policy_engine=get_policy_engine(),
            audit_store=get_audit_store(),
            client=_default_client_from_env(),
            approval_store=get_approval_store(),
        )
    return _default_adapter


def reset_bentolab_adapter(
    adapter: BentoLabAdapter | None = None,
) -> None:
    """Test helper: replace or clear the singleton."""
    global _default_adapter
    _default_adapter = adapter
