"""Wallac adapter for the lab copilot gateway (C18).

Orchestrates gateway-enforced calls to the Wallac Victor2 vm-agent service
for instrument status reads.  The vm-agent is a Python HTTP service running
on a Windows 7 VM (192.168.122.203:8420) that controls the Wallac Victor2
plate reader via COM.

This module follows the orchestration template established by the eLabFTW
adapter (C08/C09) and OpenCloning adapter (C16):

    context token verification
        → identity resolution (mapped user required)
        → policy engine decision (tier 2 OPERATIONAL_READ_ONLY, read-only)
        → downstream client call (HTTP for production, stub for tests)
        → audit record with tool_args_hash

Key differences from the eLabFTW/OpenCloning adapters:

    * The vm-agent uses optional Bearer token auth (not API key).
    * Wallac status is not experiment-scoped — the context token's
      experiment_id is carried as a context_ref for audit correlation
      but is not used in the downstream call.
    * The vm-agent may be offline (VM down, network issue) — the adapter
      handles this gracefully with a structured error.

Tools exposed (already in the C06 tool registry):

    wallac.get_status — read instrument status and current job

Future chunks (C19/C20) will add:
    wallac.propose_generated_protocol
    wallac.validate_generated_protocol
    wallac.prepare_submission_package
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from lab_copilot_gateway.approval import compute_args_hash
from lab_copilot_gateway.audit import AuditRecord, AuditStore
from lab_copilot_gateway.elabftw import (
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
TOOL_TIER = Tier.OPERATIONAL_READ_ONLY


@dataclass
class WallacAdapter:
    """Orchestrates Wallac vm-agent calls through the gateway-enforced path.

    Dependencies are injected so tests can swap the policy engine, audit
    store, and downstream client individually.  In production, the
    module-level ``get_wallac_adapter()`` wires all the singletons.
    """

    policy_engine: PolicyEngine
    audit_store: AuditStore
    client: WallacClient | None
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

    # --- shared orchestration --------------------------------------------

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
    ) -> WallacResult:
        """Shared orchestration: token → identity → policy → client → audit.

        Mirrors the OpenCloning adapter's _execute method.
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

        # 4. Policy decision (tier 2 read-only, no approval needed).
        policy_req = PolicyRequest(
            tool_name=tool_name,
            tier=TOOL_TIER,
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
        if self.client is None:
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
        try:
            result = exec_fn(self.client)
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
        from lab_copilot_gateway.audit import get_audit_store
        from lab_copilot_gateway.policy import get_policy_engine

        _default_adapter = WallacAdapter(
            policy_engine=get_policy_engine(),
            audit_store=get_audit_store(),
            client=_default_client_from_env(),
        )
    return _default_adapter


def reset_wallac_adapter(adapter: WallacAdapter | None = None) -> None:
    """Test helper: replace or clear the singleton."""
    global _default_adapter
    _default_adapter = adapter
