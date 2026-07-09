"""Plan execution engine for batch change-set approval (C39).

Executes validated execution plans (C38 schema) step-by-step with
per-step audit, partial-failure handling, and plan-hash-bound approval.

Flow:

    1. Validate the plan (C38 ``validate_plan``).
    2. Verify + consume the plan-level approval (bound to plan hash).
    3. Execute read steps (using ``ElabftwReadAdapter``).
    4. Execute write steps in declared order (using ``ElabftwWriteAdapter``
       with ``plan_approval_id`` — skips per-step approval consume because
       the plan-level approval already covers all step args).
    5. Record per-step results (success / failure / skipped).
    6. Stop on first write failure (partial failure — remaining steps
       are marked ``skipped``).
    7. Return ``PlanExecutionResult`` with per-step status and audit IDs.

Security model:

    * The approval token is bound to ``plan.hash()`` — any change to the
      plan content invalidates the approval.
    * Write steps still enforce append-only checks and body-drift guards
      (the write adapter's safety checks are NOT skipped).
    * Per-step audit records are written by the write adapter's
      ``_execute()`` method; the plan executor records its own audit
      row for the plan-level approval consume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lab_copilot_gateway.approval import ApprovalStore
from lab_copilot_gateway.elabftw import (
    ElabftwAdapterError,
    ElabftwReadAdapter,
    ElabftwWriteAdapter,
    MappedIdentity,
    TOOL_NAME_READ_BY_ID,
    TOOL_NAME_SEARCH,
)
from lab_copilot_gateway.bentolab import (
    BentoLabAdapter,
    BentoLabAdapterError,
    TOOL_DRY_RUN as TOOL_BENTOLAB_DRY_RUN,
    TOOL_GET_STATUS as TOOL_BENTOLAB_STATUS,
    TOOL_SUBMIT as TOOL_BENTOLAB_SUBMIT,
    TOOL_VALIDATE as TOOL_BENTOLAB_VALIDATE,
)
from lab_copilot_gateway.opencloning import (
    OpenCloningAdapter,
    OpenCloningAdapterError,
    TOOL_ASSEMBLY,
    TOOL_CALL,
    TOOL_MANUAL,
    TOOL_OLIGO,
    TOOL_PARSE,
    TOOL_WRITEBACK,
)
from lab_copilot_gateway.plan import (
    Plan,
    PlanRiskTier,
    PlanStep,
    PlanValidationError,
    compute_plan_hash,
    validate_or_raise,
)
from lab_copilot_gateway.wallac import (
    WallacAdapter,
    WallacAdapterError,
    TOOL_GET_STATUS as TOOL_WALLAC_STATUS,
    TOOL_SUBMIT as TOOL_WALLAC_SUBMIT,
    TOOL_VALIDATE as TOOL_WALLAC_VALIDATE,
)


# --- retry semantics (C40) ---------------------------------------------------
#
# Error reasons that are transient — the step can be retried with a fresh
# plan-level approval.  Permanent failures (body drift, append-only
# violation, approval denied) require the user to re-preview and re-approve
# a new plan, not just retry.

RETRYABLE_REASONS: frozenset[str] = frozenset(
    {
        "client_error",
        "state_check_failed",
        "audit_persistence_failed",
    }
)


def _is_retryable(error_str: str) -> bool:
    """Check if an error string starts with a retryable reason code."""
    return any(error_str.startswith(r) for r in RETRYABLE_REASONS)


def _check_writeback_failure(
    result: dict[str, Any] | None,
    tool_name: str,
) -> str | None:
    """Check if a writeback result has a failed/partial status.

    Returns an error string (starting with "client_error:" for retryability)
    if the writeback failed or was partial, or None if it succeeded.
    """
    if tool_name != TOOL_WRITEBACK or not isinstance(result, dict):
        return None
    inner = result.get("result")
    if not isinstance(inner, dict):
        return None
    wb = inner.get("writeback_result")
    if not isinstance(wb, dict) or wb.get("status") not in ("failed", "partial"):
        return None
    failed_steps = [s for s in wb.get("steps", []) if s.get("status") == "failed"]
    detail = "; ".join(
        f"{s.get('step_name', '?')}: {s.get('error', 'unknown')}" for s in failed_steps
    )
    return f"client_error: writeback {wb['status']} — {detail}"


# --- result models -----------------------------------------------------------


@dataclass(frozen=True)
class StepResult:
    """Outcome of one step in a plan execution (C40 enriched).

    Fields:
        step_index: position in the plan's read+write sequence.
        tool_name: the gateway tool invoked.
        status: "success" | "failed" | "skipped".
        result: the tool's return value (dict) on success.
        error: error string on failure (format: "reason: message").
        audit_action_id: the audit row's action_id for mutating steps
            (extracted from WriteResult).  None for reads or failures
            before the audit row was written.
        retryable: whether the step can be retried with a fresh plan-level
            approval (C40).  True for transient failures (client_error,
            state_check_failed, audit_persistence_failed).  False for
            permanent failures (body_drift, append_only_violation, etc.).
    """

    step_index: int
    tool_name: str
    status: str  # "success" | "failed" | "skipped"
    result: dict[str, Any] | None = None
    error: str | None = None
    audit_action_id: str | None = None
    retryable: bool = False
    # Internal: set when a write step fails, to skip remaining steps.
    # Not serialized — only used within execute().
    write_failed: bool = False

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "step_index": self.step_index,
            "tool_name": self.tool_name,
            "status": self.status,
            "retryable": self.retryable,
        }
        if self.result is not None:
            d["result"] = self.result
        if self.error is not None:
            d["error"] = self.error
        if self.audit_action_id is not None:
            d["audit_action_id"] = self.audit_action_id
        return d


@dataclass(frozen=True)
class PlanExecutionResult:
    """Outcome of a plan execution (C39).

    ``status`` is one of:
        - ``completed``: all steps succeeded.
        - ``partial_failure``: some steps succeeded, at least one failed.
        - ``rejected``: plan validation or approval consume failed (no steps executed).
        - ``error``: unexpected error during execution.
    """

    plan_id: str
    plan_hash: str
    status: str
    step_results: list[StepResult]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "plan_hash": self.plan_hash,
            "status": self.status,
            "step_results": [r.to_dict() for r in self.step_results],
            "summary": self.summary,
        }


# --- executor ----------------------------------------------------------------


class PlanExecutor:
    """Executes validated plans with plan-hash-bound approval (C39).

    Dependencies are injected for testability.  In production,
    ``get_plan_executor()`` wires the singletons.

    The ``opencloning_adapter`` is optional — plans without OpenCloning
    steps do not require it (C41).
    """

    def __init__(
        self,
        *,
        approval_store: ApprovalStore,
        read_adapter: ElabftwReadAdapter,
        write_adapter: ElabftwWriteAdapter,
        opencloning_adapter: OpenCloningAdapter | None = None,
        wallac_adapter: WallacAdapter | None = None,
        bentolab_adapter: BentoLabAdapter | None = None,
    ) -> None:
        self.approval_store = approval_store
        self.read_adapter = read_adapter
        self.write_adapter = write_adapter
        self.opencloning_adapter = opencloning_adapter
        self.wallac_adapter = wallac_adapter
        self.bentolab_adapter = bentolab_adapter

    def _consume_plan_approval(
        self,
        plan: Plan,
        plan_hash: str,
        approval_id: str | None,
        autonomy_active: bool,
    ) -> str:
        """Consume plan-level approval. Returns effective_approval_id or error message.

        Returns the effective_approval_id on success, or an error
        string starting with "error:" on failure.
        """
        if autonomy_active:
            return f"autonomous:{plan.plan_id}"

        if not approval_id:
            return "error:Approval required but no approval_id provided"

        try:
            self.approval_store.consume(
                approval_id,
                tool_name="plan.execute",
                args_hash=plan_hash,
                target_record=f"plan:{plan.plan_id}",
            )
            return approval_id
        except Exception as exc:
            return f"error:Approval consume failed: {exc}"

    def _resolve_plan_approval(
        self,
        *,
        plan: Plan,
        plan_hash: str,
        approval_id: str | None,
    ) -> tuple[bool, str] | PlanExecutionResult:
        """Resolve autonomy status, budget, and effective plan approval.

        Returns ``(autonomy_active, effective_approval_id)`` on success,
        or a ``PlanExecutionResult`` error on budget failure.
        """
        from lab_copilot_gateway.config import (
            get_autonomy_max_steps,
            is_autonomy_enabled,
        )

        autonomy_active = (
            plan.risk_tier == PlanRiskTier.AUTONOMOUS and is_autonomy_enabled()
        )

        if autonomy_active:
            max_steps = get_autonomy_max_steps()
            total_steps = len(plan.reads) + len(plan.writes)
            if total_steps > max_steps:
                return PlanExecutionResult(
                    plan_id=plan.plan_id,
                    plan_hash=plan_hash,
                    status="rejected",
                    step_results=[],
                    summary=(
                        f"Autonomous plan exceeds budget: {total_steps} steps "
                        f"> {max_steps} max (LAB_COPILOT_AUTONOMY_MAX_STEPS)"
                    ),
                )
            return (True, f"autonomous:{plan.plan_id}")

        # Non-autonomous: consume plan-level approval.
        approval_result = self._consume_plan_approval(
            plan, plan_hash, approval_id, autonomy_active=False
        )
        if approval_result.startswith("error:"):
            return PlanExecutionResult(
                plan_id=plan.plan_id,
                plan_hash=plan_hash,
                status="rejected",
                step_results=[],
                summary=approval_result.removeprefix("error:"),
            )
        return (False, approval_result)

    def _execute_all_steps(
        self,
        *,
        plan: Plan,
        effective_approval_id: str,
        context_token: str,
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None,
        request_id: str | None,
        keycloak_subject: str | None,
        librechat_user_id: str | None,
        provider: str | None,
        model_id: str | None,
    ) -> list[StepResult]:
        """Execute all plan steps, stopping writes after first failure."""
        step_results: list[StepResult] = []
        all_steps = [(i, step, "read") for i, step in enumerate(plan.reads)]
        all_steps += [
            (len(plan.reads) + i, step, "write") for i, step in enumerate(plan.writes)
        ]

        write_failed = False
        for step_index, step, step_type in all_steps:
            if write_failed:
                step_results.append(
                    StepResult(
                        step_index=step_index,
                        tool_name=step.tool_name,
                        status="skipped",
                        error="skipped due to prior write failure",
                    )
                )
                continue

            step_result = self._execute_single_step(
                step=step,
                step_type=step_type,
                step_index=step_index,
                context_token=context_token,
                plan_approval_id=effective_approval_id,
                mapped_identity=mapped_identity,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
            )
            if step_result.write_failed:
                write_failed = True
            step_results.append(step_result)

        return step_results

    def _execute_single_step(
        self,
        *,
        step: PlanStep,
        step_type: str,
        step_index: int,
        context_token: str,
        plan_approval_id: str,
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None,
        request_id: str | None,
        keycloak_subject: str | None,
        librechat_user_id: str | None,
        provider: str | None,
        model_id: str | None,
    ) -> StepResult:
        """Execute a single plan step and return its result (never raises)."""
        try:
            if step_type == "read":
                result = self._execute_read(
                    step=step,
                    context_token=context_token,
                    mapped_identity=mapped_identity,
                    conversation_id=conversation_id,
                    request_id=request_id,
                    keycloak_subject=keycloak_subject,
                    librechat_user_id=librechat_user_id,
                    provider=provider,
                    model_id=model_id,
                )
            else:
                result = self._execute_write(
                    step=step,
                    context_token=context_token,
                    plan_approval_id=plan_approval_id,
                    mapped_identity=mapped_identity,
                    conversation_id=conversation_id,
                    request_id=request_id,
                    keycloak_subject=keycloak_subject,
                    librechat_user_id=librechat_user_id,
                    provider=provider,
                    model_id=model_id,
                )
            # C52: check if the writeback returned a failed/partial result.
            # The writeback no longer raises on step failures — it returns a
            # WritebackResult with status "failed" or "partial". Convert that
            # to a failed StepResult so the plan executor reports it correctly.
            wb_failure = _check_writeback_failure(result, step.tool_name)
            if wb_failure is not None:
                return StepResult(
                    step_index=step_index,
                    tool_name=step.tool_name,
                    status="failed",
                    result=result,
                    error=wb_failure,
                    retryable=True,
                    write_failed=True,
                )
            return StepResult(
                step_index=step_index,
                tool_name=step.tool_name,
                status="success",
                result=result,
                audit_action_id=(
                    result.get("audit_action_id")
                    if step_type == "write" and isinstance(result, dict)
                    else None
                ),
            )
        except (
            ElabftwAdapterError,
            OpenCloningAdapterError,
            WallacAdapterError,
            BentoLabAdapterError,
        ) as exc:
            error_str = f"{exc.reason}: {exc.message}"
            return StepResult(
                step_index=step_index,
                tool_name=step.tool_name,
                status="failed",
                error=error_str,
                retryable=_is_retryable(exc.reason),
                write_failed=(step_type == "write"),
            )
        except Exception as exc:
            error_str = f"unexpected error: {type(exc).__name__}: {exc}"
            return StepResult(
                step_index=step_index,
                tool_name=step.tool_name,
                status="failed",
                error=error_str,
                retryable=True,
                write_failed=(step_type == "write"),
            )

    def execute(
        self,
        plan: Plan,
        *,
        approval_id: str | None = None,
        context_token: str,
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None = None,
        request_id: str | None = None,
        keycloak_subject: str | None = None,
        librechat_user_id: str | None = None,
        provider: str | None = None,
        model_id: str | None = None,
    ) -> PlanExecutionResult:
        """Execute a validated plan with plan-level approval.

        Returns a ``PlanExecutionResult`` — never raises (all errors are
        captured as step results or in the result status).

        C29: When the plan's risk tier is AUTONOMOUS and autonomy is
        enabled (``LAB_COPILOT_AUTONOMY_ENABLED=1``), the plan-level
        approval consume is skipped and a synthetic ``plan_approval_id``
        (``"autonomous:{plan_id}"``) is used for write-step audit records.
        Budget enforcement (``LAB_COPILOT_AUTONOMY_MAX_STEPS``) limits
        the total number of steps.
        """
        plan_hash = compute_plan_hash(plan)

        # 1. Validate the plan.
        try:
            validate_or_raise(plan)
        except PlanValidationError as exc:
            return PlanExecutionResult(
                plan_id=plan.plan_id,
                plan_hash=plan_hash,
                status="rejected",
                step_results=[],
                summary=f"Plan validation failed: {'; '.join(exc.errors)}",
            )

        # C29: Autonomy setup and approval resolution.
        autonomy_result = self._resolve_plan_approval(
            plan=plan,
            plan_hash=plan_hash,
            approval_id=approval_id,
        )
        if isinstance(autonomy_result, PlanExecutionResult):
            return autonomy_result
        autonomy_active, effective_approval_id = autonomy_result

        # At this point effective_approval_id is guaranteed to be non-None
        # (autonomous synthetic ID or validated through the approval_id guard).
        assert effective_approval_id is not None

        # 3. Execute steps.
        step_results = self._execute_all_steps(
            plan=plan,
            effective_approval_id=effective_approval_id,
            context_token=context_token,
            mapped_identity=mapped_identity,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
        )

        # 4. Build result summary.
        succeeded = sum(1 for r in step_results if r.status == "success")
        failed = sum(1 for r in step_results if r.status == "failed")
        skipped = sum(1 for r in step_results if r.status == "skipped")

        if failed == 0:
            status = "completed"
            summary = f"All {succeeded} steps completed successfully."
        else:
            status = "partial_failure"
            summary = f"{succeeded} succeeded, {failed} failed, {skipped} skipped."

        return PlanExecutionResult(
            plan_id=plan.plan_id,
            plan_hash=plan_hash,
            status=status,
            step_results=step_results,
            summary=summary,
        )

    def _execute_read(
        self,
        *,
        step: PlanStep,
        context_token: str,
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None,
        request_id: str | None,
        keycloak_subject: str | None,
        librechat_user_id: str | None,
        provider: str | None,
        model_id: str | None,
    ) -> dict[str, Any]:
        """Execute a read step using the appropriate adapter."""
        # --- eLabFTW reads ---
        if step.tool_name == TOOL_NAME_SEARCH:
            result = self.read_adapter.search_my_experiments(
                query=step.args.get("query", ""),
                limit=step.args.get("limit", 20),
                offset=step.args.get("offset", 0),
                mapped_identity=mapped_identity,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
            )
            return result.to_dict()
        elif step.tool_name == TOOL_NAME_READ_BY_ID:
            experiment_id = int(step.args.get("experiment_id", 0))
            result = self.read_adapter.read_experiment_by_id(
                experiment_id=experiment_id,
                mapped_identity=mapped_identity,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
            )
            return result.to_dict()

        # --- OpenCloning reads (C41) ---
        opencloning_tools = {
            TOOL_PARSE,
            TOOL_MANUAL,
            TOOL_OLIGO,
            TOOL_ASSEMBLY,
            TOOL_CALL,
        }
        if step.tool_name in opencloning_tools:
            return self._execute_opencloning_read(
                step=step,
                context_token=context_token,
                mapped_identity=mapped_identity,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
            )

        # --- Wallac reads (C42) ---
        wallac_read_tools = {TOOL_WALLAC_STATUS, TOOL_WALLAC_VALIDATE}
        if step.tool_name in wallac_read_tools:
            return self._execute_wallac_read(
                step=step,
                context_token=context_token,
                mapped_identity=mapped_identity,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
            )

        # --- BentoLab reads (C43) ---
        bentolab_read_tools = {
            TOOL_BENTOLAB_STATUS,
            TOOL_BENTOLAB_VALIDATE,
            TOOL_BENTOLAB_DRY_RUN,
        }
        if step.tool_name in bentolab_read_tools:
            if self.bentolab_adapter is None:
                raise BentoLabAdapterError(
                    reason="bentolab_not_configured",
                    message=(
                        "BentoLab adapter not configured for plan execution "
                        "(set LAB_COPILOT_BENTOLAB_BASE_URL)"
                    ),
                )
            profile = step.args.get("profile", {})
            if step.tool_name == TOOL_BENTOLAB_STATUS:
                result = self.bentolab_adapter.get_status(
                    context_token=context_token,
                    mapped_identity=mapped_identity,
                    conversation_id=conversation_id,
                    request_id=request_id,
                    keycloak_subject=keycloak_subject,
                    librechat_user_id=librechat_user_id,
                    provider=provider,
                    model_id=model_id,
                )
            elif step.tool_name == TOOL_BENTOLAB_VALIDATE:
                result = self.bentolab_adapter.validate_pcr_profile(
                    context_token=context_token,
                    mapped_identity=mapped_identity,
                    profile=profile,
                    conversation_id=conversation_id,
                    request_id=request_id,
                    keycloak_subject=keycloak_subject,
                    librechat_user_id=librechat_user_id,
                    provider=provider,
                    model_id=model_id,
                )
            else:  # TOOL_BENTOLAB_DRY_RUN
                result = self.bentolab_adapter.dry_run_pcr_profile(
                    context_token=context_token,
                    mapped_identity=mapped_identity,
                    profile=profile,
                    conversation_id=conversation_id,
                    request_id=request_id,
                    keycloak_subject=keycloak_subject,
                    librechat_user_id=librechat_user_id,
                    provider=provider,
                    model_id=model_id,
                )
            return result.to_dict()

        raise ElabftwAdapterError(
            reason="unsupported_read_tool",
            message=f"plan read step tool {step.tool_name!r} not supported by plan executor",
        )

    def _execute_opencloning_read(
        self,
        *,
        step: PlanStep,
        context_token: str,
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None,
        request_id: str | None,
        keycloak_subject: str | None,
        librechat_user_id: str | None,
        provider: str | None,
        model_id: str | None,
    ) -> dict[str, Any]:
        """Execute an OpenCloning read step."""
        if self.opencloning_adapter is None:
            raise OpenCloningAdapterError(
                reason="opencloning_not_configured",
                message=(
                    "OpenCloning adapter not configured for plan execution "
                    "(set LAB_COPILOT_OPENCLONING_BASE_URL)"
                ),
            )
        if step.tool_name == TOOL_PARSE:
            result = self.opencloning_adapter.parse_sequence_file(
                context_token=context_token,
                file_content=step.args.get("file_content", ""),
                file_format=step.args.get("file_format", "genbank"),
                mapped_identity=mapped_identity,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
            )
        elif step.tool_name == TOOL_MANUAL:
            result = self.opencloning_adapter.manual_sequence(
                context_token=context_token,
                sequence=step.args.get("sequence", ""),
                circular=step.args.get("circular", False),
                mapped_identity=mapped_identity,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
            )
        elif step.tool_name == TOOL_OLIGO:
            result = self.opencloning_adapter.oligo_hybridization(
                context_token=context_token,
                forward_oligo=step.args.get("forward_oligo", ""),
                reverse_oligo=step.args.get("reverse_oligo", ""),
                minimal_annealing=step.args.get("minimal_annealing", 20),
                mapped_identity=mapped_identity,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
            )
        elif step.tool_name == TOOL_CALL:
            result = self.opencloning_adapter.call_endpoint(
                context_token=context_token,
                endpoint=step.args.get("endpoint", "/"),
                body=step.args.get("body", {}),
                mapped_identity=mapped_identity,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
            )
        else:  # TOOL_ASSEMBLY
            result = self.opencloning_adapter.simulate_assembly(
                context_token=context_token,
                sequences=step.args.get("sequences", []),
                source=step.args.get(
                    "source", {"id": 0, "type": "GibsonAssemblySource"}
                ),
                mapped_identity=mapped_identity,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
            )
        return result.to_dict()

    def _execute_wallac_read(
        self,
        *,
        step: PlanStep,
        context_token: str,
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None,
        request_id: str | None,
        keycloak_subject: str | None,
        librechat_user_id: str | None,
        provider: str | None,
        model_id: str | None,
    ) -> dict[str, Any]:
        """Execute a Wallac read step."""
        if self.wallac_adapter is None:
            raise WallacAdapterError(
                reason="wallac_not_configured",
                message=(
                    "Wallac adapter not configured for plan execution "
                    "(set LAB_COPILOT_WALLAC_VM_AGENT_URL)"
                ),
            )
        if step.tool_name == TOOL_WALLAC_STATUS:
            result = self.wallac_adapter.get_status(
                context_token=context_token,
                mapped_identity=mapped_identity,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
            )
        else:  # TOOL_WALLAC_VALIDATE
            result = self.wallac_adapter.validate_generated_protocol(
                context_token=context_token,
                mapped_identity=mapped_identity,
                job_item_id=step.args.get("job_item_id", 0),
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
            )
        return result.to_dict()

    def _execute_write(
        self,
        *,
        step: PlanStep,
        context_token: str,
        plan_approval_id: str,
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None,
        request_id: str | None,
        keycloak_subject: str | None,
        librechat_user_id: str | None,
        provider: str | None,
        model_id: str | None,
    ) -> dict[str, Any]:
        """Execute a write step using the appropriate adapter with plan-level approval."""
        if step.tool_name == "elabftw.edit_experiment_section":
            result = self.write_adapter.edit_experiment_section(
                context_token=context_token,
                approval_id=plan_approval_id,
                approval_args=step.args,
                mapped_identity=mapped_identity,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                plan_approval_id=plan_approval_id,
            )
            return result.to_dict()
        elif step.tool_name == TOOL_WRITEBACK:
            # C41: OpenCloning writeback artifact — uses the OpenCloning
            # adapter with plan_approval_id to skip per-step approval consume.
            if self.opencloning_adapter is None:
                raise OpenCloningAdapterError(
                    reason="opencloning_not_configured",
                    message=(
                        "OpenCloning adapter not configured for plan execution "
                        "(set LAB_COPILOT_OPENCLONING_BASE_URL)"
                    ),
                )
            result = self.opencloning_adapter.writeback_artifact(
                context_token=context_token,
                approval_id=plan_approval_id,
                approval_args=step.args,
                mapped_identity=mapped_identity,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                plan_approval_id=plan_approval_id,
            )
            return result.to_dict()
        elif step.tool_name == TOOL_WALLAC_SUBMIT:
            # C42: Wallac submit — uses the Wallac adapter with
            # plan_approval_id to skip per-step approval consume.
            if self.wallac_adapter is None:
                raise WallacAdapterError(
                    reason="wallac_not_configured",
                    message=(
                        "Wallac adapter not configured for plan execution "
                        "(set LAB_COPILOT_WALLAC_VM_AGENT_URL)"
                    ),
                )
            result = self.wallac_adapter.submit_generated_protocol(
                context_token=context_token,
                approval_id=plan_approval_id,
                approval_args=step.args,
                job_item_id=step.args.get("job_item_id", 0),
                mapped_identity=mapped_identity,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                plan_approval_id=plan_approval_id,
            )
            return result.to_dict()
        elif step.tool_name == TOOL_BENTOLAB_SUBMIT:
            # C43: BentoLab submit — uses the BentoLab adapter with
            # plan_approval_id to skip per-step approval consume.
            if self.bentolab_adapter is None:
                raise BentoLabAdapterError(
                    reason="bentolab_not_configured",
                    message=(
                        "BentoLab adapter not configured for plan execution "
                        "(set LAB_COPILOT_BENTOLAB_BASE_URL)"
                    ),
                )
            result = self.bentolab_adapter.submit_pcr_run(
                context_token=context_token,
                approval_id=plan_approval_id,
                approval_args=step.args,
                profile=step.args.get("profile", {}),
                mapped_identity=mapped_identity,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                plan_approval_id=plan_approval_id,
            )
            return result.to_dict()
        else:
            raise ElabftwAdapterError(
                reason="unsupported_write_tool",
                message=f"plan write step tool {step.tool_name!r} not supported by plan executor",
            )


# --- module-level singleton --------------------------------------------------

_default_executor: PlanExecutor | None = None


def get_plan_executor() -> PlanExecutor:
    """Return the process-wide plan executor (created lazily)."""
    global _default_executor
    if _default_executor is None:
        from lab_copilot_gateway.approval import get_approval_store
        from lab_copilot_gateway.elabftw import (
            get_elabftw_read_adapter,
            get_elabftw_write_adapter,
        )
        from lab_copilot_gateway.bentolab import get_bentolab_adapter
        from lab_copilot_gateway.opencloning import get_opencloning_adapter
        from lab_copilot_gateway.wallac import get_wallac_adapter

        _default_executor = PlanExecutor(
            approval_store=get_approval_store(),
            read_adapter=get_elabftw_read_adapter(),
            write_adapter=get_elabftw_write_adapter(),
            opencloning_adapter=get_opencloning_adapter(),
            wallac_adapter=get_wallac_adapter(),
            bentolab_adapter=get_bentolab_adapter(),
        )
    return _default_executor


def reset_plan_executor(executor: PlanExecutor | None = None) -> None:
    """Test helper: replace or clear the singleton."""
    global _default_executor
    _default_executor = executor
