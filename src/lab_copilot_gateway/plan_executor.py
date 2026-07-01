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
from lab_copilot_gateway.opencloning import (
    OpenCloningAdapter,
    OpenCloningAdapterError,
    TOOL_ASSEMBLY,
    TOOL_MANUAL,
    TOOL_OLIGO,
    TOOL_PARSE,
    TOOL_WRITEBACK,
)
from lab_copilot_gateway.plan import (
    Plan,
    PlanStep,
    PlanValidationError,
    compute_plan_hash,
    validate_or_raise,
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
    ) -> None:
        self.approval_store = approval_store
        self.read_adapter = read_adapter
        self.write_adapter = write_adapter
        self.opencloning_adapter = opencloning_adapter

    def execute(
        self,
        plan: Plan,
        *,
        approval_id: str,
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

        # 2. Consume the plan-level approval (bound to plan hash).
        try:
            self.approval_store.consume(
                approval_id,
                tool_name="plan.execute",
                args_hash=plan_hash,
                target_record=f"plan:{plan.plan_id}",
            )
        except Exception as exc:
            return PlanExecutionResult(
                plan_id=plan.plan_id,
                plan_hash=plan_hash,
                status="rejected",
                step_results=[],
                summary=f"Approval consume failed: {exc}",
            )

        # 3. Execute steps.
        step_results: list[StepResult] = []
        all_steps = [(i, step, "read") for i, step in enumerate(plan.reads)]
        all_steps += [
            (len(plan.reads) + i, step, "write") for i, step in enumerate(plan.writes)
        ]

        write_failed = False
        for step_index, step, step_type in all_steps:
            if write_failed:
                # Remaining steps are skipped after a write failure.
                step_results.append(
                    StepResult(
                        step_index=step_index,
                        tool_name=step.tool_name,
                        status="skipped",
                        error="skipped due to prior write failure",
                    )
                )
                continue

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
                        plan_approval_id=approval_id,
                        mapped_identity=mapped_identity,
                        conversation_id=conversation_id,
                        request_id=request_id,
                        keycloak_subject=keycloak_subject,
                        librechat_user_id=librechat_user_id,
                        provider=provider,
                        model_id=model_id,
                    )
                step_results.append(
                    StepResult(
                        step_index=step_index,
                        tool_name=step.tool_name,
                        status="success",
                        result=result,
                        # C40: extract audit_action_id from write results
                        # (WriteResult.to_dict() includes it).
                        audit_action_id=(
                            result.get("audit_action_id")
                            if step_type == "write" and isinstance(result, dict)
                            else None
                        ),
                    )
                )
            except (ElabftwAdapterError, OpenCloningAdapterError) as exc:
                error_str = f"{exc.reason}: {exc.message}"
                step_results.append(
                    StepResult(
                        step_index=step_index,
                        tool_name=step.tool_name,
                        status="failed",
                        error=error_str,
                        # C40: transient failures are retryable with a
                        # fresh plan-level approval.
                        retryable=_is_retryable(exc.reason),
                    )
                )
                if step_type == "write":
                    write_failed = True
            except Exception as exc:
                error_str = f"unexpected error: {type(exc).__name__}: {exc}"
                step_results.append(
                    StepResult(
                        step_index=step_index,
                        tool_name=step.tool_name,
                        status="failed",
                        error=error_str,
                        # C40: unexpected errors are treated as transient
                        # (retryable) — the caller should investigate before
                        # retrying, but the step itself may succeed on retry.
                        retryable=True,
                    )
                )
                if step_type == "write":
                    write_failed = True

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
        opencloning_tools = {TOOL_PARSE, TOOL_MANUAL, TOOL_OLIGO, TOOL_ASSEMBLY}
        if step.tool_name in opencloning_tools:
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
            else:  # TOOL_ASSEMBLY
                result = self.opencloning_adapter.simulate_assembly(
                    context_token=context_token,
                    fragments=step.args.get("fragments", []),
                    assembly_config=step.args.get("assembly_config", {}),
                    mapped_identity=mapped_identity,
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
        from lab_copilot_gateway.opencloning import get_opencloning_adapter

        _default_executor = PlanExecutor(
            approval_store=get_approval_store(),
            read_adapter=get_elabftw_read_adapter(),
            write_adapter=get_elabftw_write_adapter(),
            opencloning_adapter=get_opencloning_adapter(),
        )
    return _default_executor


def reset_plan_executor(executor: PlanExecutor | None = None) -> None:
    """Test helper: replace or clear the singleton."""
    global _default_executor
    _default_executor = executor
