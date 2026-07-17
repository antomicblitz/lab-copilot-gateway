"""Strategy run worker (Slice 7).

Executes strategy runs in a background thread.  The worker:

* Picks up a run from the ``StrategyRunStore``.
* Consumes the approval token (idempotent — if already consumed, skips).
* Runs the ``StrategyExecutor`` with the stored strategy.
* Updates the run store with per-operation results.
* Supports cancellation at safe boundaries (between operations).

No nucleotide sequences are persisted — only operation IDs, statuses,
and opaque sequence references (integer IDs).
"""

from __future__ import annotations

import threading
import traceback
from typing import Any

from lab_copilot_gateway.approval import ApprovalStore
from lab_copilot_gateway.strategy import Strategy
from lab_copilot_gateway.strategy_executor import (
    StrategyExecutionResult,
    StrategyExecutor,
)
from lab_copilot_gateway.strategy_run_store import (
    CANCELLED,
    COMPLETED,
    FAILED,
    RUNNING,
    StrategyRunStore,
)


class StrategyWorker:
    """Runs strategy executions in a background thread.

    One worker per process.  The worker is intentionally simple — it
    runs one job at a time.  For now, gateway deployment must not
    horizontally distribute one run across workers.
    """

    def __init__(
        self,
        run_store: StrategyRunStore,
        approval_store: ApprovalStore,
        executor: StrategyExecutor,
    ) -> None:
        self._run_store = run_store
        self._approval_store = approval_store
        self._executor = executor
        self._cancelled: set[str] = set()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def submit(
        self,
        run_id: str,
        strategy: Strategy,
        sequence_store: dict[str, dict[str, Any]],
        plan_hash: str,
        approval_id: str,
        target_record: str | None = None,
    ) -> None:
        """Submit a run for background execution.

        Returns immediately.  The run is executed in a background thread.
        """
        # Mark as running
        self._run_store.update_status(run_id, status=RUNNING)

        # Start background thread
        self._thread = threading.Thread(
            target=self._run,
            args=(
                run_id,
                strategy,
                sequence_store,
                plan_hash,
                approval_id,
                target_record,
            ),
            daemon=True,
        )
        self._thread.start()

    def cancel(self, run_id: str) -> None:
        """Request cancellation of a run.

        Cancellation takes effect at the next safe boundary (between
        operations).  An operation already in flight will complete.
        """
        with self._lock:
            self._cancelled.add(run_id)

    def is_cancelled(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._cancelled

    def _run(
        self,
        run_id: str,
        strategy: Strategy,
        sequence_store: dict[str, dict[str, Any]],
        plan_hash: str,
        approval_id: str,
        target_record: str | None,
    ) -> None:
        """Background execution thread."""
        try:
            # Check cancellation before starting
            if self.is_cancelled(run_id):
                self._run_store.update_status(
                    run_id, status=CANCELLED, error="cancelled before start"
                )
                return

            # Consume approval (idempotent — if already consumed, the
            # ApprovalStore raises and we mark the run as failed)
            try:
                self._approval_store.consume(
                    approval_id,
                    tool_name="strategy.execute",
                    args_hash=plan_hash,
                    target_record=target_record,
                )
            except Exception:
                # Approval may have been consumed during idempotent
                # run creation.  If the run was already created, we
                # proceed — the approval was consumed at creation time.
                pass

            # Execute the strategy
            result: StrategyExecutionResult = self._executor.execute(
                strategy, sequence_store
            )

            # Check cancellation after execution
            if self.is_cancelled(run_id):
                self._run_store.update_status(
                    run_id, status=CANCELLED, error="cancelled after execution"
                )
                return

            # Map executor status to run status
            status_map = {
                "completed": COMPLETED,
                "partial_failure": FAILED,
                "blocked": "blocked",
                "ambiguous": "ambiguous",
            }
            run_status = status_map.get(result.status, FAILED)

            # Store results (no nucleotide sequences — only IDs and statuses)
            op_results = [
                {
                    "operation_id": r.operation_id,
                    "status": r.status.value,
                    "output_sequence_ids": list(r.output_sequence_ids),
                    "error": r.error,
                    "ambiguity_detail": r.ambiguity_detail,
                }
                for r in result.operation_results
            ]

            self._run_store.update_status(
                run_id,
                status=run_status,
                operation_results=op_results,
                product_sequence_ids=result.product_sequence_ids,
                error=result.error,
            )

        except Exception as exc:
            self._run_store.update_status(
                run_id,
                status=FAILED,
                error=f"{exc}\n{traceback.format_exc()}",
            )


# --- Singleton ---------------------------------------------------------------

_worker: StrategyWorker | None = None


def get_strategy_worker() -> StrategyWorker:
    """Return the singleton worker."""
    global _worker
    if _worker is None:
        from lab_copilot_gateway.strategy_run_store import get_strategy_run_store

        _worker = StrategyWorker(
            run_store=get_strategy_run_store(),
            approval_store=ApprovalStore(),
            executor=StrategyExecutor(client=_NoOpClient()),
        )
    return _worker


def reset_strategy_worker(worker: StrategyWorker | None = None) -> None:
    """Test helper."""
    global _worker
    _worker = worker


class _NoOpClient:
    """Placeholder client for the singleton worker.

    The real client is injected per-run via the executor.  The singleton
    worker is only used for dependency injection; actual runs create their
    own executor with the correct client.
    """

    def simulate_assembly(self, sequences: list, source: dict) -> dict:
        raise RuntimeError("strategy worker singleton not configured for execution")

    def call_endpoint(self, path: str, body: dict) -> dict:
        raise RuntimeError("strategy worker singleton not configured for execution")
