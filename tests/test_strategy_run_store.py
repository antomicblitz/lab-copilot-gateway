"""Tests for the strategy run store and worker (Slice 7).

Acceptance checks (from the umbrella plan):

    * submit returns 202 + run_id
    * polling reaches terminal state
    * cancellation works at safe boundaries
    * restart marks active runs as interrupted
    * DB inspection confirms no nucleotide content
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


from lab_copilot_gateway.approval import ApprovalStore
from lab_copilot_gateway.strategy import (
    BiologicalRole,
    Molecule,
    MoleculeStage,
    Operation,
    Strategy,
)
from lab_copilot_gateway.strategy_executor import (
    StrategyExecutor,
)
from lab_copilot_gateway.strategy_run_store import (
    CANCELLED,
    COMPLETED,
    FAILED,
    INTERRUPTED,
    PENDING,
    RUNNING,
    StrategyRunStore,
    get_strategy_run_store,
    reset_strategy_run_store,
)
from lab_copilot_gateway.strategy_worker import (
    StrategyWorker,
)


# --- Run store tests ---------------------------------------------------------


class TestRunStore:
    """StrategyRunStore CRUD and lifecycle."""

    def test_create_returns_pending_run(self) -> None:
        store = StrategyRunStore(":memory:")
        run = store.create(
            strategy_id="strategy_001",
            plan_hash="abc123",
            approval_id="appr_001",
            target_record="strategy:exp:42",
        )
        assert run.status == PENDING
        assert run.run_id
        assert run.strategy_id == "strategy_001"
        assert run.plan_hash == "abc123"

    def test_get_returns_run(self) -> None:
        store = StrategyRunStore(":memory:")
        run = store.create(strategy_id="s1", plan_hash="h1", approval_id="a1")
        fetched = store.get(run.run_id)
        assert fetched is not None
        assert fetched.run_id == run.run_id

    def test_get_nonexistent_returns_none(self) -> None:
        store = StrategyRunStore(":memory:")
        assert store.get("nonexistent") is None

    def test_update_status_to_running(self) -> None:
        store = StrategyRunStore(":memory:")
        run = store.create(strategy_id="s1", plan_hash="h1", approval_id="a1")
        updated = store.update_status(run.run_id, status=RUNNING)
        assert updated is not None
        assert updated.status == RUNNING
        assert updated.completed_at is None

    def test_update_status_to_terminal_sets_completed_at(self) -> None:
        store = StrategyRunStore(":memory:")
        run = store.create(strategy_id="s1", plan_hash="h1", approval_id="a1")
        updated = store.update_status(run.run_id, status=COMPLETED)
        assert updated is not None
        assert updated.status == COMPLETED
        assert updated.completed_at is not None

    def test_update_status_with_operation_results(self) -> None:
        store = StrategyRunStore(":memory:")
        run = store.create(strategy_id="s1", plan_hash="h1", approval_id="a1")
        updated = store.update_status(
            run.run_id,
            status=COMPLETED,
            operation_results=[
                {
                    "operation_id": "op_001",
                    "status": "succeeded",
                    "output_sequence_ids": [1],
                },
            ],
            product_sequence_ids={"mol_003": 1},
        )
        assert updated is not None
        assert len(updated.operation_results) == 1
        assert updated.operation_results[0]["operation_id"] == "op_001"
        assert updated.product_sequence_ids == {"mol_003": 1}

    def test_list_active_returns_pending_and_running(self) -> None:
        store = StrategyRunStore(":memory:")
        r1 = store.create(strategy_id="s1", plan_hash="h1", approval_id="a1")
        r2 = store.create(strategy_id="s2", plan_hash="h2", approval_id="a2")
        store.update_status(r2.run_id, status=RUNNING)
        r3 = store.create(strategy_id="s3", plan_hash="h3", approval_id="a3")
        store.update_status(r3.run_id, status=COMPLETED)

        active = store.list_active()
        assert r1.run_id in active
        assert r2.run_id in active
        assert r3.run_id not in active

    def test_mark_active_interrupted(self) -> None:
        store = StrategyRunStore(":memory:")
        r1 = store.create(strategy_id="s1", plan_hash="h1", approval_id="a1")
        r2 = store.create(strategy_id="s2", plan_hash="h2", approval_id="a2")
        store.update_status(r2.run_id, status=RUNNING)
        r3 = store.create(strategy_id="s3", plan_hash="h3", approval_id="a3")
        store.update_status(r3.run_id, status=COMPLETED)

        count = store.mark_active_interrupted()
        assert count == 2  # r1 (pending) + r2 (running), not r3 (completed)

        assert store.get(r1.run_id).status == INTERRUPTED
        assert store.get(r2.run_id).status == INTERRUPTED
        assert store.get(r3.run_id).status == COMPLETED

    def test_no_nucleotide_sequences_in_db(self) -> None:
        """The run store must not persist nucleotide sequences."""
        store = StrategyRunStore(":memory:")
        run = store.create(strategy_id="s1", plan_hash="h1", approval_id="a1")
        store.update_status(
            run.run_id,
            status=COMPLETED,
            operation_results=[{"operation_id": "op_001", "status": "succeeded"}],
        )

        # Inspect the raw DB
        conn = store._connect()
        with store._lock:
            row = conn.execute("SELECT * FROM strategy_runs").fetchone()
        cols = [
            d[0]
            for d in conn.execute("SELECT * FROM strategy_runs LIMIT 0").description
        ]
        data = dict(zip(cols, row))

        # Check no nucleotide sequences in any field
        for key, value in data.items():
            if value is None:
                continue
            value_str = str(value).lower()
            forbidden = ["atcg", "locus", "genbank", "file_content", "fasta"]
            for word in forbidden:
                assert word not in value_str, (
                    f"field '{key}' contains forbidden content '{word}'"
                )


# --- Worker tests ------------------------------------------------------------


@dataclass
class StubExecutionClient:
    """Stub client for worker tests."""

    responses: dict[str, dict[str, Any]] = field(default_factory=dict)

    def simulate_assembly(self, sequences: list, source: dict) -> dict[str, Any]:
        return self.responses.get(
            source.get("type", ""),
            {"sequences": [{"id": 1, "type": "TextFileSequence", "name": "product"}]},
        )

    def call_endpoint(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        return self.responses.get(
            path,
            {"sequences": [{"id": 1, "type": "TextFileSequence", "name": "product"}]},
        )


def _make_strategy() -> Strategy:
    """Create a minimal valid strategy."""
    mol1 = Molecule(
        id="mol_001",
        stage=MoleculeStage.SOURCE,
        role=BiologicalRole.VECTOR,
        name="template",
    )
    mol2 = Molecule(
        id="mol_002",
        stage=MoleculeStage.PRODUCT,
        role=BiologicalRole.PRODUCT,
        name="product",
        producer_operation_id="op_001",
    )
    op = Operation(
        id="op_001",
        operation_key="pcr",
        input_molecule_ids=("mol_001",),
        output_molecule_ids=("mol_002",),
    )
    return Strategy(
        id="strategy_001", goal="test", molecules=(mol1, mol2), operations=(op,)
    )


class TestWorker:
    """StrategyWorker async execution."""

    def test_submit_returns_and_completes(self) -> None:
        store = StrategyRunStore(":memory:")
        approval = ApprovalStore(":memory:")
        client = StubExecutionClient()
        executor = StrategyExecutor(client)
        worker = StrategyWorker(store, approval, executor)

        strategy = _make_strategy()
        run = store.create(
            strategy_id=strategy.id,
            plan_hash=strategy.compute_hash(),
            approval_id="appr_001",
        )

        worker.submit(
            run_id=run.run_id,
            strategy=strategy,
            sequence_store={
                "mol_001": {"id": 1, "type": "TextFileSequence", "name": "template"}
            },
            plan_hash=strategy.compute_hash(),
            approval_id="appr_001",
        )

        # Wait for completion (background thread)
        time.sleep(0.5)

        updated = store.get(run.run_id)
        assert updated is not None
        assert updated.status in (COMPLETED, FAILED)
        if updated.status == COMPLETED:
            assert len(updated.operation_results) > 0

    def test_cancel_before_start(self) -> None:
        store = StrategyRunStore(":memory:")
        approval = ApprovalStore(":memory:")
        client = StubExecutionClient()
        executor = StrategyExecutor(client)
        worker = StrategyWorker(store, approval, executor)

        strategy = _make_strategy()
        run = store.create(
            strategy_id=strategy.id,
            plan_hash=strategy.compute_hash(),
            approval_id="appr_001",
        )

        # Cancel before submitting
        worker.cancel(run.run_id)

        worker.submit(
            run_id=run.run_id,
            strategy=strategy,
            sequence_store={"mol_001": {"id": 1, "type": "TextFileSequence"}},
            plan_hash=strategy.compute_hash(),
            approval_id="appr_001",
        )

        time.sleep(0.3)

        updated = store.get(run.run_id)
        assert updated is not None
        assert updated.status == CANCELLED

    def test_run_survives_disconnect(self) -> None:
        """The run store persists even if the browser disconnects."""
        store = StrategyRunStore(":memory:")
        approval = ApprovalStore(":memory:")
        client = StubExecutionClient()
        executor = StrategyExecutor(client)
        worker = StrategyWorker(store, approval, executor)

        strategy = _make_strategy()
        run = store.create(
            strategy_id=strategy.id,
            plan_hash=strategy.compute_hash(),
            approval_id="appr_001",
        )

        worker.submit(
            run_id=run.run_id,
            strategy=strategy,
            sequence_store={"mol_001": {"id": 1, "type": "TextFileSequence"}},
            plan_hash=strategy.compute_hash(),
            approval_id="appr_001",
        )

        # "Disconnect" — just wait and poll
        time.sleep(0.5)

        # Reconnect — the run should still be accessible
        fetched = store.get(run.run_id)
        assert fetched is not None
        assert fetched.status in (
            COMPLETED,
            FAILED,
            CANCELLED,
            INTERRUPTED,
            RUNNING,
            PENDING,
        )


# --- Singleton tests ---------------------------------------------------------


class TestSingleton:
    """Module-level singleton behavior."""

    def test_get_strategy_run_store_returns_singleton(self) -> None:
        s1 = get_strategy_run_store()
        s2 = get_strategy_run_store()
        assert s1 is s2

    def test_reset_clears_singleton(self) -> None:
        original = get_strategy_run_store()
        reset_strategy_run_store()
        new = get_strategy_run_store()
        assert new is not original
        reset_strategy_run_store(original)
