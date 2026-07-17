"""Tests for method-family expansions (Slices 10-12).

Scenarios 15-26: HR, CRISPR-HDR, Cre/Lox, Gateway, recombinase,
oligo hybridization, polymerase extension, reverse complement.

These verify that the executor correctly dispatches all cataloged
operations to the right endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


from lab_copilot_gateway.strategy import (
    BiologicalRole,
    Molecule,
    MoleculeStage,
    Operation,
    Strategy,
)
from lab_copilot_gateway.strategy_executor import StrategyExecutor


@dataclass
class StubClient:
    """Stub client that records calls and returns canned responses."""

    responses: dict[str, dict[str, Any]] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def simulate_assembly(self, sequences: list, source: dict) -> dict[str, Any]:
        self.calls.append(
            {
                "method": "simulate_assembly",
                "source_type": source.get("type"),
                "count": len(sequences),
            }
        )
        return self.responses.get(
            source.get("type", ""),
            {"sequences": [{"id": 1, "type": "TextFileSequence", "name": "product"}]},
        )

    def call_endpoint(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append({"method": "call_endpoint", "path": path})
        return self.responses.get(
            path,
            {"sequences": [{"id": 1, "type": "TextFileSequence", "name": "product"}]},
        )


def _make_strategy(operation_key: str, n_inputs: int = 2) -> Strategy:
    mols = []
    for i in range(n_inputs):
        mols.append(
            Molecule(
                id=f"mol_{i + 1:03d}",
                stage=MoleculeStage.SOURCE,
                role=BiologicalRole.VECTOR if i == 0 else BiologicalRole.INSERT,
                name=f"input_{i + 1}",
            )
        )
    mols.append(
        Molecule(
            id="mol_999",
            stage=MoleculeStage.PRODUCT,
            role=BiologicalRole.PRODUCT,
            name="product",
            producer_operation_id="op_001",
        )
    )
    op = Operation(
        id="op_001",
        operation_key=operation_key,
        input_molecule_ids=tuple(f"mol_{i + 1:03d}" for i in range(n_inputs)),
        output_molecule_ids=("mol_999",),
    )
    return Strategy(
        id="strategy_test", goal="test", molecules=tuple(mols), operations=(op,)
    )


def _make_store(n: int) -> dict:
    return {
        f"mol_{i + 1:03d}": {
            "id": i + 1,
            "type": "TextFileSequence",
            "name": f"seq_{i + 1}",
        }
        for i in range(n)
    }


# --- Slice 10: HR and CRISPR-HDR (scenarios 15-17) --------------------------


class TestHomologousRecombination:
    """Scenario 15: Homologous recombination."""

    def test_hr_succeeds(self) -> None:
        client = StubClient()
        executor = StrategyExecutor(client)
        result = executor.execute(
            _make_strategy("homologous_recombination", 2), _make_store(2)
        )
        assert result.status == "completed"
        assert client.calls[0]["source_type"] == "HomologousRecombinationSource"


class TestCRISPRHDR:
    """Scenario 16: CRISPR-HDR."""

    def test_crispr_succeeds(self) -> None:
        client = StubClient()
        executor = StrategyExecutor(client)
        result = executor.execute(_make_strategy("crispr_hdr", 2), _make_store(2))
        assert result.status == "completed"
        assert client.calls[0]["source_type"] == "CRISPRSource"


# --- Slice 11: Site-specific recombination (scenarios 18-23) ----------------


class TestCreLox:
    """Scenario 18: Cre/Lox recombination."""

    def test_cre_lox_succeeds(self) -> None:
        client = StubClient()
        executor = StrategyExecutor(client)
        result = executor.execute(_make_strategy("cre_lox", 2), _make_store(2))
        assert result.status == "completed"
        assert client.calls[0]["source_type"] == "CreLoxRecombinationSource"


class TestGateway:
    """Scenario 20: Gateway recombination."""

    def test_gateway_succeeds(self) -> None:
        client = StubClient()
        executor = StrategyExecutor(client)
        result = executor.execute(_make_strategy("gateway", 2), _make_store(2))
        assert result.status == "completed"
        assert client.calls[0]["source_type"] == "GatewaySource"


class TestRecombinase:
    """Scenario 23: Generic recombinase."""

    def test_recombinase_succeeds(self) -> None:
        client = StubClient()
        executor = StrategyExecutor(client)
        result = executor.execute(_make_strategy("recombinase", 2), _make_store(2))
        assert result.status == "completed"
        assert client.calls[0]["source_type"] == "RecombinaseSource"


# --- Slice 12: Small product transformations (scenarios 24-26) --------------


class TestOligoHybridization:
    """Scenario 24: Oligo hybridization."""

    def test_oligo_hybridization_succeeds(self) -> None:
        client = StubClient()
        executor = StrategyExecutor(client)
        result = executor.execute(
            _make_strategy("oligo_hybridization", 2), _make_store(2)
        )
        assert result.status == "completed"
        assert client.calls[0]["path"] == "/oligonucleotide_hybridization"


class TestPolymeraseExtension:
    """Scenario 25: Polymerase extension."""

    def test_polymerase_extension_succeeds(self) -> None:
        client = StubClient()
        executor = StrategyExecutor(client)
        result = executor.execute(
            _make_strategy("polymerase_extension", 1), _make_store(1)
        )
        assert result.status == "completed"
        assert client.calls[0]["path"] == "/polymerase_extension"


class TestReverseComplement:
    """Scenario 26: Reverse complement."""

    def test_reverse_complement_succeeds(self) -> None:
        client = StubClient()
        executor = StrategyExecutor(client)
        result = executor.execute(
            _make_strategy("reverse_complement", 1), _make_store(1)
        )
        assert result.status == "completed"
        assert client.calls[0]["path"] == "/reverse_complement"


# --- Slice 13: Batch execution (scenarios 27-28) ----------------------------


class TestBatchExecution:
    """Scenario 27: Independent batch of multiple assemblies."""

    def test_batch_of_5_succeeds(self) -> None:
        """5 independent Gibson assemblies should all succeed."""
        from lab_copilot_gateway.strategy import BatchMember

        mols = []
        ops = []
        # Common backbone
        mols.append(
            Molecule(
                id="mol_001",
                stage=MoleculeStage.SOURCE,
                role=BiologicalRole.VECTOR,
                name="pUC19",
            )
        )
        # 5 inserts + 5 products + 5 operations
        for i in range(5):
            insert_id = f"mol_{i + 2:03d}"
            product_id = f"mol_{i + 7:03d}"
            op_id = f"op_{i + 1:03d}"
            mols.append(
                Molecule(
                    id=insert_id,
                    stage=MoleculeStage.SOURCE,
                    role=BiologicalRole.INSERT,
                    name=f"Gene {chr(65 + i)}",
                )
            )
            mols.append(
                Molecule(
                    id=product_id,
                    stage=MoleculeStage.PRODUCT,
                    role=BiologicalRole.PRODUCT,
                    name=f"pUC19-Gene{chr(65 + i)}",
                    producer_operation_id=op_id,
                )
            )
            ops.append(
                Operation(
                    id=op_id,
                    operation_key="gibson_assembly",
                    input_molecule_ids=("mol_001", insert_id),
                    output_molecule_ids=(product_id,),
                )
            )

        batch_members = tuple(
            BatchMember(id=f"batch_{i + 1:03d}", label=f"Gene {chr(65 + i)} construct")
            for i in range(5)
        )

        strategy = Strategy(
            id="strategy_batch",
            goal="Batch assemble 5 constructs",
            molecules=tuple(mols),
            operations=tuple(ops),
            batch_members=batch_members,
        )

        store = {"mol_001": {"id": 1, "type": "TextFileSequence", "name": "pUC19"}}
        for i in range(5):
            store[f"mol_{i + 2:03d}"] = {
                "id": i + 2,
                "type": "TextFileSequence",
                "name": f"Gene {chr(65 + i)}",
            }

        client = StubClient()
        executor = StrategyExecutor(client)
        result = executor.execute(strategy, store)

        assert result.status == "completed"
        assert len(result.operation_results) == 5
        assert all(r.status.value == "succeeded" for r in result.operation_results)
        assert len(result.product_sequence_ids) == 5
