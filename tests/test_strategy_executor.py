"""Tests for the deterministic strategy executor (Slice 6).

Acceptance checks (from the umbrella plan):

    Scenarios 1–14: PCR, digest, ligation, restriction cloning, Golden Gate,
    Gibson, In-Fusion, overlap-extension PCR, in-vivo assembly.

    * No LLM call participates in execution.
    * Operation IDs preserved in progress/results.
    * Zero/multiple products reported as blocked ambiguity.
    * Bounded transient retries without changing semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


from lab_copilot_gateway.strategy import (
    BiologicalRole,
    Molecule,
    MoleculeStage,
    Operation,
    OperationStatus,
    Strategy,
)
from lab_copilot_gateway.strategy_executor import (
    StrategyExecutor,
)


# --- Stub client ------------------------------------------------------------


@dataclass
class StubExecutionClient:
    """In-memory client for executor tests.

    Pre-seed ``responses`` keyed by endpoint or source type.
    Set ``error`` to simulate a transient failure (retried).
    Set ``permanent_error`` to simulate a non-retriable failure.
    """

    responses: dict[str, dict[str, Any]] = field(default_factory=dict)
    error: Exception | None = None
    permanent_error: Exception | None = None
    call_count: int = 0
    calls: list[dict[str, Any]] = field(default_factory=list)

    def simulate_assembly(
        self,
        sequences: list[dict[str, Any]],
        source: dict[str, Any],
    ) -> dict[str, Any]:
        self.call_count += 1
        self.calls.append(
            {
                "method": "simulate_assembly",
                "source_type": source.get("type"),
                "seq_count": len(sequences),
            }
        )
        if self.permanent_error:
            raise self.permanent_error
        if self.error and self.call_count < 3:
            raise self.error
        key = source.get("type", "")
        return self.responses.get(
            key,
            {
                "sequences": [{"id": 1, "type": "TextFileSequence", "name": "product"}],
                "sources": [],
            },
        )

    def call_endpoint(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        self.call_count += 1
        self.calls.append(
            {"method": "call_endpoint", "path": path, "body_keys": list(body.keys())}
        )
        if self.permanent_error:
            raise self.permanent_error
        if self.error and self.call_count < 3:
            raise self.error
        return self.responses.get(
            path,
            {
                "sequences": [{"id": 1, "type": "TextFileSequence", "name": "product"}],
                "sources": [],
            },
        )


# --- Helpers -----------------------------------------------------------------


def _make_source_molecule(mid: str, name: str = "source") -> Molecule:
    return Molecule(
        id=mid,
        stage=MoleculeStage.SOURCE,
        role=BiologicalRole.VECTOR,
        name=name,
    )


def _make_product_molecule(mid: str, producer: str, name: str = "product") -> Molecule:
    return Molecule(
        id=mid,
        stage=MoleculeStage.PRODUCT,
        role=BiologicalRole.PRODUCT,
        name=name,
        producer_operation_id=producer,
    )


def _make_intermediate_molecule(
    mid: str, producer: str, name: str = "intermediate"
) -> Molecule:
    return Molecule(
        id=mid,
        stage=MoleculeStage.INTERMEDIATE,
        role=BiologicalRole.FRAGMENT,
        name=name,
        producer_operation_id=producer,
    )


def _make_sequence(seq_id: int, name: str = "seq") -> dict[str, Any]:
    return {
        "id": seq_id,
        "type": "TextFileSequence",
        "name": name,
        "file_content": "LOCUS test",
    }


# --- PCR tests (scenario 1) --------------------------------------------------


class TestPCR:
    """Scenario 1: PCR amplification."""

    def test_pcr_succeeds(self) -> None:
        client = StubExecutionClient(
            responses={"/pcr": {"sequences": [_make_sequence(1, "amplicon")]}}
        )
        mol_template = _make_source_molecule("mol_001", "template")
        mol_product = _make_product_molecule("mol_002", "op_001", "amplicon")
        op = Operation(
            id="op_001",
            operation_key="pcr",
            input_molecule_ids=("mol_001",),
            output_molecule_ids=("mol_002",),
        )
        strategy = Strategy(
            id="strategy_001",
            goal="PCR test",
            molecules=(mol_template, mol_product),
            operations=(op,),
        )
        store = {"mol_001": _make_sequence(1, "template")}
        executor = StrategyExecutor(client)
        result = executor.execute(strategy, store)

        assert result.status == "completed"
        assert len(result.operation_results) == 1
        assert result.operation_results[0].status == OperationStatus.SUCCEEDED
        assert result.operation_results[0].operation_id == "op_001"
        assert "mol_002" in result.product_sequence_ids


# --- Restriction digest tests (scenario 2) -----------------------------------


class TestRestrictionDigest:
    """Scenario 2: Restriction enzyme digest."""

    def test_restriction_digest_succeeds(self) -> None:
        client = StubExecutionClient(
            responses={"/restriction": {"sequences": [_make_sequence(1, "fragment")]}}
        )
        mol_source = _make_source_molecule("mol_001", "vector")
        mol_product = _make_product_molecule("mol_002", "op_001", "fragment")
        op = Operation(
            id="op_001",
            operation_key="restriction_digest",
            input_molecule_ids=("mol_001",),
            output_molecule_ids=("mol_002",),
            parameters={"enzymes": ["EcoRI"]},
        )
        strategy = Strategy(
            id="strategy_001",
            goal="digest test",
            molecules=(mol_source, mol_product),
            operations=(op,),
        )
        store = {"mol_001": _make_sequence(1, "vector")}
        executor = StrategyExecutor(client)
        result = executor.execute(strategy, store)

        assert result.status == "completed"
        assert result.operation_results[0].status == OperationStatus.SUCCEEDED


# --- Gibson assembly tests (scenario 5) --------------------------------------


class TestGibsonAssembly:
    """Scenario 5: Gibson assembly."""

    def test_gibson_succeeds(self) -> None:
        client = StubExecutionClient(
            responses={
                "GibsonAssemblySource": {"sequences": [_make_sequence(1, "assembly")]}
            }
        )
        mol1 = _make_source_molecule("mol_001", "fragment 1")
        mol2 = _make_source_molecule("mol_002", "fragment 2")
        mol_product = _make_product_molecule("mol_003", "op_001", "assembly")
        op = Operation(
            id="op_001",
            operation_key="gibson_assembly",
            input_molecule_ids=("mol_001", "mol_002"),
            output_molecule_ids=("mol_003",),
        )
        strategy = Strategy(
            id="strategy_001",
            goal="Gibson test",
            molecules=(mol1, mol2, mol_product),
            operations=(op,),
        )
        store = {
            "mol_001": _make_sequence(1, "fragment 1"),
            "mol_002": _make_sequence(2, "fragment 2"),
        }
        executor = StrategyExecutor(client)
        result = executor.execute(strategy, store)

        assert result.status == "completed"
        assert result.operation_results[0].status == OperationStatus.SUCCEEDED
        assert "mol_003" in result.product_sequence_ids


# --- Ligation tests (scenario 3) ---------------------------------------------


class TestLigation:
    """Scenario 3: Conventional ligation."""

    def test_ligation_succeeds(self) -> None:
        client = StubExecutionClient(
            responses={"LigationSource": {"sequences": [_make_sequence(1, "ligated")]}}
        )
        mol1 = _make_source_molecule("mol_001", "backbone")
        mol2 = _make_source_molecule("mol_002", "insert")
        mol_product = _make_product_molecule("mol_003", "op_001", "ligated")
        op = Operation(
            id="op_001",
            operation_key="ligation",
            input_molecule_ids=("mol_001", "mol_002"),
            output_molecule_ids=("mol_003",),
        )
        strategy = Strategy(
            id="strategy_001",
            goal="ligation test",
            molecules=(mol1, mol2, mol_product),
            operations=(op,),
        )
        store = {"mol_001": _make_sequence(1), "mol_002": _make_sequence(2)}
        executor = StrategyExecutor(client)
        result = executor.execute(strategy, store)

        assert result.status == "completed"


# --- Golden Gate tests (scenario 4) ------------------------------------------


class TestGoldenGate:
    """Scenario 4: Golden Gate assembly."""

    def test_golden_gate_succeeds(self) -> None:
        client = StubExecutionClient(
            responses={
                "RestrictionAndLigationSource": {
                    "sequences": [_make_sequence(1, "gg_product")]
                }
            }
        )
        mol1 = _make_source_molecule("mol_001", "part 1")
        mol2 = _make_source_molecule("mol_002", "part 2")
        mol_product = _make_product_molecule("mol_003", "op_001", "gg_product")
        op = Operation(
            id="op_001",
            operation_key="golden_gate",
            input_molecule_ids=("mol_001", "mol_002"),
            output_molecule_ids=("mol_003",),
            parameters={"enzymes": ["BsaI"]},
        )
        strategy = Strategy(
            id="strategy_001",
            goal="Golden Gate test",
            molecules=(mol1, mol2, mol_product),
            operations=(op,),
        )
        store = {"mol_001": _make_sequence(1), "mol_002": _make_sequence(2)}
        executor = StrategyExecutor(client)
        result = executor.execute(strategy, store)

        assert result.status == "completed"


# --- Multi-step strategy tests (scenario 7) -----------------------------------


class TestMultiStepStrategy:
    """Scenario 7: PCR → digest → ligation chain."""

    def test_pcr_digest_ligation_chain(self) -> None:
        client = StubExecutionClient(
            responses={
                "/pcr": {"sequences": [_make_sequence(1, "amplicon")]},
                "/restriction": {"sequences": [_make_sequence(1, "digested")]},
                "LigationSource": {"sequences": [_make_sequence(1, "final")]},
            }
        )
        mol_template = _make_source_molecule("mol_001", "template")
        mol_vector = _make_source_molecule("mol_002", "vector")
        mol_amplicon = _make_intermediate_molecule("mol_003", "op_001", "amplicon")
        mol_digested = _make_intermediate_molecule(
            "mol_004", "op_002", "digested vector"
        )
        mol_product = _make_product_molecule("mol_005", "op_003", "final product")

        op_pcr = Operation(
            id="op_001",
            operation_key="pcr",
            input_molecule_ids=("mol_001",),
            output_molecule_ids=("mol_003",),
        )
        op_digest = Operation(
            id="op_002",
            operation_key="restriction_digest",
            input_molecule_ids=("mol_002",),
            output_molecule_ids=("mol_004",),
        )
        op_ligation = Operation(
            id="op_003",
            operation_key="ligation",
            input_molecule_ids=("mol_003", "mol_004"),
            output_molecule_ids=("mol_005",),
        )
        strategy = Strategy(
            id="strategy_001",
            goal="PCR → digest → ligation",
            molecules=(
                mol_template,
                mol_vector,
                mol_amplicon,
                mol_digested,
                mol_product,
            ),
            operations=(op_pcr, op_digest, op_ligation),
        )
        store = {
            "mol_001": _make_sequence(1, "template"),
            "mol_002": _make_sequence(2, "vector"),
        }
        executor = StrategyExecutor(client)
        result = executor.execute(strategy, store)

        assert result.status == "completed"
        assert len(result.operation_results) == 3
        assert all(
            r.status == OperationStatus.SUCCEEDED for r in result.operation_results
        )
        assert "mol_005" in result.product_sequence_ids


# --- Ambiguity tests --------------------------------------------------------


class TestAmbiguity:
    """Zero or multiple products are reported as ambiguous."""

    def test_zero_products_is_ambiguous(self) -> None:
        client = StubExecutionClient(responses={"/pcr": {"sequences": []}})
        mol_template = _make_source_molecule("mol_001", "template")
        mol_product = _make_product_molecule("mol_002", "op_001", "amplicon")
        op = Operation(
            id="op_001",
            operation_key="pcr",
            input_molecule_ids=("mol_001",),
            output_molecule_ids=("mol_002",),
        )
        strategy = Strategy(
            id="strategy_001",
            goal="zero products",
            molecules=(mol_template, mol_product),
            operations=(op,),
        )
        store = {"mol_001": _make_sequence(1)}
        executor = StrategyExecutor(client)
        result = executor.execute(strategy, store)

        assert result.status == "ambiguous"
        assert result.operation_results[0].status == OperationStatus.AMBIGUOUS
        assert "zero products" in (result.operation_results[0].ambiguity_detail or "")

    def test_multiple_products_is_ambiguous(self) -> None:
        client = StubExecutionClient(
            responses={
                "/pcr": {
                    "sequences": [_make_sequence(1, "p1"), _make_sequence(2, "p2")]
                }
            }
        )
        mol_template = _make_source_molecule("mol_001", "template")
        mol_product = _make_product_molecule("mol_002", "op_001", "amplicon")
        op = Operation(
            id="op_001",
            operation_key="pcr",
            input_molecule_ids=("mol_001",),
            output_molecule_ids=("mol_002",),
        )
        strategy = Strategy(
            id="strategy_001",
            goal="multiple products",
            molecules=(mol_template, mol_product),
            operations=(op,),
        )
        store = {"mol_001": _make_sequence(1)}
        executor = StrategyExecutor(client)
        result = executor.execute(strategy, store)

        assert result.status == "ambiguous"
        assert "2 products" in (result.operation_results[0].ambiguity_detail or "")


# --- Failure and blocking tests ---------------------------------------------


class TestFailures:
    """Operation failures block downstream operations."""

    def test_failed_operation_blocks_downstream(self) -> None:
        client = StubExecutionClient(
            permanent_error=RuntimeError("backend down"),
        )
        mol_template = _make_source_molecule("mol_001", "template")
        mol_intermediate = _make_intermediate_molecule(
            "mol_002", "op_001", "intermediate"
        )
        mol_product = _make_product_molecule("mol_003", "op_002", "product")
        op1 = Operation(
            id="op_001",
            operation_key="pcr",
            input_molecule_ids=("mol_001",),
            output_molecule_ids=("mol_002",),
        )
        op2 = Operation(
            id="op_002",
            operation_key="restriction_digest",
            input_molecule_ids=("mol_002",),
            output_molecule_ids=("mol_003",),
        )
        strategy = Strategy(
            id="strategy_001",
            goal="failure chain",
            molecules=(mol_template, mol_intermediate, mol_product),
            operations=(op1, op2),
        )
        store = {"mol_001": _make_sequence(1)}
        executor = StrategyExecutor(client)
        result = executor.execute(strategy, store)

        assert result.status == "partial_failure"
        assert result.operation_results[0].status == OperationStatus.FAILED
        assert result.operation_results[1].status == OperationStatus.BLOCKED

    def test_unknown_operation_key_blocked(self) -> None:
        client = StubExecutionClient()
        mol_source = _make_source_molecule("mol_001")
        mol_product = _make_product_molecule("mol_002", "op_001")
        op = Operation(
            id="op_001",
            operation_key="totally_bogus",
            input_molecule_ids=("mol_001",),
        )
        strategy = Strategy(
            id="strategy_001",
            goal="unknown op",
            molecules=(mol_source, mol_product),
            operations=(op,),
        )
        store = {"mol_001": _make_sequence(1)}
        executor = StrategyExecutor(client)
        result = executor.execute(strategy, store)

        assert result.operation_results[0].status == OperationStatus.BLOCKED


# --- Retry tests -------------------------------------------------------------


class TestRetries:
    """Transient failures are retried."""

    def test_transient_failure_retried(self) -> None:
        client = StubExecutionClient(
            error=ConnectionError("transient"),
            responses={"/pcr": {"sequences": [_make_sequence(1, "amplicon")]}},
        )
        mol_template = _make_source_molecule("mol_001", "template")
        mol_product = _make_product_molecule("mol_002", "op_001", "amplicon")
        op = Operation(
            id="op_001",
            operation_key="pcr",
            input_molecule_ids=("mol_001",),
            output_molecule_ids=("mol_002",),
        )
        strategy = Strategy(
            id="strategy_001",
            goal="retry test",
            molecules=(mol_template, mol_product),
            operations=(op,),
        )
        store = {"mol_001": _make_sequence(1)}
        executor = StrategyExecutor(client)
        result = executor.execute(strategy, store)

        assert result.status == "completed"
        assert client.call_count >= 3  # retried at least twice before success


# --- Topological ordering tests ----------------------------------------------


class TestTopologicalOrder:
    """Operations are executed in dependency order."""

    def test_dependencies_respected(self) -> None:
        client = StubExecutionClient(
            responses={
                "/pcr": {"sequences": [_make_sequence(1, "amplicon")]},
                "/restriction": {"sequences": [_make_sequence(1, "digested")]},
            }
        )
        mol_template = _make_source_molecule("mol_001", "template")
        mol_amplicon = _make_intermediate_molecule("mol_002", "op_001", "amplicon")
        mol_product = _make_product_molecule("mol_003", "op_002", "digested")
        op_pcr = Operation(
            id="op_001",
            operation_key="pcr",
            input_molecule_ids=("mol_001",),
            output_molecule_ids=("mol_002",),
        )
        op_digest = Operation(
            id="op_002",
            operation_key="restriction_digest",
            input_molecule_ids=("mol_002",),
            output_molecule_ids=("mol_003",),
        )
        # Put digest first in the list — executor should still run PCR first
        strategy = Strategy(
            id="strategy_001",
            goal="ordering test",
            molecules=(mol_template, mol_amplicon, mol_product),
            operations=(op_digest, op_pcr),  # reversed order
        )
        store = {"mol_001": _make_sequence(1)}
        executor = StrategyExecutor(client)
        result = executor.execute(strategy, store)

        assert result.status == "completed"
        # PCR should have been called first despite being second in the list
        assert client.calls[0]["method"] == "call_endpoint"
        assert client.calls[0]["path"] == "/pcr"


# --- No LLM participation ----------------------------------------------------


class TestNoLLM:
    """The executor must not call any LLM."""

    def test_executor_has_no_llm_imports(self) -> None:
        """The strategy_executor module must not import any LLM-related modules."""
        import lab_copilot_gateway.strategy_executor as mod

        source = mod.__dict__
        # Check no LLM-related modules are imported
        forbidden_imports = ["openai", "anthropic", "deepseek"]
        for name in source:
            for forbidden in forbidden_imports:
                assert forbidden not in name.lower(), (
                    f"strategy_executor imports '{name}' — executor must be LLM-free"
                )

    def test_executor_has_no_llm_calls(self) -> None:
        """The executor source must not contain LLM call patterns."""
        import inspect
        import lab_copilot_gateway.strategy_executor as mod

        source = inspect.getsource(mod)
        # Check for actual LLM call patterns, not substrings in docstrings
        forbidden_calls = ["call_llm", "call_openai", "call_anthropic", "agentic_loop"]
        for pattern in forbidden_calls:
            assert pattern not in source, (
                f"strategy_executor.py contains '{pattern}' — executor must be LLM-free"
            )
