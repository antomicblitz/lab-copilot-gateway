"""Tests for canonical strategy schema v3 (Slice 2).

Acceptance checks (from the umbrella plan):

1. Deterministic round-trip/hash tests.
2. All structural error cases (scenarios 32–37): duplicate IDs, dangling
   references, cycles, unknown operation keys, batch limit, payload size.
3. Warnings do not invalidate the graph.
4. No nucleotide sequences in the schema.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from lab_copilot_gateway.strategy import (
    MAX_BATCH_MEMBERS,
    MAX_LABEL_LENGTH,
    MAX_STRATEGY_PAYLOAD_BYTES,
    Annotation,
    BatchMember,
    BiologicalRole,
    Molecule,
    MoleculeStage,
    Operation,
    ProductComposition,
    Strategy,
)
from lab_copilot_gateway.strategy_validation import (
    has_blockers,
    is_approved,
    validate_strategy,
)


# --- Helpers -----------------------------------------------------------------


def _make_minimal_strategy(**overrides: object) -> Strategy:
    """Create a minimal valid strategy for testing."""
    mol1 = Molecule(
        id="mol_001",
        stage=MoleculeStage.SOURCE,
        role=BiologicalRole.VECTOR,
        name="pUC19",
    )
    mol2 = Molecule(
        id="mol_002",
        stage=MoleculeStage.SOURCE,
        role=BiologicalRole.INSERT,
        name="GFP insert",
    )
    mol3 = Molecule(
        id="mol_003",
        stage=MoleculeStage.PRODUCT,
        role=BiologicalRole.PRODUCT,
        name="pUC19-GFP",
        producer_operation_id="op_001",
    )
    op = Operation(
        id="op_001",
        operation_key="gibson_assembly",
        input_molecule_ids=("mol_001", "mol_002"),
        output_molecule_ids=("mol_003",),
    )
    defaults: dict[str, object] = {
        "id": "strategy_001",
        "goal": "Assemble pUC19-GFP by Gibson assembly",
        "molecules": (mol1, mol2, mol3),
        "operations": (op,),
    }
    defaults.update(overrides)
    return Strategy(**defaults)  # type: ignore[arg-type]


# --- Schema structure tests --------------------------------------------------


class TestSchemaStructure:
    """Basic schema dataclass behavior."""

    def test_minimal_strategy_valid(self) -> None:
        s = _make_minimal_strategy()
        issues = validate_strategy(s)
        assert not has_blockers(issues), [i.message for i in issues]

    def test_molecule_id_must_start_with_mol(self) -> None:
        with pytest.raises(ValueError, match="must start with 'mol_'"):
            Molecule(
                id="bad_id",
                stage=MoleculeStage.SOURCE,
                role=BiologicalRole.VECTOR,
            )

    def test_operation_id_must_start_with_op(self) -> None:
        with pytest.raises(ValueError, match="must start with 'op_'"):
            Operation(
                id="bad_id",
                operation_key="pcr",
                input_molecule_ids=("mol_001",),
            )

    def test_operation_requires_input(self) -> None:
        with pytest.raises(ValueError, match="at least one input"):
            Operation(
                id="op_001",
                operation_key="pcr",
                input_molecule_ids=(),
            )

    def test_strategy_requires_molecules(self) -> None:
        with pytest.raises(ValueError, match="at least one molecule"):
            Strategy(
                id="strategy_001",
                goal="test",
                molecules=(),
                operations=(
                    Operation(
                        id="op_001",
                        operation_key="pcr",
                        input_molecule_ids=("mol_001",),
                    ),
                ),
            )

    def test_strategy_requires_operations(self) -> None:
        with pytest.raises(ValueError, match="at least one operation"):
            Strategy(
                id="strategy_001",
                goal="test",
                molecules=(
                    Molecule(
                        id="mol_001",
                        stage=MoleculeStage.SOURCE,
                        role=BiologicalRole.VECTOR,
                    ),
                ),
                operations=(),
            )


# --- Canonical JSON and hash tests -------------------------------------------


class TestCanonicalJson:
    """Deterministic serialization and hashing."""

    def test_canonical_json_is_deterministic(self) -> None:
        s = _make_minimal_strategy()
        json1 = s.to_canonical_json()
        json2 = s.to_canonical_json()
        assert json1 == json2

    def test_hash_is_deterministic(self) -> None:
        s = _make_minimal_strategy()
        assert s.compute_hash() == s.compute_hash()

    def test_hash_changes_with_content(self) -> None:
        s1 = _make_minimal_strategy()
        s2 = _make_minimal_strategy(goal="Different goal")
        assert s1.compute_hash() != s2.compute_hash()

    def test_hash_is_sha256(self) -> None:
        s = _make_minimal_strategy()
        expected = hashlib.sha256(s.to_canonical_json().encode("utf-8")).hexdigest()
        assert s.compute_hash() == expected

    def test_canonical_json_sorted_keys(self) -> None:
        """JSON keys must be sorted for deterministic hashing."""
        s = _make_minimal_strategy()
        data = json.loads(s.to_canonical_json())
        # Top-level keys should be sorted
        assert list(data.keys()) == sorted(data.keys())

    def test_round_trip_preserves_structure(self) -> None:
        """Canonical JSON can be parsed back into the same hash."""
        s = _make_minimal_strategy()
        json_str = s.to_canonical_json()
        # Parse and re-serialize
        parsed = json.loads(json_str)
        re_json = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        assert json_str == re_json


# --- Validation: duplicate IDs -----------------------------------------------


class TestDuplicateIds:
    """Scenario 32: duplicate IDs are blockers."""

    def test_duplicate_molecule_id_blocked(self) -> None:
        mol1 = Molecule(
            id="mol_001",
            stage=MoleculeStage.SOURCE,
            role=BiologicalRole.VECTOR,
        )
        mol2 = Molecule(
            id="mol_001",  # duplicate!
            stage=MoleculeStage.SOURCE,
            role=BiologicalRole.INSERT,
        )
        op = Operation(
            id="op_001",
            operation_key="gibson_assembly",
            input_molecule_ids=("mol_001",),
        )
        s = Strategy(
            id="strategy_001",
            goal="test",
            molecules=(mol1, mol2),
            operations=(op,),
        )
        issues = validate_strategy(s)
        blockers = [i for i in issues if i.code == "duplicate_molecule_id"]
        assert len(blockers) == 1
        assert not is_approved(issues)

    def test_duplicate_operation_id_blocked(self) -> None:
        mol1 = Molecule(
            id="mol_001",
            stage=MoleculeStage.SOURCE,
            role=BiologicalRole.VECTOR,
        )
        op1 = Operation(
            id="op_001",
            operation_key="pcr",
            input_molecule_ids=("mol_001",),
        )
        op2 = Operation(
            id="op_001",  # duplicate!
            operation_key="restriction_digest",
            input_molecule_ids=("mol_001",),
        )
        s = Strategy(
            id="strategy_001",
            goal="test",
            molecules=(mol1,),
            operations=(op1, op2),
        )
        issues = validate_strategy(s)
        blockers = [i for i in issues if i.code == "duplicate_operation_id"]
        assert len(blockers) == 1
        assert not is_approved(issues)


# --- Validation: dangling references -----------------------------------------


class TestDanglingReferences:
    """Scenario 33: references to non-existent nodes."""

    def test_dangling_input_reference_blocked(self) -> None:
        mol1 = Molecule(
            id="mol_001",
            stage=MoleculeStage.SOURCE,
            role=BiologicalRole.VECTOR,
        )
        op = Operation(
            id="op_001",
            operation_key="pcr",
            input_molecule_ids=("mol_001", "mol_999"),  # mol_999 doesn't exist
        )
        s = Strategy(
            id="strategy_001",
            goal="test",
            molecules=(mol1,),
            operations=(op,),
        )
        issues = validate_strategy(s)
        blockers = [i for i in issues if i.code == "dangling_input_reference"]
        assert len(blockers) == 1
        assert not is_approved(issues)

    def test_dangling_output_reference_blocked(self) -> None:
        mol1 = Molecule(
            id="mol_001",
            stage=MoleculeStage.SOURCE,
            role=BiologicalRole.VECTOR,
        )
        op = Operation(
            id="op_001",
            operation_key="pcr",
            input_molecule_ids=("mol_001",),
            output_molecule_ids=("mol_999",),  # doesn't exist
        )
        s = Strategy(
            id="strategy_001",
            goal="test",
            molecules=(mol1,),
            operations=(op,),
        )
        issues = validate_strategy(s)
        blockers = [i for i in issues if i.code == "dangling_output_reference"]
        assert len(blockers) == 1
        assert not is_approved(issues)

    def test_dangling_producer_reference_blocked(self) -> None:
        mol1 = Molecule(
            id="mol_001",
            stage=MoleculeStage.SOURCE,
            role=BiologicalRole.VECTOR,
        )
        mol2 = Molecule(
            id="mol_002",
            stage=MoleculeStage.PRODUCT,
            role=BiologicalRole.PRODUCT,
            producer_operation_id="op_999",  # doesn't exist
        )
        op = Operation(
            id="op_001",
            operation_key="pcr",
            input_molecule_ids=("mol_001",),
            output_molecule_ids=("mol_002",),
        )
        s = Strategy(
            id="strategy_001",
            goal="test",
            molecules=(mol1, mol2),
            operations=(op,),
        )
        issues = validate_strategy(s)
        blockers = [i for i in issues if i.code == "dangling_producer_reference"]
        assert len(blockers) == 1
        assert not is_approved(issues)


# --- Validation: cycles ------------------------------------------------------


class TestCycleDetection:
    """Scenario 34: cycles in the operation dependency graph."""

    def test_simple_cycle_blocked(self) -> None:
        """op_001 produces mol_002, op_002 consumes mol_002 and produces
        mol_001, op_001 consumes mol_001 → cycle."""
        mol1 = Molecule(
            id="mol_001",
            stage=MoleculeStage.INTERMEDIATE,
            role=BiologicalRole.FRAGMENT,
            producer_operation_id="op_002",
        )
        mol2 = Molecule(
            id="mol_002",
            stage=MoleculeStage.INTERMEDIATE,
            role=BiologicalRole.FRAGMENT,
            producer_operation_id="op_001",
        )
        op1 = Operation(
            id="op_001",
            operation_key="pcr",
            input_molecule_ids=("mol_001",),  # produced by op_002
            output_molecule_ids=("mol_002",),
        )
        op2 = Operation(
            id="op_002",
            operation_key="pcr",
            input_molecule_ids=("mol_002",),  # produced by op_001
            output_molecule_ids=("mol_001",),
        )
        s = Strategy(
            id="strategy_001",
            goal="test cycle",
            molecules=(mol1, mol2),
            operations=(op1, op2),
        )
        issues = validate_strategy(s)
        blockers = [i for i in issues if i.code == "cycle_detected"]
        assert len(blockers) >= 1
        assert not is_approved(issues)

    def test_no_false_positive_for_linear_chain(self) -> None:
        """A → B → C is not a cycle."""
        mol_a = Molecule(
            id="mol_001",
            stage=MoleculeStage.SOURCE,
            role=BiologicalRole.VECTOR,
        )
        mol_b = Molecule(
            id="mol_002",
            stage=MoleculeStage.INTERMEDIATE,
            role=BiologicalRole.FRAGMENT,
            producer_operation_id="op_001",
        )
        mol_c = Molecule(
            id="mol_003",
            stage=MoleculeStage.PRODUCT,
            role=BiologicalRole.PRODUCT,
            producer_operation_id="op_002",
        )
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
        s = Strategy(
            id="strategy_001",
            goal="linear chain",
            molecules=(mol_a, mol_b, mol_c),
            operations=(op1, op2),
        )
        issues = validate_strategy(s)
        assert is_approved(issues), [i.message for i in issues]


# --- Validation: unknown operation keys --------------------------------------


class TestUnknownOperationKeys:
    """Scenario 35: unknown operation keys are blockers."""

    def test_unknown_operation_key_blocked(self) -> None:
        mol1 = Molecule(
            id="mol_001",
            stage=MoleculeStage.SOURCE,
            role=BiologicalRole.VECTOR,
        )
        op = Operation(
            id="op_001",
            operation_key="totally_bogus_operation",
            input_molecule_ids=("mol_001",),
        )
        s = Strategy(
            id="strategy_001",
            goal="test",
            molecules=(mol1,),
            operations=(op,),
        )
        issues = validate_strategy(s)
        blockers = [i for i in issues if i.code == "unknown_operation_key"]
        assert len(blockers) == 1
        assert not is_approved(issues)

    def test_experimental_operation_is_warning(self) -> None:
        """Experimental operations produce warnings, not blockers."""
        mol1 = Molecule(
            id="mol_001",
            stage=MoleculeStage.SOURCE,
            role=BiologicalRole.VECTOR,
        )
        op = Operation(
            id="op_001",
            operation_key="batch_domestication",
            input_molecule_ids=("mol_001",),
        )
        s = Strategy(
            id="strategy_001",
            goal="test experimental",
            molecules=(mol1,),
            operations=(op,),
        )
        issues = validate_strategy(s)
        warnings = [i for i in issues if i.code == "experimental_operation"]
        assert len(warnings) == 1
        assert is_approved(issues)  # warnings don't block


# --- Validation: batch limits -----------------------------------------------


class TestBatchLimits:
    """Scenario 36: batch limit enforcement."""

    def test_batch_at_limit_ok(self) -> None:
        s = _make_minimal_strategy(
            batch_members=tuple(
                BatchMember(id=f"batch_{i:03d}") for i in range(MAX_BATCH_MEMBERS)
            ),
        )
        issues = validate_strategy(s)
        assert is_approved(issues), [i.message for i in issues]

    def test_batch_over_limit_blocked(self) -> None:
        s = _make_minimal_strategy(
            batch_members=tuple(
                BatchMember(id=f"batch_{i:03d}") for i in range(MAX_BATCH_MEMBERS + 1)
            ),
        )
        issues = validate_strategy(s)
        blockers = [i for i in issues if i.code == "batch_limit_exceeded"]
        assert len(blockers) == 1
        assert not is_approved(issues)


# --- Validation: payload size -----------------------------------------------


class TestPayloadSize:
    """Scenario 37: payload size limit."""

    def test_normal_payload_ok(self) -> None:
        s = _make_minimal_strategy()
        issues = validate_strategy(s)
        assert not any(i.code == "payload_too_large" for i in issues)

    def test_oversized_payload_blocked(self) -> None:
        """Create a strategy with a very long goal to exceed the payload limit."""
        mol1 = Molecule(
            id="mol_001",
            stage=MoleculeStage.SOURCE,
            role=BiologicalRole.VECTOR,
            name="x" * (MAX_LABEL_LENGTH),
        )
        op = Operation(
            id="op_001",
            operation_key="pcr",
            input_molecule_ids=("mol_001",),
            parameters={"data": "x" * (MAX_STRATEGY_PAYLOAD_BYTES)},
        )
        s = Strategy(
            id="strategy_001",
            goal="oversized",
            molecules=(mol1,),
            operations=(op,),
        )
        issues = validate_strategy(s)
        blockers = [i for i in issues if i.code == "payload_too_large"]
        assert len(blockers) == 1
        assert not is_approved(issues)


# --- Validation: forbidden sequence fields -----------------------------------


class TestForbiddenSequenceFields:
    """No nucleotide sequences in the strategy."""

    def test_forbidden_field_in_parameters_blocked(self) -> None:
        mol1 = Molecule(
            id="mol_001",
            stage=MoleculeStage.SOURCE,
            role=BiologicalRole.VECTOR,
        )
        op = Operation(
            id="op_001",
            operation_key="pcr",
            input_molecule_ids=("mol_001",),
            parameters={"sequence": "ATCGATCG"},  # forbidden!
        )
        s = Strategy(
            id="strategy_001",
            goal="test",
            molecules=(mol1,),
            operations=(op,),
        )
        issues = validate_strategy(s)
        blockers = [i for i in issues if i.code == "forbidden_sequence_field"]
        assert len(blockers) == 1
        assert not is_approved(issues)

    def test_forbidden_field_in_annotation_properties_blocked(self) -> None:
        mol1 = Molecule(
            id="mol_001",
            stage=MoleculeStage.SOURCE,
            role=BiologicalRole.VECTOR,
            annotations=(
                Annotation(
                    kind="primer",
                    label="forward",
                    properties={"sequence": "ATCG"},  # forbidden!
                ),
            ),
        )
        op = Operation(
            id="op_001",
            operation_key="pcr",
            input_molecule_ids=("mol_001",),
        )
        s = Strategy(
            id="strategy_001",
            goal="test",
            molecules=(mol1,),
            operations=(op,),
        )
        issues = validate_strategy(s)
        blockers = [i for i in issues if i.code == "forbidden_sequence_field"]
        assert len(blockers) == 1
        assert not is_approved(issues)

    def test_safe_annotation_properties_ok(self) -> None:
        """Non-sequence fields in annotations are fine."""
        mol1 = Molecule(
            id="mol_001",
            stage=MoleculeStage.SOURCE,
            role=BiologicalRole.VECTOR,
            annotations=(
                Annotation(
                    kind="restriction_site",
                    label="BsaI",
                    properties={"position": 1234, "enzyme": "BsaI"},
                ),
            ),
        )
        op = Operation(
            id="op_001",
            operation_key="pcr",
            input_molecule_ids=("mol_001",),
        )
        s = Strategy(
            id="strategy_001",
            goal="test",
            molecules=(mol1,),
            operations=(op,),
        )
        issues = validate_strategy(s)
        assert is_approved(issues), [i.message for i in issues]


# --- Validation: product compositions ---------------------------------------


class TestProductCompositions:
    """Product composition reference validation."""

    def test_valid_composition_ok(self) -> None:
        s = _make_minimal_strategy(
            product_compositions=(
                ProductComposition(
                    product_molecule_id="mol_003",
                    segment_molecule_ids=("mol_001", "mol_002"),
                    segment_labels=("backbone", "insert"),
                ),
            ),
        )
        issues = validate_strategy(s)
        assert is_approved(issues), [i.message for i in issues]

    def test_dangling_product_reference_blocked(self) -> None:
        s = _make_minimal_strategy(
            product_compositions=(
                ProductComposition(
                    product_molecule_id="mol_999",  # doesn't exist
                ),
            ),
        )
        issues = validate_strategy(s)
        blockers = [i for i in issues if i.code == "dangling_product_reference"]
        assert len(blockers) == 1
        assert not is_approved(issues)


# --- Validation: batch member references -------------------------------------


class TestBatchMemberReferences:
    """Batch member reference validation."""

    def test_valid_batch_member_ok(self) -> None:
        s = _make_minimal_strategy(
            batch_members=(
                BatchMember(
                    id="batch_001",
                    molecule_ids=("mol_001",),
                    operation_ids=("op_001",),
                ),
            ),
        )
        issues = validate_strategy(s)
        assert is_approved(issues), [i.message for i in issues]

    def test_dangling_batch_molecule_blocked(self) -> None:
        s = _make_minimal_strategy(
            batch_members=(
                BatchMember(
                    id="batch_001",
                    molecule_ids=("mol_999",),
                ),
            ),
        )
        issues = validate_strategy(s)
        blockers = [i for i in issues if i.code == "dangling_batch_molecule_reference"]
        assert len(blockers) == 1
        assert not is_approved(issues)


# --- Label length validation -------------------------------------------------


class TestLabelLength:
    """Label length limits."""

    def test_normal_label_ok(self) -> None:
        s = _make_minimal_strategy()
        issues = validate_strategy(s)
        assert not any(i.code == "label_too_long" for i in issues)

    def test_oversized_molecule_name_blocked(self) -> None:
        mol1 = Molecule(
            id="mol_001",
            stage=MoleculeStage.SOURCE,
            role=BiologicalRole.VECTOR,
            name="x" * (MAX_LABEL_LENGTH + 1),
        )
        op = Operation(
            id="op_001",
            operation_key="pcr",
            input_molecule_ids=("mol_001",),
        )
        s = Strategy(
            id="strategy_001",
            goal="test",
            molecules=(mol1,),
            operations=(op,),
        )
        issues = validate_strategy(s)
        blockers = [i for i in issues if i.code == "label_too_long"]
        assert len(blockers) == 1
        assert not is_approved(issues)


# --- Multi-step strategy test -----------------------------------------------


class TestMultiStepStrategy:
    """A realistic multi-step strategy: PCR → digest → ligation."""

    def test_pcr_digest_ligation_valid(self) -> None:
        mol_template = Molecule(
            id="mol_001",
            stage=MoleculeStage.SOURCE,
            role=BiologicalRole.TARGET,
            name="template DNA",
        )
        mol_vector = Molecule(
            id="mol_002",
            stage=MoleculeStage.SOURCE,
            role=BiologicalRole.VECTOR,
            name="pUC19",
        )
        mol_amplicon = Molecule(
            id="mol_003",
            stage=MoleculeStage.PREPARED_PIECE,
            role=BiologicalRole.AMPLICON,
            name="GFP amplicon",
            producer_operation_id="op_001",
        )
        mol_vector_digested = Molecule(
            id="mol_004",
            stage=MoleculeStage.PREPARED_PIECE,
            role=BiologicalRole.BACKBONE,
            name="digested pUC19",
            producer_operation_id="op_002",
        )
        mol_product = Molecule(
            id="mol_005",
            stage=MoleculeStage.PRODUCT,
            role=BiologicalRole.PRODUCT,
            name="pUC19-GFP",
            producer_operation_id="op_003",
        )
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
        s = Strategy(
            id="strategy_001",
            goal="PCR → digest → ligation",
            molecules=(
                mol_template,
                mol_vector,
                mol_amplicon,
                mol_vector_digested,
                mol_product,
            ),
            operations=(op_pcr, op_digest, op_ligation),
            product_compositions=(
                ProductComposition(
                    product_molecule_id="mol_005",
                    segment_molecule_ids=("mol_004", "mol_003"),
                    segment_labels=("backbone", "insert"),
                ),
            ),
        )
        issues = validate_strategy(s)
        assert is_approved(issues), [i.message for i in issues]
        # Hash should be stable
        assert s.compute_hash() == s.compute_hash()
