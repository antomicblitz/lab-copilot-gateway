"""Canonical strategy schema v3 (Slice 2).

Defines the molecule/operation DAG that represents a complete cloning
strategy.  This is the hardest-to-reverse contract in the v3 plan — it
must cleanly represent:

* Source molecules (circular vectors, linear fragments, typed sequences)
* Prepared assembly pieces (PCR products, digested fragments, amplicons)
* Operations that transform inputs into outputs (PCR, digest, ligation, etc.)
* Dependencies between molecules and operations (acyclic graph)
* Biological roles (backbone, insert, amplicon, target, repair template)
* Stages (source, prepared piece, intermediate, product)
* Conceptual annotations (primers, homology, sites, overhangs, guides, cuts)
* Ordered product composition
* Optional batch members (up to 100)
* Validation issues (blockers and warnings)

Design principles
-----------------

* **No nucleotide sequences** in the schema.  Molecules reference opaque
  sequence IDs that the gateway resolves during execution.
* **No method-specific display hacks** in the core graph model.  Visual
  families are derived from the operation catalog, not embedded here.
* **Stable canonical JSON** — the schema serializes deterministically so
  SHA-256 hashing is reproducible.
* **Stage ≠ biological role** — a molecule can be a "source" stage with
  a "backbone" role, or an "intermediate" stage with an "insert" role.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# --- Enums -------------------------------------------------------------------


class MoleculeStage(str, Enum):
    """Where a molecule sits in the preparation pipeline."""

    SOURCE = "source"
    """Original input: circular vector, linear fragment, typed sequence."""

    PREPARED_PIECE = "prepared_piece"
    """Derived from a source via a preparation operation (PCR product,
    digested fragment, amplicon)."""

    INTERMEDIATE = "intermediate"
    """Output of a multi-step operation that is itself input to another."""

    PRODUCT = "product"
    """Final output of the strategy — the intended cloning product."""


class BiologicalRole(str, Enum):
    """Biological function of a molecule in the strategy.

    A molecule's role is independent of its stage — a source vector can
    be a backbone, and a prepared PCR product can be an insert.
    """

    BACKBONE = "backbone"
    """Circular vector providing the replicon/selection backbone."""

    INSERT = "insert"
    """DNA fragment being cloned into a backbone."""

    PART = "part"
    """Standardized genetic part (promoter, CDS, terminator, etc.)."""

    AMPLICON = "amplicon"
    """PCR product amplified from a template."""

    TARGET = "target"
    """Molecule being modified (HR target, CRISPR target)."""

    REPAIR_TEMPLATE = "repair_template"
    """Donor template for homologous recombination or CRISPR-HDR."""

    FRAGMENT = "fragment"
    """Generic prepared fragment without a specific biological role."""

    OLIGO = "oligo"
    """Short single-stranded oligonucleotide."""

    VECTOR = "vector"
    """Circular source molecule (before backbone/insert roles are assigned)."""

    PRODUCT = "product"
    """Final assembled product."""


class OperationStatus(str, Enum):
    """Status of an operation in the strategy lifecycle."""

    PROPOSED = "proposed"
    """Operation is part of the approved plan but not yet executed."""

    RUNNING = "running"
    """Operation is currently executing."""

    SUCCEEDED = "succeeded"
    """Operation completed successfully."""

    FAILED = "failed"
    """Operation failed during execution."""

    AMBIGUOUS = "ambiguous"
    """Operation produced multiple products; selection required."""

    BLOCKED = "blocked"
    """Operation could not execute due to upstream failure or ambiguity."""

    INTERRUPTED = "interrupted"
    """Operation was interrupted by gateway restart or cancellation."""


class ValidationSeverity(str, Enum):
    """Severity of a validation issue."""

    BLOCKER = "blocker"
    """Structural error — approval is refused."""

    WARNING = "warning"
    """Unknown or optional detail — approval proceeds with a warning."""


# --- Dataclasses -------------------------------------------------------------


@dataclass(frozen=True)
class Annotation:
    """Conceptual annotation on a molecule or operation.

    Annotations are display-only metadata — they do not affect execution.
    Examples: primer names, homology regions, restriction sites, overhangs,
    guide sequences, cut positions, recombination sites.
    """

    kind: str
    """Annotation type: 'primer', 'homology', 'restriction_site', 'overhang',
    'guide', 'cut', 'recombination_site', etc."""

    label: str
    """Human-readable label for display."""

    properties: dict[str, Any] = field(default_factory=dict)
    """Structured properties (e.g. {'position': 1234, 'enzyme': 'BsaI'}).
    No nucleotide sequences — only positions, names, and counts."""


@dataclass(frozen=True)
class Molecule:
    """A molecule node in the strategy DAG.

    Molecules reference opaque sequence IDs — the gateway resolves these
    to actual sequences during execution.  No nucleotide sequences are
    stored in the schema.
    """

    id: str
    """Stable, unique identifier within the strategy (e.g. 'mol_001')."""

    stage: MoleculeStage
    """Where this molecule sits in the preparation pipeline."""

    role: BiologicalRole
    """Biological function of this molecule."""

    sequence_ref: str | None = None
    """Opaque reference to a sequence in the gateway's sequence store.
    None for molecules that haven't been created yet (future products)."""

    name: str = ""
    """Human-readable name for display."""

    annotations: tuple[Annotation, ...] = field(default_factory=tuple)
    """Conceptual annotations (primers, sites, homology, etc.)."""

    producer_operation_id: str | None = None
    """ID of the operation that produced this molecule.
    None for source molecules."""

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Molecule.id must not be empty")
        if not self.id.startswith("mol_"):
            raise ValueError(f"Molecule.id must start with 'mol_': {self.id}")


@dataclass(frozen=True)
class Operation:
    """An operation node in the strategy DAG.

    Operations transform input molecules into output molecules.
    The operation_key must exist in the strategy catalog.
    """

    id: str
    """Stable, unique identifier (e.g. 'op_001')."""

    operation_key: str
    """Canonical operation key from the strategy catalog (e.g. 'pcr',
    'gibson_assembly', 'golden_gate')."""

    input_molecule_ids: tuple[str, ...]
    """IDs of input molecules (must exist in the strategy)."""

    output_molecule_ids: tuple[str, ...] = field(default_factory=tuple)
    """IDs of output molecules (must exist in the strategy)."""

    parameters: dict[str, Any] = field(default_factory=dict)
    """Operation-specific parameters (e.g. {'enzymes': ['BsaI'], 'minimal_annealing': 20}).
    No nucleotide sequences — only enzyme names, temperatures, counts."""

    annotations: tuple[Annotation, ...] = field(default_factory=tuple)
    """Conceptual annotations (guide sequences, cut sites, etc.)."""

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Operation.id must not be empty")
        if not self.id.startswith("op_"):
            raise ValueError(f"Operation.id must start with 'op_': {self.id}")
        if not self.operation_key:
            raise ValueError("Operation.operation_key must not be empty")
        if not self.input_molecule_ids:
            # Batch native workflows (e.g. ziqiang_et_al2024) may use a
            # bundled template with no external input molecules.
            if not self.operation_key.startswith("batch_"):
                raise ValueError(
                    f"Operation {self.id} must have at least one input molecule"
                )


@dataclass(frozen=True)
class ProductComposition:
    """Ordered composition of a final product.

    Describes how the product is assembled from its constituent pieces,
    in order.  This is display metadata — the actual assembly is
    determined by the operation chain.
    """

    product_molecule_id: str
    """ID of the product molecule."""

    segment_molecule_ids: tuple[str, ...] = field(default_factory=tuple)
    """IDs of molecules that form the segments of the product, in order."""

    segment_labels: tuple[str, ...] = field(default_factory=tuple)
    """Optional labels for each segment (e.g. 'backbone', 'insert')."""


@dataclass(frozen=True)
class BatchMember:
    """An independent member of a batch strategy.

    Each member has its own sub-graph of molecules and operations.
    Up to 100 members per strategy.
    """

    id: str
    """Stable, unique identifier (e.g. 'batch_001')."""

    label: str = ""
    """Human-readable label for display."""

    molecule_ids: tuple[str, ...] = field(default_factory=tuple)
    """IDs of molecules in this member's sub-graph."""

    operation_ids: tuple[str, ...] = field(default_factory=tuple)
    """IDs of operations in this member's sub-graph."""


@dataclass(frozen=True)
class ValidationIssue:
    """A validation issue found during strategy validation."""

    severity: ValidationSeverity
    code: str
    """Machine-readable code (e.g. 'duplicate_id', 'cycle_detected',
    'unknown_operation_key')."""

    message: str
    """Human-readable description."""

    node_id: str | None = None
    """ID of the molecule or operation that caused the issue."""


@dataclass(frozen=True)
class Strategy:
    """A canonical cloning strategy v3.

    This is the root dataclass that gets hashed and approved.  The
    canonical JSON serialization is deterministic (sorted keys, no
    whitespace) so the SHA-256 hash is reproducible.
    """

    id: str
    """Stable strategy identifier (e.g. 'strategy_001')."""

    goal: str
    """Human-readable goal statement (e.g. 'Assemble pUC19-GFP by Gibson assembly')."""

    molecules: tuple[Molecule, ...]
    """All molecule nodes in the DAG."""

    operations: tuple[Operation, ...]
    """All operation nodes in the DAG."""

    product_compositions: tuple[ProductComposition, ...] = field(default_factory=tuple)
    """Ordered composition of final products."""

    batch_members: tuple[BatchMember, ...] = field(default_factory=tuple)
    """Independent batch members (up to 100)."""

    revision: int = 1
    """Revision number (incremented on each supersession)."""

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Strategy.id must not be empty")
        if not self.goal:
            raise ValueError("Strategy.goal must not be empty")
        if not self.molecules:
            raise ValueError("Strategy must have at least one molecule")
        if not self.operations:
            raise ValueError("Strategy must have at least one operation")

    def to_canonical_json(self) -> str:
        """Serialize to deterministic JSON for hashing.

        Keys are sorted, no extra whitespace, enums serialized as values.
        """
        return json.dumps(self._to_dict(), sort_keys=True, separators=(",", ":"))

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of the canonical JSON."""
        return hashlib.sha256(self.to_canonical_json().encode("utf-8")).hexdigest()

    def _to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "revision": self.revision,
            "molecules": [self._molecule_to_dict(m) for m in self.molecules],
            "operations": [self._operation_to_dict(o) for o in self.operations],
            "product_compositions": [
                self._composition_to_dict(c) for c in self.product_compositions
            ],
            "batch_members": [
                self._batch_member_to_dict(b) for b in self.batch_members
            ],
        }

    @staticmethod
    def _annotation_to_dict(a: Annotation) -> dict[str, Any]:
        return {
            "kind": a.kind,
            "label": a.label,
            "properties": dict(sorted(a.properties.items())),
        }

    @classmethod
    def _molecule_to_dict(cls, m: Molecule) -> dict[str, Any]:
        return {
            "id": m.id,
            "stage": m.stage.value,
            "role": m.role.value,
            "sequence_ref": m.sequence_ref,
            "name": m.name,
            "annotations": [cls._annotation_to_dict(a) for a in m.annotations],
            "producer_operation_id": m.producer_operation_id,
        }

    @classmethod
    def _operation_to_dict(cls, o: Operation) -> dict[str, Any]:
        return {
            "id": o.id,
            "operation_key": o.operation_key,
            "input_molecule_ids": list(o.input_molecule_ids),
            "output_molecule_ids": list(o.output_molecule_ids),
            "parameters": dict(sorted(o.parameters.items())),
            "annotations": [cls._annotation_to_dict(a) for a in o.annotations],
        }

    @staticmethod
    def _composition_to_dict(c: ProductComposition) -> dict[str, Any]:
        return {
            "product_molecule_id": c.product_molecule_id,
            "segment_molecule_ids": list(c.segment_molecule_ids),
            "segment_labels": list(c.segment_labels),
        }

    @staticmethod
    def _batch_member_to_dict(b: BatchMember) -> dict[str, Any]:
        return {
            "id": b.id,
            "label": b.label,
            "molecule_ids": list(b.molecule_ids),
            "operation_ids": list(b.operation_ids),
        }


# --- Constants ---------------------------------------------------------------

MAX_BATCH_MEMBERS = 100
MAX_LABEL_LENGTH = 200
MAX_STRATEGY_PAYLOAD_BYTES = 512 * 1024  # 512 KiB
FORBIDDEN_SEQUENCE_FIELDS = frozenset(
    {
        "sequence",
        "nucleotide_sequence",
        "dna_sequence",
        "file_content",
        "genbank",
        "fasta",
        "raw_sequence",
    }
)
