"""Validation for canonical strategy schema v3 (Slice 2).

Validates a ``Strategy`` DAG for structural correctness and returns a list
of ``ValidationIssue`` objects.  Blockers prevent approval; warnings do not.

Validation checks:

1. Unique molecule IDs
2. Unique operation IDs
3. All operation input_molecule_ids reference existing molecules
4. All operation output_molecule_ids reference existing molecules
5. All producer_operation_id references on molecules point to existing operations
6. No cycles in the operation dependency graph
7. All operation_keys exist in the strategy catalog
8. Batch member count ≤ MAX_BATCH_MEMBERS
9. Label lengths ≤ MAX_LABEL_LENGTH
10. Payload size ≤ MAX_STRATEGY_PAYLOAD_BYTES
11. No forbidden sequence fields in parameters or annotations
12. Product compositions reference existing molecules
13. Batch members reference existing molecules and operations
"""

from __future__ import annotations

from typing import Any

from lab_copilot_gateway.strategy import (
    FORBIDDEN_SEQUENCE_FIELDS,
    MAX_BATCH_MEMBERS,
    MAX_LABEL_LENGTH,
    MAX_STRATEGY_PAYLOAD_BYTES,
    Operation,
    Strategy,
    ValidationIssue,
    ValidationSeverity,
)
from lab_copilot_gateway.strategy_catalog import get_catalog


def validate_strategy(strategy: Strategy) -> list[ValidationIssue]:
    """Validate a strategy and return all issues found.

    Returns an empty list if the strategy is valid.
    Blockers must be resolved before approval; warnings are informational.
    """
    issues: list[ValidationIssue] = []

    issues.extend(_check_unique_ids(strategy))
    issues.extend(_check_references(strategy))
    issues.extend(_check_acyclic(strategy))
    issues.extend(_check_operation_keys(strategy))
    issues.extend(_check_batch_limits(strategy))
    issues.extend(_check_label_lengths(strategy))
    issues.extend(_check_payload_size(strategy))
    issues.extend(_check_forbidden_fields(strategy))
    issues.extend(_check_product_compositions(strategy))
    issues.extend(_check_batch_members(strategy))

    return issues


def has_blockers(issues: list[ValidationIssue]) -> bool:
    """Return True if any issue is a blocker."""
    return any(i.severity == ValidationSeverity.BLOCKER for i in issues)


def is_approved(issues: list[ValidationIssue]) -> bool:
    """Return True if the strategy has no blockers (warnings are OK)."""
    return not has_blockers(issues)


# --- Individual checks -------------------------------------------------------


def _check_unique_ids(strategy: Strategy) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    mol_ids = [m.id for m in strategy.molecules]
    seen: set[str] = set()
    for mid in mol_ids:
        if mid in seen:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.BLOCKER,
                    code="duplicate_molecule_id",
                    message=f"duplicate molecule ID: {mid}",
                    node_id=mid,
                )
            )
        seen.add(mid)

    op_ids = [o.id for o in strategy.operations]
    seen = set()
    for oid in op_ids:
        if oid in seen:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.BLOCKER,
                    code="duplicate_operation_id",
                    message=f"duplicate operation ID: {oid}",
                    node_id=oid,
                )
            )
        seen.add(oid)

    return issues


def _check_references(strategy: Strategy) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    mol_ids = {m.id for m in strategy.molecules}
    op_ids = {o.id for o in strategy.operations}

    for op in strategy.operations:
        issues.extend(_check_operation_references(op, mol_ids))

    for mol in strategy.molecules:
        if mol.producer_operation_id and mol.producer_operation_id not in op_ids:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.BLOCKER,
                    code="dangling_producer_reference",
                    message=f"molecule {mol.id} references unknown producer operation {mol.producer_operation_id}",
                    node_id=mol.id,
                )
            )

    return issues


def _check_operation_references(
    op: Operation, mol_ids: set[str]
) -> list[ValidationIssue]:
    """Check that an operation's input/output molecule references exist."""
    issues: list[ValidationIssue] = []
    for mid in op.input_molecule_ids:
        if mid not in mol_ids:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.BLOCKER,
                    code="dangling_input_reference",
                    message=f"operation {op.id} references unknown input molecule {mid}",
                    node_id=op.id,
                )
            )
    for mid in op.output_molecule_ids:
        if mid not in mol_ids:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.BLOCKER,
                    code="dangling_output_reference",
                    message=f"operation {op.id} references unknown output molecule {mid}",
                    node_id=op.id,
                )
            )
    return issues


def _check_acyclic(strategy: Strategy) -> list[ValidationIssue]:
    """Check that the operation dependency graph is acyclic.

    An operation depends on another if it consumes a molecule produced
    by that other operation.
    """
    issues: list[ValidationIssue] = []
    deps = _build_dependency_graph(strategy)
    _detect_cycles(deps, issues)
    return issues


def _build_dependency_graph(strategy: Strategy) -> dict[str, set[str]]:
    """Build op_id → set of op_ids it depends on."""
    producer: dict[str, str] = {}
    for mol in strategy.molecules:
        if mol.producer_operation_id:
            producer[mol.id] = mol.producer_operation_id

    deps: dict[str, set[str]] = {}
    for op in strategy.operations:
        op_deps: set[str] = set()
        for mid in op.input_molecule_ids:
            if mid in producer:
                op_deps.add(producer[mid])
        deps[op.id] = op_deps
    return deps


def _detect_cycles(deps: dict[str, set[str]], issues: list[ValidationIssue]) -> None:
    """DFS cycle detection on the dependency graph."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {oid: WHITE for oid in deps}

    for oid in deps:
        if color[oid] == WHITE:
            _dfs_visit(oid, [], deps, color, GRAY, BLACK, issues)


def _dfs_visit(
    node: str,
    path: list[str],
    deps: dict[str, set[str]],
    color: dict[str, int],
    gray: int,
    black: int,
    issues: list[ValidationIssue],
) -> bool:
    """Single DFS visit for cycle detection."""
    color[node] = gray
    for neighbor in deps.get(node, set()):
        if color[neighbor] == gray:
            cycle = path + [node, neighbor]
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.BLOCKER,
                    code="cycle_detected",
                    message=f"cycle detected in operation dependency graph: {' → '.join(cycle)}",
                    node_id=node,
                )
            )
            return True
        if color[neighbor] == 0:  # WHITE
            if _dfs_visit(neighbor, path + [node], deps, color, gray, black, issues):
                return True
    color[node] = black
    return False


def _check_operation_keys(strategy: Strategy) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    catalog = get_catalog()

    for op in strategy.operations:
        cap = catalog.get_by_operation_key(op.operation_key)
        if cap is None:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.BLOCKER,
                    code="unknown_operation_key",
                    message=f"operation {op.id} has unknown operation_key: {op.operation_key}",
                    node_id=op.id,
                )
            )
        elif cap.status.value == "experimental":
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    code="experimental_operation",
                    message=f"operation {op.id} uses experimental operation {op.operation_key}",
                    node_id=op.id,
                )
            )

    return issues


def _check_batch_limits(strategy: Strategy) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if len(strategy.batch_members) > MAX_BATCH_MEMBERS:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.BLOCKER,
                code="batch_limit_exceeded",
                message=f"batch has {len(strategy.batch_members)} members, max is {MAX_BATCH_MEMBERS}",
            )
        )
    return issues


def _check_label_lengths(strategy: Strategy) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    for mol in strategy.molecules:
        if len(mol.name) > MAX_LABEL_LENGTH:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.BLOCKER,
                    code="label_too_long",
                    message=f"molecule {mol.id} name exceeds {MAX_LABEL_LENGTH} chars",
                    node_id=mol.id,
                )
            )

    for op in strategy.operations:
        for ann in op.annotations:
            if len(ann.label) > MAX_LABEL_LENGTH:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.BLOCKER,
                        code="label_too_long",
                        message=f"operation {op.id} annotation label exceeds {MAX_LABEL_LENGTH} chars",
                        node_id=op.id,
                    )
                )

    return issues


def _check_payload_size(strategy: Strategy) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    payload = strategy.to_canonical_json()
    if len(payload.encode("utf-8")) > MAX_STRATEGY_PAYLOAD_BYTES:
        issues.append(
            ValidationIssue(
                severity=ValidationSeverity.BLOCKER,
                code="payload_too_large",
                message=f"strategy payload {len(payload)} bytes exceeds limit {MAX_STRATEGY_PAYLOAD_BYTES}",
            )
        )
    return issues


def _check_forbidden_fields(strategy: Strategy) -> list[ValidationIssue]:
    """Reject nucleotide-like fields in parameters and annotations."""
    issues: list[ValidationIssue] = []

    for op in strategy.operations:
        _check_dict_for_forbidden_keys(
            op.parameters, f"operation {op.id} parameters", op.id, issues
        )
        for ann in op.annotations:
            _check_dict_for_forbidden_keys(
                ann.properties,
                f"operation {op.id} annotation {ann.kind}",
                op.id,
                issues,
            )

    for mol in strategy.molecules:
        for ann in mol.annotations:
            _check_dict_for_forbidden_keys(
                ann.properties,
                f"molecule {mol.id} annotation {ann.kind}",
                mol.id,
                issues,
            )

    return issues


def _check_dict_for_forbidden_keys(
    d: dict[str, Any], context: str, node_id: str, issues: list[ValidationIssue]
) -> None:
    for key in d:
        if key.lower() in FORBIDDEN_SEQUENCE_FIELDS:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.BLOCKER,
                    code="forbidden_sequence_field",
                    message=f"{context} contains forbidden field '{key}' (nucleotide sequences not allowed in strategy)",
                    node_id=node_id,
                )
            )


def _check_product_compositions(strategy: Strategy) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    mol_ids = {m.id for m in strategy.molecules}

    for comp in strategy.product_compositions:
        if comp.product_molecule_id not in mol_ids:
            issues.append(
                ValidationIssue(
                    severity=ValidationSeverity.BLOCKER,
                    code="dangling_product_reference",
                    message=f"product composition references unknown molecule {comp.product_molecule_id}",
                    node_id=comp.product_molecule_id,
                )
            )
        for sid in comp.segment_molecule_ids:
            if sid not in mol_ids:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.BLOCKER,
                        code="dangling_segment_reference",
                        message=f"product composition references unknown segment molecule {sid}",
                        node_id=sid,
                    )
                )

    return issues


def _check_batch_members(strategy: Strategy) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    mol_ids = {m.id for m in strategy.molecules}
    op_ids = {o.id for o in strategy.operations}

    for member in strategy.batch_members:
        for mid in member.molecule_ids:
            if mid not in mol_ids:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.BLOCKER,
                        code="dangling_batch_molecule_reference",
                        message=f"batch member {member.id} references unknown molecule {mid}",
                        node_id=member.id,
                    )
                )
        for oid in member.operation_ids:
            if oid not in op_ids:
                issues.append(
                    ValidationIssue(
                        severity=ValidationSeverity.BLOCKER,
                        code="dangling_batch_operation_reference",
                        message=f"batch member {member.id} references unknown operation {oid}",
                        node_id=member.id,
                    )
                )

    return issues
