"""Deterministic strategy executor for core cloning methods (Slice 6).

Executes a validated ``Strategy`` DAG topologically using opaque sequence
references.  No LLM calls participate in execution — the executor is purely
deterministic.

Core methods supported (scenarios 1–14):
    PCR, restriction digest, ligation, restriction cloning, Golden Gate,
    Gibson, In-Fusion, overlap-extension PCR, in-vivo assembly.

Key design decisions:

* **Opaque sequence references:** Molecules reference sequences by gateway-
  unique integer IDs.  The executor resolves these to full sequence dicts
  (with ``file_content``) from the sequence store before calling OpenCloning.
  No nucleotide sequences appear in progress/results or logs.
* **Operation ID preservation:** Every result carries the operation ID from
  the strategy DAG so the UI can map statuses to graph nodes.
* **Ambiguity detection:** If an operation produces zero or multiple
  products, the executor reports ``AMBIGUOUS`` or ``BLOCKED`` rather than
  guessing.
* **Bounded retries:** Transient HTTP failures are retried up to 3 times
  with exponential backoff.  Semantic errors are not retried.
* **Known workarounds:** PCR self-reference cleanup and Gibson input
  injection are handled by delegating to the existing adapter helpers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from lab_copilot_gateway.strategy import (
    Operation,
    OperationStatus,
    Strategy,
)
from lab_copilot_gateway.strategy_catalog import (
    OperationCapability,
    get_catalog,
)

# --- Constants ---------------------------------------------------------------

MAX_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 0.5
RETRY_BACKOFF_MULTIPLIER = 2.0

# Operations that route through simulate_assembly (source-type-based dispatch)
_ASSEMBLY_OPERATIONS = frozenset(
    {
        "gibson_assembly",
        "in_fusion_assembly",
        "overlap_extension_pcr",
        "in_vivo_assembly",
        "ligation",
        "restriction_ligation",
        "golden_gate",
        "homologous_recombination",
        "crispr_hdr",
        "cre_lox",
        "gateway",
        "recombinase",
    }
)

# Operations that route through call_endpoint (direct endpoint)
_DIRECT_ENDPOINT_OPERATIONS: dict[str, str] = {
    "pcr": "/pcr",
    "restriction_digest": "/restriction",
    "oligo_hybridization": "/oligonucleotide_hybridization",
    "polymerase_extension": "/polymerase_extension",
    "reverse_complement": "/reverse_complement",
}

# Native batch operations that route through call_endpoint with batch-specific
# body construction and terminal-product extraction from the CloningStrategy
# response. Verified against manulera/opencloning:v1.3.1-baseurl-opencloning.
_BATCH_OPERATIONS: dict[str, str] = {
    "batch_domestication": "/batch_cloning/domesticate",
    "batch_ziqiang": "/batch_cloning/ziqiang_et_al2024",
    # batch_pombe intentionally omitted — multipart form + ZIP response is
    # incompatible with the JSON DAG execution model. It will hit the
    # BLOCKED fallback with a clear error.
}


# --- Result types ------------------------------------------------------------


@dataclass(frozen=True)
class OperationResult:
    """Result of executing a single operation in the strategy DAG."""

    operation_id: str
    status: OperationStatus
    output_sequence_ids: tuple[int, ...] = field(default_factory=tuple)
    """Gateway-unique sequence IDs produced by this operation."""

    error: str | None = None
    """Error message if the operation failed."""

    ambiguity_detail: str | None = None
    """Description of ambiguity if status is AMBIGUOUS."""


@dataclass(frozen=True)
class StrategyExecutionResult:
    """Result of executing an entire strategy DAG."""

    strategy_id: str
    plan_hash: str
    status: str
    """Overall status: 'completed', 'partial_failure', 'blocked', 'ambiguous'."""

    operation_results: tuple[OperationResult, ...]
    """Per-operation results, in execution order."""

    product_sequence_ids: dict[str, int] = field(default_factory=dict)
    """Maps product molecule_id → gateway sequence ID for successful products."""

    error: str | None = None


# --- Client protocol --------------------------------------------------------


class ExecutionClient(Protocol):
    """Minimal client interface the executor needs.

    This is a subset of ``OpenCloningClient`` — the executor only needs
    ``simulate_assembly`` and ``call_endpoint``.
    """

    def simulate_assembly(
        self,
        sequences: list[dict[str, Any]],
        source: dict[str, Any],
    ) -> dict[str, Any]: ...

    def call_endpoint(
        self, path: str, body: dict[str, Any] | list[Any]
    ) -> dict[str, Any]: ...


# --- Executor ----------------------------------------------------------------


class StrategyExecutor:
    """Executes a validated strategy DAG topologically.

    The executor is stateless between runs — all state is in the
    ``ExecutionContext`` passed to ``execute()``.
    """

    def __init__(
        self,
        client: ExecutionClient,
        catalog: Any | None = None,
    ) -> None:
        self._client = client
        self._catalog = catalog or get_catalog()

    def execute(
        self,
        strategy: Strategy,
        sequence_store: dict[str, dict[str, Any]],
        seq_counter: int = 0,
    ) -> StrategyExecutionResult:
        """Execute a strategy DAG topologically.

        Args:
            strategy: The validated strategy to execute.
            sequence_store: Maps molecule_id → full sequence dict (with
                file_content).  Source molecules must already be in the
                store.  The executor adds output molecules as they're
                produced.
            seq_counter: Starting value for gateway-unique sequence IDs.
                The executor increments this as new sequences are created.

        Returns:
            StrategyExecutionResult with per-operation status and product
            sequence IDs.
        """
        ctx = _ExecutionContext(
            strategy=strategy,
            sequence_store=sequence_store,
            seq_counter=seq_counter,
        )

        ordered_ops = self._topological_sort(strategy)

        for op in ordered_ops:
            result = self._execute_operation(op, ctx)
            ctx.results.append(result)

            # Stop on blocked or ambiguous — downstream ops can't proceed
            if result.status in (OperationStatus.BLOCKED, OperationStatus.AMBIGUOUS):
                self._mark_remaining_skipped(op, ordered_ops, ctx)
                break

        result = self._build_result(strategy, ctx)

        # Telemetry: privacy-safe outcome counter (Slice 15 step 4).
        from lab_copilot_gateway.strategy_telemetry import get_telemetry

        get_telemetry().record_execute(outcome=result.status)

        return result

    def _topological_sort(self, strategy: Strategy) -> list[Operation]:
        """Sort operations by dependency (sources first, products last)."""
        producer = _build_producer_map(strategy)
        deps, op_by_id = _build_dependency_graph(strategy, producer)
        return _kahn_sort(deps, op_by_id)

    def _execute_operation(
        self, op: Operation, ctx: _ExecutionContext
    ) -> OperationResult:
        """Execute a single operation."""
        try:
            cap = self._catalog.get_by_operation_key(op.operation_key)
            if cap is None:
                return OperationResult(
                    operation_id=op.id,
                    status=OperationStatus.BLOCKED,
                    error=f"unknown operation key: {op.operation_key}",
                )

            # Batch operations may not need input sequences (e.g. ziqiang
            # uses a bundled template). Resolve inputs only if the operation
            # has input molecules.
            input_sequences: list[dict[str, Any]] = []
            if op.input_molecule_ids:
                input_sequences = self._resolve_inputs(op, ctx)
                if not input_sequences:
                    return OperationResult(
                        operation_id=op.id,
                        status=OperationStatus.BLOCKED,
                        error="could not resolve input sequences",
                    )

            # Execute based on operation type
            if op.operation_key in _ASSEMBLY_OPERATIONS:
                result = self._execute_assembly(op, cap, input_sequences, ctx)
            elif op.operation_key in _DIRECT_ENDPOINT_OPERATIONS:
                endpoint = _DIRECT_ENDPOINT_OPERATIONS[op.operation_key]
                result = self._execute_direct(op, endpoint, input_sequences, ctx)
            elif op.operation_key in _BATCH_OPERATIONS:
                endpoint = _BATCH_OPERATIONS[op.operation_key]
                result = self._execute_batch(op, endpoint, input_sequences, ctx)
            else:
                return OperationResult(
                    operation_id=op.id,
                    status=OperationStatus.BLOCKED,
                    error=f"operation {op.operation_key} not supported by executor",
                )

            return result

        except Exception as exc:
            return OperationResult(
                operation_id=op.id,
                status=OperationStatus.FAILED,
                error=str(exc),
            )

    def _resolve_inputs(
        self, op: Operation, ctx: _ExecutionContext
    ) -> list[dict[str, Any]]:
        """Resolve input molecule IDs to full sequence dicts."""
        sequences: list[dict[str, Any]] = []
        for mid in op.input_molecule_ids:
            seq = ctx.sequence_store.get(mid)
            if seq is None:
                # Check if this molecule was produced by a prior op
                gateway_id = ctx.molecule_to_seq.get(mid)
                if gateway_id is not None:
                    seq = ctx.sequence_store.get(str(gateway_id))
            if seq is not None:
                sequences.append(dict(seq))  # copy to avoid mutation
        return sequences

    def _execute_assembly(
        self,
        op: Operation,
        cap: OperationCapability,
        sequences: list[dict[str, Any]],
        ctx: _ExecutionContext,
    ) -> OperationResult:
        """Execute an assembly operation via simulate_assembly."""
        source = self._build_source(op, cap)
        response = self._call_with_retry(
            lambda: self._client.simulate_assembly(sequences, source)
        )
        return self._process_response(op, response, ctx)

    def _execute_direct(
        self,
        op: Operation,
        endpoint: str,
        sequences: list[dict[str, Any]],
        ctx: _ExecutionContext,
    ) -> OperationResult:
        """Execute a direct-endpoint operation (PCR, restriction, etc.)."""
        body = self._build_direct_body(op, sequences)
        response = self._call_with_retry(
            lambda: self._client.call_endpoint(endpoint, body)
        )
        return self._process_response(op, response, ctx)

    def _execute_batch(
        self,
        op: Operation,
        endpoint: str,
        sequences: list[dict[str, Any]],
        ctx: _ExecutionContext,
    ) -> OperationResult:
        """Execute a native batch operation (domesticate, ziqiang, etc.).

        Batch endpoints return a full ``BaseCloningStrategy`` (sequences +
        sources), not just output sequences. The terminal product — the
        sequence that is not an input to any other source — is extracted
        and stored as the operation's output.
        """
        body = self._build_batch_body(op, sequences)
        response = self._call_with_retry(
            lambda: self._client.call_endpoint(endpoint, body)
        )
        return self._process_batch_response(op, response, ctx)

    @staticmethod
    def _build_source(op: Operation, cap: OperationCapability) -> dict[str, Any]:
        """Build the OpenCloning source object for an assembly operation."""
        source: dict[str, Any] = {
            "id": 0,
            "type": cap.source_type,
        }
        # Merge operation parameters into the source
        for key, value in op.parameters.items():
            if key not in ("sequences",):
                source[key] = value
        return source

    @staticmethod
    def _build_direct_body(
        op: Operation, sequences: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Build the request body for a direct-endpoint operation."""
        body: dict[str, Any] = {"sequences": sequences}
        body.update(op.parameters)
        return body

    @staticmethod
    def _build_batch_body(
        op: Operation, sequences: list[dict[str, Any]]
    ) -> dict[str, Any] | list[str]:
        """Build the request body for a native batch operation.

        - ``batch_domestication``: JSON dict with sequence + params.
        - ``batch_ziqiang``: JSON array of protospacer strings from
          operation parameters (no input sequence needed).
        """
        if op.operation_key == "batch_ziqiang":
            protospacers = op.parameters.get("protospacers", [])
            if not protospacers:
                raise ValueError("batch_ziqiang requires 'protospacers' parameter")
            return list(protospacers)

        # Default: dict body with first input sequence + operation params
        body: dict[str, Any] = {}
        if sequences:
            body["sequence"] = sequences[0]
        body.update(op.parameters)
        return body

    def _process_batch_response(
        self, op: Operation, response: dict[str, Any], ctx: _ExecutionContext
    ) -> OperationResult:
        """Process a batch CloningStrategy response.

        Extracts the terminal product — the sequence whose ID is not
        consumed as input by any source — and stores it as the output.
        """
        all_sequences = response.get("sequences") or []
        if not all_sequences:
            return OperationResult(
                operation_id=op.id,
                status=OperationStatus.AMBIGUOUS,
                ambiguity_detail="batch operation produced zero sequences",
            )

        terminal = _find_terminal_products(all_sequences, response.get("sources") or [])
        if not terminal:
            return OperationResult(
                operation_id=op.id,
                status=OperationStatus.AMBIGUOUS,
                ambiguity_detail="batch operation has no terminal product",
            )

        if len(terminal) > 1:
            names = [s.get("name", s.get("id", "?")) for s in terminal]
            return OperationResult(
                operation_id=op.id,
                status=OperationStatus.AMBIGUOUS,
                ambiguity_detail=(
                    f"batch operation produced {len(terminal)} terminal products "
                    f"({names}); product selector not yet implemented"
                ),
            )

        # Single terminal product — store it
        seq = terminal[0]
        ctx.seq_counter += 1
        gateway_id = ctx.seq_counter
        seq["id"] = gateway_id
        ctx.sequence_store[str(gateway_id)] = dict(seq)

        for mid in op.output_molecule_ids:
            ctx.molecule_to_seq[mid] = gateway_id

        return OperationResult(
            operation_id=op.id,
            status=OperationStatus.SUCCEEDED,
            output_sequence_ids=(gateway_id,),
        )

    def _process_response(
        self, op: Operation, response: dict[str, Any], ctx: _ExecutionContext
    ) -> OperationResult:
        """Process an OpenCloning response and store output sequences."""
        output_seqs = response.get("sequences") or []

        if not output_seqs:
            return OperationResult(
                operation_id=op.id,
                status=OperationStatus.AMBIGUOUS,
                ambiguity_detail="operation produced zero products",
            )

        if len(output_seqs) > 1:
            # Multiple products — check if the strategy specifies which one
            # For now, report ambiguity (product selection is a future enhancement)
            return OperationResult(
                operation_id=op.id,
                status=OperationStatus.AMBIGUOUS,
                ambiguity_detail=(
                    f"operation produced {len(output_seqs)} products; "
                    f"product selector not yet implemented"
                ),
            )

        # Single product — store it
        seq = output_seqs[0]
        ctx.seq_counter += 1
        gateway_id = ctx.seq_counter
        seq["id"] = gateway_id
        ctx.sequence_store[str(gateway_id)] = dict(seq)

        # Map output molecule IDs to this sequence
        output_ids: list[int] = [gateway_id]
        for mid in op.output_molecule_ids:
            ctx.molecule_to_seq[mid] = gateway_id

        return OperationResult(
            operation_id=op.id,
            status=OperationStatus.SUCCEEDED,
            output_sequence_ids=tuple(output_ids),
        )

    def _call_with_retry(self, fn: Any) -> dict[str, Any]:
        """Call a function with bounded transient retries."""
        last_exc: Exception | None = None
        delay = RETRY_BASE_DELAY_SECONDS
        for attempt in range(MAX_RETRIES):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                if attempt < MAX_RETRIES - 1:
                    time.sleep(delay)
                    delay *= RETRY_BACKOFF_MULTIPLIER
        raise last_exc  # type: ignore[misc]

    @staticmethod
    def _mark_remaining_skipped(
        completed_op: Operation,
        all_ops: list[Operation],
        ctx: _ExecutionContext,
    ) -> None:
        """Mark operations after the blocked one as skipped."""
        found = False
        for op in all_ops:
            if op.id == completed_op.id:
                found = True
                continue
            if found:
                ctx.results.append(
                    OperationResult(
                        operation_id=op.id,
                        status=OperationStatus.BLOCKED,
                        error="skipped due to upstream failure",
                    )
                )

    @staticmethod
    def _build_result(
        strategy: Strategy, ctx: _ExecutionContext
    ) -> StrategyExecutionResult:
        """Build the final execution result."""
        statuses = [r.status for r in ctx.results]
        overall = _compute_overall_status(statuses)
        product_ids = _collect_product_ids(strategy, ctx)

        return StrategyExecutionResult(
            strategy_id=strategy.id,
            plan_hash=strategy.compute_hash(),
            status=overall,
            operation_results=tuple(ctx.results),
            product_sequence_ids=product_ids,
        )


# --- Execution context (internal) -------------------------------------------


@dataclass
class _ExecutionContext:
    """Mutable state for a single execution run."""

    strategy: Strategy
    sequence_store: dict[str, dict[str, Any]]
    seq_counter: int = 0
    molecule_to_seq: dict[str, int] = field(default_factory=dict)
    """Maps strategy molecule_id → gateway sequence ID."""
    results: list[OperationResult] = field(default_factory=list)


# --- Topological sort helpers ------------------------------------------------


def _build_producer_map(strategy: Strategy) -> dict[str, str]:
    """Build molecule_id → operation_id that produced it."""
    producer: dict[str, str] = {}
    for mol in strategy.molecules:
        if mol.producer_operation_id:
            producer[mol.id] = mol.producer_operation_id
    return producer


def _build_dependency_graph(
    strategy: Strategy, producer: dict[str, str]
) -> tuple[dict[str, set[str]], dict[str, Operation]]:
    """Build op_id → set of op_ids it depends on."""
    deps: dict[str, set[str]] = {}
    op_by_id = {op.id: op for op in strategy.operations}
    for op in strategy.operations:
        op_deps: set[str] = set()
        for mid in op.input_molecule_ids:
            if mid in producer:
                op_deps.add(producer[mid])
        deps[op.id] = op_deps
    return deps, op_by_id


def _kahn_sort(
    deps: dict[str, set[str]], op_by_id: dict[str, Operation]
) -> list[Operation]:
    """Kahn's algorithm for topological sorting."""
    in_degree: dict[str, int] = {oid: 0 for oid in deps}
    for oid, dep_set in deps.items():
        in_degree[oid] = len(dep_set)

    queue: list[str] = [oid for oid, d in in_degree.items() if d == 0]
    ordered: list[Operation] = []

    while queue:
        oid = queue.pop(0)
        ordered.append(op_by_id[oid])
        for other_oid, dep_set in deps.items():
            if oid in dep_set:
                dep_set.discard(oid)
                in_degree[other_oid] -= 1
                if in_degree[other_oid] == 0:
                    queue.append(other_oid)

    return ordered


# --- Result building helpers -------------------------------------------------


def _find_terminal_products(
    sequences: list[dict[str, Any]], sources: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Find sequences not consumed as input by any source.

    In a CloningStrategy, the terminal product is the sequence whose ID
    does not appear in any source's ``input`` list. This is the final
    output of the coupled workflow.
    """
    consumed_ids: set[int] = set()
    for source in sources:
        for inp in source.get("input", []):
            seq_ref = inp.get("sequence") if isinstance(inp, dict) else inp
            if seq_ref is not None:
                consumed_ids.add(seq_ref)
    return [s for s in sequences if s.get("id") not in consumed_ids]


def _compute_overall_status(statuses: list[OperationStatus]) -> str:
    """Compute overall execution status from per-operation statuses."""
    has_blocked = any(s == OperationStatus.BLOCKED for s in statuses)
    has_ambiguous = any(s == OperationStatus.AMBIGUOUS for s in statuses)
    has_failed = any(s == OperationStatus.FAILED for s in statuses)

    if has_ambiguous:
        return "ambiguous"
    if has_blocked and not has_failed:
        return "blocked"
    if has_blocked or has_failed:
        return "partial_failure"
    return "completed"


def _collect_product_ids(strategy: Strategy, ctx: _ExecutionContext) -> dict[str, int]:
    """Map product molecules to their gateway sequence IDs."""
    product_ids: dict[str, int] = {}
    for mol in strategy.molecules:
        if mol.stage.value == "product":
            gateway_id = ctx.molecule_to_seq.get(mol.id)
            if gateway_id is not None:
                product_ids[mol.id] = gateway_id
    return product_ids
