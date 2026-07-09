"""OpenCloning adapter for the lab copilot gateway (C16).

Orchestrates gateway-enforced calls to the OpenCloning backend service for
computational cloning design.  OpenCloning is a stateless FastAPI application
running in the Docker compose network with no authentication of its own —
the gateway is the auth/policy/audit boundary.

This module follows the orchestration template established by the eLabFTW
adapter (C08/C09):

    context token verification
        → identity resolution (mapped user required)
        → policy engine decision (tier 3 VALIDATION_DRY_RUN, read-only)
        → downstream client call (HTTP for production, stub for tests)
        → audit record with tool_args_hash

Key differences from the eLabFTW adapter:

    * OpenCloning has no API key — the HTTP client only needs a base_url.
    * All four V1 tools are tier 3 read-only (no approval tokens needed).
    * File-size limits and file-type allowlists are enforced at the adapter
      layer before any downstream call (OpenCloning itself has no file-size
      limit — see upstream TODO in read_from_file).
    * The context token's experiment_id is carried through as a context_ref
      for audit correlation but is not used in the downstream call —
      OpenCloning operations are computational, not experiment-scoped.

Tools exposed (all already in the C06 tool registry):

    opencloning.parse_sequence_file   — parse an uploaded sequence file
    opencloning.manual_sequence       — validate a manually typed sequence
    opencloning.oligo_hybridization  — compute oligo hybridization product
    opencloning.simulate_assembly     — simulate a cloning assembly
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

import requests

from lab_copilot_gateway.approval import ApprovalStore, compute_args_hash
from lab_copilot_gateway.artifact_bundle import (
    ArtifactBundleError,
    ArtifactBundleStore,
    get_artifact_bundle_store,
)
from lab_copilot_gateway.audit import AuditRecord, AuditStore
from lab_copilot_gateway.elabftw import (
    ElabftwClient,
    InvalidContextToken,
    UnmappedCaller,
    _merge_provenance_into_metadata,
    verify_context_token,
)
from lab_copilot_gateway.identity import MappedIdentity
from lab_copilot_gateway.policy import Decision, PolicyEngine, PolicyRequest, Tier


# --- configuration --------------------------------------------------------

# Maximum file size accepted by parse_sequence_file (1 MB).  OpenCloning
# itself has no limit (upstream TODO), so the gateway enforces one.
MAX_FILE_SIZE_BYTES = 1_048_576  # 1 MiB

# Allowed file extensions for parse_sequence_file.  Matches the
# SequenceFileFormat enum in OpenCloning's backend.
ALLOWED_FILE_EXTENSIONS: frozenset[str] = frozenset(
    {".gb", ".gbk", ".ape", ".fasta", ".fa", ".dna", ".embl"}
)

# Request timeout for downstream OpenCloning calls (seconds).  Assembly
# simulation can take a few seconds; 15s is generous without being unbounded.
DEFAULT_TIMEOUT_SECONDS = 15.0


# --- exceptions -----------------------------------------------------------


class OpenCloningAdapterError(Exception):
    """Base class for OpenCloning adapter errors.

    ``reason`` is a machine-readable code; ``message`` is human-readable.
    Mirrors ``ElabftwAdapterError`` so the /invoke endpoint can handle
    both uniformly.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message

    def to_dict(self) -> dict[str, str]:
        return {"reason": self.reason, "message": self.message}


class PolicyDenied(OpenCloningAdapterError):
    """Policy engine refused the call."""

    def __init__(self, decision: Decision) -> None:
        super().__init__(
            reason=f"policy_denied:{decision.reason}",
            message=f"policy engine denied call ({decision.reason})",
        )


class FileTooLarge(OpenCloningAdapterError):
    """Uploaded file exceeds the gateway's size limit."""

    def __init__(self, size: int, limit: int) -> None:
        super().__init__(
            reason="file_too_large",
            message=f"file size {size} bytes exceeds limit {limit} bytes",
        )
        self.size = size
        self.limit = limit


class DisallowedFileType(OpenCloningAdapterError):
    """File extension is not in the allowlist."""

    def __init__(self, extension: str) -> None:
        super().__init__(
            reason="disallowed_file_type",
            message=(
                f"file extension {extension!r} not allowed "
                f"(allowed: {sorted(ALLOWED_FILE_EXTENSIONS)})"
            ),
        )
        self.extension = extension


# --- downstream client ---------------------------------------------------


class OpenCloningClient(Protocol):
    """OpenCloning backend client surface the gateway adapter requires.

    Implementations: ``StubOpenCloningClient`` for tests;
    ``HttpOpenCloningClient`` for live calls.

    The OpenCloning API (https://github.com/manulera/OpenCloning) uses a
    {sources, sequences} response shape for all endpoints. Request bodies
    follow the pattern {"source": <SourceObject>, ...payload} where the
    source object carries the type and metadata.
    """

    def parse_sequence_file(
        self, file_content: bytes, file_format: str
    ) -> dict[str, Any]:
        """POST /read_from_file — parse an uploaded sequence file (multipart).

        Returns {sources: [...], sequences: [...]}.
        """
        ...

    def manual_sequence(self, sequence: str, circular: bool = False) -> dict[str, Any]:
        """POST /manually_typed — validate a manually typed sequence.

        Returns {sources: [...], sequences: [...]}.
        """
        ...

    def oligo_hybridization(
        self, forward_oligo: str, reverse_oligo: str, minimal_annealing: int = 20
    ) -> dict[str, Any]:
        """POST /oligonucleotide_hybridization — compute hybridization product.

        Returns {sources: [...], sequences: [...]}.
        """
        ...

    def simulate_assembly(
        self,
        sequences: list[dict[str, Any]],
        source: dict[str, Any],
    ) -> dict[str, Any]:
        """POST /gibson_assembly (or /ligation, etc.) — simulate assembly.

        ``sequences`` is the array of TextFileSequence objects from prior
        parse/PCR steps. ``source`` is the assembly source object (e.g.
        {"id": 0, "type": "GibsonAssemblySource", "input": [...]}).

        Returns {sources: [...], sequences: [...]}.
        """
        ...

    def call_endpoint(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST to any OpenCloning endpoint — generic escape hatch.

        For endpoints without a dedicated method (PCR, restriction digest,
        repository import, primer design, CRISPR, etc.).
        """
        ...


@dataclass
class StubOpenCloningClient:
    """In-memory ``OpenCloningClient`` for tests and offline dev.

    Pre-seed ``responses`` with known return values keyed by method name.
    Methods that aren't pre-seeded return a default empty-success response.
    Set ``error`` to simulate a downstream failure.
    """

    responses: dict[str, dict[str, Any]] = field(default_factory=dict)
    error: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def _record(self, method: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"method": method, **kwargs})
        if self.error is not None:
            raise self.error
        return dict(self.responses.get(method, {"ok": True}))

    def parse_sequence_file(
        self, file_content: bytes, file_format: str
    ) -> dict[str, Any]:
        return self._record(
            "parse_sequence_file",
            file_size=len(file_content),
            file_format=file_format,
        )

    def manual_sequence(self, sequence: str, circular: bool = False) -> dict[str, Any]:
        return self._record("manual_sequence", sequence=sequence, circular=circular)

    def oligo_hybridization(
        self, forward_oligo: str, reverse_oligo: str, minimal_annealing: int = 20
    ) -> dict[str, Any]:
        return self._record(
            "oligo_hybridization",
            forward_oligo=forward_oligo,
            reverse_oligo=reverse_oligo,
            minimal_annealing=minimal_annealing,
        )

    def simulate_assembly(
        self,
        sequences: list[dict[str, Any]],
        source: dict[str, Any],
    ) -> dict[str, Any]:
        return self._record(
            "simulate_assembly",
            sequence_count=len(sequences),
            source=source,
        )

    def call_endpoint(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        return self._record("call_endpoint", path=path, body_keys=list(body.keys()))


@dataclass
class HttpOpenCloningClient:
    """Live OpenCloning backend client.

    OpenCloning has no authentication — the client only needs a base_url.
    Uses ``requests.Session`` for connection reuse, matching the lab HTTP
    pattern (see AGENTS.md → "eLabFTW API — HTTP client patterns").
    """

    base_url: str
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    _session: Any = None

    def __post_init__(self) -> None:
        if not self.base_url:
            raise ValueError("HttpOpenCloningClient requires non-empty base_url")

    def _connect(self) -> Any:
        if self._session is None:
            import requests  # local import; not all deployments exercise HTTP

            s = requests.Session()
            s.verify = False  # internal service, self-signed or no TLS
            self._session = s
        return self._session

    def _post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}{path}"
        s = self._connect()
        resp = s.post(url, json=body, timeout=self.timeout_seconds)
        if not resp.ok:
            raise self._http_error(resp, path)
        return resp.json()

    def _post_multipart(
        self, path: str, files: dict[str, Any], data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}{path}"
        s = self._connect()
        resp = s.post(url, files=files, data=data or {}, timeout=self.timeout_seconds)
        if not resp.ok:
            raise self._http_error(resp, path)
        return resp.json()

    @staticmethod
    def _http_error(resp: Any, path: str) -> "requests.HTTPError":
        """Build an HTTPError that includes the response body so the LLM
        sees the downstream error message (e.g. 'wrong sequence accession')
        and can correct its next call."""
        import requests as _requests

        try:
            detail = resp.json()
            body_text = str(detail.get("detail", detail))
        except Exception:
            body_text = resp.text[:200]
        return _requests.HTTPError(
            f"{resp.status_code} {resp.reason} for {path}: {body_text}",
            response=resp,
        )

    def parse_sequence_file(
        self, file_content: bytes, file_format: str
    ) -> dict[str, Any]:
        # OpenCloning's read_from_file expects a multipart upload.
        # The file_format is inferred from the extension by the backend.
        filename = _format_to_filename(file_format)
        files = {"file": (filename, file_content)}
        return self._post_multipart("/read_from_file", files=files)

    def manual_sequence(self, sequence: str, circular: bool = False) -> dict[str, Any]:
        # OpenCloning API expects {"source": ManuallyTypedSource, "sequence": ManuallyTypedSequence}
        return self._post_json(
            "/manually_typed",
            {
                "source": {
                    "id": 0,
                    "type": "ManuallyTypedSource",
                },
                "sequence": {
                    "id": 0,
                    "type": "ManuallyTypedSequence",
                    "sequence": sequence,
                    "circular": circular,
                },
            },
        )

    def oligo_hybridization(
        self, forward_oligo: str, reverse_oligo: str, minimal_annealing: int = 20
    ) -> dict[str, Any]:
        # OpenCloning API expects {"source": OligoHybridizationSource, "primers": [forward, reverse]}
        return self._post_json(
            "/oligonucleotide_hybridization",
            {
                "source": {
                    "id": 0,
                    "type": "OligoHybridizationSource",
                },
                "primers": [
                    {
                        "id": 0,
                        "type": "Primer",
                        "sequence": forward_oligo,
                        "name": "forward",
                    },
                    {
                        "id": 1,
                        "type": "Primer",
                        "sequence": reverse_oligo,
                        "name": "reverse",
                    },
                ],
            },
        )

    def simulate_assembly(
        self,
        sequences: list[dict[str, Any]],
        source: dict[str, Any],
    ) -> dict[str, Any]:
        # The assembly endpoint is determined by source["type"].
        # Map source type → endpoint path.
        source_type = source.get("type", "")
        type_to_path = {
            "GibsonAssemblySource": "/gibson_assembly",
            "LigationSource": "/ligation",
            "RestrictionAndLigationSource": "/restriction_and_ligation",
            "OverlapExtensionPCRLigationSource": "/gibson_assembly",
            "InFusionSource": "/gibson_assembly",
            "InVivoAssemblySource": "/gibson_assembly",
            "HomologousRecombinationSource": "/homologous_recombination",
            "CRISPRSource": "/crispr",
            "CreLoxRecombinationSource": "/cre_lox_recombination",
            "GatewaySource": "/gateway",
            "RecombinaseSource": "/recombinase",
        }
        path = type_to_path.get(source_type, "/gibson_assembly")
        return self._post_json(
            path,
            {"sequences": sequences, "source": source},
        )

    def call_endpoint(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST to any OpenCloning endpoint — generic escape hatch.

        Use this for endpoints that don't have a dedicated method.
        The path should start with '/' (e.g. '/pcr', '/repository_id/addgene').
        """
        return self._post_json(path, body)

    def get(self, path: str) -> dict[str, Any]:
        """GET any OpenCloning endpoint (e.g. /version, /restriction_enzyme_list)."""
        url = f"{self.base_url.rstrip('/')}{path}"
        s = self._connect()
        resp = s.get(url, timeout=self.timeout_seconds)
        if not resp.ok:
            raise self._http_error(resp, path)
        return resp.json()


def _inject_one_sequence(seq: Any, store: dict[str, dict[str, Any]]) -> None:
    """Inject file_content into a single sequence dict if needed."""
    if not isinstance(seq, dict):
        return
    fc = seq.get("file_content")
    if fc is None or (isinstance(fc, dict) and fc.get("redacted")):
        seq_id = str(seq.get("id", ""))
        stored = store.get(seq_id)
        if stored and "file_content" in stored:
            seq["file_content"] = stored["file_content"]
        elif fc is not None:
            del seq["file_content"]


def _inject_file_content_from_store(
    body: dict[str, Any], store: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Inject file_content from the sequence store into sequence objects.

    When the LLM passes sequences to opencloning.call, it omits
    file_content (because the gateway redacts it in responses). But
    OpenCloning needs file_content to process the sequences. This function
    looks up each sequence by its id in the store and injects the stored
    file_content if the sequence doesn't already have it.

    Also strips any redacted file_content placeholder the LLM might have
    included (e.g. {redacted: true, ...}).

    Handles sequences in multiple locations in the request body:
    - body["sequences"] (top-level array, used by /pcr, /gibson_assembly, etc.)
    - body["pcr_templates"][*]["sequence"] (used by /primer_design/*)
    - body["pcr_template"]["sequence"] (used by /primer_design/simple_pair)
    - body["homologous_recombination_target"]["sequence"] (used by /primer_design/hr)
    - body["sequence"] (used by /align_sanger, /primer_details)
    """

    def _inject_one(seq: Any) -> None:
        _inject_one_sequence(seq, store)

    # Top-level sequences array (used by /pcr, /gibson_assembly, /restriction, etc.)
    sequences = body.get("sequences")
    if isinstance(sequences, list):
        for seq in sequences:
            _inject_one(seq)

    # Primer design: pcr_templates is a list of {sequence, location, ...}
    pcr_templates = body.get("pcr_templates")
    if isinstance(pcr_templates, list):
        for template in pcr_templates:
            if isinstance(template, dict):
                _inject_one(template.get("sequence"))

    # Primer design (single): pcr_template is {sequence, location, ...}
    pcr_template = body.get("pcr_template")
    if isinstance(pcr_template, dict):
        _inject_one(pcr_template.get("sequence"))

    # Homologous recombination primer design: homologous_recombination_target
    hr_target = body.get("homologous_recombination_target")
    if isinstance(hr_target, dict):
        _inject_one(hr_target.get("sequence"))

    # Align sanger: sequence is a single sequence object (not in an array)
    single_seq = body.get("sequence")
    if isinstance(single_seq, dict):
        _inject_one(single_seq)

    # Rename: the body IS the sequence object itself (no wrapper key).
    # If the body has type=TextFileSequence and no file_content, inject it.
    if isinstance(body, dict) and body.get("type") == "TextFileSequence":
        _inject_one(body)

    return body


def _rewrite_opencloning_result_ids(
    result: dict[str, Any], store: dict[str, dict[str, Any]], next_id: int
) -> int:
    """Assign gateway-unique ids to returned sequences and their sources.

    OpenCloning is stateless and many endpoints return sequence/source ids that
    start back at 0. The chat loop needs stable gateway ids, and the history JSON
    must keep source ids aligned with their output sequence ids so OpenCloning's
    frontend can reconstruct the graph.
    """
    next_id, id_map = _rewrite_sequence_ids(result, store, next_id)
    if id_map:
        _rewrite_source_ids(result, id_map)
    return next_id


def _rewrite_sequence_ids(
    result: dict[str, Any],
    store: dict[str, dict[str, Any]],
    next_id: int,
) -> tuple[int, dict[str, int]]:
    """Assign gateway-unique ids to sequences and store them."""
    id_map: dict[str, int] = {}
    for seq in result.get("sequences", []):
        if not isinstance(seq, dict) or "file_content" not in seq:
            continue
        old_id = str(seq.get("id", ""))
        if not old_id:
            continue
        next_id += 1
        new_id = next_id
        id_map[old_id] = new_id
        stored = dict(seq)
        stored["original_id"] = old_id
        store[str(new_id)] = stored
        seq["id"] = new_id
    return next_id, id_map


def _rewrite_source_ids(result: dict[str, Any], id_map: dict[str, int]) -> None:
    """Rewrite source ids and input sequence references using id_map."""
    for src in result.get("sources", []):
        if not isinstance(src, dict):
            continue
        src_id = str(src.get("id", ""))
        if src_id in id_map:
            src["id"] = id_map[src_id]
        _rewrite_source_inputs(src, id_map)


def _rewrite_source_inputs(src: dict[str, Any], id_map: dict[str, int]) -> None:
    """Rewrite sequence references in a source's input list."""
    inputs = src.get("input")
    if not isinstance(inputs, list):
        return
    for item in inputs:
        if not isinstance(item, dict):
            continue
        seq_ref = str(item.get("sequence", ""))
        if seq_ref in id_map:
            item["sequence"] = id_map[seq_ref]


def _strip_self_references(sources: list[dict[str, Any]]) -> None:
    """Convert sources with self-referencing inputs to DatabaseSource.

    OpenCloning's /pcr endpoint returns sources whose inputs include the PCR
    product itself (describing primer-binding regions as fragments of the
    product). This creates a cycle (source → output sequence → source input)
    that causes /validate to reject the history JSON with 422 and triggers
    infinite recursion in the frontend's graph traversal.

    Instead of stripping individual self-referencing inputs (which leaves PCR
    sources with fewer inputs than the frontend expects), we convert the
    entire source to a DatabaseSource — the same approach the OpenCloning
    frontend uses when loading history files with missing ancestor sequences
    (see useLoadDatabaseFile.js). DatabaseSource has no inputs, so the
    frontend renders it as a leaf node without trying to traverse inputs.
    """
    for i, src in enumerate(sources):
        if not isinstance(src, dict):
            continue
        src_id = src.get("id")
        inputs = src.get("input")
        if not isinstance(inputs, list) or src_id is None:
            continue
        has_self_ref = any(
            isinstance(inp, dict) and inp.get("sequence") == src_id for inp in inputs
        )
        if has_self_ref:
            sources[i] = {
                "id": src_id,
                "type": "DatabaseSource",
                "input": [],
                "output_name": src.get("output_name"),
                # database_id is required and must be an integer for
                # DatabaseSource. Use src_id as a placeholder — the
                # sequence is already in the history JSON, so the
                # frontend uses the local copy.
                "database_id": src_id,
            }


def _strip_sequence_names(sequences: list[dict[str, Any]]) -> None:
    """Strip the 'name' field from each sequence dict in-place.

    TextFileSequence has ``extra='forbid'`` in its schema, so passing any
    unrecognised field (including ``name``, which is already encoded in the
    GenBank LOCUS line of ``file_content``) causes ``/validate`` to return 422.
    """
    for seq in sequences:
        seq.pop("name", None)


def _generic_sequence_name(name: Any) -> bool:
    value = str(name or "").strip().lower()
    return value in {"", "name", "sequence", "untitled", "untitled_sequence"}


def _enrich_strategy_names(result: dict[str, Any]) -> None:
    """Copy source output_name into generic/missing sequence names."""
    output_names = {
        src.get("id"): src.get("output_name")
        for src in result.get("sources", [])
        if isinstance(src, dict) and src.get("output_name")
    }
    for seq in result.get("sequences", []):
        if not isinstance(seq, dict) or not _generic_sequence_name(seq.get("name")):
            continue
        output_name = output_names.get(seq.get("id"))
        if output_name:
            seq["name"] = output_name


def _format_to_filename(file_format: str) -> str:
    """Map a file_format identifier to a dummy filename with the right
    extension, so OpenCloning's backend can infer the parser to use."""
    ext_map = {
        "genbank": ".gb",
        "fasta": ".fasta",
        "snapgene": ".dna",
        "embl": ".embl",
    }
    ext = ext_map.get(file_format, ".gb")
    return f"sequence{ext}"


def _default_client_from_env() -> OpenCloningClient | None:
    """Construct the live HTTP client if env config is present.

    Returns None if base_url is unset — the adapter then refuses calls
    with ``opencloning_not_configured`` at call time.
    """
    base_url = os.getenv("LAB_COPILOT_OPENCLONING_BASE_URL", "")
    if not base_url:
        return None
    timeout = float(
        os.getenv("LAB_COPILOT_OPENCLONING_TIMEOUT", str(DEFAULT_TIMEOUT_SECONDS))
    )
    return HttpOpenCloningClient(
        base_url=base_url,
        timeout_seconds=timeout,
    )


# --- result ---------------------------------------------------------------


@dataclass(frozen=True)
class OpenCloningResult:
    """Outcome of a successful OpenCloning adapter call."""

    tool_name: str
    result: dict[str, Any]
    audit_action_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "result": self.result,
            "audit_action_id": self.audit_action_id,
        }


# --- C52: writeback package data structures --------------------------------


@dataclass
class WritebackStepResult:
    """Result of a single writeback step (upload, append, patch).

    Each step in the expanded writeback package produces one of these so
    the caller can report exactly what succeeded, what failed, and what
    upload IDs were assigned.
    """

    step_name: str  # "upload_genbank", "upload_history", etc.
    status: str  # "success" | "failed" | "skipped"
    upload_id: int | None = None
    error: str | None = None


@dataclass
class WritebackResult:
    """Aggregate result of the expanded writeback package.

    ``status`` is "success" only if all required steps succeeded.
    "partial" means some steps succeeded but a non-blocking step failed
    (e.g. metadata patch). "failed" means a blocking step failed (e.g.
    audit report upload) or the primary artifact upload failed.
    """

    status: str  # "success" | "partial" | "failed"
    upload_ids: dict[str, int]  # {"genbank": 123, "history": 124, ...}
    notebook_appended: bool
    metadata_patched: bool
    steps: list[WritebackStepResult]
    rollback_instructions: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "upload_ids": dict(self.upload_ids),
            "notebook_appended": self.notebook_appended,
            "metadata_patched": self.metadata_patched,
            "steps": [
                {
                    "step_name": s.step_name,
                    "status": s.status,
                    "upload_id": s.upload_id,
                    "error": s.error,
                }
                for s in self.steps
            ],
            "rollback_instructions": self.rollback_instructions,
            "warnings": list(self.warnings),
        }


# --- C52: writeback package hash -------------------------------------------


def _compute_writeback_package_hash(
    artifact_manifest: dict[str, Any],
    final_construct_hash: str,
    history_json_hash: str,
    audit_report_hash: str,
    notebook_summary_hash: str,
    selected_intermediate_attachments: list[dict[str, Any]],
    protocol_source_refs: list[str],
    validation_bundle_version: str,
) -> str:
    """Compute a hash that binds approval to the exact writeback package.

    The hash covers every component of the writeback package: the final
    GenBank construct, the history JSON, the audit report, the notebook
    summary, the selected intermediate attachments, protocol source
    references, and the validation bundle version. If any component
    changes, the hash changes and the old approval is invalid.

    This uses the same canonical-JSON + SHA-256 approach as
    ``compute_args_hash`` so it is byte-stable across runs.
    """
    return compute_args_hash(
        {
            "artifact_manifest": artifact_manifest,
            "final_construct_hash": final_construct_hash,
            "history_json_hash": history_json_hash,
            "audit_report_hash": audit_report_hash,
            "notebook_summary_hash": notebook_summary_hash,
            "selected_intermediate_attachments": selected_intermediate_attachments,
            "protocol_source_refs": protocol_source_refs,
            "validation_bundle_version": validation_bundle_version,
        }
    )


def _wrap_amendment_html(existing_body: str, amendment_html: str) -> str:
    """Wrap an amendment in an HTML section marker.

    Mirrors ``ElabftwWriteAdapter._wrap_amendment`` so the OpenCloning
    adapter can append notebook entries directly via the eLabFTW client
    without going through the write adapter. The wrapper is a styled
    ``<section>`` so eLabFTW viewers can visually identify AI-generated
    amendments.
    """
    if not amendment_html:
        return existing_body
    section = (
        '<section data-source="lab-copilot-gateway" '
        'data-amendment="1">'
        f"{amendment_html}"
        "</section>"
    )
    if not existing_body:
        return section
    return existing_body.rstrip() + "\n\n" + section


def _build_rollback_instructions(
    upload_ids: dict[str, int],
    experiment_id: int,
) -> str:
    """Build actionable rollback instructions for a partial writeback.

    Lists the uploaded attachment IDs that should be manually deleted from
    the experiment if the user wants to undo the partial writeback.
    """
    uploaded = [
        f"{name} (upload_id={uid})"
        for name, uid in upload_ids.items()
        if uid is not None
    ]
    if not uploaded:
        return "No attachments were uploaded — no rollback needed."
    return (
        f"To undo this partial writeback on experiment {experiment_id}, "
        f"manually delete the following attachments in eLabFTW: "
        + ", ".join(uploaded)
        + ". Then remove the lab-copilot amendment section from the "
        "experiment body if one was appended."
    )


def _step_flags(steps: list[WritebackStepResult]) -> dict[str, bool]:
    """Extract success/failure flags for each writeback step name.

    Returns a dict with keys like ``notebook_appended``, ``genbank_failed``,
    etc. — one per step name × status combination the assembler checks.
    """
    result: dict[str, bool] = {
        "notebook_appended": False,
        "metadata_patched": False,
        "genbank_failed": False,
        "audit_failed": False,
        "notebook_failed": False,
        "metadata_failed": False,
    }
    for s in steps:
        if s.step_name == "append_notebook" and s.status == "success":
            result["notebook_appended"] = True
        elif s.step_name == "append_notebook" and s.status == "failed":
            result["notebook_failed"] = True
        elif s.step_name == "patch_metadata" and s.status == "success":
            result["metadata_patched"] = True
        elif s.step_name == "patch_metadata" and s.status == "failed":
            result["metadata_failed"] = True
        elif s.step_name == "upload_genbank" and s.status == "failed":
            result["genbank_failed"] = True
        elif s.step_name == "upload_audit_report" and s.status == "failed":
            result["audit_failed"] = True
    return result


# --- adapter --------------------------------------------------------------


# Tool names from the C06 registry.
TOOL_PARSE = "opencloning.parse_sequence_file"
TOOL_MANUAL = "opencloning.manual_sequence"
TOOL_OLIGO = "opencloning.oligo_hybridization"
TOOL_ASSEMBLY = "opencloning.simulate_assembly"
TOOL_WRITEBACK = "opencloning.writeback_artifact"
TOOL_CALL = "opencloning.call"
TOOL_TIER = Tier.VALIDATION_DRY_RUN
TOOL_TIER_WRITE = Tier.BOUNDED_WRITES


def _parse_artifact_reference(
    approval_args: dict[str, Any],
    artifact_content: str | None,
) -> dict[str, Any] | None:
    """Parse artifact reference from approval_args, validating required fields.

    Raises OpenCloningAdapterError if neither content nor valid reference is found.
    Returns None if artifact_content is not None (legacy inline path).
    Returns the reference dict if artifact_id is present and valid.
    """
    if artifact_content is not None:
        return None
    if approval_args.get("artifact_id"):
        artifact_reference = {
            "artifact_id": approval_args.get("artifact_id"),
            "plan_id": approval_args.get("plan_id"),
            "plan_hash": approval_args.get("plan_hash"),
            "expected_sha256": approval_args.get("artifact_sha256"),
            "expected_size_bytes": approval_args.get("artifact_size_bytes"),
        }
        required_fields = {
            "artifact_id",
            "plan_id",
            "plan_hash",
            "expected_sha256",
            "expected_size_bytes",
        }
        missing = [
            key
            for key, value in artifact_reference.items()
            if key in required_fields and (value is None or value == "")
        ]
        if missing:
            raise OpenCloningAdapterError(
                reason="missing_artifact_reference",
                message=f"artifact bundle reference missing required fields: {missing}",
            )
        return artifact_reference
    raise OpenCloningAdapterError(
        reason="missing_artifact_payload",
        message="writeback requires artifact_content or a complete artifact bundle reference",
    )


@dataclass
class OpenCloningAdapter:
    """Orchestrates OpenCloning calls through the gateway-enforced path.

    Dependencies are injected so tests can swap the policy engine, audit
    store, and downstream client individually.  In production, the
    module-level ``get_opencloning_adapter()`` wires all the singletons.
    """

    policy_engine: PolicyEngine
    audit_store: AuditStore
    client: OpenCloningClient | None
    elabftw_client: ElabftwClient | None = None
    approval_store: ApprovalStore | None = None
    artifact_store: ArtifactBundleStore | None = None
    action_id_prefix: str = "opencloning"
    # Process-local store of full sequence objects (with file_content) from
    # prior OpenCloning responses. Used to inject file_content into
    # opencloning.call requests when the LLM passes redacted sequences.
    _sequence_store: dict[str, dict[str, Any]] = field(default_factory=dict)
    _seq_counter: int = 0
    # Cloning strategy accumulator: tracks all sequences, sources, and
    # primers across the agentic loop so they can be assembled into a
    # cloning history JSON during writeback. This matches what the
    # OpenCloning frontend saves to eLabFTW as {title}_history.json.
    _strategy_sequences: list[dict[str, Any]] = field(default_factory=list)
    _strategy_sources: list[dict[str, Any]] = field(default_factory=list)
    _strategy_primers: list[dict[str, Any]] = field(default_factory=list)
    _primer_counter: int = 0

    # --- public API: four C16 tool entrypoints --------------------------

    def parse_sequence_file(
        self,
        *,
        context_token: str,
        file_content: str,
        file_format: str,
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None = None,
        request_id: str | None = None,
        keycloak_subject: str | None = None,
        librechat_user_id: str | None = None,
        provider: str | None = None,
        model_id: str | None = None,
    ) -> OpenCloningResult:
        """Parse an uploaded sequence file via OpenCloning.

        ``file_content`` is the raw file content as a string (base64-decoded
        by the /invoke endpoint if needed).  ``file_format`` is one of
        ``genbank``, ``fasta``, ``snapgene``, ``embl``.

        Enforces file-size limit and file-type allowlist before any
        downstream call.
        """
        # Decode file content to bytes for size check and downstream call.
        raw_bytes = (
            file_content.encode("utf-8")
            if isinstance(file_content, str)
            else file_content
        )

        # File-size limit (enforced before any downstream call).
        if len(raw_bytes) > MAX_FILE_SIZE_BYTES:
            self._audit(
                tool_name=TOOL_PARSE,
                policy_decision="deny",
                reason="file_too_large",
                mapped_identity=mapped_identity,
                experiment_id=None,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=compute_args_hash(
                    {"file_format": file_format, "file_size": len(raw_bytes)}
                ),
                error={
                    "code": "FILE_TOO_LARGE",
                    "size": len(raw_bytes),
                    "limit": MAX_FILE_SIZE_BYTES,
                },
            )
            raise FileTooLarge(len(raw_bytes), MAX_FILE_SIZE_BYTES)

        # File-type allowlist.
        ext = _infer_extension(file_format)
        if ext not in ALLOWED_FILE_EXTENSIONS:
            self._audit(
                tool_name=TOOL_PARSE,
                policy_decision="deny",
                reason="disallowed_file_type",
                mapped_identity=mapped_identity,
                experiment_id=None,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=compute_args_hash(
                    {"file_format": file_format, "file_size": len(raw_bytes)}
                ),
                error={
                    "code": "DISALLOWED_FILE_TYPE",
                    "extension": ext,
                    "allowed": sorted(ALLOWED_FILE_EXTENSIONS),
                },
            )
            raise DisallowedFileType(ext)

        return self._execute(
            tool_name=TOOL_PARSE,
            context_token=context_token,
            mapped_identity=mapped_identity,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            args_for_hash={"file_format": file_format, "file_size": len(raw_bytes)},
            api_call_summary={"method": "POST", "path": "/read_from_file"},
            exec_fn=lambda _client: _client.parse_sequence_file(raw_bytes, file_format),
        )

    def manual_sequence(
        self,
        *,
        context_token: str,
        sequence: str,
        circular: bool = False,
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None = None,
        request_id: str | None = None,
        keycloak_subject: str | None = None,
        librechat_user_id: str | None = None,
        provider: str | None = None,
        model_id: str | None = None,
    ) -> OpenCloningResult:
        """Validate a manually typed DNA sequence via OpenCloning."""
        return self._execute(
            tool_name=TOOL_MANUAL,
            context_token=context_token,
            mapped_identity=mapped_identity,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            args_for_hash={"sequence_length": len(sequence), "circular": circular},
            api_call_summary={"method": "POST", "path": "/manually_typed"},
            exec_fn=lambda _client: _client.manual_sequence(sequence, circular),
        )

    def oligo_hybridization(
        self,
        *,
        context_token: str,
        forward_oligo: str,
        reverse_oligo: str,
        minimal_annealing: int = 20,
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None = None,
        request_id: str | None = None,
        keycloak_subject: str | None = None,
        librechat_user_id: str | None = None,
        provider: str | None = None,
        model_id: str | None = None,
    ) -> OpenCloningResult:
        """Compute the product of oligo hybridization via OpenCloning."""
        return self._execute(
            tool_name=TOOL_OLIGO,
            context_token=context_token,
            mapped_identity=mapped_identity,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            args_for_hash={
                "forward_length": len(forward_oligo),
                "reverse_length": len(reverse_oligo),
                "minimal_annealing": minimal_annealing,
            },
            api_call_summary={
                "method": "POST",
                "path": "/oligonucleotide_hybridization",
            },
            exec_fn=lambda _client: _client.oligo_hybridization(
                forward_oligo, reverse_oligo, minimal_annealing
            ),
        )

    def simulate_assembly(
        self,
        *,
        context_token: str,
        sequences: list[dict[str, Any]],
        source: dict[str, Any],
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None = None,
        request_id: str | None = None,
        keycloak_subject: str | None = None,
        librechat_user_id: str | None = None,
        provider: str | None = None,
        model_id: str | None = None,
    ) -> OpenCloningResult:
        """Simulate a cloning assembly via OpenCloning.

        ``sequences`` is the array of TextFileSequence objects from prior
        parse/PCR steps. ``source`` is the assembly source object (e.g.
        {"id": 0, "type": "GibsonAssemblySource", "input": [...]}).

        Injects stored file_content into any sequences that are missing it
        (same as ``call_endpoint``), so the OpenCloning backend receives
        complete GenBank bytes instead of redacted stubs.
        """
        body = _inject_file_content_from_store(
            {"sequences": sequences, "source": source}, self._sequence_store
        )
        sequences = body.get("sequences", sequences)
        source = body.get("source", source)
        source_type = source.get("type", "GibsonAssemblySource")
        return self._execute(
            tool_name=TOOL_ASSEMBLY,
            context_token=context_token,
            mapped_identity=mapped_identity,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            args_for_hash={
                "sequence_count": len(sequences),
                "source_type": source_type,
            },
            api_call_summary={"method": "POST", "path": f"/{source_type}"},
            exec_fn=lambda _client: _client.simulate_assembly(sequences, source),
        )

    # --- Generic endpoint call (covers all OpenCloning API operations) -----

    def call_endpoint(
        self,
        *,
        context_token: str,
        endpoint: str,
        body: dict[str, Any],
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None = None,
        request_id: str | None = None,
        keycloak_subject: str | None = None,
        librechat_user_id: str | None = None,
        provider: str | None = None,
        model_id: str | None = None,
    ) -> OpenCloningResult:
        """Call any OpenCloning API endpoint — generic escape hatch.

        This covers all operations without dedicated methods:
        - Repository imports: /repository_id/addgene, /repository_id/genbank, etc.
        - PCR: /pcr
        - Restriction digest: /restriction
        - Restriction/Ligation (Golden Gate): /restriction_and_ligation
        - CRISPR: /crispr
        - Homologous recombination: /homologous_recombination
        - Cre/Lox: /cre_lox_recombination
        - Gateway: /gateway
        - Recombinase: /recombinase
        - Polymerase extension: /polymerase_extension
        - Reverse complement: /reverse_complement
        - Primer design: /primer_design/gibson_assembly, etc.
        - Primer details: /primer_details
        - Validate: /validate, /validate_syntax
        - Sanger alignment: /align_sanger
        - Rename: /rename_sequence
        - Batch cloning: /batch_cloning/*
        """
        # Inject file_content from the sequence store for any sequences
        # that are missing it. The LLM sees redacted sequences (file_content
        # is replaced with {redacted: true, ...}) but OpenCloning needs the
        # raw file_content to process the sequences. The gateway stores the
        # full sequences from prior OpenCloning responses and injects them
        # here so the LLM never needs to handle raw sequence bytes.
        body = _inject_file_content_from_store(body, self._sequence_store)

        result = self._execute(
            tool_name=TOOL_CALL,
            context_token=context_token,
            mapped_identity=mapped_identity,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            args_for_hash={"endpoint": endpoint, "body_keys": sorted(body.keys())},
            api_call_summary={"method": "POST", "path": endpoint},
            exec_fn=lambda _client: _client.call_endpoint(endpoint, body),
        )

        # Feature-loss detection for PCR (single-template operations only).
        # Assembly operations legitimately change feature counts, so we
        # only warn for PCR where feature loss indicates a degraded template.
        if endpoint == "/pcr":
            from lab_copilot_gateway.opencloning_features import detect_feature_loss

            warning = detect_feature_loss(endpoint, body, result.result)
            if warning:
                result.result["feature_loss_warning"] = warning

        return result

    # --- C17a: capture preview refs ---------------------------------------

    def capture_preview_refs(
        self,
        result: dict[str, Any],
        run_id: str,
        step_id: str,
        context_token_hash: str,
    ) -> dict[int, str]:
        """Store sequence file_content in the preview store.

        Iterates ``result["sequences"]`` and stores each sequence with
        ``file_content`` in the short-lived preview store.  Returns a
        mapping ``{sequence_id: preview_ref}`` that callers can use to
        annotate manifests or redacted responses.

        Returns an empty dict if the result has no sequences with
        downloadable content.
        """
        from lab_copilot_gateway.opencloning_previews import get_preview_store

        store = get_preview_store()
        preview_map: dict[int, str] = {}

        sequences = result.get("sequences")
        if not isinstance(sequences, list):
            return preview_map

        for seq in sequences:
            if not isinstance(seq, dict):
                continue
            file_content = seq.get("file_content")
            if not isinstance(file_content, str) or not file_content:
                continue

            seq_id = seq.get("id")
            if seq_id is None:
                continue

            file_format = str(seq.get("sequence_file_format") or "genbank").lower()
            is_circular = seq.get("circular")
            is_circular_val: bool | None = (
                is_circular if isinstance(is_circular, bool) else None
            )

            preview_ref = store.store(
                run_id=run_id,
                step_id=step_id,
                context_token_hash=context_token_hash,
                sequence_id=int(seq_id) if seq_id else None,
                file_format=file_format,
                file_content=file_content,
                is_circular=is_circular_val,
            )

            preview_map[int(seq_id)] = preview_ref

        return preview_map

    # --- C17c: deterministic validation bundle ----------------------------

    def build_validation_bundle(
        self,
        *,
        run_id: str,
        approval_args: dict[str, Any],
        step_manifests: list[dict[str, Any]] | None = None,
        protocol_result: dict[str, Any] | None = None,
        preview_refs_count: int = 0,
        notebook_draft_generated: bool = False,
        audit_draft_generated: bool = False,
    ) -> dict[str, Any]:
        """Build a deterministic validation bundle from accumulated state.

        The bundle is the gate between "simulation done" and "user can approve
        writeback."  It is built from captured facts (strategy sequences,
        sources, primers, artifact reference), not from LLM output, so the LLM
        cannot influence validation results.

        Returns the bundle as a JSON-serializable dict (schema
        ``opencloning_validation_bundle_v1``) with ``is_approved`` set to
        ``False`` if any blocker check failed.
        """
        from lab_copilot_gateway.opencloning_validation import (
            build_validation_bundle as _build_bundle,
        )

        bundle = _build_bundle(
            run_id=run_id,
            sequences=list(self._strategy_sequences),
            sources=list(self._strategy_sources),
            primers=list(self._strategy_primers),
            artifacts=approval_args,
            step_manifests=list(step_manifests or []),
            protocol_result=protocol_result,
            preview_refs_count=preview_refs_count,
            notebook_draft_generated=notebook_draft_generated,
            audit_draft_generated=audit_draft_generated,
        )
        return bundle.to_dict()

    # --- C17b: writeback artifact to eLabFTW ------------------------------

    def _compute_package_hash_for_writeback(
        self,
        *,
        artifact_bytes: bytes,
        artifact_filename: str,
        approval_args: dict[str, Any],
    ) -> str:
        """Compute the writeback package hash from actual content.

        Renders the history JSON and reports from accumulated strategy
        state so the hash binds to the exact package that will be uploaded.
        The orchestrator computes the same hash before requesting approval
        and includes it in ``approval_args["writeback_package_hash"]``.
        """
        history_data = self._build_history_json(artifact_filename=artifact_filename)
        history_json_bytes = history_data[0] if history_data else b""
        reports = self._render_writeback_reports(
            upload_id=None,
            history_upload_id=None,
            approval_args=approval_args,
        )
        notebook_html = reports.get("notebook_html") or ""
        audit_markdown = reports.get("audit_markdown") or ""
        final_construct_hash = hashlib.sha256(artifact_bytes).hexdigest()
        history_json_hash = hashlib.sha256(history_json_bytes).hexdigest()
        audit_report_hash = hashlib.sha256(audit_markdown.encode("utf-8")).hexdigest()
        notebook_summary_hash = hashlib.sha256(
            notebook_html.encode("utf-8")
        ).hexdigest()
        artifact_manifest = {
            "artifact_filename": artifact_filename,
            "artifact_size": len(artifact_bytes),
        }
        selected_ids = approval_args.get("selected_intermediate_ids") or []
        selected_intermediate_attachments = [{"id": sid} for sid in selected_ids]
        protocol_source_refs = list(approval_args.get("protocol_source_refs") or [])
        return _compute_writeback_package_hash(
            artifact_manifest=artifact_manifest,
            final_construct_hash=final_construct_hash,
            history_json_hash=history_json_hash,
            audit_report_hash=audit_report_hash,
            notebook_summary_hash=notebook_summary_hash,
            selected_intermediate_attachments=selected_intermediate_attachments,
            protocol_source_refs=protocol_source_refs,
            validation_bundle_version="opencloning_validation_bundle_v1",
        )

    def writeback_artifact(
        self,
        *,
        context_token: str,
        approval_id: str,
        approval_args: dict[str, Any],
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None = None,
        request_id: str | None = None,
        keycloak_subject: str | None = None,
        librechat_user_id: str | None = None,
        provider: str | None = None,
        model_id: str | None = None,
        plan_approval_id: str | None = None,
    ) -> OpenCloningResult:
        """Attach an approved OpenCloning design artifact to the user's experiment.

        This is the ``opencloning.writeback_artifact`` tool (C17).  It bridges
        OpenCloning (which produced the artifact) and eLabFTW (where the
        experiment lives).  The artifact content and filename come from
        ``approval_args`` — they are hash-bound by the approval token so the
        user approved exactly what gets uploaded.

        The tool:
            1. Verifies the context token + identity (same as C16 reads).
            2. Checks policy (tier 4 BOUNDED_WRITES, requires approval).
            3. Consumes the approval token (single-use, args-hash-bound).
            4. Uploads the artifact as an attachment to the experiment via
               the eLabFTW client.
            5. Writes provenance (audit_action_id) to the experiment metadata.
            6. Records an audit row.

        The eLabFTW client must be configured (``elabftw_client`` field).
        The OpenCloning client is NOT needed for this tool.  V1 accepts either
        legacy inline ``artifact_content`` or a bundle reference
        (``artifact_id`` + ``plan_id`` + ``plan_hash`` + expected hash/size).
        """
        artifact_filename = approval_args.get("artifact_filename", "construct.gb")
        artifact_content = approval_args.get("artifact_content")
        artifact_comment = approval_args.get(
            "artifact_comment", "OpenCloning design artifact"
        )
        artifact_reference = _parse_artifact_reference(approval_args, artifact_content)

        # The artifact content is the hash-bound payload.  Convert to bytes
        # for the eLabFTW upload_attachment call.
        artifact_bytes = (
            b""
            if artifact_reference
            else (
                artifact_content.encode("utf-8")
                if isinstance(artifact_content, str)
                else artifact_content or b""
            )
        )

        # Enforce file-size limit on the artifact too (defense in depth).
        if not artifact_reference and len(artifact_bytes) > MAX_FILE_SIZE_BYTES:
            self._audit(
                tool_name=TOOL_WRITEBACK,
                policy_decision="deny",
                reason="file_too_large",
                mapped_identity=mapped_identity,
                experiment_id=None,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=compute_args_hash(
                    {
                        "artifact_filename": artifact_filename,
                        "artifact_size": len(artifact_bytes),
                    }
                ),
                error={
                    "code": "FILE_TOO_LARGE",
                    "size": len(artifact_bytes),
                    "limit": MAX_FILE_SIZE_BYTES,
                },
            )
            raise FileTooLarge(len(artifact_bytes), MAX_FILE_SIZE_BYTES)

        # Build the deterministic validation bundle from accumulated cloning
        # state BEFORE the writeback upload.  The bundle is the gate between
        # "simulation done" and "user can approve writeback" — it is built from
        # captured facts (strategy sequences/sources/primers + artifact ref),
        # not from LLM output.  Included in the result so the orchestrator/
        # widget can display validation status on the approval card.
        validation_bundle = self.build_validation_bundle(
            run_id=request_id or f"writeback-{conversation_id or 'unknown'}",
            approval_args=approval_args,
        )

        # C52: compute the writeback package hash from actual content. If
        # approval_args contains writeback_package_hash, verify it matches
        # BEFORE consuming the approval (fail closed on mismatch — the
        # strategy state changed between approval and writeback).
        package_hash = self._compute_package_hash_for_writeback(
            artifact_bytes=artifact_bytes,
            artifact_filename=artifact_filename,
            approval_args=approval_args,
        )
        expected_hash = approval_args.get("writeback_package_hash")
        if expected_hash and package_hash != expected_hash:
            self._audit(
                tool_name=TOOL_WRITEBACK,
                policy_decision="deny",
                reason="package_hash_mismatch",
                mapped_identity=mapped_identity,
                experiment_id=None,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=compute_args_hash(
                    {
                        "artifact_filename": artifact_filename,
                        "expected_hash": expected_hash,
                        "actual_hash": package_hash,
                    }
                ),
                error={
                    "code": "PACKAGE_HASH_MISMATCH",
                    "expected": expected_hash,
                    "actual": package_hash,
                },
            )
            raise OpenCloningAdapterError(
                reason="package_hash_mismatch",
                message=(
                    "writeback package hash does not match the approved hash "
                    "— the cloning strategy changed between approval and "
                    "writeback. Request a new approval."
                ),
            )

        # Use the shared orchestration for token verify + identity + policy.
        # The exec_fn uploads the artifact via the eLabFTW client.
        result = self._execute_writeback(
            context_token=context_token,
            approval_id=approval_id,
            approval_args=approval_args,
            mapped_identity=mapped_identity,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            artifact_filename=artifact_filename,
            artifact_bytes=artifact_bytes,
            artifact_comment=artifact_comment,
            artifact_reference=artifact_reference,
            plan_approval_id=plan_approval_id,
        )
        # Surface the validation bundle in the result dict so the orchestrator
        # can emit it as an SSE event and the widget can display it.
        result.result["validation_bundle"] = validation_bundle
        # C52: include the package hash in the result for transparency.
        result.result["writeback_package_hash"] = package_hash

        # C52: rebuild bundle with report-draft checks set to "pass" now that
        # the reports have been generated inside _execute_writeback.
        wb_result = result.result.get("writeback_result")
        if isinstance(wb_result, dict):
            notebook_appended = wb_result.get("notebook_appended", False)
        else:
            notebook_appended = result.result.get("notebook_summary_html") is not None
        if notebook_appended:
            updated_bundle = self.build_validation_bundle(
                run_id=request_id or f"writeback-{conversation_id or 'unknown'}",
                approval_args=approval_args,
                notebook_draft_generated=True,
                audit_draft_generated=True,
            )
            result.result["validation_bundle"] = updated_bundle

        return result

    def _verify_identity_binding(
        self,
        *,
        tool_name: str,
        claims: Any,
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None,
        request_id: str | None,
        keycloak_subject: str | None,
        librechat_user_id: str | None,
        provider: str | None,
        model_id: str | None,
        tool_args_hash: str,
    ) -> None:
        """Verify identity resolution and token-identity binding.

        Raises UnmappedCaller or InvalidContextToken on failure (after audit).
        """
        if mapped_identity is None:
            self._audit(
                tool_name=tool_name,
                policy_decision="deny",
                reason="unmapped_caller",
                mapped_identity=None,
                experiment_id=claims.experiment_id,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=tool_args_hash,
                error={"code": "UNMAPPED_CALLER"},
            )
            raise UnmappedCaller()

        if claims.mapped_elabftw_user_id != mapped_identity.elabftw_user_id:
            self._audit(
                tool_name=tool_name,
                policy_decision="deny",
                reason="context_token_user_mismatch",
                mapped_identity=mapped_identity,
                experiment_id=claims.experiment_id,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=tool_args_hash,
                error={"code": "CONTEXT_TOKEN_USER_MISMATCH"},
            )
            raise InvalidContextToken("token user does not match resolved identity")

        if (
            claims.keycloak_subject is not None
            and mapped_identity.keycloak_subject is not None
        ):
            if claims.keycloak_subject != mapped_identity.keycloak_subject:
                self._audit(
                    tool_name=tool_name,
                    policy_decision="deny",
                    reason="context_token_kc_subject_mismatch",
                    mapped_identity=mapped_identity,
                    experiment_id=claims.experiment_id,
                    conversation_id=conversation_id,
                    request_id=request_id,
                    keycloak_subject=keycloak_subject,
                    librechat_user_id=librechat_user_id,
                    provider=provider,
                    model_id=model_id,
                    tool_args_hash=tool_args_hash,
                    error={"code": "CONTEXT_TOKEN_KC_SUBJECT_MISMATCH"},
                )
                raise InvalidContextToken(
                    "token keycloak_subject does not match resolved identity"
                )

        if (
            claims.librechat_user_id is not None
            and mapped_identity.librechat_user_id is not None
        ):
            if claims.librechat_user_id != mapped_identity.librechat_user_id:
                self._audit(
                    tool_name=tool_name,
                    policy_decision="deny",
                    reason="context_token_lc_user_mismatch",
                    mapped_identity=mapped_identity,
                    experiment_id=claims.experiment_id,
                    conversation_id=conversation_id,
                    request_id=request_id,
                    keycloak_subject=keycloak_subject,
                    librechat_user_id=librechat_user_id,
                    provider=provider,
                    model_id=model_id,
                    tool_args_hash=tool_args_hash,
                    error={"code": "CONTEXT_TOKEN_LC_USER_MISMATCH"},
                )
                raise InvalidContextToken(
                    "token librechat_user_id does not match resolved identity"
                )

    def _check_policy_decision(
        self,
        *,
        tool_name: str,
        tier: Tier,
        mapped_identity: MappedIdentity,
        experiment_id: int,
        conversation_id: str | None,
        request_id: str | None,
        keycloak_subject: str | None,
        librechat_user_id: str | None,
        provider: str | None,
        model_id: str | None,
        tool_args_hash: str,
        has_approval: bool = False,
        approval_id: str | None = None,
    ) -> None:
        """Check policy decision. Raises PolicyDenied on denial (after audit)."""
        policy_req = PolicyRequest(
            tool_name=tool_name,
            tier=tier,
            adapter="opencloning",
            user_id=mapped_identity.elabftw_user_id,
            team_id=mapped_identity.elabftw_team_id,
            has_approval=has_approval,
            approval_id=approval_id,
        )
        decision = self.policy_engine.decide(policy_req)

        if decision.decision != "allow":
            self._audit(
                tool_name=tool_name,
                policy_decision="deny",
                reason=decision.reason,
                mapped_identity=mapped_identity,
                experiment_id=experiment_id,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=tool_args_hash,
                approval_id=approval_id,
                error={
                    "code": "POLICY_DENIED",
                    "reason": decision.reason,
                    "tier": decision.tier,
                },
            )
            raise PolicyDenied(decision)

    def _verify_and_decode_token(
        self,
        *,
        tool_name: str,
        context_token: str,
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None,
        request_id: str | None,
        keycloak_subject: str | None,
        librechat_user_id: str | None,
        provider: str | None,
        model_id: str | None,
        early_hash: str,
    ) -> Any:
        """Verify context token signature and expiry.  Returns claims or raises."""
        if not context_token:
            self._audit(
                tool_name=tool_name,
                policy_decision="deny",
                reason="missing_context_token",
                mapped_identity=mapped_identity,
                experiment_id=None,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=early_hash,
                error={"code": "MISSING_CONTEXT_TOKEN"},
            )
            raise InvalidContextToken("missing")
        try:
            return verify_context_token(context_token)
        except InvalidContextToken as exc:
            self._audit(
                tool_name=tool_name,
                policy_decision="deny",
                reason=f"invalid_context_token:{exc.detail}",
                mapped_identity=mapped_identity,
                experiment_id=None,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=early_hash,
                error={"code": "INVALID_CONTEXT_TOKEN", "detail": exc.detail},
            )
            raise

    def _require_client(
        self,
        *,
        tool_name: str,
        mapped_identity: MappedIdentity | None,
        experiment_id: int,
        conversation_id: str | None,
        request_id: str | None,
        keycloak_subject: str | None,
        librechat_user_id: str | None,
        provider: str | None,
        model_id: str | None,
        tool_args_hash: str,
    ) -> None:
        """Raise OpenCloningAdapterError if the client is not configured."""
        if self.client is not None:
            return
        self._audit(
            tool_name=tool_name,
            policy_decision="deny",
            reason="opencloning_not_configured",
            mapped_identity=mapped_identity,
            experiment_id=experiment_id,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            tool_args_hash=tool_args_hash,
            error={"code": "OPENCLONING_NOT_CONFIGURED"},
        )
        raise OpenCloningAdapterError(
            reason="opencloning_not_configured",
            message=(
                "OpenCloning client not configured "
                "(set LAB_COPILOT_OPENCLONING_BASE_URL)"
            ),
        )

    def _accumulate_strategy_data(self, result: dict[str, Any]) -> None:
        """Accumulate sequences, sources, and primers for cloning history JSON."""
        existing_seq_ids = {s.get("id") for s in self._strategy_sequences}
        for seq in result.get("sequences", []):
            if isinstance(seq, dict) and seq.get("id") not in existing_seq_ids:
                self._strategy_sequences.append(dict(seq))
                existing_seq_ids.add(seq.get("id"))
        existing_src_ids = {s.get("id") for s in self._strategy_sources}
        for src in result.get("sources", []):
            if isinstance(src, dict) and src.get("id") not in existing_src_ids:
                self._strategy_sources.append(dict(src))
                existing_src_ids.add(src.get("id"))
        existing_primer_names = {p.get("name") for p in self._strategy_primers}
        for primer in result.get("primers", []):
            if (
                isinstance(primer, dict)
                and primer.get("name") not in existing_primer_names
            ):
                # Assign a unique sequential ID — OpenCloning's primer design
                # endpoints return id=0 for all new primers, which causes
                # validation failures when the history JSON is reloaded.
                self._primer_counter += 1
                primer_copy = dict(primer)
                primer_copy["id"] = self._primer_counter
                self._strategy_primers.append(primer_copy)
                existing_primer_names.add(primer.get("name"))

    def _resolve_artifact_bundle(
        self,
        *,
        artifact_reference: dict[str, Any],
        artifact_filename: str,
        claims: Any,
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None,
        request_id: str | None,
        keycloak_subject: str | None,
        librechat_user_id: str | None,
        provider: str | None,
        model_id: str | None,
        early_hash: str,
    ) -> tuple[str, bytes]:
        """Resolve artifact content from the artifact bundle store."""
        try:
            stored = (self.artifact_store or get_artifact_bundle_store()).get(
                plan_id=str(artifact_reference.get("plan_id") or ""),
                plan_hash=str(artifact_reference.get("plan_hash") or ""),
                artifact_id=str(artifact_reference.get("artifact_id") or ""),
                expected_sha256=artifact_reference.get("expected_sha256"),
                expected_size_bytes=artifact_reference.get("expected_size_bytes"),
                binding={
                    "record_type": claims.record_type,
                    "record_id": str(claims.experiment_id),
                    "mapped_elabftw_user_id": claims.mapped_elabftw_user_id,
                },
            )
        except ArtifactBundleError as exc:
            self._audit(
                tool_name=TOOL_WRITEBACK,
                policy_decision="deny",
                reason=f"artifact_bundle_error:{type(exc).__name__}",
                mapped_identity=mapped_identity,
                experiment_id=claims.experiment_id,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=early_hash,
                error={
                    "code": "ARTIFACT_BUNDLE_ERROR",
                    "exception": type(exc).__name__,
                    "message": str(exc),
                },
            )
            raise OpenCloningAdapterError(
                reason="artifact_bundle_error",
                message=f"artifact bundle lookup failed: {exc}",
            ) from exc
        resolved_filename = str(artifact_filename or stored.filename or "construct.gb")
        if len(stored.bytes_data) > MAX_FILE_SIZE_BYTES:
            raise FileTooLarge(len(stored.bytes_data), MAX_FILE_SIZE_BYTES)
        return resolved_filename, stored.bytes_data

    def _consume_writeback_approval(
        self,
        *,
        plan_approval_id: str | None,
        approval_id: str,
        approval_args: dict[str, Any],
        claims: Any,
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None,
        request_id: str | None,
        keycloak_subject: str | None,
        librechat_user_id: str | None,
        provider: str | None,
        model_id: str | None,
        tool_args_hash: str,
    ) -> str:
        """Consume writeback approval token (or use plan-level bypass)."""
        if plan_approval_id:
            return plan_approval_id
        if self.approval_store is not None and approval_id:
            try:
                self.approval_store.consume(
                    approval_id,
                    tool_name=TOOL_WRITEBACK,
                    args_hash=compute_args_hash(approval_args),
                    target_record=str(claims.experiment_id),
                )
                return approval_id
            except Exception as exc:
                self._audit(
                    tool_name=TOOL_WRITEBACK,
                    policy_decision="deny",
                    reason=f"approval_consume_failed:{exc}",
                    mapped_identity=mapped_identity,
                    experiment_id=claims.experiment_id,
                    conversation_id=conversation_id,
                    request_id=request_id,
                    keycloak_subject=keycloak_subject,
                    librechat_user_id=librechat_user_id,
                    provider=provider,
                    model_id=model_id,
                    tool_args_hash=tool_args_hash,
                    approval_id=approval_id,
                    error={
                        "code": "APPROVAL_CONSUME_FAILED",
                        "exception": type(exc).__name__,
                        "message": str(exc),
                    },
                )
                raise OpenCloningAdapterError(
                    reason="approval_consume_failed",
                    message=f"approval token consume failed: {exc}",
                ) from exc
        return approval_id

    def _maybe_rewrite_genbank_features(self, artifact_bytes: bytes) -> bytes:
        """Re-annotate GenBank with features recovered from templates."""
        if not artifact_bytes or not self._strategy_sequences:
            return artifact_bytes
        try:
            from lab_copilot_gateway.opencloning_features import (
                rewrite_genbank_features,
            )

            genbank_str = (
                artifact_bytes.decode("utf-8")
                if isinstance(artifact_bytes, bytes)
                else artifact_bytes
            )
            rewritten = rewrite_genbank_features(genbank_str, self._strategy_sequences)
            return (
                rewritten.encode("utf-8")
                if isinstance(artifact_bytes, bytes)
                else rewritten
            )
        except Exception:
            return artifact_bytes

    def _render_writeback_reports(
        self,
        *,
        upload_id: int | None,
        history_upload_id: int | None,
        approval_args: dict[str, Any],
    ) -> dict[str, str | None]:
        """Render notebook summary HTML and Markdown audit report (T10/T52).

        Both are deterministic — built from accumulated cloning state.
        No upload is performed here; the caller uploads the audit report
        and appends the notebook HTML in separate steps so partial failures
        can be tracked individually.
        """
        try:
            from lab_copilot_gateway.opencloning_reports import (
                render_cloning_audit_report_markdown,
                render_notebook_summary,
            )

            goal = str(approval_args.get("goal", ""))
            method = str(
                approval_args.get("method", approval_args.get("cloning_method", ""))
            )
            llm_rationale = str(approval_args.get("llm_rationale", ""))

            artifact_ids = {
                "genbank": upload_id,
                "history_json": history_upload_id,
            }

            notebook_html = render_notebook_summary(
                sequences=list(self._strategy_sequences),
                sources=list(self._strategy_sources),
                primers=list(self._strategy_primers),
                validation_bundle={},
                artifact_ids=artifact_ids,
                goal=goal,
                method=method,
                llm_rationale=llm_rationale,
            )

            audit_markdown = render_cloning_audit_report_markdown(
                sequences=list(self._strategy_sequences),
                sources=list(self._strategy_sources),
                primers=list(self._strategy_primers),
                validation_bundle={},
                step_manifests=[],
                artifact_ids=artifact_ids,
                goal=goal,
                method=method,
            )

            return {"notebook_html": notebook_html, "audit_markdown": audit_markdown}
        except Exception:
            return {"notebook_html": None, "audit_markdown": None}

    def _build_history_json(
        self,
        *,
        artifact_filename: str,
    ) -> tuple[bytes, str] | None:
        """Build the cloning history JSON bytes and filename.

        Returns ``None`` if no strategy data has been accumulated.
        Extracted from ``_execute_writeback`` so the hash can be computed
        before the approval is consumed.
        """
        if not (self._strategy_sequences or self._strategy_sources):
            return None
        import json as _json
        import os as _os

        strategy = {
            "sequences": [dict(s) for s in self._strategy_sequences],
            "sources": [dict(s) for s in self._strategy_sources],
            "primers": [dict(p) for p in self._strategy_primers],
        }
        _strip_self_references(strategy["sources"])
        _strip_sequence_names(strategy["sequences"])
        try:
            resp = self.client.get("/version")  # type: ignore[union-attr]
            strategy["backend_version"] = resp.get("backend_version", "")
            strategy["schema_version"] = resp.get("schema_version", "")
        except Exception:
            pass
        strategy_json = _json.dumps(strategy, indent=2).encode("utf-8")
        strategy_filename = _os.path.splitext(artifact_filename)[0] + "_history.json"
        return strategy_json, strategy_filename

    def _upload_genbank_step(
        self,
        *,
        claims: Any,
        rt_param: str,
        artifact_filename: str,
        artifact_bytes: bytes,
        artifact_comment: str,
    ) -> WritebackStepResult:
        """Upload the final GenBank construct as an attachment."""
        assert self.elabftw_client is not None
        try:
            upload_id = self.elabftw_client.upload_attachment(
                experiment_id=claims.experiment_id,
                filename=artifact_filename,
                data=artifact_bytes,
                comment=artifact_comment,
                record_type=rt_param,
            )
            return WritebackStepResult(
                step_name="upload_genbank",
                status="success",
                upload_id=upload_id,
            )
        except Exception as exc:
            return WritebackStepResult(
                step_name="upload_genbank",
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )

    def _upload_history_step(
        self,
        *,
        claims: Any,
        rt_param: str,
        history_json: bytes | None,
        history_filename: str | None,
    ) -> WritebackStepResult:
        """Upload the cloning history JSON as an attachment."""
        if history_json is None or history_filename is None:
            return WritebackStepResult(step_name="upload_history", status="skipped")
        assert self.elabftw_client is not None
        try:
            upload_id = self.elabftw_client.upload_attachment(
                experiment_id=claims.experiment_id,
                filename=history_filename,
                data=history_json,
                comment="cloning history - generated by OpenCloning via Lab Copilot",
                record_type=rt_param,
            )
            return WritebackStepResult(
                step_name="upload_history",
                status="success",
                upload_id=upload_id,
            )
        except Exception as exc:
            return WritebackStepResult(
                step_name="upload_history",
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )

    def _upload_audit_report_step(
        self,
        *,
        claims: Any,
        rt_param: str,
        audit_markdown: str | None,
        artifact_filename: str,
    ) -> WritebackStepResult:
        """Upload the Markdown audit report as an attachment (BLOCKING)."""
        if not audit_markdown:
            return WritebackStepResult(
                step_name="upload_audit_report",
                status="failed",
                error="audit report markdown was not generated",
            )
        assert self.elabftw_client is not None
        try:
            audit_bytes = audit_markdown.encode("utf-8")
            audit_filename = artifact_filename.rsplit(".", 1)[0] + "_audit_report.md"
            upload_id = self.elabftw_client.upload_attachment(
                experiment_id=claims.experiment_id,
                filename=audit_filename,
                data=audit_bytes,
                comment="Cloning audit report — generated by Lab Copilot Gateway",
                record_type=rt_param,
            )
            return WritebackStepResult(
                step_name="upload_audit_report",
                status="success",
                upload_id=upload_id,
            )
        except Exception as exc:
            return WritebackStepResult(
                step_name="upload_audit_report",
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )

    def _upload_intermediates_step(
        self,
        *,
        claims: Any,
        rt_param: str,
        approval_args: dict[str, Any],
        artifact_filename: str,
    ) -> list[WritebackStepResult]:
        """Upload selected intermediate sequences (opt-in via checkbox).

        When ``include_intermediate_attachments`` is True in approval_args,
        each sequence whose id is in ``selected_intermediate_ids`` is
        uploaded as a separate GenBank file. When False (default), this
        step is skipped entirely.
        """
        if not approval_args.get("include_intermediate_attachments"):
            return [
                WritebackStepResult(step_name="upload_intermediates", status="skipped")
            ]
        selected_ids = approval_args.get("selected_intermediate_ids") or []
        if not selected_ids:
            return [
                WritebackStepResult(step_name="upload_intermediates", status="skipped")
            ]
        return self._upload_selected_intermediates(
            claims=claims,
            rt_param=rt_param,
            selected_ids=selected_ids,
            artifact_filename=artifact_filename,
        )

    def _upload_selected_intermediates(
        self,
        *,
        claims: Any,
        rt_param: str,
        selected_ids: list[Any],
        artifact_filename: str,
    ) -> list[WritebackStepResult]:
        """Upload each selected intermediate sequence as a GenBank file."""
        assert self.elabftw_client is not None
        results: list[WritebackStepResult] = []
        base_name = artifact_filename.rsplit(".", 1)[0]
        for seq in self._strategy_sequences:
            seq_id = seq.get("id")
            if seq_id not in selected_ids:
                continue
            file_content = seq.get("file_content")
            if not file_content:
                continue
            content_bytes = (
                file_content.encode("utf-8")
                if isinstance(file_content, str)
                else file_content
            )
            seq_name = str(seq.get("name") or seq_id)
            filename = f"{base_name}_intermediate_{seq_name}.gb"
            try:
                upload_id = self.elabftw_client.upload_attachment(
                    experiment_id=claims.experiment_id,
                    filename=filename,
                    data=content_bytes,
                    comment=f"Intermediate sequence {seq_name} — generated by Lab Copilot Gateway",
                    record_type=rt_param,
                )
                results.append(
                    WritebackStepResult(
                        step_name=f"upload_intermediate_{seq_id}",
                        status="success",
                        upload_id=upload_id,
                    )
                )
            except Exception as exc:
                results.append(
                    WritebackStepResult(
                        step_name=f"upload_intermediate_{seq_id}",
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
        return results

    def _append_notebook_step(
        self,
        *,
        claims: Any,
        rt_param: str,
        notebook_html: str | None,
    ) -> WritebackStepResult:
        """Append the notebook summary to the experiment body."""
        if not notebook_html:
            return WritebackStepResult(
                step_name="append_notebook",
                status="skipped",
                error="notebook HTML was not generated",
            )
        assert self.elabftw_client is not None
        try:
            current = self.elabftw_client.get_experiment(int(claims.experiment_id))
            existing_body = str(current.get("body", "") or "")
            amended_body = _wrap_amendment_html(existing_body, notebook_html)
            self.elabftw_client.patch_experiment_body(
                int(claims.experiment_id), amended_body
            )
            return WritebackStepResult(step_name="append_notebook", status="success")
        except Exception as exc:
            return WritebackStepResult(
                step_name="append_notebook",
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )

    def _patch_metadata_step(
        self,
        *,
        claims: Any,
        rt_param: str,
        action_id: str,
        upload_id: int | None,
    ) -> WritebackStepResult:
        """Patch metadata/provenance (non-blocking)."""
        assert self.elabftw_client is not None
        try:
            if rt_param == "items":
                current_record = self.elabftw_client.get_item(int(claims.experiment_id))
            else:
                current_record = self.elabftw_client.get_experiment(
                    int(claims.experiment_id)
                )
            existing_metadata = current_record.get("metadata") or {}
            updated_metadata = _merge_provenance_into_metadata(
                existing_metadata,
                action_id,
                {
                    "lab_copilot_audit_id": action_id,
                    "lab_copilot_tool": TOOL_WRITEBACK,
                    "lab_copilot_upload_id": str(upload_id or 0),
                },
            )
            self.elabftw_client.patch_experiment_metadata(
                experiment_id=claims.experiment_id,
                metadata=updated_metadata,
                record_type=rt_param,
            )
            return WritebackStepResult(step_name="patch_metadata", status="success")
        except Exception as exc:
            return WritebackStepResult(
                step_name="patch_metadata",
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )

    def _assemble_writeback_result(
        self,
        *,
        steps: list[WritebackStepResult],
        upload_ids: dict[str, int],
        experiment_id: int,
    ) -> WritebackResult:
        """Determine overall status from step results.

        Rules:
        - GenBank upload failure → "failed" (primary deliverable missing)
        - Audit report upload failure → "failed" (audit trail incomplete)
        - Notebook append failure after uploads → "partial" with rollback
        - Metadata patch failure after notebook → "success" with warning
        """
        flags = _step_flags(steps)
        rollback = _build_rollback_instructions(upload_ids, experiment_id)
        if flags["genbank_failed"] or flags["audit_failed"]:
            return WritebackResult(
                status="failed",
                upload_ids=upload_ids,
                notebook_appended=flags["notebook_appended"],
                metadata_patched=flags["metadata_patched"],
                steps=steps,
                rollback_instructions=rollback,
            )
        if flags["notebook_failed"]:
            return WritebackResult(
                status="partial",
                upload_ids=upload_ids,
                notebook_appended=flags["notebook_appended"],
                metadata_patched=flags["metadata_patched"],
                steps=steps,
                rollback_instructions=rollback,
            )
        warnings: list[str] = []
        if flags["metadata_failed"]:
            warnings.append(
                "Metadata/provenance patch failed — the notebook entry is "
                "visible but provenance metadata was not updated."
            )
        return WritebackResult(
            status="success",
            upload_ids=upload_ids,
            notebook_appended=flags["notebook_appended"],
            metadata_patched=flags["metadata_patched"],
            steps=steps,
            warnings=warnings,
        )

    def _require_elabftw_client(
        self,
        *,
        mapped_identity: MappedIdentity | None,
        experiment_id: int,
        conversation_id: str | None,
        request_id: str | None,
        keycloak_subject: str | None,
        librechat_user_id: str | None,
        provider: str | None,
        model_id: str | None,
        tool_args_hash: str,
        approval_id: str,
    ) -> None:
        """Raise OpenCloningAdapterError if eLabFTW client is not configured."""
        if self.elabftw_client is not None:
            return
        self._audit(
            tool_name=TOOL_WRITEBACK,
            policy_decision="deny",
            reason="elabftw_not_configured",
            mapped_identity=mapped_identity,
            experiment_id=experiment_id,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            tool_args_hash=tool_args_hash,
            approval_id=approval_id,
            error={"code": "ELABFTW_NOT_CONFIGURED"},
        )
        raise OpenCloningAdapterError(
            reason="elabftw_not_configured",
            message=(
                "eLabFTW client not configured for writeback "
                "(set LAB_COPILOT_ELABFTW_BASE_URL and LAB_COPILOT_ELABFTW_API_KEY)"
            ),
        )

    def _execute_writeback(
        self,
        *,
        context_token: str,
        approval_id: str,
        approval_args: dict[str, Any],
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None,
        request_id: str | None,
        keycloak_subject: str | None,
        librechat_user_id: str | None,
        provider: str | None,
        model_id: str | None,
        artifact_filename: str,
        artifact_bytes: bytes,
        artifact_comment: str,
        artifact_reference: dict[str, Any] | None = None,
        plan_approval_id: str | None = None,
    ) -> OpenCloningResult:
        """Writeback orchestration: token → identity → policy → approval → upload → provenance → audit.

        When ``plan_approval_id`` is set (C41), the plan executor has already
        consumed the plan-level approval (bound to the plan hash, which covers
        all step args).  The per-step approval consume is skipped; the plan
        approval_id is used in audit records.

        C52: the writeback package now includes the GenBank construct, history
        JSON, Markdown audit report, optional intermediate attachments, a
        notebook entry appended to the experiment body, and metadata
        provenance. Each step is tracked as a ``WritebackStepResult`` so
        partial failures are reported clearly with rollback instructions.
        """
        early_hash = compute_args_hash(
            {
                "artifact_filename": artifact_filename,
                "artifact_size": (
                    artifact_reference.get("expected_size_bytes")
                    if artifact_reference
                    else len(artifact_bytes)
                ),
                "token_present": bool(context_token),
            }
        )

        # 1. Verify context token.
        claims = self._verify_and_decode_token(
            tool_name=TOOL_WRITEBACK,
            context_token=context_token,
            mapped_identity=mapped_identity,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            early_hash=early_hash,
        )

        if artifact_reference:
            artifact_filename, artifact_bytes = self._resolve_artifact_bundle(
                artifact_reference=artifact_reference,
                artifact_filename=artifact_filename,
                claims=claims,
                mapped_identity=mapped_identity,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                early_hash=early_hash,
            )

        tool_args_hash = compute_args_hash(
            {
                "artifact_filename": artifact_filename,
                "artifact_size": len(artifact_bytes),
                "experiment_id": claims.experiment_id,
            }
        )

        # 2-3. Identity resolution + token-identity binding.
        self._verify_identity_binding(
            tool_name=TOOL_WRITEBACK,
            claims=claims,
            mapped_identity=mapped_identity,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            tool_args_hash=tool_args_hash,
        )

        # 4. Policy decision (tier 4 write, requires approval).
        assert mapped_identity is not None
        self._check_policy_decision(
            tool_name=TOOL_WRITEBACK,
            tier=TOOL_TIER_WRITE,
            mapped_identity=mapped_identity,
            experiment_id=claims.experiment_id,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            tool_args_hash=tool_args_hash,
            has_approval=bool(approval_id),
            approval_id=approval_id,
        )

        # 5. Approval token consume (single-use, args-hash-bound).
        # C41: plan_approval_id skips per-step consume.
        effective_approval_id = self._consume_writeback_approval(
            plan_approval_id=plan_approval_id,
            approval_id=approval_id,
            approval_args=approval_args,
            claims=claims,
            mapped_identity=mapped_identity,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            tool_args_hash=tool_args_hash,
        )

        # 6. eLabFTW client must be configured for writeback.
        self._require_elabftw_client(
            mapped_identity=mapped_identity,
            experiment_id=claims.experiment_id,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            tool_args_hash=tool_args_hash,
            approval_id=effective_approval_id,
        )

        # 6b. Rewrite GenBank features to fix pydna annotation gaps.
        artifact_bytes = self._maybe_rewrite_genbank_features(artifact_bytes)

        # 7. Run all writeback steps (uploads, notebook append, metadata).
        import secrets as _secrets
        import time as _time

        action_id = (
            f"{self.action_id_prefix}-writeback-"
            f"{int(_time.time() * 1000)}-{_secrets.token_hex(4)}"
        )
        rt_param = "items" if claims.record_type == "resource" else "experiments"
        wb_result = self._run_writeback_steps(
            claims=claims,
            rt_param=rt_param,
            artifact_filename=artifact_filename,
            artifact_bytes=artifact_bytes,
            artifact_comment=artifact_comment,
            approval_args=approval_args,
            mapped_identity=mapped_identity,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            tool_args_hash=tool_args_hash,
            effective_approval_id=effective_approval_id,
            action_id=action_id,
        )

        # 8. Audit the writeback result.
        self._audit_writeback_result(
            wb_result=wb_result,
            artifact_filename=artifact_filename,
            artifact_bytes=artifact_bytes,
            claims=claims,
            mapped_identity=mapped_identity,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            tool_args_hash=tool_args_hash,
            effective_approval_id=effective_approval_id,
            action_id=action_id,
        )

        return self._build_writeback_opencloning_result(
            wb_result=wb_result,
            claims=claims,
            artifact_filename=artifact_filename,
            artifact_bytes=artifact_bytes,
            action_id=action_id,
        )

    def _run_writeback_steps(
        self,
        *,
        claims: Any,
        rt_param: str,
        artifact_filename: str,
        artifact_bytes: bytes,
        artifact_comment: str,
        approval_args: dict[str, Any],
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None,
        request_id: str | None,
        keycloak_subject: str | None,
        librechat_user_id: str | None,
        provider: str | None,
        model_id: str | None,
        tool_args_hash: str,
        effective_approval_id: str,
        action_id: str,
    ) -> WritebackResult:
        """Run all writeback upload/append/patch steps in order.

        Steps:
        1. Upload GenBank (blocking on failure)
        2. Upload history JSON (non-blocking)
        3. Upload audit report (BLOCKING — audit trail must be complete)
        4. Upload intermediate attachments (opt-in, non-blocking)
        5. Append notebook entry (non-blocking, but affects status)
        6. Patch metadata/provenance (non-blocking)
        """
        steps: list[WritebackStepResult] = []
        upload_ids: dict[str, int] = {}

        # Step 1: Upload GenBank (blocking).
        genbank_step = self._upload_genbank_step(
            claims=claims,
            rt_param=rt_param,
            artifact_filename=artifact_filename,
            artifact_bytes=artifact_bytes,
            artifact_comment=artifact_comment,
        )
        steps.append(genbank_step)
        if genbank_step.status == "failed":
            self._audit_writeback_error(
                genbank_step,
                claims,
                mapped_identity,
                conversation_id,
                request_id,
                keycloak_subject,
                librechat_user_id,
                provider,
                model_id,
                tool_args_hash,
                effective_approval_id,
            )
            return WritebackResult(
                status="failed",
                upload_ids=upload_ids,
                notebook_appended=False,
                metadata_patched=False,
                steps=steps,
                rollback_instructions=_build_rollback_instructions(
                    upload_ids, claims.experiment_id
                ),
            )
        upload_ids["genbank"] = genbank_step.upload_id or 0

        # Step 2: Build + upload history JSON (non-blocking).
        history_data = self._build_history_json(artifact_filename=artifact_filename)
        history_json = history_data[0] if history_data else None
        history_filename = history_data[1] if history_data else None
        history_step = self._upload_history_step(
            claims=claims,
            rt_param=rt_param,
            history_json=history_json,
            history_filename=history_filename,
        )
        steps.append(history_step)
        if history_step.upload_id is not None:
            upload_ids["history_json"] = history_step.upload_id

        # Step 3: Render reports + upload audit report (BLOCKING).
        reports = self._render_writeback_reports(
            upload_id=genbank_step.upload_id,
            history_upload_id=history_step.upload_id,
            approval_args=approval_args,
        )
        notebook_html = reports.get("notebook_html")
        audit_markdown = reports.get("audit_markdown")
        audit_step = self._upload_audit_report_step(
            claims=claims,
            rt_param=rt_param,
            audit_markdown=audit_markdown,
            artifact_filename=artifact_filename,
        )
        steps.append(audit_step)
        if audit_step.status == "failed":
            self._audit_writeback_error(
                audit_step,
                claims,
                mapped_identity,
                conversation_id,
                request_id,
                keycloak_subject,
                librechat_user_id,
                provider,
                model_id,
                tool_args_hash,
                effective_approval_id,
            )
            return WritebackResult(
                status="failed",
                upload_ids=upload_ids,
                notebook_appended=False,
                metadata_patched=False,
                steps=steps,
                rollback_instructions=_build_rollback_instructions(
                    upload_ids, claims.experiment_id
                ),
            )
        if audit_step.upload_id is not None:
            upload_ids["audit_report"] = audit_step.upload_id

        # Step 4: Upload intermediate attachments (opt-in, non-blocking).
        intermediate_steps = self._upload_intermediates_step(
            claims=claims,
            rt_param=rt_param,
            approval_args=approval_args,
            artifact_filename=artifact_filename,
        )
        steps.extend(intermediate_steps)
        for s in intermediate_steps:
            if s.upload_id is not None:
                upload_ids[f"intermediate_{s.step_name}"] = s.upload_id

        # Step 5: Append notebook entry (non-blocking, affects status).
        notebook_step = self._append_notebook_step(
            claims=claims,
            rt_param=rt_param,
            notebook_html=notebook_html,
        )
        steps.append(notebook_step)

        # Step 6: Patch metadata/provenance (non-blocking).
        metadata_step = self._patch_metadata_step(
            claims=claims,
            rt_param=rt_param,
            action_id=action_id,
            upload_id=genbank_step.upload_id,
        )
        steps.append(metadata_step)

        return self._assemble_writeback_result(
            steps=steps,
            upload_ids=upload_ids,
            experiment_id=claims.experiment_id,
        )

    def _audit_writeback_error(
        self,
        step: WritebackStepResult,
        claims: Any,
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None,
        request_id: str | None,
        keycloak_subject: str | None,
        librechat_user_id: str | None,
        provider: str | None,
        model_id: str | None,
        tool_args_hash: str,
        effective_approval_id: str,
    ) -> None:
        """Audit a blocking writeback step failure."""
        self._audit(
            tool_name=TOOL_WRITEBACK,
            policy_decision="allow",
            reason=f"writeback_step_failed:{step.step_name}",
            mapped_identity=mapped_identity,
            experiment_id=claims.experiment_id,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            tool_args_hash=tool_args_hash,
            approval_id=effective_approval_id,
            error={
                "code": "WRITEBACK_STEP_FAILED",
                "step": step.step_name,
                "message": step.error or "unknown error",
            },
        )

    def _audit_writeback_result(
        self,
        *,
        wb_result: WritebackResult,
        artifact_filename: str,
        artifact_bytes: bytes,
        claims: Any,
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None,
        request_id: str | None,
        keycloak_subject: str | None,
        librechat_user_id: str | None,
        provider: str | None,
        model_id: str | None,
        tool_args_hash: str,
        effective_approval_id: str,
        action_id: str,
    ) -> None:
        """Audit the final writeback result (success, partial, or failed)."""
        reason = f"writeback_{wb_result.status}"
        self._audit(
            tool_name=TOOL_WRITEBACK,
            policy_decision="allow",
            reason=reason,
            mapped_identity=mapped_identity,
            experiment_id=claims.experiment_id,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            tool_args_hash=tool_args_hash,
            approval_id=effective_approval_id,
            api_call_summary={
                "method": "POST",
                "path": f"/api/v2/experiments/{claims.experiment_id}/uploads",
            },
            result_summary={
                "status": wb_result.status,
                "upload_ids": wb_result.upload_ids,
                "artifact_filename": artifact_filename,
                "artifact_size": len(artifact_bytes),
                "steps": [s.step_name for s in wb_result.steps],
            },
            require_persistence=True,
            action_id=action_id,
        )

    def _build_writeback_opencloning_result(
        self,
        *,
        wb_result: WritebackResult,
        claims: Any,
        artifact_filename: str,
        artifact_bytes: bytes,
        action_id: str,
    ) -> OpenCloningResult:
        """Build the final OpenCloningResult from a WritebackResult."""
        genbank_id = wb_result.upload_ids.get("genbank")
        history_id = wb_result.upload_ids.get("history_json")
        audit_id = wb_result.upload_ids.get("audit_report")

        # Re-render the notebook HTML for the result dict (it was already
        # appended to the experiment body; including it in the result lets
        # the widget display what was written).
        notebook_html = None
        for s in wb_result.steps:
            if s.step_name == "append_notebook" and s.status == "success":
                notebook_html = "(appended to experiment body)"
                break

        return OpenCloningResult(
            tool_name=TOOL_WRITEBACK,
            result={
                "upload_id": genbank_id,
                "experiment_id": claims.experiment_id,
                "artifact_filename": artifact_filename,
                "history_upload_id": history_id,
                "notebook_summary_html": notebook_html,
                "audit_report_markdown": None,
                "audit_report_upload_id": audit_id,
                "artifact_ids": {
                    "genbank": genbank_id,
                    "history_json": history_id,
                    "audit_report": audit_id,
                },
                "writeback_result": wb_result.to_dict(),
            },
            audit_action_id=action_id,
        )

    # --- shared orchestration --------------------------------------------

    def _execute(
        self,
        *,
        tool_name: str,
        context_token: str,
        mapped_identity: MappedIdentity | None,
        conversation_id: str | None,
        request_id: str | None,
        keycloak_subject: str | None,
        librechat_user_id: str | None,
        provider: str | None,
        model_id: str | None,
        args_for_hash: dict[str, Any],
        api_call_summary: dict[str, Any],
        exec_fn: Any,  # callable(client) -> dict
    ) -> OpenCloningResult:
        """Shared orchestration: token → identity → policy → client → audit.

        All four tool methods delegate here.  The ``exec_fn`` lambda is the
        only tool-specific part — it receives the client and returns the
        downstream result dict.
        """
        # Compute early redacted args hash for audit on failure paths.
        early_hash = compute_args_hash(
            {**args_for_hash, "token_present": bool(context_token)}
        )

        claims = self._verify_and_decode_token(
            tool_name=tool_name,
            context_token=context_token,
            mapped_identity=mapped_identity,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            early_hash=early_hash,
        )

        # Once claims verified, use the experiment_id from the token for
        # context_refs correlation.
        tool_args_hash = compute_args_hash(
            {**args_for_hash, "experiment_id": claims.experiment_id}
        )

        # 2-3. Identity resolution + token-identity binding.
        self._verify_identity_binding(
            tool_name=tool_name,
            claims=claims,
            mapped_identity=mapped_identity,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            tool_args_hash=tool_args_hash,
        )

        # 4. Policy decision (tier 3 read-only, no approval needed).
        assert mapped_identity is not None
        self._check_policy_decision(
            tool_name=tool_name,
            tier=TOOL_TIER,
            mapped_identity=mapped_identity,
            experiment_id=claims.experiment_id,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            tool_args_hash=tool_args_hash,
        )

        # 5. Client-not-configured check.
        self._require_client(
            tool_name=tool_name,
            mapped_identity=mapped_identity,
            experiment_id=claims.experiment_id,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            tool_args_hash=tool_args_hash,
        )

        # 6. Downstream call.
        try:
            result = exec_fn(self.client)
        except Exception as exc:
            self._audit(
                tool_name=tool_name,
                policy_decision="allow",
                reason="client_error",
                mapped_identity=mapped_identity,
                experiment_id=claims.experiment_id,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=tool_args_hash,
                api_call_summary=api_call_summary,
                error={
                    "code": "CLIENT_ERROR",
                    "exception": type(exc).__name__,
                    "message": str(exc),
                },
            )
            raise OpenCloningAdapterError(
                reason="client_error",
                message=f"OpenCloning client raised {type(exc).__name__}: {exc}",
            ) from exc

        # 7. Success — audit + return.
        import secrets as _secrets
        import time as _time

        action_id = f"{self.action_id_prefix}-{int(_time.time() * 1000)}-{_secrets.token_hex(4)}"

        self._audit(
            tool_name=tool_name,
            policy_decision="allow",
            reason="call_succeeded",
            mapped_identity=mapped_identity,
            experiment_id=claims.experiment_id,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            provider=provider,
            model_id=model_id,
            tool_args_hash=tool_args_hash,
            api_call_summary=api_call_summary,
            result_summary={
                "result_keys": list(result.keys()) if isinstance(result, dict) else []
            },
            require_persistence=True,
            action_id=action_id,
        )

        # 7b. Store sequences for later injection into opencloning.call.
        # The full sequences (with file_content) are stored server-side.
        # Redaction happens in _opencloning_invoke_success (app.py) after
        # artifact normalization, so the artifact builder can still access
        # the raw file_content to create downloadable artifacts.
        #
        # ID rewriting: OpenCloning is stateless, so each call returns
        # whatever id was in the request body (often 0). The gateway assigns
        # unique ids and rewrites matching source ids so the saved history JSON
        # remains a valid source→sequence graph.
        if isinstance(result, dict):
            self._seq_counter = _rewrite_opencloning_result_ids(
                result, self._sequence_store, self._seq_counter
            )
            _enrich_strategy_names(result)
            self._accumulate_strategy_data(result)

        return OpenCloningResult(
            tool_name=tool_name,
            result=result,
            audit_action_id=action_id,
        )

    # --- audit writer ----------------------------------------------------

    def _audit(
        self,
        *,
        tool_name: str,
        policy_decision: str,
        reason: str,
        mapped_identity: MappedIdentity | None,
        experiment_id: int | None,
        conversation_id: str | None,
        request_id: str | None,
        keycloak_subject: str | None,
        librechat_user_id: str | None,
        provider: str | None,
        model_id: str | None,
        tool_args_hash: str | None = None,
        approval_id: str | None = None,
        api_call_summary: dict[str, Any] | None = None,
        result_summary: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        require_persistence: bool = False,
        action_id: str | None = None,
    ) -> None:
        """Persist an audit record.

        Mirrors the eLabFTW adapter's audit pattern: deny/error paths
        swallow audit-write failures; success paths raise if audit
        persistence fails.
        """
        import secrets as _secrets
        import time as _time

        if action_id is None:
            action_id = f"{self.action_id_prefix}-{int(_time.time() * 1000)}-{_secrets.token_hex(4)}"

        context_refs: list[dict[str, Any]] = []
        if experiment_id is not None:
            context_refs.append({"kind": "experiment", "id": experiment_id})

        record = AuditRecord(
            action_id=action_id,
            conversation_id=conversation_id,
            request_id=request_id,
            keycloak_subject=keycloak_subject,
            librechat_user_id=librechat_user_id,
            mapped_elabftw_user_id=(
                mapped_identity.elabftw_user_id if mapped_identity else None
            ),
            mapped_elabftw_team_id=(
                mapped_identity.elabftw_team_id if mapped_identity else None
            ),
            provider=provider,
            model_id=model_id,
            tool_name=tool_name,
            tool_args_hash=tool_args_hash,
            context_refs=context_refs,
            policy_decision=policy_decision,
            approval_id=approval_id,
            api_call_summary=api_call_summary or {},
            result_summary=result_summary or {},
            error=error,
        )
        try:
            self.audit_store.append(record)
        except Exception as exc:
            if require_persistence:
                raise OpenCloningAdapterError(
                    reason="audit_persistence_failed",
                    message=f"audit append failed on success path: {exc}",
                ) from exc
            pass


# --- helpers --------------------------------------------------------------


def _infer_extension(file_format: str) -> str:
    """Map a file_format identifier to its canonical extension.

    Used for the file-type allowlist check in parse_sequence_file.
    """
    ext_map = {
        "genbank": ".gb",
        "gb": ".gb",
        "gbk": ".gbk",
        "ape": ".ape",
        "fasta": ".fasta",
        "fa": ".fa",
        "snapgene": ".dna",
        "dna": ".dna",
        "embl": ".embl",
    }
    return ext_map.get(file_format.lower(), f".{file_format.lower()}")


# --- module-level singleton for dependency injection ----------------------

_default_adapter: OpenCloningAdapter | None = None


def get_opencloning_adapter() -> OpenCloningAdapter:
    """Return the process-wide OpenCloning adapter (created lazily)."""
    global _default_adapter
    if _default_adapter is None:
        from lab_copilot_gateway.approval import get_approval_store
        from lab_copilot_gateway.audit import get_audit_store
        from lab_copilot_gateway.elabftw import (
            _default_client_from_env as _elabftw_client_from_env,
        )
        from lab_copilot_gateway.policy import get_policy_engine

        _default_adapter = OpenCloningAdapter(
            policy_engine=get_policy_engine(),
            audit_store=get_audit_store(),
            client=_default_client_from_env(),
            elabftw_client=_elabftw_client_from_env(),
            approval_store=get_approval_store(),
        )
    return _default_adapter


def reset_opencloning_adapter(
    adapter: OpenCloningAdapter | None = None,
) -> None:
    """Test helper: replace or clear the singleton."""
    global _default_adapter
    _default_adapter = adapter
