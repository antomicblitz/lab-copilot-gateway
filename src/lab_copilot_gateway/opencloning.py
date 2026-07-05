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

    if not id_map:
        return next_id

    for src in result.get("sources", []):
        if not isinstance(src, dict):
            continue
        src_id = str(src.get("id", ""))
        if src_id in id_map:
            src["id"] = id_map[src_id]
        inputs = src.get("input")
        if isinstance(inputs, list):
            for item in inputs:
                if not isinstance(item, dict):
                    continue
                seq_ref = str(item.get("sequence", ""))
                if seq_ref in id_map:
                    item["sequence"] = id_map[seq_ref]

    return next_id


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

    # --- C17: writeback artifact to eLabFTW -------------------------------

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
        artifact_reference = None
        if artifact_content is None and approval_args.get("artifact_id"):
            artifact_reference = {
                "artifact_id": approval_args.get("artifact_id"),
                "plan_id": approval_args.get("plan_id"),
                "plan_hash": approval_args.get("plan_hash"),
                "expected_sha256": approval_args.get("artifact_sha256"),
                "expected_size_bytes": approval_args.get("artifact_size_bytes"),
            }
            missing = [
                key
                for key, value in artifact_reference.items()
                if key
                in {
                    "artifact_id",
                    "plan_id",
                    "plan_hash",
                    "expected_sha256",
                    "expected_size_bytes",
                }
                and (value is None or value == "")
            ]
            if missing:
                raise OpenCloningAdapterError(
                    reason="missing_artifact_reference",
                    message=f"artifact bundle reference missing required fields: {missing}",
                )
        elif artifact_content is None:
            raise OpenCloningAdapterError(
                reason="missing_artifact_payload",
                message="writeback requires artifact_content or a complete artifact bundle reference",
            )

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

        # Use the shared orchestration for token verify + identity + policy.
        # The exec_fn uploads the artifact via the eLabFTW client.
        return self._execute_writeback(
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
        if not context_token:
            self._audit(
                tool_name=TOOL_WRITEBACK,
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
            claims = verify_context_token(context_token)
        except InvalidContextToken as exc:
            self._audit(
                tool_name=TOOL_WRITEBACK,
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

        if artifact_reference:
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
            artifact_filename = str(
                artifact_filename or stored.filename or "construct.gb"
            )
            artifact_bytes = stored.bytes_data

            if len(artifact_bytes) > MAX_FILE_SIZE_BYTES:
                raise FileTooLarge(len(artifact_bytes), MAX_FILE_SIZE_BYTES)

        tool_args_hash = compute_args_hash(
            {
                "artifact_filename": artifact_filename,
                "artifact_size": len(artifact_bytes),
                "experiment_id": claims.experiment_id,
            }
        )

        # 2. Identity resolution.
        if mapped_identity is None:
            self._audit(
                tool_name=TOOL_WRITEBACK,
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

        # 3. Token-identity binding.
        if claims.mapped_elabftw_user_id != mapped_identity.elabftw_user_id:
            self._audit(
                tool_name=TOOL_WRITEBACK,
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
                    tool_name=TOOL_WRITEBACK,
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
                    tool_name=TOOL_WRITEBACK,
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

        # 4. Policy decision (tier 4 write, requires approval).
        policy_req = PolicyRequest(
            tool_name=TOOL_WRITEBACK,
            tier=TOOL_TIER_WRITE,
            adapter="opencloning",
            user_id=mapped_identity.elabftw_user_id,
            team_id=mapped_identity.elabftw_team_id,
            has_approval=bool(approval_id),
            approval_id=approval_id,
        )
        decision = self.policy_engine.decide(policy_req)

        if decision.decision != "allow":
            self._audit(
                tool_name=TOOL_WRITEBACK,
                policy_decision="deny",
                reason=decision.reason,
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
                    "code": "POLICY_DENIED",
                    "reason": decision.reason,
                    "tier": decision.tier,
                },
            )
            raise PolicyDenied(decision)

        # 5. Approval token consume (single-use, args-hash-bound).
        # C41: when ``plan_approval_id`` is set, the plan executor has
        # already consumed the plan-level approval (bound to the plan hash,
        # which covers all step args).  Skip the per-step consume and use
        # the plan approval_id in audit records.
        if plan_approval_id:
            effective_approval_id = plan_approval_id
        elif self.approval_store is not None and approval_id:
            try:
                self.approval_store.consume(
                    approval_id,
                    tool_name=TOOL_WRITEBACK,
                    args_hash=compute_args_hash(approval_args),
                    target_record=str(claims.experiment_id),
                )
                effective_approval_id = approval_id
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
        else:
            effective_approval_id = approval_id

        # 6. eLabFTW client must be configured for writeback.
        if self.elabftw_client is None:
            self._audit(
                tool_name=TOOL_WRITEBACK,
                policy_decision="deny",
                reason="elabftw_not_configured",
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
                error={"code": "ELABFTW_NOT_CONFIGURED"},
            )
            raise OpenCloningAdapterError(
                reason="elabftw_not_configured",
                message=(
                    "eLabFTW client not configured for writeback "
                    "(set LAB_COPILOT_ELABFTW_BASE_URL and LAB_COPILOT_ELABFTW_API_KEY)"
                ),
            )

        # 6b. Rewrite GenBank features to fix pydna annotation gaps.
        # pydna's Gibson assembly and PCR do not transfer CDS/promoter/
        # terminator features from insert templates to the assembled product.
        # We recover them by searching for each template feature's nucleotide
        # sequence in the final product (see opencloning_features.py).
        if artifact_bytes and self._strategy_sequences:
            try:
                from lab_copilot_gateway.opencloning_features import (
                    rewrite_genbank_features,
                )

                genbank_str = (
                    artifact_bytes.decode("utf-8")
                    if isinstance(artifact_bytes, bytes)
                    else artifact_bytes
                )
                rewritten = rewrite_genbank_features(
                    genbank_str, self._strategy_sequences
                )
                artifact_bytes = (
                    rewritten.encode("utf-8")
                    if isinstance(artifact_bytes, bytes)
                    else rewritten
                )
            except Exception:
                # Non-fatal — upload the original if rewriting fails.
                pass

        # 7. Upload the artifact as an attachment.
        # record_type: "resource" → /api/v2/items/{id}, else /api/v2/experiments/{id}
        rt_param = "items" if claims.record_type == "resource" else "experiments"
        try:
            upload_id = self.elabftw_client.upload_attachment(
                experiment_id=claims.experiment_id,
                filename=artifact_filename,
                data=artifact_bytes,
                comment=artifact_comment,
                record_type=rt_param,
            )
        except Exception as exc:
            self._audit(
                tool_name=TOOL_WRITEBACK,
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
                approval_id=effective_approval_id,
                api_call_summary={
                    "method": "POST",
                    "path": f"/api/v2/experiments/{claims.experiment_id}/uploads",
                },
                error={
                    "code": "CLIENT_ERROR",
                    "exception": type(exc).__name__,
                    "message": str(exc),
                },
            )
            raise OpenCloningAdapterError(
                reason="client_error",
                message=f"eLabFTW upload failed {type(exc).__name__}: {exc}",
            ) from exc

        # 7b. Build and upload the cloning history JSON.
        # This matches what the OpenCloning frontend saves to eLabFTW as
        # {title}_history.json — the full reproducible workflow including
        # all intermediate sequences, their sources (provenance chain), and
        # primers. The JSON can be loaded back into OpenCloning via
        # "File > Load cloning history from file" or via the URL parameter
        # ?source=database&item_id=<id>&file_id=<upload_id>.
        history_upload_id = None
        if self._strategy_sequences or self._strategy_sources:
            import json as _json
            import os as _os

            # Build the cloning strategy from accumulated state
            strategy = {
                "sequences": self._strategy_sequences,
                "sources": self._strategy_sources,
                "primers": self._strategy_primers,
            }

            # Add version info (best-effort)
            try:
                resp = self.client.get("/version")  # type: ignore[union-attr]
                strategy["backend_version"] = resp.get("backend_version", "")
                strategy["schema_version"] = resp.get("schema_version", "")
            except Exception:
                pass

            strategy_json = _json.dumps(strategy, indent=2).encode("utf-8")
            strategy_filename = (
                _os.path.splitext(artifact_filename)[0] + "_history.json"
            )

            try:
                history_upload_id = self.elabftw_client.upload_attachment(
                    experiment_id=claims.experiment_id,
                    filename=strategy_filename,
                    data=strategy_json,
                    comment="cloning history - generated by OpenCloning via Lab Copilot",
                    record_type=rt_param,
                )
            except Exception:
                # Non-fatal — the GenBank file is already uploaded.
                # The history is a bonus, not a requirement.
                history_upload_id = None

        # 8. Write provenance (audit_action_id in metadata).
        import secrets as _secrets
        import time as _time

        action_id = f"{self.action_id_prefix}-writeback-{int(_time.time() * 1000)}-{_secrets.token_hex(4)}"

        try:
            self.elabftw_client.patch_experiment_metadata(
                experiment_id=claims.experiment_id,
                metadata={
                    "extra_fields": {
                        "lab_copilot_audit_id": action_id,
                        "lab_copilot_tool": TOOL_WRITEBACK,
                        "lab_copilot_upload_id": upload_id,
                    }
                },
                record_type=rt_param,
            )
        except Exception:
            # Provenance write failure is not fatal — the artifact is
            # already uploaded.  Log to audit but don't raise.
            pass

        # 9. Success audit.
        self._audit(
            tool_name=TOOL_WRITEBACK,
            policy_decision="allow",
            reason="writeback_succeeded",
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
                "upload_id": upload_id,
                "artifact_filename": artifact_filename,
                "artifact_size": len(artifact_bytes),
            },
            require_persistence=True,
            action_id=action_id,
        )

        return OpenCloningResult(
            tool_name=TOOL_WRITEBACK,
            result={
                "upload_id": upload_id,
                "experiment_id": claims.experiment_id,
                "artifact_filename": artifact_filename,
                "history_upload_id": history_upload_id,
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

        # 1. Verify context token (signature + expiry).
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
            claims = verify_context_token(context_token)
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

        # Once claims verified, use the experiment_id from the token for
        # context_refs correlation.
        tool_args_hash = compute_args_hash(
            {**args_for_hash, "experiment_id": claims.experiment_id}
        )

        # 2. Identity resolution — unmapped caller denies before any HTTP.
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

        # 3. Token-identity binding (three-way check).
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

        # 4. Policy decision (tier 3 read-only, no approval needed).
        policy_req = PolicyRequest(
            tool_name=tool_name,
            tier=TOOL_TIER,
            adapter="opencloning",
            user_id=mapped_identity.elabftw_user_id,
            team_id=mapped_identity.elabftw_team_id,
            has_approval=False,
            approval_id=None,
        )
        decision = self.policy_engine.decide(policy_req)

        if decision.decision != "allow":
            self._audit(
                tool_name=tool_name,
                policy_decision="deny",
                reason=decision.reason,
                mapped_identity=mapped_identity,
                experiment_id=claims.experiment_id,
                conversation_id=conversation_id,
                request_id=request_id,
                keycloak_subject=keycloak_subject,
                librechat_user_id=librechat_user_id,
                provider=provider,
                model_id=model_id,
                tool_args_hash=tool_args_hash,
                error={
                    "code": "POLICY_DENIED",
                    "reason": decision.reason,
                    "tier": decision.tier,
                },
            )
            raise PolicyDenied(decision)

        # 5. Client-not-configured check.
        if self.client is None:
            self._audit(
                tool_name=tool_name,
                policy_decision="deny",
                reason="opencloning_not_configured",
                mapped_identity=mapped_identity,
                experiment_id=claims.experiment_id,
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

            # 7c. Accumulate sequences, sources, and primers for the
            # cloning strategy JSON. This matches what the OpenCloning
            # frontend saves to eLabFTW as {title}_history.json — the
            # full reproducible workflow including all intermediate
            # sequences, their sources (provenance chain), and primers.
            # Deduplicate by sequence ID to avoid accumulating duplicates
            # across multiple runs in the same gateway process.
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
                    self._strategy_primers.append(dict(primer))
                    existing_primer_names.add(primer.get("name"))

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
