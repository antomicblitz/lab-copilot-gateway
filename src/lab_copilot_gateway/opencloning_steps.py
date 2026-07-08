"""OpenCloning step manifest contract — typed data model for intermediate operations.

This module defines ``StepManifest``, a structured dataclass that captures
intermediate OpenCloning operations (PCR, assembly, primer design, etc.)
as typed manifests.  Each manifest records what was called, which inputs
were used, what outputs were produced, and any warnings or errors.

Typed extractors dispatch by endpoint path and extract operation-specific
fields from the raw OpenCloning API response.  Unknown endpoints fall
through to the generic extractor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# --- StepManifest dataclass --------------------------------------------------


@dataclass
class StepManifest:
    """Structured manifest for an intermediate OpenCloning cloning step.

    This is the ``opencloning_step_manifest_v1`` contract.  It captures
    what operation was performed, its inputs and outputs (by ref/ID only —
    no raw sequence bytes), and any diagnostic information.

    All sequence-bearing fields use refs/IDs/hashes only.  The manifest
    MUST NOT contain raw sequence bytes.
    """

    schema: str = "opencloning_step_manifest_v1"
    run_id: str = ""
    step_id: str = ""
    parent_step_ids: list[str] = field(default_factory=list)
    operation_type: str = ""
    operation_label: str = ""
    status: str = "succeeded"
    tool_name: str = "opencloning.call"
    endpoint: str = ""
    input_refs: list[str] = field(default_factory=list)
    output_refs: list[str] = field(default_factory=list)
    template_refs: list[str] = field(default_factory=list)
    region_refs: list[str] = field(default_factory=list)
    primer_refs: list[str] = field(default_factory=list)
    protocol_refs: list[str] = field(default_factory=list)
    validation_refs: list[str] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    preview_ref: str | None = None
    created_at: str = field(default_factory=lambda: _now_iso())
    sequence_byte_policy: str = "server_side_only"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict matching the manifest schema."""
        return {
            "schema": self.schema,
            "run_id": self.run_id,
            "step_id": self.step_id,
            "parent_step_ids": list(self.parent_step_ids),
            "operation_type": self.operation_type,
            "operation_label": self.operation_label,
            "status": self.status,
            "tool_name": self.tool_name,
            "endpoint": self.endpoint,
            "input_refs": list(self.input_refs),
            "output_refs": list(self.output_refs),
            "template_refs": list(self.template_refs),
            "region_refs": list(self.region_refs),
            "primer_refs": list(self.primer_refs),
            "protocol_refs": list(self.protocol_refs),
            "validation_refs": list(self.validation_refs),
            "warnings": [dict(w) for w in self.warnings],
            "errors": [dict(e) for e in self.errors],
            "preview_ref": self.preview_ref,
            "created_at": self.created_at,
            "sequence_byte_policy": self.sequence_byte_policy,
        }


# --- helpers -----------------------------------------------------------------


def _now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _str_or(value: Any, fallback: str = "") -> str:
    """Return a string value or fallback."""
    if isinstance(value, str):
        return value
    return fallback


def _list_of_str(value: Any) -> list[str]:
    """Normalize a value to a list of strings."""
    if isinstance(value, list):
        return [str(v) for v in value if v is not None]
    return []


def _list_of_dict(value: Any) -> list[dict[str, str]]:
    """Normalize a value to a list of dicts with str values."""
    if not isinstance(value, list):
        return []
    result: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            result.append({str(k): str(v) for k, v in item.items()})
    return result


def _extract_ids_from_sequences(sequences: Any) -> list[str]:
    """Extract string IDs from an OpenCloning sequences list."""
    if not isinstance(sequences, list):
        return []
    ids: list[str] = []
    for seq in sequences:
        if isinstance(seq, dict):
            sid = seq.get("id")
            if sid is not None:
                ids.append(str(sid))
    return ids


def _extract_input_seq_ids(sources: Any) -> list[str]:
    """Extract input sequence IDs from OpenCloning source objects.

    Sources carry their inputs in a list of ``{sequence: <id>, ...}`` dicts.
    """
    if not isinstance(sources, list):
        return []
    ids: list[str] = []
    for src in sources:
        if not isinstance(src, dict):
            continue
        inputs = src.get("input")
        if isinstance(inputs, list):
            for inp in inputs:
                if isinstance(inp, dict):
                    seq_ref = inp.get("sequence")
                    if seq_ref is not None:
                        ids.append(str(seq_ref))
    return ids


def _extract_warnings(result: dict[str, Any]) -> list[dict[str, str]]:
    """Normalize OpenCloning warnings from a result dict."""
    raw = result.get("warnings")
    if not raw:
        return []
    return _list_of_dict(raw)


def _extract_errors(result: dict[str, Any]) -> list[dict[str, str]]:
    """Normalize OpenCloning errors from a result dict."""
    raw = result.get("errors")
    if not raw:
        return []
    # Errors can be a list of strings or dicts.
    if isinstance(raw, list):
        normalized: list[dict[str, str]] = []
        for item in raw:
            if isinstance(item, dict):
                normalized.append({str(k): str(v) for k, v in item.items()})
            elif isinstance(item, str):
                normalized.append({"message": item})
        return normalized
    return []


def _endpoint_label(endpoint: str) -> str:
    """Derive a human-readable operation label from an endpoint path."""
    known: dict[str, str] = {
        "/read_from_file": "Import sequence file",
        "/manually_typed": "Manual sequence entry",
        "/pcr": "PCR amplification",
        "/restriction": "Restriction digest",
        "/gibson_assembly": "Gibson assembly",
        "/ligation": "Ligation",
        "/restriction_and_ligation": "Restriction-ligation (Golden Gate)",
        "/homologous_recombination": "Homologous recombination",
        "/crispr": "CRISPR editing",
        "/cre_lox_recombination": "Cre/Lox recombination",
        "/gateway": "Gateway cloning",
        "/recombinase": "Recombinase cloning",
        "/polymerase_extension": "Polymerase extension",
        "/reverse_complement": "Reverse complement",
        "/oligonucleotide_hybridization": "Oligo hybridization",
        "/annotate/plannotate": "Plannotate annotation",
        "/primer_details": "Primer details",
        "/validate": "Validation",
        "/validate_syntax": "Syntax validation",
    }
    # Check exact match first.
    label = known.get(endpoint)
    if label:
        return label

    # Check prefix matches for primer_design/* paths.
    if endpoint.startswith("/primer_design/"):
        method = endpoint.removeprefix("/primer_design/").replace("_", " ").title()
        return f"Primer design — {method}"

    # Check prefix matches for batch_cloning/* paths.
    if endpoint.startswith("/batch_cloning/"):
        return f"Batch cloning — {endpoint.removeprefix('/batch_cloning/')}"

    # Check repository imports.
    if endpoint.startswith("/repository_id/"):
        repo = endpoint.removeprefix("/repository_id/").capitalize()
        return f"Import from {repo}"

    # Fallback: humanize the endpoint path.
    parts = endpoint.strip("/").split("/")
    label_parts = [p.replace("_", " ").title() for p in parts]
    return " — ".join(label_parts)


def _infer_status(result: dict[str, Any]) -> str:
    """Infer status from result dict — 'succeeded' unless errors present."""
    errors = result.get("errors")
    if isinstance(errors, list) and len(errors) > 0:
        return "failed"
    return "succeeded"


# --- Typed extractors --------------------------------------------------------


def extract_sequence_import(
    endpoint: str,
    request_body: dict[str, Any],
    result: dict[str, Any],
) -> StepManifest:
    """Extract a sequence import step manifest.

    Handles ``/read_from_file`` (file upload) and ``/manually_typed``
    (manual sequence entry).
    """
    output_refs = _extract_ids_from_sequences(result.get("sequences"))
    input_refs = _extract_input_seq_ids(result.get("sources"))

    # Detect file format from request body (read_from_file uses multipart
    # with a filename; manual_sequence carries no file format).
    file_format = request_body.get("file_format", "")

    label = _endpoint_label(endpoint)
    if file_format:
        label = f"{label} ({file_format})"

    return StepManifest(
        operation_type="sequence_import",
        operation_label=label,
        endpoint=endpoint,
        input_refs=input_refs,
        output_refs=output_refs,
        status=_infer_status(result),
        warnings=_extract_warnings(result),
        errors=_extract_errors(result),
    )


def extract_primer_design(
    endpoint: str,
    request_body: dict[str, Any],
    result: dict[str, Any],
) -> StepManifest:
    """Extract a primer design step manifest.

    Handles ``/primer_design/*`` endpoints.  Extract primer sequences
    (IDs only, no raw sequence bytes) and melting temperatures.
    """
    primer_refs = _list_of_str(
        [
            str(p.get("id"))
            for p in (result.get("primers") or [])
            if isinstance(p, dict) and p.get("id") is not None
        ]
    )
    template_refs = _extract_ids_from_sequences(request_body.get("sequences"))
    # Also check pcr_templates (used by primer_design endpoints).
    pcr_templates = request_body.get("pcr_templates")
    if isinstance(pcr_templates, list):
        for tmpl in pcr_templates:
            if isinstance(tmpl, dict):
                seq = tmpl.get("sequence")
                if isinstance(seq, dict) and seq.get("id") is not None:
                    template_refs.append(str(seq["id"]))
    # Single pcr_template.
    pcr_template = request_body.get("pcr_template")
    if isinstance(pcr_template, dict):
        seq = pcr_template.get("sequence")
        if isinstance(seq, dict) and seq.get("id") is not None:
            template_refs.append(str(seq["id"]))
    # Homologous recombination target.
    hr_target = request_body.get("homologous_recombination_target")
    if isinstance(hr_target, dict):
        seq = hr_target.get("sequence")
        if isinstance(seq, dict) and seq.get("id") is not None:
            template_refs.append(str(seq["id"]))

    # Generate warning for primers with high Tm differences or other issues.
    warnings = list(_extract_warnings(result))
    primers = result.get("primers") or []
    if isinstance(primers, list) and len(primers) >= 2:
        tm_values: list[float] = []
        for p in primers:
            if isinstance(p, dict):
                raw_tm = p.get("tm") or p.get("melting_temperature")
                if raw_tm is not None:
                    try:
                        tm_values.append(float(raw_tm))
                    except (ValueError, TypeError):
                        pass
        if len(tm_values) >= 2 and max(tm_values) - min(tm_values) > 5:
            warnings.append(
                {
                    "severity": "warning",
                    "code": "primer_tm_mismatch",
                    "message": (
                        f"Primer melting temperatures differ by "
                        f"{max(tm_values) - min(tm_values):.1f}°C "
                        f"(recommended ≤ 5°C)."
                    ),
                }
            )

    return StepManifest(
        operation_type="primer_design",
        operation_label=_endpoint_label(endpoint),
        endpoint=endpoint,
        template_refs=template_refs,
        primer_refs=primer_refs,
        status=_infer_status(result),
        warnings=warnings,
        errors=_extract_errors(result),
    )


def extract_pcr(
    endpoint: str,
    request_body: dict[str, Any],
    result: dict[str, Any],
) -> StepManifest:
    """Extract a PCR step manifest.

    Handles ``/pcr`` endpoint.  Extracts template source, region
    (start, end, strand), forward/reverse primer IDs, product
    sequence ID, product length.
    """
    output_refs = _extract_ids_from_sequences(result.get("sequences"))
    input_seq_ids = _extract_input_seq_ids(result.get("sources"))

    # Template refs are the input sequences to the PCR source.
    template_refs = list(input_seq_ids)

    # Region refs from the request body — PCR typically specifies
    # a region on the template in pcr_templates or via source inputs.
    region_refs: list[str] = []
    pcr_templates = request_body.get("pcr_templates")
    if isinstance(pcr_templates, list):
        for i, tmpl in enumerate(pcr_templates):
            if isinstance(tmpl, dict):
                loc = tmpl.get("location")
                if isinstance(loc, dict):
                    parts: list[str] = []
                    start = loc.get("start")
                    end = loc.get("end")
                    strand = loc.get("strand")
                    if start is not None:
                        parts.append(f"start={start}")
                    if end is not None:
                        parts.append(f"end={end}")
                    if strand:
                        parts.append(f"strand={strand}")
                    if parts:
                        region_refs.append(f"template-{i}:{','.join(parts)}")

    # Primer refs from the result.
    primers = result.get("primers") or result.get("products") or []
    primer_refs: list[str] = []
    if isinstance(primers, list):
        for p in primers:
            if isinstance(p, dict):
                pid = p.get("id")
                if pid is not None:
                    primer_refs.append(str(pid))
                else:
                    pname = p.get("name", "")
                    if pname:
                        primer_refs.append(pname)

    warnings = list(_extract_warnings(result))

    # Add feature-loss warning if OpenCloning returned one.
    feature_loss = result.get("feature_loss_warning")
    if isinstance(feature_loss, dict):
        warnings.append(feature_loss)

    return StepManifest(
        operation_type="pcr",
        operation_label="PCR amplification",
        endpoint=endpoint,
        input_refs=input_seq_ids,
        output_refs=output_refs,
        template_refs=template_refs,
        region_refs=region_refs,
        primer_refs=primer_refs,
        status=_infer_status(result),
        warnings=warnings,
        errors=_extract_errors(result),
    )


def extract_restriction_digest(
    endpoint: str,
    request_body: dict[str, Any],
    result: dict[str, Any],
) -> StepManifest:
    """Extract a restriction digest step manifest.

    Handles ``/restriction`` endpoint.  Extracts enzyme list,
    input sequence IDs, fragment output IDs.
    """
    output_refs = _extract_ids_from_sequences(result.get("sequences"))
    input_refs = _extract_input_seq_ids(result.get("sources"))
    # Fall back to sequences in the request body if sources have no inputs.
    if not input_refs:
        input_refs = _extract_ids_from_sequences(request_body.get("sequences"))

    # Extract enzyme names from the result.
    enzymes: list[str] = []
    for src in result.get("sources") or []:
        if isinstance(src, dict) and src.get("type") == "RestrictionSource":
            raw_enzymes = src.get("enzymes") or src.get("enzyme_names") or []
            if isinstance(raw_enzymes, list):
                enzymes.extend(str(e) for e in raw_enzymes)

    # Also check for enzymes in the result itself.
    if not enzymes:
        raw_enzymes = result.get("enzymes") or result.get("enzyme_names") or []
        if isinstance(raw_enzymes, list):
            enzymes = [str(e) for e in raw_enzymes]

    warnings = list(_extract_warnings(result))
    if not enzymes:
        warnings.append(
            {
                "severity": "warning",
                "code": "no_enzymes_detected",
                "message": "No restriction enzymes detected in digest result.",
            }
        )

    return StepManifest(
        operation_type="restriction_digest",
        operation_label="Restriction digest",
        endpoint=endpoint,
        input_refs=input_refs,
        output_refs=output_refs,
        status=_infer_status(result),
        warnings=warnings,
        errors=_extract_errors(result),
    )


def extract_assembly(
    endpoint: str,
    request_body: dict[str, Any],
    result: dict[str, Any],
) -> StepManifest:
    """Extract an assembly step manifest.

    Handles ``/gibson_assembly``, ``/ligation``,
    ``/restriction_and_ligation``, ``/homologous_recombination``,
    ``/crispr``, ``/cre_lox_recombination``, ``/gateway``,
    ``/recombinase``, ``/polymerase_extension``.

    Extracts input fragment IDs, assembly method, circular/linear
    result, output sequence ID.
    """
    output_refs = _extract_ids_from_sequences(result.get("sequences"))
    input_refs = _extract_input_seq_ids(result.get("sources"))
    # Fall back to sequences in the request body.
    if not input_refs:
        input_refs = _extract_ids_from_sequences(request_body.get("sequences"))

    method = _endpoint_label(endpoint)

    warnings = list(_extract_warnings(result))

    return StepManifest(
        operation_type="assembly",
        operation_label=method,
        endpoint=endpoint,
        input_refs=input_refs,
        output_refs=output_refs,
        status=_infer_status(result),
        warnings=warnings,
        errors=_extract_errors(result),
    )


def extract_annotation(
    endpoint: str,
    request_body: dict[str, Any],
    result: dict[str, Any],
) -> StepManifest:
    """Extract an annotation step manifest.

    Handles ``/annotate/plannotate`` endpoint.  Extracts input
    sequence ID, features found, annotation status.
    """
    # Input sequence from the request body (sequences list).
    input_refs = _extract_ids_from_sequences(request_body.get("sequences"))
    if not input_refs:
        input_refs = _extract_input_seq_ids(result.get("sources"))

    output_refs = _extract_ids_from_sequences(result.get("sequences"))

    # Detect number of features found.
    features = result.get("features") or []
    feature_count = len(features) if isinstance(features, list) else 0

    warnings = list(_extract_warnings(result))
    if feature_count == 0:
        warnings.append(
            {
                "severity": "warning",
                "code": "no_features_found",
                "message": "No features were found by Plannotate annotation.",
            }
        )
    else:
        warnings.append(
            {
                "severity": "info",
                "code": "features_found",
                "message": f"Found {feature_count} feature(s) in the sequence.",
            }
        )

    return StepManifest(
        operation_type="annotation",
        operation_label="Plannotate annotation",
        endpoint=endpoint,
        input_refs=input_refs,
        output_refs=output_refs,
        status=_infer_status(result),
        warnings=warnings,
        errors=_extract_errors(result),
    )


# --- Placeholder extractors --------------------------------------------------


def extract_protocol_lookup(
    endpoint: str,
    request_body: dict[str, Any],
    result: dict[str, Any],
) -> StepManifest:
    """Placeholder: extract a protocol lookup step manifest.

    Populated by later slices — currently returns a generic manifest
    with protocol refs from the result if detectable.
    """
    protocol_refs: list[str] = []
    # Attempt to extract protocol references from result.
    for key in ("protocol", "protocol_id", "protocol_ref"):
        val = result.get(key)
        if val is not None:
            protocol_refs.append(str(val))

    return StepManifest(
        operation_type="protocol_lookup",
        operation_label="Protocol lookup",
        endpoint=endpoint,
        protocol_refs=protocol_refs,
        status=_infer_status(result),
        warnings=_extract_warnings(result),
        errors=_extract_errors(result),
    )


def extract_validation(
    endpoint: str,
    request_body: dict[str, Any],
    result: dict[str, Any],
) -> StepManifest:
    """Placeholder: extract a validation step manifest.

    Populated by later slices — currently returns a generic manifest
    with validation refs from the result if detectable.
    """
    validation_refs: list[str] = []
    for key in ("validation_id", "validated", "is_valid"):
        val = result.get(key)
        if val is not None:
            validation_refs.append(str(val))

    return StepManifest(
        operation_type="validation",
        operation_label="Validation",
        endpoint=endpoint,
        validation_refs=validation_refs,
        status=_infer_status(result),
        warnings=_extract_warnings(result),
        errors=_extract_errors(result),
    )


def extract_writeback_draft(
    endpoint: str,
    request_body: dict[str, Any],
    result: dict[str, Any],
) -> StepManifest:
    """Placeholder: extract a writeback draft step manifest.

    Populated by later slices — currently returns a generic manifest.
    """
    output_refs = _extract_ids_from_sequences(result.get("sequences"))

    return StepManifest(
        operation_type="writeback_draft",
        operation_label="Writeback draft",
        endpoint=endpoint,
        output_refs=output_refs,
        status=_infer_status(result),
        warnings=_extract_warnings(result),
        errors=_extract_errors(result),
    )


# --- Generic fallback --------------------------------------------------------


def extract_generic(
    endpoint: str,
    request_body: dict[str, Any],
    result: dict[str, Any],
) -> StepManifest:
    """Fallback extractor for unrecognized endpoints.

    Extracts endpoint path, top-level input/output refs from the
    result if detectable.
    """
    output_refs = _extract_ids_from_sequences(result.get("sequences"))
    if not output_refs:
        # Maybe the result has a single 'id' field at the top level.
        rid = result.get("id")
        if rid is not None:
            output_refs = [str(rid)]

    input_refs = _extract_input_seq_ids(result.get("sources"))
    if not input_refs:
        input_refs = _extract_ids_from_sequences(request_body.get("sequences"))

    return StepManifest(
        operation_type="generic_opencloning_operation",
        operation_label=_endpoint_label(endpoint),
        endpoint=endpoint,
        input_refs=input_refs,
        output_refs=output_refs,
        status=_infer_status(result),
        warnings=_extract_warnings(result),
        errors=_extract_errors(result),
    )


# --- Dispatcher --------------------------------------------------------------


_ENDPOINT_ROUTING: dict[str, str] = {
    "/read_from_file": "sequence_import",
    "/manually_typed": "sequence_import",
    "/pcr": "pcr",
    "/restriction": "restriction_digest",
    "/gibson_assembly": "assembly",
    "/ligation": "assembly",
    "/restriction_and_ligation": "assembly",
    "/homologous_recombination": "assembly",
    "/crispr": "assembly",
    "/cre_lox_recombination": "assembly",
    "/gateway": "assembly",
    "/recombinase": "assembly",
    "/polymerase_extension": "assembly",
    "/reverse_complement": "assembly",
    "/oligonucleotide_hybridization": "assembly",
    "/annotate/plannotate": "annotation",
    "/primer_details": "primer_design",
    "/validate": "validation",
    "/validate_syntax": "validation",
}

_EXTRACTOR_DISPATCH: dict[str, Any] = {
    "sequence_import": extract_sequence_import,
    "primer_design": extract_primer_design,
    "pcr": extract_pcr,
    "restriction_digest": extract_restriction_digest,
    "assembly": extract_assembly,
    "annotation": extract_annotation,
    "protocol_lookup": extract_protocol_lookup,
    "validation": extract_validation,
    "writeback_draft": extract_writeback_draft,
}


def _route_endpoint(endpoint: str) -> str:
    """Map an endpoint path to an operation type."""
    exact = _ENDPOINT_ROUTING.get(endpoint)
    if exact:
        return exact
    # Prefix matches.
    if endpoint.startswith("/primer_design/"):
        return "primer_design"
    if endpoint.startswith("/batch_cloning/"):
        return "generic_opencloning_operation"
    if endpoint.startswith("/repository_id/"):
        return "sequence_import"
    if endpoint.startswith("/protocol"):
        return "protocol_lookup"
    if endpoint.startswith("/annotate/"):
        return "annotation"
    return "generic_opencloning_operation"


def extract_step_manifest(
    endpoint: str,
    request_body: dict[str, Any],
    result: dict[str, Any],
    run_id: str,
    step_id: str,
    parent_step_ids: list[str] | None = None,
) -> StepManifest:
    """Dispatch to the appropriate typed extractor based on endpoint.

    Args:
        endpoint: The OpenCloning API endpoint path (e.g. ``/pcr``).
        request_body: The request body sent to the endpoint.
        result: The raw OpenCloning API result dict.
        run_id: Identifier for the overall plan run.
        step_id: Unique identifier for this step.
        parent_step_ids: Optional list of parent step IDs.

    Returns:
        A populated ``StepManifest``.
    """
    operation_type = _route_endpoint(endpoint)
    extractor = _EXTRACTOR_DISPATCH.get(operation_type)
    if extractor is None:
        extractor = extract_generic

    manifest = extractor(endpoint, request_body, result)
    manifest.schema = "opencloning_step_manifest_v1"
    manifest.run_id = run_id
    manifest.step_id = step_id
    manifest.parent_step_ids = list(parent_step_ids or [])
    if not manifest.operation_type:
        manifest.operation_type = operation_type
    return manifest
