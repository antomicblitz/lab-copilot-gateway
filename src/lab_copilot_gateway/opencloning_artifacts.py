"""OpenCloning result normalization for chat artifact manifests (C52).

This module converts OpenCloning's raw ``{sources, sequences}`` responses into
compact, user-facing artifact manifests.  It does not infer biological
correctness; it only exposes data already present in OpenCloning output and
marks missing deterministic bytes as blocking for writeback.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from lab_copilot_gateway.artifact_bundle import DEFAULT_MAX_ARTIFACT_BYTES


GENBANK_MIME_TYPE = "chemical/x-genbank"
FASTA_MIME_TYPE = "chemical/x-fasta"


@dataclass(frozen=True)
class NormalizedOpenCloningArtifacts:
    """Normalized OpenCloning artifact payload.

    ``artifact_bytes`` is intentionally separate from ``artifacts`` so callers
    can store bytes server-side while sending only manifests to the chat UI.
    """

    summary: str
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    artifact_bytes: dict[str, bytes] = field(default_factory=dict)
    warnings: list[dict[str, str]] = field(default_factory=list)
    blocking_errors: list[dict[str, str]] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_event_payload(self) -> dict[str, Any]:
        """Return the chat/SSE-safe payload without raw artifact bytes."""
        return {
            "summary": self.summary,
            "artifacts": self.artifacts,
            "warnings": self.warnings,
            "blocking_errors": self.blocking_errors,
            "provenance": self.provenance,
        }


def _normalize_empty_sequences_result(
    result: dict[str, Any],
    warnings: list[dict[str, Any]],
    blocking_errors: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> NormalizedOpenCloningArtifacts:
    """Handle OpenCloning results that have no sequences.

    Determines whether the result is a valid non-sequence output (primers,
    analysis, rename) or a genuine failure, and returns the appropriate
    NormalizedOpenCloningArtifacts.
    """
    # Primer design endpoints return {primers: [...]} with no sequences.
    if result.get("primers"):
        return NormalizedOpenCloningArtifacts(
            summary=f"Designed {len(result['primers'])} primers.",
            warnings=warnings,
            blocking_errors=blocking_errors,
            provenance=provenance,
        )
    # Analysis endpoints return analysis results, not sequences.
    analysis_keys = {
        "melting_temperature", "tm", "gc_content",
        "heterodimer", "self_dimer", "hairpin",
    }
    if any(k in result for k in analysis_keys):
        return NormalizedOpenCloningArtifacts(
            summary="Analysis completed.",
            warnings=warnings,
            blocking_errors=blocking_errors,
            provenance=provenance,
        )
    # Rename returns sources but no new sequences — not a failure.
    if result.get("sources") and any(
        s.get("type") == "RenameSequenceSource"
        for s in result.get("sources", [])
        if isinstance(s, dict)
    ):
        return NormalizedOpenCloningArtifacts(
            summary="Sequence renamed.",
            warnings=warnings,
            blocking_errors=blocking_errors,
            provenance=provenance,
        )
    blocking_errors.append({
        "severity": "blocking_error",
        "code": "no_final_sequence",
        "message": "OpenCloning returned no final product sequence.",
    })
    return NormalizedOpenCloningArtifacts(
        summary="OpenCloning produced no final construct.",
        warnings=warnings,
        blocking_errors=blocking_errors,
        provenance=provenance,
    )


def normalize_opencloning_artifacts(
    result: dict[str, Any],
    *,
    plan_id: str,
    plan_hash: str,
    operation_label: str = "OpenCloning workflow",
    source_inputs: list[str] | None = None,
    artifact_id_prefix: str = "oc-artifact",
) -> NormalizedOpenCloningArtifacts:
    """Build compact artifact manifests from an OpenCloning response.

    Candidate final products are the returned ``sequences`` entries.  If there
    are multiple sequences, V1 exposes each as a separate artifact and adds a
    non-blocking warning instead of guessing which construct the user intended.
    """
    raw_sources = result.get("sources")
    sources: list[Any] = raw_sources if isinstance(raw_sources, list) else []
    raw_sequences = result.get("sequences")
    sequences: list[Any] = raw_sequences if isinstance(raw_sequences, list) else []
    warnings = _normalize_warnings(result.get("warnings"))
    blocking_errors = _normalize_blocking_errors(result.get("errors"))

    provenance = {
        "source_inputs": list(source_inputs or []),
        "opencloning_steps": _provenance_steps(operation_label, sources, sequences),
        "source_count": len(sources),
        "sequence_count": len(sequences),
        "operation": operation_label,
    }

    if not sequences:
        return _normalize_empty_sequences_result(
            result, warnings, blocking_errors, provenance
        )

    if len(sequences) > 1:
        warnings.append(
            {
                "severity": "warning",
                "code": "multiple_final_products",
                "message": (
                    "OpenCloning returned multiple possible final products; "
                    "review filenames before writeback."
                ),
            }
        )

    artifacts: list[dict[str, Any]] = []
    artifact_bytes: dict[str, bytes] = {}
    for index, sequence in enumerate(sequences, start=1):
        if not isinstance(sequence, dict):
            continue
        normalized = _normalize_sequence_artifact(
            sequence,
            sources=sources,
            index=index,
            plan_id=plan_id,
            plan_hash=plan_hash,
            artifact_id=f"{artifact_id_prefix}-{index}",
            operation_label=operation_label,
            warnings=warnings,
            provenance=provenance,
        )
        if normalized is None:
            blocking_errors.append(
                {
                    "severity": "blocking_error",
                    "code": "missing_artifact_bytes",
                    "message": (
                        "OpenCloning returned a sequence without deterministic "
                        "file_content for download/writeback."
                    ),
                }
            )
            continue
        artifact, data = normalized
        artifacts.append(artifact)
        artifact_bytes[artifact["artifact_id"]] = data

    if not artifacts and not blocking_errors:
        blocking_errors.append(
            {
                "severity": "blocking_error",
                "code": "no_writeback_eligible_artifact",
                "message": "No deterministic OpenCloning artifact could be created.",
            }
        )

    return NormalizedOpenCloningArtifacts(
        summary=_summary_for_artifacts(artifacts, blocking_errors),
        artifacts=artifacts,
        artifact_bytes=artifact_bytes,
        warnings=warnings,
        blocking_errors=blocking_errors,
        provenance=provenance,
    )


def _normalize_sequence_artifact(
    sequence: dict[str, Any],
    *,
    sources: list[Any],
    index: int,
    plan_id: str,
    plan_hash: str,
    artifact_id: str,
    operation_label: str,
    warnings: list[dict[str, str]],
    provenance: dict[str, Any],
) -> tuple[dict[str, Any], bytes] | None:
    file_content = sequence.get("file_content")
    if not isinstance(file_content, str) or not file_content:
        return None

    file_format = str(sequence.get("sequence_file_format") or "genbank").lower()
    kind = "fasta" if file_format in {"fasta", "fa"} else "genbank"
    mime_type = FASTA_MIME_TYPE if kind == "fasta" else GENBANK_MIME_TYPE
    fallback_name = _source_output_name(sequence, sources) or f"construct-{index}"
    summary = _sequence_summary(sequence, file_content, fallback_name=fallback_name)
    filename = _filename_for(summary["name"], kind, index)
    data = file_content.encode("utf-8")
    artifact_warnings = list(warnings)

    writeback_eligible = len(data) <= DEFAULT_MAX_ARTIFACT_BYTES
    if not writeback_eligible:
        artifact_warnings.append(
            {
                "severity": "blocking_error",
                "code": "artifact_too_large",
                "message": "Artifact exceeds the 1 MiB writeback limit.",
            }
        )

    artifact = {
        "artifact_id": artifact_id,
        "plan_id": plan_id,
        "plan_hash": plan_hash,
        "kind": kind,
        "role": "final_product",
        "filename": filename,
        "mime_type": mime_type,
        "size_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "description": f"{operation_label} product",
        "sequence_summary": summary,
        "warnings": artifact_warnings,
        "provenance": provenance,
        "exports": [
            {
                "kind": kind,
                "artifact_id": artifact_id,
                "filename": filename,
            }
        ],
        "opencloning_handoff": {
            "mode": "manual_import",
            "url": "/opencloning",
            "instructions": (
                "Open OpenCloning and load the downloaded artifact or "
                "cloning-history JSON when available."
            ),
        },
        "download_url": f"/__copilot/api/v1/artifacts/{artifact_id}/download",
        "writeback_eligible": writeback_eligible,
    }
    return artifact, data


def _sequence_summary(
    sequence: dict[str, Any], file_content: str, *, fallback_name: str
) -> dict[str, Any]:
    raw_name = sequence.get("name") or _genbank_locus_name(file_content)
    name = str(raw_name if raw_name and not _generic_name(raw_name) else fallback_name)
    length_bp = _int_or_none(sequence.get("length")) or _genbank_length(file_content)
    circular = _bool_or_none(sequence.get("circular"))
    if circular is None:
        circular = _genbank_circular(file_content)
    feature_count = _int_or_none(sequence.get("feature_count"))
    if feature_count is None:
        feature_count = _genbank_feature_count(file_content)
    features = _genbank_features(file_content)
    return {
        "name": name,
        "length_bp": length_bp,
        "circular": circular,
        "feature_count": feature_count,
        "features": features,
    }


def _generic_name(name: Any) -> bool:
    return str(name or "").strip().lower() in {
        "name",
        "sequence",
        "untitled",
        "untitled_sequence",
    }


def _normalize_warnings(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    normalized: list[dict[str, str]] = []
    for index, item in enumerate(raw, start=1):
        if isinstance(item, dict):
            severity = str(item.get("severity") or "warning")
            code = str(item.get("code") or f"opencloning_warning_{index}")
            message = str(item.get("message") or item.get("detail") or item)
        else:
            severity = "warning"
            code = f"opencloning_warning_{index}"
            message = str(item)
        normalized.append({"severity": severity, "code": code, "message": message})
    return normalized


def _normalize_blocking_errors(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list):
        return []
    normalized: list[dict[str, str]] = []
    for index, item in enumerate(raw, start=1):
        if isinstance(item, dict):
            code = str(item.get("code") or f"opencloning_error_{index}")
            message = str(item.get("message") or item.get("detail") or item)
        else:
            code = f"opencloning_error_{index}"
            message = str(item)
        normalized.append(
            {"severity": "blocking_error", "code": code, "message": message}
        )
    return normalized


def _provenance_steps(
    operation_label: str, sources: list[Any], sequences: list[Any]
) -> list[str]:
    steps = []
    if sources:
        steps.append(f"Processed {len(sources)} OpenCloning source(s)")
    steps.append(operation_label)
    if sequences:
        steps.append(f"Produced {len(sequences)} final sequence(s)")
    return steps


def _summary_for_artifacts(
    artifacts: list[dict[str, Any]], blocking_errors: list[dict[str, str]]
) -> str:
    if not artifacts:
        return "OpenCloning did not produce a writeback-ready construct."
    if len(artifacts) == 1:
        seq = artifacts[0]["sequence_summary"]
        topology = "circular" if seq.get("circular") else "linear"
        return f"Produced 1 {topology} final construct ({seq.get('length_bp')} bp)."
    if blocking_errors:
        return f"Produced {len(artifacts)} construct artifacts with blocking warnings."
    return f"Produced {len(artifacts)} possible final construct artifacts."


def _filename_for(name: str, kind: str, index: int) -> str:
    if _generic_name(name):
        name = f"construct-{index}"
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", name).strip(".-") or f"construct-{index}"
    extension = ".fasta" if kind == "fasta" else ".gb"
    if safe.lower().endswith(extension):
        return safe
    return f"{safe}{extension}"


def _source_output_name(sequence: dict[str, Any], sources: list[Any]) -> str | None:
    seq_id = sequence.get("id")
    for source in sources:
        if not isinstance(source, dict):
            continue
        if source.get("id") == seq_id and source.get("output_name"):
            return str(source["output_name"])
    return None


def _genbank_locus_name(file_content: str) -> str | None:
    match = re.search(r"^LOCUS\s+(\S+)", file_content, flags=re.MULTILINE)
    return match.group(1) if match else None


def _genbank_length(file_content: str) -> int | None:
    match = re.search(r"^LOCUS\s+\S+\s+(\d+)\s+bp\b", file_content, flags=re.MULTILINE)
    if match:
        return int(match.group(1))
    origin_match = re.search(
        r"^ORIGIN\n(?P<body>.*?)(?:^//|\Z)",
        file_content,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not origin_match:
        return None
    bases = re.sub(r"[^A-Za-z]", "", origin_match.group("body"))
    return len(bases) if bases else None


def _genbank_circular(file_content: str) -> bool | None:
    locus = next(
        (line for line in file_content.splitlines() if line.startswith("LOCUS")), ""
    )
    lowered = locus.lower()
    if "circular" in lowered:
        return True
    if "linear" in lowered:
        return False
    return None


def _genbank_feature_count(file_content: str) -> int:
    in_features = False
    count = 0
    for line in file_content.splitlines():
        if line.startswith("FEATURES"):
            in_features = True
            continue
        if in_features and (line.startswith("ORIGIN") or line.startswith("//")):
            break
        if in_features and re.match(r"^\s{5}\S", line):
            count += 1
    return count


def _genbank_features(file_content: str) -> list[dict[str, str]]:
    """Extract feature types, locations, and labels from a GenBank file.

    Returns a list of {type, location, label, strand} dicts.  The LLM
    uses these to decide which regions to PCR-amplify, digest, etc.
    """
    features: list[dict[str, str]] = []
    in_features = False
    current: dict[str, str] | None = None
    for line in file_content.splitlines():
        if line.startswith("FEATURES"):
            in_features = True
            continue
        if in_features and (line.startswith("ORIGIN") or line.startswith("//")):
            break
        if not in_features:
            continue
        # Feature lines start at column 5 with a type name.
        feat_match = re.match(r"^\s{5}(\S+)\s+(.+)$", line)
        if feat_match:
            if current:
                features.append(current)
            ftype = feat_match.group(1)
            location = feat_match.group(2).strip()
            strand = "reverse" if "complement" in location else "forward"
            current = {
                "type": ftype,
                "location": location,
                "strand": strand,
                "label": "",
            }
        elif current and re.match(r"^\s+/", line):
            qual_match = re.match(r"^\s+/(\w+)=(.*)$", line)
            if qual_match and qual_match.group(1) == "label":
                current["label"] = qual_match.group(2).strip().strip('"')
            elif qual_match and qual_match.group(1) == "gene":
                if not current["label"]:
                    current["label"] = qual_match.group(2).strip().strip('"')
            elif qual_match and qual_match.group(1) == "product":
                if not current["label"]:
                    current["label"] = qual_match.group(2).strip().strip('"')
    if current:
        features.append(current)
    # Limit to 50 features to keep the manifest compact.
    return features[:50]


def _int_or_none(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None
