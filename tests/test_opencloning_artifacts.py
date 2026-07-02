"""Tests for OpenCloning result normalization (C52)."""

from __future__ import annotations

import hashlib

from lab_copilot_gateway.opencloning_artifacts import normalize_opencloning_artifacts


GENBANK = """LOCUS       pDemo                   12 bp    DNA     circular SYN 01-JUL-2026
FEATURES             Location/Qualifiers
     source          1..12
     CDS             1..9
ORIGIN
        1 atgcgatcgatc
//
"""


def test_normalize_gibson_result_creates_final_product_manifest() -> None:
    """A standard OpenCloning response becomes one GenBank manifest."""
    result = {
        "sources": [{"id": 0, "type": "UploadedFileSource"}],
        "sequences": [
            {
                "id": 1,
                "type": "TextFileSequence",
                "sequence_file_format": "genbank",
                "file_content": GENBANK,
            }
        ],
    }

    normalized = normalize_opencloning_artifacts(
        result,
        plan_id="plan-1",
        plan_hash="hash-1",
        operation_label="Simulated Gibson assembly",
        source_inputs=["insert.gb", "vector.gb"],
    )

    assert normalized.summary == "Produced 1 circular final construct (12 bp)."
    assert normalized.blocking_errors == []
    assert len(normalized.artifacts) == 1
    artifact = normalized.artifacts[0]
    assert artifact["artifact_id"] == "oc-artifact-1"
    assert artifact["plan_id"] == "plan-1"
    assert artifact["plan_hash"] == "hash-1"
    assert artifact["role"] == "final_product"
    assert artifact["kind"] == "genbank"
    assert artifact["filename"] == "pDemo.gb"
    assert artifact["mime_type"] == "chemical/x-genbank"
    assert artifact["size_bytes"] == len(GENBANK.encode())
    assert artifact["sha256"] == hashlib.sha256(GENBANK.encode()).hexdigest()
    assert artifact["sequence_summary"]["name"] == "pDemo"
    assert artifact["sequence_summary"]["length_bp"] == 12
    assert artifact["sequence_summary"]["circular"] is True
    assert artifact["sequence_summary"]["feature_count"] == 2
    assert len(artifact["sequence_summary"]["features"]) == 2
    assert artifact["sequence_summary"]["features"][0]["type"] == "source"
    assert artifact["sequence_summary"]["features"][1]["type"] == "CDS"
    assert artifact["provenance"]["source_inputs"] == ["insert.gb", "vector.gb"]
    assert artifact["opencloning_handoff"]["url"] == "/opencloning"
    assert artifact["writeback_eligible"] is True
    assert "file_content" not in artifact
    assert normalized.artifact_bytes["oc-artifact-1"] == GENBANK.encode()
    assert "artifact_bytes" not in normalized.to_event_payload()


def test_generic_locus_name_uses_source_output_name() -> None:
    """OpenCloning's placeholder LOCUS 'name' should not become name.gb."""
    generic_genbank = GENBANK.replace("pDemo", "name")
    normalized = normalize_opencloning_artifacts(
        {
            "sources": [
                {
                    "id": 5,
                    "type": "GibsonAssemblySource",
                    "output_name": "pCambia2300-Pat-EGFP-chlorella",
                }
            ],
            "sequences": [
                {
                    "id": 5,
                    "type": "TextFileSequence",
                    "sequence_file_format": "genbank",
                    "file_content": generic_genbank,
                }
            ],
        },
        plan_id="plan-1",
        plan_hash="hash-1",
    )

    artifact = normalized.artifacts[0]
    assert artifact["filename"] == "pCambia2300-Pat-EGFP-chlorella.gb"
    assert artifact["sequence_summary"]["name"] == "pCambia2300-Pat-EGFP-chlorella"


def test_normalize_multiple_products_returns_manifests_with_warning() -> None:
    """Multiple returned sequences are exposed without guessing intent."""
    result = {
        "sources": [],
        "sequences": [
            {
                "id": 1,
                "type": "TextFileSequence",
                "sequence_file_format": "genbank",
                "file_content": GENBANK.replace("pDemo", "pOne"),
            },
            {
                "id": 2,
                "type": "TextFileSequence",
                "sequence_file_format": "genbank",
                "file_content": GENBANK.replace("pDemo", "pTwo"),
            },
        ],
    }

    normalized = normalize_opencloning_artifacts(
        result,
        plan_id="plan-1",
        plan_hash="hash-1",
        operation_label="Simulated ligation",
    )

    assert len(normalized.artifacts) == 2
    assert [a["filename"] for a in normalized.artifacts] == ["pOne.gb", "pTwo.gb"]
    assert any(w["code"] == "multiple_final_products" for w in normalized.warnings)
    assert normalized.blocking_errors == []


def test_normalize_no_final_sequence_blocks_writeback() -> None:
    """No sequences means no downloadable/writeback artifact."""
    normalized = normalize_opencloning_artifacts(
        {"sources": [{"id": 0}], "sequences": []},
        plan_id="plan-1",
        plan_hash="hash-1",
    )

    assert normalized.artifacts == []
    assert normalized.artifact_bytes == {}
    assert normalized.blocking_errors[0]["code"] == "no_final_sequence"
    assert normalized.summary == "OpenCloning produced no final construct."


def test_normalize_missing_file_content_blocks_writeback() -> None:
    """A sequence without deterministic bytes is not writeback eligible."""
    normalized = normalize_opencloning_artifacts(
        {"sources": [], "sequences": [{"id": 1, "length": 12, "circular": True}]},
        plan_id="plan-1",
        plan_hash="hash-1",
    )

    assert normalized.artifacts == []
    assert normalized.blocking_errors[0]["code"] == "missing_artifact_bytes"


def test_normalize_validation_warnings_are_non_blocking_with_final_product() -> None:
    """OpenCloning warnings remain visible but do not block valid artifacts."""
    result = {
        "warnings": [
            {
                "severity": "warning",
                "code": "low_homology",
                "message": "Overlap is shorter than recommended.",
            }
        ],
        "sources": [],
        "sequences": [
            {
                "id": 1,
                "type": "TextFileSequence",
                "sequence_file_format": "genbank",
                "file_content": GENBANK,
            }
        ],
    }

    normalized = normalize_opencloning_artifacts(
        result,
        plan_id="plan-1",
        plan_hash="hash-1",
    )

    assert normalized.blocking_errors == []
    assert normalized.artifacts[0]["writeback_eligible"] is True
    assert normalized.warnings == [
        {
            "severity": "warning",
            "code": "low_homology",
            "message": "Overlap is shorter than recommended.",
        }
    ]
    assert normalized.artifacts[0]["warnings"] == normalized.warnings
