"""Tests for OpenCloning step manifest contract and typed extractors."""

from __future__ import annotations

import json

from lab_copilot_gateway.opencloning_steps import (
    StepManifest,
    extract_step_manifest,
    extract_pcr,
    extract_assembly,
    extract_sequence_import,
)

# --- Constants ---------------------------------------------------------------

RUN_ID = "run-abc-123"
STEP_ID = "step-001"
PARENT_IDS = ["step-000"]

GENBANK_TEMPLATE = """LOCUS       pTemplate              3200 bp    DNA     circular SYN 01-JUL-2026
FEATURES             Location/Qualifiers
     source          1..3200
     CDS             complement(100..1200)
     promoter        1400..1600
     terminator      1800..2000
ORIGIN
        1 atgcgatcgatcagctagctagctagctagctagctagctagctagctagc
       61 tagctagctagctagctagctagctagctagctagctagctagctagctagc
      121 tagctagctagctagctagctagctagctagctagctagctagctagctagc
      181 tagctagctagctagctagctagctagctagctagctagctagctagctagc
      ...
     3200 atgcatgcatgc
//
"""

GENBANK_PRODUCT = """LOCUS       pProduct                1200 bp    DNA     linear SYN 01-JUL-2026
FEATURES             Location/Qualifiers
     source          1..1200
     CDS             complement(1..1100)
ORIGIN
        1 atgcgatcgatcagctagctagctagctagctagctagctagctagctagc
       61 tagctagctagctagctagctagctagctagctagctagctagctagctagc
      ...
     1200 atgcatgcatgc
//
"""


# --- Helpers -----------------------------------------------------------------


def _make_manifest(endpoint: str, body: dict, result: dict) -> StepManifest:
    return extract_step_manifest(
        endpoint=endpoint,
        request_body=body,
        result=result,
        run_id=RUN_ID,
        step_id=STEP_ID,
        parent_step_ids=PARENT_IDS,
    )


# --- PCR result: typed manifest with template, region, primers, product -------


def test_pcr_produces_typed_manifest() -> None:
    """PCR result produces a typed manifest with template, region, primers,
    product refs, warnings, and preview_ref=None."""
    request_body = {
        "sequences": [
            {
                "id": 1,
                "type": "TextFileSequence",
                "file_content": GENBANK_TEMPLATE,
                "circular": True,
            }
        ],
        "pcr_templates": [
            {
                "sequence": {"id": 1, "type": "TextFileSequence"},
                "location": {"start": 100, "end": 1300, "strand": "forward"},
            }
        ],
        "source": {"id": 0, "type": "PCRSource", "input": [{"sequence": 1}]},
    }
    result = {
        "sources": [
            {
                "id": 2,
                "type": "PCRSource",
                "input": [{"sequence": 1}],
                "output_name": "pcr_product",
            }
        ],
        "sequences": [
            {
                "id": 3,
                "type": "TextFileSequence",
                "file_content": GENBANK_PRODUCT,
                "length": 1200,
                "circular": False,
            }
        ],
        "primers": [
            {"id": 1, "name": "fwd_primer", "sequence": "ATGCATGC"},
            {"id": 2, "name": "rev_primer", "sequence": "GCATGCAT"},
        ],
        "warnings": [
            {
                "severity": "warning",
                "code": "primer_tm_warning",
                "message": "Primer Tm difference exceeds 5°C.",
            }
        ],
    }

    manifest = _make_manifest("/pcr", request_body, result)

    assert manifest.schema == "opencloning_step_manifest_v1"
    assert manifest.run_id == RUN_ID
    assert manifest.step_id == STEP_ID
    assert manifest.parent_step_ids == PARENT_IDS
    assert manifest.operation_type == "pcr"
    assert manifest.operation_label == "PCR amplification"
    assert manifest.endpoint == "/pcr"
    assert manifest.status == "succeeded"
    assert manifest.tool_name == "opencloning.call"
    assert manifest.input_refs == ["1"]
    assert manifest.output_refs == ["3"]
    assert manifest.template_refs == ["1"]
    assert manifest.region_refs == ["template-0:start=100,end=1300,strand=forward"]
    assert manifest.primer_refs == ["1", "2"]
    assert manifest.preview_ref is None
    assert manifest.sequence_byte_policy == "server_side_only"
    assert len(manifest.warnings) == 1
    assert manifest.warnings[0]["code"] == "primer_tm_warning"


def test_pcr_manifest_to_dict_matches_schema() -> None:
    """to_dict() produces the expected schema shape with snake_case keys."""
    request_body = {
        "sequences": [{"id": 1, "type": "TextFileSequence"}],
        "source": {"id": 0, "type": "PCRSource", "input": [{"sequence": 1}]},
    }
    result = {
        "sources": [{"id": 2, "type": "PCRSource", "input": [{"sequence": 1}]}],
        "sequences": [{"id": 3, "type": "TextFileSequence", "length": 500}],
    }

    manifest = _make_manifest("/pcr", request_body, result)
    d = manifest.to_dict()

    assert d["schema"] == "opencloning_step_manifest_v1"
    assert d["run_id"] == RUN_ID
    assert d["step_id"] == STEP_ID
    assert d["parent_step_ids"] == PARENT_IDS
    assert d["operation_type"] == "pcr"
    assert d["operation_label"] == "PCR amplification"
    assert d["endpoint"] == "/pcr"
    assert d["status"] == "succeeded"
    assert d["tool_name"] == "opencloning.call"
    assert d["input_refs"] == ["1"]
    assert d["output_refs"] == ["3"]
    assert d["template_refs"] == ["1"]
    assert d["region_refs"] == []
    assert d["primer_refs"] == []
    assert d["preview_ref"] is None
    assert d["sequence_byte_policy"] == "server_side_only"
    assert d["warnings"] == []
    assert d["errors"] == []
    assert "created_at" in d

    # Verify JSON serializable.
    serialized = json.dumps(d)
    assert isinstance(serialized, str)
    assert "server_side_only" in serialized


def test_pcr_with_feature_loss_warning() -> None:
    """PCR with feature loss warning carries the warning in the manifest."""
    request_body = {
        "sequences": [{"id": 1, "type": "TextFileSequence"}],
        "source": {"id": 0, "type": "PCRSource", "input": [{"sequence": 1}]},
    }
    result = {
        "sources": [{"id": 2, "type": "PCRSource", "input": [{"sequence": 1}]}],
        "sequences": [{"id": 3, "type": "TextFileSequence", "length": 500}],
        "feature_loss_warning": {
            "feature_loss_warning": True,
            "template_features": 3,
            "product_features": 1,
            "message": "PCR product has 1 features but template had 3.",
        },
    }

    manifest = _make_manifest("/pcr", request_body, result)
    assert len(manifest.warnings) == 1
    assert manifest.warnings[0]["feature_loss_warning"] is True


def test_pcr_with_errors_sets_failed_status() -> None:
    """PCR with errors produces 'failed' status."""
    request_body = {
        "sequences": [{"id": 1, "type": "TextFileSequence"}],
        "source": {"id": 0, "type": "PCRSource", "input": [{"sequence": 1}]},
    }
    result = {
        "errors": [{"code": "pcr_failed", "message": "No product generated from PCR."}],
        "sources": [{"id": 2, "type": "PCRSource", "input": [{"sequence": 1}]}],
        "sequences": [],
    }

    manifest = _make_manifest("/pcr", request_body, result)
    assert manifest.status == "failed"
    assert len(manifest.errors) == 1
    assert manifest.errors[0]["code"] == "pcr_failed"
    assert manifest.output_refs == []


# --- Assembly result: typed manifest with fragments, method, topology ---------


def test_assembly_produces_typed_manifest() -> None:
    """Assembly result produces a typed manifest with fragment inputs,
    circular/linear result, output ref, warnings."""
    request_body = {
        "sequences": [
            {"id": 1, "type": "TextFileSequence"},
            {"id": 2, "type": "TextFileSequence"},
        ],
        "source": {
            "id": 0,
            "type": "GibsonAssemblySource",
            "input": [{"sequence": 1}, {"sequence": 2}],
        },
    }
    result = {
        "sources": [
            {
                "id": 3,
                "type": "GibsonAssemblySource",
                "input": [{"sequence": 1}, {"sequence": 2}],
                "output_name": "assembled_construct",
            }
        ],
        "sequences": [
            {
                "id": 4,
                "type": "TextFileSequence",
                "file_content": "LOCUS...",
                "length": 5000,
                "circular": True,
            }
        ],
        "warnings": [
            {
                "severity": "warning",
                "code": "low_homology",
                "message": "Overlap is shorter than recommended.",
            }
        ],
    }

    manifest = _make_manifest("/gibson_assembly", request_body, result)

    assert manifest.operation_type == "assembly"
    assert manifest.operation_label == "Gibson assembly"
    assert manifest.endpoint == "/gibson_assembly"
    assert manifest.status == "succeeded"
    assert manifest.input_refs == ["1", "2"]
    assert manifest.output_refs == ["4"]
    assert manifest.preview_ref is None
    assert len(manifest.warnings) == 1
    assert manifest.warnings[0]["code"] == "low_homology"


def test_assembly_ligation_roundtrip() -> None:
    """Ligation assembly produces correct operation_type and label."""
    request_body = {
        "sequences": [{"id": 1, "type": "TextFileSequence"}],
        "source": {
            "id": 0,
            "type": "LigationSource",
            "input": [{"sequence": 1}],
        },
    }
    result = {
        "sources": [{"id": 2, "type": "LigationSource", "input": [{"sequence": 1}]}],
        "sequences": [
            {"id": 3, "type": "TextFileSequence", "length": 100, "circular": True}
        ],
    }

    manifest = _make_manifest("/ligation", request_body, result)
    assert manifest.operation_type == "assembly"
    assert manifest.operation_label == "Ligation"
    assert manifest.output_refs == ["3"]


def test_assembly_circular_detected() -> None:
    """Assembly with circular output is correctly detected."""
    request_body = {
        "sequences": [{"id": 1, "type": "TextFileSequence"}],
        "source": {
            "id": 0,
            "type": "GibsonAssemblySource",
            "input": [{"sequence": 1}],
        },
    }
    result = {
        "sources": [
            {"id": 2, "type": "GibsonAssemblySource", "input": [{"sequence": 1}]}
        ],
        "sequences": [
            {"id": 3, "type": "TextFileSequence", "length": 100, "circular": True}
        ],
    }

    manifest = _make_manifest("/gibson_assembly", request_body, result)
    assert manifest.operation_type == "assembly"
    assert manifest.output_refs == ["3"]


# --- Primer design: typed manifest with primer data --------------------------


def test_primer_design_produces_typed_manifest() -> None:
    """Primer design result produces a typed manifest with primer IDs."""
    request_body = {
        "pcr_templates": [{"sequence": {"id": 1, "type": "TextFileSequence"}}],
        "source": {"id": 0, "type": "PrimerDesignSource"},
    }
    result = {
        "primers": [
            {
                "id": 1,
                "name": "forward_primer",
                "sequence": "ATGCATGCATGC",
                "tm": 62.5,
            },
            {
                "id": 2,
                "name": "reverse_primer",
                "sequence": "GCATGCATGCAT",
                "tm": 55.0,
            },
        ],
        "warnings": [],
    }

    manifest = _make_manifest("/primer_design/gibson_assembly", request_body, result)

    assert manifest.operation_type == "primer_design"
    assert manifest.operation_label == "Primer design — Gibson Assembly"
    assert manifest.endpoint == "/primer_design/gibson_assembly"
    assert manifest.primer_refs == ["1", "2"]
    assert manifest.template_refs == ["1"]
    # Tm difference > 5°C should add a warning.
    assert any(w["code"] == "primer_tm_mismatch" for w in manifest.warnings)


def test_primer_design_no_tm_warning_when_tm_close() -> None:
    """Primer design with close Tm values should NOT add tm_mismatch warning."""
    request_body = {
        "pcr_templates": [{"sequence": {"id": 1, "type": "TextFileSequence"}}],
        "source": {"id": 0, "type": "PrimerDesignSource"},
    }
    result = {
        "primers": [
            {"id": 1, "name": "fwd", "sequence": "ATGC", "tm": 60.0},
            {"id": 2, "name": "rev", "sequence": "GCAT", "tm": 62.0},
        ],
    }

    manifest = _make_manifest("/primer_design/simple_pair", request_body, result)
    assert manifest.operation_type == "primer_design"
    assert manifest.primer_refs == ["1", "2"]
    assert not any(w["code"] == "primer_tm_mismatch" for w in manifest.warnings)


def test_primer_design_with_single_template_dict() -> None:
    """Primer design with single pcr_template (dict, not list)."""
    request_body = {
        "pcr_template": {"sequence": {"id": 5, "type": "TextFileSequence"}},
        "source": {"id": 0, "type": "PrimerDesignSource"},
    }
    result = {
        "primers": [{"id": 10, "name": "fwd", "sequence": "ATGC"}],
    }

    manifest = _make_manifest("/primer_design/simple_pair", request_body, result)
    assert manifest.template_refs == ["5"]
    assert manifest.primer_refs == ["10"]


# --- Sequence import: typed manifest with file format ------------------------


def test_sequence_import_read_from_file() -> None:
    """Sequence import from file produces typed manifest with format."""
    request_body = {
        "file_format": "genbank",
    }
    result = {
        "sources": [{"id": 1, "type": "UploadedFileSource"}],
        "sequences": [
            {
                "id": 2,
                "type": "TextFileSequence",
                "file_content": "LOCUS...",
                "length": 3200,
                "circular": True,
            }
        ],
    }

    manifest = _make_manifest("/read_from_file", request_body, result)

    assert manifest.operation_type == "sequence_import"
    assert manifest.operation_label == "Import sequence file (genbank)"
    assert manifest.endpoint == "/read_from_file"
    assert manifest.output_refs == ["2"]
    assert manifest.status == "succeeded"


def test_sequence_import_manual_sequence() -> None:
    """Manual sequence entry produces typed manifest."""
    request_body = {
        "sequence": {
            "id": 0,
            "type": "ManuallyTypedSequence",
            "sequence": "ATCGATCG",
            "circular": False,
        },
        "source": {"id": 0, "type": "ManuallyTypedSource"},
    }
    result = {
        "sources": [{"id": 1, "type": "ManuallyTypedSource"}],
        "sequences": [
            {
                "id": 2,
                "type": "TextFileSequence",
                "length": 8,
                "circular": False,
            }
        ],
    }

    manifest = _make_manifest("/manually_typed", request_body, result)

    assert manifest.operation_type == "sequence_import"
    assert manifest.operation_label == "Manual sequence entry"
    assert manifest.endpoint == "/manually_typed"
    assert manifest.output_refs == ["2"]


# --- Restriction digest ------------------------------------------------------


def test_restriction_digest_produces_typed_manifest() -> None:
    """Restriction digest result produces typed manifest with enzyme info."""
    request_body = {
        "sequences": [{"id": 1, "type": "TextFileSequence"}],
        "source": {
            "id": 0,
            "type": "RestrictionSource",
            "enzymes": ["EcoRI", "HindIII"],
            "input": [{"sequence": 1}],
        },
    }
    result = {
        "sources": [
            {
                "id": 2,
                "type": "RestrictionSource",
                "enzymes": ["EcoRI", "HindIII"],
                "input": [{"sequence": 1}],
            }
        ],
        "sequences": [
            {"id": 3, "type": "TextFileSequence", "length": 200},
            {"id": 4, "type": "TextFileSequence", "length": 3000},
        ],
    }

    manifest = _make_manifest("/restriction", request_body, result)

    assert manifest.operation_type == "restriction_digest"
    assert manifest.operation_label == "Restriction digest"
    assert manifest.input_refs == ["1"]
    assert manifest.output_refs == ["3", "4"]


# --- Unknown / generic endpoint ----------------------------------------------


def test_generic_endpoint_produces_valid_manifest() -> None:
    """Unknown/generic endpoints still produce valid step manifests."""
    request_body = {"some_field": "value"}
    result = {
        "id": 42,
        "status": "ok",
        "data": {"result": "computed"},
    }

    manifest = _make_manifest("/some/unknown/endpoint", request_body, result)

    assert manifest.operation_type == "generic_opencloning_operation"
    assert manifest.operation_label == "Some — Unknown — Endpoint"
    assert manifest.endpoint == "/some/unknown/endpoint"
    assert manifest.status == "succeeded"
    assert manifest.output_refs == ["42"]
    assert manifest.input_refs == []


def test_generic_endpoint_with_output_refs_from_sequences() -> None:
    """Generic endpoint with sequences in result extracts output refs."""
    request_body = {"sequences": [{"id": 1}]}
    result = {
        "sequences": [{"id": 5, "type": "TextFileSequence"}],
    }

    manifest = _make_manifest("/custom/operation", request_body, result)

    assert manifest.operation_type == "generic_opencloning_operation"
    assert manifest.output_refs == ["5"]
    assert manifest.input_refs == ["1"]


# --- sequence_byte_policy guarantee -----------------------------------------


def test_manifest_never_contains_raw_sequence_bytes() -> None:
    """The manifest's sequence_byte_policy is always 'server_side_only'.
    No field in the manifest should contain raw sequence DNA."""
    request_body = {
        "sequences": [{"id": 1, "type": "TextFileSequence"}],
        "source": {"id": 0, "type": "PCRSource", "input": [{"sequence": 1}]},
    }
    result = {
        "sources": [{"id": 2, "type": "PCRSource", "input": [{"sequence": 1}]}],
        "sequences": [{"id": 3, "type": "TextFileSequence", "length": 500}],
    }

    manifest = _make_manifest("/pcr", request_body, result)
    d = manifest.to_dict()

    assert d["sequence_byte_policy"] == "server_side_only"

    # Check no field has raw DNA-like content.
    raw_text = json.dumps(d)
    # No long stretches of A/C/G/T (sequence bytes).
    assert "ATGCATGC" not in raw_text
    assert "atgcatgc" not in raw_text


def test_primer_design_manifest_no_raw_sequences() -> None:
    """Primer design manifest carries primer IDs, not raw primer sequences."""
    request_body = {
        "sequences": [{"id": 1, "type": "TextFileSequence"}],
        "source": {"id": 0, "type": "PrimerDesignSource"},
    }
    result = {
        "primers": [
            {"id": 1, "name": "fwd", "sequence": "ATCGATCGATCGATCG"},
            {"id": 2, "name": "rev", "sequence": "CGATCGATCGATCGAT"},
        ],
    }

    manifest = _make_manifest("/primer_design/simple_pair", request_body, result)
    d = manifest.to_dict()

    assert d["primer_refs"] == ["1", "2"]
    assert d["sequence_byte_policy"] == "server_side_only"
    # Primer sequences should NOT be in the manifest.
    raw = json.dumps(d)
    assert "ATCGATCGATCGATCG" not in raw


# --- Edge cases: missing fields ----------------------------------------------


def test_empty_result_does_not_crash() -> None:
    """An empty result dict does not crash any extractor."""
    manifest = _make_manifest("/pcr", {}, {})

    assert manifest.operation_type == "pcr"
    assert manifest.input_refs == []
    assert manifest.output_refs == []
    assert manifest.template_refs == []
    assert manifest.region_refs == []
    assert manifest.primer_refs == []
    assert manifest.warnings == []
    assert manifest.errors == []
    assert manifest.status == "succeeded"


def test_missing_fields_return_empty_lists() -> None:
    """Each extractor handles missing fields gracefully (empty lists, not crash)."""
    # Test the PCR extractor directly with minimal input.
    manifest = extract_pcr("/pcr", {"sequences": None, "pcr_templates": None}, {})
    assert manifest.input_refs == []
    assert manifest.output_refs == []
    assert manifest.warnings == []

    # Test assembly directly with minimal input.
    manifest = extract_assembly("/gibson_assembly", {}, {})
    assert manifest.input_refs == []
    assert manifest.output_refs == []

    # Test sequence_import directly with empty result.
    manifest = extract_sequence_import("/read_from_file", {}, {})
    assert manifest.input_refs == []
    assert manifest.output_refs == []


def test_extract_step_manifest_passthrough_run_id_and_step_id() -> None:
    """Ensure run_id, step_id, parent_step_ids are passed through."""
    manifest = extract_step_manifest(
        endpoint="/pcr",
        request_body={},
        result={},
        run_id="my-run",
        step_id="my-step",
        parent_step_ids=["p1", "p2"],
    )
    assert manifest.run_id == "my-run"
    assert manifest.step_id == "my-step"
    assert manifest.parent_step_ids == ["p1", "p2"]

    manifest_no_parents = extract_step_manifest(
        endpoint="/pcr",
        request_body={},
        result={},
        run_id="r",
        step_id="s",
        parent_step_ids=None,
    )
    assert manifest_no_parents.parent_step_ids == []
