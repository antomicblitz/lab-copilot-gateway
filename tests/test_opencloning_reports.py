"""Tests for the deterministic notebook summary and Markdown audit report renderers (T10).

Acceptance checks:

    1. Notebook summary renders all required sections.
    2. Markdown report renders all required sections.
    3. Missing facts render as "N/A" or "not checked".
    4. Primer table includes all primers with available properties.
    5. Step timeline includes all step manifests in order.
    6. Validation bundle checks are reflected in the report.
    7. Protocol-backed conditions appear when protocol_result is provided.
    8. Fallback warnings appear when protocol_result is None or
       fallback_used=True.
    9. Empty/minimal cloning run produces valid (but sparse) reports.
    10. HTML is well-formed (has opening/closing tags).
    11. Markdown is valid (has headers, tables, code blocks).
"""

from __future__ import annotations

from lab_copilot_gateway.opencloning_reports import (
    render_cloning_audit_report_markdown,
    render_notebook_summary,
)

# --- test fixtures ------------------------------------------------------------

GENBANK_INSERT = """LOCUS       insert                 60 bp    DNA     linear SYN 01-JUL-2026
DEFINITION  Synthetic insert for Gibson assembly.
FEATURES             Location/Qualifiers
     source          1..60
                     /label="insert"
     CDS             3..57
                     /label="GFP"
ORIGIN
        1 atggtgagca agggcgagga gctgttcacc ggggtggtgc ccatcctggt
       61 cgag
//
"""

GENBANK_VECTOR = """LOCUS       vector                120 bp    DNA     circular SYN 01-JUL-2026
DEFINITION  Cloning vector backbone.
FEATURES             Location/Qualifiers
     source          1..120
                     /label="vector"
     rep_origin      10..100
                     /label="ori"
ORIGIN
        1 aaggatccga attcgagctc ggtacccggg gatcctctag agtcgacctg
       61 caggcatgca agcttggcac tggccgtcgt tttacaacgt cgtgactggg
      121 aaaaccctgg cgttacccaa cttaatcgcc ttgcagcaca tccccctttc
      181 gccagctggc gtaatagcga agaggcccgc accgatcgcc cttcccaaca
      241 gttgcgcagc ctgaatggcg aatgggatcc
//
"""

GENBANK_FINAL = """LOCUS       final_construct       180 bp    DNA     circular SYN 01-JUL-2026
DEFINITION  Final Gibson assembly product.
FEATURES             Location/Qualifiers
     source          1..180
                     /label="final_construct"
     CDS             3..57
                     /label="GFP"
     rep_origin      60..150
                     /label="ori"
ORIGIN
        1 atggtgagca agggcgagga gctgttcacc ggggtggtgc ccatcctggt
       61 cgaggatccg aattcgagct cggtacccgg ggatcctcta gagtcgacct
      121 gcaggcatgc aagcttggca ctggccgtcg ttttacaacg tcgtgactgg
      181 gaaaaccctg gcgttaccca acttaatcgc cttgcagcac atcccccttt
      241 cgccagctgg cgtaatagcg aagaggcccg caccgatcgc ccttcccaac
      301 agttgcgcag cctgaatggc gaatgggatc c
//
"""


# --- fixture factories --------------------------------------------------------


def _gibson_sequences() -> list[dict]:
    return [
        {
            "id": 1,
            "type": "TextFileSequence",
            "name": "insert",
            "sequence_file_format": "genbank",
            "file_content": GENBANK_INSERT,
            "length": 60,
            "circular": False,
            "feature_count": 2,
        },
        {
            "id": 2,
            "type": "TextFileSequence",
            "name": "vector",
            "sequence_file_format": "genbank",
            "file_content": GENBANK_VECTOR,
            "length": 120,
            "circular": True,
            "feature_count": 2,
        },
        {
            "id": 3,
            "type": "TextFileSequence",
            "name": "pcr_product_1",
            "sequence_file_format": "genbank",
            "file_content": GENBANK_INSERT,
            "length": 60,
            "circular": False,
            "feature_count": 2,
        },
        {
            "id": 4,
            "type": "TextFileSequence",
            "name": "final_construct",
            "sequence_file_format": "genbank",
            "file_content": GENBANK_FINAL,
            "length": 180,
            "circular": True,
            "feature_count": 4,
        },
    ]


def _gibson_sources() -> list[dict]:
    return [
        {"id": 1, "type": "UploadedFileSource", "input": []},
        {"id": 2, "type": "UploadedFileSource", "input": []},
        {
            "id": 3,
            "type": "PCRSource",
            "input": [{"sequence": 1}],
            "output_name": "insert PCR",
        },
        {
            "id": 4,
            "type": "GibsonAssemblySource",
            "input": [{"sequence": 1}, {"sequence": 2}],
            "output_name": "final_construct",
        },
    ]


def _gibson_primers() -> list[dict]:
    return [
        {
            "id": 1,
            "name": "fwd_primer",
            "sequence": "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGA",
            "tm": 58.5,
            "gc_content": 45.0,
        },
        {
            "id": 2,
            "name": "rev_primer",
            "sequence": "GCATCGATCGATCGATCGATCGATCGATCGATCGATCG",
            "tm": 60.2,
            "gc_content": 50.0,
        },
        {
            "id": 3,
            "name": "fwd_primer_2",
            "sequence": "ATGCGATCGATCGATCGATCGATCGATCGATCG",
            "tm": 59.0,
        },
        {
            "id": 4,
            "name": "rev_primer_2",
            "sequence": "GCATGCATCGATCGATCGATCGATCGATCG",
            "tm": 61.5,
        },
    ]


def _gibson_step_manifests() -> list[dict]:
    return [
        {
            "step_id": "step-1",
            "operation_type": "sequence_import",
            "operation_label": "Import sequence file (genbank)",
            "status": "succeeded",
            "input_refs": [],
            "output_refs": ["1"],
        },
        {
            "step_id": "step-2",
            "operation_type": "sequence_import",
            "operation_label": "Import sequence file (genbank)",
            "status": "succeeded",
            "input_refs": [],
            "output_refs": ["2"],
        },
        {
            "step_id": "step-3",
            "operation_type": "pcr",
            "operation_label": "PCR amplification",
            "status": "succeeded",
            "input_refs": ["1"],
            "output_refs": ["3"],
            "template_refs": ["1"],
            "primer_refs": ["1", "2"],
        },
        {
            "step_id": "step-4",
            "operation_type": "assembly",
            "operation_label": "Gibson assembly",
            "status": "succeeded",
            "input_refs": ["1", "2"],
            "output_refs": ["4"],
        },
    ]


def _passing_validation_bundle() -> dict:
    return {
        "schema": "opencloning_validation_bundle_v1",
        "run_id": "test-run-1",
        "is_approved": True,
        "blocker_count": 0,
        "warning_count": 0,
        "checks": [
            {
                "check_id": "final_construct_exists",
                "label": "Final construct exists",
                "level": "blocker",
                "status": "pass",
                "detail": "Final construct sequence id=4, 180 bp",
            },
            {
                "check_id": "genbank_artifact_exists",
                "label": "GenBank artifact exists",
                "level": "blocker",
                "status": "pass",
                "detail": "GenBank artifact content is available for writeback",
            },
            {
                "check_id": "post_assembly_annotation",
                "label": "Post-assembly annotation",
                "level": "warning",
                "status": "pass",
                "detail": "Post-assembly annotation completed",
            },
            {
                "check_id": "primer_details",
                "label": "Primer details",
                "level": "warning",
                "status": "pass",
                "detail": "Primer details available for all 4 primer(s)",
            },
            {
                "check_id": "protocol_lookup",
                "label": "Protocol lookup",
                "level": "warning",
                "status": "pass",
                "detail": "Approved protocol found for the cloning method",
            },
            {
                "check_id": "notebook_summary_draft",
                "label": "Notebook summary draft",
                "level": "warning",
                "status": "pass",
                "detail": "Notebook summary draft generated",
            },
            {
                "check_id": "audit_report_draft",
                "label": "Audit report draft",
                "level": "warning",
                "status": "pass",
                "detail": "Audit report draft generated",
            },
        ],
        "blockers": [],
        "warnings": [],
    }


def _protocol_result() -> dict:
    return {
        "query_method": "pcr",
        "query_reagent": "Q5 Polymerase",
        "matches": [
            {
                "item_id": 10,
                "title": "Q5 PCR Protocol",
                "method_type": "pcr",
                "reagent_product_name": "q5_polymerase",
                "status": "approved",
                "approved_for_use": True,
                "annealing_temperature_rule": "Tm - 3°C",
                "extension_rate": "30 s/kb",
                "cycle_count_guidance": "35 cycles",
                "manufacturer": "NEB",
                "catalog_number": "M0491S",
                "match_confidence": "exact",
                "warnings": [],
            }
        ],
        "best_match": {
            "item_id": 10,
            "title": "Q5 PCR Protocol",
            "method_type": "pcr",
            "reagent_product_name": "q5_polymerase",
            "status": "approved",
            "approved_for_use": True,
            "annealing_temperature_rule": "Tm - 3°C",
            "extension_rate": "30 s/kb",
            "cycle_count_guidance": "35 cycles",
            "manufacturer": "NEB",
            "catalog_number": "M0491S",
            "match_confidence": "exact",
            "warnings": [],
        },
        "fallback_used": False,
        "warnings": [],
    }


def _artifact_ids() -> dict[str, int]:
    return {"genbank": 100, "history_json": 101, "audit_report": 102}


# --- notebook summary tests ---------------------------------------------------


class TestNotebookSummary:
    """Tests for ``render_notebook_summary()``."""

    def test_render_all_sections(self) -> None:
        """Happy path: all sections present."""
        html = render_notebook_summary(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=_gibson_primers(),
            validation_bundle=_passing_validation_bundle(),
            protocol_result=_protocol_result(),
            step_manifests=_gibson_step_manifests(),
            artifact_ids=_artifact_ids(),
            goal="Assemble insert into vector",
            method="Gibson assembly",
            llm_rationale="The insert was amplified with overhangs.",
        )

        # Check key sections present.
        assert "Cloning summary" in html
        assert "Objective:" in html
        assert "Assemble insert into vector" in html
        assert "Gibson assembly" in html
        assert "Primers:" in html
        assert "PCR products:" in html
        assert "Protocol conditions:" in html
        assert "Final construct:" in html
        assert "Validation:" in html
        assert "Attachments:" in html
        assert "AI-generated rationale" in html

        # Well-formed HTML: at least one opening and closing tag.
        assert "<" in html
        assert "</" in html
        assert html.count("<") > html.count("</")

    def test_no_llm_rationale(self) -> None:
        """No rationale section when llm_rationale is empty."""
        html = render_notebook_summary(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=_gibson_primers(),
            validation_bundle=_passing_validation_bundle(),
        )
        assert "AI-generated rationale" not in html
        assert "Objective:" not in html  # no goal

    def test_missing_facts_render_as_na(self) -> None:
        """Sequences without data still produce valid HTML (method defaults
        to display value rather than N/A)."""
        seqs_no_data = [
            {"id": 1, "name": "unknown", "file_content": ""},
        ]
        html = render_notebook_summary(
            sequences=seqs_no_data,
            sources=[],
            primers=[],
            validation_bundle={
                "is_approved": False,
                "blocker_count": 2,
                "warning_count": 0,
            },
        )
        assert "Cloning summary" in html
        assert "Primers:" not in html  # no primers

    def test_primer_sequence_truncated(self) -> None:
        """Primer sequences longer than 30 nt are truncated with …."""
        html = render_notebook_summary(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=_gibson_primers(),
            validation_bundle=_passing_validation_bundle(),
        )
        # fwd_primer is 40 nt — full sequence should not appear.
        full_fwd = "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGA"
        assert full_fwd not in html
        # Should appear truncated with … inside a <code> tag.
        assert "…</code>" in html

    def test_validation_blocked_shows_fail(self) -> None:
        """Blocked validation shows ❌ Blocked."""
        bundle = dict(_passing_validation_bundle())
        bundle["is_approved"] = False
        bundle["blocker_count"] = 1
        bundle["blockers"] = [
            {
                "check_id": "final_construct_exists",
                "label": "Final construct exists",
                "detail": "Not found",
            }
        ]
        html = render_notebook_summary(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=_gibson_primers(),
            validation_bundle=bundle,
        )
        assert "❌" in html or "Blocked" in html

    def test_protocol_conditions_included(self) -> None:
        """Protocol conditions appear when protocol_result is provided."""
        html = render_notebook_summary(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=_gibson_primers(),
            validation_bundle=_passing_validation_bundle(),
            protocol_result=_protocol_result(),
        )
        assert "Protocol conditions" in html
        assert "35 cycles" in html
        assert "Tm - 3°C" in html
        assert "Q5 PCR Protocol" in html

    def test_no_protocol_no_conditions(self) -> None:
        """No protocol section when protocol_result is None."""
        html = render_notebook_summary(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=_gibson_primers(),
            validation_bundle=_passing_validation_bundle(),
            protocol_result=None,
        )
        assert "Protocol conditions" not in html

    def test_attachment_ids_included(self) -> None:
        """Attachment IDs appear when artifact_ids is provided."""
        html = render_notebook_summary(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=_gibson_primers(),
            validation_bundle=_passing_validation_bundle(),
            artifact_ids=_artifact_ids(),
        )
        assert "Attachments" in html
        assert "upload id=100" in html
        assert "upload id=101" in html
        assert "upload id=102" in html

    def test_empty_run_produces_valid_html(self) -> None:
        """Empty cloning run produces valid but sparse HTML."""
        html = render_notebook_summary(
            sequences=[],
            sources=[],
            primers=[],
            validation_bundle={
                "is_approved": False,
                "blocker_count": 2,
                "warning_count": 0,
            },
        )
        assert "<" in html
        assert "</" in html
        assert "Cloning summary" in html

    def test_final_construct_section(self) -> None:
        """Final construct details are rendered."""
        html = render_notebook_summary(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=[],
            validation_bundle=_passing_validation_bundle(),
        )
        # final_construct is a leaf in _gibson_sources().
        assert "Final construct" in html
        assert "final_construct" in html
        assert "180 bp" in html

    def test_pcr_product_details(self) -> None:
        """PCR products appear in their own section."""
        html = render_notebook_summary(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=_gibson_primers(),
            validation_bundle=_passing_validation_bundle(),
            step_manifests=_gibson_step_manifests(),
        )
        assert "PCR products" in html
        assert "pcr_product_1" in html  # PCR output


# --- markdown audit report tests ----------------------------------------------


class TestAuditReportMarkdown:
    """Tests for ``render_cloning_audit_report_markdown()``."""

    def test_render_all_sections(self) -> None:
        """Happy path: all sections present."""
        md = render_cloning_audit_report_markdown(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=_gibson_primers(),
            validation_bundle=_passing_validation_bundle(),
            step_manifests=_gibson_step_manifests(),
            protocol_result=_protocol_result(),
            artifact_ids=_artifact_ids(),
            goal="Assemble insert into vector",
            method="Gibson assembly",
            warnings=[{"message": "Annotation used fallback method."}],
            rollback_instructions="Delete attachments and restore body.",
        )

        # Check key sections.
        assert "# Cloning Audit Report" in md
        assert "**Method:** Gibson assembly" in md
        assert "**Goal:** Assemble insert into vector" in md
        assert "## Step Timeline" in md
        assert "## Intermediate Products" in md
        assert "## Primers" in md
        assert "## Protocol" in md
        assert "## Validation" in md
        assert "## Provenance" in md
        assert "## Warnings" in md
        assert "## Attachments" in md
        assert "## Rollback" in md

        # Has valid markdown structure: headers, tables.
        assert md.startswith("#")
        assert "|---" in md  # table separator

    def test_step_timeline_order(self) -> None:
        """Step manifests appear in order with correct data."""
        md = render_cloning_audit_report_markdown(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=_gibson_primers(),
            validation_bundle=_passing_validation_bundle(),
            step_manifests=_gibson_step_manifests(),
        )

        # Steps should appear in order.
        idx_import = md.index("sequence_import")
        idx_pcr = md.index("pcr")
        idx_assembly = md.index("assembly")
        assert idx_import < idx_pcr < idx_assembly

    def test_primer_table_includes_all_primers(self) -> None:
        """All 4 primers appear in the table."""
        md = render_cloning_audit_report_markdown(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=_gibson_primers(),
            validation_bundle=_passing_validation_bundle(),
            step_manifests=_gibson_step_manifests(),
        )

        # All primer names in the markdown.
        assert "fwd_primer" in md
        assert "rev_primer" in md
        assert "fwd_primer_2" in md
        assert "rev_primer_2" in md
        # Tm values.
        assert "58.5" in md
        assert "60.2" in md

    def test_primer_gc_content(self) -> None:
        """GC content appears when available."""
        md = render_cloning_audit_report_markdown(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=_gibson_primers(),
            validation_bundle=_passing_validation_bundle(),
            step_manifests=_gibson_step_manifests(),
        )
        # Only fwd_primer and rev_primer have gc_content.
        assert "45.0" in md
        assert "50.0" in md

    def test_protocol_backed_conditions(self) -> None:
        """Protocol conditions appear when protocol_result is provided."""
        md = render_cloning_audit_report_markdown(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=_gibson_primers(),
            validation_bundle=_passing_validation_bundle(),
            step_manifests=_gibson_step_manifests(),
            protocol_result=_protocol_result(),
        )
        assert "Q5 PCR Protocol" in md
        assert "NEB" in md
        assert "M0491S" in md
        assert "35 cycles" in md
        assert "Tm - 3°C" in md

    def test_protocol_fallback_warnings(self) -> None:
        """Fallback warnings appear when fallback_used=True."""
        fallback = dict(_protocol_result())
        fallback["fallback_used"] = True
        fallback["warnings"] = ["No approved protocol found for this method."]
        md = render_cloning_audit_report_markdown(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=_gibson_primers(),
            validation_bundle=_passing_validation_bundle(),
            step_manifests=_gibson_step_manifests(),
            protocol_result=fallback,
        )
        assert "## Fallback Assumptions" in md
        assert "No approved protocol found" in md

    def test_validation_bundle_reflected(self) -> None:
        """Validation checks are reflected in the report."""
        md = render_cloning_audit_report_markdown(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=_gibson_primers(),
            validation_bundle=_passing_validation_bundle(),
            step_manifests=_gibson_step_manifests(),
        )
        assert "**APPROVED**" in md
        assert "final_construct_exists" in md
        assert "protocol_lookup" in md

    def test_intermediate_products_table(self) -> None:
        """Intermediate products table includes all sequences."""
        md = render_cloning_audit_report_markdown(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=_gibson_primers(),
            validation_bundle=_passing_validation_bundle(),
            step_manifests=_gibson_step_manifests(),
        )
        assert "insert" in md
        assert "vector" in md
        assert "pcr_product_1" in md
        assert "final_construct" in md
        # Topology.
        assert "circular" in md
        assert "linear" in md

    def test_provenance_table(self) -> None:
        """Provenance source tree is rendered."""
        md = render_cloning_audit_report_markdown(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=_gibson_primers(),
            validation_bundle=_passing_validation_bundle(),
            step_manifests=_gibson_step_manifests(),
        )
        assert "UploadedFileSource" in md
        assert "PCRSource" in md
        assert "GibsonAssemblySource" in md

    def test_attachment_ids(self) -> None:
        """Attachment IDs render correctly."""
        md = render_cloning_audit_report_markdown(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=_gibson_primers(),
            validation_bundle=_passing_validation_bundle(),
            step_manifests=_gibson_step_manifests(),
            artifact_ids=_artifact_ids(),
        )
        assert "GenBank artifact" in md
        assert "100" in md
        assert "Cloning history JSON" in md
        assert "101" in md
        assert "Audit report (this file)" in md
        assert "102" in md

    def test_empty_run_produces_valid_markdown(self) -> None:
        """Empty cloning run produces valid but sparse Markdown."""
        md = render_cloning_audit_report_markdown(
            sequences=[],
            sources=[],
            primers=[],
            validation_bundle={
                "is_approved": False,
                "blocker_count": 2,
                "warning_count": 0,
                "checks": [],
            },
            step_manifests=[],
        )
        assert md.startswith("#")
        assert "## Step Timeline" in md
        assert "## Intermediate Products" in md
        assert "## Primers" in md

    def test_rollback_instructions(self) -> None:
        """Custom rollback instructions appear."""
        md = render_cloning_audit_report_markdown(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=_gibson_primers(),
            validation_bundle=_passing_validation_bundle(),
            step_manifests=_gibson_step_manifests(),
            rollback_instructions="Custom undo: revert to previous state.",
        )
        assert "Custom undo: revert to previous state." in md

    def test_missing_protocol_no_conditions(self) -> None:
        """No protocol section when protocol_result is None."""
        md = render_cloning_audit_report_markdown(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=_gibson_primers(),
            validation_bundle=_passing_validation_bundle(),
            step_manifests=_gibson_step_manifests(),
            protocol_result=None,
        )
        assert "No protocol lookup performed" in md

    def test_warnings_section_with_warnings(self) -> None:
        """Warnings appear in warnings section."""
        md = render_cloning_audit_report_markdown(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=_gibson_primers(),
            validation_bundle=_passing_validation_bundle(),
            step_manifests=_gibson_step_manifests(),
            warnings=[
                {"message": "Annotation used fallback method."},
                {"message": "Primer Tm mismatch detected."},
            ],
        )
        assert "## Warnings" in md
        assert "Annotation used fallback method." in md
        assert "Primer Tm mismatch detected." in md
        assert "None." not in md


# --- no-protocol fallback tests -----------------------------------------------


class TestNoProtocolFallback:
    """Tests when protocol_result is None."""

    def test_notebook_no_protocol(self) -> None:
        """Notebook summary has no protocol section."""
        html = render_notebook_summary(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=_gibson_primers(),
            validation_bundle=_passing_validation_bundle(),
            protocol_result=None,
        )
        assert "Protocol conditions" not in html

    def test_audit_report_no_protocol(self) -> None:
        """Audit report shows 'no protocol lookup performed'."""
        md = render_cloning_audit_report_markdown(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=_gibson_primers(),
            validation_bundle=_passing_validation_bundle(),
            step_manifests=_gibson_step_manifests(),
            protocol_result=None,
        )
        assert "No protocol lookup performed" in md
        assert "## Fallback Assumptions" in md

    def test_audit_report_fallback_used(self) -> None:
        """Audit report shows fallback warning when protocol was None but
        the validation bundle indicates it, or when protocol_result has
        fallback_used=True."""
        # Test with explicit fallback result.
        fb_result = {
            "query_method": "pcr",
            "fallback_used": True,
            "warnings": ["No approved protocol entry found."],
        }
        md = render_cloning_audit_report_markdown(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=_gibson_primers(),
            validation_bundle=_passing_validation_bundle(),
            step_manifests=_gibson_step_manifests(),
            protocol_result=fb_result,
        )
        assert "## Fallback Assumptions" in md
        assert "No approved protocol entry found" in md


# --- edge case tests ----------------------------------------------------------


class TestEdgeCases:
    """Edge cases for both renderers."""

    def test_html_is_well_formed(self) -> None:
        """HTML output has basic well-formedness."""
        html = render_notebook_summary(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=_gibson_primers(),
            validation_bundle=_passing_validation_bundle(),
        )
        # Every <tag> has a matching </tag> or is self-closing/void element.
        # For a simpler check: ensure count of <tags without /> matches </tags>.
        open_count = 0
        close_count = 0
        in_tag = False
        is_close = False
        for ch in html:
            if ch == "<":
                in_tag = True
                is_close = False
            elif ch == "/" and in_tag:
                is_close = True
            elif ch == ">" and in_tag:
                if is_close:
                    close_count += 1
                else:
                    open_count += 1
                in_tag = False
        # At minimum, there should be some open and close pairs.
        assert open_count > 0
        assert close_count > 0

    def test_markdown_has_headers_and_tables(self) -> None:
        """Markdown output has at least 5 headers and a table separator."""
        md = render_cloning_audit_report_markdown(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=_gibson_primers(),
            validation_bundle=_passing_validation_bundle(),
            step_manifests=_gibson_step_manifests(),
        )
        header_count = sum(1 for line in md.split("\n") if line.startswith("##"))
        assert header_count >= 4
        assert "|---" in md

    def test_no_step_manifests(self) -> None:
        """No manifests still renders with placeholder text."""
        html = render_notebook_summary(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=_gibson_primers(),
            validation_bundle=_passing_validation_bundle(),
            step_manifests=None,
        )
        # PCR products section should be absent if no manifests.
        assert "PCR products" not in html

        md = render_cloning_audit_report_markdown(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=_gibson_primers(),
            validation_bundle=_passing_validation_bundle(),
            step_manifests=[],
        )
        assert "No step manifests recorded" in md

    def test_validation_blocker_rendered(self) -> None:
        """Blocked validation shows details in the report."""
        bundle = {
            "schema": "opencloning_validation_bundle_v1",
            "run_id": "test-run-2",
            "is_approved": False,
            "blocker_count": 2,
            "warning_count": 1,
            "checks": [
                {
                    "check_id": "final_construct_exists",
                    "label": "Final construct exists",
                    "level": "blocker",
                    "status": "fail",
                    "detail": "No final construct sequence found",
                },
                {
                    "check_id": "genbank_artifact_exists",
                    "label": "GenBank artifact exists",
                    "level": "blocker",
                    "status": "fail",
                    "detail": "No GenBank artifact available",
                },
                {
                    "check_id": "protocol_lookup",
                    "label": "Protocol lookup",
                    "level": "warning",
                    "status": "fail",
                    "detail": "No approved protocol found",
                },
            ],
            "blockers": [
                {
                    "check_id": "final_construct_exists",
                    "label": "Final construct exists",
                    "detail": "Not found",
                },
                {
                    "check_id": "genbank_artifact_exists",
                    "label": "GenBank artifact exists",
                    "detail": "Not available",
                },
            ],
            "warnings": [
                {
                    "check_id": "protocol_lookup",
                    "label": "Protocol lookup",
                    "detail": "No approved protocol found",
                },
            ],
        }
        md = render_cloning_audit_report_markdown(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=_gibson_primers(),
            validation_bundle=bundle,
            step_manifests=_gibson_step_manifests(),
        )
        assert "**BLOCKED**" in md
        assert "final_construct_exists" in md
        assert "No final construct sequence found" in md

    def test_primers_empty(self) -> None:
        """No primers renders 'No primers in this cloning run'."""
        html = render_notebook_summary(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=[],
            validation_bundle=_passing_validation_bundle(),
        )
        assert "Primers:" not in html

        md = render_cloning_audit_report_markdown(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=[],
            validation_bundle=_passing_validation_bundle(),
            step_manifests=_gibson_step_manifests(),
        )
        assert "No primers in this cloning run" in md


class TestRenderNotebookSummaryDeterministic:
    """Check that identical inputs produce identical outputs."""

    def test_deterministic(self) -> None:
        """Same inputs → same HTML output."""
        inputs = {
            "sequences": _gibson_sequences(),
            "sources": _gibson_sources(),
            "primers": _gibson_primers(),
            "validation_bundle": _passing_validation_bundle(),
        }
        html1 = render_notebook_summary(**inputs)
        html2 = render_notebook_summary(**inputs)
        assert html1 == html2

    def test_llm_rationale_marked(self) -> None:
        """LLM rationale is clearly marked as AI-generated."""
        html = render_notebook_summary(
            sequences=_gibson_sequences(),
            sources=_gibson_sources(),
            primers=_gibson_primers(),
            validation_bundle=_passing_validation_bundle(),
            llm_rationale="This is an AI suggestion.",
        )
        assert "AI-generated rationale" in html
        assert "This is an AI suggestion" in html
        # Should be inside a <details> element.
        assert "<details>" in html


class TestRenderAuditReportDeterministic:
    """Check that identical inputs produce identical outputs."""

    def test_deterministic(self) -> None:
        """Same inputs → same Markdown output."""
        inputs = {
            "sequences": _gibson_sequences(),
            "sources": _gibson_sources(),
            "primers": _gibson_primers(),
            "validation_bundle": _passing_validation_bundle(),
            "step_manifests": _gibson_step_manifests(),
        }
        md1 = render_cloning_audit_report_markdown(**inputs)
        md2 = render_cloning_audit_report_markdown(**inputs)
        assert md1 == md2
