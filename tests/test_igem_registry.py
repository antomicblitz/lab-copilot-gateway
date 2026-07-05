"""Tests for the iGEM Registry API client (igem_registry.py).

Tests the GenBank conversion logic without hitting the live API.
The _to_genbank() function is tested directly with known inputs.
The fetch_igem_part_as_genbank() function is tested with mocked HTTP.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lab_copilot_gateway.igem_registry import (
    _format_location,
    _format_sequence,
    _slug_from_name,
    _to_genbank,
    fetch_igem_part_as_genbank,
)


# --- _slug_from_name ------------------------------------------------------


class TestSlugFromName:
    def test_simple_part_name(self) -> None:
        assert _slug_from_name("BBa_J23105") == "bba-j23105"

    def test_part_with_multiple_underscores(self) -> None:
        assert _slug_from_name("BBa_K4736002") == "bba-k4736002"

    def test_already_lowercase(self) -> None:
        assert _slug_from_name("bba-j23105") == "bba-j23105"

    def test_empty_string(self) -> None:
        assert _slug_from_name("") == ""


# --- _format_location -----------------------------------------------------


class TestFormatLocation:
    def test_forward_strand(self) -> None:
        assert _format_location(0, 35, "forward") == "1..35"

    def test_forward_strand_nonzero_start(self) -> None:
        assert _format_location(2, 676, "forward") == "3..676"

    def test_reverse_strand(self) -> None:
        assert _format_location(0, 35, "reverse") == "complement(1..35)"

    def test_reverse_strand_nonzero_start(self) -> None:
        assert _format_location(2, 676, "reverse") == "complement(3..676)"


# --- _format_sequence -----------------------------------------------------


class TestFormatSequence:
    def test_short_sequence(self) -> None:
        result = _format_sequence("TTTACGGCTAGCTCAGTCCTAGGTACTATGCTAGC")
        assert "TTTACGGCTA GCTCAGTCCT AGGTACTATG CTAGC" in result
        assert result.startswith("        1 ")

    def test_long_sequence_wraps_at_60(self) -> None:
        seq = "A" * 120
        result = _format_sequence(seq)
        lines = result.strip().split("\n")
        assert len(lines) == 2
        assert "1" in lines[0]
        assert "61" in lines[1]

    def test_empty_sequence(self) -> None:
        result = _format_sequence("")
        assert result == ""


# --- _to_genbank ----------------------------------------------------------


class TestToGenbank:
    def test_basic_part_no_annotations(self) -> None:
        gb = _to_genbank(
            "BBa_J23105",
            "TTTACGGCTAGCTCAGTCCTAGGTACTATGCTAGC",
            [],
        )
        assert "LOCUS       BBa_J23105" in gb
        assert "35 bp" in gb
        assert "TTTACGGCTA GCTCAGTCCT AGGTACTATG CTAGC" in gb
        assert gb.endswith("//\n")

    def test_part_with_cds_annotation(self) -> None:
        annotations = [
            {
                "label": "mrfp1",
                "role": {"accession": "SO:0000316", "label": "CDS"},
                "strand": "forward",
                "locations": [{"start": 2, "end": 676}],
            }
        ]
        gb = _to_genbank(
            "BBa_E1010",
            "ATGGCTTCCT" * 70 + "ATG",
            annotations,
        )
        assert "CDS" in gb
        assert "3..676" in gb
        assert '/label="mrfp1"' in gb

    def test_part_with_reverse_strand_annotation(self) -> None:
        annotations = [
            {
                "label": "gfp",
                "role": {"accession": "SO:0000316", "label": "CDS"},
                "strand": "reverse",
                "locations": [{"start": 10, "end": 720}],
            }
        ]
        gb = _to_genbank(
            "BBa_test",
            "A" * 730,
            annotations,
        )
        assert "complement(11..720)" in gb

    def test_part_with_promoter_annotation(self) -> None:
        annotations = [
            {
                "label": "J23105",
                "role": {"accession": "SO:0000167", "label": "promoter"},
                "strand": "forward",
                "locations": [{"start": 0, "end": 35}],
            }
        ]
        gb = _to_genbank(
            "BBa_J23105",
            "TTTACGGCTAGCTCAGTCCTAGGTACTATGCTAGC",
            annotations,
        )
        assert "promoter" in gb
        assert "1..35" in gb
        assert '/label="J23105"' in gb

    def test_annotation_with_no_locations_skipped(self) -> None:
        annotations = [
            {
                "label": "empty",
                "role": {"accession": "SO:0000316", "label": "CDS"},
                "strand": "forward",
                "locations": [],
            }
        ]
        gb = _to_genbank("test", "ATGC", annotations)
        assert "CDS" not in gb

    def test_annotation_with_zero_length_skipped(self) -> None:
        annotations = [
            {
                "label": "empty",
                "role": {"accession": "SO:0000316", "label": "CDS"},
                "strand": "forward",
                "locations": [{"start": 5, "end": 5}],
            }
        ]
        gb = _to_genbank("test", "ATGCATGC", annotations)
        assert "CDS" not in gb

    def test_unknown_so_accession_falls_back_to_misc_feature(self) -> None:
        annotations = [
            {
                "label": "weird",
                "role": {"accession": "SO:9999999", "label": "unknown"},
                "strand": "forward",
                "locations": [{"start": 0, "end": 10}],
            }
        ]
        gb = _to_genbank("test", "ATGCATGCAT", annotations)
        assert "misc_feature" in gb

    def test_label_with_quotes_escaped(self) -> None:
        annotations = [
            {
                "label": 'gene "X"',
                "role": {"accession": "SO:0000316", "label": "CDS"},
                "strand": "forward",
                "locations": [{"start": 0, "end": 10}],
            }
        ]
        gb = _to_genbank("test", "ATGCATGCAT", annotations)
        assert "/label=\"gene 'X'\"" in gb

    def test_long_locus_name_truncated(self) -> None:
        long_name = "BBa_VERYLONGPARTNAME123"
        gb = _to_genbank(long_name, "ATGC", [])
        # LOCUS name should be truncated to 16 chars
        assert "BBa_VERYLONGPART" in gb
        assert long_name not in gb.split("\n")[0]


# --- fetch_igem_part_as_genbank (mocked HTTP) -----------------------------


class TestFetchIgemPartAsGenbank:
    """Test the full fetch flow with mocked HTTP responses."""

    @patch("lab_copilot_gateway.igem_registry.requests.get")
    def test_fetch_simple_part(self, mock_get: MagicMock) -> None:
        """Fetch a part with no annotations."""
        # Mock the part lookup response.
        part_response = MagicMock()
        part_response.status_code = 200
        part_response.json.return_value = {
            "uuid": "df048256-91c5-4a48-920c-b3a7d571b4af",
            "name": "BBa_J23105",
            "title": "constitutive promoter family member",
            "description": "Later",
            "sequence": "TTTACGGCTAGCTCAGTCCTAGGTACTATGCTAGC",
        }

        # Mock the annotations response.
        ann_response = MagicMock()
        ann_response.status_code = 200
        ann_response.json.return_value = {"data": []}

        mock_get.side_effect = [part_response, ann_response]

        gb = fetch_igem_part_as_genbank("BBa_J23105")
        assert "LOCUS       BBa_J23105" in gb
        assert "TTTACGGCTA GCTCAGTCCT AGGTACTATG CTAGC" in gb
        assert gb.endswith("//\n")

    @patch("lab_copilot_gateway.igem_registry.requests.get")
    def test_fetch_part_with_annotations(self, mock_get: MagicMock) -> None:
        """Fetch a part with CDS annotations."""
        part_response = MagicMock()
        part_response.status_code = 200
        part_response.json.return_value = {
            "uuid": "test-uuid",
            "name": "BBa_E1010",
            "title": "mRFP1",
            "description": "Red fluorescent protein",
            "sequence": "ATGGCTTCCT" * 70 + "ATG",
        }

        ann_response = MagicMock()
        ann_response.status_code = 200
        ann_response.json.return_value = {
            "data": [
                {
                    "uuid": "ann-uuid",
                    "label": "mrfp1",
                    "role": {
                        "uuid": "role-uuid",
                        "accession": "SO:0000316",
                        "label": "CDS",
                    },
                    "locations": [{"start": 2, "end": 676}],
                    "strand": "forward",
                }
            ]
        }

        mock_get.side_effect = [part_response, ann_response]

        gb = fetch_igem_part_as_genbank("BBa_E1010")
        assert "CDS" in gb
        assert "3..676" in gb
        assert '/label="mrfp1"' in gb

    @patch("lab_copilot_gateway.igem_registry.requests.get")
    def test_part_not_found_raises_value_error(self, mock_get: MagicMock) -> None:
        """404 on part lookup raises ValueError."""
        part_response = MagicMock()
        part_response.status_code = 404

        mock_get.return_value = part_response

        with pytest.raises(ValueError, match="not found"):
            fetch_igem_part_as_genbank("BBa_NONEXISTENT")

    @patch("lab_copilot_gateway.igem_registry.requests.get")
    def test_part_with_empty_sequence_raises_value_error(
        self, mock_get: MagicMock
    ) -> None:
        """Part with no sequence data raises ValueError."""
        part_response = MagicMock()
        part_response.status_code = 200
        part_response.json.return_value = {
            "uuid": "test-uuid",
            "name": "BBa_EMPTY",
            "sequence": "",
        }

        mock_get.return_value = part_response

        with pytest.raises(ValueError, match="no sequence data"):
            fetch_igem_part_as_genbank("BBa_EMPTY")

    @patch("lab_copilot_gateway.igem_registry.requests.get")
    def test_annotation_fetch_failure_ignored(self, mock_get: MagicMock) -> None:
        """If annotation fetch fails, the part is still returned without features."""
        import requests as _requests

        part_response = MagicMock()
        part_response.status_code = 200
        part_response.json.return_value = {
            "uuid": "test-uuid",
            "name": "BBa_J23105",
            "sequence": "TTTACGGCTAGCTCAGTCCTAGGTACTATGCTAGC",
        }

        ann_response = MagicMock()
        ann_response.status_code = 500
        ann_response.raise_for_status.side_effect = _requests.HTTPError("Server error")

        mock_get.side_effect = [part_response, ann_response]

        # Should not raise — annotations are optional.
        gb = fetch_igem_part_as_genbank("BBa_J23105")
        assert "LOCUS" in gb
        assert "TTTACGGCTA" in gb

    @patch("lab_copilot_gateway.igem_registry.requests.get")
    def test_correct_slug_used(self, mock_get: MagicMock) -> None:
        """The part name is converted to a slug for the API call."""
        part_response = MagicMock()
        part_response.status_code = 200
        part_response.json.return_value = {
            "uuid": "test-uuid",
            "name": "BBa_J23105",
            "sequence": "ATGC",
        }

        ann_response = MagicMock()
        ann_response.status_code = 200
        ann_response.json.return_value = {"data": []}

        mock_get.side_effect = [part_response, ann_response]

        fetch_igem_part_as_genbank("BBa_J23105")

        # First call should be to the slug endpoint.
        first_call = mock_get.call_args_list[0]
        assert "bba-j23105" in first_call.kwargs.get("url", "") or "bba-j23105" in str(
            first_call
        )
