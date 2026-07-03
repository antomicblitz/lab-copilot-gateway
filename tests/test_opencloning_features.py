"""Tests for post-assembly GenBank feature annotation recovery (C53).

Validates that ``rewrite_genbank_features`` correctly:
  * recovers CDS annotations from insert templates
  * skips features already present in the final product
  * skips features whose sequence is absent (e.g., removed parts)
  * handles complement (reverse strand) features
  * is a no-op when no templates or no matches exist
"""

from __future__ import annotations

from lab_copilot_gateway.opencloning_features import rewrite_genbank_features


def _genbank(name: str, length: int, topology: str, features: str, origin_seq: str) -> str:
    """Build a minimal GenBank string from components."""
    return (
        f"LOCUS       {name:<24s} {length} bp    DNA     {topology:<8s} UNK\n"
        f"FEATURES             Location/Qualifiers\n"
        f"{features}"
        f"ORIGIN\n"
        f"        1 {origin_seq}\n"
        f"//\n"
    )


def _wrap_origin(seq: str, width: int = 60) -> str:
    """Format a sequence string into GenBank ORIGIN lines (no numbering)."""
    return " ".join(seq[i:i+width] for i in range(0, len(seq), width))


# --- Test sequences ---------------------------------------------------------

# 40 bp CDS sequence for "EGFP"
_EGFP_CDS = "GTGACCATGGAGAGCAACACGGATCCTAGAGATCTAGAATGCTAGCTAGCTGATCGATCG"

# 30 bp "mScarlet" CDS — NOT present in the final product
_MSCARLET_CDS = "TTTTTTTTTTTTTTTTTTTTTTTTTTTTTT"

# Backbone flanking sequences (promoter + vector)
_BACKBONE_5 = "ATGCATGCATGCATGCATGCATGC"  # 24 bp
_BACKBONE_3 = "ATCGATCGATCGATCG"          # 16 bp

_FINAL_SEQ = _BACKBONE_5 + _EGFP_CDS + _BACKBONE_3
_FINAL_LEN = len(_FINAL_SEQ)


def _final_no_cds() -> str:
    features = (
        '     promoter        1..24\n'
        '                     /label="test promoter"\n'
    )
    return _genbank("name", _FINAL_LEN, "circular", features, _wrap_origin(_FINAL_SEQ))


def _egfp_template() -> str:
    features = (
        '     CDS             join(1..3,4..6,7..40)\n'
        '                     /label="EGFP"\n'
        '                     /product="enhanced GFP"\n'
    )
    return _genbank("EGFP", len(_EGFP_CDS), "linear", features,
                    _wrap_origin(_EGFP_CDS))


def _mscarlet_template() -> str:
    features = (
        '     CDS             1..30\n'
        '                     /label="mScarlet"\n'
    )
    return _genbank("mScarlet", len(_MSCARLET_CDS), "linear", features,
                    _wrap_origin(_MSCARLET_CDS))


def _make_template(file_content: str) -> dict:
    return {"id": 1, "type": "TextFileSequence", "file_content": file_content}


# --- Tests ------------------------------------------------------------------


def test_recovers_cds_from_insert_template() -> None:
    """EGFP CDS annotation should be added when its sequence is found."""
    result = rewrite_genbank_features(_final_no_cds(), [_make_template(_egfp_template())])

    assert "CDS" in result
    assert '/label="EGFP"' in result
    assert '/product="enhanced GFP"' in result
    # The promoter should still be there.
    assert '/label="test promoter"' in result
    # CDS should be at the right position (25..64, 1-based).
    # Backbone is 24 bp, then EGFP CDS is 40 bp → 25..64.
    assert "25..64" in result


def test_skips_features_not_in_product() -> None:
    """mScarlet CDS (not in final product) should NOT be annotated."""
    result = rewrite_genbank_features(_final_no_cds(), [_make_template(_mscarlet_template())])

    assert "mScarlet" not in result
    assert '/label="test promoter"' in result


def test_skips_already_present_features() -> None:
    """Features already in the final product should not be duplicated."""
    features = (
        '     promoter        1..24\n'
        '                     /label="test promoter"\n'
        '     CDS             25..64\n'
        '                     /label="EGFP"\n'
    )
    final_with_cds = _genbank("name", _FINAL_LEN, "circular", features,
                              _wrap_origin(_FINAL_SEQ))
    result = rewrite_genbank_features(final_with_cds, [_make_template(_egfp_template())])

    assert result.count("CDS") == 1
    assert result.count('/label="EGFP"') == 1


def test_no_templates_noop() -> None:
    """No templates → no changes."""
    original = _final_no_cds()
    result = rewrite_genbank_features(original, [])
    assert result == original


def test_empty_origin_noop() -> None:
    """Malformed GenBank without ORIGIN → no changes."""
    malformed = "LOCUS test\nFEATURES\n//\n"
    result = rewrite_genbank_features(malformed, [_make_template(_egfp_template())])
    assert result == malformed


def test_complement_feature() -> None:
    """A reverse-strand CDS should be found via reverse complement search."""
    from lab_copilot_gateway.opencloning_features import _reverse_complement

    cds_seq = "ATGCATGCATGCATGCATGCATGCATGC"  # 28 bp
    rc_cds = _reverse_complement(cds_seq)

    template = _genbank("insert", len(cds_seq), "linear",
                        '     CDS             1..28\n'
                        '                     /label="rc_test"\n',
                        _wrap_origin(cds_seq))

    product_seq = "AAAAAAAAAA" + rc_cds + "TTTTTTTTTT"
    final = _genbank("product", len(product_seq), "circular", "",
                     _wrap_origin(product_seq))

    result = rewrite_genbank_features(final, [_make_template(template)])
    assert "CDS" in result
    assert '/label="rc_test"' in result
    assert "complement(" in result


def test_multiple_templates() -> None:
    """Multiple templates: only matching ones get annotated."""
    result = rewrite_genbank_features(
        _final_no_cds(),
        [_make_template(_egfp_template()), _make_template(_mscarlet_template())],
    )
    assert '/label="EGFP"' in result
    assert "mScarlet" not in result


def test_short_features_skipped() -> None:
    """Features shorter than _MIN_FEATURE_LEN should be skipped."""
    short_seq = "ATGCA"
    template = _genbank("short", len(short_seq), "linear",
                        '     CDS             1..5\n'
                        '                     /label="tiny"\n',
                        _wrap_origin(short_seq))
    # Final product contains the short sequence but feature is too short.
    final_seq = "XXXXX" + short_seq + "XXXXX"
    final = _genbank("name", len(final_seq), "circular", "",
                     _wrap_origin(final_seq))
    result = rewrite_genbank_features(final, [_make_template(template)])
    assert "tiny" not in result


def test_circular_wrap_around() -> None:
    """A feature spanning the origin of a circular plasmid should match."""
    cds_seq = "GGGGCCCCGGGGCCCCGGGGCCCCGGGGCCCC"  # 32 bp
    # Place the CDS split across origin: last 16 bp at end, first 16 at start.
    product_seq = cds_seq[16:] + "AATTCCGG" * 10 + cds_seq[:16]
    template = _genbank("insert", len(cds_seq), "linear",
                        '     CDS             1..32\n'
                        '                     /label="origin_span"\n',
                        _wrap_origin(cds_seq))
    final = _genbank("product", len(product_seq), "circular", "",
                     _wrap_origin(product_seq))
    result = rewrite_genbank_features(final, [_make_template(template)])
    assert "CDS" in result
    assert '/label="origin_span"' in result
