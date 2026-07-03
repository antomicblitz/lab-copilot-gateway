"""Post-assembly GenBank feature annotation recovery (C53).

pydna's Gibson assembly and PCR functions do not transfer CDS, promoter,
terminator, or RBS features from insert templates to the assembled product.
The resulting GenBank file has the correct sequence but is missing annotations
for inserted parts.

This module recovers missing annotations by searching for each template
feature's nucleotide sequence in the final assembled product.  If the exact
subsequence is found, the feature is re-annotated at the match position.

Approach (sequence-based, not coordinate-based):

    1.  Extract the nucleotide sequence from the final product's ORIGIN.
    2.  Parse existing features to avoid duplicates.
    3.  For each template in the strategy store, extract its features and
        their nucleotide subsequences.
    4.  Search for each subsequence in the final product (forward and
        reverse complement).  If found and not a duplicate, add the
        annotation.

This is robust because a CDS's nucleotide sequence is unique enough to match
exactly.  Parts that were removed (e.g., mScarlet replaced by EGFP) will not
match and will not be erroneously annotated.
"""

from __future__ import annotations

import re
from typing import Any, Sequence


# Feature types worth recovering from templates.  Excludes misc_feature,
# source, primer_bind, etc. which are either ubiquitous or not useful
# for downstream cloning operations.
_FEATURE_TYPES_OF_INTEREST: frozenset[str] = frozenset(
    {"cds", "promoter", "terminator", "rbs", "gene", "exon", "rrna", "trna"}
)

# Minimum feature length (bp) to search for.  Shorter sequences produce
# too many false-positive matches.
_MIN_FEATURE_LEN = 15

# Maximum number of features to add (safety valve against pathological
# templates with hundreds of annotations).
_MAX_NEW_FEATURES = 50

_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def rewrite_genbank_features(
    final_genbank: str,
    template_sequences: Sequence[dict[str, Any]],
) -> str:
    """Add missing feature annotations from templates to the final product.

    Returns the (possibly modified) GenBank string.  If parsing fails or no
    features need to be added, returns the original string unchanged.

    ``template_sequences`` is the list of sequence dicts from the adapter's
    strategy store — each has a ``file_content`` key with the GenBank text.
    """
    final_seq = _extract_origin_sequence(final_genbank)
    if not final_seq:
        return final_genbank

    existing = _parse_features(final_genbank)
    existing_labels = {
        (f["type"], f["start"], f["end"]) for f in existing
    }

    new_features: list[dict[str, Any]] = []
    seen_seqs: set[str] = set()

    for template in template_sequences:
        if not isinstance(template, dict):
            continue
        fc = template.get("file_content")
        if not isinstance(fc, str) or fc == final_genbank:
            continue

        tmpl_seq = _extract_origin_sequence(fc)
        if not tmpl_seq:
            continue

        tmpl_features = _parse_features(fc)

        for feat in tmpl_features:
            if len(new_features) >= _MAX_NEW_FEATURES:
                break
            if feat["type"].lower() not in _FEATURE_TYPES_OF_INTEREST:
                continue

            feat_seq = _extract_feature_sequence(feat, tmpl_seq)
            if not feat_seq or len(feat_seq) < _MIN_FEATURE_LEN:
                continue

            # Skip if we've already matched this exact sequence.
            if feat_seq in seen_seqs:
                continue

            match = _find_in_product(feat_seq, final_seq)
            if match is None:
                continue

            start, end, strand = match

            # Skip duplicates (same type at same position).
            key = (feat["type"], start, end)
            if key in existing_labels:
                continue
            existing_labels.add(key)

            seen_seqs.add(feat_seq)
            new_features.append(
                {
                    "type": feat["type"],
                    "start": start,
                    "end": end,
                    "strand": strand,
                    "qualifiers": feat["qualifiers"],
                }
            )

    if not new_features:
        return final_genbank

    return _insert_features(final_genbank, new_features)


# ---------------------------------------------------------------------------
# GenBank parsing helpers (string-based, consistent with opencloning_artifacts)
# ---------------------------------------------------------------------------


def _extract_origin_sequence(genbank: str) -> str:
    """Extract the nucleotide sequence from the ORIGIN section."""
    origin_match = re.search(
        r"^ORIGIN\s*\n(.*?)(?:^//|\Z)",
        genbank,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not origin_match:
        return ""
    return re.sub(r"[^A-Za-z]", "", origin_match.group(1)).upper()


def _parse_features(genbank: str) -> list[dict[str, Any]]:
    """Parse FEATURES section into structured dicts.

    Each dict has: type, start, end, strand, location_str, qualifiers.
    ``start`` and ``end`` are 1-based, inclusive.  ``strand`` is 1 or -1.
    """
    features: list[dict[str, Any]] = []
    in_features = False
    current: dict[str, Any] | None = None

    for line in genbank.splitlines():
        if line.startswith("FEATURES"):
            in_features = True
            continue
        if in_features and (line.startswith("ORIGIN") or line.startswith("//")):
            break
        if not in_features:
            continue

        feat_match = re.match(r"^\s{5}(\S+)\s+(.+)$", line)
        if feat_match:
            if current:
                features.append(current)
            ftype = feat_match.group(1)
            location_str = feat_match.group(2).strip()
            start, end, strand = _parse_location(location_str)
            current = {
                "type": ftype,
                "start": start,
                "end": end,
                "strand": strand,
                "location_str": location_str,
                "qualifiers": {},
            }
        elif current and re.match(r"^\s+/", line):
            qual_match = re.match(r"^\s+/(\w+)=(.*)$", line)
            if qual_match:
                key = qual_match.group(1)
                val = qual_match.group(2).strip().strip('"')
                current["qualifiers"][key] = val

    if current:
        features.append(current)
    return features


def _parse_location(location_str: str) -> tuple[int, int, int]:
    """Parse a GenBank location string into (start, end, strand).

    Handles:
      ``8978..9673``           → (8978, 9673, 1)
      ``complement(100..500)`` → (100, 500, -1)
      ``join(1..3,4..6,7..720)`` → (1, 720, 1)
      ``join(9000..9752,1..100)`` → (9000, 100, 1)  [spanning origin]
    """
    strand = 1
    loc = location_str
    if loc.startswith("complement("):
        strand = -1
        loc = loc[len("complement(") :].rstrip(")")

    # Extract all position numbers.
    numbers = [int(n) for n in re.findall(r"\d+", loc)]
    if not numbers:
        return (0, 0, 1)

    if "join" in loc:
        start = min(numbers)
        end = max(numbers)
    else:
        start = numbers[0]
        end = numbers[1] if len(numbers) > 1 else numbers[0]
    return (start, end, strand)


def _extract_feature_sequence(
    feat: dict[str, Any], template_seq: str
) -> str:
    """Extract the nucleotide subsequence for a feature from a template.

    Handles join() locations with multiple parts and complement().  For
    circular plasmids, handles features spanning the origin.
    """
    loc = feat.get("location_str", "")
    if not loc:
        return ""

    # Determine complement.
    is_complement = loc.startswith("complement(")
    inner = loc[len("complement(") :] if is_complement else loc
    inner = inner.rstrip(")")

    # Parse all location parts.
    parts: list[tuple[int, int]] = []
    if inner.startswith("join("):
        inner = inner[len("join(") :].rstrip(")")
        for part_str in inner.split(","):
            part_str = part_str.strip()
            nums = [int(n) for n in re.findall(r"\d+", part_str)]
            if len(nums) >= 2:
                parts.append((nums[0], nums[1]))
            elif len(nums) == 1:
                parts.append((nums[0], nums[0]))
    else:
        nums = [int(n) for n in re.findall(r"\d+", inner)]
        if len(nums) >= 2:
            parts.append((nums[0], nums[1]))
        elif len(nums) == 1:
            parts.append((nums[0], nums[0]))

    if not parts:
        return ""

    seq_len = len(template_seq)

    # Extract and concatenate each part.
    chunks: list[str] = []
    for start, end in parts:
        if start <= end:
            # Normal: within the same strand.
            if end > seq_len:
                return ""  # malformed
            chunks.append(template_seq[start - 1 : end])
        else:
            # Spans the origin (circular): start > end.
            # E.g., join(9700..9752,1..100) with parts (9700,9752) and (1,100).
            # The wrapping part has start > end only when the origin-spanning
            # segment is encoded as a single part like 9700..100.  In practice
            # join() with two separate parts handles this (see above).
            # Handle the edge case anyway.
            if start > seq_len:
                return ""
            chunks.append(template_seq[start - 1 : seq_len])
            chunks.append(template_seq[0 : end])

    result = "".join(chunks)

    if is_complement:
        result = _reverse_complement(result)

    return result


def _reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def _find_in_product(
    feat_seq: str, product_seq: str
) -> tuple[int, int, int] | None:
    """Search for feat_seq in product_seq (forward and reverse complement).

    Returns (start_1based, end_inclusive, strand) or None.
    Also searches across the origin for circular plasmids by appending
    the first ``len(feat_seq)`` bases of the product to the end.
    """
    if not feat_seq or not product_seq:
        return None

    # Forward strand search.
    pos = product_seq.find(feat_seq)
    if pos != -1:
        return (pos + 1, pos + len(feat_seq), 1)

    # Reverse complement search.
    rc = _reverse_complement(feat_seq)
    pos = product_seq.find(rc)
    if pos != -1:
        return (pos + 1, pos + len(feat_seq), -1)

    # Circular search: wrap around the origin.
    if len(feat_seq) < len(product_seq):
        extended = product_seq + product_seq[: len(feat_seq) - 1]
        pos = extended.find(feat_seq)
        if pos != -1:
            end = pos + len(feat_seq)
            if end <= len(product_seq):
                return (pos + 1, end, 1)
            # Spans origin — represent as join.
            # For simplicity, return the wrapped coordinates.
            return (pos + 1, end - len(product_seq), 1)

        pos = extended.find(rc)
        if pos != -1:
            end = pos + len(feat_seq)
            if end <= len(product_seq):
                return (pos + 1, end, -1)
            return (pos + 1, end - len(product_seq), -1)

    return None


# ---------------------------------------------------------------------------
# GenBank writing helpers
# ---------------------------------------------------------------------------


def _format_feature(feature: dict[str, Any]) -> str:
    """Format a feature dict as GenBank feature lines."""
    ftype = feature["type"]
    start = feature["start"]
    end = feature["end"]
    strand = feature["strand"]

    if strand == -1:
        location = f"complement({start}..{end})"
    else:
        location = f"{start}..{end}"

    lines: list[str] = [f"     {ftype:<16s}{location}"]

    qualifiers = feature.get("qualifiers") or {}
    # Prioritize label, then gene, then product for the /label qualifier.
    label = qualifiers.get("label") or qualifiers.get("gene") or qualifiers.get("product")
    if label:
        lines.append(f'                     /label="{label}"')
    # Preserve other useful qualifiers.
    for key in ("product", "note", "codon_start", "translation"):
        val = qualifiers.get(key)
        if val and key not in ("label", "gene"):
            # Handle long values (wrap at 58 chars, GenBank convention).
            val_str = f'/{key}="{val}"'
            if len(val_str) <= 68:
                lines.append(f"                     {val_str}")
            else:
                # Wrap long qualifier values.
                prefix = f'/{key}="'
                suffix = '"'
                inner = val
                first_line_len = 68 - len(prefix)
                lines.append(f"                     {prefix}{inner[:first_line_len]}")
                inner = inner[first_line_len:]
                while inner:
                    chunk = inner[:58]
                    inner = inner[58:]
                    if not inner:
                        lines.append(f"                     {chunk}{suffix}")
                    else:
                        lines.append(f"                     {chunk}")
    return "\n".join(lines)


def _insert_features(genbank: str, new_features: list[dict[str, Any]]) -> str:
    """Insert new features into the FEATURES section of a GenBank string.

    Features are inserted before the ORIGIN line, after existing features.
    """
    # Find the ORIGIN line (features go before it).
    origin_match = re.search(r"^ORIGIN\s", genbank, flags=re.MULTILINE)
    if not origin_match:
        return genbank

    insert_pos = origin_match.start()

    # Build the feature block text.
    feature_lines = []
    for feat in sorted(new_features, key=lambda f: f["start"]):
        feature_lines.append(_format_feature(feat))

    feature_text = "\n".join(feature_lines) + "\n"

    return genbank[:insert_pos] + feature_text + genbank[insert_pos:]
