"""iGEM Registry API client for fetching biological parts.

Queries the official iGEM Registry REST API (api.registry.igem.org, 2025 Beta)
to retrieve part sequences and annotations, then converts them to GenBank
format for import into OpenCloning via ``parse_sequence_file``.

This replaces the broken OpenCloning ``/repository_id/igem`` endpoint, which
requires an internal ``sequence_file_url`` field that the LLM cannot discover.

The raw-HTTP pattern matches the lab convention (see AGENTS.md → "eLabFTW API
— HTTP client patterns"): ``requests.Session`` with a ``User-Agent`` header,
no authentication needed for public parts.
"""

from __future__ import annotations

from typing import Any

import requests

# Base URL for the iGEM Registry REST API (2025 Beta).
IGEM_REGISTRY_BASE = "https://api.registry.igem.org/v1"

# Request timeout (seconds). The registry is usually fast.
IGEM_REGISTRY_TIMEOUT = 15.0

# SO ontology accession → GenBank feature type mapping.
# The iGEM API returns role objects with SO accessions (e.g. "SO:0000316").
# GenBank uses simpler type names. This mapping covers the common cases.
_SO_TO_GENBANK: dict[str, str] = {
    "SO:0000316": "CDS",            # CDS
    "SO:0000167": "promoter",       # promoter
    "SO:0000139": "ribosome_entry_site",  # RBS
    "SO:0000141": "terminator",     # terminator
    "SO:0000296": "origin_of_replication",  # ori
    "SO:0000286": "primer_bind",    # primer binding site
    "SO:0000410": "stem_loop",      # stem-loop
    "SO:0000001": "misc_feature",   # misc
}

# Fallback mapping by label keyword (when SO accession is missing/unknown).
_LABEL_KEYWORD_TO_TYPE: list[tuple[str, str]] = [
    ("cds", "CDS"),
    ("promoter", "promoter"),
    ("rbs", "ribosome_entry_site"),
    ("ribosome", "ribosome_entry_site"),
    ("terminator", "terminator"),
    ("origin", "origin_of_replication"),
    ("ori", "origin_of_replication"),
    ("primer", "primer_bind"),
    ("stem", "stem_loop"),
    ("scar", "misc_feature"),
    ("barcode", "misc_feature"),
    ("restriction", "misc_feature"),
]


def _slug_from_name(part_name: str) -> str:
    """Convert a part name like 'BBa_J23105' to a slug like 'bba-j23105'."""
    return part_name.lower().replace("_", "-")


def _fetch_part(part_name: str) -> dict[str, Any]:
    """Fetch a part from the iGEM Registry by name (via slug lookup).

    Returns the raw JSON dict from the API.
    Raises ``ValueError`` if the part is not found.
    """
    slug = _slug_from_name(part_name)
    url = f"{IGEM_REGISTRY_BASE}/parts/slugs/{slug}"
    resp = requests.get(
        url,
        headers={"User-Agent": "LabCopilot/1.0"},
        timeout=IGEM_REGISTRY_TIMEOUT,
    )
    if resp.status_code == 404:
        raise ValueError(f"iGEM part '{part_name}' not found (slug: {slug})")
    resp.raise_for_status()
    return resp.json()


def _fetch_annotations(part_uuid: str) -> list[dict[str, Any]]:
    """Fetch sequence annotations (features) for a part by UUID.

    Returns a list of annotation dicts with keys: label, type (SO accession),
    strand, start, end.
    """
    url = f"{IGEM_REGISTRY_BASE}/parts/{part_uuid}/sequence-features"
    resp = requests.get(
        url,
        headers={"User-Agent": "LabCopilot/1.0"},
        timeout=IGEM_REGISTRY_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def _annotation_to_genbank_type(ann: dict[str, Any]) -> str:
    """Map an iGEM annotation to a GenBank feature type."""
    # Try SO accession from the role object.
    role = ann.get("role", {})
    accession = role.get("accession", "")
    if accession in _SO_TO_GENBANK:
        return _SO_TO_GENBANK[accession]

    # Fallback: try the role label.
    role_label = (role.get("label", "") or "").lower()
    for keyword, gb_type in _LABEL_KEYWORD_TO_TYPE:
        if keyword in role_label:
            return gb_type

    # Fallback: try the annotation label.
    label = (ann.get("label", "") or "").lower()
    for keyword, gb_type in _LABEL_KEYWORD_TO_TYPE:
        if keyword in label:
            return gb_type

    return "misc_feature"


def _format_location(start: int, end: int, strand: str) -> str:
    """Format a GenBank location string.

    iGEM API uses 0-indexed [start, end) — GenBank uses 1-indexed [start, end].
    So we add 1 to start, keep end as-is (since end is exclusive in 0-indexed
    = inclusive in 1-indexed).
    """
    gb_start = start + 1
    gb_end = end
    if strand == "reverse":
        return f"complement({gb_start}..{gb_end})"
    return f"{gb_start}..{gb_end}"


def _format_sequence(sequence: str) -> str:
    """Format a DNA sequence into GenBank ORIGIN block format.

    GenBank wraps sequences at 60 chars per line, with a leading position
    number, in groups of 10 separated by spaces.
    """
    seq = sequence.upper()
    lines: list[str] = []
    for i in range(0, len(seq), 60):
        chunk = seq[i : i + 60]
        # Split into groups of 10
        groups = [chunk[j : j + 10] for j in range(0, len(chunk), 10)]
        line = f"{i + 1:>9} " + " ".join(groups)
        lines.append(line)
    return "\n".join(lines)


def _to_genbank(
    part_name: str,
    sequence: str,
    annotations: list[dict[str, Any]],
    title: str = "",
    description: str = "",
) -> str:
    """Convert part data to a GenBank format string.

    Produces valid GenBank that Biopython's SeqIO.parse can read and that
    OpenCloning's ``/read_from_file`` endpoint accepts.
    """
    seq_len = len(sequence)
    # LOCUS line: name padded to 16 chars, length right-justified.
    locus_name = part_name[:16] if part_name else "igem_part"
    locus_line = (
        f"LOCUS       {locus_name:<16} {seq_len:>11} bp    DNA     "
        f"linear   UNK 01-JAN-1980"
    )

    # FEATURES section.
    feature_lines: list[str] = ["FEATURES             Location/Qualifiers"]
    for ann in annotations:
        gb_type = _annotation_to_genbank_type(ann)
        locations = ann.get("locations", [])
        if not locations:
            # Some annotations have start/end at top level.
            start = ann.get("start", 0)
            end = ann.get("end", 0)
            if start is None or end is None or start == end:
                continue
            locations = [{"start": start, "end": end}]

        strand = ann.get("strand", "forward")
        label = ann.get("label", "")

        for loc in locations:
            start = loc.get("start", 0)
            end = loc.get("end", 0)
            if start is None or end is None or start == end:
                continue
            location_str = _format_location(start, end, strand)
            feature_lines.append(f"     {gb_type:<16} {location_str}")
            if label:
                # Escape quotes in label.
                safe_label = label.replace('"', "'")
                feature_lines.append(f'                     /label="{safe_label}"')

    # ORIGIN section.
    origin_block = _format_sequence(sequence)

    # Assemble.
    parts = [
        locus_line,
        *feature_lines,
        "ORIGIN",
        origin_block,
        "//",
    ]
    return "\n".join(parts) + "\n"


def fetch_igem_part_as_genbank(part_name: str) -> str:
    """Fetch an iGEM Registry part and return it as a GenBank string.

    This is the main entry point. It:
    1. Looks up the part by name (e.g. ``BBa_J23105``) via slug.
    2. Fetches its sequence annotations.
    3. Converts both to a GenBank format string.

    The returned GenBank string should be passed to
    ``opencloning.parse_sequence_file`` with ``file_format='genbank'``.

    Args:
        part_name: The iGEM part name (e.g. ``BBa_J23105``, ``BBa_E1010``).

    Returns:
        A GenBank format string containing the part's sequence and features.

    Raises:
        ValueError: If the part is not found in the registry.
        requests.HTTPError: For other API errors.
    """
    part = _fetch_part(part_name)
    sequence = part.get("sequence", "")
    if not sequence:
        raise ValueError(
            f"iGEM part '{part_name}' has no sequence data"
        )

    part_uuid = part.get("uuid", "")
    annotations: list[dict[str, Any]] = []
    if part_uuid:
        try:
            annotations = _fetch_annotations(part_uuid)
        except requests.HTTPError:
            # Annotations are optional — a part without features is still
            # valid. Log and continue with empty annotations.
            annotations = []

    title = part.get("title", "")
    description = part.get("description", "")

    return _to_genbank(part_name, sequence, annotations, title, description)
