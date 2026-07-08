"""Tests for the protocol lookup service (Slice 7).

Uses ``StubElabftwClient`` with pre-seeded protocol entries matching the
convention (Phusion PCR, NEBuilder Gibson, T4 Ligase) to verify:

    1. Exact reagent match returns approved protocol with all fields.
    2. Alias match works (query by alias returns the entry).
    3. Method-only match returns the first approved entry for that method.
    4. No approved protocol → fallback with warning.
    5. Deprecated entries are never selected.
    6. ``validate_corpus`` flags missing required fields.
    7. ``validate_corpus`` flags duplicate aliases.
    8. Empty category returns fallback.
"""

from __future__ import annotations

import json

import pytest

from lab_copilot_gateway.elabftw import StubElabftwClient
from lab_copilot_gateway.protocol_lookup import (
    ProtocolLookupService,
    ProtocolMatch,
)


# ---------------------------------------------------------------------------
# Fixtures — eLabFTW protocol entries
# ---------------------------------------------------------------------------

# Helper: build an item dict with metadata in the extra_fields shape the
# ProtocolLookupService expects (dict values with optional "type"/"value"
# sub-keys, matching how eLabFTW stores them).


def _protocol_item(
    item_id: int,
    title: str,
    *,
    method_type: str = "pcr",
    reagent: str = "",
    aliases: str = "",
    manufacturer: str = "",
    catalog: str = "",
    status: str = "approved",
    approved: bool = True,
    tm_rule: str = "",
    ext_rate: str = "",
    cycles: str = "",
    additive: str = "",
    incubation: str = "",
) -> dict:
    """Build an eLabFTW item dict representing a protocol resource entry."""
    extra_fields: dict = {}
    field_map = {
        "method_type": method_type,
        "reagent_product_name": reagent,
        "aliases": aliases,
        "manufacturer": manufacturer,
        "catalog_number": catalog,
        "status": status,
        "approved_for_use": "1" if approved else "0",
        "annealing_temperature_rule": tm_rule,
        "extension_rate": ext_rate,
        "cycle_count_guidance": cycles,
        "additive_guidance": additive,
        "assembly_incubation_guidance": incubation,
    }
    # Match the eLabFTW extra_fields schema: each field has a dict
    # with "type" and "value" keys.
    for key, value in field_map.items():
        if value:
            extra_fields[key] = {"type": "text", "value": value}
        else:
            extra_fields[key] = {"type": "text", "value": ""}

    metadata = json.dumps({"extra_fields": extra_fields})
    return {
        "id": item_id,
        "title": title,
        "body": (
            f"<p>{title} protocol description</p>\n<p>Method type: {method_type}</p>"
        ),
        "metadata": metadata,
    }


@pytest.fixture
def phusion_item() -> dict:
    return _protocol_item(
        item_id=101,
        title="Phusion High-Fidelity PCR Master Mix",
        method_type="pcr",
        reagent="Phusion High-Fidelity PCR Master Mix",
        aliases="Phusion, F-531L, F-531S",
        manufacturer="Thermo Fisher Scientific",
        catalog="F-531L",
        status="approved",
        approved=True,
        tm_rule="Tm-5",
        ext_rate="15-30 s/kb",
        cycles="25-35",
        additive="3% DMSO for GC-rich templates; 1x GC Buffer as alternative",
    )


@pytest.fixture
def nebuilder_item() -> dict:
    return _protocol_item(
        item_id=102,
        title="NEBuilder HiFi DNA Assembly Master Mix",
        method_type="gibson_assembly",
        reagent="NEBuilder HiFi DNA Assembly Master Mix",
        aliases="NEBuilder, HiFi Assembly, E2621, E2622",
        manufacturer="New England Biolabs",
        catalog="E2621L",
        status="approved",
        approved=True,
        incubation="50°C for 15-60 min (1-3 fragments: 15 min; 4-6: 30 min; 6+: 60 min)",
    )


@pytest.fixture
def t4_ligase_item() -> dict:
    return _protocol_item(
        item_id=103,
        title="T4 DNA Ligase (NEB M0202)",
        method_type="restriction_ligation",
        reagent="T4 DNA Ligase",
        aliases="T4 Ligase, NEB M0202, DNA Ligase",
        manufacturer="New England Biolabs",
        catalog="M0202L",
        status="approved",
        approved=True,
        additive="5% PEG 4000 for blunt-end ligation",
        incubation="16°C for 1-16 h (cohesive ends: 1 h; blunt ends: 16 h) or 22°C for 10 min (quick ligation)",
    )


@pytest.fixture
def draft_pcr_item() -> dict:
    """Non-approved PCR entry (draft) — should NOT be selected."""
    return _protocol_item(
        item_id=201,
        title="Draft PCR Protocol (unreviewed)",
        method_type="pcr",
        reagent="Draft PCR Master Mix",
        status="draft",
        approved=False,
    )


@pytest.fixture
def deprecated_pcr_item() -> dict:
    """Deprecated PCR entry — should NOT be selected."""
    return _protocol_item(
        item_id=202,
        title="Old PCR Master Mix (deprecated)",
        method_type="pcr",
        reagent="Old PCR Mix",
        aliases="old-pcr, deprecated-mix",
        status="deprecated",
        approved=False,
    )


@pytest.fixture
def ligation_item() -> dict:
    return _protocol_item(
        item_id=104,
        title="T4 DNA Ligation (Rapid)",
        method_type="ligation",
        reagent="T4 DNA Ligase Rapid",
        status="approved",
        approved=True,
        incubation="22°C for 5 min",
    )


@pytest.fixture
def stub_client(
    phusion_item: dict,
    nebuilder_item: dict,
    t4_ligase_item: dict,
    draft_pcr_item: dict,
    deprecated_pcr_item: dict,
    ligation_item: dict,
) -> StubElabftwClient:
    """Stub client pre-seeded with protocol entries.

    The stub's ``search_items`` does free-text matching against title/body,
    so entries are findable by method-type keywords in their titles.
    """
    seeds: dict[int, dict] = {
        101: phusion_item,
        102: nebuilder_item,
        103: t4_ligase_item,
        104: ligation_item,
        201: draft_pcr_item,
        202: deprecated_pcr_item,
    }
    return StubElabftwClient(seeds=seeds)


@pytest.fixture
def lookup_service(stub_client: StubElabftwClient) -> ProtocolLookupService:
    return ProtocolLookupService(elabftw_client=stub_client)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first_approved(
    items: list[ProtocolMatch],
) -> ProtocolMatch | None:
    """Return the first entry with approved status."""
    for m in items:
        if m.status == "approved" and m.approved_for_use:
            return m
    return None


# ---------------------------------------------------------------------------
# Tests: lookup
# ---------------------------------------------------------------------------


class TestLookupExactMatch:
    """1. Exact reagent name match returns approved protocol with all fields."""

    def test_phusion_pcr_exact_match(
        self, lookup_service: ProtocolLookupService
    ) -> None:
        result = lookup_service.lookup(
            method_type="pcr",
            reagent_name="Phusion High-Fidelity PCR Master Mix",
        )
        assert not result.fallback_used
        assert len(result.matches) >= 1
        assert result.best_match is not None
        assert result.best_match.match_confidence == "exact"
        assert (
            result.best_match.reagent_product_name
            == "Phusion High-Fidelity PCR Master Mix"
        )
        assert result.best_match.annealing_temperature_rule == "Tm-5"
        assert result.best_match.extension_rate == "15-30 s/kb"
        assert result.best_match.cycle_count_guidance == "25-35"
        assert (
            result.best_match.additive_guidance
            == "3% DMSO for GC-rich templates; 1x GC Buffer as alternative"
        )
        assert result.best_match.manufacturer == "Thermo Fisher Scientific"
        assert result.best_match.catalog_number == "F-531L"
        assert result.best_match.aliases == ["Phusion", "F-531L", "F-531S"]

    def test_nebuilder_exact_match(self, lookup_service: ProtocolLookupService) -> None:
        result = lookup_service.lookup(
            method_type="gibson_assembly",
            reagent_name="NEBuilder HiFi DNA Assembly Master Mix",
        )
        assert not result.fallback_used
        assert result.best_match is not None
        assert result.best_match.match_confidence == "exact"
        assert result.best_match.assembly_incubation_guidance is not None


class TestLookupAliasMatch:
    """2. Alias match works."""

    def test_alias_match_by_product_name(
        self, lookup_service: ProtocolLookupService
    ) -> None:
        """Query "Phusion" matches the Phusion entry (it's in the aliases list)."""
        result = lookup_service.lookup(
            method_type="pcr",
            reagent_name="Phusion",
        )
        assert result.best_match is not None
        assert result.best_match.match_confidence == "alias"
        assert "Phusion" in result.best_match.reagent_product_name

    def test_alias_match_by_f_531(self, lookup_service: ProtocolLookupService) -> None:
        """Query "F-531L" matches via catalog number alias."""
        result = lookup_service.lookup(
            method_type="pcr",
            reagent_name="F-531L",
        )
        assert result.best_match is not None
        assert result.best_match.match_confidence == "alias"

    def test_alias_match_by_supplied_aliases(
        self, lookup_service: ProtocolLookupService
    ) -> None:
        """Query with user-supplied alias ("Phusion MM") matches."""
        result = lookup_service.lookup(
            method_type="pcr",
            reagent_name="Some Unknown Name",
            aliases=["Phusion High-Fidelity PCR Master Mix"],
        )
        assert result.best_match is not None
        assert result.best_match.match_confidence == "alias"

    def test_alias_match_nebuilder(self, lookup_service: ProtocolLookupService) -> None:
        result = lookup_service.lookup(
            method_type="gibson_assembly",
            reagent_name="NEBuilder",
        )
        assert result.best_match is not None
        assert result.best_match.match_confidence == "alias"

    def test_alias_match_e2621(self, lookup_service: ProtocolLookupService) -> None:
        result = lookup_service.lookup(
            method_type="gibson_assembly",
            reagent_name="E2621",
        )
        assert result.best_match is not None
        assert result.best_match.match_confidence == "alias"


class TestLookupMethodOnly:
    """3. Method-only match returns the first approved entry."""

    def test_method_only_pcr(self, lookup_service: ProtocolLookupService) -> None:
        """No reagent name — returns the first approved PCR entry."""
        result = lookup_service.lookup(method_type="pcr")
        assert result.best_match is not None
        assert result.best_match.match_confidence == "method_only"
        assert result.best_match.method_type == "pcr"
        assert result.best_match.status == "approved"
        assert result.best_match.approved_for_use

    def test_method_only_gibson(self, lookup_service: ProtocolLookupService) -> None:
        result = lookup_service.lookup(method_type="gibson_assembly")
        assert result.best_match is not None
        assert result.best_match.match_confidence == "method_only"
        assert result.best_match.method_type == "gibson_assembly"
        assert (
            result.best_match.reagent_product_name
            == "NEBuilder HiFi DNA Assembly Master Mix"
        )

    def test_method_only_ligation(self, lookup_service: ProtocolLookupService) -> None:
        result = lookup_service.lookup(method_type="ligation")
        assert result.best_match is not None
        assert result.best_match.match_confidence == "method_only"
        assert result.best_match.method_type == "ligation"


class TestLookupFallback:
    """4. No approved protocol → fallback with warning."""

    def test_no_protocols_at_all(self, lookup_service: ProtocolLookupService) -> None:
        """Method type that doesn't match any entry."""
        result = lookup_service.lookup(method_type="golden_gate")
        assert result.fallback_used
        assert result.best_match is None
        assert len(result.matches) == 0
        assert len(result.warnings) > 0

    def test_only_draft_entries_available(
        self, lookup_service: ProtocolLookupService
    ) -> None:
        """When no approved but unapproved entries exist, fallback + show them."""
        result = lookup_service.lookup(
            method_type="pcr", reagent_name="Unmatched Reagent X"
        )
        # At least one entry is approved (Phusion), so this should find it
        # with a method_only fallback.
        assert result.best_match is not None
        # The warning says "no exact or alias match" since the reagent doesn't match
        assert len(result.best_match.warnings) > 0


class TestLookupDeprecated:
    """5. Deprecated entries are never selected."""

    def test_deprecated_not_in_matches(
        self, lookup_service: ProtocolLookupService
    ) -> None:
        """Deprecated items should NOT appear in matches, only approved ones."""
        # Search for PCR — should find Phusion but NOT the deprecated entry
        result = lookup_service.lookup(method_type="pcr")
        for match in result.matches:
            assert match.status == "approved", (
                f"Deprecated entry {match.title} was selected"
            )
            assert match.approved_for_use

    def test_deprecated_not_returned_by_reagent_query(
        self, lookup_service: ProtocolLookupService
    ) -> None:
        """Even if the deprecated entry matches the reagent name, it's excluded."""
        result = lookup_service.lookup(
            method_type="pcr",
            reagent_name="Old PCR Mix",
        )
        # The deprecated entry shares method_type=pcr but should be excluded.
        # We may still find the Phusion entry if it matches.
        assert result.best_match is not None
        # Make sure Phusion is the best_match, not the deprecated one
        assert result.best_match.reagent_product_name != "Old PCR Mix"


# ---------------------------------------------------------------------------
# Tests: validate_corpus
# ---------------------------------------------------------------------------


class TestValidateCorpus:
    """6-7. Corpus validation."""

    def test_validate_corpus_happy_path(
        self, lookup_service: ProtocolLookupService
    ) -> None:
        """Valid entries produce no errors."""
        issues = lookup_service.validate_corpus()
        # Should find the Phusion, NEBuilder, T4, and Ligation entries.
        # No errors expected for valid entries.
        errors = [i for i in issues if i.get("severity") == "error"]
        assert len(errors) == 0, f"Unexpected errors: {errors}"

    def test_validate_corpus_missing_required_fields(
        self, stub_client: StubElabftwClient
    ) -> None:
        """Add an entry missing method_type and status."""
        bad_item = {
            "id": 301,
            "title": "Bad Protocol Entry",
            "body": "<p>Missing required fields</p><p>Method type: pcr</p>",
            "metadata": json.dumps(
                {
                    "extra_fields": {
                        "reagent_product_name": {
                            "type": "text",
                            "value": "Some Reagent",
                        },
                    }
                }
            ),
        }
        stub_client.seeds[301] = bad_item
        service = ProtocolLookupService(elabftw_client=stub_client)
        issues = service.validate_corpus()
        errors = [i for i in issues if i.get("severity") == "error"]
        assert any(
            "method_type" in i["message"] or "status" in i["message"] for i in errors
        ), f"No missing-field error: {errors}"

    def test_validate_corpus_duplicate_aliases(
        self, stub_client: StubElabftwClient
    ) -> None:
        """Two entries sharing an alias should be flagged."""
        # Add an entry with a duplicate alias
        # Reason: the stub search is text-based, so we must ensure the
        # duplicate alias entry has a searchable keyword in its title/body.
        dup_item = _protocol_item(
            item_id=401,
            title="Another Phusion-like PCR Mix",
            method_type="pcr",
            reagent="Phusion Clone",
            aliases="Phusion, copy-cat",  # "Phusion" is already used by item 101
            status="draft",
            approved=False,
        )
        stub_client.seeds[401] = dup_item
        service = ProtocolLookupService(elabftw_client=stub_client)
        issues = service.validate_corpus()
        alias_warnings = [
            i for i in issues if "Duplicate alias" in i.get("message", "")
        ]
        assert len(alias_warnings) >= 1, f"No duplicate alias warning: {issues}"

    def test_validate_corpus_deprecated_flag(
        self, lookup_service: ProtocolLookupService
    ) -> None:
        """Deprecated entries should generate warnings."""
        issues = lookup_service.validate_corpus()
        deprecated_warnings = [
            i for i in issues if "deprecated" in i.get("message", "").lower()
        ]
        assert len(deprecated_warnings) >= 1, f"No deprecated warnings: {issues}"

    def test_validate_corpus_non_approved_warnings(
        self, lookup_service: ProtocolLookupService
    ) -> None:
        """Draft and review entries generate status warnings."""
        issues = lookup_service.validate_corpus()
        status_warnings = [
            i for i in issues if "not selectable" in i.get("message", "")
        ]
        # We have: draft_pcr_item (draft), deprecated_pcr_item (deprecated)
        assert len(status_warnings) >= 2, (
            f"Not enough status warnings: {status_warnings}"
        )


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestLookupEdgeCases:
    """Edge cases and error handling."""

    def test_empty_method_type(self, lookup_service: ProtocolLookupService) -> None:
        result = lookup_service.lookup(method_type="")
        assert result.fallback_used
        assert result.best_match is None
        assert "method_type is required" in result.warnings

    def test_empty_category(self) -> None:
        """No items in the stub → fallback."""
        empty_client = StubElabftwClient(seeds={})
        service = ProtocolLookupService(elabftw_client=empty_client)
        result = service.lookup(method_type="pcr")
        assert result.fallback_used
        assert result.best_match is None

    def test_item_without_metadata(self, stub_client: StubElabftwClient) -> None:
        """An item without matching metadata should be skipped silently."""
        stub_client.seeds[501] = {
            "id": 501,
            "title": "Random Plasmid",
            "body": "<p>Not a protocol</p>",
            "metadata": None,
        }
        service = ProtocolLookupService(elabftw_client=stub_client)
        result = service.lookup(method_type="pcr")
        assert result.best_match is not None
        # The random item should be ignored, and the Phusion entry should be found

    def test_to_dict_roundtrip(self, lookup_service: ProtocolLookupService) -> None:
        """ProtocolLookupResult.to_dict() is JSON-safe."""
        result = lookup_service.lookup(
            method_type="pcr",
            reagent_name="Phusion High-Fidelity PCR Master Mix",
        )
        d = result.to_dict()
        assert d["query_method"] == "pcr"
        assert d["fallback_used"] is False
        assert len(d["matches"]) >= 1
        bm = d["best_match"]
        assert bm is not None
        assert bm["match_confidence"] == "exact"
        assert bm["reagent_product_name"] == "Phusion High-Fidelity PCR Master Mix"
        assert "warnings" in bm
        json.dumps(d)  # Must be JSON-serializable
