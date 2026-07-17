"""Tests for the canonical OpenCloning capability catalog (Slice 1).

Acceptance checks (from the umbrella plan):

    1. Catalog covers every agreed product operation exactly once.
    2. Version drift and unknown source types fail clearly.
    3. No external-service success dependency in required CI.
    4. GoldenBraid/NCBI-dependent methods are experimental.
    5. Aliases (Golden Gate vs. restriction/ligation) are distinguished.
"""

from __future__ import annotations

import pytest

from lab_copilot_gateway.strategy_catalog import (
    PINNED_OPENCLONING_IMAGE,
    CapabilityStatus,
    Cardinality,
    OperationCapability,
    StrategyCatalog,
    UnknownSourceTypeError,
    VisualFamily,
    get_catalog,
    reset_catalog,
)


# --- Catalog structure -------------------------------------------------------


class TestCatalogStructure:
    """Catalog invariants: uniqueness, completeness, no gaps."""

    def test_catalog_is_singleton(self) -> None:
        """get_catalog() returns the same instance."""
        c1 = get_catalog()
        c2 = get_catalog()
        assert c1 is c2

    def test_catalog_has_entries(self) -> None:
        """Catalog is not empty."""
        assert len(get_catalog().list_all()) > 0

    def test_operation_keys_are_unique(self) -> None:
        """Every operation_key appears exactly once."""
        keys = [e.operation_key for e in get_catalog().list_all()]
        assert len(keys) == len(set(keys))

    def test_source_types_can_be_shared(self) -> None:
        """Multiple operation keys can share a source type (e.g. Golden Gate
        and restriction/ligation both use RestrictionAndLigationSource)."""
        catalog = get_catalog()
        rl_ops = catalog.get_by_source_type("RestrictionAndLigationSource")
        assert len(rl_ops) == 2  # restriction_ligation + golden_gate
        keys = {op.operation_key for op in rl_ops}
        assert keys == {"restriction_ligation", "golden_gate"}

    def test_all_endpoints_start_with_slash(self) -> None:
        """Every endpoint path starts with '/'."""
        for e in get_catalog().list_all():
            assert e.endpoint.startswith("/"), f"{e.operation_key}: {e.endpoint}"

    def test_all_descriptions_nonempty(self) -> None:
        """Every entry has a non-empty description."""
        for e in get_catalog().list_all():
            assert e.description, f"{e.operation_key} has empty description"


# --- Required operations -----------------------------------------------------


class TestRequiredOperations:
    """Every in-scope product-generating operation is cataloged."""

    REQUIRED_OPERATION_KEYS = {
        # PCR
        "pcr",
        # Restriction
        "restriction_digest",
        # Assembly
        "gibson_assembly",
        "in_fusion_assembly",
        "overlap_extension_pcr",
        "in_vivo_assembly",
        "ligation",
        "restriction_ligation",
        "golden_gate",
        # Homologous recombination
        "homologous_recombination",
        "crispr_hdr",
        # Site-specific recombination
        "cre_lox",
        "gateway",
        "recombinase",
        # Small products
        "oligo_hybridization",
        "polymerase_extension",
        "reverse_complement",
    }

    REQUIRED_SOURCE_TYPES = {
        "PCRSource",
        "RestrictionEnzymeDigestionSource",
        "GibsonAssemblySource",
        "InFusionSource",
        "OverlapExtensionPCRLigationSource",
        "InVivoAssemblySource",
        "LigationSource",
        "RestrictionAndLigationSource",
        "HomologousRecombinationSource",
        "CRISPRSource",
        "CreLoxRecombinationSource",
        "GatewaySource",
        "RecombinaseSource",
        "OligoHybridizationSource",
        "PolymeraseExtensionSource",
        "ReverseComplementSource",
    }

    def test_all_required_operation_keys_present(self) -> None:
        catalog_keys = {e.operation_key for e in get_catalog().list_all()}
        missing = self.REQUIRED_OPERATION_KEYS - catalog_keys
        assert not missing, f"missing operation keys: {sorted(missing)}"

    def test_all_required_source_types_present(self) -> None:
        catalog_types = {e.source_type for e in get_catalog().list_all()}
        missing = self.REQUIRED_SOURCE_TYPES - catalog_types
        assert not missing, f"missing source types: {sorted(missing)}"

    def test_golden_gate_distinct_from_restriction_ligation(self) -> None:
        """Golden Gate and restriction/ligation share an endpoint but are
        distinct operation keys."""
        catalog = get_catalog()
        gg = catalog.get_by_operation_key("golden_gate")
        rl = catalog.get_by_operation_key("restriction_ligation")
        assert gg is not None
        assert rl is not None
        assert gg.endpoint == rl.endpoint
        assert gg.operation_key != rl.operation_key
        assert "restriction_ligation" in gg.aliases

    def test_in_fusion_uses_gibson_endpoint(self) -> None:
        """In-Fusion, overlap-extension PCR, and in-vivo assembly all
        route through /gibson_assembly but are distinct operations."""
        catalog = get_catalog()
        gibson = catalog.get_by_operation_key("gibson_assembly")
        infusion = catalog.get_by_operation_key("in_fusion_assembly")
        overlap = catalog.get_by_operation_key("overlap_extension_pcr")
        invivo = catalog.get_by_operation_key("in_vivo_assembly")
        assert all(e is not None for e in (gibson, infusion, overlap, invivo))
        assert gibson.endpoint == "/gibson_assembly"
        assert infusion.endpoint == "/gibson_assembly"
        assert overlap.endpoint == "/gibson_assembly"
        assert invivo.endpoint == "/gibson_assembly"
        assert "gibson_assembly" in infusion.aliases


# --- Unknown source type rejection ------------------------------------------


class TestUnknownSourceTypeRejection:
    """Unknown source types must fail explicitly — no Gibson fallback."""

    def test_unknown_source_type_raises(self) -> None:
        with pytest.raises(UnknownSourceTypeError) as exc_info:
            get_catalog().resolve_endpoint("NonexistentSource")
        assert "NonexistentSource" in str(exc_info.value)

    def test_unknown_source_type_error_lists_known_types(self) -> None:
        """Error message should help the caller understand what's valid."""
        with pytest.raises(UnknownSourceTypeError) as exc_info:
            get_catalog().resolve_endpoint("BogusSource")
        msg = str(exc_info.value)
        assert "GibsonAssemblySource" in msg
        assert "PCRSource" in msg

    def test_empty_source_type_raises(self) -> None:
        with pytest.raises(UnknownSourceTypeError):
            get_catalog().resolve_endpoint("")

    def test_simulate_assembly_no_gibson_fallback(self) -> None:
        """HttpOpenCloningClient.simulate_assembly must not fall back to
        /gibson_assembly for unknown source types."""
        from lab_copilot_gateway.opencloning import HttpOpenCloningClient

        client = HttpOpenCloningClient(base_url="http://fake:9999")
        with pytest.raises(UnknownSourceTypeError):
            # We can't actually make an HTTP call, but the catalog
            # resolution happens before any HTTP request.
            client.simulate_assembly(
                sequences=[],
                source={"type": "TotallyBogusSource", "id": 0},
            )


# --- Experimental gating -----------------------------------------------------


class TestExperimentalGating:
    """GoldenBraid/NCBI-dependent methods are experimental."""

    EXPERIMENTAL_KEYS = {
        "batch_domestication",
        "batch_gibson",
        "batch_restriction_ligation",
        "batch_homologous_recombination",
    }

    def test_experimental_operations_marked(self) -> None:
        experimental = {e.operation_key for e in get_catalog().list_experimental()}
        assert self.EXPERIMENTAL_KEYS.issubset(experimental)

    def test_stable_operations_exclude_experimental(self) -> None:
        stable = {e.operation_key for e in get_catalog().list_stable()}
        assert not (self.EXPERIMENTAL_KEYS & stable)

    def test_public_dict_excludes_experimental_by_default(self) -> None:
        """to_public_dict() must not advertise experimental operations
        unless explicitly requested."""
        pub = get_catalog().to_public_dict()
        experimental_keys = {e.operation_key for e in get_catalog().list_experimental()}
        advertised_keys = {op["operation_key"] for op in pub["operations"]}
        assert not (experimental_keys & advertised_keys)

    def test_public_dict_includes_experimental_when_requested(self) -> None:
        pub = get_catalog().to_public_dict(include_experimental=True)
        advertised_keys = {op["operation_key"] for op in pub["operations"]}
        assert "batch_domestication" in advertised_keys


# --- Endpoint resolution -----------------------------------------------------


class TestEndpointResolution:
    """resolve_endpoint maps source types to correct endpoints."""

    @pytest.mark.parametrize(
        "source_type,expected_endpoint",
        [
            ("GibsonAssemblySource", "/gibson_assembly"),
            ("LigationSource", "/ligation"),
            ("RestrictionAndLigationSource", "/restriction_and_ligation"),
            ("OverlapExtensionPCRLigationSource", "/gibson_assembly"),
            ("InFusionSource", "/gibson_assembly"),
            ("InVivoAssemblySource", "/gibson_assembly"),
            ("HomologousRecombinationSource", "/homologous_recombination"),
            ("CRISPRSource", "/crispr"),
            ("CreLoxRecombinationSource", "/cre_lox_recombination"),
            ("GatewaySource", "/gateway"),
            ("RecombinaseSource", "/recombinase"),
        ],
    )
    def test_endpoint_resolution(
        self, source_type: str, expected_endpoint: str
    ) -> None:
        assert get_catalog().resolve_endpoint(source_type) == expected_endpoint


# --- Visual family coverage --------------------------------------------------


class TestVisualFamilyCoverage:
    """Every visual family is represented in the catalog."""

    def test_all_visual_families_present(self) -> None:
        families = {e.visual_family for e in get_catalog().list_all()}
        for family in VisualFamily:
            assert family in families, f"missing visual family: {family.value}"

    def test_assembly_family_has_multiple_operations(self) -> None:
        assembly_ops = [
            e
            for e in get_catalog().list_all()
            if e.visual_family == VisualFamily.ASSEMBLY
        ]
        assert (
            len(assembly_ops) >= 5
        )  # Gibson, InFusion, overlap, in-vivo, ligation, R&L, Golden Gate

    def test_recombination_family_has_three_operations(self) -> None:
        recomb_ops = [
            e
            for e in get_catalog().list_all()
            if e.visual_family == VisualFamily.RECOMBINATION
        ]
        assert len(recomb_ops) == 3  # Cre/Lox, Gateway, recombinase


# --- Cardinality coverage ----------------------------------------------------


class TestCardinalityCoverage:
    """Every cardinality type is represented."""

    def test_unary_operations_exist(self) -> None:
        unary = [
            e for e in get_catalog().list_all() if e.cardinality == Cardinality.UNARY
        ]
        assert (
            len(unary) >= 3
        )  # PCR, restriction, polymerase extension, reverse complement

    def test_binary_operations_exist(self) -> None:
        binary = [
            e for e in get_catalog().list_all() if e.cardinality == Cardinality.BINARY
        ]
        assert len(binary) >= 1  # oligo hybridization

    def test_n_ary_operations_exist(self) -> None:
        n_ary = [
            e for e in get_catalog().list_all() if e.cardinality == Cardinality.N_ARY
        ]
        assert len(n_ary) >= 5  # Gibson, ligation, R&L, HR, CRISPR, etc.

    def test_batch_coupled_operations_exist(self) -> None:
        batch = [
            e
            for e in get_catalog().list_all()
            if e.cardinality == Cardinality.BATCH_COUPLED
        ]
        assert len(batch) >= 1


# --- Pinned image constant ---------------------------------------------------


class TestPinnedImage:
    """The catalog is validated against a specific pinned image."""

    def test_pinned_image_constant_exists(self) -> None:
        assert PINNED_OPENCLONING_IMAGE
        assert "opencloning" in PINNED_OPENCLONING_IMAGE

    def test_pinned_image_has_version_tag(self) -> None:
        assert ":" in PINNED_OPENCLONING_IMAGE
        tag = PINNED_OPENCLONING_IMAGE.split(":")[-1]
        assert tag  # not empty


# --- Catalog construction validation -----------------------------------------


class TestCatalogValidation:
    """Catalog construction rejects invalid entries."""

    def test_duplicate_operation_keys_rejected(self) -> None:
        entry = OperationCapability(
            operation_key="pcr",
            source_type="PCRSource",
            endpoint="/pcr",
            cardinality=Cardinality.UNARY,
            visual_family=VisualFamily.PCR,
            status=CapabilityStatus.STABLE,
            description="test",
        )
        with pytest.raises(ValueError, match="duplicate operation_key"):
            StrategyCatalog((entry, entry))

    def test_conflicting_endpoints_for_same_source_type_rejected(self) -> None:
        """If two entries share a source_type, they must agree on endpoint."""
        e1 = OperationCapability(
            operation_key="op_a",
            source_type="SharedSource",
            endpoint="/endpoint_a",
            cardinality=Cardinality.UNARY,
            visual_family=VisualFamily.PCR,
            status=CapabilityStatus.STABLE,
            description="test 1",
        )
        e2 = OperationCapability(
            operation_key="op_b",
            source_type="SharedSource",
            endpoint="/endpoint_b",
            cardinality=Cardinality.UNARY,
            visual_family=VisualFamily.PCR,
            status=CapabilityStatus.STABLE,
            description="test 2",
        )
        with pytest.raises(ValueError, match="conflicting endpoints"):
            StrategyCatalog((e1, e2))

    def test_empty_operation_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="operation_key must not be empty"):
            OperationCapability(
                operation_key="",
                source_type="TestSource",
                endpoint="/test",
                cardinality=Cardinality.UNARY,
                visual_family=VisualFamily.PCR,
                status=CapabilityStatus.STABLE,
                description="test",
            )

    def test_endpoint_without_slash_rejected(self) -> None:
        with pytest.raises(ValueError, match="endpoint must start with '/'"):
            OperationCapability(
                operation_key="test",
                source_type="TestSource",
                endpoint="test",
                cardinality=Cardinality.UNARY,
                visual_family=VisualFamily.PCR,
                status=CapabilityStatus.STABLE,
                description="test",
            )

    def test_empty_description_rejected(self) -> None:
        with pytest.raises(ValueError, match="description must not be empty"):
            OperationCapability(
                operation_key="test",
                source_type="TestSource",
                endpoint="/test",
                cardinality=Cardinality.UNARY,
                visual_family=VisualFamily.PCR,
                status=CapabilityStatus.STABLE,
                description="",
            )


# --- Reset singleton ---------------------------------------------------------


class TestResetCatalog:
    """reset_catalog is a test helper that clears the singleton."""

    def test_reset_clears_singleton(self) -> None:
        original = get_catalog()
        reset_catalog()
        new = get_catalog()
        assert new is not original
        # Restore for other tests
        reset_catalog(original)

    def test_reset_with_custom_catalog(self) -> None:
        custom = StrategyCatalog(
            (
                OperationCapability(
                    operation_key="custom_op",
                    source_type="CustomSource",
                    endpoint="/custom",
                    cardinality=Cardinality.UNARY,
                    visual_family=VisualFamily.PCR,
                    status=CapabilityStatus.STABLE,
                    description="custom test entry",
                ),
            )
        )
        reset_catalog(custom)
        assert get_catalog().get_by_operation_key("custom_op") is not None
        assert get_catalog().get_by_operation_key("pcr") is None
        reset_catalog()  # restore default
