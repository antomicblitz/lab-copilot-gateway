"""Canonical OpenCloning capability catalog (Slice 1).

Defines every product-generating operation supported by the pinned
OpenCloning backend (``manulera/opencloning:v1.3.1-baseurl-opencloning``)
as a structured, validated catalog entry.

This module is the **single source of truth** for what the gateway
advertises as supported.  The catalog is consumed by:

* ``GET /strategy/capabilities`` (Slice 3) — surfaces the catalog to the
  orchestrator and UI.
* ``strategy_executor.py`` (Slice 6) — maps operation keys to endpoints.
* ``strategy_validation.py`` (Slice 2) — validates that a strategy DAG
  only references cataloged operations.

Design principles
-----------------

* Every in-scope product-generating operation appears **exactly once**.
* Aliases (Golden Gate vs. restriction/ligation) are distinguished by
  ``operation_key``, not by overloading source types.
* Unknown source types **must** fail explicitly — no Gibson fallback.
* GoldenBraid/NCBI-dependent native batch methods are gated as
  ``CapabilityStatus.EXPERIMENTAL``.
* No nucleotide sequences are stored or logged by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class CapabilityStatus(str, Enum):
    """Lifecycle status of a cataloged operation."""

    STABLE = "stable"
    """Operation is fully supported and tested against the pinned backend."""

    EXPERIMENTAL = "experimental"
    """Operation depends on external services (GoldenBraid, NCBI) and may
    break independently of this deployment.  Must not be advertised unless
    explicitly enabled."""


class VisualFamily(str, Enum):
    """Visual family for rendering in the strategy review UI.

    Determines the topology shape and method-specific motif.  The UI
    (Slice 5) uses this to select the correct rendering template — it
    does not infer biology from node shapes.
    """

    ASSEMBLY = "assembly"
    """Multi-input convergence: Gibson, ligation, Golden Gate, etc."""

    RESTRICTION = "restriction"
    """Single-input cut: restriction digest."""

    PCR = "pcr"
    """Single-input amplification: PCR, overlap-extension PCR."""

    RECOMBINATION = "recombination"
    """Site-specific recombination: Cre/Lox, Gateway, recombinase."""

    HOMOLOGOUS_RECOMBINATION = "homologous_recombination"
    """HR integration/excision/inversion + CRISPR-HDR."""

    OLIGO = "oligo"
    """Small product: oligo hybridization, polymerase extension, reverse complement."""

    BATCH = "batch"
    """Native batch endpoint: domestication, pombe, etc."""


class Cardinality(str, Enum):
    """Input/output cardinality for an operation."""

    UNARY = "unary"
    """One input → one or more outputs (restriction, PCR, extension)."""

    BINARY = "binary"
    """Two inputs → one output (oligo hybridization)."""

    N_ARY = "n_ary"
    """N inputs → one or more outputs (Gibson, ligation, Golden Gate)."""

    BATCH_INDEPENDENT = "batch_independent"
    """Up to 100 independent members, each with its own sub-graph."""

    BATCH_COUPLED = "batch_coupled"
    """Native batch endpoint with all-or-nothing semantics."""


@dataclass(frozen=True)
class OperationCapability:
    """A single product-generating operation in the catalog.

    Fields:
        operation_key: Stable, unique identifier (e.g. ``"pcr"``,
            ``"gibson_assembly"``).  Used in strategy schema v3.
        source_type: OpenCloning source type string (e.g.
            ``"GibsonAssemblySource"``).  Used to route to the correct
            backend endpoint.
        endpoint: OpenCloning API path (e.g. ``"/gibson_assembly"``).
        cardinality: Input/output shape.
        visual_family: Rendering family for the UI.
        status: Capability lifecycle status.
        description: Human-readable summary for the catalog API.
        aliases: Other operation keys that map to the same endpoint but
            represent distinct biological methods (e.g. Golden Gate and
            conventional restriction/ligation both use
            ``/restriction_and_ligation`` but are different operations).
    """

    operation_key: str
    source_type: str
    endpoint: str
    cardinality: Cardinality
    visual_family: VisualFamily
    status: CapabilityStatus
    description: str
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.operation_key:
            raise ValueError("operation_key must not be empty")
        if not self.source_type:
            raise ValueError("source_type must not be empty")
        if not self.endpoint.startswith("/"):
            raise ValueError(f"endpoint must start with '/': {self.endpoint}")
        if not self.description:
            raise ValueError("description must not be empty")


# --- Catalog entries ---------------------------------------------------------
#
# Every product-generating operation in the pinned OpenCloning release.
# Ordered by visual family for readability.

_CATALOG_ENTRIES: tuple[OperationCapability, ...] = (
    # --- PCR family -------------------------------------------------------
    OperationCapability(
        operation_key="pcr",
        source_type="PCRSource",
        endpoint="/pcr",
        cardinality=Cardinality.UNARY,
        visual_family=VisualFamily.PCR,
        status=CapabilityStatus.STABLE,
        description="Polymerase chain reaction amplification from a template and primer pair.",
    ),
    # --- Restriction family ----------------------------------------------
    OperationCapability(
        operation_key="restriction_digest",
        source_type="RestrictionEnzymeDigestionSource",
        endpoint="/restriction",
        cardinality=Cardinality.UNARY,
        visual_family=VisualFamily.RESTRICTION,
        status=CapabilityStatus.STABLE,
        description="Restriction enzyme digest of a single substrate.",
    ),
    # --- Assembly family -------------------------------------------------
    OperationCapability(
        operation_key="gibson_assembly",
        source_type="GibsonAssemblySource",
        endpoint="/gibson_assembly",
        cardinality=Cardinality.N_ARY,
        visual_family=VisualFamily.ASSEMBLY,
        status=CapabilityStatus.STABLE,
        description="Gibson isothermal assembly of overlapping fragments.",
    ),
    OperationCapability(
        operation_key="in_fusion_assembly",
        source_type="InFusionSource",
        endpoint="/gibson_assembly",
        cardinality=Cardinality.N_ARY,
        visual_family=VisualFamily.ASSEMBLY,
        status=CapabilityStatus.STABLE,
        description="In-Fusion cloning assembly (routed through Gibson endpoint).",
        aliases=("gibson_assembly",),
    ),
    OperationCapability(
        operation_key="overlap_extension_pcr",
        source_type="OverlapExtensionPCRLigationSource",
        endpoint="/gibson_assembly",
        cardinality=Cardinality.N_ARY,
        visual_family=VisualFamily.ASSEMBLY,
        status=CapabilityStatus.STABLE,
        description="Overlap-extension PCR assembly (routed through Gibson endpoint).",
        aliases=("gibson_assembly",),
    ),
    OperationCapability(
        operation_key="in_vivo_assembly",
        source_type="InVivoAssemblySource",
        endpoint="/gibson_assembly",
        cardinality=Cardinality.N_ARY,
        visual_family=VisualFamily.ASSEMBLY,
        status=CapabilityStatus.STABLE,
        description="In-vivo homologous recombination assembly (routed through Gibson endpoint).",
        aliases=("gibson_assembly",),
    ),
    OperationCapability(
        operation_key="ligation",
        source_type="LigationSource",
        endpoint="/ligation",
        cardinality=Cardinality.N_ARY,
        visual_family=VisualFamily.ASSEMBLY,
        status=CapabilityStatus.STABLE,
        description="Conventional ligation of compatible-ended fragments.",
    ),
    OperationCapability(
        operation_key="restriction_ligation",
        source_type="RestrictionAndLigationSource",
        endpoint="/restriction_and_ligation",
        cardinality=Cardinality.N_ARY,
        visual_family=VisualFamily.ASSEMBLY,
        status=CapabilityStatus.STABLE,
        description="Combined restriction digest and ligation in one reaction.",
    ),
    OperationCapability(
        operation_key="golden_gate",
        source_type="RestrictionAndLigationSource",
        endpoint="/restriction_and_ligation",
        cardinality=Cardinality.N_ARY,
        visual_family=VisualFamily.ASSEMBLY,
        status=CapabilityStatus.STABLE,
        description=(
            "Golden Gate assembly using Type IIS restriction enzymes and "
            "compatible overhangs.  Distinct from conventional restriction/ligation "
            "despite sharing the same backend endpoint."
        ),
        aliases=("restriction_ligation",),
    ),
    # --- Homologous recombination family --------------------------------
    OperationCapability(
        operation_key="homologous_recombination",
        source_type="HomologousRecombinationSource",
        endpoint="/homologous_recombination",
        cardinality=Cardinality.N_ARY,
        visual_family=VisualFamily.HOMOLOGOUS_RECOMBINATION,
        status=CapabilityStatus.STABLE,
        description="Homologous recombination: integration, excision, or inversion.",
    ),
    OperationCapability(
        operation_key="crispr_hdr",
        source_type="CRISPRSource",
        endpoint="/crispr",
        cardinality=Cardinality.N_ARY,
        visual_family=VisualFamily.HOMOLOGOUS_RECOMBINATION,
        status=CapabilityStatus.STABLE,
        description="CRISPR-mediated homology-directed repair with guide RNA and repair template.",
    ),
    # --- Site-specific recombination family ------------------------------
    OperationCapability(
        operation_key="cre_lox",
        source_type="CreLoxRecombinationSource",
        endpoint="/cre_lox_recombination",
        cardinality=Cardinality.N_ARY,
        visual_family=VisualFamily.RECOMBINATION,
        status=CapabilityStatus.STABLE,
        description="Cre/Lox site-specific recombination: integration, excision, or inversion.",
    ),
    OperationCapability(
        operation_key="gateway",
        source_type="GatewaySource",
        endpoint="/gateway",
        cardinality=Cardinality.N_ARY,
        visual_family=VisualFamily.RECOMBINATION,
        status=CapabilityStatus.STABLE,
        description="Gateway BP/LR recombination between att-containing sites.",
    ),
    OperationCapability(
        operation_key="recombinase",
        source_type="RecombinaseSource",
        endpoint="/recombinase",
        cardinality=Cardinality.N_ARY,
        visual_family=VisualFamily.RECOMBINATION,
        status=CapabilityStatus.STABLE,
        description="Generic recombinase-mediated site-specific recombination.",
    ),
    # --- Small product family --------------------------------------------
    OperationCapability(
        operation_key="oligo_hybridization",
        source_type="OligoHybridizationSource",
        endpoint="/oligonucleotide_hybridization",
        cardinality=Cardinality.BINARY,
        visual_family=VisualFamily.OLIGO,
        status=CapabilityStatus.STABLE,
        description="Hybridization of two complementary oligonucleotides to form a dsDNA product.",
    ),
    OperationCapability(
        operation_key="polymerase_extension",
        source_type="PolymeraseExtensionSource",
        endpoint="/polymerase_extension",
        cardinality=Cardinality.UNARY,
        visual_family=VisualFamily.OLIGO,
        status=CapabilityStatus.STABLE,
        description="Polymerase extension/fill-in of recessed 3' ends to generate blunt products.",
    ),
    OperationCapability(
        operation_key="reverse_complement",
        source_type="ReverseComplementSource",
        endpoint="/reverse_complement",
        cardinality=Cardinality.UNARY,
        visual_family=VisualFamily.OLIGO,
        status=CapabilityStatus.STABLE,
        description="Reverse complement of a sequence.",
    ),
    # --- Native batch family (experimental) ------------------------------
    # Verified against manulera/opencloning:v1.3.1-baseurl-opencloning.
    # Only three /batch_cloning/* POST endpoints exist on the pinned image:
    #   /batch_cloning/domesticate       (JSON → BaseCloningStrategy)
    #   /batch_cloning/pombe             (multipart form → ZIP file)
    #   /batch_cloning/ziqiang_et_al2024 (JSON → BaseCloningStrategy)
    # The previously-cataloged gibson/restriction_ligation/homologous_recombination
    # batch endpoints do not exist on this backend version (405).
    OperationCapability(
        operation_key="batch_domestication",
        source_type="BatchDomesticationSource",
        endpoint="/batch_cloning/domesticate",
        cardinality=Cardinality.BATCH_COUPLED,
        visual_family=VisualFamily.BATCH,
        status=CapabilityStatus.EXPERIMENTAL,
        description=(
            "Native batch domestication via OpenCloning. Removes internal "
            "restriction sites and adds GoldenBraid compatibility overhangs. "
            "Depends on external GoldenBraid service (goldenbraidpro.com)."
        ),
    ),
    OperationCapability(
        operation_key="batch_pombe",
        source_type="BatchPombeSource",
        endpoint="/batch_cloning/pombe",
        cardinality=Cardinality.BATCH_COUPLED,
        visual_family=VisualFamily.BATCH,
        status=CapabilityStatus.EXPERIMENTAL,
        description=(
            "S. pombe gene tagging workflow via OpenCloning. Takes a gene list "
            "and plasmid, returns a ZIP archive of cloning results. "
            "Depends on external NCBI and Addgene services. "
            "Multipart form input, ZIP output — incompatible with JSON DAG execution."
        ),
    ),
    OperationCapability(
        operation_key="batch_ziqiang",
        source_type="BatchZiqiangSource",
        endpoint="/batch_cloning/ziqiang_et_al2024",
        cardinality=Cardinality.BATCH_COUPLED,
        visual_family=VisualFamily.BATCH,
        status=CapabilityStatus.EXPERIMENTAL,
        description=(
            "Ziqiang et al. 2024 CRISPR multiplex cloning workflow. Takes a list "
            "of protospacers, internally runs PCR → Golden Gate → Gateway BP/LR. "
            "No external service dependency — uses a bundled template."
        ),
    ),
)


# --- Catalog API -------------------------------------------------------------


class StrategyCatalog:
    """Immutable catalog of supported OpenCloning operations.

    Use ``get_catalog()`` to obtain the singleton instance.
    """

    def __init__(self, entries: tuple[OperationCapability, ...]) -> None:
        # Validate uniqueness of operation_key
        keys = [e.operation_key for e in entries]
        duplicates = {k for k in keys if keys.count(k) > 1}
        if duplicates:
            raise ValueError(f"duplicate operation_key(s): {sorted(duplicates)}")

        self._entries: tuple[OperationCapability, ...] = entries
        self._by_key: dict[str, OperationCapability] = {
            e.operation_key: e for e in entries
        }
        # Multiple operation keys can share a source type (e.g. Golden Gate
        # and restriction/ligation both use RestrictionAndLigationSource).
        # For endpoint resolution, we map source_type → endpoint (unique per
        # source type since the backend routes by source type, not operation key).
        self._endpoint_by_source_type: dict[str, str] = {}
        self._by_source_type: dict[str, list[OperationCapability]] = {}
        for e in entries:
            self._by_source_type.setdefault(e.source_type, []).append(e)
            # All entries with the same source_type must agree on endpoint
            if e.source_type in self._endpoint_by_source_type:
                if self._endpoint_by_source_type[e.source_type] != e.endpoint:
                    raise ValueError(
                        f"source_type {e.source_type!r} has conflicting endpoints: "
                        f"{self._endpoint_by_source_type[e.source_type]!r} vs {e.endpoint!r}"
                    )
            else:
                self._endpoint_by_source_type[e.source_type] = e.endpoint

    def get_by_operation_key(self, key: str) -> OperationCapability | None:
        """Look up a capability by its canonical operation key."""
        return self._by_key.get(key)

    def get_by_source_type(self, source_type: str) -> list[OperationCapability]:
        """Look up capabilities by OpenCloning source type.

        Returns an empty list for unknown source types — callers must
        handle this as an explicit error (no Gibson fallback).

        Multiple operation keys can share a source type (e.g. Golden Gate
        and restriction/ligation both use RestrictionAndLigationSource).
        """
        return self._by_source_type.get(source_type, [])

    def list_all(self) -> tuple[OperationCapability, ...]:
        """Return all cataloged capabilities."""
        return self._entries

    def list_stable(self) -> tuple[OperationCapability, ...]:
        """Return only stable (non-experimental) capabilities."""
        return tuple(e for e in self._entries if e.status == CapabilityStatus.STABLE)

    def list_experimental(self) -> tuple[OperationCapability, ...]:
        """Return only experimental capabilities."""
        return tuple(
            e for e in self._entries if e.status == CapabilityStatus.EXPERIMENTAL
        )

    def resolve_endpoint(self, source_type: str) -> str:
        """Resolve the OpenCloning endpoint for a source type.

        Raises ``UnknownSourceTypeError`` for unrecognized source types.
        This replaces the previous Gibson fallback in
        ``HttpOpenCloningClient.simulate_assembly``.
        """
        endpoint = self._endpoint_by_source_type.get(source_type)
        if endpoint is None:
            raise UnknownSourceTypeError(source_type)
        return endpoint

    def list_known_source_types(self) -> list[str]:
        """Return all known source types, sorted."""
        return sorted(self._endpoint_by_source_type.keys())

    def to_public_dict(self, include_experimental: bool = False) -> dict[str, Any]:
        """Serialize the catalog for ``GET /strategy/capabilities``.

        No nucleotide sequences or internal state — only operation keys,
        endpoints, cardinality, visual family, and status.
        """
        entries = self._entries if include_experimental else self.list_stable()
        return {
            "operations": [
                {
                    "operation_key": e.operation_key,
                    "source_type": e.source_type,
                    "endpoint": e.endpoint,
                    "cardinality": e.cardinality.value,
                    "visual_family": e.visual_family.value,
                    "status": e.status.value,
                    "description": e.description,
                    "aliases": list(e.aliases),
                }
                for e in entries
            ],
            "operation_count": len(entries),
        }


class UnknownSourceTypeError(Exception):
    """Raised when a source type is not in the catalog.

    Replaces the previous Gibson fallback behavior.
    """

    def __init__(self, source_type: str) -> None:
        super().__init__(
            f"unknown OpenCloning source type: {source_type!r}. "
            f"Known types: {get_catalog().list_known_source_types()}"
        )
        self.source_type = source_type


# --- Singleton ---------------------------------------------------------------

_catalog: StrategyCatalog | None = None


def get_catalog() -> StrategyCatalog:
    """Return the singleton catalog instance."""
    global _catalog
    if _catalog is None:
        _catalog = StrategyCatalog(_CATALOG_ENTRIES)
    return _catalog


def reset_catalog(catalog: StrategyCatalog | None = None) -> None:
    """Reset the singleton (test helper)."""
    global _catalog
    _catalog = catalog


# --- Pinned backend version --------------------------------------------------

PINNED_OPENCLONING_IMAGE = "manulera/opencloning:v1.3.1-baseurl-opencloning"
"""The Docker image tag this catalog is validated against.

The real-container contract test (``make test-opencloning-contract``)
checks that the running backend reports a compatible version.
"""
