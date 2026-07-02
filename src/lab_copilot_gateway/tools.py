"""Curated lab tool registry exposed to LibreChat (C06).

The registry is the single source of truth for which lab actions the copilot
may perform on behalf of a user.  Tools are high-level and bounded — raw
privileged API endpoints of eLabFTW, OpenCloning, Wallac, or BentoLab are
never exposed directly.

Each declared tool carries the fields the policy engine, audit store, and
identity mapper need to evaluate and record a call:

    name               : stable identifier used by the policy engine kill
                         switches (exact or fnmatch patterns).  Names are
                         dotted lowercase, e.g. ``elabftw.read_current_experiment``.
    tier               : action tier from the policy engine's ``Tier`` enum.
    adapter            : which downstream service the tool eventually calls
                         (``elabftw`` | ``opencloning`` | ``wallac`` |
                         ``bentolab`` | ``help``).
    requires_approval  : whether the tool requires a single-use approval token
                         before execution.  Mutating tools always set this True
                         in V1; read-only tools set it False.
    mutability         : one of ``read`` / ``append`` / ``mutate``.  ``read``
                         makes no downstream change.  ``append`` only adds content
                         or attachments.  ``mutate`` rewrites existing content
                         (blocked in V1; only present so the registry can
                         describe tools we will support later).
    description        : short human-readable summary, surfaced to LibreChat in
                         ``GET /tools``.

The registry construction itself enforces the invariants — any tool missing a
required field or carrying a forbidden raw-endpoint hint raises ValueError at
import time, so a misconfigured registry fails fast instead of silently
exposing dangerous tools.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from lab_copilot_gateway.policy import Tier


# Adapters that a declared tool may target.  Keeping the set explicit makes a
# typo in a tool entry fail closed.
_ALLOWED_ADAPTERS: frozenset[str] = frozenset(
    {"elabftw", "opencloning", "wallac", "bentolab", "help"}
)

# Mutability classes.  ``mutate`` rewrites existing content and is reserved for
# future approval-gated flows; V1 only allows ``read`` and ``append`` to be
# declared for tools that ship.
_ALLOWED_MUTABILITY: frozenset[str] = frozenset({"read", "append", "mutate"})

# Field names that would indicate a raw low-level endpoint surfaced to the LLM.
# Tools must be higher-level than "call this URL"; any entry attempting to set
# one of these attributes is rejected.
_FORBIDDEN_RAW_ENDPOINT_FIELDS: frozenset[str] = frozenset(
    {
        "url",
        "endpoint",
        "endpoint_url",
        "raw_endpoint",
        "raw_url",
        "http_method",
        "http_path",
    }
)


@dataclass(frozen=True)
class Tool:
    """A curated lab tool exposed to LibreChat through the gateway.

    Invariants validated in ``__post_init__``:

        * ``name`` is non-empty dotted lowercase, no whitespace.
        * ``adapter`` is in the allowed set.
        * ``tier`` is a valid policy ``Tier``.
        * ``mutability`` is ``read`` | ``append`` | ``mutate``.
        * Mutating tools always ``requires_approval`` in V1.
        * No forbidden raw-endpoint attribute is set on the instance.
    """

    name: str
    tier: Tier
    adapter: str
    requires_approval: bool
    mutability: str
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise ValueError(f"tool name must be non-empty and trimmed: {self.name!r}")
        if any(ch.isspace() for ch in self.name):
            raise ValueError(f"tool name must not contain whitespace: {self.name!r}")
        # Reason: reject uppercase so kill-switch fnmatch patterns behave
        # deterministically across the registry, policy engine, and audit log.
        if self.name != self.name.lower():
            raise ValueError(f"tool name must be lowercase: {self.name!r}")
        if self.adapter not in _ALLOWED_ADAPTERS:
            raise ValueError(
                f"tool {self.name!r}: adapter {self.adapter!r} not allowed "
                f"(allowed: {sorted(_ALLOWED_ADAPTERS)})"
            )
        if not isinstance(self.tier, Tier):
            raise ValueError(
                f"tool {self.name!r}: tier must be a policy.Tier, got {type(self.tier).__name__}"
            )
        if self.mutability not in _ALLOWED_MUTABILITY:
            raise ValueError(
                f"tool {self.name!r}: mutability {self.mutability!r} not allowed "
                f"(allowed: {sorted(_ALLOWED_MUTABILITY)})"
            )
        # Reason: in V1 any write-side action (append or mutate) must require
        # explicit single-use approval.  Read-only tools may still require
        # approval in future (e.g. for permissioned reads), but the reverse
        # direction is always enforced here.
        if self.mutability in {"append", "mutate"} and not self.requires_approval:
            raise ValueError(
                f"tool {self.name!r}: mutability={self.mutability!r} requires approval"
            )
        # Reason: reject raw-endpoint fields.  ``object.__setattr__`` is the
        # way to introspect a frozen dataclass after construction; if any
        # forbidden attribute leaked in (e.g. via subclassing or dynamic
        # assignment), fail closed.
        for forbidden in _FORBIDDEN_RAW_ENDPOINT_FIELDS:
            if forbidden in self.__dict__:
                raise ValueError(
                    f"tool {self.name!r}: forbidden raw-endpoint field {forbidden!r}"
                )

    def to_dict(self) -> dict[str, object]:
        """Serialise to the JSON-compatible shape returned by ``GET /tools``.

        Mutability, tier, adapter, and approval requirement are always present
        so policy-engine and audit callers can rely on the schema.  No URL,
        HTTP method, or downstream path is ever emitted.
        """
        return {
            "name": self.name,
            "tier": int(self.tier),
            "tier_name": self.tier.name.lower(),
            "adapter": self.adapter,
            "requires_approval": self.requires_approval,
            "mutability": self.mutability,
            "description": self.description,
        }


@dataclass
class ToolRegistry:
    """Ordered, name-unique collection of curated tools.

    Construction validates:

        * Every entry is a ``Tool`` (dataclass invariants also run).
        * Names are unique within the registry.
        * No tool exposes a raw-endpoint field.
    """

    tools: tuple[Tool, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for tool in self.tools:
            if not isinstance(tool, Tool):
                raise TypeError(
                    f"registry entries must be Tool instances, got {type(tool).__name__}"
                )
            # Re-run the raw-endpoint guard on the live instance — this is the
            # acceptance-check hook ("registry rejects entries exposing raw
            # endpoint URLs as callable strings").  Calling it at registry
            # construction means an attacker-supplied Tool subclass cannot slip
            # a url field past validation.
            for forbidden in _FORBIDDEN_RAW_ENDPOINT_FIELDS:
                if hasattr(tool, forbidden):
                    raise ValueError(
                        f"tool {tool.name!r}: forbidden raw-endpoint field {forbidden!r}"
                    )
            if tool.name in seen:
                raise ValueError(f"duplicate tool name in registry: {tool.name!r}")
            seen.add(tool.name)

    def list(self) -> list[Tool]:
        return list(self.tools)

    def find(self, name: str) -> Tool | None:
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def to_list(self) -> list[dict[str, object]]:
        return [tool.to_dict() for tool in self.tools]

    @classmethod
    def from_iterable(cls, tools: Iterable[Tool]) -> "ToolRegistry":
        return cls(tools=tuple(tools))


# --- Curated V1 tool catalog ---------------------------------------------
#
# These are the 13 tools called out in the C06 build plan.  Add or remove tools
# only by editing this tuple — the singleton below wraps it so callers see one
# consistent registry for the lifetime of the process.

_CATALOG: tuple[Tool, ...] = (
    # --- eLabFTW reads and writes ----------------------------------------
    Tool(
        name="elabftw.read_current_experiment",
        tier=Tier.PERMISSIONED_ELABFTW_READ,
        adapter="elabftw",
        requires_approval=False,
        mutability="read",
        description="Read the title, body, and metadata of the user's current experiment context.",
    ),
    Tool(
        name="elabftw.search_my_experiments",
        tier=Tier.PERMISSIONED_ELABFTW_READ,
        adapter="elabftw",
        requires_approval=False,
        mutability="read",
        description="Search the user's accessible experiments by free-text query. Returns compact summaries (id, title, dates) suitable for citation; use read_experiment_by_id for full content.",
    ),
    Tool(
        name="elabftw.read_experiment_by_id",
        tier=Tier.PERMISSIONED_ELABFTW_READ,
        adapter="elabftw",
        requires_approval=False,
        mutability="read",
        description="Read the full title, body, and metadata of a specific experiment by its numeric id. Per-record permissions enforced server-side by eLabFTW.",
    ),
    Tool(
        name="elabftw.draft_experiment_update",
        tier=Tier.BOUNDED_WRITES,
        adapter="elabftw",
        requires_approval=True,
        mutability="append",
        description="Draft a proposed amendment section for the user's experiment for review prior to append.",
    ),
    Tool(
        name="elabftw.amend_my_experiment_after_approval",
        tier=Tier.BOUNDED_WRITES,
        adapter="elabftw",
        requires_approval=True,
        mutability="append",
        description="Append an approved AI-generated amendment section and provenance to the user's experiment.",
    ),
    Tool(
        name="elabftw.edit_experiment_section",
        tier=Tier.BOUNDED_WRITES,
        adapter="elabftw",
        requires_approval=True,
        mutability="mutate",
        description="Replace the body of the user's experiment with approved content (direct edit). Rollback via eLabFTW revision history; approval binds old_body_hash + new_body to prevent stale-edit clobbering.",
    ),
    # --- OpenCloning computational design (no hardware) ------------------
    Tool(
        name="opencloning.parse_sequence_file",
        tier=Tier.VALIDATION_DRY_RUN,
        adapter="opencloning",
        requires_approval=False,
        mutability="read",
        description="Parse an allowed sequence file (FASTA/GenBank/SnapGene) into a structured sequence description.",
    ),
    Tool(
        name="opencloning.manual_sequence",
        tier=Tier.VALIDATION_DRY_RUN,
        adapter="opencloning",
        requires_approval=False,
        mutability="read",
        description="Validate and describe a manually typed or pasted DNA sequence.",
    ),
    Tool(
        name="opencloning.oligo_hybridization",
        tier=Tier.VALIDATION_DRY_RUN,
        adapter="opencloning",
        requires_approval=False,
        mutability="read",
        description="Compute the product of oligo hybridization from provided primer/oligo sequences.",
    ),
    Tool(
        name="opencloning.simulate_assembly",
        tier=Tier.VALIDATION_DRY_RUN,
        adapter="opencloning",
        requires_approval=False,
        mutability="read",
        description="Simulate a cloning assembly from input fragments and return the predicted construct.",
    ),
    Tool(
        name="opencloning.writeback_artifact",
        tier=Tier.BOUNDED_WRITES,
        adapter="opencloning",
        requires_approval=True,
        mutability="append",
        description="Attach an approved OpenCloning design artifact (GenBank/FASTA) to the user's experiment with provenance.",
    ),
    # --- Wallac status, proposal, validation (no execution in V1) --------
    Tool(
        name="wallac.get_status",
        tier=Tier.OPERATIONAL_READ_ONLY,
        adapter="wallac",
        requires_approval=False,
        mutability="read",
        description="Read Wallac Victor2 service and current job status.",
    ),
    Tool(
        name="wallac.call",
        tier=Tier.VALIDATION_DRY_RUN,
        adapter="wallac",
        requires_approval=False,
        mutability="read",
        description=(
            "Call any Wallac Victor2 vm-agent API endpoint. Covers: "
            "GET /health, /status, /instrument, /protocols, /runs/{id}, "
            "/runs/{id}/results, /jobs, /jobs/{id}, /jobs/{id}/results, "
            "/jobs/{id}/export. "
            "POST /runs (forces dry_run=true), /runs/{id}/abort, "
            "/admin/reconnect. "
            "PATCH /mdb/protocols/{id}/plate_map (set which wells to "
            "measure — body: {\"plate_map\": [108 ints]}). "
            "Args: method (GET, POST, or PATCH), endpoint (e.g. "
            "'/protocols'), body (dict, for POST/PATCH)."
        ),
    ),
    Tool(
        name="wallac.propose_generated_protocol",
        tier=Tier.VALIDATION_DRY_RUN,
        adapter="wallac",
        requires_approval=False,
        mutability="read",
        description="Propose a generated-protocol package (method/layout/analysis/job) without execution.",
    ),
    Tool(
        name="wallac.validate_generated_protocol",
        tier=Tier.VALIDATION_DRY_RUN,
        adapter="wallac",
        requires_approval=False,
        mutability="read",
        description="Validate a generated-protocol package against the signed-spec schema.",
    ),
    Tool(
        name="wallac.prepare_submission_package",
        tier=Tier.VALIDATION_DRY_RUN,
        adapter="wallac",
        requires_approval=False,
        mutability="read",
        description="Prepare an approval-ready Wallac submission package; execution remains blocked in v1.",
    ),
    # --- Wallac hardware execution (approval-gated, NOT v1-blocked) -----
    Tool(
        name="wallac.run",
        tier=Tier.BOUNDED_WRITES,
        adapter="wallac",
        requires_approval=True,
        mutability="mutate",
        description=(
            "Start a REAL measurement run on the Wallac Victor2. "
            "Calling this tool automatically triggers an approval card "
            "in the UI — do NOT ask the user for confirmation first, "
            "just call it and the approval card appears. "
            "Args: protocol_id (int, from GET /protocols), plate_id (int, optional)."
        ),
    ),
    # --- Wallac hardware execution (v1.1 — blocked by policy in v1) ------
    Tool(
        name="wallac.submit_generated_protocol",
        tier=Tier.HARDWARE_EXECUTION,
        adapter="wallac",
        requires_approval=True,
        mutability="mutate",
        description="Submit an approved Wallac generated protocol for hardware execution; requires explicit user approval (blocked in v1).",
    ),
    # --- BentoLab status and validation (no execution until wrapper ready) --
    Tool(
        name="bentolab.get_status",
        tier=Tier.OPERATIONAL_READ_ONLY,
        adapter="bentolab",
        requires_approval=False,
        mutability="read",
        description="Read BentoLab HTTP wrapper service and device status.",
    ),
    Tool(
        name="bentolab.validate_pcr_profile",
        tier=Tier.VALIDATION_DRY_RUN,
        adapter="bentolab",
        requires_approval=False,
        mutability="read",
        description="Validate a PCR temperature/cycle profile against the device contract without hardware side effects.",
    ),
    Tool(
        name="bentolab.dry_run_pcr_profile",
        tier=Tier.VALIDATION_DRY_RUN,
        adapter="bentolab",
        requires_approval=False,
        mutability="read",
        description="Simulate a PCR run on BentoLab without hardware side effects; returns step breakdown and timing.",
    ),
    Tool(
        name="bentolab.submit_pcr_run",
        tier=Tier.HARDWARE_EXECUTION,
        adapter="bentolab",
        requires_approval=True,
        mutability="mutate",
        description="Submit an approved PCR profile for hardware execution on BentoLab; requires explicit user approval (blocked in v1 by default).",
    ),
    # --- OpenCloning generic endpoint (covers all API operations) ---------
    Tool(
        name="opencloning.call",
        tier=Tier.VALIDATION_DRY_RUN,
        adapter="opencloning",
        requires_approval=False,
        mutability="read",
        description=(
            "Call any OpenCloning API endpoint. Covers repository imports "
            "(Addgene, GenBank, Benchling, SnapGene, Euroscarf, iGEM, SEVA), "
            "PCR, restriction digest, Golden Gate, CRISPR, homologous "
            "recombination, Cre/Lox, Gateway, primer design, validation, "
            "Sanger alignment, and more. "
            "Args: endpoint (e.g. '/repository_id/addgene'), body (request dict)."
        ),
    ),
    # --- NCBI sequence search (discover accessions by gene name) ----------
    Tool(
        name="opencloning.search_ncbi",
        tier=Tier.OPERATIONAL_READ_ONLY,
        adapter="opencloning",
        requires_approval=False,
        mutability="read",
        description=(
            "Search NCBI's nuccore database for sequences by gene name, "
            "organism, or keyword. Returns accession.version, title, "
            "organism, and length for each result. Use this to find the "
            "correct GenBank accession before importing via "
            "/repository_id/genbank. "
            "Args: query (str, e.g. 'nptII[Title]'), retmax (int, default 5)."
        ),
    ),
    # --- SnapGene plasmid search (discover plasmids by name) -------------
    Tool(
        name="opencloning.search_snapgene",
        tier=Tier.OPERATIONAL_READ_ONLY,
        adapter="opencloning",
        requires_approval=False,
        mutability="read",
        description=(
            "Search the SnapGene plasmid catalog by name. Returns "
            "repository_id (category/plasmid_name), name, and category. "
            "Use this to find cloning vectors, expression vectors, and "
            "other plasmids before importing via /repository_id/snapgene. "
            "Args: query (str, e.g. 'pUC19'), retmax (int, default 10)."
        ),
    ),
)


# --- module-level singleton for dependency injection --------------------
_default_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """Return the process-wide tool registry (created lazily from the catalog)."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry(tools=_CATALOG)
    return _default_registry


def reset_tool_registry(registry: ToolRegistry | None = None) -> None:
    """Test helper: replace or clear the singleton."""
    global _default_registry
    _default_registry = registry


def list_tools() -> list[dict[str, object]]:
    """Return the curated tool catalog as JSON-compatible dicts.

    Used by ``GET /tools``.  Always returns dataclasses' ``to_dict`` output —
    no URL, HTTP method, or downstream path is ever included.
    """
    return get_tool_registry().to_list()
