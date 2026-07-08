"""Tests for the curated tool registry (C06).

Acceptance checks:

    1. Every declared tool has all required fields (name, tier, adapter,
       requires_approval, mutability, description).
    2. Registry rejects entries exposing raw endpoint URLs as callable strings
       (any forbidden attribute, including via Tool subclasses or dynamic
       assignment, fails closed at registry construction).
    3. GET /tools returns the curated list — see test_app.py instead.

Additional coverage:

    * Name uniqueness inside the registry.
    * Mutating tools always require approval.
    * Adapter must be from the allowed set.
    * Mutability must be one of read/append/mutate.
    * Tool names are lowercase dotted identifiers (kill-switch fnmatch hygiene).
    * ToolRegistry.find returns declared tools and None for unknown names.
    * Round-trip via to_list / to_dict preserves all fields.
    * Singleton get_tool_registry / reset_tool_registry works.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from lab_copilot_gateway import tools as toolsmod
from lab_copilot_gateway.policy import Tier
from lab_copilot_gateway.tools import (
    Tool,
    ToolRegistry,
    get_tool_registry,
    list_tools,
    reset_tool_registry,
)


# --- Catalog-level acceptance checks -----------------------------------------


def test_catalog_has_all_13_c06_tools() -> None:
    """C06 plan lists 13 initial tools (now 20 with C15, C24, C35, C36, C43); the catalog must declare exactly these."""
    catalog = list_tools()
    names = {entry["name"] for entry in catalog}
    expected = {
        "elabftw.read_current_experiment",
        "elabftw.search_my_experiments",
        "elabftw.read_experiment_by_id",
        "elabftw.search_items",
        "elabftw.read_item_by_id",
        "elabftw.download_upload",
        "elabftw.draft_experiment_update",
        "elabftw.amend_my_experiment_after_approval",
        "elabftw.edit_experiment_section",
        "opencloning.parse_sequence_file",
        "opencloning.manual_sequence",
        "opencloning.oligo_hybridization",
        "opencloning.simulate_assembly",
        "opencloning.writeback_artifact",
        "opencloning.call",
        "opencloning.search_parts",
        "opencloning.fetch_igem_part",
        "opencloning.lookup_protocol",
        "protocols.validate_corpus",
        "wallac.get_status",
        "wallac.call",
        "wallac.run",
        "wallac.bridge_status",
        "wallac.propose_generated_protocol",
        "wallac.validate_generated_protocol",
        "wallac.prepare_submission_package",
        "wallac.submit_generated_protocol",
        "bentolab.get_status",
        "bentolab.validate_pcr_profile",
        "bentolab.dry_run_pcr_profile",
        "bentolab.submit_pcr_run",
    }
    assert len(catalog) == 31  # noqa: PLR2004 — V1 catalog size is a contract
    assert names == expected


# --- Acceptance check 1: every declared tool has all required fields ----------


REQUIRED_FIELDS = {
    "name",
    "tier",
    "tier_name",
    "adapter",
    "requires_approval",
    "mutability",
    "description",
}


@pytest.mark.parametrize("entry", list_tools())
def test_each_tool_exposes_all_required_fields(entry: dict) -> None:
    assert REQUIRED_FIELDS.issubset(entry.keys())
    assert isinstance(entry["name"], str) and entry["name"]
    assert isinstance(entry["tier"], int)
    assert isinstance(entry["adapter"], str) and entry["adapter"]
    assert isinstance(entry["requires_approval"], bool)
    assert entry["mutability"] in {"read", "append", "mutate"}
    assert isinstance(entry["description"], str)


# --- Acceptance check 2: registry rejects raw-endpoint fields ----------------


def _tool(name: str = "elabftw.read_current_experiment", **overrides) -> Tool:
    base = dict(
        name=name,
        tier=Tier.PERMISSIONED_ELABFTW_READ,
        adapter="elabftw",
        requires_approval=False,
        mutability="read",
        description="test tool",
    )
    base.update(overrides)
    return Tool(**base)


def test_tool_constructor_rejects_dynamically_assigned_url_field() -> None:
    """A Tool cannot have a url/endpoint/raw_endpoint attribute even if assigned
    after construction (defence-in-depth for subclasses or dynamic assignment)."""
    t = _tool()
    # Reason: forbidden field set via setattr must still be rejected when we
    # re-validate at registry construction time.
    object.__setattr__(t, "url", "https://elabftw.example.org/api/v2/experiments/1")
    with pytest.raises(ValueError, match="forbidden raw-endpoint field"):
        ToolRegistry(tools=(t,))


def test_registry_rejects_subclass_exposing_raw_endpoint() -> None:
    """A Tool subclass that adds an ``endpoint_url`` field fails closed at
    construction — the constructor-level guard runs before any registry sees
    the instance."""
    # Reason: use exec to define the subclass without instantiating it at
    # module scope (frozen dataclass field with default "" still triggers
    # __post_init__ when endpoint_url is supplied to __init__).

    @dataclass(frozen=True)
    class _LeakyTool(Tool):
        endpoint_url: str = ""

    with pytest.raises(ValueError, match="forbidden raw-endpoint field"):
        _LeakyTool(
            name="elabftw.read_current_experiment",
            tier=Tier.PERMISSIONED_ELABFTW_READ,
            adapter="elabftw",
            requires_approval=False,
            mutability="read",
            description="test leaky subclass",
            endpoint_url="https://elabftw.example.org/api/v2/experiments/1",
        )


def test_tool_to_dict_omits_any_raw_endpoint_keys() -> None:
    """Acceptance check: the dict returned to LibreChat has no URL-like field."""
    entry = _tool().to_dict()
    forbidden = {
        "url",
        "endpoint",
        "endpoint_url",
        "raw_endpoint",
        "raw_url",
        "http_method",
        "http_path",
    }
    assert not (forbidden & set(entry.keys()))


def test_tool_constructor_validates_required_fields() -> None:
    """Direct construction must validate each field of the contract."""
    with pytest.raises(ValueError, match="name must be non-empty"):
        Tool(
            name="",
            tier=Tier.OPERATIONAL_READ_ONLY,
            adapter="elabftw",
            requires_approval=False,
            mutability="read",
        )
    with pytest.raises(ValueError, match="whitespace"):
        Tool(
            name="elabftw.read current",
            tier=Tier.OPERATIONAL_READ_ONLY,
            adapter="elabftw",
            requires_approval=False,
            mutability="read",
        )
    with pytest.raises(ValueError, match="lowercase"):
        Tool(
            name="ElabFTW.read_current_experiment",
            tier=Tier.OPERATIONAL_READ_ONLY,
            adapter="elabftw",
            requires_approval=False,
            mutability="read",
        )


def test_tool_constructor_validates_adapter_allowlist() -> None:
    with pytest.raises(ValueError, match="adapter.*not allowed"):
        Tool(
            name="custom.read_thing",
            tier=Tier.OPERATIONAL_READ_ONLY,
            adapter="my_custom_service",
            requires_approval=False,
            mutability="read",
        )


def test_tool_constructor_validates_mutability_allowlist() -> None:
    with pytest.raises(ValueError, match="mutability.*not allowed"):
        Tool(
            name="elabftw.read_thing",
            tier=Tier.OPERATIONAL_READ_ONLY,
            adapter="elabftw",
            requires_approval=False,
            mutability="delete",
        )


def test_mutating_tool_requires_approval() -> None:
    """Acceptance check: mutating tools always require approval in V1."""
    with pytest.raises(ValueError, match=r"mutability='append' requires approval"):
        Tool(
            name="elabftw.amend_thing",
            tier=Tier.BOUNDED_WRITES,
            adapter="elabftw",
            requires_approval=False,
            mutability="append",
        )
    with pytest.raises(ValueError, match=r"mutability='mutate' requires approval"):
        Tool(
            name="elabftw.mutate_thing",
            tier=Tier.BOUNDED_WRITES,
            adapter="elabftw",
            requires_approval=False,
            mutability="mutate",
        )


def test_tool_constructor_validates_tier_type() -> None:
    with pytest.raises(ValueError, match="tier must be a policy.Tier"):
        Tool(
            name="elabftw.read_thing",
            tier=99,  # type: ignore[arg-type]
            adapter="elabftw",
            requires_approval=False,
            mutability="read",
        )


# --- Registry-level invariants -----------------------------------------------


def test_registry_rejects_duplicate_names() -> None:
    t1 = _tool(name="elabftw.read_current_experiment")
    t2 = _tool(name="elabftw.read_current_experiment")
    with pytest.raises(ValueError, match="duplicate tool name"):
        ToolRegistry(tools=(t1, t2))


def test_registry_rejects_non_tool_entries() -> None:
    with pytest.raises(TypeError, match="must be Tool instances"):
        ToolRegistry(tools=("not-a-tool",))  # type: ignore[arg-type]


def test_registry_from_iterable_preserves_order() -> None:
    t1 = _tool(name="elabftw.read_current_experiment")
    t2 = _tool(
        name="elabftw.draft_experiment_update",
        tier=Tier.BOUNDED_WRITES,
        requires_approval=True,
        mutability="append",
    )
    reg = ToolRegistry.from_iterable([t1, t2])
    assert [t.name for t in reg.list()] == [
        "elabftw.read_current_experiment",
        "elabftw.draft_experiment_update",
    ]


def test_registry_find_returns_named_tool() -> None:
    reg = get_tool_registry()
    found = reg.find("elabftw.read_current_experiment")
    assert isinstance(found, Tool)
    assert found.adapter == "elabftw"


def test_registry_find_returns_none_for_unknown_name() -> None:
    reg = get_tool_registry()
    assert reg.find("elabftw.bogus_tool") is None


def test_registry_to_list_round_trips_all_fields() -> None:
    """to_list/to_dict preserves all required fields and the tier mapping."""
    reg = ToolRegistry.from_iterable(
        [
            _tool(name="elabftw.read_current_experiment"),
            _tool(
                name="elabftw.amend_my_experiment_after_approval",
                tier=Tier.BOUNDED_WRITES,
                requires_approval=True,
                mutability="append",
                description="append amendment",
            ),
        ]
    )
    out = reg.to_list()
    assert len(out) == 2  # noqa: PLR2004 — explicit pair
    amended = next(
        e for e in out if e["name"] == "elabftw.amend_my_experiment_after_approval"
    )
    assert amended["tier"] == int(Tier.BOUNDED_WRITES)
    assert amended["tier_name"] == "bounded_writes"
    assert amended["requires_approval"] is True
    assert amended["mutability"] == "append"
    assert amended["description"] == "append amendment"


# --- Catalog content checks -------------------------------------------------


@pytest.mark.parametrize(
    "tool_name, expected_tier, expected_adapter, expected_approval, expected_mutability",
    [
        (
            "elabftw.read_current_experiment",
            Tier.PERMISSIONED_ELABFTW_READ,
            "elabftw",
            False,
            "read",
        ),
        (
            "elabftw.draft_experiment_update",
            Tier.BOUNDED_WRITES,
            "elabftw",
            True,
            "append",
        ),
        (
            "elabftw.amend_my_experiment_after_approval",
            Tier.BOUNDED_WRITES,
            "elabftw",
            True,
            "append",
        ),
        (
            "elabftw.edit_experiment_section",
            Tier.BOUNDED_WRITES,
            "elabftw",
            True,
            "mutate",
        ),
        (
            "opencloning.parse_sequence_file",
            Tier.VALIDATION_DRY_RUN,
            "opencloning",
            False,
            "read",
        ),
        (
            "opencloning.manual_sequence",
            Tier.VALIDATION_DRY_RUN,
            "opencloning",
            False,
            "read",
        ),
        (
            "opencloning.oligo_hybridization",
            Tier.VALIDATION_DRY_RUN,
            "opencloning",
            False,
            "read",
        ),
        (
            "opencloning.simulate_assembly",
            Tier.VALIDATION_DRY_RUN,
            "opencloning",
            False,
            "read",
        ),
        (
            "opencloning.writeback_artifact",
            Tier.BOUNDED_WRITES,
            "opencloning",
            True,
            "append",
        ),
        ("wallac.get_status", Tier.OPERATIONAL_READ_ONLY, "wallac", False, "read"),
        (
            "wallac.propose_generated_protocol",
            Tier.VALIDATION_DRY_RUN,
            "wallac",
            False,
            "read",
        ),
        (
            "wallac.validate_generated_protocol",
            Tier.VALIDATION_DRY_RUN,
            "wallac",
            False,
            "read",
        ),
        (
            "wallac.prepare_submission_package",
            Tier.VALIDATION_DRY_RUN,
            "wallac",
            False,
            "read",
        ),
        ("bentolab.get_status", Tier.OPERATIONAL_READ_ONLY, "bentolab", False, "read"),
        (
            "bentolab.validate_pcr_profile",
            Tier.VALIDATION_DRY_RUN,
            "bentolab",
            False,
            "read",
        ),
    ],
)
def test_catalog_entry_matches_plan_contract(
    tool_name: str,
    expected_tier: Tier,
    expected_adapter: str,
    expected_approval: bool,
    expected_mutability: str,
) -> None:
    """Every C06 tool matches the tier/adapter/approval/mutability contract."""
    reg = get_tool_registry()
    tool = reg.find(tool_name)
    assert tool is not None, f"missing tool in catalog: {tool_name}"
    assert tool.tier == expected_tier
    assert tool.adapter == expected_adapter
    assert tool.requires_approval is expected_approval
    assert tool.mutability == expected_mutability


def test_catalog_has_no_hardware_execution_tools() -> None:
    """Acceptance check: V1 policy blocks hardware execution; the catalog must
    not declare any tier-5 (HARDWARE_EXECUTION) or higher tool."""
    # wallac.submit_generated_protocol and bentolab.submit_pcr_run are v1.1
    # tools registered at HARDWARE_EXECUTION tier but blocked by the policy
    # engine in v1.
    for tool in get_tool_registry().list():
        if tool.name in (
            "wallac.submit_generated_protocol",
            "bentolab.submit_pcr_run",
        ):
            # v1.1 tools — registered but policy-blocked; skip the tier assertion
            continue
        assert tool.tier < Tier.HARDWARE_EXECUTION, (
            f"tool {tool.name!r} declares tier {tool.tier!r} which exceeds V1 limits"
        )
    # Verify the policy engine blocks tier 5 by default
    from lab_copilot_gateway.policy import PolicyEngine, PolicyRequest

    engine = PolicyEngine()
    req = PolicyRequest(
        tool_name="wallac.submit_generated_protocol",
        tier=Tier.HARDWARE_EXECUTION,
        adapter="wallac",
        user_id="test",
        team_id="test",
        has_approval=True,
        approval_id="test",
    )
    decision = engine.decide(req)
    assert decision.decision != "allow", (
        "wallac.submit_generated_protocol should be blocked by V1 policy"
    )
    assert "hardware_blocked" in decision.reason


# --- Singleton behaviour ----------------------------------------------------


def test_get_tool_registry_returns_singleton() -> None:
    reset_tool_registry()
    first = get_tool_registry()
    second = get_tool_registry()
    assert first is second


def test_reset_tool_registry_clears_singleton() -> None:
    reset_tool_registry()
    assert toolsmod._default_registry is None
    reg = get_tool_registry()
    assert toolsmod._default_registry is reg
    reset_tool_registry(registry=None)
    assert toolsmod._default_registry is None


def test_default_registry_is_the_v1_catalog() -> None:
    """The default registry is built from the curated _CATALOG tuple (13 tools)."""
    reset_tool_registry()
    reg = get_tool_registry()
    assert len(reg.list()) == 31  # noqa: PLR2004 — V1 catalog size is a contract
    reset_tool_registry()
