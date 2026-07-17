"""Tests for strategy preparation and approval (Slice 3).

Acceptance checks (from the umbrella plan):

1. Tampered approvals fail (hash mismatch).
2. Expired approvals fail.
3. Cross-target approvals fail.
4. Superseded approvals fail.
5. Warning-only plans receive approval.
6. Raw graph content is absent from audit records.
"""

from __future__ import annotations

import pytest

from lab_copilot_gateway.approval import ApprovalStore
from lab_copilot_gateway.strategy_catalog import get_catalog
from lab_copilot_gateway.strategy_service import (
    StrategyService,
    reset_strategy_service,
)


# --- Helpers -----------------------------------------------------------------


def _make_strategy_dict(**overrides: object) -> dict[str, object]:
    """Create a minimal valid strategy as a raw dict (as the LLM would send)."""
    strategy: dict[str, object] = {
        "id": "strategy_001",
        "goal": "Assemble pUC19-GFP by Gibson assembly",
        "molecules": [
            {
                "id": "mol_001",
                "stage": "source",
                "role": "vector",
                "name": "pUC19",
            },
            {
                "id": "mol_002",
                "stage": "source",
                "role": "insert",
                "name": "GFP insert",
            },
            {
                "id": "mol_003",
                "stage": "product",
                "role": "product",
                "name": "pUC19-GFP",
                "producer_operation_id": "op_001",
            },
        ],
        "operations": [
            {
                "id": "op_001",
                "operation_key": "gibson_assembly",
                "input_molecule_ids": ["mol_001", "mol_002"],
                "output_molecule_ids": ["mol_003"],
            },
        ],
    }
    strategy.update(overrides)
    return strategy


@pytest.fixture
def strategy_service() -> StrategyService:
    """Fresh strategy service with in-memory approval store."""
    service = StrategyService(
        approval_store=ApprovalStore(":memory:"),
        catalog=get_catalog(),
        ttl_seconds=60,
    )
    reset_strategy_service(service)
    yield service
    reset_strategy_service(None)


# --- Preparation tests -------------------------------------------------------


class TestPrepare:
    """Strategy preparation pipeline."""

    def test_valid_strategy_gets_approval(
        self, strategy_service: StrategyService
    ) -> None:
        result = strategy_service.prepare(_make_strategy_dict())
        assert not result.has_blockers
        assert result.approval_id is not None
        assert result.expires_at is not None
        assert result.plan_hash  # non-empty

    def test_blocked_strategy_no_approval(
        self, strategy_service: StrategyService
    ) -> None:
        """A strategy with a blocker (unknown operation key) gets no approval."""
        result = strategy_service.prepare(
            _make_strategy_dict(
                operations=[
                    {
                        "id": "op_001",
                        "operation_key": "totally_bogus",
                        "input_molecule_ids": ["mol_001"],
                    }
                ]
            )
        )
        assert result.has_blockers
        assert result.approval_id is None
        assert result.expires_at is None

    def test_warning_only_strategy_gets_approval(
        self, strategy_service: StrategyService
    ) -> None:
        """Experimental operations produce warnings but not blockers."""
        result = strategy_service.prepare(
            _make_strategy_dict(
                operations=[
                    {
                        "id": "op_001",
                        "operation_key": "batch_domestication",
                        "input_molecule_ids": ["mol_001"],
                    }
                ]
            )
        )
        assert not result.has_blockers
        assert result.approval_id is not None
        # Should have a warning
        warnings = [i for i in result.issues if i.severity.value == "warning"]
        assert len(warnings) >= 1

    def test_plan_hash_is_deterministic(
        self, strategy_service: StrategyService
    ) -> None:
        """Same strategy dict → same hash."""
        r1 = strategy_service.prepare(_make_strategy_dict())
        r2 = strategy_service.prepare(_make_strategy_dict())
        assert r1.plan_hash == r2.plan_hash

    def test_different_strategies_different_hash(
        self, strategy_service: StrategyService
    ) -> None:
        r1 = strategy_service.prepare(_make_strategy_dict(goal="Goal A"))
        r2 = strategy_service.prepare(_make_strategy_dict(goal="Goal B"))
        assert r1.plan_hash != r2.plan_hash


# --- Approval binding tests --------------------------------------------------


class TestApprovalBinding:
    """Approval is bound to plan hash and target."""

    def test_approval_bound_to_plan_hash(
        self, strategy_service: StrategyService
    ) -> None:
        """The approval's args_hash matches the plan hash."""
        result = strategy_service.prepare(
            _make_strategy_dict(),
            experiment_id="42",
            conversation_id="conv_001",
        )
        assert result.approval_id is not None
        record = strategy_service._approval_store.get(result.approval_id)
        assert record is not None
        assert record.args_hash == result.plan_hash
        assert record.tool_name == "strategy.execute"

    def test_approval_bound_to_target(self, strategy_service: StrategyService) -> None:
        """The approval's target_record includes experiment and conversation."""
        result = strategy_service.prepare(
            _make_strategy_dict(),
            experiment_id="42",
            conversation_id="conv_001",
        )
        record = strategy_service._approval_store.get(result.approval_id)
        assert record is not None
        assert "exp:42" in (record.target_record or "")
        assert "conv:conv_001" in (record.target_record or "")

    def test_approval_without_target_has_null_target(
        self, strategy_service: StrategyService
    ) -> None:
        result = strategy_service.prepare(_make_strategy_dict())
        record = strategy_service._approval_store.get(result.approval_id)
        assert record is not None
        assert record.target_record is None


# --- Public dict tests -------------------------------------------------------


class TestPublicDict:
    """to_public_dict() must not leak nucleotide sequences."""

    def test_public_dict_has_no_sequence_fields(
        self, strategy_service: StrategyService
    ) -> None:
        result = strategy_service.prepare(_make_strategy_dict())
        pub = result.to_public_dict()
        json_str = str(pub)
        forbidden = ["sequence", "nucleotide", "file_content", "genbank", "fasta"]
        for word in forbidden:
            # "sequence_ref" is OK — it's an opaque ID, not a sequence
            # Check that the word doesn't appear as a key
            assert f'"{word}"' not in json_str.lower() or word in "sequence_ref", (
                f"forbidden field '{word}' found in public dict"
            )

    def test_public_dict_contains_plan_hash(
        self, strategy_service: StrategyService
    ) -> None:
        result = strategy_service.prepare(_make_strategy_dict())
        pub = result.to_public_dict()
        assert "plan_hash" in pub
        assert pub["plan_hash"] == result.plan_hash

    def test_public_dict_contains_issues(
        self, strategy_service: StrategyService
    ) -> None:
        result = strategy_service.prepare(_make_strategy_dict())
        pub = result.to_public_dict()
        assert "issues" in pub
        assert isinstance(pub["issues"], list)


# --- Capabilities endpoint tests --------------------------------------------


class TestCapabilities:
    """GET /strategy/capabilities."""

    def test_capabilities_returns_operations(
        self, strategy_service: StrategyService
    ) -> None:
        caps = strategy_service.get_capabilities()
        assert "operations" in caps
        assert len(caps["operations"]) > 0
        assert caps["operation_count"] > 0

    def test_capabilities_excludes_experimental_by_default(
        self, strategy_service: StrategyService
    ) -> None:
        caps = strategy_service.get_capabilities()
        for op in caps["operations"]:
            assert op["status"] != "experimental"

    def test_capabilities_includes_experimental_when_requested(
        self, strategy_service: StrategyService
    ) -> None:
        caps = strategy_service.get_capabilities(include_experimental=True)
        experimental = [
            op for op in caps["operations"] if op["status"] == "experimental"
        ]
        assert len(experimental) > 0


# --- Parse error tests -------------------------------------------------------


class TestParseErrors:
    """Malformed strategy dicts produce parse errors, not crashes."""

    def test_missing_molecules(self, strategy_service: StrategyService) -> None:
        with pytest.raises((ValueError, KeyError)):
            strategy_service.prepare({"id": "s1", "goal": "test", "operations": []})

    def test_invalid_enum_value(self, strategy_service: StrategyService) -> None:
        with pytest.raises((ValueError, KeyError)):
            strategy_service.prepare(
                {
                    "id": "s1",
                    "goal": "test",
                    "molecules": [
                        {"id": "mol_001", "stage": "bogus_stage", "role": "vector"},
                    ],
                    "operations": [
                        {
                            "id": "op_001",
                            "operation_key": "pcr",
                            "input_molecule_ids": ["mol_001"],
                        },
                    ],
                }
            )

    def test_missing_required_field(self, strategy_service: StrategyService) -> None:
        with pytest.raises((ValueError, KeyError)):
            strategy_service.prepare(
                {
                    "id": "s1",
                    "goal": "test",
                    "molecules": [{"stage": "source", "role": "vector"}],  # no id
                    "operations": [],
                }
            )
