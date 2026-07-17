"""Tests for strategy telemetry counters (Slice 15 step 4)."""

from __future__ import annotations

from lab_copilot_gateway.strategy_telemetry import (
    StrategyTelemetry,
    get_telemetry,
    reset_telemetry,
)


def test_record_prepare_increments_counters():
    t = StrategyTelemetry()
    t.record_prepare(
        operation_keys=["pcr", "gibson_assembly"],
        molecule_count=3,
        operation_count=2,
        revision=1,
        issue_codes=["duplicate_id"],
    )
    assert t.prepare_count == 1
    assert t.operation_family["pcr"] == 1
    assert t.operation_family["gibson_assembly"] == 1
    assert t.graph_size_bucket["1-5"] == 1
    assert t.revision["1"] == 1
    assert t.issue_code["duplicate_id"] == 1


def test_record_execute_increments_outcome():
    t = StrategyTelemetry()
    t.record_execute(outcome="completed")
    t.record_execute(outcome="completed")
    t.record_execute(outcome="blocked")
    assert t.execute_count == 3
    assert t.outcome["completed"] == 2
    assert t.outcome["blocked"] == 1


def test_record_approval_mismatch():
    t = StrategyTelemetry()
    t.record_approval_mismatch(mismatch_type="mismatch")
    t.record_approval_mismatch(mismatch_type="expired")
    assert t.approval_mismatch["mismatch"] == 1
    assert t.approval_mismatch["expired"] == 1


def test_to_dict_returns_all_counters():
    t = StrategyTelemetry()
    t.record_prepare(
        operation_keys=["pcr"],
        molecule_count=1,
        operation_count=1,
        revision=1,
        issue_codes=[],
    )
    t.record_execute(outcome="completed")
    d = t.to_dict()
    assert d["prepare_count"] == 1
    assert d["execute_count"] == 1
    assert d["operation_family"]["pcr"] == 1
    assert d["outcome"]["completed"] == 1
    assert "graph_size_bucket" in d
    assert "issue_code" in d
    assert "revision" in d
    assert "approval_mismatch" in d


def test_reset_clears_all_counters():
    t = StrategyTelemetry()
    t.record_prepare(
        operation_keys=["pcr"],
        molecule_count=1,
        operation_count=1,
        revision=1,
        issue_codes=["cycle_detected"],
    )
    t.record_execute(outcome="completed")
    t.reset()
    assert t.prepare_count == 0
    assert t.execute_count == 0
    assert len(t.operation_family) == 0
    assert len(t.outcome) == 0


def test_size_buckets():
    t = StrategyTelemetry()
    t.record_prepare(
        operation_keys=["pcr"],
        molecule_count=2,
        operation_count=1,
        revision=1,
        issue_codes=[],
    )
    t.record_prepare(
        operation_keys=["pcr"],
        molecule_count=5,
        operation_count=5,
        revision=1,
        issue_codes=[],
    )
    t.record_prepare(
        operation_keys=["pcr"],
        molecule_count=10,
        operation_count=10,
        revision=1,
        issue_codes=[],
    )
    t.record_prepare(
        operation_keys=["pcr"],
        molecule_count=30,
        operation_count=20,
        revision=1,
        issue_codes=[],
    )
    t.record_prepare(
        operation_keys=["pcr"],
        molecule_count=25,
        operation_count=15,
        revision=1,
        issue_codes=[],
    )
    assert t.graph_size_bucket["1-5"] == 1
    assert t.graph_size_bucket["6-10"] == 1
    assert t.graph_size_bucket["11-20"] == 1
    assert t.graph_size_bucket["21-50"] == 2
    assert t.graph_size_bucket["51+"] == 0


def test_singleton_get_and_reset():
    reset_telemetry()
    t1 = get_telemetry()
    t2 = get_telemetry()
    assert t1 is t2
    mock = StrategyTelemetry()
    reset_telemetry(mock)
    assert get_telemetry() is mock
    reset_telemetry()
