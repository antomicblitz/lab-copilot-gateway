"""Privacy-safe telemetry counters for strategy v3 (Slice 15 step 4).

Counts aggregate events — operation families, graph sizes, issue codes,
revisions, execution outcomes, and approval mismatches — without storing
sequences, user identities, or strategy content.

Use ``get_telemetry()`` to obtain the singleton. Call ``reset_telemetry()``
in tests.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StrategyTelemetry:
    """In-memory aggregate counters for strategy v3.

    All counters are privacy-safe: they store only category labels and
    integer counts. No sequence content, user IDs, or strategy payloads
    are retained.
    """

    operation_family: Counter = field(default_factory=Counter)
    """Count by operation_key (pcr, gibson_assembly, etc.)."""

    graph_size_bucket: Counter = field(default_factory=Counter)
    """Count by graph-size bucket (1-5, 6-10, 11-20, 21-50, 51+)."""

    issue_code: Counter = field(default_factory=Counter)
    """Count by validation issue code (duplicate_id, cycle_detected, etc.)."""

    revision: Counter = field(default_factory=Counter)
    """Count by revision number."""

    outcome: Counter = field(default_factory=Counter)
    """Count by execution outcome (completed, partial_failure, blocked, ambiguous)."""

    approval_mismatch: Counter = field(default_factory=Counter)
    """Count by approval mismatch type (not_found, mismatch, expired, already_consumed)."""

    prepare_count: int = 0
    """Total prepare() calls."""

    execute_count: int = 0
    """Total execute() calls."""

    # --- Recording methods ---------------------------------------------------

    def record_prepare(
        self,
        *,
        operation_keys: list[str],
        molecule_count: int,
        operation_count: int,
        revision: int,
        issue_codes: list[str],
    ) -> None:
        """Record counters from a strategy prepare() call."""
        self.prepare_count += 1
        for key in operation_keys:
            self.operation_family[key] += 1
        self.graph_size_bucket[_size_bucket(molecule_count + operation_count)] += 1
        self.revision[str(revision)] += 1
        for code in issue_codes:
            self.issue_code[code] += 1

    def record_execute(self, *, outcome: str) -> None:
        """Record counters from a strategy execute() call."""
        self.execute_count += 1
        self.outcome[outcome] += 1

    def record_approval_mismatch(self, *, mismatch_type: str) -> None:
        """Record an approval mismatch during run creation."""
        self.approval_mismatch[mismatch_type] += 1

    # --- Query methods -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return all counters as a JSON-serializable dict."""
        return {
            "prepare_count": self.prepare_count,
            "execute_count": self.execute_count,
            "operation_family": dict(self.operation_family),
            "graph_size_bucket": dict(self.graph_size_bucket),
            "issue_code": dict(self.issue_code),
            "revision": dict(self.revision),
            "outcome": dict(self.outcome),
            "approval_mismatch": dict(self.approval_mismatch),
        }

    def reset(self) -> None:
        """Clear all counters."""
        self.operation_family.clear()
        self.graph_size_bucket.clear()
        self.issue_code.clear()
        self.revision.clear()
        self.outcome.clear()
        self.approval_mismatch.clear()
        self.prepare_count = 0
        self.execute_count = 0


def _size_bucket(total: int) -> str:
    """Map a total node count to a histogram bucket label."""
    if total <= 5:
        return "1-5"
    if total <= 10:
        return "6-10"
    if total <= 20:
        return "11-20"
    if total <= 50:
        return "21-50"
    return "51+"


# --- Singleton ---------------------------------------------------------------

_telemetry: StrategyTelemetry | None = None


def get_telemetry() -> StrategyTelemetry:
    """Return the singleton StrategyTelemetry instance."""
    global _telemetry
    if _telemetry is None:
        _telemetry = StrategyTelemetry()
    return _telemetry


def reset_telemetry(telemetry: StrategyTelemetry | None = None) -> None:
    """Reset the singleton. Pass a mock for testing; pass None to reset to a fresh instance."""
    global _telemetry
    _telemetry = telemetry
