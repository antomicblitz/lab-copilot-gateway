"""Tests for short-lived OpenCloning artifact bundle custody (C51)."""

from __future__ import annotations

import hashlib

import pytest

from lab_copilot_gateway.artifact_bundle import (
    ArtifactBundleStore,
    ArtifactContextMismatch,
    ArtifactExpired,
    ArtifactHashMismatch,
    ArtifactManifestCollision,
    ArtifactMissing,
    ArtifactSizeMismatch,
    ArtifactTooLarge,
    get_artifact_bundle_store,
)


class FakeClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def test_put_returns_manifest_without_raw_bytes() -> None:
    """Stored bytes stay server-side; manifests carry hash/size only."""
    data = b"LOCUS test\n//\n"
    store = ArtifactBundleStore(clock=FakeClock())

    manifest = store.put(
        plan_id="plan-1",
        plan_hash="hash-1",
        artifact_id="oc-artifact-1",
        filename="construct.gb",
        mime_type="chemical/x-genbank",
        data=data,
        manifest_extra={"kind": "genbank", "writeback_eligible": True},
    )

    assert manifest["sha256"] == hashlib.sha256(data).hexdigest()
    assert manifest["size_bytes"] == len(data)
    assert manifest["kind"] == "genbank"
    assert manifest["writeback_eligible"] is True
    assert "bytes" not in manifest
    assert "bytes_data" not in manifest
    assert "data" not in manifest


def test_get_returns_bytes_when_plan_hash_and_context_match() -> None:
    """Artifact lookup is scoped by plan hash and binding metadata."""
    binding = {"record_type": "experiment", "record_id": "200"}
    store = ArtifactBundleStore(clock=FakeClock())
    manifest = store.put(
        plan_id="plan-1",
        plan_hash="hash-1",
        artifact_id="oc-artifact-1",
        filename="construct.gb",
        mime_type="chemical/x-genbank",
        data="LOCUS test\n//\n",
        binding=binding,
    )

    artifact = store.get(
        plan_id="plan-1",
        plan_hash="hash-1",
        artifact_id="oc-artifact-1",
        expected_sha256=manifest["sha256"],
        expected_size_bytes=manifest["size_bytes"],
        binding=binding,
    )

    assert artifact.bytes_data == b"LOCUS test\n//\n"


def test_wrong_plan_hash_fails_closed() -> None:
    """A stale plan hash cannot resolve an artifact from another bundle."""
    store = ArtifactBundleStore(clock=FakeClock())
    store.put(
        plan_id="plan-1",
        plan_hash="hash-1",
        artifact_id="oc-artifact-1",
        filename="construct.gb",
        mime_type="chemical/x-genbank",
        data=b"abc",
    )

    with pytest.raises(ArtifactMissing):
        store.get(
            plan_id="plan-1",
            plan_hash="hash-2",
            artifact_id="oc-artifact-1",
        )


def test_expired_bundle_fails_closed_and_is_removed() -> None:
    """Expired artifacts cannot be downloaded or written back."""
    clock = FakeClock()
    store = ArtifactBundleStore(ttl_seconds=10, clock=clock)
    store.put(
        plan_id="plan-1",
        plan_hash="hash-1",
        artifact_id="oc-artifact-1",
        filename="construct.gb",
        mime_type="chemical/x-genbank",
        data=b"abc",
    )

    clock.advance(11)

    with pytest.raises(ArtifactExpired):
        store.get(plan_id="plan-1", plan_hash="hash-1", artifact_id="oc-artifact-1")
    with pytest.raises(ArtifactMissing):
        store.get(plan_id="plan-1", plan_hash="hash-1", artifact_id="oc-artifact-1")


def test_hash_and_size_mismatch_fail_closed() -> None:
    """The caller's expected identity must match stored bytes."""
    store = ArtifactBundleStore(clock=FakeClock())
    manifest = store.put(
        plan_id="plan-1",
        plan_hash="hash-1",
        artifact_id="oc-artifact-1",
        filename="construct.gb",
        mime_type="chemical/x-genbank",
        data=b"abc",
    )

    with pytest.raises(ArtifactHashMismatch):
        store.get(
            plan_id="plan-1",
            plan_hash="hash-1",
            artifact_id="oc-artifact-1",
            expected_sha256="0" * 64,
        )
    with pytest.raises(ArtifactSizeMismatch):
        store.get(
            plan_id="plan-1",
            plan_hash="hash-1",
            artifact_id="oc-artifact-1",
            expected_size_bytes=manifest["size_bytes"] + 1,
        )


def test_manifest_extra_cannot_override_reserved_identity_fields() -> None:
    """Caller display metadata must not shadow hash/size identity fields."""
    store = ArtifactBundleStore(clock=FakeClock())

    with pytest.raises(ArtifactManifestCollision):
        store.put(
            plan_id="plan-1",
            plan_hash="hash-1",
            artifact_id="oc-artifact-1",
            filename="construct.gb",
            mime_type="chemical/x-genbank",
            data=b"abc",
            manifest_extra={"sha256": "0" * 64},
        )


def test_context_mismatch_fails_closed() -> None:
    """Artifacts are bound to the source record/request context."""
    store = ArtifactBundleStore(clock=FakeClock())
    store.put(
        plan_id="plan-1",
        plan_hash="hash-1",
        artifact_id="oc-artifact-1",
        filename="construct.gb",
        mime_type="chemical/x-genbank",
        data=b"abc",
        binding={"record_type": "experiment", "record_id": "200"},
    )

    with pytest.raises(ArtifactContextMismatch):
        store.get(
            plan_id="plan-1",
            plan_hash="hash-1",
            artifact_id="oc-artifact-1",
            binding={"record_type": "experiment", "record_id": "201"},
        )


def test_oversized_artifact_rejected_before_storage() -> None:
    """The bundle store enforces the writeback size limit."""
    store = ArtifactBundleStore(max_artifact_bytes=3, clock=FakeClock())

    with pytest.raises(ArtifactTooLarge):
        store.put(
            plan_id="plan-1",
            plan_hash="hash-1",
            artifact_id="oc-artifact-1",
            filename="construct.gb",
            mime_type="chemical/x-genbank",
            data=b"abcd",
        )


def test_purge_expired_removes_only_expired_artifacts() -> None:
    """Manual cleanup can remove stale in-memory bundles."""
    clock = FakeClock()
    store = ArtifactBundleStore(ttl_seconds=10, clock=clock)
    store.put(
        plan_id="plan-1",
        plan_hash="hash-1",
        artifact_id="expired",
        filename="expired.gb",
        mime_type="chemical/x-genbank",
        data=b"abc",
    )
    clock.advance(5)
    store.put(
        plan_id="plan-1",
        plan_hash="hash-1",
        artifact_id="fresh",
        filename="fresh.gb",
        mime_type="chemical/x-genbank",
        data=b"abc",
    )
    clock.advance(6)

    assert store.purge_expired() == 1
    with pytest.raises(ArtifactMissing):
        store.get(plan_id="plan-1", plan_hash="hash-1", artifact_id="expired")
    assert store.get(plan_id="plan-1", plan_hash="hash-1", artifact_id="fresh")


def test_get_artifact_bundle_store_returns_singleton() -> None:
    """The default process store is reused across callers."""
    assert get_artifact_bundle_store() is get_artifact_bundle_store()
