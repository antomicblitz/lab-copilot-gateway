"""Tests for the OpenCloning preview store and fetch endpoint.

Acceptance checks:

    1. Store and fetch a preview payload successfully.
    2. Expired payloads return None.
    3. Wrong context token hash returns None.
    4. Per-run limit eviction works.
    5. The fetch endpoint returns the correct payload shape.
    6. Missing/nonexistent preview_ref returns 404.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from lab_copilot_gateway.app import create_app
from lab_copilot_gateway.opencloning_previews import (
    OpenCloningPreviewStore,
    get_preview_store,
    reset_preview_store,
)


# --- Preview store unit tests -----------------------------------------------


def test_store_and_fetch() -> None:
    """Store a payload and fetch it back with the correct token hash."""
    store = OpenCloningPreviewStore(ttl_seconds=3600)
    token = "test-token"
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    preview_ref = store.store(
        run_id="run-1",
        step_id="step-1",
        context_token_hash=token_hash,
        sequence_id=42,
        file_format="genbank",
        file_content="LOCUS     test    100 bp    DNA    circular",
        is_circular=True,
    )

    assert preview_ref.startswith("preview-")

    fetched = store.fetch(preview_ref, token_hash)
    assert fetched is not None
    assert fetched.run_id == "run-1"
    assert fetched.step_id == "step-1"
    assert fetched.sequence_id == 42
    assert fetched.file_format == "genbank"
    assert fetched.is_circular is True
    assert "LOCUS" in fetched.file_content
    assert fetched.sequence_length == len("LOCUS     test    100 bp    DNA    circular")


def test_fetch_wrong_token() -> None:
    """Wrong context token hash should return None."""
    store = OpenCloningPreviewStore(ttl_seconds=3600)

    preview_ref = store.store(
        run_id="run-1",
        step_id="step-1",
        context_token_hash="correct-hash",
        sequence_id=1,
        file_format="genbank",
        file_content="LOCUS     test    100 bp",
    )

    assert store.fetch(preview_ref, "wrong-hash") is None


def test_fetch_expired() -> None:
    """Expired payloads should return None."""
    store = OpenCloningPreviewStore(ttl_seconds=0)  # expires immediately

    preview_ref = store.store(
        run_id="run-1",
        step_id="step-1",
        context_token_hash="hash",
        sequence_id=1,
        file_format="genbank",
        file_content="LOCUS     test    100 bp",
    )

    # Give it a moment to expire
    time.sleep(0.01)
    assert store.fetch(preview_ref, "hash") is None


def test_fetch_missing() -> None:
    """Nonexistent preview_ref should return None."""
    store = OpenCloningPreviewStore()
    assert store.fetch("preview-nonexistent", "hash") is None


def test_per_run_limit() -> None:
    """Per-run limit eviction should remove oldest previews."""
    store = OpenCloningPreviewStore(ttl_seconds=3600)

    # Store MAX_PREVIEWS_PER_RUN + 1 previews for the same run
    token_hash = "test-hash"
    refs: list[str] = []
    for i in range(51):  # MAX_PREVIEWS_PER_RUN = 50
        ref = store.store(
            run_id="run-limited",
            step_id=f"step-{i}",
            context_token_hash=token_hash,
            sequence_id=i,
            file_format="genbank",
            file_content=f"LOCUS     test_{i}    100 bp",
        )
        refs.append(ref)

    # The oldest ref should have been evicted
    assert store.fetch(refs[0], token_hash) is None
    # The most recent ref should still exist
    assert store.fetch(refs[-1], token_hash) is not None


def test_empty_sequence_content() -> None:
    """Empty file_content should still be stored with length 0."""
    store = OpenCloningPreviewStore(ttl_seconds=3600)
    token_hash = "hash"

    preview_ref = store.store(
        run_id="run-1",
        step_id="step-1",
        context_token_hash=token_hash,
        sequence_id=1,
        file_format="fasta",
        file_content="",
    )

    fetched = store.fetch(preview_ref, token_hash)
    assert fetched is not None
    assert fetched.sequence_length == 0
    assert fetched.file_format == "fasta"


def test_multiple_runs_independent() -> None:
    """Previews from different runs should not interfere."""
    store = OpenCloningPreviewStore(ttl_seconds=3600)
    token_hash = "hash"

    ref_a = store.store(
        run_id="run-a",
        step_id="step-1",
        context_token_hash=token_hash,
        sequence_id=1,
        file_format="genbank",
        file_content="SEQUENCE A",
    )
    ref_b = store.store(
        run_id="run-b",
        step_id="step-1",
        context_token_hash=token_hash,
        sequence_id=2,
        file_format="genbank",
        file_content="SEQUENCE B",
    )

    fetched_a = store.fetch(ref_a, token_hash)
    fetched_b = store.fetch(ref_b, token_hash)
    assert fetched_a is not None
    assert fetched_b is not None
    assert fetched_a.run_id == "run-a"
    assert fetched_b.run_id == "run-b"


# --- Adapter integration test -----------------------------------------------


def test_capture_preview_refs_produces_mapping() -> None:
    """capture_preview_refs should return a {seq_id: preview_ref} mapping."""
    from lab_copilot_gateway.opencloning import get_opencloning_adapter

    reset_preview_store()
    adapter = get_opencloning_adapter()

    result: dict[str, Any] = {
        "sequences": [
            {"id": 0, "file_content": "LOCUS     seq1    100 bp", "circular": True},
            {"id": 1, "file_content": "LOCUS     seq2    200 bp"},
            {"id": 2},  # No file_content — should be skipped
        ]
    }

    token_hash = hashlib.sha256(b"test-token").hexdigest()
    preview_map = adapter.capture_preview_refs(
        result=result,
        run_id="run-test",
        step_id="step-test",
        context_token_hash=token_hash,
    )

    assert len(preview_map) == 2
    assert 0 in preview_map
    assert 1 in preview_map
    assert preview_map[0].startswith("preview-")
    assert preview_map[1].startswith("preview-")


def test_capture_preview_refs_empty_result() -> None:
    """Empty or no sequences should return empty mapping."""
    from lab_copilot_gateway.opencloning import get_opencloning_adapter

    reset_preview_store()
    adapter = get_opencloning_adapter()

    token_hash = hashlib.sha256(b"test-token").hexdigest()
    assert (
        adapter.capture_preview_refs(
            result={}, run_id="r", step_id="s", context_token_hash=token_hash
        )
        == {}
    )
    assert (
        adapter.capture_preview_refs(
            result={"sequences": []},
            run_id="r",
            step_id="s",
            context_token_hash=token_hash,
        )
        == {}
    )


# --- HTTP endpoint test -----------------------------------------------------


@pytest.fixture
def client() -> TestClient:
    """Create a fresh app with a clean preview store for each test."""
    reset_preview_store()
    app = create_app()
    return TestClient(app)


def _make_preview(
    store: OpenCloningPreviewStore,
    run_id: str = "run-test",
    step_id: str = "step-test",
) -> tuple[str, str]:
    """Helper: store a preview and return (preview_ref, token_hash)."""
    token = "test-token"
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    ref = store.store(
        run_id=run_id,
        step_id=step_id,
        context_token_hash=token_hash,
        sequence_id=42,
        file_format="genbank",
        file_content="LOCUS     test    100 bp    DNA    circular",
        is_circular=True,
    )
    return ref, token


def test_fetch_endpoint_success(client: TestClient) -> None:
    """Fetch endpoint returns payload with correct shape."""
    store = get_preview_store()
    preview_ref, token = _make_preview(store)

    response = client.get(
        f"/v1/opencloning/previews/{preview_ref}",
        headers={"X-Lab-Copilot-Context-Token": token},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["preview_ref"] == preview_ref
    assert data["run_id"] == "run-test"
    assert data["step_id"] == "step-test"
    assert data["sequence_id"] == 42
    assert data["file_format"] == "genbank"
    assert "LOCUS" in data["file_content"]
    assert data["is_circular"] is True
    assert data["sequence_length"] == len("LOCUS     test    100 bp    DNA    circular")


def test_fetch_endpoint_404_missing(client: TestClient) -> None:
    """Missing preview_ref returns 404."""
    response = client.get(
        "/v1/opencloning/previews/preview-nonexistent",
        headers={"X-Lab-Copilot-Context-Token": "some-token"},
    )
    assert response.status_code == 404


def test_fetch_endpoint_404_wrong_token(client: TestClient) -> None:
    """Wrong context token returns 404 (token mismatch treated as not found)."""
    store = get_preview_store()
    preview_ref, _token = _make_preview(store)  # stored with "test-token"

    response = client.get(
        f"/v1/opencloning/previews/{preview_ref}",
        headers={"X-Lab-Copilot-Context-Token": "wrong-token"},
    )
    assert response.status_code == 404


def test_fetch_endpoint_404_expired(client: TestClient) -> None:
    """Expired payload returns 404."""
    store = OpenCloningPreviewStore(ttl_seconds=-1)  # already expired
    token = "test-token"
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    preview_ref = store.store(
        run_id="run-x",
        step_id="step-x",
        context_token_hash=token_hash,
        sequence_id=1,
        file_format="genbank",
        file_content="LOCUS expired",
    )

    # Manually add to the singleton store
    reset_preview_store()
    live_store = get_preview_store()
    # Override the singleton's store directly
    live_store._store[preview_ref] = store._store[preview_ref]

    response = client.get(
        f"/v1/opencloning/previews/{preview_ref}",
        headers={"X-Lab-Copilot-Context-Token": token},
    )
    assert response.status_code == 404


def test_fetch_endpoint_requires_token(client: TestClient) -> None:
    """Missing context token header should return 422 (validation error)."""
    response = client.get("/v1/opencloning/previews/some-ref")
    assert response.status_code == 422
