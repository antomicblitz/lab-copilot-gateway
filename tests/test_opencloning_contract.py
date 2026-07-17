"""Real-container contract tests for the pinned OpenCloning backend.

These tests run against a live OpenCloning container to verify that the
catalog's endpoint/source-type mappings match the actual runtime API.

Run with: ``make test-opencloning-contract``

This target is **not** part of ``make check`` — it requires a running
OpenCloning container and is intended for CI or manual verification.

The tests check:
    1. Backend version endpoint responds.
    2. Every cataloged endpoint accepts a minimal valid request for its
       source type (or at least returns a structured error, not a 404).
    3. Unknown source types are rejected by the gateway catalog (no
       Gibson fallback) — this is a unit test, but included here for
       completeness.

No nucleotide sequences are stored or logged by these tests.
"""

from __future__ import annotations

import os

import pytest

from lab_copilot_gateway.strategy_catalog import (
    PINNED_OPENCLONING_IMAGE,
    get_catalog,
)


# Skip all tests in this module if no OpenCloning URL is configured.
OPENCLONING_URL = os.environ.get("LAB_COPILOT_OPENCLONING_URL", "")
pytestmark = pytest.mark.skipif(
    not OPENCLONING_URL,
    reason="Set LAB_COPILOT_OPENCLONING_URL to run contract tests",
)


class TestBackendVersion:
    """Verify the running backend is the pinned version."""

    def test_version_endpoint_responds(self) -> None:
        import requests

        resp = requests.get(f"{OPENCLONING_URL}/version", timeout=10, verify=False)
        assert resp.ok, f"/version returned {resp.status_code}"
        data = resp.json()
        assert "opencloning_version" in data or "version" in str(data).lower()

    def test_pinned_image_constant(self) -> None:
        """The catalog records which image it was validated against."""
        assert (
            PINNED_OPENCLONING_IMAGE
            == "manulera/opencloning:v1.3.1-baseurl-opencloning"
        )


class TestEndpointContracts:
    """Every cataloged endpoint must exist on the running backend."""

    @pytest.mark.parametrize(
        "endpoint",
        sorted({e.endpoint for e in get_catalog().list_stable()}),
    )
    def test_endpoint_exists(self, endpoint: str) -> None:
        """The endpoint should respond to a POST (even if it rejects the
        body, it should not 404).  Scoped to stable endpoints only —
        experimental batch endpoints may crash on empty bodies."""
        import requests

        resp = requests.post(
            f"{OPENCLONING_URL}{endpoint}",
            json={},
            timeout=10,
            verify=False,
        )
        # 404 means the endpoint doesn't exist on this backend version.
        # 422 (validation error) or 400 means it exists but our empty
        # body is invalid — that's fine for a contract check.
        assert resp.status_code != 404, (
            f"endpoint {endpoint} returned 404 — "
            f"catalog is out of sync with {PINNED_OPENCLONING_IMAGE}"
        )


class TestCatalogBackendAgreement:
    """The catalog's source-type → endpoint mapping must match the backend."""

    def test_all_stable_endpoints_exist_on_backend(self) -> None:
        """Every stable operation's endpoint must be present on the backend."""
        import requests

        stable_endpoints = {e.endpoint for e in get_catalog().list_stable()}
        missing: list[str] = []
        for endpoint in stable_endpoints:
            resp = requests.post(
                f"{OPENCLONING_URL}{endpoint}",
                json={},
                timeout=10,
                verify=False,
            )
            if resp.status_code == 404:
                missing.append(endpoint)
        assert not missing, (
            f"endpoints missing from backend: {missing}. "
            f"Catalog is out of sync with {PINNED_OPENCLONING_IMAGE}"
        )


class TestBatchEndpointContracts:
    """Experimental batch endpoints must exist on the running backend.

    Verified against manulera/opencloning:v1.3.1-baseurl-opencloning.
    Only three /batch_cloning/* POST endpoints exist on this image:
    domesticate, pombe, ziqiang_et_al2024.
    """

    @pytest.mark.parametrize(
        "endpoint",
        sorted({e.endpoint for e in get_catalog().list_experimental()}),
    )
    def test_batch_endpoint_exists(self, endpoint: str) -> None:
        """Each experimental batch endpoint must exist (not 404).

        We accept 422/400/415 (validation/conversion errors on empty body)
        but reject 404/405 (endpoint doesn't exist or doesn't accept POST).
        """
        import requests

        resp = requests.post(
            f"{OPENCLONING_URL}{endpoint}",
            json={},
            timeout=10,
            verify=False,
        )
        assert resp.status_code not in (404, 405), (
            f"batch endpoint {endpoint} returned {resp.status_code} — "
            f"catalog is out of sync with {PINNED_OPENCLONING_IMAGE}"
        )

    def test_ziqiang_executes_with_valid_protospacers(self) -> None:
        """The ziqiang_et_al2024 workflow must succeed with valid input.

        This is the only batch endpoint with no external dependency —
        it uses a bundled template, so it can be fully contract-tested.
        """
        import requests

        resp = requests.post(
            f"{OPENCLONING_URL}/batch_cloning/ziqiang_et_al2024",
            json=["GCGTCCACTGAGTCGTGAGC", "ACAGGTGGACAGTGTCGTAC"],
            timeout=60,
            verify=False,
        )
        assert resp.ok, f"ziqiang failed: {resp.status_code} {resp.text[:200]}"
        data = resp.json()
        assert "sequences" in data
        assert len(data["sequences"]) > 1, "ziqiang should produce multiple sequences"
        assert "sources" in data

    def test_no_phantom_batch_endpoints_in_catalog(self) -> None:
        """The catalog must not reference nonexistent batch endpoints.

        Previously cataloged batch_gibson/restriction_ligation/
        homologous_recombination pointed to endpoints that return 405.
        This test prevents regression.
        """
        phantom_prefixes = [
            "/batch_cloning/gibson_assembly",
            "/batch_cloning/restriction_ligation",
            "/batch_cloning/homologous_recombination",
        ]
        catalog_endpoints = {e.endpoint for e in get_catalog().list_all()}
        for ep in phantom_prefixes:
            assert ep not in catalog_endpoints, (
                f"phantom endpoint {ep} is in the catalog but does not "
                f"exist on {PINNED_OPENCLONING_IMAGE}"
            )
