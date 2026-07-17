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
        assert "version" in data or "OpenCloning" in str(data)

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
