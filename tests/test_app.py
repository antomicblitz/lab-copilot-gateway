"""Smoke tests for the gateway scaffold endpoints."""

from fastapi.testclient import TestClient

from lab_copilot_gateway import __version__
from lab_copilot_gateway.app import create_app


def make_client() -> TestClient:
    return TestClient(create_app())


def test_health_returns_version_and_dependency_status() -> None:
    response = make_client().get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "service": "lab-copilot-gateway",
        "version": __version__,
        "status": "ok",
        "dependencies": {
            "audit_db": "not_configured",
            "elabftw": "not_configured",
            "opencloning": "not_configured",
            "wallac": "not_configured",
            "bentolab": "not_configured",
        },
    }


def test_public_config_is_deterministic_and_non_secret() -> None:
    response = make_client().get("/config/public")

    assert response.status_code == 200
    assert response.json() == {
        "service": "lab-copilot-gateway",
        "version": __version__,
        "environment": "local",
        "approvals_required_for_mutations": True,
        "hardware_execution_enabled": False,
        "autonomous_execution_enabled": False,
    }


def test_tools_registry_starts_empty() -> None:
    response = make_client().get("/tools")

    assert response.status_code == 200
    assert response.json() == {"tools": []}
