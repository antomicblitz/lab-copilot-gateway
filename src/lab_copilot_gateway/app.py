"""FastAPI application for the lab copilot gateway scaffold."""

from __future__ import annotations

from fastapi import FastAPI

from lab_copilot_gateway import __version__
from lab_copilot_gateway.config import get_public_config
from lab_copilot_gateway.tools import list_tools


def create_app() -> FastAPI:
    """Create the ASGI application."""
    service_name = "lab-copilot-gateway"
    api = FastAPI(title="Lab Copilot Gateway", version=__version__)

    # CORS is intentionally closed in the scaffold. Configure allowed LibreChat
    # origins when browser-based calls are introduced.

    @api.get("/health")
    def health() -> dict[str, object]:
        return {
            "service": service_name,
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

    @api.get("/config/public")
    def public_config() -> dict[str, object]:
        return get_public_config(service_name=service_name, version=__version__)

    @api.get("/tools")
    def tools() -> dict[str, list[dict[str, object]]]:
        return {"tools": list_tools()}

    return api


app = create_app()
