# lab-copilot-gateway

Policy, approval, audit, and tool-mediation gateway for the Lambda Biolab eLabFTW lab copilot.

This service is intentionally separate from eLabFTW and LibreChat. LibreChat is the chat UI; this gateway is the deterministic boundary for lab actions.

## Current scaffold

- `GET /health` returns service version and dependency status.
- `GET /config/public` returns non-secret public configuration.
- `GET /tools` returns the current curated tool registry, empty in this scaffold.

Real eLabFTW, OpenCloning, Wallac, and BentoLab adapters are intentionally not implemented yet.

## Development

```bash
uv run pytest
make check
make run
```
