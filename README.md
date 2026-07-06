# lab-copilot-gateway

Policy, approval, audit, and tool-mediation gateway for the Lambda Biolab eLabFTW lab copilot.

This service is intentionally separate from eLabFTW and LibreChat. LibreChat is the chat UI; this gateway is the deterministic boundary for lab actions.

## Endpoints

### Unauthenticated

- `GET /health` — service version, dependency status, auth readiness
- `GET /config/public` — non-secret public configuration
- `GET /tools` — curated tool registry catalog (C06)

### Protected (requires JWT or dev-auth)

- `POST /audit` — append audit record
- `GET /audit/{action_id}` — read audit record
- `POST /policy/evaluate` — policy engine decision
- `POST /identity/resolve` — resolve Keycloak sub to eLabFTW identity
- `POST /approval/request` — issue single-use approval token
- `POST /approval/consume` — consume approval token
- `GET /approval/{approval_id}` — read approval token
- `POST /elabftw/mint_context_token` — mint experiment-scoped context token
- `POST /elabftw/read_current_experiment` — read experiment (C08)
- `POST /elabftw/draft_experiment_update` — create draft (C09)
- `POST /elabftw/amend_my_experiment_after_approval` — append amendment (C09)
- `POST /invoke` — LibreChat tool dispatch (C13)

## OpenCloning tools

Tools dispatched via `POST /invoke` for the OpenCloning cloning design API
(see `docs/opencloning-integration.md` in `eLabFTW-lambdabiolab`).

| Tool | Tier | Approval | Description |
|------|------|----------|-------------|
| `opencloning.parse_sequence_file` | 3 (VALIDATION_DRY_RUN) | No | Parse a GenBank/FASTA file into sequences |
| `opencloning.manual_sequence` | 3 | No | Create a sequence from raw text |
| `opencloning.oligo_hybridization` | 3 | No | Compute oligo hybridization product |
| `opencloning.simulate_assembly` | 3 | No | Simulate cloning assembly from fragments |
| `opencloning.writeback_artifact` | 4 (BOUNDED_WRITES) | Yes | Attach an approved design artifact to an experiment |
| `opencloning.call` | 3 | No | Generic OpenCloning API access (imports, PCR, restriction, Gibson, Golden Gate, CRISPR, primer design, etc.) |

Env vars for configuration:

| Variable | Purpose |
|----------|---------|
| `LAB_COPILOT_OPENCLONING_BASE_URL` | OpenCloning backend URL (empty = stub client) |
| `LAB_COPILOT_OPENCLONING_TIMEOUT` | Downstream call timeout in seconds (default 30) |

**Sequence store:** The gateway stores full sequence objects (with `file_content`) from OpenCloning responses in a process-local dict. When subsequent `opencloning.call` requests receive sequences without `file_content`, the gateway injects the stored bytes so OpenCloning can process them. This enables multi-step workflows (import → PCR → Gibson → writeback) where each step's output becomes the next step's input.

**Feature extraction:** Artifact manifests include a `sequence_summary.features` list with `{type, location, strand, label}` for every GenBank feature (CDS, promoter, terminator, RBS, etc.). The LLM uses these coordinates to plan PCR amplification, restriction digests, and assembly.

## Wallac tools

Tools dispatched via `POST /invoke` for the Wallac Victor2 plate reader
integration. The gateway reaches the vm-agent on the Windows VM via an
SSH tunnel (see `docs/runbook.md` in `eLabFTW-lambdabiolab`).

| Tool | Tier | Approval | Description |
|------|------|----------|-------------|
| `wallac.get_status` | OPERATIONAL_READ_ONLY | no | Read instrument status, current job, and hardware state |
| `wallac.call` | VALIDATION_DRY_RUN | no | Generic vm-agent API access (GET/POST/PATCH). Forces `dry_run=true` for `POST /runs` |
| `wallac.run` | BOUNDED_WRITES | **yes** | Real hardware execution — starts a measurement on the Victor2. Approval card appears automatically |

Env vars for configuration:

| Variable | Purpose |
|----------|---------|
| `LAB_COPILOT_WALLAC_BASE_URL` | vm-agent HTTP endpoint (via SSH tunnel) |
| `LAB_COPILOT_WALLAC_BEARER_TOKEN` | Bearer token for vm-agent auth (optional) |
| `LAB_COPILOT_WALLAC_TIMEOUT` | Downstream call timeout in seconds (default 10) |

## Authentication (C14)

### Production mode (`LAB_COPILOT_KEYCLOAK_VERIFY=true`)

All protected routes require `Authorization: Bearer <JWT>` header. The JWT
is verified against Keycloak JWKS. Identity is derived from the JWT's `sub`
claim — body fields are correlation metadata only.

Required environment variables:

| Variable | Purpose |
|---|---|
| `LAB_COPILOT_KEYCLOAK_ISSUER` | Expected JWT `iss` claim |
| `LAB_COPILOT_KEYCLOAK_AUDIENCE` | Expected JWT `aud` claim |
| `LAB_COPILOT_KEYCLOAK_JWKS_URL` | Keycloak JWKS endpoint |
| `LAB_COPILOT_KEYCLOAK_ALLOWED_ALGS` | Allowed algorithms (default: `RS256`) |
| `LAB_COPILOT_KEYCLOAK_JWKS_CACHE_TTL` | JWKS cache TTL in seconds (default: `300`) |
| `LAB_COPILOT_KEYCLOAK_JWKS_TIMEOUT` | JWKS fetch timeout (default: `10`) |

Fail-closed behavior:

- Missing issuer/audience/JWKS config → gateway refuses to start
- First-boot JWKS fetch failure → gateway refuses to start
- Unknown `kid` during JWKS outage → 401
- Expired JWKS cache → 401 (no grace period)

### Dev mode (`LAB_COPILOT_KEYCLOAK_VERIFY=false`)

Allowed only in `local`, `test`, or `dev` environments. Requires
`LAB_COPILOT_DEV_AUTH_SECRET` to be set. Requests must include:

```text
X-Lab-Copilot-Dev-Auth: <secret>
X-Lab-Copilot-Dev-Sub: <keycloak_subject>   # optional, defaults to "dev-unverified"
```

### Rollback

Production rollback = redeploy previous known-good image/config. **Never**
disable `LAB_COPILOT_KEYCLOAK_VERIFY` in a non-local environment.

## Development

```bash
uv run pytest
make check
make run
```

Tests use dev-auth mode by default (no Keycloak needed). The test fixture
in `tests/test_app.py` sets up dev auth config automatically.
