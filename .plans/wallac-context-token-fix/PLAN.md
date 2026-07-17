# Plan: Fix Wallac (and BentoLab) context_token requirement blocking read-only operations

## Symptom

When a user opens the Lab Copilot chat without an experiment/resource open in
eLabFTW and asks for a plate reader measurement (e.g. "Measure 610 nm
absorbance, 1s exposure, all wells"), the LLM reports:

> "I'm having trouble accessing the Wallac plate reader due to a missing
> context token. This typically happens when no experiment is currently open
> in eLabFTW."

The LLM then refuses to proceed, even though the Wallac vm-agent is running
and reachable.

## Root Cause

The gateway's Wallac adapter (`wallac.py`) requires a `context_token` for
**every** tool call, including read-only operations like listing protocols
(`wallac.call GET /protocols`) and checking instrument status
(`wallac.get_status`). This contradicts the C60 "no-context mode" design,
which made the `context_token` optional at the orchestrator level.

### The contradiction

1. **Orchestrator (server.js:2607):** `context_token` is optional. The chat
   works without an experiment open. The orchestrator passes
   `context_token: context.context_token || ""` (empty string when absent).

2. **System prompt (server.js:379-391):** Tells the LLM: "No experiment or
   resource is currently open. The chat is still usable — you can plan a
   cloning strategy, search OpenCloning parts, **look up wallac protocols**..."

3. **Orchestrator comment (server.js:79):** "opencloning.search_parts,
   wallac.get_status, anything [doesn't need context_token]"

4. **Gateway (wallac.py:1741):** `if not context_token: raise
   InvalidContextToken("missing")` — rejects ALL Wallac calls when
   `context_token` is empty/absent.

The C60 feature was implemented at the orchestrator level but the gateway
adapters were never updated to match.

### Key finding: experiment_id is NOT used for Wallac API calls

The `context_token`'s `experiment_id` is used in the Wallac adapter ONLY for:
- Audit trail records (`experiment_id` field in audit rows)
- Token-identity binding (verifying the user matches)

It is **NOT** used for the actual vm-agent API calls. The `exec_fn` just calls
`client.get_status()` or `client.call(method, endpoint, body)` — none of which
need an `experiment_id`.

Even `wallac.run` (real hardware execution) does NOT use the
`context_token`'s `experiment_id` for writeback. The bridge creates a NEW
experiment for the results:

```python
# wallac.py:898-906
job_payload = {
    "title": f"Lab Copilot — Protocol {protocol_id}",
    "execution_mode": "existing_protocol",
    "protocol_id": protocol_id,
    # Don't pass the current experiment_id — the bridge creates
    # a NEW experiment for the results.
    "elabftw_experiment_id": 0,
}
```

### Affected code paths

The `missing_context_token` check appears in these locations:

| File | Line | Method | Tool | Tier | Mutability |
|------|------|--------|------|------|------------|
| `wallac.py` | 1741 | `_execute()` | get_status, call, propose, validate, prepare | OPERATIONAL_READ_ONLY / VALIDATION_DRY_RUN | read |
| `wallac.py` | 690 | `_execute_run()` | wallac.run | BOUNDED_WRITES | mutate |
| `wallac.py` | 1026 | `bridge_status()` | wallac.bridge_status | OPERATIONAL_READ_ONLY | read |
| `wallac.py` | 1495 | `_execute_submit()` | submit_generated_protocol | HARDWARE_EXECUTION | mutate |
| `bentolab.py` | 438 | `_execute()` | bentolab.get_status, bentolab.call | OPERATIONAL_READ_ONLY | read |

### Dead env var

`LAB_COPILOT_RELAX_TOKEN_CHECK` is set to `"true"` in `docker-compose.yml:344`
for the `copilot-orchestrator` service, but is **not referenced anywhere** in
the orchestrator's `server.js` or the gateway's Python code. It appears to be
a leftover from an earlier design that was superseded by C60's
`LAB_COPILOT_REQUIRE_CONTEXT_TOKEN`.

## Fix

### Principle

Make the `context_token` check conditional on the tool's mutability:
- **Read-only tools** (mutability="read"): `context_token` is OPTIONAL. If
  present, verify it and use for audit/identity binding. If absent, skip
  token verification and identity binding, proceed with
  `experiment_id=None` in audit records.
- **Write/mutate tools** (mutability="mutate"): `context_token` is OPTIONAL
  for `wallac.run` (the bridge creates its own experiment). Still REQUIRED for
  `wallac.submit_generated_protocol` (tier 5, blocked in v1).

### Changes

#### 1. `wallac.py:_execute()` (line ~1741) — make context_token optional for read-only tools

Add a `requires_context_token: bool = True` parameter to `_execute()`. When
`False`, skip the context_token check and identity binding, but still audit
with `experiment_id=None`.

Set `requires_context_token=False` in the calls from:
- `get_status()` (line 516)
- `call()` (line 568)
- `propose_generated_protocol()` 
- `validate_generated_protocol()`
- `prepare_submission_package()`

For `_execute()`, when `context_token` is absent and
`requires_context_token=False`:
- Skip step 1 (token verify) — set `claims = None`, `experiment_id = None`
- Skip step 3 (token-identity binding) — only check `mapped_identity is None`
- Proceed with steps 4-6 (policy, client check, downstream call)
- In audit records, use `experiment_id=None`

#### 2. `wallac.py:_execute_run()` (line ~690) — make context_token optional for wallac.run

Since the bridge creates its own experiment (`elabftw_experiment_id: 0`),
`wallac.run` does NOT need the context_token's experiment_id. Make the
context_token optional:
- If present: verify token, do identity binding, use `claims.experiment_id` in audit
- If absent: skip token verify and identity binding, use `experiment_id=None` in audit
- The approval system still works (it uses `compute_args_hash(raw_args)`, not the token)

#### 3. `wallac.py:bridge_status()` (line ~1026) — make context_token optional

This is a read-only operation. Apply the same pattern: if `context_token` is
absent, skip token verification and proceed with `experiment_id=None` in audit.

#### 4. `bentolab.py:_execute()` (line ~438) — make context_token optional for read-only tools

Same pattern as `wallac.py:_execute()`. Add `requires_context_token` parameter,
set to `False` for read-only BentoLab tools.

#### 5. Update tests

- `tests/test_wallac.py:test_missing_context_token_denies` (line 309): Update
  to reflect that `get_status` (read-only) NO LONGER denies without a token.
  Split into two tests:
  - `test_read_only_tool_works_without_context_token` — verifies get_status
    succeeds without a token
  - `test_mutate_tool_denies_without_context_token` — verifies wallac.run
    (or submit_generated_protocol) still denies without a token

- Add new tests for `wallac.call` and `wallac.bridge_status` without token.

- `tests/test_wallac.py:test_submit_missing_context_token` (line 1564): Keep
  as-is (submit_generated_protocol still requires token).

#### 6. System prompt update (server.js)

Update the no-context-mode branch of `buildSystemPrompt()` to explicitly tell
the LLM that Wallac tools (including `wallac.run`) work without an experiment
open, and the bridge creates a new experiment for the results.

Current (server.js:385-391):
```
"No experiment or resource is currently open. The chat is still
usable — you can plan a cloning strategy, search OpenCloning
parts, look up wallac protocols, search eLabFTW items, and
answer general questions. If the user wants to act on a
specific experiment or resource (read, amend, attach a design),
tell them to open it in eLabFTW first; the widget passes the
context_token automatically when one is open."
```

Updated:
```
"No experiment or resource is currently open. The chat is still
usable — you can plan a cloning strategy, search OpenCloning
parts, run Wallac plate reader measurements (the bridge creates
a new experiment for the results automatically), search eLabFTW
items, and answer general questions. If the user wants to act on
a specific experiment or resource (read, amend, attach a design),
tell them to open it in eLabFTW first; the widget passes the
context_token automatically when one is open."
```

#### 7. Clean up dead env var

Remove `LAB_COPILOT_RELAX_TOKEN_CHECK: "true"` from `docker-compose.yml:344`
(it's not referenced anywhere). Or repurpose it if needed.

## Verification

1. **Unit tests:** `cd /home/antonio/repos/lab-copilot-gateway && make test`
   (or `uv run pytest tests/test_wallac.py tests/test_bentolab.py -v`)
2. **Rebuild gateway image:** 
   ```bash
   cd /home/antonio/repos/lab-copilot-gateway
   docker build -t lab-copilot-gateway:dev .
   cd /home/antonio/repos/eLabFTW-lambdabiolab
   docker compose up -d gateway
   ```
3. **Rebuild orchestrator image:**
   ```bash
   docker compose up -d --build copilot-orchestrator
   ```
4. **End-to-end test:** Open the chat without an experiment open, ask for a
   plate reader measurement, verify the LLM can list protocols and run the
   measurement without a "missing context token" error.

## Files to modify

| File | Repo | Changes |
|------|------|---------|
| `src/lab_copilot_gateway/wallac.py` | lab-copilot-gateway | Make context_token optional in `_execute()`, `_execute_run()`, `bridge_status()` |
| `src/lab_copilot_gateway/bentolab.py` | lab-copilot-gateway | Make context_token optional in `_execute()` |
| `tests/test_wallac.py` | lab-copilot-gateway | Update tests for no-token read-only path |
| `tests/test_bentolab.py` | lab-copilot-gateway | Update tests for no-token read-only path |
| `copilot/orchestrator/server.js` | eLabFTW-lambdabiolab | Update system prompt no-context branch |
| `docker-compose.yml` | eLabFTW-lambdabiolab | Remove dead `LAB_COPILOT_RELAX_TOKEN_CHECK` env var |
