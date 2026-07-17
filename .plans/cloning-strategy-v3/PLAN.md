# Cloning Strategy v3 — Gateway-Local Plan

Authoritative umbrella plan:
`elabftw-lambdabiolab/docs/plans/cloning-strategy-v3-review-workspace.md`

This file tracks gateway-local acceptance results without duplicating
requirements. Update the status column as slices complete.

## Slice status

| Slice | Description | Status | Evidence |
|-------|-------------|--------|----------|
| 1 | Pinned OpenCloning capability contract | DONE | `strategy_catalog.py` (20 ops), 45 tests, `make test-opencloning-contract`, CI contract job; commit `4bf68e6` |
| 2 | Canonical strategy schema v3 | TODO | — |
| 3 | Canonical preparation + secure approval | TODO | — |
| 6 | Deterministic execution (core methods) | TODO | — |
| 7 | Durable async strategy runs | TODO | — |
| 10 | HR + CRISPR-HDR | TODO | — |
| 11 | Site-specific recombination | TODO | — |
| 12 | Small product transformations | TODO | — |
| 13 | Independent batch execution | TODO | — |
| 14 | Native batch (experimental) | TODO | — |
| 15 | Full matrix + rollout | TODO | — |

Slices 4, 5, 8, 9 live in the deployment repo (`elabftw-lambdabiolab`).

## Local acceptance commands

- `make check` — lint, typecheck, format, complexity, tests
- `make test-opencloning-contract` — real pinned-container contract (Slice 1+)
