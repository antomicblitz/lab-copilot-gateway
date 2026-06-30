"""Tests for the OpenCloning adapter (C16).

Acceptance checks (from build plan):

    1. Adapter can parse a small allowed test sequence file.
    2. Oversized or disallowed file is rejected before OpenCloning.
    3. Errors are structured and audited.

Additional coverage (mirrors the eLabFTW adapter test patterns):

    * Unmapped caller denies before any downstream call.
    * Invalid/expired context token denies with audit.
    * Kill switch denies via policy engine; audit deny record preserved.
    * Client error propagates after audit record persisted.
    * Token-identity mismatch denies with audit.
    * Client-not-configured raises structured error after audit.
    * All four tool methods (parse, manual, oligo, assembly) succeed.
    * Audit records carry tool_args_hash on every path.
    * Module-level singleton get/reset behaviour.
"""

from __future__ import annotations

import datetime as _dt

import pytest

from lab_copilot_gateway.audit import AuditStore
from lab_copilot_gateway.elabftw import (
    ContextTokenClaims,
    InvalidContextToken,
    UnmappedCaller,
    mint_context_token,
)
from lab_copilot_gateway.identity import MappedIdentity
from lab_copilot_gateway.opencloning import (
    ALLOWED_FILE_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    DisallowedFileType,
    FileTooLarge,
    HttpOpenCloningClient,
    OpenCloningAdapter,
    OpenCloningAdapterError,
    OpenCloningResult,
    PolicyDenied,
    StubOpenCloningClient,
    _default_client_from_env,
    _infer_extension,
    get_opencloning_adapter,
    reset_opencloning_adapter,
)
from lab_copilot_gateway.policy import PolicyEngine


SECRET = b"test-secret-do-not-use-in-prod"


@pytest.fixture(autouse=True)
def _stable_token_secret(monkeypatch) -> None:
    """Pin the HMAC secret so tokens minted in tests verify in the adapter."""
    monkeypatch.setenv("LAB_COPILOT_TOKEN_SECRET", SECRET.decode("latin-1"))


# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def audit() -> AuditStore:
    s = AuditStore(db_path=":memory:")
    yield s
    s.close()


@pytest.fixture
def policy() -> PolicyEngine:
    return PolicyEngine()  # default: max_tier=6, no kill switches


@pytest.fixture
def stub_client() -> StubOpenCloningClient:
    return StubOpenCloningClient(
        responses={
            "parse_sequence_file": {
                "sequences": [{"id": "seq1", "length": 100, "circular": False}],
            },
            "manual_sequence": {
                "sequence": "ATCG",
                "length": 4,
                "circular": False,
            },
            "oligo_hybridization": {
                "product": {"length": 24, "overhang": "blunt"},
            },
            "simulate_assembly": {
                "construct": {"length": 5000, "circular": True},
            },
        }
    )


@pytest.fixture
def adapter(
    policy: PolicyEngine,
    audit: AuditStore,
    stub_client: StubOpenCloningClient,
) -> OpenCloningAdapter:
    return OpenCloningAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=stub_client,
    )


# --- helpers ----------------------------------------------------------------


def _claims(
    *,
    experiment_id: int = 42,
    mapped_elabftw_user_id: str = "elab-human-1",
    keycloak_subject: str | None = "kc-human-1",
    librechat_user_id: str | None = "lc-human-1",
    ttl_seconds: int = 600,
) -> ContextTokenClaims:
    issued_at = _dt.datetime.now(_dt.timezone.utc)
    expires_at = issued_at + _dt.timedelta(seconds=ttl_seconds)
    return ContextTokenClaims(
        experiment_id=experiment_id,
        mapped_elabftw_user_id=mapped_elabftw_user_id,
        keycloak_subject=keycloak_subject,
        librechat_user_id=librechat_user_id,
        issued_at=issued_at.isoformat(),
        expires_at=expires_at.isoformat(),
    )


def _token(claims: ContextTokenClaims | None = None) -> str:
    return mint_context_token(claims or _claims())


def _identity(
    *,
    elab_user_id: str = "elab-human-1",
    team_ids: list[str] | None = None,
    kc_subject: str = "kc-human-1",
    lc_user_id: str = "lc-human-1",
) -> MappedIdentity:
    return MappedIdentity(
        keycloak_subject=kc_subject,
        librechat_user_id=lc_user_id,
        elabftw_user_id=elab_user_id,
        elabftw_team_ids=team_ids or ["team-1"],
    )


def _audit_rows(audit: AuditStore) -> list[dict]:
    """Read all audit rows as dicts for assertions.

    JSON columns (``*_json``) are parsed back to Python objects so tests
    can assert on structured fields directly.
    """
    import json

    conn = audit._connect()
    with audit._lock:
        rows = conn.execute(
            "SELECT * FROM audit_records ORDER BY created_at"
        ).fetchall()
        cols = [
            d[0]
            for d in conn.execute("SELECT * FROM audit_records LIMIT 0").description
        ]
    result = []
    for row in rows:
        d = dict(zip(cols, row))
        # Parse JSON columns (stored with _json suffix).
        for key in list(d.keys()):
            if key.endswith("_json") and d[key]:
                d[key.removesuffix("_json")] = json.loads(d[key])
        result.append(d)
    return result


# --- parse_sequence_file: happy path ----------------------------------------


def test_parse_sequence_file_succeeds(
    adapter: OpenCloningAdapter,
    audit: AuditStore,
    stub_client: StubOpenCloningClient,
) -> None:
    """Acceptance #1: adapter can parse a small allowed test sequence file."""
    result = adapter.parse_sequence_file(
        context_token=_token(),
        file_content="ATCGATCGATCG",
        file_format="fasta",
        mapped_identity=_identity(),
    )
    assert isinstance(result, OpenCloningResult)
    assert result.tool_name == "opencloning.parse_sequence_file"
    assert "sequences" in result.result
    # Stub recorded the call
    assert len(stub_client.calls) == 1
    assert stub_client.calls[0]["method"] == "parse_sequence_file"
    assert stub_client.calls[0]["file_format"] == "fasta"
    # Audit row written
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["policy_decision"] == "allow"
    assert rows[0]["tool_name"] == "opencloning.parse_sequence_file"
    assert rows[0]["tool_args_hash"] is not None
    assert rows[0]["context_refs"] == [{"kind": "experiment", "id": 42}]


def test_parse_sequence_file_all_formats(
    adapter: OpenCloningAdapter,
    audit: AuditStore,
) -> None:
    """All allowed file formats are accepted by the adapter."""
    for fmt in ("genbank", "fasta", "snapgene", "embl"):
        result = adapter.parse_sequence_file(
            context_token=_token(),
            file_content="ATCG",
            file_format=fmt,
            mapped_identity=_identity(),
        )
        assert result.tool_name == "opencloning.parse_sequence_file"


# --- parse_sequence_file: file-size limit -----------------------------------


def test_parse_sequence_file_rejects_oversized(
    adapter: OpenCloningAdapter,
    audit: AuditStore,
    stub_client: StubOpenCloningClient,
) -> None:
    """Acceptance #2: oversized file is rejected before OpenCloning."""
    oversized = "A" * (MAX_FILE_SIZE_BYTES + 1)
    with pytest.raises(FileTooLarge) as exc_info:
        adapter.parse_sequence_file(
            context_token=_token(),
            file_content=oversized,
            file_format="fasta",
            mapped_identity=_identity(),
        )
    assert exc_info.value.size == MAX_FILE_SIZE_BYTES + 1
    assert exc_info.value.limit == MAX_FILE_SIZE_BYTES
    # No downstream call was made
    assert len(stub_client.calls) == 0
    # Audit deny record written
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["policy_decision"] == "deny"
    assert rows[0]["error"]["code"] == "FILE_TOO_LARGE"


def test_parse_sequence_file_accepts_max_size(
    adapter: OpenCloningAdapter,
    stub_client: StubOpenCloningClient,
) -> None:
    """File at exactly the size limit is accepted (boundary check)."""
    content = "A" * MAX_FILE_SIZE_BYTES
    result = adapter.parse_sequence_file(
        context_token=_token(),
        file_content=content,
        file_format="fasta",
        mapped_identity=_identity(),
    )
    assert result.tool_name == "opencloning.parse_sequence_file"


# --- parse_sequence_file: file-type allowlist --------------------------------


def test_parse_sequence_file_rejects_disallowed_type(
    adapter: OpenCloningAdapter,
    audit: AuditStore,
    stub_client: StubOpenCloningClient,
) -> None:
    """Acceptance #2: disallowed file type is rejected before OpenCloning."""
    with pytest.raises(DisallowedFileType) as exc_info:
        adapter.parse_sequence_file(
            context_token=_token(),
            file_content="ATCG",
            file_format="exe",
            mapped_identity=_identity(),
        )
    assert ".exe" in exc_info.value.extension or "exe" in exc_info.value.extension
    # No downstream call was made
    assert len(stub_client.calls) == 0
    # Audit deny record written
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["policy_decision"] == "deny"
    assert rows[0]["error"]["code"] == "DISALLOWED_FILE_TYPE"


# --- manual_sequence: happy path --------------------------------------------


def test_manual_sequence_succeeds(
    adapter: OpenCloningAdapter,
    audit: AuditStore,
    stub_client: StubOpenCloningClient,
) -> None:
    """manual_sequence tool returns validated sequence description."""
    result = adapter.manual_sequence(
        context_token=_token(),
        sequence="ATCGATCG",
        circular=False,
        mapped_identity=_identity(),
    )
    assert result.tool_name == "opencloning.manual_sequence"
    assert result.result["sequence"] == "ATCG"
    assert len(stub_client.calls) == 1
    assert stub_client.calls[0]["method"] == "manual_sequence"
    # Audit row
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["policy_decision"] == "allow"


# --- oligo_hybridization: happy path ----------------------------------------


def test_oligo_hybridization_succeeds(
    adapter: OpenCloningAdapter,
    audit: AuditStore,
    stub_client: StubOpenCloningClient,
) -> None:
    """oligo_hybridization tool returns product description."""
    result = adapter.oligo_hybridization(
        context_token=_token(),
        forward_oligo="ATCGATCGATCGATCGATCG",
        reverse_oligo="CGATCGATCGATCGATCGAT",
        minimal_annealing=15,
        mapped_identity=_identity(),
    )
    assert result.tool_name == "opencloning.oligo_hybridization"
    assert "product" in result.result
    assert len(stub_client.calls) == 1
    assert stub_client.calls[0]["method"] == "oligo_hybridization"
    assert stub_client.calls[0]["minimal_annealing"] == 15


# --- simulate_assembly: happy path ------------------------------------------


def test_simulate_assembly_succeeds(
    adapter: OpenCloningAdapter,
    audit: AuditStore,
    stub_client: StubOpenCloningClient,
) -> None:
    """simulate_assembly tool returns predicted construct."""
    fragments = [
        {"id": "frag1", "sequence": "ATCG"},
        {"id": "frag2", "sequence": "GCTA"},
    ]
    assembly_config = {"type": "ligation"}
    result = adapter.simulate_assembly(
        context_token=_token(),
        fragments=fragments,
        assembly_config=assembly_config,
        mapped_identity=_identity(),
    )
    assert result.tool_name == "opencloning.simulate_assembly"
    assert "construct" in result.result
    assert len(stub_client.calls) == 1
    assert stub_client.calls[0]["method"] == "simulate_assembly"
    assert stub_client.calls[0]["fragment_count"] == 2


# --- unmapped caller --------------------------------------------------------


def test_unmapped_caller_denies_before_http(
    adapter: OpenCloningAdapter,
    audit: AuditStore,
    stub_client: StubOpenCloningClient,
) -> None:
    """Unmapped caller denies before any downstream call."""
    with pytest.raises(UnmappedCaller):
        adapter.manual_sequence(
            context_token=_token(),
            sequence="ATCG",
            mapped_identity=None,
        )
    assert len(stub_client.calls) == 0
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["policy_decision"] == "deny"
    assert rows[0]["error"]["code"] == "UNMAPPED_CALLER"
    assert rows[0]["tool_args_hash"] is not None


# --- invalid context token --------------------------------------------------


def test_missing_context_token_denies(
    adapter: OpenCloningAdapter,
    audit: AuditStore,
) -> None:
    """Missing context token denies with audit."""
    with pytest.raises(InvalidContextToken, match="missing"):
        adapter.manual_sequence(
            context_token="",
            sequence="ATCG",
            mapped_identity=_identity(),
        )
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["error"]["code"] == "MISSING_CONTEXT_TOKEN"


def test_invalid_context_token_denies(
    adapter: OpenCloningAdapter,
    audit: AuditStore,
) -> None:
    """Malformed context token denies with audit."""
    with pytest.raises(InvalidContextToken):
        adapter.manual_sequence(
            context_token="garbage.payload",
            sequence="ATCG",
            mapped_identity=_identity(),
        )
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["error"]["code"] == "INVALID_CONTEXT_TOKEN"


def test_expired_context_token_denies(
    adapter: OpenCloningAdapter,
    audit: AuditStore,
) -> None:
    """Expired context token denies with audit."""
    expired_claims = _claims(ttl_seconds=-10)
    with pytest.raises(InvalidContextToken, match="expired"):
        adapter.manual_sequence(
            context_token=_token(expired_claims),
            sequence="ATCG",
            mapped_identity=_identity(),
        )
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert "expired" in rows[0]["error"]["detail"]


# --- token-identity mismatch ------------------------------------------------


def test_token_user_mismatch_denies(
    adapter: OpenCloningAdapter,
    audit: AuditStore,
    stub_client: StubOpenCloningClient,
) -> None:
    """Token bound to user A cannot be used by user B."""
    with pytest.raises(InvalidContextToken, match="user does not match"):
        adapter.manual_sequence(
            context_token=_token(),
            sequence="ATCG",
            mapped_identity=_identity(elab_user_id="different-user"),
        )
    assert len(stub_client.calls) == 0
    rows = _audit_rows(audit)
    assert rows[0]["error"]["code"] == "CONTEXT_TOKEN_USER_MISMATCH"


def test_token_kc_subject_mismatch_denies(
    adapter: OpenCloningAdapter,
    stub_client: StubOpenCloningClient,
) -> None:
    """Token keycloak_subject mismatch denies."""
    with pytest.raises(InvalidContextToken, match="keycloak_subject"):
        adapter.manual_sequence(
            context_token=_token(),
            sequence="ATCG",
            mapped_identity=_identity(kc_subject="different-kc"),
        )
    assert len(stub_client.calls) == 0


# --- policy deny ------------------------------------------------------------


def test_kill_switch_denies(
    audit: AuditStore,
    stub_client: StubOpenCloningClient,
) -> None:
    """Kill switch denies via policy engine; audit deny record preserved."""
    policy = PolicyEngine(kill_switches=("opencloning.*",))
    adapter = OpenCloningAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=stub_client,
    )
    with pytest.raises(PolicyDenied):
        adapter.manual_sequence(
            context_token=_token(),
            sequence="ATCG",
            mapped_identity=_identity(),
        )
    assert len(stub_client.calls) == 0
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["policy_decision"] == "deny"
    assert "kill_switch" in rows[0]["error"]["reason"]


# --- client not configured --------------------------------------------------


def test_client_not_configured_raises(
    audit: AuditStore,
    policy: PolicyEngine,
) -> None:
    """Client=None raises structured error after audit."""
    adapter = OpenCloningAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=None,
    )
    with pytest.raises(OpenCloningAdapterError) as exc_info:
        adapter.manual_sequence(
            context_token=_token(),
            sequence="ATCG",
            mapped_identity=_identity(),
        )
    assert exc_info.value.reason == "opencloning_not_configured"
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["error"]["code"] == "OPENCLONING_NOT_CONFIGURED"


# --- client error -----------------------------------------------------------


def test_client_error_propagates_after_audit(
    audit: AuditStore,
    policy: PolicyEngine,
) -> None:
    """Acceptance #3: errors are structured and audited."""
    stub = StubOpenCloningClient(error=RuntimeError("connection refused"))
    adapter = OpenCloningAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=stub,
    )
    with pytest.raises(OpenCloningAdapterError) as exc_info:
        adapter.manual_sequence(
            context_token=_token(),
            sequence="ATCG",
            mapped_identity=_identity(),
        )
    assert exc_info.value.reason == "client_error"
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["policy_decision"] == "allow"  # policy allowed, client failed
    assert rows[0]["error"]["code"] == "CLIENT_ERROR"
    assert rows[0]["error"]["exception"] == "RuntimeError"


# --- audit hash on all paths ------------------------------------------------


def test_audit_hash_present_on_success(
    adapter: OpenCloningAdapter,
    audit: AuditStore,
) -> None:
    """tool_args_hash is present on success path."""
    adapter.manual_sequence(
        context_token=_token(),
        sequence="ATCG",
        mapped_identity=_identity(),
    )
    rows = _audit_rows(audit)
    assert rows[0]["tool_args_hash"] is not None
    assert len(rows[0]["tool_args_hash"]) == 64  # SHA-256 hex


def test_audit_hash_present_on_deny(
    adapter: OpenCloningAdapter,
    audit: AuditStore,
) -> None:
    """tool_args_hash is present on deny path (C08/C09 invariant)."""
    with pytest.raises(UnmappedCaller):
        adapter.manual_sequence(
            context_token=_token(),
            sequence="ATCG",
            mapped_identity=None,
        )
    rows = _audit_rows(audit)
    assert rows[0]["tool_args_hash"] is not None


# --- audit context_refs -----------------------------------------------------


def test_audit_context_refs_include_experiment_id(
    adapter: OpenCloningAdapter,
    audit: AuditStore,
) -> None:
    """context_refs include the experiment_id from the token."""
    adapter.manual_sequence(
        context_token=_token(_claims(experiment_id=99)),
        sequence="ATCG",
        mapped_identity=_identity(),
    )
    rows = _audit_rows(audit)
    assert rows[0]["context_refs"] == [{"kind": "experiment", "id": 99}]


# --- api_call_summary -------------------------------------------------------


def test_audit_api_call_summary_on_success(
    adapter: OpenCloningAdapter,
    audit: AuditStore,
) -> None:
    """api_call_summary is populated on success."""
    adapter.manual_sequence(
        context_token=_token(),
        sequence="ATCG",
        mapped_identity=_identity(),
    )
    rows = _audit_rows(audit)
    assert rows[0]["api_call_summary"]["method"] == "POST"
    assert rows[0]["api_call_summary"]["path"] == "/manually_typed"


# --- singleton --------------------------------------------------------------


def test_singleton_get_and_reset() -> None:
    """get_opencloning_adapter returns a singleton; reset clears it."""
    reset_opencloning_adapter()
    a1 = get_opencloning_adapter()
    a2 = get_opencloning_adapter()
    assert a1 is a2
    reset_opencloning_adapter()
    a3 = get_opencloning_adapter()
    assert a3 is not a1


# --- _default_client_from_env -----------------------------------------------


def test_default_client_from_env_returns_none_when_unset(monkeypatch) -> None:
    """No env vars → None (adapter will refuse calls)."""
    monkeypatch.delenv("LAB_COPILOT_OPENCLONING_BASE_URL", raising=False)
    assert _default_client_from_env() is None


def test_default_client_from_env_returns_client_when_set(monkeypatch) -> None:
    """Env vars set → HttpOpenCloningClient."""
    monkeypatch.setenv("LAB_COPILOT_OPENCLONING_BASE_URL", "http://opencloning:8000")
    client = _default_client_from_env()
    assert client is not None
    assert isinstance(client, HttpOpenCloningClient)
    assert client.base_url == "http://opencloning:8000"


# --- _infer_extension -------------------------------------------------------


def test_infer_extension_maps_known_formats() -> None:
    """Known file formats map to allowed extensions."""
    assert _infer_extension("genbank") == ".gb"
    assert _infer_extension("fasta") == ".fasta"
    assert _infer_extension("snapgene") == ".dna"
    assert _infer_extension("embl") == ".embl"


def test_infer_extension_passes_through_unknown() -> None:
    """Unknown formats get a .{format} extension (will be rejected by allowlist)."""
    assert _infer_extension("exe") == ".exe"


# --- ALLOWED_FILE_EXTENSIONS ------------------------------------------------


def test_allowed_extensions_include_all_supported() -> None:
    """All OpenCloning-supported formats are in the allowlist."""
    for ext in (".gb", ".gbk", ".ape", ".fasta", ".fa", ".dna", ".embl"):
        assert ext in ALLOWED_FILE_EXTENSIONS


# --- HttpOpenCloningClient construction -------------------------------------


def test_http_client_requires_base_url() -> None:
    """HttpOpenCloningClient rejects empty base_url."""
    with pytest.raises(ValueError, match="non-empty base_url"):
        HttpOpenCloningClient(base_url="")


# --- StubOpenCloningClient default response ----------------------------------


def test_stub_client_default_response() -> None:
    """Stub returns default ok response when no pre-seeded data."""
    stub = StubOpenCloningClient()
    result = stub.manual_sequence("ATCG")
    assert result == {"ok": True}
    assert len(stub.calls) == 1


def test_stub_client_records_calls() -> None:
    """Stub records all calls for assertion."""
    stub = StubOpenCloningClient()
    stub.parse_sequence_file(b"ATCG", "fasta")
    stub.manual_sequence("ATCG", circular=True)
    stub.oligo_hybridization("forward", "reverse", 15)
    stub.simulate_assembly([{"id": "f1"}], {"type": "ligation"})
    assert len(stub.calls) == 4
    assert stub.calls[0]["method"] == "parse_sequence_file"
    assert stub.calls[1]["method"] == "manual_sequence"
    assert stub.calls[2]["method"] == "oligo_hybridization"
    assert stub.calls[3]["method"] == "simulate_assembly"
