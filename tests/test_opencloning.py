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

from lab_copilot_gateway.approval import (
    ApprovalRequest,
    ApprovalStore,
    compute_args_hash,
)
from lab_copilot_gateway.artifact_bundle import ArtifactBundleStore
from lab_copilot_gateway.audit import AuditStore
from lab_copilot_gateway.elabftw import (
    ContextTokenClaims,
    InvalidContextToken,
    StubElabftwClient,
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
    _rewrite_opencloning_result_ids,
    get_opencloning_adapter,
    reset_opencloning_adapter,
)
from lab_copilot_gateway.policy import PolicyEngine, Tier


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
    sequences = [
        {"id": 0, "type": "TextFileSequence", "sequence_file_format": "genbank"},
        {"id": 1, "type": "TextFileSequence", "sequence_file_format": "genbank"},
    ]
    source = {"id": 0, "type": "GibsonAssemblySource"}
    result = adapter.simulate_assembly(
        context_token=_token(),
        sequences=sequences,
        source=source,
        mapped_identity=_identity(),
    )
    assert result.tool_name == "opencloning.simulate_assembly"
    assert "construct" in result.result
    assert len(stub_client.calls) == 1
    assert stub_client.calls[0]["method"] == "simulate_assembly"
    assert stub_client.calls[0]["sequence_count"] == 2


def test_rewrite_opencloning_result_ids_keeps_history_graph_consistent() -> None:
    """Returned source ids must follow gateway-rewritten output sequence ids."""
    result = {
        "sequences": [
            {"id": 0, "type": "TextFileSequence", "file_content": "LOCUS a\n//"}
        ],
        "sources": [
            {
                "id": 0,
                "type": "GibsonAssemblySource",
                "input": [{"sequence": 0}, {"sequence": 9}],
            }
        ],
    }
    store: dict[str, dict[str, object]] = {}

    next_id = _rewrite_opencloning_result_ids(result, store, 10)

    assert next_id == 11
    assert result["sequences"][0]["id"] == 11
    assert result["sources"][0]["id"] == 11
    assert result["sources"][0]["input"] == [{"sequence": 11}, {"sequence": 9}]
    assert store["11"]["original_id"] == "0"


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
    stub.simulate_assembly([{"id": 0}], {"id": 0, "type": "GibsonAssemblySource"})
    assert len(stub.calls) == 4
    assert stub.calls[0]["method"] == "parse_sequence_file"
    assert stub.calls[1]["method"] == "manual_sequence"
    assert stub.calls[2]["method"] == "oligo_hybridization"
    assert stub.calls[3]["method"] == "simulate_assembly"


# ============================================================================
# C17 — OpenCloning eLabFTW writeback tests
# ============================================================================
#
# Acceptance checks (from build plan):
#     1. User approves artifact writeback.
#     2. Artifact is attached or linked in eLabFTW.
#     3. Audit ID is visible in provenance.


@pytest.fixture
def approval_store() -> ApprovalStore:
    s = ApprovalStore(db_path=":memory:")
    yield s
    s.close()


@pytest.fixture
def elabftw_stub() -> StubElabftwClient:
    """Pre-seed experiment 42 for writeback tests."""
    return StubElabftwClient(
        seeds={
            42: {
                "id": 42,
                "title": "Cloning experiment",
                "body": "<p>Original body</p>",
                "metadata": {"extra_fields": {}},
                "state": 1,
                "locked": 0,
                "uploads": [],
            },
        }
    )


@pytest.fixture
def writeback_adapter(
    policy: PolicyEngine,
    audit: AuditStore,
    approval_store: ApprovalStore,
    elabftw_stub: StubElabftwClient,
) -> OpenCloningAdapter:
    return OpenCloningAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=None,  # OpenCloning client not needed for writeback
        elabftw_client=elabftw_stub,
        approval_store=approval_store,
    )


def _issue_writeback_approval(
    store: ApprovalStore,
    *,
    args: dict,
    target_record: str | None = "42",
) -> str:
    """Issue and return an approval_id bound to the given args."""
    req = ApprovalRequest(
        tool_name="opencloning.writeback_artifact",
        args_hash=compute_args_hash(args),
        target_record=target_record,
        tier=int(Tier.BOUNDED_WRITES),
        keycloak_subject="kc-human-1",
        librechat_user_id="lc-human-1",
        mapped_elabftw_user_id="elab-human-1",
        provider="openai",
        model_id="gpt-4o-mini",
        ttl_seconds=600,
    )
    approval_id, _expires = store.request(req)
    return approval_id


# --- writeback: happy path --------------------------------------------------


def test_writeback_attaches_artifact_to_experiment(
    writeback_adapter: OpenCloningAdapter,
    elabftw_stub: StubElabftwClient,
    approval_store: ApprovalStore,
    audit: AuditStore,
) -> None:
    """Acceptance #1+#2: approved artifact is attached to eLabFTW experiment."""
    artifact_content = "LOCUS test 100 bp DNA\nORIGIN\nATCG\n//"
    approval_args = {
        "artifact_filename": "construct.gb",
        "artifact_content": artifact_content,
        "artifact_comment": "OpenCloning design artifact",
    }
    approval_id = _issue_writeback_approval(approval_store, args=approval_args)

    result = writeback_adapter.writeback_artifact(
        context_token=_token(),
        approval_id=approval_id,
        approval_args=approval_args,
        mapped_identity=_identity(),
    )
    assert result.tool_name == "opencloning.writeback_artifact"
    assert result.result["upload_id"] is not None
    assert result.result["experiment_id"] == 42
    assert result.result["artifact_filename"] == "construct.gb"

    # Verify the attachment was uploaded to the eLabFTW stub.
    uploads = elabftw_stub.seeds[42]["uploads"]
    assert len(uploads) == 1
    assert uploads[0]["real_name"] == "construct.gb"
    assert uploads[0]["comment"] == "OpenCloning design artifact"

    # Verify provenance was written to metadata.
    metadata = elabftw_stub.seeds[42]["metadata"]
    assert "extra_fields" in metadata
    assert "lab_copilot_audit_id" in metadata["extra_fields"]
    assert (
        metadata["extra_fields"]["lab_copilot_tool"] == "opencloning.writeback_artifact"
    )

    # Verify audit row.
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["policy_decision"] == "allow"
    assert rows[0]["tool_name"] == "opencloning.writeback_artifact"
    assert rows[0]["approval_id"] == approval_id
    assert rows[0]["tool_args_hash"] is not None


def test_writeback_resolves_artifact_bundle_reference(
    writeback_adapter: OpenCloningAdapter,
    elabftw_stub: StubElabftwClient,
    approval_store: ApprovalStore,
) -> None:
    """C56: approved writeback uploads exact bytes from bundle custody."""
    data = b"LOCUS bundled 100 bp DNA\nORIGIN\nATCG\n//"
    store = ArtifactBundleStore()
    manifest = store.put(
        plan_id="plan-oc-1",
        plan_hash="hash-oc-1",
        artifact_id="oc-artifact-1",
        filename="construct.gb",
        mime_type="chemical/x-genbank",
        data=data,
        binding={
            "record_type": "experiment",
            "record_id": "42",
            "mapped_elabftw_user_id": "elab-human-1",
        },
    )
    writeback_adapter.artifact_store = store
    approval_args = {
        "artifact_id": "oc-artifact-1",
        "plan_id": "plan-oc-1",
        "plan_hash": "hash-oc-1",
        "artifact_filename": "construct.gb",
        "artifact_sha256": manifest["sha256"],
        "artifact_size_bytes": manifest["size_bytes"],
        "artifact_comment": "OpenCloning bundled artifact",
    }
    approval_id = _issue_writeback_approval(approval_store, args=approval_args)

    result = writeback_adapter.writeback_artifact(
        context_token=_token(),
        approval_id=approval_id,
        approval_args=approval_args,
        mapped_identity=_identity(),
    )

    assert result.result["artifact_filename"] == "construct.gb"
    uploads = elabftw_stub.seeds[42]["uploads"]
    assert len(uploads) == 1
    assert uploads[0]["real_name"] == "construct.gb"
    assert uploads[0]["size"] == len(data)
    assert uploads[0]["comment"] == "OpenCloning bundled artifact"


def test_writeback_bundle_hash_mismatch_fails_before_upload(
    writeback_adapter: OpenCloningAdapter,
    elabftw_stub: StubElabftwClient,
    approval_store: ApprovalStore,
    audit: AuditStore,
) -> None:
    """C56: expected hash mismatch fails closed before eLabFTW upload."""
    store = ArtifactBundleStore()
    manifest = store.put(
        plan_id="plan-oc-1",
        plan_hash="hash-oc-1",
        artifact_id="oc-artifact-1",
        filename="construct.gb",
        mime_type="chemical/x-genbank",
        data=b"LOCUS bundled\n//",
        binding={
            "record_type": "experiment",
            "record_id": "42",
            "mapped_elabftw_user_id": "elab-human-1",
        },
    )
    writeback_adapter.artifact_store = store
    approval_args = {
        "artifact_id": "oc-artifact-1",
        "plan_id": "plan-oc-1",
        "plan_hash": "hash-oc-1",
        "artifact_filename": "construct.gb",
        "artifact_sha256": "0" * 64,
        "artifact_size_bytes": manifest["size_bytes"],
    }
    approval_id = _issue_writeback_approval(approval_store, args=approval_args)

    with pytest.raises(OpenCloningAdapterError) as exc_info:
        writeback_adapter.writeback_artifact(
            context_token=_token(),
            approval_id=approval_id,
            approval_args=approval_args,
            mapped_identity=_identity(),
        )

    assert exc_info.value.reason == "artifact_bundle_error"
    assert elabftw_stub.seeds[42]["uploads"] == []
    rows = _audit_rows(audit)
    assert rows[0]["policy_decision"] == "deny"
    assert rows[0]["error"]["code"] == "ARTIFACT_BUNDLE_ERROR"


def test_writeback_missing_payload_fails_closed(
    writeback_adapter: OpenCloningAdapter,
) -> None:
    """C56: filename-only writeback must not upload an empty artifact."""
    with pytest.raises(OpenCloningAdapterError) as exc_info:
        writeback_adapter.writeback_artifact(
            context_token=_token(),
            approval_id="approval-id",
            approval_args={"artifact_filename": "construct.gb"},
            mapped_identity=_identity(),
        )

    assert exc_info.value.reason == "missing_artifact_payload"


# --- writeback: approval consumed ------------------------------------------


def test_writeback_consumes_approval_token(
    writeback_adapter: OpenCloningAdapter,
    approval_store: ApprovalStore,
) -> None:
    """Approval token is consumed (single-use) after writeback."""
    approval_args = {
        "artifact_filename": "construct.gb",
        "artifact_content": "LOCUS test\n",
    }
    approval_id = _issue_writeback_approval(approval_store, args=approval_args)

    writeback_adapter.writeback_artifact(
        context_token=_token(),
        approval_id=approval_id,
        approval_args=approval_args,
        mapped_identity=_identity(),
    )

    # The approval token should now be consumed.
    record = approval_store.get(approval_id)
    assert record is not None
    assert record.consumed_at is not None


# --- writeback: no approval ------------------------------------------------


def test_writeback_without_approval_denies(
    adapter: OpenCloningAdapter,
    audit: AuditStore,
) -> None:
    """Writeback without approval token is denied by policy."""
    # This adapter has no approval_store and no elabftw_client.
    with pytest.raises(PolicyDenied):
        adapter.writeback_artifact(
            context_token=_token(),
            approval_id="",
            approval_args={"artifact_filename": "x.gb", "artifact_content": "x"},
            mapped_identity=_identity(),
        )
    rows = _audit_rows(audit)
    assert len(rows) == 1
    assert rows[0]["policy_decision"] == "deny"


# --- writeback: eLabFTW not configured --------------------------------------


def test_writeback_without_elabftw_client_denies(
    policy: PolicyEngine,
    audit: AuditStore,
    approval_store: ApprovalStore,
) -> None:
    """Writeback without eLabFTW client raises structured error."""
    adapter = OpenCloningAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=None,
        elabftw_client=None,
        approval_store=approval_store,
    )
    approval_args = {
        "artifact_filename": "construct.gb",
        "artifact_content": "LOCUS test\n",
    }
    approval_id = _issue_writeback_approval(approval_store, args=approval_args)
    with pytest.raises(OpenCloningAdapterError) as exc_info:
        adapter.writeback_artifact(
            context_token=_token(),
            approval_id=approval_id,
            approval_args=approval_args,
            mapped_identity=_identity(),
        )
    assert exc_info.value.reason == "elabftw_not_configured"
    rows = _audit_rows(audit)
    assert rows[0]["error"]["code"] == "ELABFTW_NOT_CONFIGURED"


# --- writeback: file-size limit --------------------------------------------


def test_writeback_rejects_oversized_artifact(
    writeback_adapter: OpenCloningAdapter,
    approval_store: ApprovalStore,
    audit: AuditStore,
) -> None:
    """Oversized artifact is rejected before upload."""
    oversized = "A" * (MAX_FILE_SIZE_BYTES + 1)
    approval_args = {
        "artifact_filename": "huge.gb",
        "artifact_content": oversized,
    }
    approval_id = _issue_writeback_approval(approval_store, args=approval_args)
    with pytest.raises(FileTooLarge):
        writeback_adapter.writeback_artifact(
            context_token=_token(),
            approval_id=approval_id,
            approval_args=approval_args,
            mapped_identity=_identity(),
        )
    rows = _audit_rows(audit)
    assert rows[0]["error"]["code"] == "FILE_TOO_LARGE"


# --- writeback: unmapped caller --------------------------------------------


def test_writeback_unmapped_caller_denies(
    writeback_adapter: OpenCloningAdapter,
    approval_store: ApprovalStore,
    audit: AuditStore,
) -> None:
    """Unmapped caller denies before any upload."""
    approval_args = {
        "artifact_filename": "construct.gb",
        "artifact_content": "LOCUS test\n",
    }
    approval_id = _issue_writeback_approval(approval_store, args=approval_args)
    with pytest.raises(UnmappedCaller):
        writeback_adapter.writeback_artifact(
            context_token=_token(),
            approval_id=approval_id,
            approval_args=approval_args,
            mapped_identity=None,
        )
    rows = _audit_rows(audit)
    assert rows[0]["error"]["code"] == "UNMAPPED_CALLER"


# --- writeback: audit ID in provenance --------------------------------------


def test_writeback_audit_id_in_provenance(
    writeback_adapter: OpenCloningAdapter,
    elabftw_stub: StubElabftwClient,
    approval_store: ApprovalStore,
) -> None:
    """Acceptance #3: audit ID is visible in experiment metadata provenance."""
    approval_args = {
        "artifact_filename": "construct.gb",
        "artifact_content": "LOCUS test\n",
    }
    approval_id = _issue_writeback_approval(approval_store, args=approval_args)

    result = writeback_adapter.writeback_artifact(
        context_token=_token(),
        approval_id=approval_id,
        approval_args=approval_args,
        mapped_identity=_identity(),
    )

    # The audit_action_id should be in the experiment's metadata.
    metadata = elabftw_stub.seeds[42]["metadata"]
    assert metadata["extra_fields"]["lab_copilot_audit_id"] == result.audit_action_id


# --- writeback: client error ------------------------------------------------


def test_writeback_client_error_propagates(
    policy: PolicyEngine,
    audit: AuditStore,
    approval_store: ApprovalStore,
) -> None:
    """eLabFTW upload error is structured and audited."""
    # Use a stub with no seeds — upload will raise KeyError.
    elabftw_stub = StubElabftwClient(seeds={})
    adapter = OpenCloningAdapter(
        policy_engine=policy,
        audit_store=audit,
        client=None,
        elabftw_client=elabftw_stub,
        approval_store=approval_store,
    )
    approval_args = {
        "artifact_filename": "construct.gb",
        "artifact_content": "LOCUS test\n",
    }
    approval_id = _issue_writeback_approval(approval_store, args=approval_args)
    with pytest.raises(OpenCloningAdapterError) as exc_info:
        adapter.writeback_artifact(
            context_token=_token(),
            approval_id=approval_id,
            approval_args=approval_args,
            mapped_identity=_identity(),
        )
    assert exc_info.value.reason == "client_error"
    rows = _audit_rows(audit)
    assert rows[0]["error"]["code"] == "CLIENT_ERROR"
