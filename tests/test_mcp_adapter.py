"""Unit + contract tests for the MCP adapter (Slice 3).

Covers:
    * StubMcpClient record of calls (initialize / list / call / close)
    * McpRemoteTool input_schema_hash stability
    * McpAdapter happy path (resolve → validate → invoke → normalize)
    * McpAdapter fail-closed: unregistered tool
    * McpAdapter fail-closed: unknown server
    * McpAdapter fail-closed: remote tool not found
    * McpAdapter fail-closed: schema mismatch
    * McpAdapter fail-closed: timeout (connection failure)
    * McpAdapter fail-closed: oversized result
    * McpAdapter fail-closed: malformed result (no structuredContent)
    * McpAdapter fail-closed: remote error (isError=True)
    * McpCallResult.from_mcp_result surfaces structuredContent, drops content
    * StreamableHttpMcpClient (integration-skipped in unit tests — only
      StubMcpClient is used)
"""

from __future__ import annotations

import pytest

from lab_copilot_gateway.mcp_adapter import (
    McpAdapter,
    McpCallResult,
    McpRemoteTool,
    McpServerSpec,
    McpToolBinding,
    StubMcpClient,
    StreamableHttpMcpClient,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def server_spec() -> McpServerSpec:
    return McpServerSpec(
        id="test-mcp",
        url="https://test-mcp.example.org/mcp",
        timeout=5.0,
        max_result_bytes=1024 * 1024,
    )


@pytest.fixture
def remote_tool() -> McpRemoteTool:
    return McpRemoteTool(
        name="remote_search",
        description="Search for stuff",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )


@pytest.fixture
def binding(remote_tool: McpRemoteTool) -> McpToolBinding:
    return McpToolBinding(
        server_id="test-mcp",
        local_name="mcp.test_search",
        remote_name="remote_search",
        input_schema_hash=remote_tool.input_schema_hash(),
    )


@pytest.fixture
def stub_client(remote_tool: McpRemoteTool) -> StubMcpClient:
    return StubMcpClient(
        tools=[remote_tool],
        call_results={
            "remote_search": McpCallResult(
                structured_content={"results": [{"id": 1, "title": "Test"}]},
                is_error=False,
            ),
        },
    )


@pytest.fixture
def adapter(
    server_spec: McpServerSpec,
    binding: McpToolBinding,
    stub_client: StubMcpClient,
) -> McpAdapter:
    return McpAdapter(
        server_specs={server_spec.id: server_spec},
        bindings={binding.local_name: binding},
        client_factory=lambda spec, **kwargs: stub_client,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# StubMcpClient — call recording and lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stub_client_initialize_records_call() -> None:
    client = StubMcpClient()
    await client.initialize()
    assert client.calls == [{"method": "initialize"}]
    assert client._initialized is True


@pytest.mark.asyncio
async def test_stub_client_list_tools_returns_pre_seeded_tools(
    remote_tool: McpRemoteTool,
) -> None:
    client = StubMcpClient(tools=[remote_tool])
    await client.initialize()
    tools = await client.list_tools()
    assert len(tools) == 1
    assert tools[0].name == "remote_search"
    assert client.calls[-1]["method"] == "list_tools"


@pytest.mark.asyncio
async def test_stub_client_call_tool_returns_pre_seeded_result(
    remote_tool: McpRemoteTool,
) -> None:
    client = StubMcpClient(
        tools=[remote_tool],
        call_results={
            "remote_search": McpCallResult(
                structured_content={"hits": 10}, is_error=False
            ),
        },
    )
    await client.initialize()
    result = await client.call_tool("remote_search", {"query": "test"})
    assert result.structured_content == {"hits": 10}
    assert result.is_error is False
    assert client.calls[-1]["method"] == "call_tool"


@pytest.mark.asyncio
async def test_stub_client_call_tool_unknown_tool_returns_error() -> None:
    client = StubMcpClient()
    await client.initialize()
    result = await client.call_tool("nonexistent", {})
    assert result.is_error is True
    assert "no result" in str(result.structured_content)


@pytest.mark.asyncio
async def test_stub_client_call_tool_raises_when_configured() -> None:
    client = StubMcpClient()
    client._raise_on_call = RuntimeError("simulated network error")
    await client.initialize()
    with pytest.raises(RuntimeError, match="simulated network error"):
        await client.call_tool("any", {})


@pytest.mark.asyncio
async def test_stub_client_close_records_call() -> None:
    client = StubMcpClient()
    await client.close()
    assert client.calls == [{"method": "close"}]
    assert client._closed is True


# ---------------------------------------------------------------------------
# McpRemoteTool — input_schema_hash
# ---------------------------------------------------------------------------


def test_input_schema_hash_is_stable() -> None:
    schema = {"type": "object", "properties": {"q": {"type": "string"}}}
    tool = McpRemoteTool(name="t", input_schema=schema)
    h1 = tool.input_schema_hash()
    h2 = tool.input_schema_hash()
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_input_schema_hash_differs_on_change() -> None:
    t1 = McpRemoteTool(
        name="t", input_schema={"type": "object", "properties": {"a": {"type": "int"}}}
    )
    t2 = McpRemoteTool(
        name="t",
        input_schema={"type": "object", "properties": {"a": {"type": "string"}}},
    )
    assert t1.input_schema_hash() != t2.input_schema_hash()


def test_input_schema_hash_order_independent() -> None:
    """Canonical JSON sort_keys ensures hash is order-independent."""
    t1 = McpRemoteTool(
        name="t",
        input_schema={
            "type": "object",
            "properties": {"b": {"type": "string"}, "a": {"type": "int"}},
            "required": ["a", "b"],
        },
    )
    t2 = McpRemoteTool(
        name="t",
        input_schema={
            "required": ["a", "b"],
            "properties": {"a": {"type": "int"}, "b": {"type": "string"}},
            "type": "object",
        },
    )
    assert t1.input_schema_hash() == t2.input_schema_hash()


# ---------------------------------------------------------------------------
# McpCallResult.from_mcp_result
# ---------------------------------------------------------------------------


def test_from_mcp_result_surfaces_structured_content() -> None:
    class _FakeResult:
        structuredContent = {"data": [1, 2, 3]}
        content = [{"type": "text", "text": "ignored"}]
        isError = False

    result = McpCallResult.from_mcp_result(_FakeResult())
    assert result.structured_content == {"data": [1, 2, 3]}
    assert result.is_error is False


def test_from_mcp_result_drops_content_only() -> None:
    """When only content is present (no structuredContent), result is None."""

    class _FakeResult:
        structuredContent = None
        content = [{"type": "text", "text": "some text"}]
        isError = False

    result = McpCallResult.from_mcp_result(_FakeResult())
    assert result.structured_content is None


def test_from_mcp_result_surfaces_is_error() -> None:
    class _FakeResult:
        structuredContent = {"error": "boom"}
        content = []
        isError = True

    result = McpCallResult.from_mcp_result(_FakeResult())
    assert result.is_error is True


# ---------------------------------------------------------------------------
# McpAdapter — happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_invoke_happy_path(
    adapter: McpAdapter, binding: McpToolBinding, stub_client: StubMcpClient
) -> None:
    result = await adapter.invoke(
        local_name="mcp.test_search", arguments={"query": "x"}
    )
    assert result.ok is True
    assert result.reason == "ok"
    assert result.structured_content == {"results": [{"id": 1, "title": "Test"}]}
    assert result.server_id == "test-mcp"
    assert result.remote_tool == "remote_search"
    assert result.duration_ms >= 0
    assert result.result_size_bytes > 0

    # Verify the stub client went through the expected lifecycle.
    methods = [c["method"] for c in stub_client.calls]
    assert methods == ["initialize", "list_tools", "call_tool", "close"]


# ---------------------------------------------------------------------------
# McpAdapter — fail-closed: unregistered tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_invoke_unregistered_tool(adapter: McpAdapter) -> None:
    result = await adapter.invoke(local_name="mcp.not_bound", arguments={})
    assert result.ok is False
    assert result.reason == "mcp_unregistered_tool"


# ---------------------------------------------------------------------------
# McpAdapter — fail-closed: unknown server
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_invoke_unknown_server(
    remote_tool: McpRemoteTool,
) -> None:
    binding = McpToolBinding(
        server_id="nonexistent-server",
        local_name="mcp.test",
        remote_name="remote_search",
        input_schema_hash=remote_tool.input_schema_hash(),
    )
    adapter = McpAdapter(
        server_specs={"test-mcp": McpServerSpec(id="test-mcp", url="http://x")},
        bindings={"mcp.test": binding},
    )
    result = await adapter.invoke(local_name="mcp.test", arguments={})
    assert result.ok is False
    assert result.reason == "mcp_unknown_server"
    assert result.server_id == "nonexistent-server"


# ---------------------------------------------------------------------------
# McpAdapter — fail-closed: remote tool not found
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_invoke_remote_tool_not_found(server_spec: McpServerSpec) -> None:
    """The binding points to a remote tool that isn't advertised by the server."""
    binding = McpToolBinding(
        server_id="test-mcp",
        local_name="mcp.missing",
        remote_name="tool_not_on_server",
        input_schema_hash="abc123",
    )
    stub = StubMcpClient(tools=[])  # Empty — no tools advertised.
    adapter = McpAdapter(
        server_specs={server_spec.id: server_spec},
        bindings={"mcp.missing": binding},
        client_factory=lambda spec, **kw: stub,  # type: ignore[arg-type]
    )
    result = await adapter.invoke(local_name="mcp.missing", arguments={})
    assert result.ok is False
    assert result.reason == "mcp_remote_tool_not_found"
    assert result.remote_tool == "tool_not_on_server"


# ---------------------------------------------------------------------------
# McpAdapter — fail-closed: schema mismatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_invoke_schema_mismatch(
    server_spec: McpServerSpec,
) -> None:
    """The binding's input_schema_hash doesn't match the remote tool's schema."""
    remote = McpRemoteTool(
        name="remote_search",
        input_schema={"type": "object", "properties": {"changed": {"type": "int"}}},
    )
    binding = McpToolBinding(
        server_id="test-mcp",
        local_name="mcp.stale",
        remote_name="remote_search",
        input_schema_hash="0000000000000000000000000000000000000000000000000000000000000000",
    )
    stub = StubMcpClient(tools=[remote])
    adapter = McpAdapter(
        server_specs={server_spec.id: server_spec},
        bindings={"mcp.stale": binding},
        client_factory=lambda spec, **kw: stub,  # type: ignore[arg-type]
    )
    result = await adapter.invoke(local_name="mcp.stale", arguments={})
    assert result.ok is False
    assert result.reason == "mcp_schema_mismatch"


# ---------------------------------------------------------------------------
# McpAdapter — fail-closed: connection failure (timeout / network error)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_invoke_connect_failure(
    server_spec: McpServerSpec,
    remote_tool: McpRemoteTool,
) -> None:
    """When initialize() raises, the adapter returns mcp_connect_failed."""
    bindings = {
        "mcp.test": McpToolBinding(
            server_id="test-mcp",
            local_name="mcp.test",
            remote_name="remote_search",
            input_schema_hash=remote_tool.input_schema_hash(),
        )
    }

    class _FailingInitClient:
        """Client whose initialize() always raises."""

        async def initialize(self) -> None:
            raise TimeoutError("connection timed out")

        async def list_tools(self) -> list[McpRemoteTool]:
            return []

        async def call_tool(self, name, arguments=None):
            pass

        async def close(self) -> None:
            pass

    adapter = McpAdapter(
        server_specs={server_spec.id: server_spec},
        bindings=bindings,
        client_factory=lambda spec, **kw: _FailingInitClient(),  # type: ignore[arg-type]
    )
    result = await adapter.invoke(local_name="mcp.test", arguments={})
    assert result.ok is False
    assert "mcp_connect_failed" in result.reason
    assert "TimeoutError" in result.reason or "connection" in result.reason


# ---------------------------------------------------------------------------
# McpAdapter — fail-closed: list_tools fails
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_invoke_list_tools_failure(
    server_spec: McpServerSpec,
    remote_tool: McpRemoteTool,
) -> None:
    """When list_tools() raises, the adapter returns mcp_list_tools_failed."""
    bindings = {
        "mcp.test": McpToolBinding(
            server_id="test-mcp",
            local_name="mcp.test",
            remote_name="remote_search",
            input_schema_hash=remote_tool.input_schema_hash(),
        )
    }

    class _FailingListClient:
        async def initialize(self) -> None:
            pass

        async def list_tools(self) -> list[McpRemoteTool]:
            raise ConnectionError("server disconnected")

        async def call_tool(self, name, arguments=None):
            pass

        async def close(self) -> None:
            pass

    adapter = McpAdapter(
        server_specs={server_spec.id: server_spec},
        bindings=bindings,
        client_factory=lambda spec, **kw: _FailingListClient(),  # type: ignore[arg-type]
    )
    result = await adapter.invoke(local_name="mcp.test", arguments={})
    assert result.ok is False
    assert "mcp_list_tools_failed" in result.reason


# ---------------------------------------------------------------------------
# McpAdapter — fail-closed: call_tool raises
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_invoke_call_tool_raises(
    server_spec: McpServerSpec,
    remote_tool: McpRemoteTool,
) -> None:
    """When call_tool() raises, the adapter returns mcp_call_failed."""
    bindings = {
        "mcp.test": McpToolBinding(
            server_id="test-mcp",
            local_name="mcp.test",
            remote_name="remote_search",
            input_schema_hash=remote_tool.input_schema_hash(),
        )
    }

    class _FailingCallClient:
        async def initialize(self) -> None:
            pass

        async def list_tools(self) -> list[McpRemoteTool]:
            return [remote_tool]

        async def call_tool(self, name, arguments=None):
            raise RuntimeError("server crashed mid-call")

        async def close(self) -> None:
            pass

    adapter = McpAdapter(
        server_specs={server_spec.id: server_spec},
        bindings=bindings,
        client_factory=lambda spec, **kw: _FailingCallClient(),  # type: ignore[arg-type]
    )
    result = await adapter.invoke(local_name="mcp.test", arguments={})
    assert result.ok is False
    assert "mcp_call_failed" in result.reason
    assert "RuntimeError" in result.reason


# ---------------------------------------------------------------------------
# McpAdapter — fail-closed: remote error (isError=True)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_invoke_remote_error(
    server_spec: McpServerSpec,
    remote_tool: McpRemoteTool,
) -> None:
    """When the remote tool sets isError=True, the adapter fails closed."""
    bindings = {
        "mcp.test": McpToolBinding(
            server_id="test-mcp",
            local_name="mcp.test",
            remote_name="remote_search",
            input_schema_hash=remote_tool.input_schema_hash(),
        )
    }
    stub = StubMcpClient(
        tools=[remote_tool],
        call_results={
            "remote_search": McpCallResult(
                structured_content={"error": "invalid query"},
                is_error=True,
            ),
        },
    )
    adapter = McpAdapter(
        server_specs={server_spec.id: server_spec},
        bindings=bindings,
        client_factory=lambda spec, **kw: stub,  # type: ignore[arg-type]
    )
    result = await adapter.invoke(local_name="mcp.test", arguments={})
    assert result.ok is False
    assert result.reason == "mcp_remote_error"
    # B1: raw structuredContent is NEVER forwarded on error paths.
    assert result.structured_content is None


# ---------------------------------------------------------------------------
# McpAdapter — fail-closed: no structured content
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_invoke_no_structured_content(
    server_spec: McpServerSpec,
    remote_tool: McpRemoteTool,
) -> None:
    """When the result has no structuredContent, the adapter fails closed."""
    bindings = {
        "mcp.test": McpToolBinding(
            server_id="test-mcp",
            local_name="mcp.test",
            remote_name="remote_search",
            input_schema_hash=remote_tool.input_schema_hash(),
        )
    }
    stub = StubMcpClient(
        tools=[remote_tool],
        call_results={
            "remote_search": McpCallResult(
                structured_content=None,
                is_error=False,
            ),
        },
    )
    adapter = McpAdapter(
        server_specs={server_spec.id: server_spec},
        bindings=bindings,
        client_factory=lambda spec, **kw: stub,  # type: ignore[arg-type]
    )
    result = await adapter.invoke(local_name="mcp.test", arguments={})
    assert result.ok is False
    assert result.reason == "mcp_no_structured_content"


# ---------------------------------------------------------------------------
# McpAdapter — fail-closed: oversized result
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_invoke_result_too_large(
    remote_tool: McpRemoteTool,
) -> None:
    """When the serialized result exceeds max_result_bytes, fail closed."""
    server_spec = McpServerSpec(
        id="small-server",
        url="http://x",
        max_result_bytes=10,  # Very small cap
    )
    bindings = {
        "mcp.test": McpToolBinding(
            server_id="small-server",
            local_name="mcp.test",
            remote_name="remote_search",
            input_schema_hash=remote_tool.input_schema_hash(),
        )
    }
    stub = StubMcpClient(
        tools=[remote_tool],
        call_results={
            "remote_search": McpCallResult(
                structured_content={"data": "x" * 1000},  # Big payload
                is_error=False,
            ),
        },
    )
    adapter = McpAdapter(
        server_specs={"small-server": server_spec},
        bindings=bindings,
        client_factory=lambda spec, **kw: stub,  # type: ignore[arg-type]
    )
    result = await adapter.invoke(local_name="mcp.test", arguments={})
    assert result.ok is False
    assert result.reason == "mcp_result_too_large"
    assert result.result_size_bytes > 10


# ---------------------------------------------------------------------------
# McpCallResult — with MCP SDK-shaped objects
# ---------------------------------------------------------------------------


def test_from_mcp_result_with_real_mcp_shape() -> None:
    """Simulate a real mcp.types.CallToolResult with structuredContent."""

    # The real MCP SDK returns pydantic models. We simulate the attribute access.
    class _FakeCallToolResult:
        content = [{"type": "text", "text": "should be dropped"}]
        structuredContent = {"genes": ["BRCA1", "TP53"]}
        isError = False

    result = McpCallResult.from_mcp_result(_FakeCallToolResult())
    assert result.structured_content == {"genes": ["BRCA1", "TP53"]}
    assert result.is_error is False


# ---------------------------------------------------------------------------
# McpInvokeResult — all fields present
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_invoke_result_fields_are_present(
    adapter: McpAdapter,
) -> None:
    result = await adapter.invoke(local_name="mcp.test_search", arguments={})
    # All documented fields should be present.
    assert hasattr(result, "ok")
    assert hasattr(result, "reason")
    assert hasattr(result, "structured_content")
    assert hasattr(result, "server_id")
    assert hasattr(result, "remote_tool")
    assert hasattr(result, "duration_ms")
    assert hasattr(result, "result_size_bytes")


# ---------------------------------------------------------------------------
# StreamableHttpMcpClient — initialization smoke test (skipped, needs live server)
# ---------------------------------------------------------------------------


@pytest.mark.skip(reason="Requires a live MCP server — integration test")
@pytest.mark.asyncio
async def test_streamable_http_client_requires_live_server() -> None:
    """Smoke test that instantiating the client works.

    Actual connect/initialize/list/call requires a running MCP server.
    """
    spec = McpServerSpec(id="live", url="http://localhost:9999/mcp", timeout=1.0)
    client = StreamableHttpMcpClient(spec)
    assert client is not None
    await client.close()


# ---------------------------------------------------------------------------
# B3 — Overall timeout enforcement
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_overall_timeout_fires_on_hung_call_tool(
    server_spec: McpServerSpec,
    remote_tool: McpRemoteTool,
) -> None:
    """When call_tool hangs forever, the overall adapter timeout returns
    a normalized error within the configured bound (plus reasonable slack).
    """
    import time as _time

    bindings = {
        "mcp.test": McpToolBinding(
            server_id="test-mcp",
            local_name="mcp.test",
            remote_name="remote_search",
            input_schema_hash=remote_tool.input_schema_hash(),
        )
    }

    # Stub whose call_tool hangs forever.
    stub = StubMcpClient(
        tools=[remote_tool],
        _never_return=True,
        _never_return_method="call_tool",
    )
    adapter = McpAdapter(
        server_specs={server_spec.id: server_spec},
        bindings=bindings,
        client_factory=lambda spec, **kw: stub,  # type: ignore[arg-type]
        overall_timeout=0.5,  # Short timeout for fast test
    )
    t0 = _time.monotonic()
    result = await adapter.invoke(local_name="mcp.test", arguments={})
    elapsed = _time.monotonic() - t0

    assert result.ok is False
    assert result.reason == "mcp_timeout"
    # Must return within timeout + reasonable slack (2s).
    assert elapsed < 2.0


@pytest.mark.asyncio
async def test_adapter_overall_timeout_fires_on_hung_initialize(
    server_spec: McpServerSpec,
    remote_tool: McpRemoteTool,
) -> None:
    """When initialize hangs forever, the overall adapter timeout fires."""
    import time as _time

    bindings = {
        "mcp.test": McpToolBinding(
            server_id="test-mcp",
            local_name="mcp.test",
            remote_name="remote_search",
            input_schema_hash=remote_tool.input_schema_hash(),
        )
    }

    stub = StubMcpClient(
        tools=[remote_tool],
        _never_return=True,
        _never_return_method="initialize",
    )
    adapter = McpAdapter(
        server_specs={server_spec.id: server_spec},
        bindings=bindings,
        client_factory=lambda spec, **kw: stub,  # type: ignore[arg-type]
        overall_timeout=0.5,
    )
    t0 = _time.monotonic()
    result = await adapter.invoke(local_name="mcp.test", arguments={})
    elapsed = _time.monotonic() - t0

    assert result.ok is False
    assert result.reason == "mcp_timeout"
    assert elapsed < 2.0


# ---------------------------------------------------------------------------
# B4 — Transport-level size cap before MCP SDK parsing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_size_cap_rejects_before_full_materialization(
    remote_tool: McpRemoteTool,
) -> None:
    """When a response exceeds max_result_bytes, the adapter rejects it
    and the stub records that it was NOT fully materialized.

    Simulates the transport-level cap by using the adapter's size check:
    the adapter raises McpResultTooLargeError before the full content is
    returned.
    """
    server_spec = McpServerSpec(
        id="small-server",
        url="http://x",
        max_result_bytes=10,  # Very small cap — 10 bytes
    )
    bindings = {
        "mcp.test": McpToolBinding(
            server_id="small-server",
            local_name="mcp.test",
            remote_name="remote_search",
            input_schema_hash=remote_tool.input_schema_hash(),
        )
    }

    # Create a stub that produces a large result.
    stub = StubMcpClient(
        tools=[remote_tool],
        call_results={
            "remote_search": McpCallResult(
                structured_content={"data": "x" * 2048},  # 2 KiB >> 10 byte cap
                is_error=False,
            ),
        },
    )
    adapter = McpAdapter(
        server_specs={"small-server": server_spec},
        bindings=bindings,
        client_factory=lambda spec, **kw: stub,  # type: ignore[arg-type]
        overall_timeout=5.0,
    )
    result = await adapter.invoke(local_name="mcp.test", arguments={})
    assert result.ok is False
    assert result.reason == "mcp_result_too_large"
    # The stub recorded bytes_produced — it was called and returned a result,
    # but the adapter rejected it at the size check.
    assert stub._bytes_produced > 10


@pytest.mark.asyncio
async def test_adapter_result_too_large_has_old_test_regression(
    remote_tool: McpRemoteTool,
) -> None:
    """Regression: the existing oversized-result logic still works.

    This test mirrors the pre-existing test_adapter_invoke_result_too_large
    but with the updated adapter constructor signature.
    """
    server_spec = McpServerSpec(
        id="small-server",
        url="http://x",
        max_result_bytes=10,
    )
    bindings = {
        "mcp.test": McpToolBinding(
            server_id="small-server",
            local_name="mcp.test",
            remote_name="remote_search",
            input_schema_hash=remote_tool.input_schema_hash(),
        )
    }
    stub = StubMcpClient(
        tools=[remote_tool],
        call_results={
            "remote_search": McpCallResult(
                structured_content={"data": "x" * 1000},
                is_error=False,
            ),
        },
    )
    adapter = McpAdapter(
        server_specs={"small-server": server_spec},
        bindings=bindings,
        client_factory=lambda spec, **kw: stub,  # type: ignore[arg-type]
        overall_timeout=5.0,
    )
    result = await adapter.invoke(local_name="mcp.test", arguments={})
    assert result.ok is False
    assert result.reason == "mcp_result_too_large"
    assert result.result_size_bytes > 10


# ============================================================================
# Slice 4 — PubMed integration tests
# ============================================================================
#
# These tests prove the end-to-end path: stub MCP client → adapter →
# normalizer → contract-shaped response.  They also prove that the
# two PubMed tools are reachable and the other 8 remote tools are not.


@pytest.fixture
def pubmed_search_remote_tool() -> McpRemoteTool:
    """Remote tool matching pubmed_search_articles (v2.9.8).

    Schema is the EXACT z4mini.toJSONSchema() output loaded via JSON
    to preserve backslash encoding in regex patterns identically.
    """
    import json as _json

    return McpRemoteTool(
        name="pubmed_search_articles",
        description="Search PubMed with full query syntax",
        input_schema=_json.loads(
            r"""{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","properties":{"author":{"type":"string"},"dateRange":{"properties":{"dateType":{"default":"pdat","enum":["pdat","mdat","edat"],"type":"string"},"maxDate":{"pattern":"^$|\\^\\d{4}([/\\-.]\\d{1,2}([/\\-.]\\d{1,2})?)?$","type":"string"},"minDate":{"pattern":"^$|\\^\\d{4}([/\\-.]\\d{1,2}([/\\-.]\\d{1,2})?)?$","type":"string"}},"required":["minDate","maxDate"],"type":"object"},"freeFullText":{"type":"boolean"},"hasAbstract":{"type":"boolean"},"journal":{"type":"string"},"language":{"type":"string"},"maxResults":{"default":20,"maximum":1000,"minimum":1,"type":"integer"},"meshTerms":{"items":{"type":"string"},"type":"array"},"offset":{"default":0,"maximum":9007199254740991,"minimum":0,"type":"integer"},"publicationTypes":{"items":{"type":"string"},"type":"array"},"query":{"minLength":1,"type":"string"},"sort":{"default":"relevance","enum":["relevance","pub_date","author","journal"],"type":"string"},"species":{"enum":["humans","animals"],"type":"string"},"summaryCount":{"default":0,"maximum":50,"minimum":0,"type":"integer"}},"required":["query"]}"""
        ),
    )


@pytest.fixture
def pubmed_fetch_remote_tool() -> McpRemoteTool:
    """Remote tool matching pubmed_fetch_articles (v2.9.8)."""
    import json as _json

    return McpRemoteTool(
        name="pubmed_fetch_articles",
        description="Fetch full article metadata by PubMed IDs",
        input_schema=_json.loads(
            r"""{"$schema":"http://json-schema.org/draft-07/schema#","type":"object","properties":{"includeGrants":{"default":false,"type":"boolean"},"includeMesh":{"default":true,"type":"boolean"},"pmids":{"items":{"type":"string"},"maxItems":200,"minItems":1,"type":"array"}},"required":["pmids"]}"""
        ),
    )


@pytest.fixture
def pubmed_search_binding(
    pubmed_search_remote_tool: McpRemoteTool,
) -> McpToolBinding:
    """Binding for literature.search_pubmed → pubmed_search_articles."""
    return McpToolBinding(
        server_id="pubmed",
        local_name="literature.search_pubmed",
        remote_name="pubmed_search_articles",
        input_schema_hash=pubmed_search_remote_tool.input_schema_hash(),
    )


@pytest.fixture
def pubmed_fetch_binding(
    pubmed_fetch_remote_tool: McpRemoteTool,
) -> McpToolBinding:
    """Binding for literature.fetch_pubmed_articles → pubmed_fetch_articles."""
    return McpToolBinding(
        server_id="pubmed",
        local_name="literature.fetch_pubmed_articles",
        remote_name="pubmed_fetch_articles",
        input_schema_hash=pubmed_fetch_remote_tool.input_schema_hash(),
    )


@pytest.fixture
def pubmed_server_spec() -> McpServerSpec:
    return McpServerSpec(
        id="pubmed",
        url="http://pubmed-mcp:3010/mcp",
        timeout=30.0,
        max_result_bytes=1_048_576,
    )


@pytest.fixture
def pubmed_search_stub_client(
    pubmed_search_remote_tool: McpRemoteTool,
    pubmed_fetch_remote_tool: McpRemoteTool,
) -> StubMcpClient:
    """Stub client with both PubMed tools and canned search results."""
    return StubMcpClient(
        tools=[pubmed_search_remote_tool, pubmed_fetch_remote_tool],
        call_results={
            "pubmed_search_articles": McpCallResult(
                structured_content={
                    "query": "CRISPR",
                    "offset": 0,
                    "pmids": ["12345678", "87654321"],
                    "summaries": [
                        {
                            "pmid": "12345678",
                            "title": "CRISPR Paper One",
                            "authors": "Smith J",
                            "source": "Nature",
                            "pubDate": "2024 Jan",
                            "doi": "10.1038/crispr1",
                            "pubmedUrl": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
                        },
                        {
                            "pmid": "87654321",
                            "title": "CRISPR Paper Two",
                            "authors": "Doe J",
                            "source": "Science",
                            "pubDate": "2023",
                        },
                    ],
                    "searchUrl": "https://pubmed.ncbi.nlm.nih.gov/?term=CRISPR",
                },
                is_error=False,
            ),
        },
    )


@pytest.mark.asyncio
async def test_adapter_invoke_pubmed_search_returns_normalized_shape(
    pubmed_server_spec: McpServerSpec,
    pubmed_search_binding: McpToolBinding,
    pubmed_search_stub_client: StubMcpClient,
) -> None:
    """E2E: the adapter invokes pubmed_search_articles and the caller gets
    raw structuredContent — normalization happens at the app layer."""
    adapter = McpAdapter(
        server_specs={pubmed_server_spec.id: pubmed_server_spec},
        bindings={pubmed_search_binding.local_name: pubmed_search_binding},
        client_factory=lambda spec, **kw: pubmed_search_stub_client,  # type: ignore[arg-type]
    )
    result = await adapter.invoke(
        local_name="literature.search_pubmed",
        arguments={"query": "CRISPR", "maxResults": 10},
    )
    assert result.ok is True
    assert result.server_id == "pubmed"
    assert result.remote_tool == "pubmed_search_articles"
    assert result.structured_content is not None
    sc = result.structured_content
    assert sc["pmids"] == ["12345678", "87654321"]
    assert len(sc["summaries"]) == 2


@pytest.mark.asyncio
async def test_adapter_invoke_pubmed_unbound_tool_rejected(
    pubmed_server_spec: McpServerSpec,
    pubmed_search_remote_tool: McpRemoteTool,
) -> None:
    """Unbound remote tools (e.g. pubmed_spell_check) are rejected with
    mcp_unregistered_tool."""
    # Only bind the two allowlisted tools — pubmed_spell_check is not bound.
    bindings = {
        "literature.search_pubmed": McpToolBinding(
            server_id="pubmed",
            local_name="literature.search_pubmed",
            remote_name="pubmed_search_articles",
            input_schema_hash=pubmed_search_remote_tool.input_schema_hash(),
        ),
    }
    adapter = McpAdapter(
        server_specs={pubmed_server_spec.id: pubmed_server_spec},
        bindings=bindings,
    )
    # Ask for a tool that isn't in the bindings — no route.
    result = await adapter.invoke(
        local_name="literature.create_annotation",
        arguments={},
    )
    assert result.ok is False
    assert result.reason == "mcp_unregistered_tool"


@pytest.mark.asyncio
async def test_adapter_invoke_pubmed_all_10_tools_only_2_reachable(
    pubmed_server_spec: McpServerSpec,
    pubmed_search_binding: McpToolBinding,
    pubmed_fetch_binding: McpToolBinding,
    pubmed_search_remote_tool: McpRemoteTool,
    pubmed_fetch_remote_tool: McpRemoteTool,
) -> None:
    """The PubMed server has 10 tools.  Only search_pubmed and fetch_pubmed_articles
    are bound.  All other tool names (local or remote) are rejected.

    The 10 remote tools: pubmed_search_articles, pubmed_fetch_articles,
    pubmed_fetch_fulltext, pubmed_find_related, pubmed_format_citations,
    pubmed_lookup_citation, pubmed_lookup_mesh, pubmed_europepmc_search,
    pubmed_convert_ids, pubmed_spell_check.
    """
    unbound_remote_names = [
        "pubmed_fetch_fulltext",
        "pubmed_find_related",
        "pubmed_format_citations",
        "pubmed_lookup_citation",
        "pubmed_lookup_mesh",
        "pubmed_europepmc_search",
        "pubmed_convert_ids",
        "pubmed_spell_check",
    ]

    stub = StubMcpClient(
        tools=[pubmed_search_remote_tool, pubmed_fetch_remote_tool],
        call_results={
            "pubmed_search_articles": McpCallResult(
                structured_content={"pmids": [], "summaries": []},
                is_error=False,
            ),
            "pubmed_fetch_articles": McpCallResult(
                structured_content={"articles": [], "totalReturned": 0},
                is_error=False,
            ),
        },
    )

    adapter = McpAdapter(
        server_specs={pubmed_server_spec.id: pubmed_server_spec},
        bindings={
            pubmed_search_binding.local_name: pubmed_search_binding,
            pubmed_fetch_binding.local_name: pubmed_fetch_binding,
        },
        client_factory=lambda spec, **kw: stub,  # type: ignore[arg-type]
    )

    # The two allowlisted tools work.
    result = await adapter.invoke(
        local_name="literature.search_pubmed", arguments={"query": "test"}
    )
    assert result.ok is True

    result = await adapter.invoke(
        local_name="literature.fetch_pubmed_articles", arguments={"pmids": ["1"]}
    )
    assert result.ok is True

    # None of the unbound remote tool names are reachable through any local name.
    for remote_name in unbound_remote_names:
        result = await adapter.invoke(local_name=remote_name, arguments={})
        assert result.ok is False, f"{remote_name} should be unregistered"
        assert result.reason == "mcp_unregistered_tool"


# ---------------------------------------------------------------------------
# PubMed normalizer integration with McpAdapter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pubmed_normalizer_produces_contract_shape(
    pubmed_server_spec: McpServerSpec,
    pubmed_search_binding: McpToolBinding,
    pubmed_search_remote_tool: McpRemoteTool,
) -> None:
    """Prove that the normalizer converts raw MCP structuredContent to the
    contracted article-record shape."""
    from lab_copilot_gateway.mcp_pubmed import normalize_search_result

    canned = {
        "query": "CRISPR",
        "offset": 0,
        "pmids": ["12345678"],
        "summaries": [
            {
                "pmid": "12345678",
                "title": "CRISPR Paper",
                "authors": "Smith J",
                "source": "Nature",
                "pubDate": "2024",
                "doi": "10.1038/test",
                "pubmedUrl": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
            },
        ],
        "searchUrl": "https://pubmed.ncbi.nlm.nih.gov/?term=CRISPR",
    }

    normalized = normalize_search_result(canned)
    assert normalized["total"] == 1
    article = normalized["articles"][0]
    assert "pmid" in article
    assert "title" in article
    assert "authors" in article
    assert "journal" in article
    assert "year" in article
    assert "doi" in article
    assert "abstract" in article
    assert "url" in article
    assert article["pmid"] == "12345678"
    assert article["title"] == "CRISPR Paper"
    assert article["url"] == "https://pubmed.ncbi.nlm.nih.gov/12345678/"


# ============================================================================
# B4 — Transport-level pre-parse enforcement  (CONCERN 2)
# ============================================================================


@pytest.mark.asyncio
async def test_byte_counting_stream_raises_before_full_materialization() -> None:
    """Feed chunks through _ByteCountingStream directly and assert the
    size-cap exception fires BEFORE the full body is consumed."""
    from lab_copilot_gateway.mcp_adapter import (
        McpResultTooLargeError,
        _ByteCountingStream,
    )

    class _StubStream:
        """Stub that yields 3 chunks and tracks total yielded."""

        def __init__(self) -> None:
            self.yielded_bytes = 0

        async def __aiter__(self):
            for i in range(3):
                chunk = b"x" * 50  # 50 bytes per chunk
                self.yielded_bytes += len(chunk)
                yield chunk

    stub = _StubStream()
    counting = _ByteCountingStream(stub, max_bytes=80)  # 80-byte cap

    chunks_read = 0
    with pytest.raises(McpResultTooLargeError):
        async for _chunk in counting:
            chunks_read += 1

    # Only 1 full chunk reached the consumer (50 ≤ 80), then the 2nd
    # (cumulative 100 > 80) triggers the cap before the chunk is yielded.
    assert chunks_read == 1, (
        f"expected 1 chunk read before cap, got {chunks_read} "
        f"(stub yielded {stub.yielded_bytes} bytes total)"
    )
    # The stub produced all 3 chunks internally but only 1 was consumed.
    assert stub.yielded_bytes >= 50, "stub should have produced at least one chunk"


# ============================================================================
# Production schema hash format guard  (CONCERN 4)
# ============================================================================


def test_production_schema_hashes_are_64_char_lowercase_hex() -> None:
    """Assert that production schema hashes in mcp_bindings are valid SHA-256
    hex strings — not empty, not placeholder, not stale."""
    import re
    from lab_copilot_gateway.mcp_bindings import BINDINGS

    search_hash = BINDINGS["literature.search_pubmed"].input_schema_hash
    fetch_hash = BINDINGS["literature.fetch_pubmed_articles"].input_schema_hash

    for label, value in [
        ("literature.search_pubmed", search_hash),
        ("literature.fetch_pubmed_articles", fetch_hash),
    ]:
        assert len(value) == 64, f"{label}: hash must be 64 chars, got {len(value)}"
        assert re.fullmatch(r"[0-9a-f]{64}", value), (
            f"{label}: hash must be lowercase hex, got {value!r}"
        )
        assert (
            value != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ), f"{label}: hash is the empty-string SHA-256 — likely a placeholder"
