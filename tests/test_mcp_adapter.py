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
    assert result.structured_content == {"error": "invalid query"}


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
