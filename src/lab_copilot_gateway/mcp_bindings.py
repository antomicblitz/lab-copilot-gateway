"""Curated MCP tool bindings (Slice 3 — placeholder, populated in Slice 4).

Each binding pins a local gateway tool name to a remote MCP tool on a
specific server.  The ``input_schema_hash`` is the SHA-256 of the remote
tool's canonical-JSON ``inputSchema`` at registration time.  The adapter
re-validates this hash on every invocation (fail-closed on mismatch).

Schema: ``{local_name: McpToolBinding(server_id, local_name, remote_name,
input_schema_hash)}``

Slice 3 leaves this empty.  Slice 4 adds PubMed bindings here.
"""

from __future__ import annotations

from lab_copilot_gateway.mcp_adapter import McpToolBinding

#: Static MCP tool bindings.  Populated by code review; no dynamic discovery.
#: Slice 4 adds PubMed bindings here.  The test_search entry exists so the
#: gateway's /invoke path can dispatch MCP tools end-to-end in tests.
BINDINGS: dict[str, McpToolBinding] = {
    "mcp.test_search": McpToolBinding(
        server_id="test-mcp",
        local_name="mcp.test_search",
        remote_name="remote_search",
        input_schema_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
}
