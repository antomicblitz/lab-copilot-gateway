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
#: Entries added in Slice 4 (PubMed bindings).
BINDINGS: dict[str, McpToolBinding] = {}
