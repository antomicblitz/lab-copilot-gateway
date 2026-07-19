"""Curated MCP tool bindings (Slice 4 — PubMed allowlist).

Each binding pins a local gateway tool name to a remote MCP tool on a
specific server.  The ``input_schema_hash`` is the SHA-256 of the remote
tool's canonical-JSON ``inputSchema`` at registration time.  The adapter
re-validates this hash on every invocation (fail-closed on mismatch).

Schema: ``{local_name: McpToolBinding(server_id, local_name, remote_name,
input_schema_hash)}``

Schema hashes below were computed *authoritatively* from the
pubmed-mcp-server v2.9.8 Zod v4 source definitions (commit ef0be2c0)
using the same canonicalization as the gateway (``json.dumps(sort_keys=True,
separators=(",", ":"))`` → SHA-256).  See ``mcp_pubmed.py`` runbook for
re-validation instructions.
"""

from __future__ import annotations

from lab_copilot_gateway.mcp_adapter import McpToolBinding

#: Static MCP tool bindings.  Populated by code review; no dynamic discovery.
#:
#: PubMed (Slice 4): two allowlisted literature tools.
#:   10 remote tools total on the server; only these 2 reachable.
#:   Remote tool names verified against actual v2.9.8 server code
#:   (pubmed_search_articles, pubmed_fetch_articles).
#:
#: The test_search entry exists so the gateway's /invoke path can dispatch
#: MCP tools end-to-end in tests.
BINDINGS: dict[str, McpToolBinding] = {
    "mcp.test_search": McpToolBinding(
        server_id="test-mcp",
        local_name="mcp.test_search",
        remote_name="remote_search",
        input_schema_hash="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    "literature.search_pubmed": McpToolBinding(
        server_id="pubmed",
        local_name="literature.search_pubmed",
        remote_name="pubmed_search_articles",
        input_schema_hash="fd48238e7d0d48a96aa415d2b1208212ea179a5f5dd3b16d495f6f53935b0106",
    ),
    "literature.fetch_pubmed_articles": McpToolBinding(
        server_id="pubmed",
        local_name="literature.fetch_pubmed_articles",
        remote_name="pubmed_fetch_articles",
        input_schema_hash="978e7bf7abf8f8884aea0c9df497b00ad8568d31736365f28a3dd8161ddee2ef",
    ),
}
