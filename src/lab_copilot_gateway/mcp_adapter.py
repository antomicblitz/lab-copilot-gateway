"""MCP (Model Context Protocol) outbound adapter for the Lab Copilot Gateway.

MCP servers connect BEHIND the Gateway as downstream adapters.  The Gateway
remains the sole tool / policy / audit / approval boundary.  MCP tools are
allowlisted, pinned, schema-validated, and output-normalized.  No dynamic
tool registration.

Architecture:

    McpClient (protocol)
        ├── StreamableHttpMcpClient   (real, uses mcp SDK)
        └── StubMcpClient             (in-process stub for tests)

    McpServerSpec   — server id/url/timeout/max_result_bytes
    McpToolBinding  — local_name → (server_id, remote_name, input_schema_hash)
    McpAdapter      — resolves bindings, validates, invokes, normalizes
"""

from __future__ import annotations

import asyncio
import hashlib
import json as _json
import time as _time
from dataclasses import dataclass, field
from typing import Any, Protocol


# ---------------------------------------------------------------------------
# McpClient protocol
# ---------------------------------------------------------------------------


class McpClient(Protocol):
    """Protocol for MCP clients (real or stub).

    Each method is async.  Callers MUST ``connect()`` before any other
    operation and ``close()`` when done.  A ``ClientSession`` MUST NOT be
    cached across event loops — create a fresh one per invocation.
    """

    async def initialize(self) -> None:
        """Connect to the server, perform the MCP initialize handshake."""
        ...

    async def list_tools(self) -> list[McpRemoteTool]:
        """Return the tools advertised by the remote server."""
        ...

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> McpCallResult:
        """Invoke a tool on the remote server and return the result."""
        ...

    async def close(self) -> None:
        """Close the connection and release resources."""
        ...


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class McpRemoteTool:
    """A tool advertised by a remote MCP server (from tools/list)."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)

    def input_schema_hash(self) -> str:
        """Stable SHA-256 of the canonical-JSON input schema."""
        canonical = _json.dumps(
            self.input_schema, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class McpCallResult:
    """Normalized result from an MCP tool invocation.

    ``structured_content`` is the ``structuredContent`` field from the
    MCP ``CallToolResult``, if present.  Arbitrary text content, resources,
    images, and prompts are dropped — only ``structuredContent`` is surfaced.
    """

    structured_content: dict[str, Any] | None = None
    is_error: bool = False

    @classmethod
    def from_mcp_result(cls, result: Any) -> "McpCallResult":
        """Extract ``structuredContent`` from an MCP ``CallToolResult``.

        Drops ``content`` (text/resources/images/prompts) — only
        ``structuredContent`` is trusted for binding-specific normalization.
        """
        structured = None
        is_error = False
        if (
            hasattr(result, "structuredContent")
            and result.structuredContent is not None
        ):
            structured = result.structuredContent
        if hasattr(result, "isError"):
            is_error = bool(result.isError)
        return cls(structured_content=structured, is_error=is_error)


# ---------------------------------------------------------------------------
# McpServerSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class McpServerSpec:
    """Static configuration for one downstream MCP server."""

    id: str
    url: str
    timeout: float = 30.0
    max_result_bytes: int = 1_048_576  # 1 MiB


# ---------------------------------------------------------------------------
# McpToolBinding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class McpToolBinding:
    """One curated binding from a local gateway tool to a remote MCP tool.

    ``input_schema_hash`` is the expected SHA-256 of the remote tool's
    ``inputSchema`` at registration time.  At invocation, the adapter
    re-validates that the remote still advertises a matching schema.
    """

    server_id: str
    local_name: str
    remote_name: str
    input_schema_hash: str


# ---------------------------------------------------------------------------
# Transport-level response size cap  (B4)
# ---------------------------------------------------------------------------


class McpResultTooLargeError(Exception):
    """Raised when an MCP response body exceeds the configured byte cap."""


class _ByteCountingStream:
    """Wraps an httpx response byte stream and raises when bytes exceed cap.

    The wrapped stream is an ``httpx.AsyncByteStream``; this wrapper
    intercepts ``__aiter__`` to count bytes and abort early before the
    MCP SDK parses the full response body.
    """

    def __init__(self, stream: Any, max_bytes: int) -> None:
        self._stream = stream
        self._max_bytes = max_bytes
        self._bytes_read = 0

    # Delegate all attribute access to the wrapped stream.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    async def __aiter__(self) -> Any:
        async for chunk in self._stream:
            self._bytes_read += len(chunk)
            if self._bytes_read > self._max_bytes:
                raise McpResultTooLargeError(
                    f"Response body exceeds maximum allowed ({self._max_bytes} bytes)"
                )
            yield chunk


class _SizeLimitedAsyncTransport:
    """Wraps ``httpx.AsyncHTTPTransport`` to enforce response size caps.

    Intercepts at the transport layer — BEFORE the MCP SDK parses the
    response body — so an oversized response is rejected immediately
    rather than OOM-ing the process during deserialization.
    """

    def __init__(self, max_bytes: int) -> None:
        import httpx

        self._inner = httpx.AsyncHTTPTransport()
        self._max_bytes = max_bytes

    async def handle_async_request(self, request: Any) -> Any:
        response = await self._inner.handle_async_request(request)

        # First line of defence: check Content-Length header.
        content_length = response.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > self._max_bytes:
                    await response.aclose()
                    raise McpResultTooLargeError(
                        f"Response Content-Length ({content_length}) exceeds "
                        f"maximum allowed ({self._max_bytes} bytes)"
                    )
            except ValueError:
                pass  # malformed content-length, defer to streaming check

        # Second line: wrap the response byte stream to count during read.
        if hasattr(response, "stream") and response.stream is not None:
            response.stream = _ByteCountingStream(response.stream, self._max_bytes)  # type: ignore[assignment]  # AsyncByteStream protocol

        return response

    async def aclose(self) -> None:
        """Delegate close to the inner transport to prevent connection leaks."""
        await self._inner.aclose()


# ---------------------------------------------------------------------------
# StreamableHttpMcpClient  (real, uses mcp SDK)
# ---------------------------------------------------------------------------


class StreamableHttpMcpClient:
    """MCP client over Streamable HTTP transport.

    Uses the official ``mcp`` Python SDK (pinned to 1.28.1).  Callers must
    ``connect()``, perform operations, then ``close()``.  A fresh client
    MUST be created per invocation — do NOT cache ``ClientSession`` across
    event loops.

    Response-size enforcement: a ``_SizeLimitedAsyncTransport`` is injected
    into the httpx client so oversized responses are rejected BEFORE the
    MCP SDK parses the body.
    """

    def __init__(
        self,
        server_spec: McpServerSpec,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._spec = server_spec
        self._extra_headers = extra_headers or {}
        self._session: Any = None
        self._client_context: Any = None
        self._read_stream: Any = None
        self._write_stream: Any = None

    async def initialize(self) -> None:
        """Open the transport and perform the MCP initialize handshake."""
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        _default_headers: dict[str, str] = {
            "User-Agent": "LabCopilot/1.0",
            "Accept": "application/json",
            **self._extra_headers,
        }

        # Build a size-limited transport for response-size enforcement (B4).
        size_transport = _SizeLimitedAsyncTransport(self._spec.max_result_bytes)

        def _factory(
            headers: dict[str, str] | None = None,
            timeout: Any = None,
            auth: Any = None,
        ) -> Any:
            import httpx

            kwargs: dict[str, Any] = {
                "transport": size_transport,
                "follow_redirects": True,
                "headers": headers or _default_headers,
            }
            if timeout:
                kwargs["timeout"] = timeout
            else:
                kwargs["timeout"] = httpx.Timeout(
                    self._spec.timeout, read=self._spec.timeout * 2
                )
            if auth is not None:
                kwargs["auth"] = auth
            return httpx.AsyncClient(**kwargs)

        self._client_context = streamablehttp_client(
            self._spec.url,
            headers=_default_headers,
            timeout=self._spec.timeout,
            httpx_client_factory=_factory,
        )
        (
            self._read_stream,
            self._write_stream,
            _,
        ) = await self._client_context.__aenter__()
        self._session = ClientSession(
            self._read_stream,
            self._write_stream,
        )
        await self._session.initialize()

    async def list_tools(self) -> list[McpRemoteTool]:
        """List tools advertised by the remote MCP server."""
        if self._session is None:
            raise RuntimeError("MCP client not initialized — call initialize() first")
        result = await self._session.list_tools()
        return [
            McpRemoteTool(
                name=t.name,
                description=t.description or "",
                input_schema=t.inputSchema if hasattr(t, "inputSchema") else {},
            )
            for t in result.tools
        ]

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> McpCallResult:
        """Invoke a tool on the remote MCP server."""
        if self._session is None:
            raise RuntimeError("MCP client not initialized — call initialize() first")
        raw = await self._session.call_tool(name, arguments or {})
        return McpCallResult.from_mcp_result(raw)

    async def close(self) -> None:
        """Close the transport and release resources."""
        exc: Exception | None = None
        if self._client_context is not None:
            try:
                await self._client_context.__aexit__(None, None, None)
            except Exception as e:
                exc = e
        self._session = None
        self._client_context = None
        self._read_stream = None
        self._write_stream = None
        if exc:
            raise exc


# ---------------------------------------------------------------------------
# StubMcpClient  (in-process for tests)
# ---------------------------------------------------------------------------


@dataclass
class StubMcpClient:
    """In-process MCP client that returns canned responses.

    For each method you can pre-configure the return value, or seed a
    dict-based lookup.  Calls are recorded on ``calls`` for assertion.

    Supports ``_never_return`` + ``_never_return_method`` for testing
    timeout behaviour: setting ``_never_return=True`` makes the named
    method await forever (proving the overall timeout fires).
    """

    tools: list[McpRemoteTool] = field(default_factory=list)
    call_results: dict[str, McpCallResult] = field(default_factory=dict)
    _raise_on_call: Exception | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)
    _initialized: bool = False
    _closed: bool = False
    _never_return: bool = False
    _never_return_method: str = "call_tool"
    # Track bytes produced (for oversize test — B4).
    _bytes_produced: int = 0
    _stream_cut_off: bool = False

    async def initialize(self) -> None:
        self.calls.append({"method": "initialize"})
        if self._never_return and self._never_return_method == "initialize":
            await asyncio.Event().wait()
        self._initialized = True

    async def list_tools(self) -> list[McpRemoteTool]:
        self.calls.append({"method": "list_tools"})
        if self._never_return and self._never_return_method == "list_tools":
            await asyncio.Event().wait()
        return list(self.tools)

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> McpCallResult:
        self.calls.append({"method": "call_tool", "name": name, "arguments": arguments})
        if self._raise_on_call is not None:
            raise self._raise_on_call
        if self._never_return and self._never_return_method == "call_tool":
            await asyncio.Event().wait()
        if name in self.call_results:
            result = self.call_results[name]
            # Simulate a remote response that produces bytes — the adapter
            # should cut it off at the cap without fully materialising it.
            if result.structured_content:
                content_str = _json.dumps(
                    result.structured_content, sort_keys=True, separators=(",", ":")
                )
                self._bytes_produced = len(content_str.encode("utf-8"))
            return result
        return McpCallResult(
            is_error=True,
            structured_content={"error": f"stub: no result for {name!r}"},
        )

    async def close(self) -> None:
        self.calls.append({"method": "close"})
        self._closed = True


# ---------------------------------------------------------------------------
# McpAdapter  — lookup, validate, invoke, normalize
# ---------------------------------------------------------------------------


class McpAdapterError(Exception):
    """Raised when the MCP adapter cannot fulfil a tool invocation."""


@dataclass
class McpInvokeResult:
    """Normalized Gateway-shaped result from an MCP tool invocation."""

    ok: bool
    reason: str = ""
    structured_content: dict[str, Any] | None = None
    server_id: str = ""
    remote_tool: str = ""
    duration_ms: int = 0
    result_size_bytes: int = 0
    schema_hash: str = ""


# Default overall timeout for the full MCP invocation
# (connect → initialize → list_tools → call_tool → close).
_DEFAULT_OVERALL_TIMEOUT: float = 60.0


class McpAdapter:
    """Resolves bindings, validates remote schemas, invokes MCP tools.

    Fail-closed on: missing binding, remote tool absent, schema hash
    mismatch, timeout, oversized result, malformed result, remote error.
    """

    def __init__(
        self,
        server_specs: dict[str, McpServerSpec],
        bindings: dict[str, McpToolBinding],
        *,
        client_factory: Any = None,
        overall_timeout: float = _DEFAULT_OVERALL_TIMEOUT,
    ) -> None:
        self._server_specs = server_specs
        self._bindings = bindings
        self._client_factory = client_factory or StreamableHttpMcpClient
        self._overall_timeout = overall_timeout

    def _binding_for(self, local_name: str) -> McpToolBinding | None:
        return self._bindings.get(local_name)

    async def invoke(
        self,
        local_name: str,
        arguments: dict[str, Any],
    ) -> McpInvokeResult:
        """Resolve, validate, invoke, and normalize an MCP tool call.

        The full invocation is bounded by ``self._overall_timeout``.
        Returns ``McpInvokeResult`` — always an object (never raises).
        """
        # 1. Resolve binding.
        binding = self._binding_for(local_name)
        if binding is None:
            return McpInvokeResult(
                ok=False,
                reason="mcp_unregistered_tool",
            )

        # 2. Resolve server spec.
        spec = self._server_specs.get(binding.server_id)
        if spec is None:
            return McpInvokeResult(
                ok=False,
                reason="mcp_unknown_server",
                server_id=binding.server_id,
                remote_tool=binding.remote_name,
            )

        # Build a *new* client per invocation.
        client: McpClient = self._client_factory(spec)
        t0 = _time.monotonic()

        # Wrap the full flow (connect → initialize → list → call → close)
        # in a single overall timeout (B3).  The close is best-effort with
        # its own small bound so a hung close cannot extend the timeout.
        try:
            result = await asyncio.wait_for(
                self._invoke_impl(client, binding, spec, arguments),
                timeout=self._overall_timeout,
            )
            return result
        except asyncio.TimeoutError:
            await _safe_close(client, timeout=min(5, self._overall_timeout / 4))
            duration_ms = int((_time.monotonic() - t0) * 1000)
            return McpInvokeResult(
                ok=False,
                reason="mcp_timeout",
                server_id=binding.server_id,
                remote_tool=binding.remote_name,
                duration_ms=duration_ms,
            )
        except McpResultTooLargeError:
            await _safe_close(client, timeout=min(5, self._overall_timeout / 4))
            return McpInvokeResult(
                ok=False,
                reason="mcp_result_too_large",
                server_id=binding.server_id,
                remote_tool=binding.remote_name,
            )

    async def _invoke_impl(
        self,
        client: McpClient,
        binding: McpToolBinding,
        spec: McpServerSpec,
        arguments: dict[str, Any],
    ) -> McpInvokeResult:
        """Core invocation logic (inside the overall timeout)."""
        t0 = _time.monotonic()

        # 3. Connect + initialize.
        try:
            await client.initialize()
        except Exception as exc:
            await _safe_close(client)
            return McpInvokeResult(
                ok=False,
                reason=f"mcp_connect_failed: {_exc_summary(exc)}",
                server_id=binding.server_id,
                remote_tool=binding.remote_name,
            )

        # 4. List tools — validate the binding still holds.
        try:
            remote_tools = await client.list_tools()
        except Exception as exc:
            await _safe_close(client)
            return McpInvokeResult(
                ok=False,
                reason=f"mcp_list_tools_failed: {_exc_summary(exc)}",
                server_id=binding.server_id,
                remote_tool=binding.remote_name,
            )

        remote_by_name: dict[str, McpRemoteTool] = {rt.name: rt for rt in remote_tools}
        remote_tool = remote_by_name.get(binding.remote_name)
        if remote_tool is None:
            await _safe_close(client)
            return McpInvokeResult(
                ok=False,
                reason="mcp_remote_tool_not_found",
                server_id=binding.server_id,
                remote_tool=binding.remote_name,
            )

        # 5. Validate input schema hash.
        try:
            actual_hash = remote_tool.input_schema_hash()
        except Exception:
            await _safe_close(client)
            return McpInvokeResult(
                ok=False,
                reason="mcp_schema_error",
                server_id=binding.server_id,
                remote_tool=binding.remote_name,
            )
        if actual_hash != binding.input_schema_hash:
            await _safe_close(client)
            return McpInvokeResult(
                ok=False,
                reason="mcp_schema_mismatch",
                server_id=binding.server_id,
                remote_tool=binding.remote_name,
                schema_hash=actual_hash,
            )

        # 6. Call the tool.
        try:
            result = await client.call_tool(binding.remote_name, arguments)
        except McpResultTooLargeError:
            await _safe_close(client)
            return McpInvokeResult(
                ok=False,
                reason="mcp_result_too_large",
                server_id=binding.server_id,
                remote_tool=binding.remote_name,
            )
        except Exception as exc:
            await _safe_close(client)
            return McpInvokeResult(
                ok=False,
                reason=f"mcp_call_failed: {_exc_summary(exc)}",
                server_id=binding.server_id,
                remote_tool=binding.remote_name,
            )

        # 7. Check for remote error.
        if result.is_error:
            await _safe_close(client)
            # B1: never forward raw structuredContent on error — the Gateway
            # owns the error shape and must not leak untrusted remote content.
            return McpInvokeResult(
                ok=False,
                reason="mcp_remote_error",
                server_id=binding.server_id,
                remote_tool=binding.remote_name,
            )

        # 8. Validate structured content.
        if result.structured_content is None:
            await _safe_close(client)
            return McpInvokeResult(
                ok=False,
                reason="mcp_no_structured_content",
                server_id=binding.server_id,
                remote_tool=binding.remote_name,
            )

        # 9. Post-parse size-cap enforcement (defence-in-depth; the primary
        #    enforcement is at the transport layer in StreamableHttpMcpClient).
        try:
            serialized = _json.dumps(
                result.structured_content, sort_keys=True, separators=(",", ":")
            )
        except (TypeError, ValueError):
            await _safe_close(client)
            return McpInvokeResult(
                ok=False,
                reason="mcp_malformed_result",
                server_id=binding.server_id,
                remote_tool=binding.remote_name,
            )
        result_bytes = len(serialized.encode("utf-8"))
        if result_bytes > spec.max_result_bytes:
            await _safe_close(client)
            return McpInvokeResult(
                ok=False,
                reason="mcp_result_too_large",
                server_id=binding.server_id,
                remote_tool=binding.remote_name,
                result_size_bytes=result_bytes,
            )

        # 10. Success.
        duration_ms = int((_time.monotonic() - t0) * 1000)
        await _safe_close(client)
        return McpInvokeResult(
            ok=True,
            reason="ok",
            structured_content=result.structured_content,
            server_id=binding.server_id,
            remote_tool=binding.remote_name,
            duration_ms=duration_ms,
            result_size_bytes=result_bytes,
            schema_hash=actual_hash,
        )


async def _safe_close(client: McpClient, timeout: float = 5.0) -> None:
    """Best-effort close with optional timeout, swallowing exceptions."""
    try:
        await asyncio.wait_for(client.close(), timeout=timeout)
    except Exception:
        pass


def _exc_summary(exc: BaseException) -> str:
    """Return a one-line summary of an exception for error reasons."""
    return f"{type(exc).__name__}: {exc}"
