from __future__ import annotations

import socket
import threading
import time
from typing import Any

from mcp.server.mcpserver import Context, MCPServer

from scripts.experiment.agent.sessions import AgentSessionRegistry


SESSION_HEADER = "X-MathER-Agent-Session"
MCP_SERVER_NAME = "mather_agent"
MCP_TOOL_NAME = "search_again"


class AgentMCPServer:
    """One local, query-isolated MCP service shared by an Agent run."""

    def __init__(
        self,
        *,
        sessions: AgentSessionRegistry,
        host: str = "127.0.0.1",
    ) -> None:
        self.sessions = sessions
        self.host = host
        self.port = _available_port(host)
        self.endpoint = f"http://{host}:{self.port}/mcp"
        self._server_error: BaseException | None = None
        self._thread: threading.Thread | None = None

        server = MCPServer(
            name=MCP_SERVER_NAME,
            instructions=(
                "This server exposes only search_again for the active MathER "
                "query. The query type and three-search limit are enforced "
                "by the server."
            ),
            log_level="ERROR",
        )

        @server.tool(
            name=MCP_TOOL_NAME,
            description=(
                "Search the frozen Canonical Knowledge Graph again using a "
                "meaning-preserving reformulation of the current mathematical "
                "query. At most two additional searches are allowed."
            ),
            structured_output=True,
        )
        def search_again(query_text: str, ctx: Context) -> dict[str, Any]:
            headers = ctx.headers or {}
            token = headers.get(SESSION_HEADER) or headers.get(
                SESSION_HEADER.lower()
            )
            if not token:
                raise ValueError("Missing MathER Agent session header.")
            return self.sessions.get(str(token)).search_again(query_text)

        self._mcp = server
        self._uvicorn: Any | None = None

    def start(self, *, timeout_seconds: float = 15.0) -> None:
        if self._thread is not None:
            raise RuntimeError("The Agent MCP server is already started.")
        import uvicorn

        app = self._mcp.streamable_http_app(
            streamable_http_path="/mcp",
            json_response=True,
            stateless_http=True,
            host=self.host,
        )
        config = uvicorn.Config(
            app,
            host=self.host,
            port=self.port,
            log_level="error",
            access_log=False,
        )
        self._uvicorn = uvicorn.Server(config)

        def serve() -> None:
            try:
                self._uvicorn.run()
            except BaseException as exc:  # pragma: no cover - startup failure
                self._server_error = exc

        self._thread = threading.Thread(
            target=serve,
            name="mather-agent-mcp",
            daemon=True,
        )
        self._thread.start()
        deadline = time.monotonic() + float(timeout_seconds)
        while time.monotonic() < deadline:
            if self._server_error is not None:
                raise RuntimeError("Agent MCP server failed to start.") from (
                    self._server_error
                )
            if bool(getattr(self._uvicorn, "started", False)) and _can_connect(
                self.host,
                self.port,
            ):
                return
            if not self._thread.is_alive():
                break
            time.sleep(0.05)
        self.stop()
        raise TimeoutError("Agent MCP server did not become ready in time.")

    def stop(self, *, timeout_seconds: float = 10.0) -> None:
        server = self._uvicorn
        thread = self._thread
        if server is not None:
            server.should_exit = True
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(timeout_seconds)))
        if thread is not None and thread.is_alive() and server is not None:
            server.force_exit = True
            thread.join(timeout=2.0)
        self._thread = None
        self._uvicorn = None


def mcp_config_template(endpoint: str = "<runtime-endpoint>") -> dict[str, Any]:
    """Return the stable MCP shape used in method hashes and manifests."""

    return {
        "mcpServers": {
            MCP_SERVER_NAME: {
                "type": "http",
                "url": endpoint,
                "headers": {SESSION_HEADER: "<query-session-token>"},
            }
        }
    }


def query_mcp_config(*, endpoint: str, session_token: str) -> dict[str, Any]:
    payload = mcp_config_template(endpoint)
    payload["mcpServers"][MCP_SERVER_NAME]["headers"][SESSION_HEADER] = (
        str(session_token)
    )
    return payload


def _available_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _can_connect(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.25):
            return True
    except OSError:
        return False
