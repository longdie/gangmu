"""Bounded Messages-to-Chat transport for the real Claude Code subprocess.

This adapter translates text and function-tool messages only. It never runs a
tool, changes a prompt, repairs model output, or substitutes a model. Claude
Code still owns the MCP loop and StructuredOutput validation. Anthropic cache
and context-management hints are transport hints, not forwarded Chat fields;
the complete visible conversation is retained. Adaptive thinking is represented
by the requested reasoning effort, not by a fabricated thinking transcript.
"""
from __future__ import annotations

import hmac
import json
import secrets
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx


class BridgeProtocolError(ValueError):
    """An unsupported or invalid protocol payload, without sensitive details."""


def _integer(value: Any, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BridgeProtocolError("Expected an integer protocol field.")
    if value < (1 if positive else 0):
        raise BridgeProtocolError("Invalid integer protocol field.")
    return value


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        raise BridgeProtocolError("Only text content is supported by this adapter.")
    parts = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "text":
            raise BridgeProtocolError("Only text content is supported by this adapter.")
        if not isinstance(block.get("text"), str):
            raise BridgeProtocolError("A text block is invalid.")
        parts.append(block["text"])
    return "\n".join(parts)


def _text_content(content: Any) -> str | list[dict[str, str]]:
    """Keep text block order and boundaries; omit only cache metadata."""
    if isinstance(content, str):
        return content
    _text(content)
    return [{"type": "text", "text": b["text"]} for b in content]


def _tools(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 64:
        raise BridgeProtocolError("Invalid tool definitions.")
    result, names = [], set()
    for tool in value:
        if not isinstance(tool, dict):
            raise BridgeProtocolError("Invalid tool definition.")
        name = tool.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise BridgeProtocolError("Tool names must be nonempty and unique.")
        if tool.get("type", "custom") != "custom":
            raise BridgeProtocolError("Only function tools are supported.")
        schema = tool.get("input_schema")
        if not isinstance(schema, dict):
            raise BridgeProtocolError("A tool input schema is missing.")
        function = {"name": name, "parameters": schema}
        if "description" in tool:
            if not isinstance(tool["description"], str):
                raise BridgeProtocolError("Invalid tool description.")
            function["description"] = tool["description"]
        result.append({"type": "function", "function": function})
        names.add(name)
    return result


def _messages(value: Any, *, tool_names: set[str]) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise BridgeProtocolError("Messages must be a nonempty list.")
    result, pending, seen = [], set(), set()
    for message in value:
        if not isinstance(message, dict) or message.get("role") not in {"user", "assistant"}:
            raise BridgeProtocolError("Unsupported message role.")
        role, content = message["role"], message.get("content")
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        if not isinstance(content, list) or not content:
            raise BridgeProtocolError("Message content must contain text or tool blocks.")
        if role == "assistant":
            if pending:
                raise BridgeProtocolError("Tool results are missing before the next assistant message.")
            texts, calls = [], []
            for block in content:
                if not isinstance(block, dict):
                    raise BridgeProtocolError("Invalid content block.")
                if block.get("type") == "text":
                    texts.extend(_text_content([block]))
                elif block.get("type") == "tool_use":
                    call_id, name, arguments = block.get("id"), block.get("name"), block.get("input")
                    if not isinstance(call_id, str) or not call_id or call_id in seen:
                        raise BridgeProtocolError("Invalid or duplicate tool call ID.")
                    if not isinstance(name, str) or name not in tool_names or not isinstance(arguments, dict):
                        raise BridgeProtocolError("Invalid tool call.")
                    calls.append({"id": call_id, "type": "function", "function": {
                        "name": name, "arguments": json.dumps(arguments, ensure_ascii=False),
                    }})
                    pending.add(call_id)
                    seen.add(call_id)
                else:
                    raise BridgeProtocolError("Unsupported assistant content block.")
            item = {"role": "assistant", "content": texts or None}
            if calls:
                item["tool_calls"] = calls
            result.append(item)
        else:
            texts = []
            for block in content:
                if not isinstance(block, dict):
                    raise BridgeProtocolError("Invalid content block.")
                if block.get("type") == "tool_result":
                    if texts:
                        raise BridgeProtocolError("Tool results must precede user text.")
                    call_id = block.get("tool_use_id")
                    if not isinstance(call_id, str) or call_id not in pending:
                        raise BridgeProtocolError("A tool result has no matching pending call.")
                    tool_text = _text(block.get("content", ""))
                    if not isinstance(block.get("is_error", False), bool):
                        raise BridgeProtocolError("Invalid tool result error flag.")
                    if block.get("is_error", False):
                        tool_text = json.dumps({"is_error": True, "content": tool_text}, ensure_ascii=False)
                    result.append({"role": "tool", "tool_call_id": call_id, "content": tool_text})
                    pending.remove(call_id)
                elif block.get("type") == "text":
                    if pending:
                        raise BridgeProtocolError("Tool results are missing before user text.")
                    texts.extend(_text_content([block]))
                else:
                    raise BridgeProtocolError("Unsupported user content block.")
            if texts:
                result.append({"role": "user", "content": texts})
    if pending:
        raise BridgeProtocolError("The request contains unresolved tool calls.")
    return result


def messages_to_chat(body: Any, *, model: str, effort: str) -> dict[str, Any]:
    if not isinstance(body, dict) or body.get("model") != model:
        raise BridgeProtocolError("Requested model does not match the configured Agent model.")
    allowed = {"model", "messages", "system", "tools", "metadata", "max_tokens", "thinking",
        "context_management", "output_config", "stream", "tool_choice", "stop_sequences",
        "temperature", "top_p", "top_k"}
    if set(body) - allowed:
        raise BridgeProtocolError("Unsupported Messages request field.")
    if not isinstance(body.get("stream", False), bool):
        raise BridgeProtocolError("Invalid streaming flag.")
    tools = _tools(body.get("tools"))
    names = {t["function"]["name"] for t in tools}
    messages = _messages(body.get("messages"), tool_names=names)
    if body.get("system"):
        messages.insert(0, {"role": "system", "content": _text_content(body["system"])})
    maximum = _integer(body.get("max_tokens"), positive=True)
    if maximum > 131072:
        raise BridgeProtocolError("The requested output token limit is unsupported.")
    result = {"model": model, "messages": messages, "stream": False, "max_completion_tokens": maximum}
    if tools:
        result["tools"] = tools
    choice = body.get("tool_choice")
    if choice is not None:
        if not isinstance(choice, dict):
            raise BridgeProtocolError("Invalid tool choice.")
        kind = choice.get("type")
        if kind in {"auto", "none"}:
            result["tool_choice"] = kind
        elif kind == "any":
            result["tool_choice"] = "required"
        elif kind == "tool" and isinstance(choice.get("name"), str) and choice["name"] in names:
            result["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
        else:
            raise BridgeProtocolError("Unsupported tool choice.")
        if "disable_parallel_tool_use" in choice:
            if not isinstance(choice["disable_parallel_tool_use"], bool):
                raise BridgeProtocolError("Invalid parallel tool setting.")
            result["parallel_tool_calls"] = not choice["disable_parallel_tool_use"]
    output_config = body.get("output_config", {})
    if not isinstance(output_config, dict) or set(output_config) - {"effort"}:
        raise BridgeProtocolError("Unsupported output configuration.")
    requested_effort = output_config.get("effort", effort)
    if not isinstance(requested_effort, str) or requested_effort not in {"low", "medium", "high", "xhigh", "max"}:
        raise BridgeProtocolError("Unsupported reasoning effort.")
    thinking = body.get("thinking")
    if thinking is not None:
        if not isinstance(thinking, dict):
            raise BridgeProtocolError("Unsupported thinking configuration.")
        if thinking.get("type") == "disabled":
            raise BridgeProtocolError("Disabled thinking has no configured Chat equivalent.")
        if thinking.get("type") != "adaptive" or "budget_tokens" in thinking:
            raise BridgeProtocolError("Only adaptive thinking is supported; explicit thinking budgets have no configured Chat equivalent.")
    result["reasoning_effort"] = requested_effort
    for name in ("temperature", "top_p"):
        if name in body:
            if isinstance(body[name], bool) or not isinstance(body[name], (int, float)):
                raise BridgeProtocolError("Invalid sampling setting.")
            result[name] = body[name]
    if body.get("stop_sequences"):
        stops = body["stop_sequences"]
        if not isinstance(stops, list) or not all(isinstance(s, str) for s in stops):
            raise BridgeProtocolError("Invalid stop sequences.")
        result["stop"] = stops
    if "top_k" in body:
        raise BridgeProtocolError("top_k has no Chat equivalent.")
    return result


def chat_to_message(body: Any, *, model: str, tool_names: set[str]) -> dict[str, Any]:
    if not isinstance(body, dict) or body.get("model") != model:
        raise BridgeProtocolError("Agent upstream model mismatch: refusing a different or missing model.")
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise BridgeProtocolError("Upstream returned invalid completion choices.")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise BridgeProtocolError("Upstream returned an invalid assistant message.")
    if message.get("refusal"):
        raise BridgeProtocolError("Upstream returned a refusal instead of an Agent result.")
    content, text = [], message.get("content")
    if text is not None and not isinstance(text, str):
        raise BridgeProtocolError("Upstream returned unsupported visible content.")
    if text:
        content.append({"type": "text", "text": text})
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list):
        raise BridgeProtocolError("Upstream returned invalid tool calls.")
    seen = set()
    for call in calls:
        if not isinstance(call, dict) or call.get("type") != "function":
            raise BridgeProtocolError("Upstream returned an unsupported tool call.")
        call_id, function = call.get("id"), call.get("function")
        if not isinstance(call_id, str) or not call_id or call_id in seen or not isinstance(function, dict):
            raise BridgeProtocolError("Upstream returned an invalid tool call ID.")
        if not isinstance(function.get("name"), str) or function["name"] not in tool_names or not isinstance(function.get("arguments"), str):
            raise BridgeProtocolError("Upstream returned an unknown or invalid tool call.")
        try:
            arguments = json.loads(function["arguments"])
        except (ValueError, TypeError) as exc:
            raise BridgeProtocolError("Upstream returned invalid tool argument JSON.") from exc
        if not isinstance(arguments, dict):
            raise BridgeProtocolError("Upstream tool arguments must be an object.")
        content.append({"type": "tool_use", "id": call_id, "name": function["name"], "input": arguments})
        seen.add(call_id)
    finish = choice.get("finish_reason")
    if finish not in {"stop", "tool_calls", "length"} or bool(calls) != (finish == "tool_calls"):
        raise BridgeProtocolError("Upstream returned an unsupported or inconsistent stop reason.")
    if not content:
        raise BridgeProtocolError("Upstream returned no visible content or tool call.")
    usage = body.get("usage")
    if not isinstance(usage, dict):
        raise BridgeProtocolError("Upstream returned no token usage.")
    prompt, completion = _integer(usage.get("prompt_tokens")), _integer(usage.get("completion_tokens"))
    details = usage.get("prompt_tokens_details") or {}
    if not isinstance(details, dict):
        raise BridgeProtocolError("Upstream returned invalid token usage.")
    cached = _integer(details.get("cached_tokens", 0))
    if cached > prompt:
        raise BridgeProtocolError("Upstream returned inconsistent cached token usage.")
    return {"id": "msg_bridge_" + uuid.uuid4().hex, "type": "message", "role": "assistant",
        "model": body["model"], "content": content,
        "stop_reason": {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens"}[finish],
        "stop_sequence": None, "usage": {"input_tokens": prompt - cached, "output_tokens": completion,
            "cache_read_input_tokens": cached, "cache_creation_input_tokens": 0}}


def message_sse(message: dict[str, Any]) -> bytes:
    chunks = []
    def event(kind: str, **values: Any) -> None:
        chunks.append("event: " + kind + "\ndata: " + json.dumps({"type": kind, **values}, ensure_ascii=False) + "\n\n")
    start = dict(message, content=[], stop_reason=None)
    start["usage"] = dict(message["usage"], output_tokens=0)
    event("message_start", message=start)
    for index, block in enumerate(message["content"]):
        if block["type"] == "text":
            event("content_block_start", index=index, content_block={"type": "text", "text": ""})
            event("content_block_delta", index=index, delta={"type": "text_delta", "text": block["text"]})
        else:
            event("content_block_start", index=index, content_block=dict(block, input={}))
            event("content_block_delta", index=index, delta={"type": "input_json_delta", "partial_json": json.dumps(block["input"], ensure_ascii=False)})
        event("content_block_stop", index=index)
    event("message_delta", delta={"stop_reason": message["stop_reason"], "stop_sequence": None}, usage={"output_tokens": message["usage"]["output_tokens"]})
    event("message_stop")
    return "".join(chunks).encode("utf-8")


class ChatCompletionsBridge:
    """Context-managed loopback endpoint with a fixed upstream and model."""

    def __init__(self, *, api_key: str, base_url: str, model: str,
                 timeout_seconds: float = 120.0, trust_env: bool = True,
                 effort: str = "medium", maximum_concurrency: int = 5):
        parts = urlsplit(base_url.rstrip("/"))
        if parts.scheme not in {"http", "https"} or not parts.netloc or parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("Configure a valid Chat base URL without URL credentials or query parameters.")
        if not api_key or not model or not 0 < timeout_seconds <= 900:
            raise ValueError("The Agent connection, model, and bounded timeout are required.")
        if not 1 <= maximum_concurrency <= 5 or effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError("Invalid Agent bridge concurrency or effort.")
        path = parts.path.rstrip("/")
        if path.endswith("/messages"):
            raise ValueError("Configure a Chat base URL, not a Messages endpoint.")
        if not path.endswith("/chat/completions"):
            path += "/chat/completions"
        self._upstream = urlunsplit((parts.scheme, parts.netloc, path, "", ""))
        self._model, self._effort = model, effort
        self._api_key, self._trust_env, self._timeout = api_key, trust_env, float(timeout_seconds)
        self.auth_token = secrets.token_urlsafe(32)
        self._slots = threading.BoundedSemaphore(maximum_concurrency)
        self._server = self._thread = self._client = None
        self._closed = False

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("The Agent bridge is not running.")
        return "http://127.0.0.1:" + str(self._server.server_port)

    def __enter__(self) -> "ChatCompletionsBridge":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def start(self) -> "ChatCompletionsBridge":
        if self._closed:
            raise RuntimeError("A closed Agent bridge cannot be restarted.")
        if self._server is not None:
            return self
        bridge = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def _reply(self, status: int, data: bytes, content_type: str = "application/json") -> None:
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                self.close_connection = True

            def _error(self, status: int, message: str) -> None:
                error_type = "invalid_request_error" if status == 400 else "api_error"
                data = {"type": "error", "error": {"type": error_type, "message": message}}
                self._reply(status, json.dumps(data).encode())

            def do_POST(self) -> None:
                supplied = self.headers.get("Authorization", "")
                key = self.headers.get("x-api-key", "")
                if not (hmac.compare_digest(supplied.encode(), ("Bearer " + bridge.auth_token).encode()) or hmac.compare_digest(key.encode(), bridge.auth_token.encode())):
                    self._error(401, "Invalid local Agent bridge credential.")
                    return
                if urlsplit(self.path).path not in {"/v1/messages", "/messages"}:
                    self._error(404, "Unsupported local Agent bridge endpoint.")
                    return
                if not bridge._slots.acquire(blocking=False):
                    self._error(429, "The local Agent bridge concurrency limit was reached.")
                    return
                try:
                    self.connection.settimeout(min(bridge._timeout, 30.0))
                    if self.headers.get("Transfer-Encoding"):
                        raise BridgeProtocolError("Chunked requests are unsupported.")
                    try:
                        length = int(self.headers.get("Content-Length", "0"))
                    except ValueError as exc:
                        raise BridgeProtocolError("Invalid request length.") from exc
                    if not 0 < length <= 16 * 1024 * 1024:
                        raise BridgeProtocolError("Invalid request length.")
                    raw = self.rfile.read(length)
                    if len(raw) != length:
                        raise BridgeProtocolError("Incomplete request body.")
                    try:
                        body = json.loads(raw)
                    except (ValueError, UnicodeError) as exc:
                        raise BridgeProtocolError("Invalid request JSON.") from exc
                    request = messages_to_chat(body, model=bridge._model, effort=bridge._effort)
                    response = bridge._client.post(bridge._upstream, json=request,
                        headers={"Authorization": "Bearer " + bridge._api_key, "Content-Type": "application/json"})
                    if response.status_code != 200:
                        status = response.status_code if response.status_code in {400, 401, 403, 408, 409, 413, 429, 500, 502, 503, 504} else 502
                        self._error(status, "The configured Agent Chat service rejected the request.")
                        return
                    try:
                        completion = response.json()
                    except ValueError as exc:
                        raise BridgeProtocolError("The Agent Chat service returned invalid JSON.") from exc
                    names = {t["function"]["name"] for t in request.get("tools", [])}
                    message = chat_to_message(completion, model=bridge._model, tool_names=names)
                    if body.get("stream", False):
                        self._reply(200, message_sse(message), "text/event-stream")
                    else:
                        self._reply(200, json.dumps(message, ensure_ascii=False).encode("utf-8"))
                except BridgeProtocolError as exc:
                    self._error(400, str(exc))
                except (httpx.TimeoutException, TimeoutError):
                    self._error(504, "The Agent bridge request timed out.")
                except httpx.HTTPError:
                    self._error(502, "The Agent bridge could not contact its configured Chat service.")
                except Exception:
                    self._error(502, "The Agent bridge request failed.")
                finally:
                    bridge._slots.release()

            def do_GET(self) -> None:
                self._error(404, "Unsupported local Agent bridge endpoint.")

        self._client = httpx.Client(timeout=self._timeout, trust_env=self._trust_env, follow_redirects=False)
        try:
            self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            self._server.daemon_threads = True
            self._server.block_on_close = False
            self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
            self._thread.start()
        except Exception:
            self.close()
            raise
        return self

    def close(self) -> None:
        self._closed = True
        if self._server is not None:
            # shutdown() waits for serve_forever(); calling it when Thread.start
            # failed would wait forever. An unstarted server only needs close.
            if self._thread is not None and self._thread.is_alive():
                self._server.shutdown()
            self._server.server_close()
        if self._client is not None:
            self._client.close()
        if self._thread is not None and self._thread.ident is not None:
            self._thread.join(timeout=2.0)
        self._server = self._client = self._thread = None
