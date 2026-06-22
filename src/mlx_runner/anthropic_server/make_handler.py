from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler
from typing import Optional

from .anthropic_error import AnthropicError
from .parsed_request import ParsedRequest
from .parse_request import parse_request
from .new_message_id import new_message_id
from .sse_event import sse_event
from .generate_full import generate_full
from .generate_stream import generate_stream
from ._constant_time_eq import _constant_time_eq
from ._constants import MAX_REQUEST_BYTES


def make_handler(runner, *, api_key: Optional[str] = None):
    """Build a BaseHTTPRequestHandler class bound to ``runner``.

    Generation is serialized with a lock — a single MLX model is not safe to run
    concurrently.
    """
    lock = threading.Lock()

    class AnthropicHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "mlx-runner-anthropic/1"

        def log_message(self, *args):  # pragma: no cover - quiet by default
            pass

        def _send_json(self, status: int, payload: dict):
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            if self.close_connection:
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)

        def _send_error(self, err: AnthropicError):
            # Error paths can return before the request body is read (404/401/413).
            # On an HTTP/1.1 keep-alive connection — which cloudflared pools to the
            # origin — the unread body would be parsed as the next request's start
            # line, corrupting it (a spurious 501). Close the connection so it is
            # never reused with an undrained body.
            self.close_connection = True
            self._send_json(err.status, err.body())

        def _check_auth(self) -> Optional[AnthropicError]:
            if api_key is None:
                return None
            provided = self.headers.get("x-api-key") or ""
            if not _constant_time_eq(provided, api_key):
                return AnthropicError(401, "authentication_error", "invalid x-api-key.")
            return None

        def do_GET(self):  # a tiny health endpoint
            if self.path.rstrip("/") == "/health":
                self._send_json(200, {"status": "ok"})
            else:
                self._send_error(AnthropicError(404, "not_found_error", "not found."))

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/messages":
                self._send_error(AnthropicError(404, "not_found_error", f"unknown path {self.path!r}."))
                return
            auth_err = self._check_auth()
            if auth_err:
                self._send_error(auth_err)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                length = 0
            if length > MAX_REQUEST_BYTES:
                self._send_error(AnthropicError(
                    413, "request_too_large",
                    f"request body exceeds the {MAX_REQUEST_BYTES} byte limit.",
                ))
                return
            # Read at most the cap even if Content-Length lies, to bound memory.
            raw = self.rfile.read(min(length, MAX_REQUEST_BYTES)) if length else b""
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                self._send_error(AnthropicError(400, "invalid_request_error", "request body is not valid JSON."))
                return
            try:
                parsed = parse_request(body)
            except AnthropicError as e:
                self._send_error(e)
                return

            message_id = new_message_id()
            if parsed.stream:
                self._serve_stream(parsed, message_id)
            else:
                self._serve_full(parsed, message_id)

        def _serve_full(self, parsed: ParsedRequest, message_id: str):
            try:
                with lock:
                    message = generate_full(runner, parsed, message_id)
            except AnthropicError as e:
                self._send_error(e)
            except Exception as e:  # surface model errors in the API envelope
                self._send_error(AnthropicError(500, "api_error", str(e)))
            else:
                self._send_json(200, message)

        def _serve_stream(self, parsed: ParsedRequest, message_id: str):
            # SSE has no Content-Length; close the connection at the end so the
            # client reads to EOF instead of hanging on an HTTP/1.1 keep-alive.
            self.close_connection = True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            try:
                with lock:
                    for chunk in generate_stream(runner, parsed, message_id):
                        self.wfile.write(chunk.encode("utf-8"))
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):  # pragma: no cover
                pass
            except Exception as e:  # best-effort error event mid-stream
                try:
                    err = {"type": "error", "error": {"type": "api_error", "message": str(e)}}
                    self.wfile.write(sse_event("error", err).encode("utf-8"))
                    self.wfile.flush()
                except Exception:
                    pass

    return AnthropicHandler
