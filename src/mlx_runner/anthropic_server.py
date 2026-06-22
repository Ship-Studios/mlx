"""An Anthropic Messages API-compatible HTTP server backed by a local MLX model.

Emulates ``POST /v1/messages`` (non-streaming and SSE streaming) so that Anthropic
SDK clients can talk to a model served by :class:`mlx_runner.runner.ModelRunner`.
The exact wire format is pinned by the vendored type stubs and ``WIRE_FORMAT.md``
under ``reference/anthropic-api/``.

The request parsing, response building, SSE event formatting, and stop-sequence
handling are pure-Python and dependency-free (unit-tested without mlx). Only
``serve()`` and the generation glue touch the Apple-silicon-only model, and those
import :mod:`mlx_runner.runner` lazily.
"""
from __future__ import annotations

import json
import secrets
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator, List, Optional

ANTHROPIC_VERSION = "2023-06-01"


# --- errors ------------------------------------------------------------------


class AnthropicError(Exception):
    """An error renderable as the Anthropic error envelope with an HTTP status."""

    def __init__(self, status: int, error_type: str, message: str):
        super().__init__(message)
        self.status = status
        self.error_type = error_type
        self.message = message

    def body(self) -> dict:
        return {"type": "error", "error": {"type": self.error_type, "message": self.message}}


# --- request parsing ---------------------------------------------------------


@dataclass
class ParsedRequest:
    """A validated, normalized ``/v1/messages`` request."""

    model: str
    max_tokens: int
    messages: List[dict]  # normalized to [{"role": ..., "content": <str>}]
    system: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: List[str] = field(default_factory=list)
    stream: bool = False


def _content_to_text(content, *, where: str) -> str:
    """Flatten a message/system ``content`` (string or block list) to plain text.

    Rejects non-text blocks (images, tool_use, …) with a 400 — a local text model
    can't honor them, and silently dropping them would corrupt the conversation.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "text":
                btype = block.get("type") if isinstance(block, dict) else type(block).__name__
                raise AnthropicError(
                    400, "invalid_request_error",
                    f"{where}: only text content blocks are supported, got {btype!r}.",
                )
            parts.append(str(block.get("text", "")))
        return "".join(parts)
    raise AnthropicError(
        400, "invalid_request_error", f"{where}: content must be a string or a list of text blocks."
    )


def parse_request(body: dict) -> ParsedRequest:
    """Validate and normalize a decoded ``/v1/messages`` JSON body."""
    if not isinstance(body, dict):
        raise AnthropicError(400, "invalid_request_error", "request body must be a JSON object.")

    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise AnthropicError(400, "invalid_request_error", "`model` is required and must be a string.")

    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
        raise AnthropicError(400, "invalid_request_error", "`max_tokens` is required and must be a positive integer.")

    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise AnthropicError(400, "invalid_request_error", "`messages` is required and must be a non-empty array.")

    messages: List[dict] = []
    for i, m in enumerate(raw_messages):
        if not isinstance(m, dict):
            raise AnthropicError(400, "invalid_request_error", f"messages[{i}] must be an object.")
        role = m.get("role")
        if role not in ("user", "assistant"):
            raise AnthropicError(
                400, "invalid_request_error", f"messages[{i}].role must be 'user' or 'assistant'."
            )
        text = _content_to_text(m.get("content"), where=f"messages[{i}]")
        messages.append({"role": role, "content": text})

    system_val = body.get("system")
    system = _content_to_text(system_val, where="system") if system_val is not None else None

    stops = body.get("stop_sequences") or []
    if not isinstance(stops, list) or any(not isinstance(s, str) for s in stops):
        raise AnthropicError(400, "invalid_request_error", "`stop_sequences` must be an array of strings.")

    def _num(name, cast):
        v = body.get(name)
        if v is None:
            return None
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise AnthropicError(400, "invalid_request_error", f"`{name}` must be a number.")
        return cast(v)

    return ParsedRequest(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        system=system,
        temperature=_num("temperature", float),
        top_p=_num("top_p", float),
        top_k=_num("top_k", int),
        stop_sequences=[s for s in stops if s],
        stream=bool(body.get("stream", False)),
    )


# --- stop sequences ----------------------------------------------------------


class StopSequenceFilter:
    """Detect stop sequences in a stream of text deltas without emitting past them.

    ``feed(text)`` returns ``(safe_to_emit, stopped)``. While no stop sequence has
    appeared it holds back the last ``max_len - 1`` characters (which could be the
    start of a future match) and emits the rest. When a stop sequence completes, it
    returns the text preceding it and ``stopped=True``; ``matched`` names the
    sequence. ``flush()`` returns any held-back tail at natural end of generation.
    """

    def __init__(self, stops: List[str]):
        self.stops = [s for s in stops if s]
        self.max_len = max((len(s) for s in self.stops), default=0)
        self.buf = ""
        self.matched: Optional[str] = None

    def feed(self, text: str):
        if not self.stops:
            return text, False
        self.buf += text
        idx, which = -1, None
        for s in self.stops:
            i = self.buf.find(s)
            if i != -1 and (idx == -1 or i < idx):
                idx, which = i, s
        if idx != -1:
            emit = self.buf[:idx]
            self.buf = ""
            self.matched = which
            return emit, True
        hold = self.max_len - 1
        if hold <= 0 or len(self.buf) <= hold:
            if hold <= 0:
                emit, self.buf = self.buf, ""
                return emit, False
            return "", False
        emit = self.buf[:-hold]
        self.buf = self.buf[-hold:]
        return emit, False

    def flush(self) -> str:
        out, self.buf = self.buf, ""
        return out


# --- response building -------------------------------------------------------


def new_message_id() -> str:
    return "msg_" + secrets.token_hex(12)


def build_message(
    *,
    message_id: str,
    model: str,
    text: str,
    stop_reason: str,
    stop_sequence: Optional[str],
    input_tokens: int,
    output_tokens: int,
) -> dict:
    """A complete non-streaming ``Message`` object."""
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "stop_sequence": stop_sequence,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def sse_event(event_type: str, data: dict) -> str:
    """Format one Server-Sent Event in the Anthropic shape."""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def message_start_event(message_id: str, model: str, input_tokens: int) -> str:
    msg = {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {"input_tokens": input_tokens, "output_tokens": 0},
    }
    return sse_event("message_start", {"type": "message_start", "message": msg})


def content_block_start_event() -> str:
    return sse_event(
        "content_block_start",
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    )


def ping_event() -> str:
    return sse_event("ping", {"type": "ping"})


def content_block_delta_event(text: str) -> str:
    return sse_event(
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
    )


def content_block_stop_event() -> str:
    return sse_event("content_block_stop", {"type": "content_block_stop", "index": 0})


def message_delta_event(stop_reason: str, stop_sequence: Optional[str], output_tokens: int) -> str:
    return sse_event(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
            "usage": {"output_tokens": output_tokens},
        },
    )


def message_stop_event() -> str:
    return sse_event("message_stop", {"type": "message_stop"})


# --- generation glue (touches the model via the runner) ----------------------


def _generation_config(parsed: ParsedRequest):
    from .runner import GenerationConfig

    kwargs = {"max_tokens": parsed.max_tokens}
    if parsed.temperature is not None:
        kwargs["temperature"] = parsed.temperature
    if parsed.top_p is not None:
        kwargs["top_p"] = parsed.top_p
    if parsed.top_k is not None:
        kwargs["top_k"] = parsed.top_k
    return GenerationConfig(**kwargs)


def _count_tokens(runner, text: str) -> int:
    """Best-effort token count via the runner's tokenizer; rough fallback otherwise."""
    tok = getattr(runner, "tokenizer", None)
    encode = getattr(tok, "encode", None)
    if encode is not None:
        try:
            return len(encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)


def _output_tokens(runner, text: str) -> int:
    stats = getattr(runner, "last_stats", None)
    gen = getattr(stats, "generation_tokens", None) if stats else None
    if isinstance(gen, int) and gen > 0:
        return gen
    return _count_tokens(runner, text)


def _input_tokens(runner, parsed: ParsedRequest) -> int:
    parts = [parsed.system or ""] + [m["content"] for m in parsed.messages]
    return _count_tokens(runner, "\n".join(p for p in parts if p))


def generate_full(runner, parsed: ParsedRequest, message_id: str) -> dict:
    """Run a non-streaming generation and return a ``Message`` dict."""
    config = _generation_config(parsed)
    sf = StopSequenceFilter(parsed.stop_sequences)
    collected: List[str] = []
    stopped = False
    for delta in runner.stream(messages=parsed.messages, system=parsed.system, config=config):
        emit, stop = sf.feed(delta)
        if emit:
            collected.append(emit)
        if stop:
            stopped = True
            break
    if not stopped:
        tail = sf.flush()
        if tail:
            collected.append(tail)

    text = "".join(collected)
    if stopped:
        stop_reason, stop_sequence = "stop_sequence", sf.matched
    else:
        stats = getattr(runner, "last_stats", None)
        finish = getattr(stats, "finish_reason", None) if stats else None
        if finish == "length":
            stop_reason, stop_sequence = "max_tokens", None
        else:
            stop_reason, stop_sequence = "end_turn", None
    return build_message(
        message_id=message_id,
        model=parsed.model,
        text=text,
        stop_reason=stop_reason,
        stop_sequence=stop_sequence,
        input_tokens=_input_tokens(runner, parsed),
        output_tokens=_output_tokens(runner, text),
    )


def generate_stream(runner, parsed: ParsedRequest, message_id: str) -> Iterator[str]:
    """Yield the full SSE event sequence for a streaming generation."""
    config = _generation_config(parsed)
    yield message_start_event(message_id, parsed.model, _input_tokens(runner, parsed))
    yield content_block_start_event()
    yield ping_event()

    sf = StopSequenceFilter(parsed.stop_sequences)
    collected: List[str] = []
    stopped = False
    for delta in runner.stream(messages=parsed.messages, system=parsed.system, config=config):
        emit, stop = sf.feed(delta)
        if emit:
            collected.append(emit)
            yield content_block_delta_event(emit)
        if stop:
            stopped = True
            break
    if not stopped:
        tail = sf.flush()
        if tail:
            collected.append(tail)
            yield content_block_delta_event(tail)

    if stopped:
        stop_reason, stop_sequence = "stop_sequence", sf.matched
    else:
        stats = getattr(runner, "last_stats", None)
        finish = getattr(stats, "finish_reason", None) if stats else None
        stop_reason = "max_tokens" if finish == "length" else "end_turn"
        stop_sequence = None

    yield content_block_stop_event()
    yield message_delta_event(stop_reason, stop_sequence, _output_tokens(runner, "".join(collected)))
    yield message_stop_event()


# --- HTTP server -------------------------------------------------------------


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
            self.end_headers()
            self.wfile.write(data)

        def _send_error(self, err: AnthropicError):
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
            raw = self.rfile.read(length) if length else b""
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


def _constant_time_eq(a: str, b: str) -> bool:
    import hmac

    return hmac.compare_digest(a, b)


def serve(
    model_path: str,
    host: str = "127.0.0.1",
    port: int = 8080,
    *,
    adapter_path: Optional[str] = None,
    trust_remote_code: bool = False,
    api_key: Optional[str] = None,
) -> int:
    """Load the model and serve the Anthropic Messages API until interrupted."""
    from .runner import ModelRunner

    runner = ModelRunner.load(model_path, adapter_path=adapter_path, trust_remote_code=trust_remote_code)
    handler = make_handler(runner, api_key=api_key)
    httpd = ThreadingHTTPServer((host, port), handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        httpd.server_close()
    return 0
