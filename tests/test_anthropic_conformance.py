"""L5 conformance + L2 SSE-frame deepening for the Anthropic Messages API server.

STAGED DRAFT — drop into tests/ only after mlx-refactor posts "split landed".
Honors the src/** + tests/ write-freeze by living outside the repo for now.

Strategy: the strongest correctness oracle for /v1/messages is the official
`anthropic` SDK's own parser. If the SDK round-trips our server's responses
(streaming and non-streaming) with no errors, we are shape-compatible with the
real API. Reuses the duck-typed `runner` seam + localhost server harness that
already exist in tests/test_anthropic_server.py, so NO mlx / mlx-lm is needed.
"""
import json
import threading
from contextlib import contextmanager
from http.server import ThreadingHTTPServer

import pytest

# Top-level (SDK-independent): doubles as a collection-time smoke check that the
# anthropic_server public surface still imports after the split.
import mlx_runner.anthropic_server as a


@pytest.fixture
def anthropic_sdk():
    """The optional `anthropic` SDK, or skip.

    The gate lives HERE, not at module top, so it only skips the L5 round-trip
    tests that actually need the SDK. The hermetic L2 SSE-frame checks below take
    no fixture and therefore still run in the dependency-free core suite.
    """
    return pytest.importorskip("anthropic")


# --- stub seam (mirrors tests/test_anthropic_server.py; duck-typed, no mlx) ---
class FakeStats:
    def __init__(self, finish_reason=None, generation_tokens=0):
        self.finish_reason = finish_reason
        self.generation_tokens = generation_tokens


class FakeRunner:
    def __init__(self, deltas, finish_reason="stop", gen_tokens=0):
        self._deltas = deltas
        self.last_stats = FakeStats(finish_reason, gen_tokens)
        self.tokenizer = None

    def stream(self, prompt=None, *, messages=None, system=None, config=None):
        for d in self._deltas:
            yield d


@contextmanager
def _running_server(runner, **kw):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), a.make_handler(runner, **kw))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


# --- L5: official SDK as the conformance oracle ------------------------------
def test_sdk_non_streaming_round_trip(anthropic_sdk):
    """The SDK parses our non-streaming Message without error."""
    with _running_server(FakeRunner(["Hello", ", ", "world"], finish_reason="stop")) as base:
        client = anthropic_sdk.Anthropic(base_url=base, api_key="x", max_retries=0)
        msg = client.messages.create(
            model="m",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
        )
    assert msg.id.startswith("msg_")
    assert msg.type == "message" and msg.role == "assistant"
    assert msg.content[0].type == "text"
    assert msg.content[0].text == "Hello, world"
    assert msg.stop_reason == "end_turn"
    assert msg.usage.output_tokens >= 0


def test_sdk_streaming_round_trip_accumulates(anthropic_sdk):
    """The SDK consumes our SSE stream and accumulates a final Message."""
    with _running_server(FakeRunner(["Hi", " there"], finish_reason="stop")) as base:
        client = anthropic_sdk.Anthropic(base_url=base, api_key="x", max_retries=0)
        with client.messages.stream(
            model="m",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
        ) as stream:
            streamed = "".join(stream.text_stream)
            final = stream.get_final_message()
    assert streamed == "Hi there"
    assert final.content[0].text == "Hi there"
    assert final.stop_reason == "end_turn"


# --- L2 deepening: SSE per-frame invariants (raw, no SDK) --------------------
def _stream_events(runner, parsed):
    return [json.loads(e.split("data: ", 1)[1]) for e in a.generate_stream(runner, parsed, "msg_1")]


def test_stop_reason_null_in_start_nonnull_in_delta():
    runner = FakeRunner(["x"], finish_reason="stop")
    parsed = a.parse_request({
        "model": "m", "max_tokens": 50,
        "messages": [{"role": "user", "content": "hi"}], "stream": True,
    })
    events = _stream_events(runner, parsed)
    start = next(e for e in events if e["type"] == "message_start")
    delta = next(e for e in events if e["type"] == "message_delta")
    assert start["message"]["stop_reason"] is None
    assert delta["delta"]["stop_reason"] is not None


def test_output_tokens_nondecreasing_across_frames():
    runner = FakeRunner(["a", "b", "c", "d"], finish_reason="stop")
    runner.tokenizer = type("T", (), {"encode": staticmethod(lambda s: list(s))})()
    parsed = a.parse_request({
        "model": "m", "max_tokens": 50,
        "messages": [{"role": "user", "content": "hi"}], "stream": True,
    })
    events = _stream_events(runner, parsed)
    seen = [
        e.get("usage", {}).get("output_tokens")
        if e["type"] == "message_delta"
        else e.get("message", {}).get("usage", {}).get("output_tokens")
        for e in events
        if e["type"] in ("message_start", "message_delta")
    ]
    seen = [v for v in seen if v is not None]
    assert seen == sorted(seen), f"output_tokens not monotonic: {seen}"


# --- L5: error must not poison a reused keep-alive connection -----------------
def test_error_response_does_not_poison_pooled_connection():
    """A 401 must not corrupt the next request on a reused keep-alive connection.

    Reproduces the live failure that the happy-path round-trip tests missed: an
    error response that doesn't drain the request body used to poison an HTTP/1.1
    keep-alive connection — which httpx (and therefore the anthropic SDK) pools —
    so the *next* request got mis-parsed into a bogus 501. Exercised over a real
    httpx pool (the SDK's transport), capped to a single connection to force reuse.
    Fails against the pre-fix server (r2 == 501); passes once errors close the conn.
    """
    httpx = pytest.importorskip("httpx")
    body = {"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
    with _running_server(FakeRunner(["ok"], finish_reason="stop"), api_key="secret") as base:
        limits = httpx.Limits(max_keepalive_connections=1, max_connections=1)
        with httpx.Client(base_url=base, limits=limits) as hc:
            r1 = hc.post("/v1/messages", json=body)  # no x-api-key -> 401 error
            assert r1.status_code == 401
            # Same pool: before the fix the undrained body poisoned this into a 501.
            r2 = hc.post("/v1/messages", json=body, headers={"x-api-key": "secret"})
            assert r2.status_code == 200, f"poisoned conn after error: {r2.status_code} {r2.text[:120]}"
            assert r2.json()["type"] == "message"
