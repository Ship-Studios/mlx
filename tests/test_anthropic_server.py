import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import ThreadingHTTPServer

import pytest

from mlx_runner import anthropic_server as a


# --- request parsing ---------------------------------------------------------


def test_parse_minimal_string_content():
    p = a.parse_request({"model": "m", "max_tokens": 10, "messages": [{"role": "user", "content": "hi"}]})
    assert p.model == "m" and p.max_tokens == 10
    assert p.messages == [{"role": "user", "content": "hi"}]
    assert p.stream is False and p.stop_sequences == []


def test_parse_text_block_content_and_system_blocks():
    p = a.parse_request({
        "model": "m", "max_tokens": 5,
        "system": [{"type": "text", "text": "be terse"}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}],
        "stop_sequences": ["</s>"], "temperature": 0.5, "top_p": 0.9, "top_k": 40, "stream": True,
    })
    assert p.system == "be terse"
    assert p.messages[0]["content"] == "ab"
    assert p.stop_sequences == ["</s>"]
    assert p.temperature == 0.5 and p.top_p == 0.9 and p.top_k == 40 and p.stream is True


@pytest.mark.parametrize("body, msg", [
    ({"max_tokens": 1, "messages": [{"role": "user", "content": "x"}]}, "model"),
    ({"model": "m", "messages": [{"role": "user", "content": "x"}]}, "max_tokens"),
    ({"model": "m", "max_tokens": 0, "messages": [{"role": "user", "content": "x"}]}, "max_tokens"),
    ({"model": "m", "max_tokens": 1, "messages": []}, "messages"),
    ({"model": "m", "max_tokens": 1, "messages": [{"role": "system", "content": "x"}]}, "role"),
])
def test_parse_invalid(body, msg):
    with pytest.raises(a.AnthropicError) as e:
        a.parse_request(body)
    assert e.value.status == 400
    assert msg in e.value.message


def test_parse_rejects_non_text_blocks():
    with pytest.raises(a.AnthropicError) as e:
        a.parse_request({
            "model": "m", "max_tokens": 1,
            "messages": [{"role": "user", "content": [{"type": "image", "source": {}}]}],
        })
    assert "text content blocks" in e.value.message


# --- stop sequence filter ----------------------------------------------------


def test_stop_filter_no_stops_passes_through():
    sf = a.StopSequenceFilter([])
    assert sf.feed("hello") == ("hello", False)


def test_stop_filter_holds_back_potential_prefix():
    sf = a.StopSequenceFilter(["END"])
    # "EN" could be the start of "END" → held back
    emit, stopped = sf.feed("abcEN")
    assert emit == "abc" and stopped is False
    # completes the stop → emit nothing more, stopped
    emit, stopped = sf.feed("D more")
    assert emit == "" and stopped is True
    assert sf.matched == "END"


def test_stop_filter_emits_text_before_stop():
    sf = a.StopSequenceFilter(["<<"])
    emit, stopped = sf.feed("keep<<drop")
    assert emit == "keep" and stopped is True


def test_stop_filter_flush_returns_tail():
    sf = a.StopSequenceFilter(["ZZZ"])  # holds back max_len-1 == 2 chars conservatively
    emit, stopped = sf.feed("tailZ")
    assert emit == "tai" and stopped is False
    assert sf.flush() == "lZ"


# --- response + SSE builders -------------------------------------------------


def test_build_message_shape():
    m = a.build_message(
        message_id="msg_1", model="m", text="hello",
        stop_reason="end_turn", stop_sequence=None, input_tokens=3, output_tokens=2,
    )
    assert m["type"] == "message" and m["role"] == "assistant"
    assert m["content"] == [{"type": "text", "text": "hello"}]
    assert m["stop_reason"] == "end_turn" and m["stop_sequence"] is None
    assert m["usage"] == {"input_tokens": 3, "output_tokens": 2}


def test_sse_event_framing():
    raw = a.content_block_delta_event("hi")
    assert raw.startswith("event: content_block_delta\ndata: ")
    assert raw.endswith("\n\n")
    payload = json.loads(raw.split("data: ", 1)[1].strip())
    assert payload == {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}}


def test_message_start_event_has_empty_content_and_null_stop():
    payload = json.loads(a.message_start_event("msg_1", "m", 7).split("data: ", 1)[1])
    msg = payload["message"]
    assert msg["content"] == [] and msg["stop_reason"] is None
    assert msg["usage"] == {"input_tokens": 7, "output_tokens": 0}


# --- generation glue (fake runner, no mlx) -----------------------------------


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


def test_generate_full_end_turn():
    runner = FakeRunner(["Hello ", "world"], finish_reason="stop")
    parsed = a.parse_request({"model": "m", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}]})
    msg = a.generate_full(runner, parsed, "msg_1")
    assert msg["content"][0]["text"] == "Hello world"
    assert msg["stop_reason"] == "end_turn"


def test_generate_full_max_tokens():
    runner = FakeRunner(["partial"], finish_reason="length")
    parsed = a.parse_request({"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]})
    msg = a.generate_full(runner, parsed, "msg_1")
    assert msg["stop_reason"] == "max_tokens"


def test_generate_full_stop_sequence_truncates():
    runner = FakeRunner(["keep", "STOPdrop"], finish_reason="stop")
    parsed = a.parse_request({
        "model": "m", "max_tokens": 50,
        "messages": [{"role": "user", "content": "hi"}], "stop_sequences": ["STOP"],
    })
    msg = a.generate_full(runner, parsed, "msg_1")
    assert msg["content"][0]["text"] == "keep"
    assert msg["stop_reason"] == "stop_sequence" and msg["stop_sequence"] == "STOP"


def test_generate_stream_event_order():
    runner = FakeRunner(["Hi", " there"], finish_reason="stop")
    parsed = a.parse_request({"model": "m", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}], "stream": True})
    events = list(a.generate_stream(runner, parsed, "msg_1"))
    types = [json.loads(e.split("data: ", 1)[1])["type"] for e in events]
    assert types[0] == "message_start"
    assert types[1] == "content_block_start"
    assert types[2] == "ping"
    assert "content_block_delta" in types
    assert types[-3] == "content_block_stop"
    assert types[-2] == "message_delta"
    assert types[-1] == "message_stop"
    # reconstruct the streamed text
    text = "".join(
        json.loads(e.split("data: ", 1)[1])["delta"]["text"]
        for e in events if "content_block_delta" in e
    )
    assert text == "Hi there"


def test_generate_stream_message_delta_carries_stop_reason():
    runner = FakeRunner(["x"], finish_reason="length")
    parsed = a.parse_request({"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}], "stream": True})
    events = [json.loads(e.split("data: ", 1)[1]) for e in a.generate_stream(runner, parsed, "msg_1")]
    delta = next(e for e in events if e["type"] == "message_delta")
    assert delta["delta"]["stop_reason"] == "max_tokens"


def test_error_body_shape():
    err = a.AnthropicError(400, "invalid_request_error", "bad")
    assert err.body() == {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}


# --- token counting (incl. stop-interruption staleness) ----------------------


class FakeTokenizer:
    def encode(self, text):
        return list(text)  # one "token" per char → len == len(text)


def test_output_tokens_uses_working_tokenizer_when_not_stopped():
    runner = FakeRunner([], finish_reason="stop", gen_tokens=0)
    runner.tokenizer = FakeTokenizer()
    assert a._output_tokens(runner, "abcd", stopped=False) == 4


def test_output_tokens_uses_stats_generation_tokens_when_positive():
    runner = FakeRunner([], gen_tokens=7)
    assert a._output_tokens(runner, "whatever", stopped=False) == 7


def test_output_tokens_ignores_stale_stats_when_stopped():
    # Stale positive stats from a prior request must NOT be reused on a stop break.
    runner = FakeRunner([], gen_tokens=999)
    runner.tokenizer = FakeTokenizer()
    assert a._output_tokens(runner, "keep", stopped=True) == 4  # counts the text, not 999


def test_output_tokens_encode_failure_falls_back():
    class BadTok:
        def encode(self, text):
            raise RuntimeError("boom")

    runner = FakeRunner([], gen_tokens=0)
    runner.tokenizer = BadTok()
    assert a._output_tokens(runner, "abcdefgh", stopped=False) == 2  # len//4


def test_generate_full_stop_sequence_reports_text_based_tokens():
    runner = FakeRunner(["keep", "STOPx"], gen_tokens=999)  # stale stats
    runner.tokenizer = FakeTokenizer()
    parsed = a.parse_request({
        "model": "m", "max_tokens": 50,
        "messages": [{"role": "user", "content": "hi"}], "stop_sequences": ["STOP"],
    })
    msg = a.generate_full(runner, parsed, "msg_1")
    assert msg["content"][0]["text"] == "keep"
    assert msg["usage"]["output_tokens"] == 4  # not the stale 999


# --- stop-filter edge cases --------------------------------------------------


def test_stop_filter_single_char_no_holdback():
    sf = a.StopSequenceFilter(["#"])  # max_len 1 → hold<=0, emit immediately
    assert sf.feed("ab") == ("ab", False)
    emit, stopped = sf.feed("c#d")
    assert emit == "c" and stopped is True and sf.matched == "#"


def test_stop_filter_earliest_match_wins_across_sequences():
    sf = a.StopSequenceFilter(["END", "STOP"])
    emit, stopped = sf.feed("aaSTOPbbENDcc")
    assert emit == "aa" and stopped is True and sf.matched == "STOP"


def test_stop_filter_reconstructs_text_across_emit_and_flush():
    sf = a.StopSequenceFilter(["ZZZ"])
    out = []
    for chunk in ["he", "llo wor", "ld"]:
        emit, stopped = sf.feed(chunk)
        out.append(emit)
        assert not stopped
    out.append(sf.flush())
    assert "".join(out) == "hello world"  # invariant: nothing lost when no stop


# --- HTTP handler (drives a real ThreadingHTTPServer with a fake runner) ------


@contextmanager
def _running_server(runner, **kw):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), a.make_handler(runner, **kw))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _post(base, body, headers=None, path="/v1/messages", raw=None):
    data = raw if raw is not None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers=headers or {}, method="POST")
    return urllib.request.urlopen(req, timeout=10)


def test_handler_health():
    with _running_server(FakeRunner(["x"])) as base:
        r = urllib.request.urlopen(base + "/health", timeout=5)
        assert r.status == 200 and json.loads(r.read())["status"] == "ok"


def test_handler_unknown_path_404():
    with _running_server(FakeRunner(["x"])) as base:
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(base, {"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}, path="/nope")
        assert e.value.code == 404
        assert json.loads(e.value.read())["error"]["type"] == "not_found_error"


def test_handler_invalid_json_400():
    with _running_server(FakeRunner(["x"])) as base:
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(base, None, raw=b"{not json")
        assert e.value.code == 400


def test_handler_auth_enforced():
    with _running_server(FakeRunner(["hi"]), api_key="secret") as base:
        body = {"model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(base, body)  # no key
        assert e.value.code == 401 and json.loads(e.value.read())["error"]["type"] == "authentication_error"
        r = _post(base, body, headers={"x-api-key": "secret", "content-type": "application/json"})
        assert r.status == 200 and json.loads(r.read())["content"][0]["text"] == "hi"


def test_handler_body_too_large_413():
    with _running_server(FakeRunner(["x"])) as base:
        headers = {"Content-Length": str(a.MAX_REQUEST_BYTES + 1), "content-type": "application/json"}
        # urllib won't send a body bigger than data; fake an oversized Content-Length.
        req = urllib.request.Request(base + "/v1/messages", data=b"{}", headers=headers, method="POST")
        # Override the header urllib would otherwise compute from len(data).
        req.add_unredirected_header("Content-Length", str(a.MAX_REQUEST_BYTES + 1))
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req, timeout=10)
        assert e.value.code == 413


def test_handler_model_error_becomes_500():
    class Boom(FakeRunner):
        def stream(self, prompt=None, *, messages=None, system=None, config=None):
            raise RuntimeError("model exploded")
            yield  # pragma: no cover

    with _running_server(Boom([])) as base:
        body = {"model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}]}
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(base, body, headers={"content-type": "application/json"})
        assert e.value.code == 500 and json.loads(e.value.read())["error"]["type"] == "api_error"
