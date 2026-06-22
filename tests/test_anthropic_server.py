import json

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
