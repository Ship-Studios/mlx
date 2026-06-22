from __future__ import annotations

from typing import List

from .parsed_request import ParsedRequest
from .stop_sequence_filter import StopSequenceFilter
from .build_message import build_message
from ._generation_config import _generation_config
from ._input_tokens import _input_tokens
from ._output_tokens import _output_tokens


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
        output_tokens=_output_tokens(runner, text, stopped=stopped),
    )
