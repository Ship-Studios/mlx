from __future__ import annotations

from typing import Iterator, List

from .parsed_request import ParsedRequest
from .stop_sequence_filter import StopSequenceFilter
from .message_start_event import message_start_event
from .content_block_start_event import content_block_start_event
from .ping_event import ping_event
from .content_block_delta_event import content_block_delta_event
from .content_block_stop_event import content_block_stop_event
from .message_delta_event import message_delta_event
from .message_stop_event import message_stop_event
from ._generation_config import _generation_config
from ._input_tokens import _input_tokens
from ._output_tokens import _output_tokens


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
    yield message_delta_event(
        stop_reason, stop_sequence, _output_tokens(runner, "".join(collected), stopped=stopped)
    )
    yield message_stop_event()
