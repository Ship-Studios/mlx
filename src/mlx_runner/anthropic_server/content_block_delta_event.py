from __future__ import annotations

from .sse_event import sse_event


def content_block_delta_event(text: str) -> str:
    return sse_event(
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
    )
