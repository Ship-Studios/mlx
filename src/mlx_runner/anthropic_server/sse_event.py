from __future__ import annotations

import json


def sse_event(event_type: str, data: dict) -> str:
    """Format one Server-Sent Event in the Anthropic shape."""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
