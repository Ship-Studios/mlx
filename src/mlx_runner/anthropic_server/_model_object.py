from __future__ import annotations


def _model_object(model_id: str) -> dict:
    """A single Anthropic-style ``model`` object for the given id.

    The id is echoed verbatim (the server serves whichever model it loaded
    regardless of the requested name), with a fixed ``created_at`` so the
    response is deterministic.
    """
    return {
        "type": "model",
        "id": model_id,
        "display_name": model_id,
        "created_at": "2025-01-01T00:00:00Z",
    }
