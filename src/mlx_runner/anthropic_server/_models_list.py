from __future__ import annotations

from typing import Optional

from ._model_object import _model_object


def _models_list(model_id: Optional[str]) -> dict:
    """An Anthropic-style ``GET /v1/models`` list payload.

    Lists the single loaded model (clients such as Claude Code call this at
    startup to validate that the configured model exists). Empty when the server
    was built without a known model id.
    """
    data = [_model_object(model_id)] if model_id else []
    return {
        "data": data,
        "has_more": False,
        "first_id": model_id,
        "last_id": model_id,
    }
