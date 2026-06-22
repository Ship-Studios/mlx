from __future__ import annotations

from ..config import UserConfig


def _model_default_note(config: UserConfig) -> str:
    return f" (default from config: {config.model})" if config.model else ""
