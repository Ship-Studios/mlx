from __future__ import annotations

from typing import Optional


class _EmbeddingRunnerInitMixin:
    def __init__(self, model, tokenizer, model_path: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.model_path = model_path
