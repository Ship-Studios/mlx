from __future__ import annotations

from typing import Any, Optional

from .generation_stats import GenerationStats


class _ModelRunnerInitMixin:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        model_path: Optional[str] = None,
    ) -> None:
        """Initialize a ModelRunner with a pre-loaded model and tokenizer.

        Args:
            model: The loaded model object from mlx-lm.
            tokenizer: The tokenizer object for tokenization.
            model_path: Optional path to the model (for display purposes).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.last_stats: Optional[GenerationStats] = None
