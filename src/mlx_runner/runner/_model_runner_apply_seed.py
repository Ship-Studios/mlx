from __future__ import annotations

from .generation_config import GenerationConfig


class _ModelRunnerApplySeedMixin:
    @staticmethod
    def _apply_seed(config: GenerationConfig) -> None:
        if config.seed is None:
            return
        try:
            import mlx.core as mx

            mx.random.seed(config.seed)
        except ImportError:  # pragma: no cover - mlx absent
            pass
