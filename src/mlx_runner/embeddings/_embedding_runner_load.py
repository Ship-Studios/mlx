from __future__ import annotations

from typing import Optional

from ._import_mlx_embeddings import _import_mlx_embeddings


class _EmbeddingRunnerLoadMixin:
    @classmethod
    def load(
        cls,
        model_path: str,
        *,
        lazy: bool = False,
        model_config: Optional[dict] = None,
    ) -> "EmbeddingRunner":
        """Load an embedding model + tokenizer from an HF repo id or local path."""
        mlx_embeddings = _import_mlx_embeddings()
        model, tokenizer = mlx_embeddings.load(
            model_path, model_config=model_config or {}, lazy=lazy
        )
        return cls(model, tokenizer, model_path=model_path)
