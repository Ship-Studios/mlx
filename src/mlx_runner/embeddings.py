"""Local text embeddings via ``mlx-embeddings``.

Mirrors :mod:`mlx_runner.runner`'s design: ``mlx_embeddings`` is imported lazily
inside the methods that need it, so this module imports cleanly on any platform
and only requires the Apple-silicon-only dependency when you actually embed.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Union

from .runner import MLXNotAvailableError


def _import_mlx_embeddings():
    try:
        import mlx_embeddings  # noqa: F401

        return mlx_embeddings
    except ImportError as e:  # pragma: no cover - exercised via monkeypatch in tests
        raise MLXNotAvailableError(
            "mlx-embeddings is not installed or not importable. Install it with "
            "`pip install mlx-embeddings` on an Apple-silicon Mac."
        ) from e


def _to_nested_list(array) -> List[List[float]]:
    """Convert an mlx (or numpy) array to a plain nested list of floats."""
    tolist = getattr(array, "tolist", None)
    if tolist is not None:
        result = tolist()
    else:  # pragma: no cover - fallback for unusual array types
        result = [list(row) for row in array]
    # A single 1-D vector becomes [[...]] so callers always get a list of rows.
    if result and not isinstance(result[0], list):
        return [result]
    return result


def cosine_similarity_matrix(vectors: Sequence[Sequence[float]]) -> List[List[float]]:
    """Pairwise cosine similarity for a list of embedding vectors."""
    norms = [math.sqrt(sum(x * x for x in v)) or 1.0 for v in vectors]
    out: List[List[float]] = []
    for i, vi in enumerate(vectors):
        row = []
        for j, vj in enumerate(vectors):
            dot = sum(a * b for a, b in zip(vi, vj))
            row.append(dot / (norms[i] * norms[j]))
        out.append(row)
    return out


class EmbeddingRunner:
    """Loads an mlx-embeddings model once and encodes text into vectors."""

    def __init__(self, model, tokenizer, model_path: Optional[str] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.model_path = model_path

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

    def embed(
        self,
        texts: Union[str, Sequence[str]],
        *,
        max_length: int = 512,
        padding: bool = True,
        truncation: bool = True,
    ) -> List[List[float]]:
        """Encode one or more texts; returns a list of embedding vectors."""
        mlx_embeddings = _import_mlx_embeddings()
        if isinstance(texts, str):
            texts = [texts]
        output = mlx_embeddings.generate(
            self.model,
            self.tokenizer,
            texts=list(texts),
            max_length=max_length,
            padding=padding,
            truncation=truncation,
        )
        return _to_nested_list(output.text_embeds)
