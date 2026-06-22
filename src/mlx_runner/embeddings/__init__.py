"""Local text embeddings via ``mlx-embeddings``.

Mirrors :mod:`mlx_runner.runner`'s design: ``mlx_embeddings`` is imported lazily
inside the methods that need it, so this module imports cleanly on any platform
and only requires the Apple-silicon-only dependency when you actually embed.
"""
from ._import_mlx_embeddings import _import_mlx_embeddings
from ._to_nested_list import _to_nested_list
from .cosine_similarity_matrix import cosine_similarity_matrix
from .embedding_runner import EmbeddingRunner

__all__ = [
    "_import_mlx_embeddings",
    "_to_nested_list",
    "cosine_similarity_matrix",
    "EmbeddingRunner",
]
