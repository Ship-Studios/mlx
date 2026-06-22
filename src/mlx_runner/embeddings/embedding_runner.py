from __future__ import annotations

from ._embedding_runner_init import _EmbeddingRunnerInitMixin
from ._embedding_runner_load import _EmbeddingRunnerLoadMixin
from ._embedding_runner_embed import _EmbeddingRunnerEmbedMixin


class EmbeddingRunner(
    _EmbeddingRunnerInitMixin,
    _EmbeddingRunnerLoadMixin,
    _EmbeddingRunnerEmbedMixin,
):
    """Loads an mlx-embeddings model once and encodes text into vectors."""
