from __future__ import annotations

from typing import List, Sequence, Union

from ._import_mlx_embeddings import _import_mlx_embeddings
from ._to_nested_list import _to_nested_list


class _EmbeddingRunnerEmbedMixin:
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
