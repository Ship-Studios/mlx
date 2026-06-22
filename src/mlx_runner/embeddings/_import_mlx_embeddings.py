from __future__ import annotations

from ..runner import MLXNotAvailableError


def _import_mlx_embeddings():
    try:
        import mlx_embeddings  # noqa: F401

        return mlx_embeddings
    except ImportError as e:  # pragma: no cover - exercised via monkeypatch in tests
        raise MLXNotAvailableError(
            "mlx-embeddings is not installed or not importable. Install it with "
            "`pip install mlx-embeddings` on an Apple-silicon Mac."
        ) from e
