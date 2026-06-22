from __future__ import annotations

from .mlx_not_available_error import MLXNotAvailableError


def _import_mlx_lm():
    try:
        import mlx_lm  # noqa: F401

        return mlx_lm
    except ImportError as e:  # pragma: no cover - exercised via monkeypatch in tests
        raise MLXNotAvailableError(
            "mlx-lm is not installed or not importable. Install it with "
            "`pip install mlx-lm` on an Apple-silicon Mac."
        ) from e
