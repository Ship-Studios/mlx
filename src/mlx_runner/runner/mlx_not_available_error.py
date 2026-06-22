from __future__ import annotations


class MLXNotAvailableError(RuntimeError):
    """Raised when mlx-lm is needed but cannot be imported (e.g. non Apple silicon)."""
