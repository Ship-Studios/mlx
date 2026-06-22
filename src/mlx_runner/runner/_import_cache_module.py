from __future__ import annotations

from ._import_mlx_lm import _import_mlx_lm


def _import_cache_module():
    """Lazily import mlx-lm's prompt-cache helpers."""
    _import_mlx_lm()  # surface a friendly error if mlx-lm is missing
    from mlx_lm.models import cache as cache_mod

    return cache_mod
