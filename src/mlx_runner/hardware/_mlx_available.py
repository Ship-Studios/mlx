from __future__ import annotations

import importlib.util


def _mlx_available() -> bool:
    """True if the ``mlx`` package is importable (does not import it)."""
    try:
        return importlib.util.find_spec("mlx") is not None
    except (ImportError, ValueError):
        return False
