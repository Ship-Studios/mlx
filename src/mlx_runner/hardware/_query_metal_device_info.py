from __future__ import annotations

from typing import Optional


def _query_metal_device_info() -> Optional[dict]:
    """Return mlx's metal device_info dict, or ``None`` if unavailable."""
    try:
        import mlx.core as mx

        info = mx.metal.device_info()
        return dict(info) if info else None
    except Exception:  # pragma: no cover - depends on hardware/mlx version
        return None
