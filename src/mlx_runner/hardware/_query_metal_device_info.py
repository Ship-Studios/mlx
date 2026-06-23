from __future__ import annotations

from typing import Optional


def _query_metal_device_info() -> Optional[dict]:
    """Return mlx's metal device_info dict, or ``None`` if unavailable."""
    try:
        import mlx.core as mx

        # Prefer the top-level `mx.device_info()` (newer mlx); fall back to the
        # deprecated `mx.metal.device_info()` only on older versions that lack it,
        # so we don't emit the deprecation warning on current mlx.
        getter = getattr(mx, "device_info", None) or mx.metal.device_info
        info = getter()
        return dict(info) if info else None
    except Exception:  # pragma: no cover - depends on hardware/mlx version
        return None
