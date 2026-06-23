from __future__ import annotations

from typing import Optional


def _apply_memory_guard(fraction: float = 0.8) -> Optional[int]:
    """Cap MLX's memory limit so an over-commit raises instead of panicking.

    MLX defaults its memory limit to **1.5x** the device's recommended working
    set size. On a memory-constrained Mac that lets a model load over-commit
    wired GPU memory and trip an Apple GPU-driver *kernel panic*
    (``IOGPUFamily: "completeMemory() prepare count underflow"``) — taking the
    whole machine down — instead of a catchable in-process ``RuntimeError``.

    We lower the limit to ``fraction`` of the recommended working set so MLX
    raises a Python exception first. The limit only sets a ceiling; it never
    restricts a model that genuinely fits, so it is safe to apply on every load.

    Best-effort and silent: returns the previous limit on success, or ``None``
    when mlx is missing, too old to expose ``set_memory_limit`` /
    ``metal.device_info``, or reports no working-set size. ``fraction`` defaults
    to ``0.8`` to match the serve/setup safety budget.
    """
    try:
        import mlx.core as mx

        info = mx.metal.device_info()
        wss = int(info["max_recommended_working_set_size"])
        if wss <= 0:
            return None
        return int(mx.set_memory_limit(int(wss * fraction)))
    except Exception:  # pragma: no cover - depends on hardware/mlx version
        return None
