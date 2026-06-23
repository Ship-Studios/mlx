"""Tests for the MLX memory guard applied before a model load.

The guard caps ``mx.set_memory_limit`` so an oversized load raises a catchable
OOM instead of panicking the GPU driver (Apple ``IOGPUFamily`` kernel panic). It
must be best-effort: a silent no-op when mlx is missing, too old to expose
``set_memory_limit``, or reports no working-set size.
"""
import sys
import types

import pytest

from mlx_runner.runner._apply_memory_guard import _apply_memory_guard

WSS = 24 * 1024 ** 3  # a plausible recommended working-set size


def _install_fake_mlx(monkeypatch, *, device_info=None, has_set_limit=True, wss=WSS):
    captured = {"limit": None}
    mlx = types.ModuleType("mlx")
    mlx_core = types.ModuleType("mlx.core")

    if device_info is None and wss is not None:
        device_info = {"max_recommended_working_set_size": wss}
    mlx_core.metal = types.SimpleNamespace(device_info=lambda: device_info)

    if has_set_limit:
        def set_memory_limit(limit):
            captured["limit"] = limit
            return 999  # previous limit
        mlx_core.set_memory_limit = set_memory_limit

    mlx.core = mlx_core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mlx_core)
    return captured


def test_guard_caps_limit_to_fraction_of_working_set(monkeypatch):
    captured = _install_fake_mlx(monkeypatch)
    prev = _apply_memory_guard(fraction=0.8)
    assert captured["limit"] == int(WSS * 0.8)
    assert prev == 999  # returns the previous limit


def test_guard_default_fraction_is_below_one(monkeypatch):
    # The whole point is to set the ceiling *below* MLX's 1.5x default.
    captured = _install_fake_mlx(monkeypatch)
    _apply_memory_guard()
    assert 0 < captured["limit"] < WSS


def test_guard_noop_when_mlx_missing(monkeypatch):
    # No mlx in sys.modules and unimportable -> None, no raise.
    monkeypatch.setitem(sys.modules, "mlx", None)
    assert _apply_memory_guard() is None


def test_guard_noop_when_set_memory_limit_absent(monkeypatch):
    # Older mlx without set_memory_limit -> silent no-op.
    _install_fake_mlx(monkeypatch, has_set_limit=False)
    assert _apply_memory_guard() is None


def test_guard_noop_when_no_working_set_size(monkeypatch):
    captured = _install_fake_mlx(monkeypatch, device_info={})  # missing the key
    assert _apply_memory_guard() is None
    assert captured["limit"] is None


def test_guard_noop_when_working_set_zero(monkeypatch):
    captured = _install_fake_mlx(monkeypatch, wss=0)
    assert _apply_memory_guard() is None
    assert captured["limit"] is None
