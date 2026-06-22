"""Persisted user defaults for mlx_runner.

A small JSON file lets you set a default model and default generation parameters
so you don't have to repeat them on every invocation. Resolution order for the
file location:

1. ``$MLX_RUNNER_CONFIG`` if set,
2. ``$XDG_CONFIG_HOME/mlx-runner/config.json`` if ``XDG_CONFIG_HOME`` is set,
3. ``~/.config/mlx-runner/config.json``.

Everything is best-effort and dependency-free: a missing or malformed file
yields built-in defaults rather than raising, so ``info``/``fit``/``generate``
keep working even with no config present.
"""
from __future__ import annotations

from ._constants import _FIELD_SPEC, _NULL_TOKENS
from .coerce_value import coerce_value
from .default_config_path import default_config_path
from .known_keys import known_keys
from .load_config import load_config
from .save_config import save_config
from .user_config import UserConfig

__all__ = [
    "_FIELD_SPEC",
    "_NULL_TOKENS",
    "UserConfig",
    "known_keys",
    "coerce_value",
    "default_config_path",
    "load_config",
    "save_config",
]
