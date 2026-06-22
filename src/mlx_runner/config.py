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

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional

# Per-field coercion metadata: name -> (python base type, is_optional).
# Used to parse ``config set KEY VALUE`` strings and to validate loaded JSON.
_FIELD_SPEC = {
    "model": (str, True),
    "system": (str, True),
    "max_tokens": (int, False),
    "temperature": (float, False),
    "top_p": (float, False),
    "top_k": (int, False),
    "min_p": (float, False),
    "repetition_penalty": (float, True),
    "seed": (int, True),
    "max_kv_size": (int, True),
    "kv_bits": (int, True),
    "safety_fraction": (float, False),
}

_NULL_TOKENS = {"", "none", "null", "default", "unset"}


@dataclass
class UserConfig:
    """User-settable defaults applied to CLI invocations.

    A value of ``None`` means "not configured" and falls back to the program's
    own built-in default for that flag.
    """

    model: Optional[str] = None
    system: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 0.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: Optional[float] = None
    seed: Optional[int] = None
    max_kv_size: Optional[int] = None
    kv_bits: Optional[int] = None
    safety_fraction: float = 0.9

    def to_dict(self) -> dict:
        return asdict(self)


def known_keys() -> tuple:
    """The set of settable configuration keys, in definition order."""
    return tuple(f.name for f in fields(UserConfig))


def coerce_value(key: str, raw: str) -> Any:
    """Parse a ``config set`` string into the correct type for ``key``.

    Raises ``KeyError`` for an unknown key and ``ValueError`` for a value that
    cannot be parsed into the field's type. Optional fields accept "none",
    "null", "default", "unset" or "" to mean *clear back to the default*.
    """
    if key not in _FIELD_SPEC:
        raise KeyError(key)
    base, optional = _FIELD_SPEC[key]
    if optional and raw.strip().lower() in _NULL_TOKENS:
        return None
    try:
        if base is int:
            return int(raw)
        if base is float:
            return float(raw)
        return raw
    except ValueError as e:
        raise ValueError(f"cannot parse {raw!r} as {base.__name__} for {key!r}") from e


def default_config_path() -> Path:
    """Resolve the config file path without creating anything."""
    env = os.environ.get("MLX_RUNNER_CONFIG")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "mlx-runner" / "config.json"


def load_config(path: Optional[os.PathLike] = None) -> UserConfig:
    """Load a :class:`UserConfig`, falling back to defaults on any problem.

    Unknown keys in the file are ignored; values are coerced to each field's
    type. A missing or unreadable/malformed file yields a default config.
    """
    cfg = UserConfig()
    p = Path(path) if path is not None else default_config_path()
    try:
        raw = p.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return cfg
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return cfg
    if not isinstance(data, dict):
        return cfg
    for key, value in data.items():
        if key not in _FIELD_SPEC:
            continue
        base, optional = _FIELD_SPEC[key]
        if value is None:
            if optional:
                setattr(cfg, key, None)
            continue
        try:
            setattr(cfg, key, base(value))
        except (TypeError, ValueError):
            # Leave the built-in default in place for a bad value.
            continue
    return cfg


def save_config(config: UserConfig, path: Optional[os.PathLike] = None) -> Path:
    """Write ``config`` to disk as pretty JSON, creating parent dirs. Returns the path."""
    p = Path(path) if path is not None else default_config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
    return p
