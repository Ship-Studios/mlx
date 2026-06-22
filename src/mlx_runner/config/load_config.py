from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from ._constants import _FIELD_SPEC
from .default_config_path import default_config_path
from .user_config import UserConfig


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
