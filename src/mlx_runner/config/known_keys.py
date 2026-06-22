from __future__ import annotations

from dataclasses import fields
from typing import Tuple

from .user_config import UserConfig


def known_keys() -> Tuple[str, ...]:
    """The set of settable configuration keys, in definition order."""
    return tuple(f.name for f in fields(UserConfig))
