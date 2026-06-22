from __future__ import annotations

from typing import Optional

from ._sysctl import _sysctl


def _sysctl_int(key: str) -> Optional[int]:
    val = _sysctl(key)
    if val is None:
        return None
    try:
        return int(val)
    except ValueError:
        return None
