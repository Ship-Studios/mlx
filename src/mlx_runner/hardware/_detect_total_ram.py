from __future__ import annotations

import os
from typing import Optional

from ._sysctl_int import _sysctl_int


def _detect_total_ram() -> Optional[int]:
    # macOS / BSD expose the physical memory size via sysctl.
    v = _sysctl_int("hw.memsize")
    if v:
        return v
    # POSIX fallback (Linux and friends).
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None
