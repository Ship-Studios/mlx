from __future__ import annotations

import subprocess
from typing import Optional


def _sysctl(key: str) -> Optional[str]:
    """Return a sysctl value as text, or ``None`` if unavailable."""
    try:
        out = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    val = out.stdout.strip()
    return val or None
