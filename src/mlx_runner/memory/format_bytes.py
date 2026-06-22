from __future__ import annotations

from typing import Optional


def format_bytes(n: Optional[float]) -> str:
    """Human-readable binary size, e.g. ``format_bytes(7e9) == '6.52 GiB'``.

    Args:
        n: Number of bytes (or None).

    Returns:
        Human-readable size string.
    """
    if n is None:
        return "unknown"
    n = float(n)
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    i = 0
    while abs(n) >= 1024.0 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    if units[i] == "B":
        return f"{int(n)} B"
    sign = "" if n >= 0 else "-"
    return f"{sign}{abs(n):.2f} {units[i]}"
