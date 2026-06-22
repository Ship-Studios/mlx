from __future__ import annotations


def parse_param_count(value) -> int:
    """Parse a parameter count like ``'7B'``, ``'1.5B'``, ``'350M'``, ``'7e9'``.

    Accepts an int, a plain numeric string, or a string with a K/M/B/T/G suffix
    (case-insensitive). Underscores are ignored.
    """
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip().lower().replace("_", "").replace(",", "")
    if not s:
        raise ValueError("empty parameter count")
    multipliers = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000, "t": 1_000_000_000_000, "g": 1_000_000_000}
    mult = 1
    if s[-1] in multipliers:
        mult = multipliers[s[-1]]
        s = s[:-1]
    try:
        n = float(s) * mult
    except ValueError as e:
        raise ValueError(f"could not parse parameter count {value!r}") from e
    if n <= 0:
        raise ValueError("parameter count must be positive")
    return int(n)
