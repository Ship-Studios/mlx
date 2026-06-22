from __future__ import annotations

from typing import Any

from ._constants import _FIELD_SPEC, _NULL_TOKENS


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
