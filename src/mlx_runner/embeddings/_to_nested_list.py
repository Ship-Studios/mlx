from __future__ import annotations

from typing import List


def _to_nested_list(array) -> List[List[float]]:
    """Convert an mlx (or numpy) array to a plain nested list of floats."""
    tolist = getattr(array, "tolist", None)
    if tolist is not None:
        result = tolist()
    else:  # pragma: no cover - fallback for unusual array types
        result = [list(row) for row in array]
    # A single 1-D vector becomes [[...]] so callers always get a list of rows.
    if result and not isinstance(result[0], list):
        return [result]
    return result
