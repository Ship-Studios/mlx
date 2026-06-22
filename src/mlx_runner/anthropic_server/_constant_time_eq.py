from __future__ import annotations


def _constant_time_eq(a: str, b: str) -> bool:
    import hmac

    return hmac.compare_digest(a, b)
