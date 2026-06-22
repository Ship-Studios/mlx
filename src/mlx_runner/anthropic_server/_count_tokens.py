from __future__ import annotations


def _count_tokens(runner, text: str) -> int:
    """Best-effort token count via the runner's tokenizer; rough fallback otherwise."""
    tok = getattr(runner, "tokenizer", None)
    encode = getattr(tok, "encode", None)
    if encode is not None:
        try:
            return len(encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)
