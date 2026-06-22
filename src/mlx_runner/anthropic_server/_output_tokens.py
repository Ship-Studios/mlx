from __future__ import annotations

from ._count_tokens import _count_tokens


def _output_tokens(runner, text: str, *, stopped: bool = False) -> int:
    # On a stop-sequence interruption the generator is abandoned before the runner
    # updates last_stats, so last_stats holds the *previous* request's count (the
    # runner is shared). Count from the emitted text in that case instead.
    if not stopped:
        stats = getattr(runner, "last_stats", None)
        gen = getattr(stats, "generation_tokens", None) if stats else None
        if isinstance(gen, int) and gen > 0:
            return gen
    return _count_tokens(runner, text)
