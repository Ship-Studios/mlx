from __future__ import annotations

from ._stop_sequence_filter_init import _StopSequenceFilterInitMixin
from ._stop_sequence_filter_feed import _StopSequenceFilterFeedMixin
from ._stop_sequence_filter_flush import _StopSequenceFilterFlushMixin


class StopSequenceFilter(
    _StopSequenceFilterInitMixin,
    _StopSequenceFilterFeedMixin,
    _StopSequenceFilterFlushMixin,
):
    """Detect stop sequences in a stream of text deltas without emitting past them.

    ``feed(text)`` returns ``(safe_to_emit, stopped)``. While no stop sequence has
    appeared it holds back the last ``max_len - 1`` characters (which could be the
    start of a future match) and emits the rest. When a stop sequence completes, it
    returns the text preceding it and ``stopped=True``; ``matched`` names the
    sequence. ``flush()`` returns any held-back tail at natural end of generation.
    """
