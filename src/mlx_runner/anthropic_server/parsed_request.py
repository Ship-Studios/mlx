from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ParsedRequest:
    """A validated, normalized ``/v1/messages`` request."""

    model: str
    max_tokens: int
    messages: List[dict]  # normalized to [{"role": ..., "content": <str>}]
    system: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: List[str] = field(default_factory=list)
    stream: bool = False
