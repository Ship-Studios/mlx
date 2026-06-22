from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class GenerationConfig:
    """Sampling and decoding parameters for a generation request."""

    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 0.0
    top_k: int = 0
    min_p: float = 0.0
    min_tokens_to_keep: int = 1
    repetition_penalty: Optional[float] = None
    repetition_context_size: int = 20
    logit_bias: Optional[Dict[int, float]] = None
    seed: Optional[int] = None
    max_kv_size: Optional[int] = None
    kv_bits: Optional[int] = None
