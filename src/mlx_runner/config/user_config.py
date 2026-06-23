from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class UserConfig:
    """User-settable defaults applied to CLI invocations.

    A value of ``None`` means "not configured" and falls back to the program's
    own built-in default for that flag.
    """

    model: Optional[str] = None
    system: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 0.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: Optional[float] = None
    seed: Optional[int] = None
    max_kv_size: Optional[int] = None
    kv_bits: Optional[int] = None
    safety_fraction: float = 0.8  # leave headroom for KV/activations; avoids GPU over-commit

    def to_dict(self) -> dict:
        return asdict(self)
