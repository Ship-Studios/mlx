"""Estimate LLM memory footprints and whether a model fits on the local machine.

Pure-Python and dependency-free so it can be unit-tested anywhere (no mlx, no
network, no model download required). All sizes are in bytes unless noted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional, Tuple

# Average bytes per parameter for common unquantized dtypes.
_DTYPE_BYTES = {
    "float32": 4.0, "fp32": 4.0, "f32": 4.0,
    "float16": 2.0, "fp16": 2.0, "f16": 2.0,
    "bfloat16": 2.0, "bf16": 2.0,
    "float64": 8.0, "fp64": 8.0,
}

# Highest-quality-first; used by recommend_quantization. ``None`` means no
# quantization (full ``dtype`` precision).
_DEFAULT_QUANT_CANDIDATES: Tuple[Optional[int], ...] = (None, 8, 6, 4, 3, 2)


def format_bytes(n: Optional[float]) -> str:
    """Human-readable binary size, e.g. ``format_bytes(7e9) == '6.52 GiB'``."""
    if n is None:
        return "unknown"
    n = float(n)
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    i = 0
    while abs(n) >= 1024.0 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    if units[i] == "B":
        return f"{int(n)} B"
    return f"{n:.2f} {units[i]}"


def parse_param_count(value) -> int:
    """Parse a parameter count like ``'7B'``, ``'1.5B'``, ``'350M'``, ``'7e9'``.

    Accepts an int, a plain numeric string, or a string with a K/M/B/T suffix
    (case-insensitive). Underscores are ignored.
    """
    if isinstance(value, (int, float)):
        return int(value)
    s = str(value).strip().lower().replace("_", "").replace(",", "")
    if not s:
        raise ValueError("empty parameter count")
    multipliers = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000, "t": 1_000_000_000_000}
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


def bytes_per_weight(
    quant_bits: Optional[int] = None,
    dtype: str = "float16",
    group_size: int = 64,
) -> float:
    """Average bytes stored per weight parameter.

    For quantized weights, MLX stores ``quant_bits`` per weight plus one fp16
    scale and one fp16 bias per group of ``group_size`` weights (4 extra bytes
    per group). For unquantized weights it is just the dtype width.
    """
    if quant_bits is not None:
        if quant_bits <= 0:
            raise ValueError("quant_bits must be positive")
        if group_size <= 0:
            raise ValueError("group_size must be positive")
        return quant_bits / 8.0 + 4.0 / group_size
    key = dtype.lower()
    if key not in _DTYPE_BYTES:
        raise ValueError(f"unknown dtype {dtype!r}; known: {sorted(_DTYPE_BYTES)}")
    return _DTYPE_BYTES[key]


def estimate_weights_memory(
    num_params: int,
    quant_bits: Optional[int] = None,
    dtype: str = "float16",
    group_size: int = 64,
) -> int:
    """Bytes required to hold the model weights."""
    if num_params <= 0:
        raise ValueError("num_params must be positive")
    return int(num_params * bytes_per_weight(quant_bits, dtype, group_size))


def estimate_kv_cache_memory(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    kv_bits: Optional[int] = None,
    batch_size: int = 1,
) -> int:
    """Bytes for the key/value cache at a given sequence length.

    Stores both K and V: ``2 * layers * kv_heads * head_dim * seq_len * batch``
    elements. Defaults to fp16 (2 bytes); pass ``kv_bits`` for a quantized cache.
    """
    if min(num_layers, num_kv_heads, head_dim, seq_len, batch_size) <= 0:
        raise ValueError("kv-cache dimensions must all be positive")
    bytes_per_elem = (kv_bits / 8.0) if kv_bits else 2.0
    elems = 2 * num_layers * num_kv_heads * head_dim * seq_len * batch_size
    return int(elems * bytes_per_elem)


@dataclass
class MemoryEstimate:
    """Breakdown of estimated memory for running a model."""

    num_params: int
    weights_bytes: int
    kv_cache_bytes: int
    overhead_bytes: int

    @property
    def total_bytes(self) -> int:
        return self.weights_bytes + self.kv_cache_bytes + self.overhead_bytes

    def human(self) -> str:
        return (
            f"weights={format_bytes(self.weights_bytes)}, "
            f"kv_cache={format_bytes(self.kv_cache_bytes)}, "
            f"overhead={format_bytes(self.overhead_bytes)}, "
            f"total={format_bytes(self.total_bytes)}"
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total_bytes"] = self.total_bytes
        return d


def estimate_model_memory(
    num_params: int,
    quant_bits: Optional[int] = None,
    dtype: str = "float16",
    group_size: int = 64,
    *,
    num_layers: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
    seq_len: int = 4096,
    kv_bits: Optional[int] = None,
    batch_size: int = 1,
    overhead_fraction: float = 0.05,
) -> MemoryEstimate:
    """Estimate total memory to load weights plus (optionally) a KV cache.

    The KV-cache term is only included when ``num_layers``, ``num_kv_heads`` and
    ``head_dim`` are all provided; otherwise it is zero. ``overhead_fraction``
    accounts for activations, framework buffers, and fragmentation.
    """
    weights = estimate_weights_memory(num_params, quant_bits, dtype, group_size)
    kv = 0
    if num_layers and num_kv_heads and head_dim:
        kv = estimate_kv_cache_memory(
            num_layers, num_kv_heads, head_dim, seq_len, kv_bits, batch_size
        )
    overhead = int(weights * overhead_fraction)
    return MemoryEstimate(
        num_params=num_params,
        weights_bytes=weights,
        kv_cache_bytes=kv,
        overhead_bytes=overhead,
    )


@dataclass
class FitResult:
    """Result of checking an estimate against an available-memory budget."""

    required_bytes: int
    available_bytes: int
    budget_bytes: int
    safety_fraction: float
    fits: bool
    headroom_bytes: int

    def human(self) -> str:
        verdict = "FITS" if self.fits else "DOES NOT FIT"
        sign = "" if self.headroom_bytes < 0 else "+"
        return (
            f"{verdict}: needs {format_bytes(self.required_bytes)} vs budget "
            f"{format_bytes(self.budget_bytes)} "
            f"({int(self.safety_fraction * 100)}% of {format_bytes(self.available_bytes)}); "
            f"headroom {sign}{format_bytes(self.headroom_bytes)}"
        )

    def to_dict(self) -> dict:
        return asdict(self)


def check_fit(estimate, available_bytes: int, safety_fraction: float = 0.9) -> FitResult:
    """Check whether ``estimate`` fits inside ``safety_fraction`` of available memory.

    ``estimate`` may be a :class:`MemoryEstimate` or a raw byte count.
    """
    if not 0 < safety_fraction <= 1:
        raise ValueError("safety_fraction must be in (0, 1]")
    required = (
        estimate.total_bytes if isinstance(estimate, MemoryEstimate) else int(estimate)
    )
    budget = int(available_bytes * safety_fraction)
    return FitResult(
        required_bytes=required,
        available_bytes=int(available_bytes),
        budget_bytes=budget,
        safety_fraction=safety_fraction,
        fits=required <= budget,
        headroom_bytes=budget - required,
    )


def recommend_quantization(
    num_params: int,
    available_bytes: int,
    dtype: str = "float16",
    group_size: int = 64,
    safety_fraction: float = 0.9,
    candidates: Tuple[Optional[int], ...] = _DEFAULT_QUANT_CANDIDATES,
) -> dict:
    """Recommend the highest-quality precision whose weights fit the budget.

    Returns a dict with ``quant_bits`` (``None`` == full precision),
    ``weights_bytes`` and ``fits``. If nothing fits, returns the smallest
    candidate with ``fits=False``.
    """
    budget = int(available_bytes * safety_fraction)
    best_small = None
    for bits in candidates:
        weights = estimate_weights_memory(num_params, bits, dtype, group_size)
        best_small = {"quant_bits": bits, "weights_bytes": weights, "fits": weights <= budget}
        if weights <= budget:
            return best_small
    return best_small or {"quant_bits": None, "weights_bytes": 0, "fits": False}
