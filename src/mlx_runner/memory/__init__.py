"""Estimate LLM memory footprints and whether a model fits on the local machine.

Pure-Python and dependency-free so it can be unit-tested anywhere (no mlx, no
network, no model download required). All sizes are in bytes unless noted.
"""
from ._constants import _DTYPE_BYTES, _DEFAULT_QUANT_CANDIDATES
from .format_bytes import format_bytes
from .parse_param_count import parse_param_count
from .bytes_per_weight import bytes_per_weight
from .estimate_weights_memory import estimate_weights_memory
from .estimate_kv_cache_memory import estimate_kv_cache_memory
from .memory_estimate import MemoryEstimate
from .estimate_model_memory import estimate_model_memory
from .fit_result import FitResult
from .check_fit import check_fit
from .recommend_quantization import recommend_quantization

__all__ = [
    "_DTYPE_BYTES",
    "_DEFAULT_QUANT_CANDIDATES",
    "format_bytes",
    "parse_param_count",
    "bytes_per_weight",
    "estimate_weights_memory",
    "estimate_kv_cache_memory",
    "MemoryEstimate",
    "estimate_model_memory",
    "FitResult",
    "check_fit",
    "recommend_quantization",
]
