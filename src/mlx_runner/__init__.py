"""mlx_runner: a hardware-aware CLI for running local LLMs on Apple silicon via mlx-lm.

Importing this package is cheap and side-effect free: the heavy, Apple-silicon-only
``mlx_lm`` dependency is imported lazily inside :mod:`mlx_runner.runner`, so the
hardware/memory helpers and the CLI scaffolding work everywhere.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .hardware import HardwareInfo, detect_hardware
from .memory import (
    FitResult,
    MemoryEstimate,
    bytes_per_weight,
    check_fit,
    estimate_kv_cache_memory,
    estimate_model_memory,
    estimate_weights_memory,
    format_bytes,
    parse_param_count,
    recommend_quantization,
)
from .runner import (
    GenerationConfig,
    GenerationStats,
    MLXNotAvailableError,
    ModelRunner,
)

__all__ = [
    "__version__",
    # hardware
    "HardwareInfo",
    "detect_hardware",
    # memory
    "FitResult",
    "MemoryEstimate",
    "bytes_per_weight",
    "check_fit",
    "estimate_kv_cache_memory",
    "estimate_model_memory",
    "estimate_weights_memory",
    "format_bytes",
    "parse_param_count",
    "recommend_quantization",
    # runner
    "GenerationConfig",
    "GenerationStats",
    "MLXNotAvailableError",
    "ModelRunner",
]
