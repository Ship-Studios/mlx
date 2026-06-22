from __future__ import annotations

import argparse

from ..config import UserConfig


def _add_generation_args(p: argparse.ArgumentParser, config: UserConfig) -> None:
    g = p.add_argument_group("generation")
    g.add_argument("--max-tokens", type=int, default=config.max_tokens)
    g.add_argument(
        "--temp", "--temperature", dest="temperature", type=float,
        default=config.temperature,
    )
    g.add_argument("--top-p", type=float, default=config.top_p)
    g.add_argument("--top-k", type=int, default=config.top_k)
    g.add_argument("--min-p", type=float, default=config.min_p)
    g.add_argument("--repetition-penalty", type=float, default=config.repetition_penalty)
    g.add_argument("--seed", type=int, default=config.seed)
    g.add_argument("--max-kv-size", type=int, default=config.max_kv_size)
    g.add_argument("--kv-bits", type=int, default=config.kv_bits)
