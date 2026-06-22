from __future__ import annotations


def _config_from_args(args) -> "object":
    # Imported lazily so `info`/`fit` work without mlx-lm present.
    from ..runner import GenerationConfig

    return GenerationConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        max_kv_size=args.max_kv_size,
        kv_bits=args.kv_bits,
    )
