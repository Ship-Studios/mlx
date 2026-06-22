from __future__ import annotations

import json
import sys

from ..hardware import detect_hardware
from ..memory import (
    check_fit,
    estimate_model_memory,
    format_bytes,
    parse_param_count,
    recommend_quantization,
)


def cmd_fit(args) -> int:
    try:
        num_params = parse_param_count(args.params)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    hw = detect_hardware()
    available = hw.recommended_working_set_bytes or hw.total_ram_bytes
    if not available:
        print("error: could not determine available memory.", file=sys.stderr)
        return 2

    est = estimate_model_memory(
        num_params,
        quant_bits=args.bits,
        dtype=args.dtype,
        group_size=args.group_size,
        num_layers=args.layers,
        num_kv_heads=args.kv_heads,
        head_dim=args.head_dim,
        seq_len=args.seq_len,
    )
    fit = check_fit(est, available, safety_fraction=args.safety)
    rec = recommend_quantization(num_params, available, safety_fraction=args.safety)

    if args.json:
        print(
            json.dumps(
                {
                    "num_params": num_params,
                    "estimate": est.to_dict(),
                    "fit": fit.to_dict(),
                    "recommended": rec,
                    "available_bytes": available,
                },
                indent=2,
            )
        )
        return 0 if fit.fits else 1

    label = f"{num_params / 1e9:.2f}B params" if num_params >= 1e9 else f"{num_params:,} params"
    quant = f"{args.bits}-bit" if args.bits else f"{args.dtype}"
    print(f"Model:     {label} @ {quant}")
    print(f"Estimate:  {est.human()}")
    print(f"Memory:    {fit.human()}")
    rec_label = f"{rec['quant_bits']}-bit" if rec["quant_bits"] else "full precision (fp16)"
    if rec["fits"]:
        print(f"Best fit:  {rec_label} ({format_bytes(rec['weights_bytes'])} weights)")
    else:
        print("Best fit:  even 2-bit weights exceed the budget on this machine.")
    return 0 if fit.fits else 1
