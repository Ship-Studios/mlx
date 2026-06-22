from __future__ import annotations

import sys

from ..config import coerce_value, load_config, save_config


def cmd_config_set(args) -> int:
    try:
        value = coerce_value(args.key, args.value)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    config = load_config()
    setattr(config, args.key, value)
    path = save_config(config)
    print(f"{args.key} = {value if value is not None else '(default)'}  -> {path}")
    return 0
