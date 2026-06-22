from __future__ import annotations

from ..config import load_config


def cmd_config_get(args) -> int:
    value = getattr(load_config(), args.key)
    print("" if value is None else value)
    return 0
