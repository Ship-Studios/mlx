from __future__ import annotations

from ..config import UserConfig, load_config, save_config


def cmd_config_unset(args) -> int:
    config = load_config()
    setattr(config, args.key, getattr(UserConfig(), args.key))
    path = save_config(config)
    print(f"{args.key} reset to default  -> {path}")
    return 0
