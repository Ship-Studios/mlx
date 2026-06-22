from __future__ import annotations

import json

from ..config import default_config_path, known_keys, load_config


def cmd_config_show(args) -> int:
    config = load_config()
    if args.json:
        print(json.dumps(config.to_dict(), indent=2))
        return 0
    print(f"# {default_config_path()}")
    for key in known_keys():
        value = getattr(config, key)
        print(f"{key} = {value if value is not None else '(default)'}")
    return 0
