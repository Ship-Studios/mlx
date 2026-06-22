from __future__ import annotations

from ..config import default_config_path


def cmd_config_path(args) -> int:
    print(default_config_path())
    return 0
