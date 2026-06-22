from __future__ import annotations

import os
from pathlib import Path


def default_config_path() -> Path:
    """Resolve the config file path without creating anything."""
    env = os.environ.get("MLX_RUNNER_CONFIG")
    if env:
        return Path(env).expanduser()
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".config"
    return base / "mlx-runner" / "config.json"
