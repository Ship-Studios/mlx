from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from .default_config_path import default_config_path
from .user_config import UserConfig


def save_config(config: UserConfig, path: Optional[os.PathLike] = None) -> Path:
    """Write ``config`` to disk as pretty JSON, creating parent dirs. Returns the path."""
    p = Path(path) if path is not None else default_config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
    return p
