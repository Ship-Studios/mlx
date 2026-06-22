from __future__ import annotations

import sys

from ._download_model import _download_model


def cmd_download(args) -> int:
    repo_id = args.model
    if not repo_id:
        print(
            "error: no model given and no default configured "
            "(pass a repo id or run `mlx-runner config set model ...`).",
            file=sys.stderr,
        )
        return 2
    print(f"Downloading {repo_id} ...", file=sys.stderr)
    try:
        path = _download_model(repo_id)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 3
    except Exception as e:  # network / repo errors from hub
        print(f"error: download failed: {e}", file=sys.stderr)
        return 1
    print(path)
    return 0
