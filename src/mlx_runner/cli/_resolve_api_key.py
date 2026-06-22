from __future__ import annotations

import sys
from typing import Optional


def _resolve_api_key(args) -> Optional[str]:
    """Resolve the effective x-api-key for `serve`, preferring secure sources.

    Precedence: ``--api-key`` (convenient but world-readable via ``ps``) wins, else
    read ``--api-key-file`` whose contents never touch argv or the process
    environment. Returns ``None`` when neither is given (an open, unauthenticated
    server).

    Fails CLOSED: if ``--api-key-file`` is given but the file is unreadable or
    empty, exit non-zero rather than silently starting a server with no auth — a
    public ``--tunnel`` with no key is worse than erroring out.
    """
    if args.api_key:
        return args.api_key

    path = getattr(args, "api_key_file", None)
    if not path:
        return None

    try:
        with open(path, "r", encoding="utf-8") as fh:
            key = fh.read().strip()
    except OSError as e:
        print(f"error: could not read --api-key-file {path!r}: {e}", file=sys.stderr)
        raise SystemExit(2)

    if not key:
        print(f"error: --api-key-file {path!r} is empty.", file=sys.stderr)
        raise SystemExit(2)

    return key
