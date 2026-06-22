from __future__ import annotations

import sys


def _load_runner_or_exit(args):
    from ..runner import MLXNotAvailableError, ModelRunner

    try:
        return ModelRunner.load(
            args.model,
            adapter_path=args.adapter_path,
            trust_remote_code=args.trust_remote_code,
        )
    except MLXNotAvailableError as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(3)
