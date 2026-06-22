from __future__ import annotations

import sys


def _serve_anthropic(args) -> int:
    if not args.model:
        print(
            "error: serve --api anthropic needs a model (pass --model or set a default "
            "with `mlx-runner config set model ...`).",
            file=sys.stderr,
        )
        return 2
    from ..anthropic_server import serve as serve_anthropic

    print(
        f"Serving Anthropic Messages API on http://{args.host}:{args.port}/v1/messages  "
        "(Ctrl-C to stop)",
        file=sys.stderr,
    )
    try:
        return serve_anthropic(
            args.model, args.host, args.port,
            adapter_path=args.adapter_path,
            trust_remote_code=args.trust_remote_code,
            api_key=args.api_key,
        )
    except KeyboardInterrupt:
        return 0
