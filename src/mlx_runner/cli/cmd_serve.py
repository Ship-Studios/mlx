from __future__ import annotations

import sys

from ..hardware import detect_hardware
from ._launch_tunnel import _launch_tunnel
from ._serve_anthropic import _serve_anthropic
from ._serve_openai import _serve_openai


def cmd_serve(args) -> int:
    hw = detect_hardware()
    if not hw.can_run_mlx:
        print(
            "warning: this is not an Apple-silicon Mac; the model may not run.",
            file=sys.stderr,
        )

    # --api-key only does anything on the anthropic path; mlx_lm.server has no auth.
    if args.api_key and args.api != "anthropic":
        print(
            "warning: --api-key is only enforced with `--api anthropic`; mlx_lm.server "
            "(the openai path) has no built-in auth, so this key is ignored.",
            file=sys.stderr,
        )

    # A public tunnel in front of an unauthenticated endpoint is a real exposure.
    if args.tunnel:
        if args.api == "anthropic" and not args.api_key:
            print(
                "WARNING: --tunnel will publish an UNAUTHENTICATED model to the public "
                "internet. Anyone with the URL can use your compute. Pass --api-key to "
                "require an x-api-key header.",
                file=sys.stderr,
            )
        elif args.api != "anthropic":
            print(
                "WARNING: --tunnel will publish mlx_lm.server, which has NO authentication, "
                "to the public internet. For an authenticated public endpoint use "
                "`--api anthropic --api-key ...`.",
                file=sys.stderr,
            )

    tunnel = _launch_tunnel(args.port) if args.tunnel else None
    try:
        if args.api == "anthropic":
            return _serve_anthropic(args)
        return _serve_openai(args)
    finally:
        if tunnel is not None:
            tunnel.terminate()
