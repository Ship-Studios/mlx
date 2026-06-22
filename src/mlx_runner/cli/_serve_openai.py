from __future__ import annotations

import subprocess
import sys


def _serve_openai(args) -> int:
    cmd = [sys.executable, "-m", "mlx_lm.server", "--host", args.host, "--port", str(args.port)]
    if args.model:
        cmd += ["--model", args.model]
    if args.adapter_path:
        cmd += ["--adapter-path", args.adapter_path]
    if args.trust_remote_code:
        cmd += ["--trust-remote-code"]
    # argparse.REMAINDER keeps a leading "--"; drop it before forwarding.
    extra = [a for a in (args.server_args or []) if a != "--"]
    cmd += extra

    print(f"Serving OpenAI-compatible API on http://{args.host}:{args.port}/v1  (Ctrl-C to stop)", file=sys.stderr)
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        print(
            "error: could not launch mlx_lm.server. Install it with `pip install mlx-lm`.",
            file=sys.stderr,
        )
        return 3
    except KeyboardInterrupt:
        return 0
