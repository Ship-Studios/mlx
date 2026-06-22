from __future__ import annotations

import subprocess
import sys
import threading


def _launch_tunnel(port: int):
    """Start a Cloudflare quick tunnel to localhost:port; print the public URL.

    Returns the cloudflared Popen (or None if cloudflared isn't installed). A
    background thread scans its output for the trycloudflare.com URL and prints it.
    """
    import shutil

    if not shutil.which("cloudflared"):
        print(
            "warning: --tunnel requested but `cloudflared` is not installed; serving "
            "locally only. Install it with `brew install cloudflared`.",
            file=sys.stderr,
        )
        return None

    proc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )

    def _watch():
        import re

        pat = re.compile(r"https://[-\w.]+\.trycloudflare\.com")
        for line in proc.stdout:  # type: ignore[union-attr]
            m = pat.search(line)
            if m:
                print(f"\n  Public URL: {m.group(0)}/v1/messages\n", file=sys.stderr)
                break

    threading.Thread(target=_watch, daemon=True).start()
    return proc
