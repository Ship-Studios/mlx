from __future__ import annotations

from http.server import ThreadingHTTPServer
from typing import Optional

from .make_handler import make_handler


def serve(
    model_path: str,
    host: str = "127.0.0.1",
    port: int = 8080,
    *,
    adapter_path: Optional[str] = None,
    trust_remote_code: bool = False,
    api_key: Optional[str] = None,
) -> int:
    """Load the model and serve the Anthropic Messages API until interrupted."""
    from ..runner import ModelRunner

    runner = ModelRunner.load(model_path, adapter_path=adapter_path, trust_remote_code=trust_remote_code)
    handler = make_handler(runner, api_key=api_key, model=model_path)
    httpd = ThreadingHTTPServer((host, port), handler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        httpd.server_close()
    return 0
