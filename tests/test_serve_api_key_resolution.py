"""`serve` API-key resolution: --api-key flag > --api-key-file, fail-closed.

The file path keeps the secret out of argv (ps) and the process environment; a
file-supplied key must enforce x-api-key exactly like the flag, and an unreadable
/ empty --api-key-file must fail closed rather than start an unauthenticated
server (a public --tunnel with no key is worse than erroring out).
"""
import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

import mlx_runner.anthropic_server as a
from mlx_runner.cli._resolve_api_key import _resolve_api_key


def _args(api_key=None, api_key_file=None):
    return SimpleNamespace(api_key=api_key, api_key_file=api_key_file)


# --- resolution precedence + parsing -----------------------------------------
def test_flag_beats_file(tmp_path):
    f = tmp_path / "key"
    f.write_text("from-file")
    assert _resolve_api_key(_args(api_key="from-flag", api_key_file=str(f))) == "from-flag"


def test_reads_and_strips_file(tmp_path):
    f = tmp_path / "key"
    f.write_text("  secret-token\n")  # trailing newline + padding must be stripped
    assert _resolve_api_key(_args(api_key_file=str(f))) == "secret-token"


def test_none_when_neither_given():
    assert _resolve_api_key(_args()) is None


# --- fail-closed --------------------------------------------------------------
def test_missing_file_exits_2(tmp_path):
    with pytest.raises(SystemExit) as ei:
        _resolve_api_key(_args(api_key_file=str(tmp_path / "nope")))
    assert ei.value.code == 2


def test_empty_file_exits_2(tmp_path):
    f = tmp_path / "key"
    f.write_text("   \n")  # whitespace-only == empty after strip
    with pytest.raises(SystemExit) as ei:
        _resolve_api_key(_args(api_key_file=str(f)))
    assert ei.value.code == 2


# --- integration: a file-resolved key enforces x-api-key like the flag --------
class _FakeRunner:
    def __init__(self, deltas):
        self._deltas = deltas
        self.last_stats = SimpleNamespace(finish_reason="stop", generation_tokens=0)
        self.tokenizer = None

    def stream(self, prompt=None, *, messages=None, system=None, config=None):
        for d in self._deltas:
            yield d


@contextmanager
def _running_server(runner, **kw):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), a.make_handler(runner, **kw))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def _post(base, headers):
    body = json.dumps(
        {"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
    ).encode()
    req = urllib.request.Request(base + "/v1/messages", data=body, headers=headers, method="POST")
    return urllib.request.urlopen(req, timeout=10)


def test_file_resolved_key_enforces_x_api_key(tmp_path):
    f = tmp_path / "key"
    f.write_text("filekey\n")
    key = _resolve_api_key(_args(api_key_file=str(f)))
    assert key == "filekey"

    with _running_server(_FakeRunner(["ok"]), api_key=key) as base:
        # Missing header → 401.
        with pytest.raises(urllib.error.HTTPError) as ei:
            _post(base, {"content-type": "application/json"})
        assert ei.value.code == 401
        # Correct header → 200.
        r = _post(base, {"content-type": "application/json", "x-api-key": "filekey"})
        assert r.status == 200
