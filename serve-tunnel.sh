#!/usr/bin/env bash
#
# serve-tunnel.sh — serve the local MLX model over an authenticated Cloudflare tunnel.
#
# Ensures an API key file exists, then runs:
#   mlx-runner serve --api anthropic --api-key-file <file> --tunnel
# so the in-process Anthropic Messages API is reachable at a public
# https://<random>.trycloudflare.com/v1/messages URL, gated by the x-api-key header.
#
# The key is read by `mlx-runner serve` directly from a chmod-600 file, so the
# secret never appears in argv (visible via `ps aux`) or the process environment.
#
# Usage:
#   ./serve-tunnel.sh                          # default model (from `mlx-runner config`)
#   ./serve-tunnel.sh --port 9000 -m repo      # extra args are forwarded to `mlx-runner serve`
#   PORT=9000 ./serve-tunnel.sh                # base port (default 8080); auto-advances if busy
#   MLX_KEY=mysecret ./serve-tunnel.sh         # set a specific key (written to the key file)
#   MLX_KEY_FILE=/path/to/key ./serve-tunnel.sh    # use a different key-file location
#   MLX_RUNNER_BIN=/path/to/mlx-runner ./serve-tunnel.sh
#
# If no explicit --port is given, the script picks the first FREE port starting at
# $PORT (default 8080), so a leftover server on the default port won't crash the
# launch. The Cloudflare tunnel follows whichever port is chosen.
#
# The key persists at $MLX_KEY_FILE (default ~/.config/mlx-runner/serve-key) so the
# URL's clients keep working across restarts. Delete that file to rotate the key.

set -euo pipefail

# --- logging -----------------------------------------------------------------
if [[ -t 1 ]]; then
  BOLD=$'\033[1m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; RESET=$'\033[0m'
else
  BOLD=""; GREEN=""; YELLOW=""; RED=""; RESET=""
fi
info() { printf '%s\n' "${BOLD}==>${RESET} $*" >&2; }
ok()   { printf '%s\n' "  ${GREEN}✓${RESET} $*" >&2; }
warn() { printf '%s\n' "  ${YELLOW}!${RESET} $*" >&2; }
die()  { printf '%s\n' "${RED}error:${RESET} $*" >&2; exit 1; }

# --- locate mlx-runner -------------------------------------------------------
RUNNER="${MLX_RUNNER_BIN:-}"
if [[ -z "$RUNNER" ]]; then
  if command -v mlx-runner >/dev/null 2>&1; then
    RUNNER="$(command -v mlx-runner)"
  elif [[ -x "$HOME/.local/bin/mlx-runner" ]]; then
    RUNNER="$HOME/.local/bin/mlx-runner"
  else
    die "mlx-runner not found. Install it first (./install.sh) or set MLX_RUNNER_BIN."
  fi
fi

command -v cloudflared >/dev/null 2>&1 \
  || die "cloudflared not installed — needed for the tunnel. Install with: brew install cloudflared"

# Fail closed: refuse to expose a tunnel unless the installed mlx-runner can read
# the key from a file (--api-key-file). An older build would ignore it and start an
# UNAUTHENTICATED public server. (serve is itself fail-closed on an unreadable/empty
# key file; this guards the staler case where the flag doesn't exist at all.)
"$RUNNER" serve --help 2>&1 | grep -q -- '--api-key-file' \
  || die "installed mlx-runner lacks --api-key-file (predates the auth fix). Reinstall: ./install.sh --skip-setup"

# --- API key file: MLX_KEY env > existing file > freshly generated -----------
KEY_FILE="${MLX_KEY_FILE:-${XDG_CONFIG_HOME:-$HOME/.config}/mlx-runner/serve-key}"

gen_key() {
  if command -v openssl >/dev/null 2>&1; then
    openssl rand -hex 24
  elif command -v python3 >/dev/null 2>&1; then
    python3 -c 'import secrets; print(secrets.token_hex(24))'
  else
    head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n'
  fi
}

umask 077                       # any file/dir we create here is owner-only
mkdir -p "$(dirname "$KEY_FILE")"
if [[ -n "${MLX_KEY:-}" ]]; then
  printf '%s\n' "$MLX_KEY" > "$KEY_FILE"
  chmod 600 "$KEY_FILE"
  ok "Wrote the provided MLX_KEY to $KEY_FILE"
elif [[ -s "$KEY_FILE" ]]; then
  ok "Using saved API key from $KEY_FILE"
else
  gen_key > "$KEY_FILE"
  chmod 600 "$KEY_FILE"
  ok "Generated a new API key at $KEY_FILE (chmod 600)"
fi

# --- choose a free port ------------------------------------------------------
EXTRA_ARGS=("$@")

# Print the first free TCP port at/above $1 on 127.0.0.1 (matches the server's
# SO_REUSEADDR so we agree on what's bindable); exit 1 if none in a 100-port window.
free_port() {
  command -v python3 >/dev/null 2>&1 || { printf '%s\n' "$1"; return 0; }
  python3 - "$1" <<'PY'
import socket, sys
start = int(sys.argv[1])
for p in range(start, start + 100):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("127.0.0.1", p))
    except OSError:
        s.close(); continue
    s.close(); print(p); sys.exit(0)
sys.exit(1)
PY
}

PORT_ARGS=()
explicit_port=0
for a in ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}; do
  case "$a" in --port|--port=*) explicit_port=1 ;; esac
done
if [[ "$explicit_port" -eq 0 ]]; then
  base_port="${PORT:-8080}"
  if port="$(free_port "$base_port")"; then
    [[ "$port" != "$base_port" ]] && warn "port $base_port is in use — using free port $port instead."
    PORT_ARGS=(--port "$port")
  else
    die "no free port found in ${base_port}-$((base_port + 99)); free one up or pass --port."
  fi
fi

# --- launch ------------------------------------------------------------------
info "Starting authenticated Cloudflare tunnel (Ctrl-C to stop)"
warn "Clients send this in the x-api-key header (also saved in $KEY_FILE):"
printf '\n    %s\n\n' "${BOLD}$(cat "$KEY_FILE")${RESET}" >&2
info "Watch below for:  Public URL: https://<random>.trycloudflare.com/v1/messages"

# exec so the server becomes this process (clean Ctrl-C / signal handling).
# --api-key-file keeps the secret out of argv AND the environment.
# ${...[@]+...} guards empty arrays under `set -u` on macOS bash 3.2.
exec "$RUNNER" serve --api anthropic --api-key-file "$KEY_FILE" --tunnel \
  ${PORT_ARGS[@]+"${PORT_ARGS[@]}"} ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
