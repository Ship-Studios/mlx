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
#   ./serve-tunnel.sh                          # default model (from `mlx-runner config`), port 8080
#   ./serve-tunnel.sh --port 9000 -m repo      # extra args are forwarded to `mlx-runner serve`
#   MLX_KEY=mysecret ./serve-tunnel.sh         # set a specific key (written to the key file)
#   MLX_KEY_FILE=/path/to/key ./serve-tunnel.sh    # use a different key-file location
#   MLX_RUNNER_BIN=/path/to/mlx-runner ./serve-tunnel.sh
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

# --- launch ------------------------------------------------------------------
EXTRA_ARGS=("$@")

info "Starting authenticated Cloudflare tunnel (Ctrl-C to stop)"
warn "Clients send this in the x-api-key header (also saved in $KEY_FILE):"
printf '\n    %s\n\n' "${BOLD}$(cat "$KEY_FILE")${RESET}" >&2
info "Watch below for:  Public URL: https://<random>.trycloudflare.com/v1/messages"

# exec so the server becomes this process (clean Ctrl-C / signal handling).
# --api-key-file keeps the secret out of argv AND the environment.
# ${EXTRA_ARGS[@]+...} guards the empty-array case under `set -u` on macOS bash 3.2.
exec "$RUNNER" serve --api anthropic --api-key-file "$KEY_FILE" --tunnel ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
