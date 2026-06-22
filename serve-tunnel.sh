#!/usr/bin/env bash
#
# serve-tunnel.sh — serve the local MLX model over an authenticated Cloudflare tunnel.
#
# Generates (and persists) an API key, then runs:
#   mlx-runner serve --api anthropic --api-key <key> --tunnel
# so the in-process Anthropic Messages API is reachable at a public
# https://<random>.trycloudflare.com/v1/messages URL, gated by the x-api-key header.
#
# Usage:
#   ./serve-tunnel.sh                      # default model (from `mlx-runner config`), port 8080
#   ./serve-tunnel.sh --port 9000 -m repo  # extra args are forwarded to `mlx-runner serve`
#   MLX_KEY=mysecret ./serve-tunnel.sh     # use a specific key instead of the persisted one
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

# --- API key: env > persisted file > freshly generated -----------------------
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

if [[ -n "${MLX_KEY:-}" ]]; then
  ok "Using API key from the MLX_KEY environment variable."
elif [[ -f "$KEY_FILE" ]]; then
  MLX_KEY="$(cat "$KEY_FILE")"
  ok "Using saved API key from $KEY_FILE"
else
  info "Generating a new API key"
  MLX_KEY="$(gen_key)"
  mkdir -p "$(dirname "$KEY_FILE")"
  printf '%s\n' "$MLX_KEY" > "$KEY_FILE"
  chmod 600 "$KEY_FILE"
  ok "Saved new API key to $KEY_FILE (chmod 600)"
fi

# --- launch ------------------------------------------------------------------
EXTRA_ARGS=("$@")

info "Starting authenticated Cloudflare tunnel (Ctrl-C to stop)"
warn "Clients must send this in the x-api-key header:"
printf '\n    %s\n\n' "${BOLD}${MLX_KEY}${RESET}" >&2
info "Watch below for:  Public URL: https://<random>.trycloudflare.com/v1/messages"

# exec so the server becomes this process (clean Ctrl-C / signal handling).
# ${EXTRA_ARGS[@]+...} guards the empty-array case under `set -u` on macOS bash 3.2.
exec "$RUNNER" serve --api anthropic --api-key "$MLX_KEY" --tunnel ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
