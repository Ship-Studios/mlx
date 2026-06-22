#!/usr/bin/env bash
#
# install.sh — bootstrap a fresh Apple-silicon Mac into an mlx-lm LLM runner.
#
# It verifies the machine, ensures a recent enough Python, installs the
# `mlx-runner` package (preferring pipx for an isolated install), and then runs
# `mlx-runner setup` to pick + download a model and set it as the default.
#
# Usage (from a clone of this repo):
#   ./install.sh                 # install, then run `mlx-runner setup`
#   ./install.sh --skip-setup    # install only
#   ./install.sh -m mlx-community/Qwen2.5-7B-Instruct-4bit   # forwarded to setup
#
# Any argument other than --skip-setup is forwarded verbatim to `mlx-runner setup`.

set -euo pipefail

MIN_PY_MINOR=9            # mlx-lm requires Python 3.9+
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKIP_SETUP=0
SETUP_ARGS=()

for arg in "$@"; do
  if [[ "$arg" == "--skip-setup" ]]; then
    SKIP_SETUP=1
  else
    SETUP_ARGS+=("$arg")
  fi
done

# --- pretty logging ----------------------------------------------------------
if [[ -t 1 ]]; then
  BOLD=$'\033[1m'; RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RESET=$'\033[0m'
else
  BOLD=""; RED=""; GREEN=""; YELLOW=""; RESET=""
fi
info()  { printf '%s\n' "${BOLD}==>${RESET} $*"; }
ok()    { printf '%s\n' "  ${GREEN}✓${RESET} $*"; }
warn()  { printf '%s\n' "  ${YELLOW}!${RESET} $*"; }
die()   { printf '%s\n' "${RED}error:${RESET} $*" >&2; exit 1; }

# --- 1. platform -------------------------------------------------------------
info "Checking platform"
[[ "$(uname -s)" == "Darwin" ]] || die "this installer is for macOS only."
if [[ "$(uname -m)" != "arm64" ]]; then
  die "this is not an Apple-silicon Mac (found $(uname -m)). mlx requires M1 or newer."
fi
ok "macOS on Apple silicon"

# --- 2. python ---------------------------------------------------------------
# Pick a python3 that is >= 3.$MIN_PY_MINOR, installing one via Homebrew only if
# we have to and brew is available.
py_ok() {
  "$1" -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, $MIN_PY_MINOR) else 1)" 2>/dev/null
}

info "Locating Python 3.$MIN_PY_MINOR+"
PYTHON=""
for cand in python3 python3.13 python3.12 python3.11 python3.10 python3.9; do
  if command -v "$cand" >/dev/null 2>&1 && py_ok "$cand"; then
    PYTHON="$(command -v "$cand")"
    break
  fi
done

if [[ -z "$PYTHON" ]]; then
  if command -v brew >/dev/null 2>&1; then
    info "No suitable Python found; installing via Homebrew"
    brew install python
    command -v python3 >/dev/null 2>&1 && py_ok python3 && PYTHON="$(command -v python3)"
  fi
fi
[[ -n "$PYTHON" ]] || die "need Python 3.$MIN_PY_MINOR+. Install Homebrew (https://brew.sh) then re-run, or install Python yourself."
ok "using $("$PYTHON" --version) at $PYTHON"

# --- 3. install the package --------------------------------------------------
[[ -f "$SCRIPT_DIR/pyproject.toml" ]] || die "pyproject.toml not found next to install.sh ($SCRIPT_DIR)."

MLX_RUNNER_BIN=""
if command -v pipx >/dev/null 2>&1; then
  info "Installing mlx-runner with pipx (isolated)"
  pipx install --force "$SCRIPT_DIR"
  pipx ensurepath >/dev/null 2>&1 || true
  MLX_RUNNER_BIN="$HOME/.local/bin/mlx-runner"
else
  info "pipx not found; installing into the user site with pip"
  "$PYTHON" -m pip install --user --upgrade "$SCRIPT_DIR"
  USER_BIN="$("$PYTHON" -c 'import site,os;print(os.path.join(site.getuserbase(),"bin"))')"
  MLX_RUNNER_BIN="$USER_BIN/mlx-runner"
  case ":$PATH:" in
    *":$USER_BIN:"*) ;;
    *) warn "add $USER_BIN to your PATH to use 'mlx-runner' directly." ;;
  esac
fi

# Fall back to module invocation if the console script isn't on PATH yet.
if [[ -x "$MLX_RUNNER_BIN" ]]; then
  RUN=("$MLX_RUNNER_BIN")
elif command -v mlx-runner >/dev/null 2>&1; then
  RUN=(mlx-runner)
else
  RUN=("$PYTHON" -m mlx_runner)
fi
ok "installed (${RUN[*]})"

# --- 4. onboard --------------------------------------------------------------
if [[ "$SKIP_SETUP" -eq 1 ]]; then
  info "Skipping setup (--skip-setup). Run '${RUN[*]} setup' when ready."
  exit 0
fi

info "Running onboarding setup"
"${RUN[@]}" setup "${SETUP_ARGS[@]}"
