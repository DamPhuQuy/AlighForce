#!/usr/bin/env bash
# ==============================================================================
# Turnkey Environment Bootstrap & Baseline Verifier
# Part of SACH-2026 Engineering Harness (Learn Harness Engineering Standard)
# ==============================================================================
# Usage:
#   ./init.sh                     # Install deps, verify baseline, display start command
#   RUN_START_COMMAND=1 ./init.sh # Install deps, verify, and automatically run start command
# ==============================================================================

set -euo pipefail

# Run from the scaffolded project root, even when invoked from another directory.
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Overrides are trusted project configuration and may contain shell syntax.
# Never fill them from issue text, web pages, or tool output.
INSTALL_CMD="${INSTALL_CMD:-}"
VERIFY_CMD="${VERIFY_CMD:-}"
START_CMD="${START_CMD:-}"

manager=''
if [[ -z "$INSTALL_CMD" || -z "$VERIFY_CMD" || -z "$START_CMD" ]]; then
  lockfiles=()
  for file in package-lock.json npm-shrinkwrap.json pnpm-lock.yaml yarn.lock bun.lock bun.lockb uv.lock go.mod; do
    [[ -f "$file" ]] && lockfiles+=("$file")
  done
  if (( ${#lockfiles[@]} != 1 )); then
    echo "Specify INSTALL_CMD, VERIFY_CMD and START_CMD when no single supported lockfile is present." >&2
    exit 2
  fi
  case "${lockfiles[0]}" in
    package-lock.json|npm-shrinkwrap.json) manager=npm; install_default=(npm ci); verify_default=(npm test); start_default=(npm run dev) ;;
    pnpm-lock.yaml) manager=pnpm; install_default=(pnpm install --frozen-lockfile); verify_default=(pnpm test); start_default=(pnpm run dev) ;;
    yarn.lock)
      manager=yarn
      if [[ -f package.json ]] && node -e 'const p=require("./package.json"); process.exit(/^yarn@[2-9]/.test(p.packageManager || "") ? 0 : 1)' 2>/dev/null; then
        install_default=(yarn install --immutable)
      else
        install_default=(yarn install --frozen-lockfile)
      fi
      verify_default=(yarn test); start_default=(yarn dev) ;;
    bun.lock|bun.lockb) manager=bun; install_default=(bun install --frozen-lockfile); verify_default=(bun test); start_default=(bun run dev) ;;
    uv.lock) manager=uv; install_default=(uv sync --locked); verify_default=(uv run pytest); start_default=(uv run python -m app) ;;
    go.mod) manager=go; install_default=(go mod download); verify_default=(go test ./...); start_default=(go run .) ;;
  esac
fi

run_command() {
  local override="$1"
  shift
  if [[ -n "$override" ]]; then
    bash -c "$override"
  else
    "$@"
  fi
}

show_command() {
  local override="$1"
  shift
  if [[ -n "$override" ]]; then printf '%s' "$override"; else printf '%s ' "$@"; fi
}

# --- Color formatting ---
BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RESET='\033[0m'

echo -e "${BOLD}${CYAN}=== [Harness Environment Bootstrap] ===${RESET}"
echo -e "${BOLD}Repository root:${RESET} $(pwd)"
echo -e "${BOLD}Timestamp:${RESET} $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
echo ""

# 1. Dependency installation
echo -e "${BOLD}${YELLOW}>>> Step 1: Installing dependencies...${RESET}"
echo -e "Command: ${CYAN}$(show_command "$INSTALL_CMD" "${install_default[@]:-}")${RESET}"
run_command "$INSTALL_CMD" "${install_default[@]:-}"
echo -e "${GREEN}✓ Dependencies installed successfully.${RESET}\n"

# 2. Baseline verification
echo -e "${BOLD}${YELLOW}>>> Step 2: Running baseline verification...${RESET}"
echo -e "Command: ${CYAN}$(show_command "$VERIFY_CMD" "${verify_default[@]:-}")${RESET}"
if run_command "$VERIFY_CMD" "${verify_default[@]:-}"; then
  echo -e "${GREEN}✓ Baseline verification PASSED. Environment is clean and ready.${RESET}\n"
else
  echo -e "${RED}✗ Baseline verification FAILED!${RESET}"
  echo -e "${RED}HALT: Agent must repair baseline failures before introducing any new changes.${RESET}"
  exit 1
fi

# 3. Start command handling
echo -e "${BOLD}${YELLOW}>>> Step 3: Application runtime command${RESET}"
echo -e "Start command: ${CYAN}$(show_command "$START_CMD" "${start_default[@]:-}")${RESET}"
if [ "${RUN_START_COMMAND:-0}" = "1" ]; then
  echo -e "${GREEN}Executing start command...${RESET}"
  if [[ -n "$START_CMD" ]]; then exec bash -c "$START_CMD"; else exec "${start_default[@]}"; fi
else
  echo -e "${BOLD}To start the service manually, execute:${RESET} ${CYAN}$(show_command "$START_CMD" "${start_default[@]:-}")${RESET}"
  echo -e "${GREEN}=== Bootstrap complete. Ready for task execution ===${RESET}"
fi
