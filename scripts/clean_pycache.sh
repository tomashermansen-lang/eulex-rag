#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DRY_RUN="${DRY_RUN:-0}"

say() { printf "%s\n" "$*"; }
run() {
  if [[ "$DRY_RUN" == "1" ]]; then
    say "[DRY_RUN] $*"
  else
    say "+ $*"
    eval "$@"
  fi
}

say "Cleaning Python cache artifacts (local-only)."

run "find . \
  -path './.venv' -prune -o \
  -path './.git' -prune -o \
  -type d -name '__pycache__' -prune -exec rm -rf {} +"

run "find . \
  -path './.venv' -prune -o \
  -path './.git' -prune -o \
  -type f -name '*.pyc' -delete"
run "rm -rf .pytest_cache || true"

say "Done."
