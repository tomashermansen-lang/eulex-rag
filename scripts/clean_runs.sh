#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TARGET_RUNS_DIR="${ROOT_DIR}/runs"

if [[ "${TARGET_RUNS_DIR}" == "/" ]] || [[ -z "${TARGET_RUNS_DIR}" ]]; then
  echo "Refusing to delete: TARGET_RUNS_DIR is unsafe: '${TARGET_RUNS_DIR}'" >&2
  exit 2
fi

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

say "Cleaning run artifacts (local-only)."

if [[ -d "$TARGET_RUNS_DIR" ]]; then
  # Keep the tracked README.md (useful docs), delete everything else.
  run "find \"$TARGET_RUNS_DIR\" -mindepth 1 -maxdepth 1 ! -name 'README.md' -exec rm -rf {} +"
else
  say "No runs/ directory found; nothing to delete."
fi

say "Done."
