  #!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Load .env if present
if [ -f "$ROOT_DIR/.env" ]; then
	set -a
	source "$ROOT_DIR/.env"
	set +a
fi

if [ -z "${PYTHONPATH:-}" ]; then
	PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR"
else
	PYTHONPATH="$ROOT_DIR/src:$ROOT_DIR:$PYTHONPATH"
fi
export PYTHONPATH

PY_BIN="${PY_BIN:-}"
if [ -z "$PY_BIN" ]; then
	if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
		PY_BIN="$ROOT_DIR/.venv/bin/python"
	elif command -v python3 >/dev/null 2>&1; then
		PY_BIN="python3"
	else
		PY_BIN="python"
	fi
fi

"$PY_BIN" -m src.eval.eval_runner "$@"
EVAL_EXIT_CODE=$?

# Update README badges after eval completes (only for full runs, not --history)
if [[ ! " $* " =~ " --history " ]] && [ $EVAL_EXIT_CODE -eq 0 ]; then
	"$PY_BIN" "$ROOT_DIR/scripts/update_readme_badges.py" 2>/dev/null || true
fi

exit $EVAL_EXIT_CODE
