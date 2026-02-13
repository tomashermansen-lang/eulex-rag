#!/bin/bash
# Stop hook: verify tests pass before Claude considers task complete
# Runs backend tests if Python files changed, frontend tests if TS/TSX/CSS files changed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

EXIT_CODE=0

# Check if any Python files in src/, tests/, or ui_react/backend/ were modified
MODIFIED_PY=$(git status --porcelain 2>/dev/null | grep -E '\.py$' | grep -E '(src/|tests/|ui_react/backend/)' || true)

if [ -n "$MODIFIED_PY" ]; then
    # Activate venv
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
    fi

    # Run fast backend tests quietly (skip slow + eval)
    python -m pytest tests/ ui_react/backend/tests/ -q -m "not slow and not eval" 2>&1
    EXIT_CODE=$?
fi

# Check if any frontend files were modified
MODIFIED_FE=$(git status --porcelain 2>/dev/null | grep -E '\.(tsx?|css|jsx?)$' | grep -E 'ui_react/frontend/' || true)

if [ -n "$MODIFIED_FE" ]; then
    # Run frontend tests
    if command -v /opt/homebrew/bin/npm &>/dev/null; then
        NPM=/opt/homebrew/bin/npm
    else
        NPM=npm
    fi
    $NPM --prefix ui_react/frontend test -- --run 2>&1
    FE_CODE=$?
    if [ $FE_CODE -ne 0 ]; then
        EXIT_CODE=$FE_CODE
    fi
fi

# No modified files of either type — skip
if [ -z "$MODIFIED_PY" ] && [ -z "$MODIFIED_FE" ]; then
    exit 0
fi

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "⚠️  Tests are failing. Fix before completing the task."
    exit 1
fi

exit 0
