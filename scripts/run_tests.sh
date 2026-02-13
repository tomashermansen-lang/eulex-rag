#!/bin/bash
# Run all tests (backend + frontend) with coverage and auto-update README badges
#
# Usage:
#   ./scripts/run_tests.sh           # Run all tests (excludes eval)
#   ./scripts/run_tests.sh --fast    # Fast tests only (excludes slow + eval)
#   ./scripts/run_tests.sh --eval    # All tests including eval
#   ./scripts/run_tests.sh --backend # Run only backend tests
#   ./scripts/run_tests.sh --frontend # Run only frontend tests
#   ./scripts/run_tests.sh -q        # Pass args to pytest

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Ensure node/npm are in PATH (Homebrew location)
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

# Parse arguments
RUN_BACKEND=true
RUN_FRONTEND=true
TEST_TIER="full"
PYTEST_ARGS=""

for arg in "$@"; do
    case $arg in
        --fast)
            TEST_TIER="fast"
            shift
            ;;
        --eval)
            TEST_TIER="eval"
            shift
            ;;
        --backend)
            RUN_FRONTEND=false
            shift
            ;;
        --frontend)
            RUN_BACKEND=false
            shift
            ;;
        *)
            PYTEST_ARGS="$PYTEST_ARGS $arg"
            ;;
    esac
done

# Apply tier-based marker filters
case $TEST_TIER in
    fast)
        PYTEST_ARGS="-m \"not slow and not eval\" $PYTEST_ARGS"
        ;;
    full)
        PYTEST_ARGS="-m \"not eval\" $PYTEST_ARGS"
        ;;
    eval)
        # Run everything including eval
        ;;
esac

BACKEND_EXIT=0
FRONTEND_EXIT=0

# Run backend tests
if [ "$RUN_BACKEND" = true ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🐍 Running backend tests (tier: $TEST_TIER)..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Use venv python directly (no need to source activate)
    if [ -f ".venv/bin/python" ]; then
        PYTHON=".venv/bin/python"
    else
        PYTHON="python"
    fi

    # Run pytest with coverage, capture output while still showing it
    # Include all tests (tests/ + ui_react/backend/tests/)
    eval "$PYTHON" -m pytest tests/ ui_react/backend/tests/ --cov=src --cov-report=term-missing $PYTEST_ARGS | tee /tmp/pytest_output.txt
    BACKEND_EXIT=${PIPESTATUS[0]}

    # Extract stats from captured output
    COVERAGE=$(grep -E "^TOTAL" /tmp/pytest_output.txt | awk '{print $NF}' | tr -d '%')
    TESTS_PASSED=$(grep -oE "[0-9]+ passed" /tmp/pytest_output.txt | head -1 | grep -oE "[0-9]+")

    # Update README if we got valid stats
    if [ -n "$COVERAGE" ] && [ -n "$TESTS_PASSED" ]; then
        sed -i '' "s/tests-[0-9]*%20passed/tests-${TESTS_PASSED}%20passed/g" README.md
        sed -i '' "s/coverage-[0-9]*%25/coverage-${COVERAGE}%25/g" README.md
        echo ""
        echo "📊 README.md updated: ${TESTS_PASSED} tests, ${COVERAGE}% coverage"
    fi

    rm -f /tmp/pytest_output.txt
fi

# Count golden eval cases (always, since it's independent of test runs)
GOLDEN_EVALS=$(grep -h "^- id:" "$PROJECT_ROOT"/data/evals/golden_cases_*.yaml 2>/dev/null | wc -l | tr -d ' ')
if [ -n "$GOLDEN_EVALS" ] && [ "$GOLDEN_EVALS" -gt 0 ]; then
    sed -i '' "s/golden%20evals-[0-9]*/golden%20evals-${GOLDEN_EVALS}/g" README.md
    echo "📊 Golden evals badge updated: ${GOLDEN_EVALS} cases"
fi

# Calculate global pass rate from latest eval run per law (if jq available)
# Only counts runs with matching golden case files and at least one pass (excludes incomplete evals)
if command -v jq &> /dev/null; then
    TOTAL_PASSED=0
    TOTAL_CASES=0
    for golden_file in "$PROJECT_ROOT"/data/evals/golden_cases_*.yaml; do
        # Extract law name from golden case file (golden_cases_ai-act.yaml -> ai-act)
        law=$(basename "$golden_file" .yaml | sed 's/golden_cases_//' | tr '_' '-')
        run_file="$PROJECT_ROOT/runs/eval_${law}.json"
        if [ -f "$run_file" ]; then
            PASSED=$(jq -r '.summary.passed // 0' "$run_file" 2>/dev/null)
            TOTAL=$(jq -r '.summary.total // 0' "$run_file" 2>/dev/null)
            # Skip incomplete evals (0 passed usually means eval failed to run properly)
            if [ "$PASSED" -gt 0 ]; then
                TOTAL_PASSED=$((TOTAL_PASSED + PASSED))
                TOTAL_CASES=$((TOTAL_CASES + TOTAL))
            fi
        fi
    done
    if [ "$TOTAL_CASES" -gt 0 ]; then
        # Round to nearest integer: (a * 100 + b/2) / b
        PASS_RATE=$(( (TOTAL_PASSED * 100 + TOTAL_CASES / 2) / TOTAL_CASES ))
        sed -i '' "s/pass%20rate-[0-9]*%25/pass%20rate-${PASS_RATE}%25/g" README.md
        echo "📊 Eval pass rate badge updated: ${PASS_RATE}% (${TOTAL_PASSED}/${TOTAL_CASES})"
    fi
fi

# Run frontend tests
if [ "$RUN_FRONTEND" = true ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "⚛️  Running frontend tests..."
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    cd "$PROJECT_ROOT/ui_react/frontend"

    # Check if node is available
    if ! command -v node &> /dev/null; then
        echo "❌ Node.js not found. Please install Node.js to run frontend tests."
        FRONTEND_EXIT=1
    else
        # Run vitest
        npx vitest --run
        FRONTEND_EXIT=$?
    fi

    cd "$PROJECT_ROOT"
fi

# Print summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 Test Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ "$RUN_BACKEND" = true ]; then
    if [ $BACKEND_EXIT -eq 0 ]; then
        echo "   ✅ Backend tests: PASSED"
    else
        echo "   ❌ Backend tests: FAILED"
    fi
fi
if [ "$RUN_FRONTEND" = true ]; then
    if [ $FRONTEND_EXIT -eq 0 ]; then
        echo "   ✅ Frontend tests: PASSED"
    else
        echo "   ❌ Frontend tests: FAILED"
    fi
fi
echo ""

# Exit with error if any tests failed
if [ $BACKEND_EXIT -ne 0 ] || [ $FRONTEND_EXIT -ne 0 ]; then
    exit 1
fi
exit 0
