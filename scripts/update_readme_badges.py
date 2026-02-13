#!/usr/bin/env python3
"""
Update README.md badges, tables, and inline numbers from codebase facts.

Scans pytest, vitest, eval YAML files, corpus registry, engine modules,
and coverage — then patches every place in README where those numbers appear.

Run manually or after eval:
    python scripts/update_readme_badges.py
    python scripts/update_readme_badges.py --with-coverage   # slower: runs pytest --cov
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Cost constants (per-case, derived from OpenAI Standard tier Feb 2025)
# ---------------------------------------------------------------------------
COST_MINI_PER_CASE = 1.30 / 226    # ~$0.00575 per case
COST_GPT5_PER_CASE = 5.30 / 226    # ~$0.02345 per case


# ── Collectors ─────────────────────────────────────────────────────────────

def get_backend_test_count(project_root: Path) -> int | None:
    """Fast: pytest --collect-only (~0.3 s)."""
    try:
        result = subprocess.run(
            [str(project_root / ".venv" / "bin" / "python"), "-m", "pytest", "--collect-only", "-q"],
            capture_output=True, text=True, cwd=project_root, timeout=30,
        )
        for line in result.stdout.strip().splitlines():
            m = re.match(r"(\d+)\s+tests? collected", line)
            if m:
                return int(m.group(1))
    except Exception:
        pass
    return None


def get_frontend_test_count(project_root: Path) -> int | None:
    """Run vitest --run and parse test count (~3 s)."""
    frontend = project_root / "ui_react" / "frontend"
    if not frontend.exists():
        return None
    try:
        result = subprocess.run(
            ["npx", "vitest", "--run"],
            capture_output=True, text=True, cwd=frontend, timeout=60,
            env={**__import__("os").environ, "PATH": f"/opt/homebrew/bin:{__import__('os').environ.get('PATH', '')}"},
        )
        for line in result.stdout.splitlines():
            m = re.search(r"Tests\s+(\d+)\s+passed", line)
            if m:
                return int(m.group(1))
    except Exception:
        pass
    return None


def get_coverage(project_root: Path, run_fresh: bool = False) -> float | None:
    """Read coverage from .coverage file, or run pytest --cov if requested."""
    python = str(project_root / ".venv" / "bin" / "python")

    if run_fresh:
        subprocess.run(
            [python, "-m", "pytest", "--cov=src", "--cov-report=term", "-q"],
            capture_output=True, text=True, cwd=project_root, timeout=120,
        )

    # Try reading existing .coverage via coverage report
    try:
        result = subprocess.run(
            [python, "-m", "coverage", "report", "--format=total"],
            capture_output=True, text=True, cwd=project_root, timeout=10,
        )
        val = result.stdout.strip()
        if val and val.replace(".", "").isdigit():
            return float(val)
    except Exception:
        pass

    # Fallback: parse "Total coverage: XX.XX%" from pytest output
    try:
        result = subprocess.run(
            [python, "-m", "coverage", "report"],
            capture_output=True, text=True, cwd=project_root, timeout=10,
        )
        for line in result.stdout.splitlines():
            m = re.search(r"TOTAL\s+\d+\s+\d+\s+\d+\s+\d+\s+(\d+)%", line)
            if m:
                return float(m.group(1))
    except Exception:
        pass

    return None


def count_all_eval_cases(project_root: Path) -> int:
    """Count ALL eval cases across all YAML files in data/evals/.

    Uses YAML parsing to avoid overcounting nested list items.
    """
    evals_dir = project_root / "data" / "evals"
    if not evals_dir.exists():
        return 0
    total = 0
    for f in evals_dir.glob("*.yaml"):
        try:
            with open(f, encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            if isinstance(data, list):
                total += len(data)
            elif isinstance(data, dict) and "cases" in data:
                total += len(data["cases"])
        except Exception:
            continue
    return total


def count_golden_cases_per_corpus(project_root: Path) -> dict[str, int]:
    """Count golden cases per corpus from golden_cases_*.yaml files."""
    evals_dir = project_root / "data" / "evals"
    counts: dict[str, int] = {}
    if not evals_dir.exists():
        return counts
    for f in evals_dir.glob("golden_cases_*.yaml"):
        corpus = f.stem.replace("golden_cases_", "")
        try:
            with open(f, encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            if isinstance(data, list):
                n = len(data)
            elif isinstance(data, dict) and "cases" in data:
                n = len(data["cases"])
            else:
                n = 0
            if n > 0:
                counts[corpus] = n
        except Exception:
            continue
    return counts


def count_corpora_from_registry(project_root: Path) -> int:
    """Count registered corpora from corpus_registry.json."""
    registry = project_root / "data" / "processed" / "corpus_registry.json"
    if not registry.exists():
        return 0
    try:
        with open(registry, encoding="utf-8") as f:
            data = json.load(f)
        return len(data)
    except Exception:
        return 0


def count_engine_modules(project_root: Path) -> int:
    """Count .py files in src/engine/ (excluding __init__.py)."""
    engine_dir = project_root / "src" / "engine"
    if not engine_dir.exists():
        return 0
    return sum(1 for p in engine_dir.glob("*.py") if p.name != "__init__.py")


def get_eval_thresholds(config_path: Path) -> tuple[float, float]:
    """Get faithfulness and relevancy thresholds from config."""
    if not config_path.exists():
        return 0.75, 0.75
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    llm_judge = config.get("eval", {}).get("llm_judge", {})
    return (
        float(llm_judge.get("faithfulness_threshold", 0.75)),
        float(llm_judge.get("relevancy_threshold", 0.75)),
    )


def get_latest_eval_results(
    runs_dir: Path, expected_corpora: set[str],
) -> tuple[dict[str, dict], str | None]:
    """Read latest eval results from runs/eval_{law}.json files."""
    if not runs_dir.exists():
        return {}, None
    latest: dict[str, dict] = {}
    latest_ts: str | None = None
    for f in runs_dir.glob("eval_*.json"):
        law = f.stem.replace("eval_", "").lower()
        if law not in expected_corpora:
            continue
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        results = data.get("results", [])
        if not results:
            continue
        total = len(results)
        passed = sum(1 for r in results if r.get("passed"))
        escalated = sum(1 for r in results if r.get("escalated"))
        esc_passed = sum(1 for r in results if r.get("escalated") and r.get("passed"))
        latest[law] = {
            "passed": passed, "total": total,
            "escalated": escalated, "esc_passed": esc_passed,
            "timestamp": data.get("meta", {}).get("timestamp"),
        }
        ts = data.get("meta", {}).get("timestamp")
        if ts and (not latest_ts or ts > latest_ts):
            latest_ts = ts
    return latest, latest_ts


# ── README patcher ─────────────────────────────────────────────────────────

def _re_badge(label: str) -> str:
    """Build regex to match a shields.io badge by alt text."""
    esc = re.escape(label)
    return rf'!\[{esc}\]\(https://img\.shields\.io/badge/[^)]+\)'


def update_readme(  # noqa: C901
    readme_path: Path,
    *,
    backend_tests: int | None,
    frontend_tests: int | None,
    coverage_pct: float | None,
    total_eval_cases: int,
    corpus_count: int,
    engine_modules: int,
    eval_results: dict[str, dict],
    latest_eval_ts: str | None,
    faith_threshold: float,
    rel_threshold: float,
) -> bool:
    """Patch every number occurrence in README. Returns True if changed."""
    if not readme_path.exists():
        print(f"README not found: {readme_path}")
        return False

    content = readme_path.read_text(encoding="utf-8")
    original = content

    total_tests = (backend_tests or 0) + (frontend_tests or 0)

    # ── Badges ─────────────────────────────────────────────────────────
    if total_tests:
        content = re.sub(
            _re_badge("Tests"),
            f"![Tests](https://img.shields.io/badge/tests-{total_tests}%20passed-green)",
            content,
        )

    if coverage_pct is not None:
        cov_int = round(coverage_pct)
        color = "green" if cov_int >= 80 else "yellow" if cov_int >= 55 else "red"
        content = re.sub(
            _re_badge("Coverage"),
            f"![Coverage](https://img.shields.io/badge/coverage-{cov_int}%25-{color})",
            content,
        )

    if total_eval_cases:
        content = re.sub(
            _re_badge("Evals"),
            f"![Evals](https://img.shields.io/badge/golden%20evals-{total_eval_cases}-green)",
            content,
        )

    if latest_eval_ts:
        date_esc = latest_eval_ts[:10].replace("-", "--")
        content = re.sub(
            _re_badge("Last Eval"),
            f"![Last Eval](https://img.shields.io/badge/last%20eval-{date_esc}-blue)",
            content,
        )

    faith_pct = int(faith_threshold * 100)
    rel_pct = int(rel_threshold * 100)
    content = re.sub(
        _re_badge("Faithfulness"),
        f"![Faithfulness](https://img.shields.io/badge/faithfulness-≥{faith_pct}%25-blue)",
        content,
    )
    content = re.sub(
        _re_badge("Relevancy"),
        f"![Relevancy](https://img.shields.io/badge/relevancy-≥{rel_pct}%25-blue)",
        content,
    )

    # ── Technical Highlights table ─────────────────────────────────────
    if total_tests and backend_tests and frontend_tests:
        content = re.sub(
            r'\| \*\*Automated tests\*\* \| .+? \| .+? \|',
            f'| **Automated tests** | {total_tests:,} passed '
            f'| {backend_tests:,} backend (pytest) + {frontend_tests:,} frontend (vitest) |',
            content,
        )

    if coverage_pct is not None:
        cov_int = round(coverage_pct)
        content = re.sub(
            r'\| \*\*Test coverage\*\* \| \d+% \|',
            f'| **Test coverage** | {cov_int}% |',
            content,
        )

    if total_eval_cases:
        content = re.sub(
            r'(\| \*\*Golden eval cases\*\* \| )\d+( \|)',
            rf'\g<1>{total_eval_cases}\2',
            content,
        )

    if corpus_count:
        content = re.sub(
            r'(Full pipeline tests across )\d+( EU regulations)',
            rf'\g<1>{corpus_count}\2',
            content,
        )

    if engine_modules:
        content = re.sub(
            r'(\| \*\*Engine modules\*\* \| )\d+( Python files)',
            rf'\g<1>{engine_modules}\2',
            content,
        )

    # ── Eval pass rate ─────────────────────────────────────────────────
    if eval_results:
        tp = sum(r["passed"] for r in eval_results.values())
        tt = sum(r["total"] for r in eval_results.values())
        if tt:
            rate = tp / tt * 100
            rate_str = "100%" if rate == 100 else f"{rate:.1f}%"
            content = re.sub(
                r'(\| \*\*Eval pass rate\*\* \| )[\d.]+%( \|)',
                rf'\g<1>{rate_str}\2',
                content,
            )

    # ── Supported Legislation heading ──────────────────────────────────
    if corpus_count:
        content = re.sub(
            r'\*\*\d+ EU laws currently indexed:\*\*',
            f'**{corpus_count} EU laws currently indexed:**',
            content,
        )

    # ── Narrative numbers ──────────────────────────────────────────────
    if total_eval_cases and corpus_count:
        # "XXX eval cases across XX EU regulations" (in "Why I Built This")
        content = re.sub(
            r'(\d+) eval cases across (\d+) EU regulations',
            f'{total_eval_cases} eval cases across {corpus_count} EU regulations',
            content,
        )
    if total_eval_cases:
        # "XXX golden eval cases" (in "What I Learned")
        content = re.sub(
            r'\d+ golden eval cases',
            f'{total_eval_cases} golden eval cases',
            content,
        )

    # ── Mermaid diagram ────────────────────────────────────────────────
    if total_eval_cases:
        content = re.sub(
            r'CASES\[\d+ cases\]',
            f'CASES[{total_eval_cases} cases]',
            content,
        )

    # ── Cost estimates ─────────────────────────────────────────────────
    if total_eval_cases:
        mini_cost = total_eval_cases * COST_MINI_PER_CASE
        gpt5_cost = total_eval_cases * COST_GPT5_PER_CASE
        # Round to nearest $0.10
        mini_str = f"~${math.ceil(mini_cost * 10) / 10:.2f}"
        gpt5_str = f"~${math.ceil(gpt5_cost * 10) / 10:.2f}"

        # Table row: | Full eval suite (XXX cases) | ~$X.XX | ~$X.XX |
        content = re.sub(
            r'\| Full eval suite \(\d+ cases\) \| ~\$[\d.]+ \| ~\$[\d.]+ \|',
            f'| Full eval suite ({total_eval_cases} cases) | {mini_str} | {gpt5_str} |',
            content,
        )
        # Inline: "~$X.XX per full eval run"
        content = re.sub(
            r'~\$[\d.]+ per full eval run',
            f'{mini_str} per full eval run',
            content,
        )

    if content != original:
        readme_path.write_text(content, encoding="utf-8")
        return True
    return False


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    project_root = Path(__file__).parent.parent
    readme_path = project_root / "README.md"
    config_path = project_root / "config" / "settings.yaml"
    runs_dir = project_root / "runs"

    with_coverage = "--with-coverage" in sys.argv

    # Collect stats
    print("Collecting stats...")
    backend = get_backend_test_count(project_root)
    frontend = get_frontend_test_count(project_root)
    coverage = get_coverage(project_root, run_fresh=with_coverage)
    total_evals = count_all_eval_cases(project_root)
    corpus_count = count_corpora_from_registry(project_root)
    modules = count_engine_modules(project_root)
    faith, rel = get_eval_thresholds(config_path)

    per_corpus = count_golden_cases_per_corpus(project_root)
    expected = set(per_corpus.keys())
    eval_results, latest_ts = get_latest_eval_results(runs_dir, expected)

    # Summary
    total_tests = (backend or 0) + (frontend or 0)
    print(f"  Backend tests:  {backend or 'N/A'}")
    print(f"  Frontend tests: {frontend or 'N/A'}")
    print(f"  Total tests:    {total_tests or 'N/A'}")
    print(f"  Coverage:       {f'{coverage:.0f}%' if coverage else 'N/A (use --with-coverage)'}")
    print(f"  Golden evals:   {total_evals}")
    print(f"  Corpora (laws): {corpus_count}")
    print(f"  Engine modules: {modules}")
    if latest_ts:
        print(f"  Last eval:      {latest_ts[:10]}")

    if not total_tests and not total_evals:
        print("\nNo stats collected — nothing to update.")
        return 1

    if update_readme(
        readme_path,
        backend_tests=backend,
        frontend_tests=frontend,
        coverage_pct=coverage,
        total_eval_cases=total_evals,
        corpus_count=corpus_count,
        engine_modules=modules,
        eval_results=eval_results,
        latest_eval_ts=latest_ts,
        faith_threshold=faith,
        rel_threshold=rel,
    ):
        print("\nREADME.md updated.")
    else:
        print("\nREADME.md already up to date.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
