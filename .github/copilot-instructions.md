# Copilot Instructions for EuLex RAG Framework

## Project Overview
- **Purpose:** Deterministic legal-domain RAG (Retrieval-Augmented Generation) lab for building and QA'ing LLM context pipelines with hard, reproducible quality gates.
- **Core Principle:** Fail-closed citation validation—system refuses to answer if it cannot cite supported articles.
- **Major Components:**
  - `src/engine/`: Modular retrieval, ranking, and orchestration (see `rag.py` as orchestrator/facade; engine modules must NOT import `rag.py`).
  - `src/services/`, `src/tools/`: Supporting logic and utilities.
  - `data/evals/`: Golden test cases for regression and quality gates.
  - `ui_react/`: React-based Admin UI (frontend/backend split).
  - `config/settings.yaml`: Central configuration.

## Key Architectural Patterns
- **Unified pipeline:** Both streaming (UI) and batch (eval/CLI) use the same retrieval logic.
- **Citation contract:** All answers must extract and validate references; fail-closed on insufficient evidence.
- **Hybrid retrieval:** Combines vector search, BM25, and citation graph boosting.
- **Model escalation:** Failed evals are retried with a more capable model for edge cases.
- **Golden evals:** Regression tests across multiple EU laws; add new cases in `data/evals/` for new behaviors.
- **No hardcoding:** Avoid hardcoded values in code or prompts—prefer generic, configurable solutions.

## Developer Workflow
- **Testing:**
  - Run `pytest -q` (unit tests, always required before marking tasks complete).
  - Run `python -m compileall src/engine` (syntax check).
  - Run `./scripts/run_eval.sh --law ai-act` (full eval with LLM-judge).
- **UI:**
  - Start dev servers: `./ui_react/start.sh` (runs both backend and frontend).
  - Frontend: `npm --prefix ui_react/frontend install` (deps), `run dev` (start), `test` (run tests).
  - Backend: `uvicorn ui_react.backend.main:app --reload --port 8000`.
- **Pre-commit:** Syntax check + pytest run automatically.
- **Conventional commits:** `feat/fix/refactor/docs/test(scope): description`.
- **Update docs:** README.md for user-facing/architecture changes, COMMANDS.md for CLI/scripts.

## Project-Specific Conventions
- **Orchestrator isolation:** Engine modules must not import `rag.py` (enforced by code search).
- **Type hints:** Required on all function signatures; use dataclasses (prefer `frozen=True`).
- **No dead/commented-out code.**
- **Composition over inheritance.**
- **Add golden evals for new retrieval logic.**

## Integration & Data Flow
- **Ingestion:** Admin UI supports browser-based ingestion of any EUR-Lex law; triggers evals to verify pipeline.
- **Retrieval:** Modular, hybrid pipeline with citation validation and abstention handling.
- **Export:** Conversations can be exported (Markdown/PDF) for audit/compliance.

## Key Files & Directories
- `src/engine/rag.py`: Orchestrator/facade (do not import from engine modules)
- `src/engine/`: Retrieval, ranking, chunking, citation logic
- `data/evals/`: Golden test cases (add here for new behaviors)
- `ui_react/`: Admin UI (frontend/backend)
- `scripts/`: Automation for evals, cleaning, validation
- `config/settings.yaml`: Central config

## Example: Add a New Retrieval Feature
1. Add logic in a new or existing `src/engine/` module (do not touch `rag.py` unless orchestrator logic changes).
2. Add/modify golden evals in `data/evals/` to cover new behavior.
3. Run `pytest -q` and `./scripts/run_eval.sh --law ai-act` to verify.
4. Update docs if user-facing or architectural change.

---
For more, see `README.md`, `CLAUDE.md`, and `COMMANDS.md`.
