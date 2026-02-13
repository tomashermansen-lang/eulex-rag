# Commands Reference

**Tip**: Alle kommandoer køres fra projektets rod (`./).

## Virtual Environment

```bash
source .venv/bin/activate
```

---

## UI Commands

```bash
# Start both backend and frontend (development)
./ui_react/start.sh

# Backend only
uvicorn ui_react.backend.main:app --reload --port 8000

# Frontend only
npm --prefix ui_react/frontend run dev
```

**Frontend Commands:**

```bash
# Install dependencies
npm --prefix ui_react/frontend install

# Run tests
npm --prefix ui_react/frontend test

# Type check
npm --prefix ui_react/frontend run typecheck

# Build
npm --prefix ui_react/frontend run build

# Backend tests
pytest ui_react/backend/tests/ -v
```

**URLs:**
- Frontend: http://localhost:5173
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/api/docs

---

## Admin UI (Add/Remove Legislation)

The Admin UI provides a browser-based way to manage EU legislation:

1. Start the UI: `./ui_react/start.sh`
2. Open hamburger menu (☰) → "Admin"
3. Browse available EUR-Lex legislation
4. Click **[+]** to add or **[🗑️]** to remove

**Features:**
- Real-time search filtering
- Progress tracking with SSE streaming
- Automatic example question generation
- Optional eval case generation with pipeline verification

**Admin API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/admin/legislation` | GET | List EUR-Lex legislation with local status |
| `/api/admin/add-law/stream` | POST | Ingest new law (SSE progress stream) |
| `/api/admin/corpus/{id}` | DELETE | Remove a corpus |
| `/api/admin/legislation/{celex}/check-update` | GET | Check if local version is outdated |

---

## Ingestion Pipeline (CLI)

### 1. Download HTML from EUR-Lex

Place HTML file in `data/raw/my-law.html` (filename without extension becomes corpus_id).

### 2. Chunk HTML → JSONL (with LLM enrichment)

```bash
# Single corpus
python -m src.ingestion.eurlex_engine --corpus my-law

# All HTML files
python -m src.ingestion.eurlex_engine
```

### 3. Index in Chroma

```bash
python -m src.main --ingest data/processed/my-law_chunks.jsonl
```

### 4. Build citation graph

```bash
python -c "from src.ingestion.citation_graph import CitationGraph; g = CitationGraph.from_corpus('my-law'); g.save()"
```

### Ingestion Eval

```bash
# Keywords method
python -m src.eval.ingestion_eval_runner --corpus ai-act --method keywords

# LLM method
python -m src.eval.ingestion_eval_runner --corpus ai-act --method llm

# Compare
python -m src.eval.ingestion_eval_runner --corpus ai-act --compare
```

---

## Eval Commands

### Full eval (default: LLM-judge enabled)

```bash
./scripts/run_eval.sh --law ai-act
./scripts/run_eval.sh --law gdpr
./scripts/run_eval.sh --law dora
./scripts/run_eval.sh --law nis2
./scripts/run_eval.sh --law all    # All corpora
```

### Common options

```bash
# Disable LLM-judge (faster)
./scripts/run_eval.sh --law ai-act --no-llm-judge

# Skip LLM entirely (retrieval only)
./scripts/run_eval.sh --law ai-act --skip-llm

# Pipeline analysis
./scripts/run_eval.sh --law ai-act --pipeline-analysis

# Specific profile
./scripts/run_eval.sh --law ai-act --profile LEGAL
./scripts/run_eval.sh --law ai-act --profile ENGINEERING

# Specific case
./scripts/run_eval.sh --law ai-act --case ai-act-13-recruitment-screening-high-risk

# Limit cases
./scripts/run_eval.sh --law ai-act --limit 10

# Verbose
./scripts/run_eval.sh --law ai-act -v

# View history
python -m src.eval.eval_runner --law ai-act --history
```

### Keep Mac awake during long eval

```bash
caffeinate -i ./scripts/run_eval.sh --law ai-act
```

### Flag Summary

| Flag | Description |
|------|-------------|
| `--law <name>` | Corpus (ai-act, gdpr, dora, nis2, all) |
| `--profile <name>` | LEGAL or ENGINEERING |
| `--skip-llm` | Retrieval only |
| `--no-llm-judge` | Disable faithfulness/relevancy scoring |
| `--pipeline-analysis` | Detailed breakdown |
| `--debug-faithfulness` | Per-claim debug |
| `--case <id>` | Single case |
| `--limit <n>` | Max cases |
| `--max-retries <n>` | Retry count (default: 3) |
| `--history` | View progression |
| `-v` | Verbose |

---

## Test Commands

```bash
# All tests (backend + frontend, updates README badges)
./scripts/run_tests.sh

# Backend only
./scripts/run_tests.sh --backend

# Frontend only
./scripts/run_tests.sh --frontend

# Quick backend tests
pytest -q

# With coverage
pytest --cov=src --cov-report=term-missing

# Specific file
pytest tests/test_rag_engine.py -v

# Pattern match
pytest -k "ranking" -v
```

---

## Utility Commands

```bash
# Clear pycache
./scripts/clean_pycache.sh

# Clear runs
./scripts/clean_runs.sh

# Verify references
python scripts/verify_references.py

# Analyze failures
python tools/analyze_failures.py

# Syntax check
python -m compileall src/engine
```

---

## Git Commands

```bash
# Conventional commits
git add -A && git commit -m "feat(scope): description"
git add -A && git commit -m "fix(scope): description"
git add -A && git commit -m "refactor(scope): description"

# Skip pre-commit hook
git commit --no-verify -m "message"
```

---

## Quick One-liners

```bash
# Quick retrieval eval (all corpora)
./scripts/run_eval.sh --law ai-act --skip-llm && ./scripts/run_eval.sh --law gdpr --skip-llm

# Fast eval without LLM-judge
./scripts/run_eval.sh --law ai-act --no-llm-judge

# Full validation before commit
python -m compileall src/engine && pytest -q && ./scripts/run_eval.sh --law ai-act --skip-llm
```
