"""Measure Chroma retrieval distances for tuning max_distance threshold.

CLI tool for analyzing vector similarity distances across a set of questions.
Used to determine appropriate max_distance settings for abstention.
"""
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from src.engine.rag import RAGEngine

from src.common.config_loader import load_settings


@dataclass(frozen=True)
class DistanceCase:
    id: str
    question: str


@dataclass(frozen=True)
class DistanceRow:
    id: str
    question: str
    best_distance: float | None
    distances: list[float]
    top_sources: list[str]


def _quantile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if q <= 0:
        return sorted_values[0]
    if q >= 1:
        return sorted_values[-1]
    idx = int(q * (len(sorted_values) - 1))
    return sorted_values[idx]


def summarize_best_distances(rows: Iterable[DistanceRow]) -> dict[str, Any]:
    bests = sorted([r.best_distance for r in rows if r.best_distance is not None])
    if not bests:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p80": None,
            "p90": None,
            "max": None,
        }

    return {
        "count": len(bests),
        "min": bests[0],
        "p50": _quantile(bests, 0.50),
        "p80": _quantile(bests, 0.80),
        "p90": _quantile(bests, 0.90),
        "max": bests[-1],
    }


def sweep_thresholds(
    rows: Iterable[DistanceRow],
    *,
    start: float,
    end: float,
    step: float,
) -> list[dict[str, Any]]:
    if step <= 0:
        raise ValueError("step must be > 0")

    rows_list = list(rows)
    bests = [r.best_distance for r in rows_list]
    total = len(bests)
    if total == 0:
        return []

    results: list[dict[str, Any]] = []
    threshold = start
    prev_answerable: set[str] = set()
    # Inclusive sweep, similar to numpy.arange but deterministic.
    while threshold <= end + 1e-12:
        abstain = 0
        answerable: set[str] = set()
        for b in bests:
            if b is None or b > threshold:
                abstain += 1

        for r in rows_list:
            b = r.best_distance
            if b is not None and b <= threshold:
                answerable.add(r.id)

        newly_answerable = sorted(answerable - prev_answerable)

        results.append(
            {
                "threshold": float(threshold),
                "total": total,
                "abstain": abstain,
                "answer": total - abstain,
                "abstain_rate": abstain / total,
                "newly_answerable_ids": newly_answerable,
            }
        )
        threshold = round(threshold + step, 10)
        prev_answerable = answerable

    return results


def _format_source(meta: dict[str, Any] | None) -> str:
    meta = meta or {}
    src = str(meta.get("source", "")).strip()
    if not src:
        src = "(ukendt)"
    article = meta.get("article")
    if article:
        return f"{src} (Artikel {article})"
    chapter = meta.get("chapter")
    if chapter:
        return f"{src} (Kapitel {chapter})"
    return src


def measure_questions(engine: "RAGEngine", cases: list[DistanceCase], k: int = 5) -> list[DistanceRow]:
    rows: list[DistanceRow] = []

    for case in cases:
        hits = engine.query(case.question, k=k)
        distances = list(getattr(engine, "_last_distances", []) or [])
        best = min(distances) if distances else None

        top_sources: list[str] = []
        for _, meta in hits[: min(3, len(hits))]:
            top_sources.append(_format_source(meta))

        rows.append(
            DistanceRow(
                id=case.id,
                question=case.question,
                best_distance=best,
                distances=distances,
                top_sources=top_sources,
            )
        )

    return rows


def _load_golden_cases(path: Path) -> list[DistanceCase]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list")

    cases: list[DistanceCase] = []
    for item in data:
        q = str(item.get("question", "")).strip()
        if not q:
            continue
        case_id = str(item.get("id", q)).strip() or q
        cases.append(DistanceCase(id=case_id, question=q))
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure Chroma retrieval distances for a set of questions, to help tune rag.max_distance."
    )
    parser.add_argument(
        "--law",
        default="",
        help="Which law/corpus to use (e.g. 'ai_act' or 'gdpr'). Defaults to rag.default_corpus.",
    )
    parser.add_argument(
        "--file",
        default="data/evals/golden_answers_ai_act.json",
        help="Path to JSON file with questions (expects objects with at least a 'question' field).",
    )
    parser.add_argument("--k", type=int, default=5, help="Number of results to retrieve per question")
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Extra question to include (repeatable).",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to write a JSON report (rows + summary).",
    )
    parser.add_argument(
        "--sweep",
        nargs=3,
        type=float,
        metavar=("START", "END", "STEP"),
        help="Optional: sweep thresholds and report abstain-rate for each (distance-only). Example: --sweep 0.60 0.90 0.01",
    )
    parser.add_argument(
        "--sweep-show-questions",
        action="store_true",
        help="When sweeping, also print which questions become answerable at each threshold step.",
    )
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    if args.k < 1:
        raise SystemExit("--k must be >= 1")

    settings = load_settings()

    corpora = settings.corpora or {}
    available_laws = sorted(corpora.keys())
    law_id = (args.law or "").strip() or settings.default_corpus
    corpus = corpora.get(law_id)
    if corpus is None:
        raise SystemExit(f"Unknown --law '{law_id}'. Available: {', '.join(available_laws)}")

    from src.engine.rag import RAGEngine
    engine = RAGEngine(
        docs_path=str(settings.docs_path),
        corpus_id=law_id,
        chunks_collection=corpus.chunks_collection,
        embedding_model=settings.embedding_model,
        chat_model=settings.chat_model,
        # top_k is now dynamic via config; use explicit k from args if needed
        top_k=args.k,
        vector_store_path=str(settings.vector_store_path),
        max_distance=None,  # do not abstain while measuring
    )

    cases_path = Path(args.file)
    cases: list[DistanceCase] = []
    if cases_path.exists():
        cases.extend(_load_golden_cases(cases_path))

    for q in args.extra:
        q = str(q).strip()
        if q:
            cases.append(DistanceCase(id=q, question=q))

    if not cases:
        raise SystemExit("No questions found. Provide --file with cases and/or one or more --extra questions.")

    rows = measure_questions(engine, cases=cases, k=args.k)

    print(f"Law/corpus: {law_id}")
    print(f"Embedding model: {settings.embedding_model}")
    print(f"Vector store: {settings.vector_store_path}")
    print(f"Questions: {len(rows)}")

    for row in rows:
        best = row.best_distance
        best_str = "(none)" if best is None else f"{best:.4f}"
        sources = "; ".join(row.top_sources) if row.top_sources else "(none)"
        print(f"\nQ: {row.question}")
        print(f"  best_distance: {best_str}")
        print(f"  top_sources: {sources}")

    summary = summarize_best_distances(rows)
    if summary["count"]:
        p80 = summary["p80"]
        p90 = summary["p90"]
        print("\nDistance distribution (best_distance over queries):")
        print(
            "  "
            f"count={summary['count']} "
            f"min={summary['min']:.4f} "
            f"p50={summary['p50']:.4f} "
            f"p80={p80:.4f} "
            f"p90={p90:.4f} "
            f"max={summary['max']:.4f}"
        )
        print("\nHeuristic starting point:")
        print(f"  Try rag.max_distance ~= {p90:.4f} (more permissive) or {p80:.4f} (stricter)")

    sweep_results: list[dict[str, Any]] = []
    if args.sweep:
        start, end, step = args.sweep
        if start > end:
            raise SystemExit("--sweep START must be <= END")

        sweep_results = sweep_thresholds(rows, start=start, end=end, step=step)
        if sweep_results:
            print("\nThreshold sweep (distance-only guardrail):")
            print("  threshold | abstain | answer | abstain_rate")
            for r in sweep_results:
                print(
                    f"  {r['threshold']:.4f} | {r['abstain']:>7} | {r['answer']:>6} | {r['abstain_rate']:.0%}"
                )

            if args.sweep_show_questions:
                id_to_row = {row.id: row for row in rows}
                print("\nQuestions that flip from abstain -> answer (by threshold):")
                printed_any = False
                for r in sweep_results:
                    ids = r.get("newly_answerable_ids") or []
                    if not ids:
                        continue
                    printed_any = True
                    threshold = float(r["threshold"])
                    print(f"\n  threshold={threshold:.4f} (newly answerable: {len(ids)})")
                    for case_id in ids:
                        row = id_to_row.get(case_id)
                        if row is None:
                            continue
                        best = row.best_distance
                        best_str = "(none)" if best is None else f"{best:.4f}"
                        print(f"    - {case_id}: best_distance={best_str} :: {row.question}")
                if not printed_any:
                    print("  (none)")

    if args.json_out:
        payload = {
            "law": law_id,
            "embedding_model": settings.embedding_model,
            "vector_store_path": str(settings.vector_store_path),
            "k": args.k,
            "rows": [
                {
                    "id": r.id,
                    "question": r.question,
                    "best_distance": r.best_distance,
                    "distances": r.distances,
                    "top_sources": r.top_sources,
                }
                for r in rows
            ],
            "summary": summary,
            "sweep": sweep_results,
        }
        Path(args.json_out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report to {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(run())
