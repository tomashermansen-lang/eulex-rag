import argparse
import sys
from pathlib import Path

from .engine.rag import RAGEngine, RAGEngineError
from .common.corpus_registry import default_registry_path, load_registry, normalize_corpus_id
from .common.config_loader import load_settings


def parse_args(*, default_docs_path: str, available_laws: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mini RAG CLI")
    parser.add_argument(
        "--law",
        default="",
        help=(
            "Which law/corpus to use (e.g. 'ai_act' or 'gdpr'). "
            f"Available: {', '.join(available_laws)}"
        ),
    )
    parser.add_argument(
        "--ingest",
        help="Path to a JSONL file (e.g., data/processed/ai-act_chunks.jsonl) to ingest into the vector store",
    )
    parser.add_argument(
        "--docs-path",
        default=default_docs_path,
        help="Directory with sample .txt docs (used when --load-sample-docs is set)",
    )
    parser.add_argument(
        "--load-sample-docs",
        action="store_true",
        help="Ingest .txt files from the docs path before starting Q&A",
    )
    parser.add_argument(
        "--corpus-registry",
        default="",
        help="Sti til corpus registry (default: data/processed/corpus_registry.json)",
    )
    return parser.parse_args()


def _choose_law_interactive(available: list[str], default_law: str) -> str:
    if not available:
        return default_law
    if len(available) == 1:
        return available[0]

    print("\nVælg lov/corpus:")
    for idx, law_id in enumerate(available, start=1):
        marker = " (default)" if law_id == default_law else ""
        print(f"  {idx}) {law_id}{marker}")

    choice = input(f"Valg [default={default_law}]: ").strip()
    if not choice:
        return default_law
    if choice.isdigit():
        pos = int(choice)
        if 1 <= pos <= len(available):
            return available[pos - 1]
    if choice in available:
        return choice
    print(f"Ugyldigt valg '{choice}'. Bruger default: {default_law}")
    return default_law


def run():
    settings = load_settings()
    corpora = settings.corpora or {}
    available_laws = sorted(corpora.keys())
    args = parse_args(default_docs_path=str(settings.docs_path), available_laws=available_laws)

    project_root = Path(__file__).resolve().parent.parent
    registry_path = (
        Path(args.corpus_registry)
        if (args.corpus_registry or "").strip()
        else default_registry_path(project_root)
    )

    default_law = settings.default_corpus
    law_id = (args.law or "").strip() or default_law

    # Only prompt interactively when running in a real terminal session.
    if not args.law and not args.ingest and sys.stdin.isatty():
        law_id = _choose_law_interactive(available_laws, default_law)

    corpus = corpora.get(law_id)
    if corpus is None:
        raise SystemExit(f"Ukendt --law '{law_id}'. Gyldige værdier: {', '.join(available_laws)}")

    reg = load_registry(registry_path)
    key = normalize_corpus_id(law_id)
    if key not in reg:
        raise SystemExit(
            f"Corpus '{law_id}' mangler i corpus registry. Tilføj entry i data/processed/corpus_registry.json eller kør ingestion med --display-name/--alias."
        )

    rag = RAGEngine(
        docs_path=args.docs_path,
        corpus_id=law_id,
        chunks_collection=corpus.chunks_collection,
        embedding_model=settings.embedding_model,
        chat_model=settings.chat_model,
        # Note: top_k is now dynamic via retrieval_pool_size and max_context_* from config
        vector_store_path=str(settings.vector_store_path),
        max_distance=(corpus.max_distance if corpus.max_distance is not None else settings.rag_max_distance),
        hybrid_vec_k=settings.hybrid_vec_k,
        ranking_weights=settings.ranking_weights,
    )

    if args.ingest:
        try:
            rag.ingest_jsonl(args.ingest)
            print(f"Ingested chunks from {args.ingest}.")
        except RAGEngineError as err:
            print(f"Failed to ingest chunks: {err}")
        return

    if args.load_sample_docs:
        try:
            rag.load_documents()
        except RAGEngineError as err:
            print(f"Failed to initialize RAG engine: {err}")
            return

    print("RAG system ready. Type 'quit' to exit.")

    while True:
        question = input("\nEnter a question: ")
        if question.strip().lower() in {"quit", "exit"}:
            print("Goodbye!")
            break

        try:
            answer = rag.answer(question)
        except RAGEngineError as err:
            print(f"Unable to answer the question: {err}")
            continue

        print("\nANSWER:")
        print(answer)


if __name__ == "__main__":
    run()
