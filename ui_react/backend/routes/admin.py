"""API routes for admin functionality - legislation management.

Single Responsibility: Handle HTTP requests for admin operations.

Security:
- All endpoints are localhost-only (enforced at server level)
- URL validation via eurlex_listing module
- CELEX number validation
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from queue import Queue, Empty
from threading import Thread
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from schemas import (
    LegislationInfo,
    LegislationListResponse,
    UpdateStatus,
    AddLawRequest,
    RemoveCorpusResponse,
    SuggestNamesRequest,
    SuggestNamesResponse,
    IngestionQuality,
    AnchorListResponse,
)

from src.ingestion.eurlex_listing import (
    list_available_legislation,
    check_for_updates,
    get_legislation_by_celex,
    validate_celex,
    EurLexValidationError,
    EurLexSecurityError,
    LegislationInfo as CoreLegislationInfo,
    DateFilterType,
    DocumentType,
    MAX_YEAR_SPAN,
)
from src.common.corpora_inventory import (
    load_corpora_inventory,
    save_corpora_inventory,
    default_corpora_path,
)
from src.common.corpus_registry import (
    load_registry,
    save_registry,
    default_registry_path,
)
from src.common.config_loader import clear_config_cache

router = APIRouter(prefix="/admin", tags=["admin"])


def _core_to_schema(info: CoreLegislationInfo, quality_data: dict | None = None) -> LegislationInfo:
    """Convert core LegislationInfo to API schema.

    Args:
        info: Core legislation info from eurlex_listing
        quality_data: Optional quality metrics dict from corpora.json
    """
    quality = None
    if quality_data:
        quality = IngestionQuality(
            unhandled_patterns=quality_data.get("unhandled_patterns", {}),
            unhandled_count=quality_data.get("unhandled_count", 0),
            unhandled_pct=quality_data.get("unhandled_pct", 0.0),
            structure_coverage_pct=quality_data.get("structure_coverage_pct", 0.0),
            chunk_count=quality_data.get("chunk_count", 0),
        )

    return LegislationInfo(
        celex_number=info.celex_number,
        title_da=info.title_da,
        title_en=info.title_en,
        last_modified=info.last_modified.isoformat() if info.last_modified else None,
        entry_into_force=info.entry_into_force.isoformat() if info.entry_into_force else None,
        in_force=info.in_force,
        amended_by=info.amended_by,
        is_ingested=info.is_ingested,
        corpus_id=info.corpus_id,
        local_version_date=info.local_version_date.isoformat() if info.local_version_date else None,
        is_outdated=info.is_outdated,
        html_url=info.html_url,
        document_type=info.document_type,
        eurovoc_labels=info.eurovoc_labels,
        quality=quality,
    )


def _get_corpora_path() -> Path:
    """Get path to corpora.json."""
    return default_corpora_path(PROJECT_ROOT)


@router.get("/legislation", response_model=LegislationListResponse)
async def list_legislation_endpoint(
    search: str = Query(default="", description="Search term for filtering"),
    year_from: int | None = Query(default=None, description="Start year (default: current year - 4)"),
    year_to: int | None = Query(default=None, description="End year (default: current year)"),
    date_filter: str = Query(default="creation", description="Date filter: 'creation' or 'modification'"),
    doc_type: str = Query(default="regulation", description="Document type: 'all', 'regulation', or 'directive'"),
    in_force_only: bool = Query(default=True, description="Only show legislation in force"),
) -> LegislationListResponse:
    """List available EU legislation with local ingestion status.

    Filters by:
    - search: Text search in title and CELEX number
    - year_from/year_to: Year range (max 5 years)
    - date_filter: 'creation' (adoption date) or 'modification' (last update)
    - doc_type: 'regulation', 'directive', or 'all'
    - in_force_only: Only show active legislation (default: true)
    """
    from datetime import datetime

    try:
        # Set defaults for year range
        current_year = datetime.now().year
        if year_to is None:
            year_to = current_year
        if year_from is None:
            year_from = year_to - (MAX_YEAR_SPAN - 1)

        # Validate year range
        if year_to - year_from + 1 > MAX_YEAR_SPAN:
            raise HTTPException(
                status_code=400,
                detail=f"Year range cannot exceed {MAX_YEAR_SPAN} years"
            )

        # Parse filter enums
        try:
            date_filter_type = DateFilterType(date_filter)
        except ValueError:
            date_filter_type = DateFilterType.CREATION

        try:
            document_type = DocumentType(doc_type)
        except ValueError:
            document_type = DocumentType.REGULATION

        # Load local corpora for status enrichment
        corpora_path = _get_corpora_path()
        local_corpora = load_corpora_inventory(corpora_path)

        # Get available legislation with filters
        core_legislation = list_available_legislation(
            search_term=search,
            year_from=year_from,
            year_to=year_to,
            date_filter_type=date_filter_type,
            document_type=document_type,
            in_force_only=in_force_only,
        )

        # Enrich with local status
        from src.ingestion.eurlex_listing import enrich_corpora_with_status, LegislationInfo as CoreLegInfo, build_html_url, get_document_type
        enriched = enrich_corpora_with_status(local_corpora, core_legislation)

        # Always include ALL locally ingested corpora (even if not in current filter)
        # This ensures users always see what's installed
        enriched_celex = {leg.celex_number for leg in enriched}
        for corpus_id, data in local_corpora.get("corpora", {}).items():
            celex = data.get("celex_number", "")
            if celex and celex not in enriched_celex:
                # Create entry for locally ingested corpus not in SPARQL results
                ingested_at = data.get("ingested_at")
                local_date = None
                if ingested_at:
                    try:
                        local_date = datetime.fromisoformat(ingested_at.replace("Z", "+00:00"))
                    except ValueError:
                        pass

                # Parse entry_into_force from corpora.json if available
                entry_into_force_str = data.get("entry_into_force")
                entry_into_force = None
                if entry_into_force_str:
                    try:
                        entry_into_force = datetime.fromisoformat(entry_into_force_str)
                    except ValueError:
                        pass

                # Parse last_modified from corpora.json if available
                last_modified_str = data.get("last_modified")
                last_modified = None
                if last_modified_str:
                    try:
                        last_modified = datetime.fromisoformat(last_modified_str)
                    except ValueError:
                        pass

                local_entry = CoreLegInfo(
                    celex_number=celex,
                    title_da=data.get("display_name", corpus_id),
                    title_en="",
                    last_modified=last_modified,
                    entry_into_force=entry_into_force,
                    in_force=True,
                    is_ingested=True,
                    corpus_id=corpus_id,
                    local_version_date=local_date,
                    is_outdated=False,
                    html_url=build_html_url(celex) if celex else "",
                    document_type=get_document_type(celex) if celex else "",
                )
                enriched.append(local_entry)

        # Convert to API schema with quality data lookup
        def convert_with_quality(info: CoreLegInfo) -> LegislationInfo:
            quality_data = None
            if info.corpus_id:
                corpus_data = local_corpora.get("corpora", {}).get(info.corpus_id, {})
                quality_data = corpus_data.get("quality")
            return _core_to_schema(info, quality_data)

        legislation = [convert_with_quality(info) for info in enriched]

        return LegislationListResponse(
            legislation=legislation,
            total=len(legislation),
            year_from=year_from,
            year_to=year_to,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list legislation: {str(e)}")


@router.get("/legislation/{celex}/check-update", response_model=UpdateStatus)
async def check_update_endpoint(celex: str) -> UpdateStatus:
    """Check if a local corpus needs to be updated.

    Args:
        celex: CELEX number of the legislation to check
    """
    try:
        validate_celex(celex)
    except EurLexValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        corpora_path = _get_corpora_path()
        local_corpora = load_corpora_inventory(corpora_path)

        # Find corpus_id by CELEX number
        corpus_id = None
        from src.ingestion.eurlex_listing import extract_celex_from_url
        for cid, data in local_corpora.get("corpora", {}).items():
            source_url = data.get("source_url", "")
            if extract_celex_from_url(source_url) == celex.upper():
                corpus_id = cid
                break

        if not corpus_id:
            return UpdateStatus(
                corpus_id="",
                celex_number=celex.upper(),
                is_outdated=False,
                local_date=None,
                remote_date=None,
                reason="Legislation not ingested locally",
            )

        status = check_for_updates(corpus_id, local_corpora)

        return UpdateStatus(
            corpus_id=status.corpus_id,
            celex_number=status.celex_number,
            is_outdated=status.is_outdated,
            local_date=status.local_date.isoformat() if status.local_date else None,
            remote_date=status.remote_date.isoformat() if status.remote_date else None,
            reason=status.reason,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check update: {str(e)}")


def _run_ingestion_in_thread(request: AddLawRequest, queue: Queue) -> None:
    """Run ingestion pipeline in a separate thread, reporting progress via queue.

    This function orchestrates the full ingestion pipeline:
    1. Download HTML from EUR-Lex
    2. Chunk and enrich with LLM
    3. Index in ChromaDB
    4. Build citation graph
    5. (Optional) Generate eval cases
    6. Update corpora inventory
    """
    try:
        # Import services here to avoid circular imports
        from services.ingestion import run_full_ingestion

        for event in run_full_ingestion(
            celex_number=request.celex_number,
            corpus_id=request.corpus_id,
            display_name=request.display_name,
            fullname=request.fullname,
            eurovoc_labels=request.eurovoc_labels,
            generate_eval=request.generate_eval,
            entry_into_force=request.entry_into_force,
            last_modified=request.last_modified,
            eval_run_mode=request.eval_run_mode,
        ):
            queue.put(("event", event))

        queue.put(("done", None))

    except Exception as e:
        queue.put(("error", str(e)))


async def _generate_ingestion_events(request: AddLawRequest) -> AsyncGenerator[str, None]:
    """Generate SSE events for ingestion progress."""
    queue: Queue = Queue()

    # Start ingestion in background thread
    thread = Thread(target=_run_ingestion_in_thread, args=(request, queue), daemon=True)
    thread.start()

    try:
        while True:
            try:
                msg_type, data = queue.get_nowait()
            except Empty:
                await asyncio.sleep(0.05)
                continue

            if msg_type == "done":
                break
            elif msg_type == "error":
                error_event = {"type": "error", "error": str(data)}
                yield f"data: {json.dumps(error_event)}\n\n"
                break
            elif msg_type == "event":
                yield f"data: {json.dumps(data)}\n\n"

    except Exception as e:
        error_event = {"type": "error", "error": f"Unexpected error: {str(e)}"}
        yield f"data: {json.dumps(error_event)}\n\n"

    yield "data: [DONE]\n\n"


@router.post("/add-law/stream")
async def add_law_stream_endpoint(request: AddLawRequest) -> StreamingResponse:
    """Add a new law/corpus with streaming progress updates.

    Streams SSE events reporting progress through:
    - download: Downloading HTML from EUR-Lex
    - chunking: Chunking and LLM enrichment
    - indexing: Vector store indexing
    - citation_graph: Building citation graph
    - eval_generation: (Optional) Generating eval cases
    - config_update: Updating configuration
    """
    # Validate CELEX number
    try:
        validate_celex(request.celex_number)
    except EurLexValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Check if corpus ID already exists
    corpora_path = _get_corpora_path()
    local_corpora = load_corpora_inventory(corpora_path)
    if request.corpus_id in local_corpora.get("corpora", {}):
        raise HTTPException(
            status_code=409,
            detail=f"Corpus ID '{request.corpus_id}' already exists. Choose a different ID.",
        )

    return StreamingResponse(
        _generate_ingestion_events(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/corpus/{corpus_id}", response_model=RemoveCorpusResponse)
async def remove_corpus_endpoint(
    corpus_id: str,
    confirm: str = Query(default="", description="Must match corpus_id exactly to confirm deletion"),
) -> RemoveCorpusResponse:
    """Remove a corpus from the system.

    This will:
    - Delete the ChromaDB collection
    - Remove entry from corpora.json
    - Delete HTML, JSONL, and citation graph files
    - Clear entire enrichment cache (ensures fresh LLM enrichment on re-ingestion)

    GUARDRAIL: Requires confirm parameter matching corpus_id exactly.
    """
    logger.info("Delete request received for corpus: %s (confirm: %s)", corpus_id, confirm)

    # GUARDRAIL 1: Require explicit confirmation
    if confirm != corpus_id:
        logger.warning(
            "Delete BLOCKED: confirm parameter '%s' does not match corpus_id '%s'",
            confirm, corpus_id
        )
        raise HTTPException(
            status_code=400,
            detail=f"Deletion requires confirm parameter matching corpus_id. Got confirm='{confirm}', expected confirm='{corpus_id}'",
        )

    try:
        corpora_path = _get_corpora_path()
        local_corpora = load_corpora_inventory(corpora_path)

        if corpus_id not in local_corpora.get("corpora", {}):
            logger.warning("Delete failed: corpus '%s' not found in inventory", corpus_id)
            raise HTTPException(
                status_code=404,
                detail=f"Corpus '{corpus_id}' not found",
            )

        corpus_data = local_corpora["corpora"][corpus_id]
        deleted_files: list[str] = []
        logger.info("Starting deletion of corpus '%s' (CELEX: %s)", corpus_id, corpus_data.get("celex_number", "unknown"))

        # Delete ChromaDB collection
        collection_name = corpus_data.get("chunks_collection", f"{corpus_id}_documents")
        try:
            import chromadb
            from src.common.config_loader import load_settings

            settings = load_settings()
            chroma_path = PROJECT_ROOT / settings.chromadb.persist_directory
            client = chromadb.PersistentClient(path=str(chroma_path))

            # Try to delete collection (may not exist)
            try:
                client.delete_collection(collection_name)
                deleted_files.append(f"ChromaDB: {collection_name}")
            except ValueError:
                # Collection doesn't exist, that's fine
                pass

        except Exception:
            # Log but don't fail - we can still remove from inventory
            pass

        # Delete associated files
        # GUARDRAIL: All file deletions verified against corpus_id
        raw_dir = PROJECT_ROOT / "data" / "raw"
        processed_dir = PROJECT_ROOT / "data" / "processed"

        # HTML file - only delete if filename exactly matches corpus_id
        html_file = raw_dir / f"{corpus_id}.html"
        if html_file.exists():
            # GUARDRAIL: Verify corpus_id is in filename
            if not html_file.name.startswith(corpus_id):
                logger.error("GUARDRAIL: Refusing to delete %s - doesn't match corpus '%s'", html_file.name, corpus_id)
            else:
                html_file.unlink()
                deleted_files.append(f"HTML: {html_file.name}")

        # JSONL chunks file - only delete if filename starts with corpus_id
        jsonl_file = processed_dir / f"{corpus_id}_chunks.jsonl"
        if jsonl_file.exists():
            # GUARDRAIL: Verify corpus_id is in filename
            if not jsonl_file.name.startswith(corpus_id):
                logger.error("GUARDRAIL: Refusing to delete %s - doesn't match corpus '%s'", jsonl_file.name, corpus_id)
            else:
                jsonl_file.unlink()
                deleted_files.append(f"JSONL: {jsonl_file.name}")

        # Citation graph file (try both naming conventions)
        for citation_pattern in [f"citation_graph_{corpus_id}.json", f"{corpus_id}_citation_graph.json"]:
            citation_file = processed_dir / citation_pattern
            if citation_file.exists():
                citation_file.unlink()
                deleted_files.append(f"Citation graph: {citation_file.name}")
                break

        # Clear entire enrichment cache (ensures fresh LLM enrichment on re-ingestion)
        enrichment_cache_dir = processed_dir / "enrichment_cache"
        if enrichment_cache_dir.exists():
            import shutil
            cache_file_count = len(list(enrichment_cache_dir.glob("*.json")))
            shutil.rmtree(enrichment_cache_dir)
            deleted_files.append(f"Enrichment cache: {cache_file_count} filer")

        # Delete eval files (golden cases)
        # GUARDRAIL: Only delete eval file with EXACT corpus_id match
        # Handle both hyphen and underscore variants (e.g., ai-act vs ai_act)
        evals_dir = PROJECT_ROOT / "data" / "evals"
        corpus_id_variants = [corpus_id, corpus_id.replace("-", "_"), corpus_id.replace("_", "-")]
        for variant in corpus_id_variants:
            eval_file = evals_dir / f"golden_cases_{variant}.yaml"
            if eval_file.exists():
                # GUARDRAIL: Verify variant derives from corpus_id
                base_id = corpus_id.replace("-", "").replace("_", "")
                file_base = variant.replace("-", "").replace("_", "")
                if base_id != file_base:
                    logger.error("GUARDRAIL: Refusing to delete %s - doesn't match corpus '%s'", eval_file.name, corpus_id)
                else:
                    eval_file.unlink()
                    deleted_files.append(f"Eval: {eval_file.name}")
                    logger.info("Deleted golden cases file: %s", eval_file.name)
                break

        # Remove from corpora.json
        del local_corpora["corpora"][corpus_id]
        save_corpora_inventory(corpora_path, local_corpora)

        # Remove from corpus_registry.json
        registry_path = default_registry_path(PROJECT_ROOT)
        try:
            registry = load_registry(registry_path)
            if corpus_id in registry:
                del registry[corpus_id]
                save_registry(registry_path, registry)
                deleted_files.append("corpus_registry.json entry")
        except Exception:
            # Non-critical - log but continue
            pass

        # Remove example questions
        try:
            from src.ingestion.example_generator import remove_corpus_examples
            if remove_corpus_examples(corpus_id):
                deleted_files.append("example_questions.json entry")
        except Exception:
            # Non-critical
            pass

        # Clear settings cache so corpus is immediately removed from chat dropdown
        clear_config_cache()

        files_msg = ", ".join(deleted_files) if deleted_files else "no additional files"
        logger.info("Deleted corpus '%s': %s", corpus_id, files_msg)

        return RemoveCorpusResponse(
            success=True,
            corpus_id=corpus_id,
            message=f"Corpus '{corpus_id}' removed. Deleted: {files_msg}.",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error while deleting corpus '%s'", corpus_id)
        raise HTTPException(status_code=500, detail=f"Failed to remove corpus: {str(e)}")


@router.post("/suggest-names", response_model=SuggestNamesResponse)
async def suggest_names_endpoint(request: SuggestNamesRequest) -> SuggestNamesResponse:
    """Use AI to suggest corpus_id, display_name, and fullname based on legislation title.

    Returns:
        SuggestNamesResponse with:
        - corpus_id: Structured ID (e.g., 'nis2-dir-2022-2555')
        - display_name: Human-readable name (e.g., 'NIS2-direktivet om cybersikkerhed')
        - fullname: Full official legal title for citations
    """
    import re

    from src.common.corpus_naming import parse_celex, generate_corpus_id

    try:
        from src.engine.llm_client import call_llm

        # Parse CELEX to get type info
        try:
            celex_info = parse_celex(request.celex_number)
            type_code = celex_info["type_code"]
            year = celex_info["year"]
            number = celex_info["number"]

            # Map type code to Danish document type
            type_names = {"dir": "direktiv", "reg": "forordning", "dec": "beslutning"}
            doc_type = type_names.get(type_code, "forordning")
        except ValueError:
            doc_type = "forordning"
            type_code = "reg"
            year = 2024
            number = 0

        prompt = f"""Du er ekspert i EU-lovgivning og skal navngive denne lov til et dansk RAG-system.

Dokumenttype: {doc_type.upper()} (baseret på CELEX-nummer)
Titel: {request.title}
CELEX: {request.celex_number}

KRITISK VIGTIGT - Identificer den OFFENTLIGT KENDTE forkortelse:
- Mange EU-love har velkendte forkortelser (f.eks. AI-ACT, GDPR, NIS2, DORA, DATA-ACT)
- Brug din viden til at identificere den etablerede BASIS-forkortelse
- VIGTIGT: known_name skal ALTID være BASIS-forkortelsen UDEN type-suffiks
  - Eksempel: For NIS2 gennemførelsesforordning skal known_name være "NIS2" (IKKE "NIS2-CIR")
  - Type-suffix (CIR, DR) tilføjes automatisk baseret på is_implementing/is_delegated flags

Foreslå:
1. known_name: BASIS-forkortelsen i UPPERCASE (f.eks. "NIS2", "GDPR", "AI-ACT")
   - ALDRIG inkluder -CIR eller -DR her - det håndteres automatisk

2. display_name på dansk:
   - For basis-love: "{doc_type.capitalize()} om [kort emne] (FORKORTELSE)"
   - For gennemførelsesforordninger: "Forordning om [kort emne] (FORKORTELSE-CIR)"
   - For delegerede forordninger: "Forordning om [kort emne] (FORKORTELSE-DR)"

3. is_implementing: true/false - Er dette en gennemførelsesforordning (Kommissionens gennemførelsesforordning)?

4. is_delegated: true/false - Er dette en delegeret forordning (Kommissionens delegerede forordning)?

Svar KUN med JSON:
{{"known_name": "...", "display_name": "...", "is_implementing": false, "is_delegated": false}}"""

        # Use project's standard LLM client
        response_text = call_llm(prompt, temperature=0.3)

        # Extract JSON from response
        json_match = re.search(r'\{[^}]+\}', response_text)
        if not json_match:
            raise ValueError("No JSON found in response")

        result = json.loads(json_match.group())

        known_name = result.get("known_name", "").upper()
        display_name = result.get("display_name", request.title)
        is_implementing = result.get("is_implementing", False)
        is_delegated = result.get("is_delegated", False)

        # Defensive: strip -CIR/-DR suffix if LLM included it (it shouldn't)
        # This prevents duplicate type codes like "nis2-cir-cir-..."
        if known_name.endswith("-CIR"):
            known_name = known_name[:-4]
            is_implementing = True  # Ensure flag is set
        elif known_name.endswith("-DR"):
            known_name = known_name[:-3]
            is_delegated = True  # Ensure flag is set

        if not known_name:
            # Fallback: extract from CELEX
            known_name = f"EU{year}"

        # Generate structured corpus_id using the naming module
        try:
            corpus_id = generate_corpus_id(
                known_name=known_name,
                celex_number=request.celex_number,
                is_implementing=is_implementing,
                is_delegated=is_delegated,
            )
        except ValueError:
            # Fallback if CELEX parsing fails
            corpus_id = known_name.lower().replace(" ", "-")

        # Ensure display_name shortname in parenthesis matches corpus_id format
        # Replace e.g. "(EU-DR)" with "(EU-DR-2024-1366)" to include year/number
        paren_match = re.search(r'\(([^)]+)\)\s*$', display_name)
        if paren_match:
            display_name = re.sub(
                r'\([^)]+\)\s*$',
                f"({corpus_id.upper()})",
                display_name,
            )

        # Use the original title as fullname (it's typically the official legal title)
        fullname = request.title

        return SuggestNamesResponse(
            corpus_id=corpus_id,
            display_name=display_name,
            fullname=fullname,
        )

    except Exception as e:
        logger.warning("AI name suggestion failed: %s", e)
        # Fallback to simple extraction
        try:
            celex_info = parse_celex(request.celex_number)
            corpus_id = f"eu{celex_info['year']}-{celex_info['type_code']}-{celex_info['number']}"
        except ValueError:
            corpus_id = request.celex_number[-8:].lower().replace("/", "").replace(" ", "")

        return SuggestNamesResponse(
            corpus_id=corpus_id,
            display_name=request.title,
            fullname=request.title,
        )


def _get_citation_graph_path(law: str) -> Path:
    """Get path to citation graph file for a law.

    Handles both underscore and hyphen naming conventions.
    """
    processed_dir = PROJECT_ROOT / "data" / "processed"

    # Try hyphen version first (ai-act), then underscore (ai_act)
    candidates = [
        processed_dir / f"citation_graph_{law}.json",
        processed_dir / f"citation_graph_{law.replace('-', '_')}.json",
    ]

    for path in candidates:
        if path.exists():
            return path

    return candidates[0]


@router.get("/corpus/{law}/anchors", response_model=AnchorListResponse)
async def list_anchors_endpoint(
    law: str,
    q: str = Query(default="", description="Search query to filter anchors"),
    limit: int = Query(default=50, ge=1, le=200, description="Maximum number of anchors to return"),
) -> AnchorListResponse:
    """List available anchors for a corpus from citation graph.

    Used for autocomplete in the eval case editor.
    """
    graph_path = _get_citation_graph_path(law)

    if not graph_path.exists():
        return AnchorListResponse(anchors=[], total=0)

    try:
        with open(graph_path, "r", encoding="utf-8") as f:
            graph = json.load(f)

        # Build anchor list from nodes (format: type:id, e.g., article:16)
        nodes = graph.get("nodes", {})
        all_anchors = []
        for node_id, node_data in nodes.items():
            node_type = node_data.get("type", "article")
            anchor = f"{node_type}:{node_id}"
            all_anchors.append(anchor)

        all_anchors = sorted(all_anchors)

        # Filter by query if provided
        if q:
            q_lower = q.lower()
            all_anchors = [a for a in all_anchors if q_lower in a.lower()]

        total = len(all_anchors)

        return AnchorListResponse(
            anchors=all_anchors[:limit],
            total=total,
        )

    except Exception as e:
        logger.warning("Failed to load citation graph for %s: %s", law, e)
        return AnchorListResponse(anchors=[], total=0)
