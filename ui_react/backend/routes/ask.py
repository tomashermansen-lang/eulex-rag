"""API routes for asking questions.

Single Responsibility: Handle HTTP requests for Q&A functionality.
"""

from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty
from threading import Thread
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from schemas import AskRequest, AskResponse, Reference
import services

router = APIRouter(tags=["ask"])

# Thread pool for running sync generators
_executor = ThreadPoolExecutor(max_workers=4)


def _build_reference(ref: dict) -> Reference:
    """Convert raw reference dict to Reference schema."""
    return Reference(
        idx=ref.get("idx", 0),
        display=ref.get("display", "Unknown source"),
        chunk_text=ref.get("chunk_text", ""),
        corpus_id=ref.get("corpus_id"),
        article=ref.get("article"),
        recital=ref.get("recital"),
        annex=ref.get("annex"),
        paragraph=ref.get("paragraph"),
        litra=ref.get("litra"),
    )


@router.post("/ask", response_model=AskResponse)
async def ask_endpoint(request: AskRequest) -> AskResponse:
    """Get a complete answer to a question (non-streaming)."""
    start = time.time()

    try:
        # Convert Pydantic models to dicts for service layer
        history = [msg.model_dump() for msg in request.history] if request.history else []
        result = services.get_answer(
            question=request.question,
            law=request.law,
            user_profile=request.user_profile,
            history=history,
            corpus_scope=request.corpus_scope,
            target_corpora=request.target_corpora,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

    elapsed = time.time() - start

    references = [_build_reference(ref) for ref in (result.references_structured or [])]

    return AskResponse(
        answer=result.answer,
        references=references,
        retrieval_metrics=result.retrieval_metrics or {},
        response_time_seconds=elapsed,
    )


def _run_stream_in_thread(request: AskRequest, queue: Queue) -> None:
    """Run the synchronous stream_answer in a separate thread, putting results in a queue."""
    try:
        # Convert Pydantic models to dicts for service layer
        history = [msg.model_dump() for msg in request.history] if request.history else []
        for chunk in services.stream_answer(
            question=request.question,
            law=request.law,
            user_profile=request.user_profile,
            history=history,
            corpus_scope=request.corpus_scope,
            target_corpora=request.target_corpora,
        ):
            queue.put(("chunk", chunk))
        queue.put(("done", None))
    except Exception as e:
        queue.put(("error", e))


async def _generate_stream_events(request: AskRequest) -> AsyncGenerator[str, None]:
    """Generate SSE events for streaming answer."""
    start = time.time()

    # Use a queue to communicate between the sync generator thread and async generator
    queue: Queue = Queue()

    # Start the sync generator in a background thread
    thread = Thread(target=_run_stream_in_thread, args=(request, queue), daemon=True)
    thread.start()

    try:
        while True:
            # Poll the queue without blocking the event loop
            try:
                msg_type, data = queue.get_nowait()
            except Empty:
                # No data yet, yield control and try again
                await asyncio.sleep(0.01)
                continue

            if msg_type == "done":
                break
            elif msg_type == "error":
                error_data = json.dumps({"type": "error", "message": str(data)})
                yield f"data: {error_data}\n\n"
                break
            elif msg_type == "chunk":
                if isinstance(data, str):
                    event_data = json.dumps({"type": "chunk", "content": data})
                    yield f"data: {event_data}\n\n"
                else:
                    # Final AskResult
                    elapsed = time.time() - start
                    references = [_build_reference(ref) for ref in (data.references_structured or [])]

                    result_data = {
                        "type": "result",
                        "data": {
                            "answer": data.answer,
                            "references": [ref.model_dump() for ref in references],
                            "retrieval_metrics": data.retrieval_metrics or {},
                            "response_time_seconds": elapsed,
                        }
                    }
                    yield f"data: {json.dumps(result_data)}\n\n"

    except Exception as e:
        error_data = json.dumps({"type": "error", "message": f"Error: {str(e)}"})
        yield f"data: {error_data}\n\n"

    yield "data: [DONE]\n\n"


@router.post("/ask/stream")
async def ask_stream_endpoint(request: AskRequest) -> StreamingResponse:
    """Stream the answer as Server-Sent Events (SSE)."""
    return StreamingResponse(
        _generate_stream_events(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )
