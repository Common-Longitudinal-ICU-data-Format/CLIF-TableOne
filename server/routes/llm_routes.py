"""LLM interpretation routes for DQA results."""

import asyncio
import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from server.services import cache_service
from server.services import llm_service
from server.routes.analysis_routes import ALL_TABLES
from modules.cli.pdf_generator import _collect_dqa_issues
from modules.utils.feedback import create_error_id

logger = logging.getLogger("clif.llm")

router = APIRouter(prefix="/api", tags=["llm"])


def _make_event_generator(stream_iter):
    """Create an async SSE event generator from a sync stream iterator."""
    async def event_generator():
        loop = asyncio.get_event_loop()
        try:
            queue = asyncio.Queue()

            def _run_stream():
                try:
                    for delta in stream_iter:
                        asyncio.run_coroutine_threadsafe(queue.put(("text", delta)), loop)
                    asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop)
                except Exception as e:
                    asyncio.run_coroutine_threadsafe(queue.put(("error", str(e))), loop)

            loop.run_in_executor(None, _run_stream)

            while True:
                kind, value = await queue.get()
                if kind == "text":
                    yield f"data: {json.dumps({'text': value, 'done': False})}\n\n"
                elif kind == "done":
                    yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"
                    break
                elif kind == "error":
                    yield f"data: {json.dumps({'error': value, 'done': True})}\n\n"
                    break
        except Exception as e:
            logger.error("LLM streaming error: %s", e)
            yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

    return event_generator()


def _build_validation_data(name: str, cached: dict) -> dict:
    """Build a validation data dict from cached analysis results."""
    validation = cached["validation"]
    category_scores, all_issues = _collect_dqa_issues(validation)
    for issue in all_issues:
        issue["error_id"] = create_error_id(issue)

    total_passed = sum(p for p, _ in category_scores.values())
    total_checks = sum(t for _, t in category_scores.values())
    return {
        "table_name": name,
        "overall_pct": round(total_passed / total_checks * 100, 1) if total_checks else 100,
        "total_passed": total_passed,
        "total_checks": total_checks,
        "error_count": sum(1 for i in all_issues if i["severity"] == "error"),
        "warning_count": sum(1 for i in all_issues if i["severity"] == "warning"),
        "category_scores": {
            cat: {"passed": p, "total": t}
            for cat, (p, t) in category_scores.items()
        },
        "issues": all_issues,
    }


@router.get("/llm/status")
async def llm_status():
    """Check if LLM interpretation is available."""
    return {
        "available": llm_service.is_available(),
        "model": llm_service.OLLAMA_MODEL,
    }


@router.post("/llm/interpret-all")
async def interpret_all():
    """Stream an LLM interpretation across all analyzed tables."""
    if not llm_service.is_available():
        raise HTTPException(503, "Ollama is not running. Start it with: ollama serve")

    all_data = []
    for name in ALL_TABLES:
        cached = cache_service.get(name)
        if not cached or not cached.get("validation"):
            continue
        all_data.append(_build_validation_data(name, cached))

    if not all_data:
        raise HTTPException(404, "No tables have been analyzed yet")

    context = llm_service.curate_all_tables_context(all_data)
    stream = llm_service.stream_interpretation(context, llm_service.SYSTEM_PROMPT_ALL)
    return StreamingResponse(_make_event_generator(stream), media_type="text/event-stream")


@router.post("/llm/interpret/{name}")
async def interpret_validation(name: str):
    """Stream an LLM interpretation of validation results for a table."""
    if not llm_service.is_available():
        raise HTTPException(503, "Ollama is not running. Start it with: ollama serve")

    cached = cache_service.get(name)
    if not cached or not cached.get("validation"):
        raise HTTPException(404, f"No validation results for {name}")

    validation_data = _build_validation_data(name, cached)
    context = llm_service.curate_table_context(validation_data)
    stream = llm_service.stream_interpretation(context)
    return StreamingResponse(_make_event_generator(stream), media_type="text/event-stream")
