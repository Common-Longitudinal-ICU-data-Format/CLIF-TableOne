"""Feedback CRUD routes."""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from server import session

logger = logging.getLogger("clif.feedback")
from server.services import cache_service
from modules.utils.feedback import (
    create_feedback_structure,
    update_user_decision,
    save_feedback,
    load_feedback,
    get_feedback_summary,
    create_error_id,
)

router = APIRouter(prefix="/api", tags=["feedback"])


class FeedbackUpdate(BaseModel):
    error_id: str
    decision: str  # 'accepted', 'rejected', 'pending'
    reason: str = ''


def _get_pending_feedback() -> dict:
    """Get the pending feedback dict from session, creating if needed."""
    pending = session.get("pending_feedback")
    if pending is None:
        pending = {}
        session.set("pending_feedback", pending)
    return pending


def _resolve_feedback(name: str, config: dict):
    """Load feedback from: pending session → disk → validation cache → DQA JSON."""
    output_dir = config.get("output_dir", "output")

    # 1. Check in-memory pending feedback (from PUT calls)
    pending = _get_pending_feedback()
    if name in pending:
        return pending[name]

    # 2. Try loading from disk
    feedback = load_feedback(output_dir, name)
    if feedback is not None:
        return feedback

    # 3. Try creating from validation cache
    cached = cache_service.get(name)
    if cached and cached.get("validation"):
        return create_feedback_structure(cached["validation"], name)

    # 4. Try creating from DQA JSON on disk
    import os, json
    dqa_path = os.path.join(output_dir, 'final', 'clifpy', f'{name}_dqa.json')
    if os.path.exists(dqa_path):
        with open(dqa_path, 'r') as f:
            validation_data = json.load(f)
        return create_feedback_structure(validation_data, name)

    return None


@router.get("/feedback/{name}")
async def get_feedback(name: str):
    """Return feedback for a table (from disk or generated from validation)."""
    config = session.get("config") or {}
    feedback = _resolve_feedback(name, config)
    if feedback is None:
        raise HTTPException(404, f"No feedback available for {name}")
    return feedback


@router.put("/feedback/{name}")
async def update_feedback(name: str, body: FeedbackUpdate):
    """Update a single feedback decision for a table."""
    config = session.get("config") or {}
    feedback = _resolve_feedback(name, config)
    if feedback is None:
        raise HTTPException(404, f"No feedback available for {name}")

    feedback = update_user_decision(feedback, body.error_id, body.decision, body.reason)

    # Store in session so save endpoint can find it
    _get_pending_feedback()[name] = feedback
    cache_service.update_feedback(name, feedback)
    logger.info("Updated feedback for %s: error_id=%s decision=%s", name, body.error_id, body.decision)

    return {"status": "updated", "summary": get_feedback_summary(feedback)}


@router.post("/feedback/{name}/save")
async def save_feedback_to_disk(name: str):
    """Persist current feedback to disk and regenerate the table's PDF report."""
    config = session.get("config") or {}
    output_dir = config.get("output_dir", "output")

    feedback = _resolve_feedback(name, config)
    if feedback is None:
        raise HTTPException(404, f"No feedback to save for {name}")

    filepath = save_feedback(feedback, output_dir, name)
    cache_service.update_feedback(name, feedback)
    # Clear from pending now that it's persisted
    _get_pending_feedback().pop(name, None)
    logger.info("Saved feedback to disk for %s: %s", name, filepath)

    # Auto-regenerate the PDF report with updated feedback
    report_regenerated = False
    try:
        from server.routes.reports_routes import _regenerate_table_pdf
        report_regenerated = _regenerate_table_pdf(name, config)

        # Also regenerate the combined report
        if report_regenerated:
            from server.routes.reports_routes import ALL_TABLES
            from modules.reports.combined_report_generator import generate_combined_report
            generate_combined_report(
                output_dir, ALL_TABLES,
                config.get('site_name'), config.get('timezone', 'UTC'),
            )
    except Exception as e:
        logger.warning("Report regeneration failed for %s: %s", name, e)

    return {
        "status": "saved",
        "filepath": filepath,
        "report_regenerated": report_regenerated,
    }
