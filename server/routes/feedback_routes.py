"""Feedback CRUD routes."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from server import session
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


@router.get("/feedback/{name}")
async def get_feedback(name: str):
    """Return feedback for a table (from disk or generated from validation)."""
    config = session.get("config") or {}
    output_dir = config.get("output_dir", "output")

    # Try loading from disk first
    feedback = load_feedback(output_dir, name)

    # If no disk feedback, try creating from validation results
    if feedback is None:
        cached = cache_service.get(name)
        if cached and cached.get("validation"):
            feedback = create_feedback_structure(cached["validation"], name)

    # If still no feedback, try creating from DQA JSON on disk
    if feedback is None:
        import os, json
        dqa_path = os.path.join(output_dir, 'final', 'clifpy', f'{name}_dqa.json')
        if os.path.exists(dqa_path):
            with open(dqa_path, 'r') as f:
                validation_data = json.load(f)
            feedback = create_feedback_structure(validation_data, name)

    if feedback is None:
        raise HTTPException(404, f"No feedback available for {name}")

    return feedback


@router.put("/feedback/{name}")
async def update_feedback(name: str, body: FeedbackUpdate):
    """Update a single feedback decision for a table."""
    config = session.get("config") or {}
    output_dir = config.get("output_dir", "output")

    feedback = load_feedback(output_dir, name)
    if feedback is None:
        cached = cache_service.get(name)
        if cached and cached.get("validation"):
            feedback = create_feedback_structure(cached["validation"], name)

    if feedback is None:
        raise HTTPException(404, f"No feedback available for {name}")

    feedback = update_user_decision(feedback, body.error_id, body.decision, body.reason)
    cache_service.update_feedback(name, feedback)

    return {"status": "updated", "summary": get_feedback_summary(feedback)}


@router.post("/feedback/{name}/save")
async def save_feedback_to_disk(name: str):
    """Persist current feedback to disk and regenerate the table's PDF report."""
    config = session.get("config") or {}
    output_dir = config.get("output_dir", "output")

    # Get current feedback from cache or disk
    feedback = load_feedback(output_dir, name)
    if feedback is None:
        raise HTTPException(404, f"No feedback to save for {name}")

    filepath = save_feedback(feedback, output_dir, name)
    cache_service.update_feedback(name, feedback)

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
    except Exception:
        pass  # Report regeneration failure shouldn't block feedback save

    return {
        "status": "saved",
        "filepath": filepath,
        "report_regenerated": report_regenerated,
    }
