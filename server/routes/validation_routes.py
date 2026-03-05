"""DQA validation result routes."""

from fastapi import APIRouter, HTTPException
from server.services import cache_service
from modules.cli.pdf_generator import _collect_dqa_issues, DQA_CATEGORIES

router = APIRouter(prefix="/api", tags=["validation"])


@router.get("/validation/{name}")
async def get_validation(name: str):
    """Return DQA validation results with scores and issues."""
    cached = cache_service.get(name)
    if not cached or not cached.get("validation"):
        raise HTTPException(404, f"No validation results for {name}")

    validation = cached["validation"]
    category_scores, all_issues = _collect_dqa_issues(validation)

    total_passed = sum(p for p, _ in category_scores.values())
    total_checks = sum(t for _, t in category_scores.values())
    error_count = sum(1 for i in all_issues if i["severity"] == "error")
    warning_count = sum(1 for i in all_issues if i["severity"] == "warning")
    overall_pct = round(total_passed / total_checks * 100, 1) if total_checks else 100

    return {
        "table_name": name,
        "overall_pct": overall_pct,
        "total_passed": total_passed,
        "total_checks": total_checks,
        "error_count": error_count,
        "warning_count": warning_count,
        "category_scores": {
            cat: {"passed": p, "total": t}
            for cat, (p, t) in category_scores.items()
        },
        "issues": all_issues,
        "raw": validation,
    }
