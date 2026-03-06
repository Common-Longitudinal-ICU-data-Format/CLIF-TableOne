"""DQA validation result routes."""

from fastapi import APIRouter, HTTPException
from server.services import cache_service
from modules.cli.pdf_generator import _collect_dqa_issues, DQA_CATEGORIES
from modules.utils.feedback import create_error_id

router = APIRouter(prefix="/api", tags=["validation"])


@router.get("/validation/summary")
async def get_validation_summary():
    """Aggregate DQA scores across all analyzed tables."""
    from server.routes.analysis_routes import ALL_TABLES

    agg_categories = {}  # cat -> [passed, total]
    total_errors = 0
    total_warnings = 0
    tables_analyzed = 0

    for name in ALL_TABLES:
        cached = cache_service.get(name)
        if not cached or not cached.get("validation"):
            continue
        tables_analyzed += 1
        category_scores, all_issues = _collect_dqa_issues(cached["validation"])

        # Adjust scores for rejected feedback (matches combined PDF logic)
        fb = cached.get("feedback")
        rejected_ids: set = set()
        if fb and fb.get("user_decisions"):
            rejected_ids = {
                eid for eid, d in fb["user_decisions"].items()
                if d.get("decision") == "rejected"
            }
        if rejected_ids:
            for cat in list(category_scores.keys()):
                cat_rejected = sum(
                    1 for i in all_issues
                    if i["category"] == cat and i["severity"] == "error"
                    and create_error_id(i) in rejected_ids
                )
                if cat_rejected:
                    p, t = category_scores[cat]
                    category_scores[cat] = (p + cat_rejected, t)

        rejected_error_count = sum(
            1 for i in all_issues
            if i["severity"] == "error" and create_error_id(i) in rejected_ids
        )
        total_errors += sum(1 for i in all_issues if i["severity"] == "error") - rejected_error_count
        total_warnings += sum(1 for i in all_issues if i["severity"] == "warning")
        for cat, (p, t) in category_scores.items():
            if cat not in agg_categories:
                agg_categories[cat] = [0, 0]
            agg_categories[cat][0] += p
            agg_categories[cat][1] += t

    total_passed = sum(v[0] for v in agg_categories.values())
    total_checks = sum(v[1] for v in agg_categories.values())
    overall_pct = round(total_passed / total_checks * 100, 1) if total_checks else 100

    return {
        "tables_analyzed": tables_analyzed,
        "tables_total": len(ALL_TABLES),
        "overall_pct": overall_pct,
        "total_passed": total_passed,
        "total_checks": total_checks,
        "total_errors": total_errors,
        "total_warnings": total_warnings,
        "categories": {
            cat: {"passed": v[0], "total": v[1]}
            for cat, v in agg_categories.items()
        },
    }


@router.get("/validation/{name}")
async def get_validation(name: str):
    """Return DQA validation results with scores and issues."""
    cached = cache_service.get(name)
    if not cached or not cached.get("validation"):
        raise HTTPException(404, f"No validation results for {name}")

    validation = cached["validation"]
    category_scores, all_issues = _collect_dqa_issues(validation)

    # Add error_id to each issue so frontend can match with feedback
    for issue in all_issues:
        issue["error_id"] = create_error_id(issue)

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
