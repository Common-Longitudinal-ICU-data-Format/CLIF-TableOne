"""DQA validation result routes."""

import math

from fastapi import APIRouter, HTTPException
from server.services import cache_service
from modules.cli.pdf_generator import _collect_dqa_issues, DQA_CATEGORIES
from modules.utils.feedback import create_error_id


def _sanitize_floats(obj):
    """Replace NaN/Inf floats with None so JSONResponse doesn't choke."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_floats(v) for v in obj]
    return obj

router = APIRouter(prefix="/api", tags=["validation"])


@router.get("/validation/summary")
async def get_validation_summary():
    """Aggregate DQA scores across all tables.

    Absent / unvalidated tables contribute ONLY their schema-derived
    conformance atoms to the totals (via
    :func:`build_absent_table_dqa_result`); their completeness and
    plausibility counts are N/A since those checks need actual data.
    This keeps the summary denominators aligned with the combined PDF
    report's DQA Overview totals.
    """
    from server.routes.analysis_routes import ALL_TABLES
    from clifpy.utils.validator import build_absent_table_dqa_result

    agg_categories = {}  # cat -> [passed, total]
    total_errors = 0
    total_warnings = 0
    tables_analyzed = 0

    for name in ALL_TABLES:
        cached = cache_service.get(name)
        has_validation = bool(cached and cached.get("validation"))
        if has_validation:
            validation = cached["validation"]
            tables_analyzed += 1
        else:
            # Missing validation → synthesize an absent-table result so
            # the table's conformance atoms still contribute to totals.
            validation = build_absent_table_dqa_result(name)

        is_absent = bool(validation.get("absent"))
        category_scores, all_issues = _collect_dqa_issues(validation)

        # Adjust scores for rejected feedback (matches combined PDF logic).
        # Absent tables have no user feedback, so this block is a no-op for them.
        fb = cached.get("feedback") if has_validation else None
        rejected_ids: set = set()
        if fb and fb.get("user_decisions"):
            rejected_ids = {
                eid for eid, d in fb["user_decisions"].items()
                if d.get("decision") == "rejected"
            }
        if rejected_ids:
            for cat in list(category_scores.keys()):
                cat_rejected = sum(
                    i.get("atomic_count", 1) for i in all_issues
                    if i["category"] == cat and i["severity"] == "error"
                    and create_error_id(i) in rejected_ids
                )
                if cat_rejected:
                    p, t = category_scores[cat]
                    category_scores[cat] = (p + cat_rejected, t)

        rejected_error_count = sum(
            i.get("atomic_count", 1) for i in all_issues
            if i["severity"] == "error" and create_error_id(i) in rejected_ids
        )
        total_errors += sum(i.get("atomic_count", 1) for i in all_issues if i["severity"] == "error") - rejected_error_count
        total_warnings += sum(i.get("atomic_count", 1) for i in all_issues if i["severity"] == "warning")
        for cat, (p, t) in category_scores.items():
            # Absent tables only contribute conformance atoms; completeness
            # and plausibility need data to evaluate and are reported N/A.
            if is_absent and cat != "conformance":
                continue
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
    """Return DQA validation results with scores and issues.

    Absent tables (site didn't submit) get a synthesized response
    using clifpy's ``build_absent_table_dqa_result`` — Conformance
    scored as ``0/N`` against the schema's expected check count, and
    Completeness / Plausibility marked N/A (``None`` in JSON).
    """
    from server.routes.analysis_routes import ALL_TABLES

    cached = cache_service.get(name)
    if cached and cached.get("validation"):
        validation = cached["validation"]
        absent = bool(validation.get("absent"))
    else:
        # Try synthesizing an absent-table result if the table is a known beta.
        if name not in ALL_TABLES:
            raise HTTPException(404, f"Unknown table: {name}")
        from clifpy.utils.validator import build_absent_table_dqa_result
        validation = build_absent_table_dqa_result(name)
        absent = True

    category_scores, all_issues = _collect_dqa_issues(validation)

    # Add error_id to each issue so frontend can match with feedback
    for issue in all_issues:
        issue["error_id"] = create_error_id(issue)

    if absent:
        # build_absent_table_dqa_result already reports conformance as 0/N
        # (N = expected_check_counts.conformance). Completeness / Plausibility
        # are N/A since no data means those checks never get to run.
        conf = category_scores.get("conformance")
        scores_out = {
            "conformance": {"passed": conf[0], "total": conf[1]}
                            if conf and conf[1] > 0 else None,
            "completeness": None,
            "plausibility": None,
        }
        total_passed = conf[0] if conf else 0
        total_checks = conf[1] if conf else 0
    else:
        scores_out = {
            cat: {"passed": p, "total": t}
            for cat, (p, t) in category_scores.items()
        }
        total_passed = sum(p for p, _ in category_scores.values())
        total_checks = sum(t for _, t in category_scores.values())

    error_count = sum(i.get("atomic_count", 1) for i in all_issues if i["severity"] == "error")
    warning_count = sum(i.get("atomic_count", 1) for i in all_issues if i["severity"] == "warning")
    overall_pct = round(total_passed / total_checks * 100, 1) if total_checks else 0

    return _sanitize_floats({
        "table_name": name,
        "absent": absent,
        "overall_pct": overall_pct,
        "total_passed": total_passed,
        "total_checks": total_checks,
        "error_count": error_count,
        "warning_count": warning_count,
        "category_scores": scores_out,
        "issues": all_issues,
        "raw": validation,
    })
