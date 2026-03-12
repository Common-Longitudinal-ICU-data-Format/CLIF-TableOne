"""Report regeneration and download routes."""

import os
import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from server import session

logger = logging.getLogger("clif.reports")

router = APIRouter(prefix="/api", tags=["reports"])

ALL_TABLES = [
    'patient', 'hospitalization', 'adt', 'code_status', 'crrt_therapy', 'ecmo_mcs',
    'hospital_diagnosis', 'labs', 'medication_admin_continuous', 'medication_admin_intermittent',
    'microbiology_culture', 'microbiology_nonculture', 'microbiology_susceptibility',
    'patient_assessments', 'patient_procedures', 'position', 'respiratory_support', 'vitals',
]


def _regenerate_table_pdf(table_name: str, config: dict) -> bool:
    """Regenerate a single table's PDF report with current feedback.

    Returns True if the PDF was regenerated successfully.
    """
    from modules.cli import ValidationPDFGenerator

    output_dir = config.get("output_dir", "output")
    clifpy_dir = os.path.join(output_dir, 'final', 'clifpy')
    results_dir = os.path.join(output_dir, 'final', 'results')
    reports_dir = os.path.join(output_dir, 'final', 'reports')
    os.makedirs(reports_dir, exist_ok=True)

    dqa_json_path = os.path.join(clifpy_dir, f'{table_name}_dqa.json')
    if not os.path.exists(dqa_json_path):
        logger.warning("DQA JSON not found: %s", dqa_json_path)
        return False

    try:
        with open(dqa_json_path, 'r') as f:
            validation_results = json.load(f)

        feedback = None
        response_file = os.path.join(results_dir, f'{table_name}_validation_response.json')
        if os.path.exists(response_file):
            with open(response_file, 'r') as f:
                feedback = json.load(f)

        pdf_path = os.path.join(reports_dir, f"{table_name}_validation_report.pdf")
        pdf_generator = ValidationPDFGenerator()
        if pdf_generator.is_available():
            pdf_generator.generate_validation_pdf(
                validation_results, table_name, pdf_path,
                config.get('site_name'), config.get('timezone', 'UTC'), feedback,
            )
            logger.info("Regenerated PDF: %s", pdf_path)
        return True
    except Exception as e:
        logger.error("Failed to regenerate PDF for %s: %s", table_name, e, exc_info=True)
        return False


@router.post("/reports/regenerate/{name}")
async def regenerate_single_report(name: str):
    """Regenerate PDF for a single table (called after feedback save)."""
    config = session.get("config")
    if not config:
        raise HTTPException(400, "No config loaded")

    if not _regenerate_table_pdf(name, config):
        raise HTTPException(404, f"Could not regenerate report for {name}")

    # Also regenerate combined report so it reflects the updated feedback
    output_dir = config.get("output_dir", "output")
    from modules.reports.combined_report_generator import generate_combined_report
    try:
        generate_combined_report(
            output_dir, ALL_TABLES,
            config.get('site_name'), config.get('timezone', 'UTC'),
        )
    except Exception:
        pass  # Combined report failure shouldn't block single-table save

    return {"status": "ok", "table": name}


@router.get("/reports/table/{name}")
async def download_table_report(name: str):
    """Download an individual table's validation PDF report."""
    config = session.get("config")
    if not config:
        raise HTTPException(400, "No config loaded")

    output_dir = config.get("output_dir", "output")
    path = os.path.join(output_dir, 'final', 'reports', f'{name}_validation_report.pdf')
    if not os.path.exists(path):
        raise HTTPException(404, f"PDF report not found for {name}")
    with open(path, "rb") as f:
        content = f.read()
    return Response(content, media_type="application/pdf", headers={
        "Content-Disposition": "inline",
        "Cache-Control": "no-store",
    })


@router.get("/reports/download/{report_type}")
async def download_report(report_type: str):
    """Download a generated report file (pdf or csv)."""
    config = session.get("config")
    if not config:
        raise HTTPException(400, "No config loaded")

    output_dir = config.get("output_dir", "output")

    if report_type == "pdf":
        path = os.path.join(output_dir, 'final', 'reports', 'combined_validation_report.pdf')
        if not os.path.exists(path):
            raise HTTPException(404, "Combined PDF not found")
        return FileResponse(path, filename="combined_validation_report.pdf", media_type="application/pdf")

    if report_type == "csv":
        path = os.path.join(output_dir, 'final', 'results', 'consolidated_validation.csv')
        if not os.path.exists(path):
            raise HTTPException(404, "Consolidated CSV not found")
        return FileResponse(path, filename="consolidated_validation.csv", media_type="text/csv")

    raise HTTPException(404, f"Unknown report type: {report_type}")
