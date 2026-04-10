"""
Combined report generator — thin wrapper around clifpy.

Delegates all PDF/CSV generation to clifpy.utils.report_generator while
maintaining the CLIF-TableOne output directory layout (pdf_reports/ and
consolidated/ subdirectories under validation/).
"""

from typing import List, Optional

from clifpy.utils.report_generator import (
    collect_table_results,
    generate_combined_validation_pdf,
    generate_consolidated_csv,
)
from modules.utils.output_paths import (
    PDF_REPORTS,
    validation_consolidated_dir,
    validation_feedback_dir,
    validation_json_reports_dir,
)


def generate_combined_report(
    output_dir: str,
    table_names: List[str],
    site_name: Optional[str] = None,
    timezone: Optional[str] = None,
) -> Optional[str]:
    """Generate a combined DQA report (PDF + CSV) for all tables.

    Parameters
    ----------
    output_dir : str
        Project output directory (unused — paths are derived from
        ``modules.utils.output_paths``).
    table_names : list[str]
        Ordered list of table names to include.
    site_name : str, optional
        Site/hospital label for the report title.
    timezone : str, optional
        Accepted for backwards compatibility; not used by clifpy.

    Returns
    -------
    str or None
        Path to the generated PDF, or *None* on failure.
    """
    try:
        json_dir = str(validation_json_reports_dir())
        feedback_dir = str(validation_feedback_dir())

        table_results, feedback_map = collect_table_results(
            json_dir, table_names, feedback_dir,
        )

        if sum(1 for r in table_results.values() if r is not None) == 0:
            return None

        # PDF → validation/pdf_reports/
        reports_dir = PDF_REPORTS
        reports_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = str(reports_dir / "combined_validation_report.pdf")
        generate_combined_validation_pdf(
            table_results, pdf_path, table_names, site_name,
            feedback_map=feedback_map,
        )

        # CSV → validation/consolidated/
        consolidated_dir = validation_consolidated_dir()
        consolidated_dir.mkdir(parents=True, exist_ok=True)
        csv_path = str(consolidated_dir / "consolidated_validation.csv")
        generate_consolidated_csv(
            table_results, csv_path, table_names,
            feedback_map=feedback_map,
        )

        return pdf_path

    except Exception:
        import traceback
        traceback.print_exc()
        return None
