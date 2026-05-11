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
from modules.utils.feedback import flatten_mcide_for_report
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

        # Split multi-value MCIDE errors with partial per-value decisions so
        # the combined PDF/CSV accounts for partial rejections. No-op for
        # tables without MCIDE errors or without sub-decisions.
        for t in list(table_results.keys()):
            if table_results[t] is None:
                continue
            adj_val, adj_fb = flatten_mcide_for_report(
                table_results[t], feedback_map.get(t),
            )
            table_results[t] = adj_val
            if adj_fb is not None:
                feedback_map[t] = adj_fb

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
