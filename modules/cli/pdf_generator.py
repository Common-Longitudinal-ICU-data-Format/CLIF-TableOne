"""PDF report generator — delegates to clifpy.utils.report_generator."""

from clifpy.utils.report_generator import (
    DQA_CATEGORIES,
    collect_dqa_issues,
    compute_table_stats,
    generate_validation_pdf,
    generate_text_report,
)

# Backward-compat alias (used by combined_report_generator and app.py)
_collect_dqa_issues = collect_dqa_issues


class ValidationPDFGenerator:
    """Thin wrapper around clifpy report functions for class-based callers."""

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def generate_validation_pdf(validation_data, table_name, output_path,
                                site_name=None, timezone=None, feedback=None):
        return generate_validation_pdf(validation_data, table_name, output_path, site_name, feedback)

    @staticmethod
    def generate_text_report(validation_data, table_name, output_path,
                             site_name=None, timezone=None, feedback=None):
        return generate_text_report(validation_data, table_name, output_path, site_name)
