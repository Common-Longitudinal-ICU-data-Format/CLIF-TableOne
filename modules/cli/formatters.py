"""Console output formatting utilities for CLI."""

from typing import Dict, Any


class ConsoleFormatter:
    """Formatter for console output with colors and symbols."""

    # Color codes
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    RESET = '\033[0m'

    # Symbols
    CHECK = '✅'
    CROSS = '❌'
    WARNING = '⚠️'
    INFO = 'ℹ️'
    SEARCH = '🔍'
    CHART = '📊'
    SAVE = '💾'
    SPARKLE = '✨'
    FOLDER = '📁'
    FILE = '📄'

    @staticmethod
    def success(message: str) -> str:
        """Format success message."""
        return f"{ConsoleFormatter.CHECK} {ConsoleFormatter.GREEN}{message}{ConsoleFormatter.RESET}"

    @staticmethod
    def error(message: str) -> str:
        """Format error message."""
        return f"{ConsoleFormatter.CROSS} {ConsoleFormatter.RED}{message}{ConsoleFormatter.RESET}"

    @staticmethod
    def warning(message: str) -> str:
        """Format warning message."""
        return f"{ConsoleFormatter.WARNING} {ConsoleFormatter.YELLOW}{message}{ConsoleFormatter.RESET}"

    @staticmethod
    def info(message: str) -> str:
        """Format info message."""
        return f"{ConsoleFormatter.INFO} {ConsoleFormatter.CYAN}{message}{ConsoleFormatter.RESET}"

    @staticmethod
    def header(message: str) -> str:
        """Format header message."""
        border = "=" * 60
        return f"\n{ConsoleFormatter.BOLD}{border}\n{message}\n{border}{ConsoleFormatter.RESET}\n"

    @staticmethod
    def section(message: str) -> str:
        """Format section message."""
        return f"\n{ConsoleFormatter.BOLD}{message}{ConsoleFormatter.RESET}"

    @staticmethod
    def progress(message: str) -> str:
        """Format progress message."""
        return f"{ConsoleFormatter.SEARCH} {message}..."

    @staticmethod
    def format_validation_summary(validation_results: Dict[str, Any], table_name: str) -> str:
        """Format validation results summary for console."""
        lines = []
        lines.append(ConsoleFormatter.section(f"Validation Results - {table_name}"))

        status = validation_results.get('status', 'unknown')
        if status == 'complete':
            lines.append(ConsoleFormatter.success(f"Status: {status.upper()}"))
        elif status == 'partial':
            lines.append(ConsoleFormatter.warning(f"Status: {status.upper()}"))
        else:
            lines.append(ConsoleFormatter.error(f"Status: {status.upper()}"))

        # Error counts
        errors = validation_results.get('errors', {})
        schema_errors = len(errors.get('schema_errors', []))
        quality_issues = len(errors.get('data_quality_issues', []))
        other_errors = len(errors.get('other_errors', []))
        total_errors = schema_errors + quality_issues + other_errors

        lines.append(f"  Total Issues: {total_errors}")
        if schema_errors > 0:
            lines.append(f"  - Schema Errors: {schema_errors}")
        if quality_issues > 0:
            lines.append(f"  - Data Quality Issues: {quality_issues}")
        if other_errors > 0:
            lines.append(f"  - Other Errors: {other_errors}")

        return "\n".join(lines)

    @staticmethod
    def format_summary_info(summary_stats: Dict[str, Any], table_name: str) -> str:
        """Format summary statistics for console."""
        lines = []
        lines.append(ConsoleFormatter.section(f"Summary Statistics - {table_name}"))

        data_info = summary_stats.get('data_info', {})
        if 'error' not in data_info:
            lines.append(f"  Total Rows: {data_info.get('row_count', 0):,}")
            lines.append(f"  Total Columns: {data_info.get('column_count', 0)}")

            if 'unique_patients' in data_info:
                lines.append(f"  Unique Patients: {data_info.get('unique_patients', 0):,}")

            if 'unique_hospitalizations' in data_info:
                lines.append(f"  Unique Hospitalizations: {data_info.get('unique_hospitalizations', 0):,}")

            # Dataset duration for hospitalization
            if 'first_admission_year' in data_info and data_info.get('first_admission_year'):
                first = data_info['first_admission_year']
                last = data_info['last_admission_year']
                years = last - first + 1
                lines.append(f"  Dataset Duration: {first} - {last} ({years} years)")

        # Missingness
        missingness = summary_stats.get('missingness', {})
        if 'error' not in missingness:
            lines.append(f"  Overall Missing: {missingness.get('overall_missing_percentage', 0):.2f}%")
            lines.append(f"  Complete Rows: {missingness.get('complete_rows_percentage', 0):.2f}%")

        return "\n".join(lines)
