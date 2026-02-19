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

    DQA_CATEGORIES = ('conformance', 'completeness', 'relational', 'plausibility')

    @staticmethod
    def format_condensed_validation(validation_results: Dict[str, Any], table_name: str) -> str:
        """Format a single-line condensed DQA validation summary."""
        total_checks = 0
        total_passed = 0
        total_errors = 0
        total_warnings = 0
        failing_categories = []

        for category in ConsoleFormatter.DQA_CATEGORIES:
            checks = validation_results.get(category, {})
            if not checks:
                continue
            cat_total = len(checks)
            cat_passed = sum(1 for r in checks.values() if r['passed'])
            cat_errors = sum(len(r['errors']) for r in checks.values())
            cat_warnings = sum(len(r['warnings']) for r in checks.values())

            total_checks += cat_total
            total_passed += cat_passed
            total_errors += cat_errors
            total_warnings += cat_warnings

            if cat_passed < cat_total:
                failing_categories.append(category)

        if total_passed == total_checks:
            return ConsoleFormatter.success(
                f"Validation: {total_passed}/{total_checks} passed — {table_name}"
            )

        parts = [f"{total_passed}/{total_checks} passed"]
        if total_errors:
            parts.append(f"{total_errors} error{'s' if total_errors != 1 else ''}")
        if total_warnings:
            parts.append(f"{total_warnings} warning{'s' if total_warnings != 1 else ''}")
        if failing_categories:
            parts.append(", ".join(c.title() for c in failing_categories))

        summary = " | ".join(parts)
        return ConsoleFormatter.warning(f"Validation: {summary} — {table_name}")

    @staticmethod
    def format_validation_summary(validation_results: Dict[str, Any], table_name: str) -> str:
        """Format DQA validation results summary for console."""
        lines = []
        lines.append(ConsoleFormatter.section(f"Validation Results - {table_name}"))

        total_checks = 0
        total_passed = 0
        total_errors = 0
        total_warnings = 0

        for category in ConsoleFormatter.DQA_CATEGORIES:
            checks = validation_results.get(category, {})
            if not checks:
                continue
            cat_total = len(checks)
            cat_passed = sum(1 for r in checks.values() if r['passed'])
            cat_errors = sum(len(r['errors']) for r in checks.values())
            cat_warnings = sum(len(r['warnings']) for r in checks.values())

            total_checks += cat_total
            total_passed += cat_passed
            total_errors += cat_errors
            total_warnings += cat_warnings

            if cat_passed == cat_total:
                lines.append(ConsoleFormatter.success(f"{category.title()}: {cat_passed}/{cat_total} passed"))
            else:
                lines.append(ConsoleFormatter.warning(f"{category.title()}: {cat_passed}/{cat_total} passed"))

        if total_checks == total_passed:
            lines.append(ConsoleFormatter.success(f"Overall: {total_passed}/{total_checks} checks passed"))
        else:
            lines.append(ConsoleFormatter.warning(
                f"Overall: {total_passed}/{total_checks} checks passed "
                f"({total_errors} errors, {total_warnings} warnings)"
            ))

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
