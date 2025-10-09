"""
Validation Utilities for CLIF Analysis

This module provides functions to format and categorize validation errors
from clifpy, similar to the clif_report_card.py implementation.
"""

from typing import Dict, Any, List


def format_clifpy_error(error: Dict[str, Any], row_count: int = 0) -> Dict[str, Any]:
    """
    Format a clifpy error object for display (from clif_report_card.py logic).

    Parameters:
    -----------
    error : dict
        Error dictionary from clifpy validation
    row_count : int
        Total number of rows in the table (for percentage calculations)

    Returns:
    --------
    dict
        Formatted error with type, description, and category
    """
    error_type = error.get('type', 'unknown')
    message = error.get('message', '')

    # Determine category based on error type
    category = 'other'
    display_type = error_type.replace('_', ' ').title()

    # Schema-related errors
    if error_type == 'missing_columns':
        category = 'schema'
        display_type = 'Missing Required Columns'
        if not message:
            columns = error.get('columns', [])
            message = f"Required columns not found: {', '.join(columns)}"

    elif error_type in ['datatype_mismatch', 'datatype_castable']:
        category = 'schema'
        display_type = 'Datatype Issue'
        if not message:
            column = error.get('column', 'unknown')
            expected = error.get('expected', 'unknown')
            actual = error.get('actual', 'unknown')
            castable = 'can be cast to' if error_type == 'datatype_castable' else 'does not match'
            message = f"Column '{column}' has type {actual} but {castable} {expected}"

    # Data quality errors
    elif error_type == 'null_values':
        category = 'data_quality'
        display_type = 'Missing Values'
        if not message:
            column = error.get('column', 'unknown')
            count = error.get('count', 0)
            percentage = (count / row_count * 100) if row_count > 0 else 0
            message = f"Column '{column}' has {count:,} missing values ({percentage:.1f}%)"

    elif error_type in ['invalid_category', 'invalid_categorical_values', 'missing_categorical_values']:
        category = 'data_quality'
        display_type = 'Invalid Categories'
        if not message:
            column = error.get('column', 'unknown')
            invalid_values = error.get('invalid_values', error.get('values', []))

            if isinstance(invalid_values, list) and invalid_values:
                truncated = invalid_values[:3]
                message = f"Column '{column}' contains invalid values: {', '.join(map(str, truncated))}"
                if len(invalid_values) > 3:
                    message += f" (and {len(invalid_values) - 3} more)"
            else:
                message = f"Column '{column}' contains invalid categorical values"

    elif error_type == 'duplicate_rows':
        category = 'data_quality'
        display_type = 'Duplicate Rows'
        if not message:
            count = error.get('count', 0)
            percentage = (count / row_count * 100) if row_count > 0 else 0
            message = f"Found {count:,} duplicate rows ({percentage:.1f}%)"

    elif error_type == 'outliers':
        category = 'data_quality'
        display_type = 'Outliers Detected'
        if not message:
            column = error.get('column', 'unknown')
            count = error.get('count', 0)
            message = f"Column '{column}' has {count} outlier values"

    elif error_type == 'file_not_found':
        category = 'schema'
        display_type = 'File Not Found'
        if not message:
            filename = error.get('filename', 'unknown')
            message = f"Could not find file: {filename}"

    elif error_type == 'empty_table':
        category = 'schema'
        display_type = 'Empty Table'
        if not message:
            message = "Table contains no data"

    # Date/time related errors
    elif error_type in ['invalid_datetime', 'datetime_format_error']:
        category = 'data_quality'
        display_type = 'DateTime Format Error'
        if not message:
            column = error.get('column', 'unknown')
            message = f"Column '{column}' contains invalid datetime values"

    # Reference errors
    elif error_type in ['invalid_reference', 'missing_reference']:
        category = 'data_quality'
        display_type = 'Reference Error'
        if not message:
            column = error.get('column', 'unknown')
            ref_table = error.get('reference_table', 'unknown')
            message = f"Column '{column}' contains invalid references to {ref_table}"

    return {
        'type': display_type,
        'description': message,
        'category': category,
        'raw_type': error_type
    }


def determine_validation_status(errors: List[Dict[str, Any]],
                               required_columns: List[str] = None) -> str:
    """
    Determine overall validation status based on errors.

    Parameters:
    -----------
    errors : list
        List of formatted error dictionaries
    required_columns : list, optional
        List of required column names

    Returns:
    --------
    str
        Status: 'complete', 'partial', or 'incomplete'
    """
    if not errors:
        return 'complete'

    # Check for critical errors that make status 'incomplete'
    has_critical_errors = False
    has_warnings = False

    for error in errors:
        raw_type = error.get('raw_type', '')
        description = error.get('description', '')

        # Critical errors (incomplete status)
        if raw_type in ['missing_columns', 'file_not_found', 'empty_table']:
            has_critical_errors = True

        # Check for non-castable datatype errors
        elif raw_type == 'datatype_mismatch' and 'can be cast to' not in description:
            has_critical_errors = True

        # Check for 100% missing data
        elif raw_type == 'null_values' and '100.0%' in description:
            has_critical_errors = True

        # Warnings (partial status)
        elif raw_type in ['invalid_category', 'invalid_categorical_values',
                         'missing_categorical_values', 'duplicate_rows',
                         'outliers', 'datatype_castable']:
            has_warnings = True

        # Check for high percentage of missing values (>50%)
        elif raw_type == 'null_values':
            import re
            match = re.search(r'(\d+\.?\d*)%', description)
            if match:
                percentage = float(match.group(1))
                if percentage > 50:
                    has_warnings = True

    # Determine final status
    if has_critical_errors:
        return 'incomplete'
    elif has_warnings:
        return 'partial'
    else:
        return 'complete'


def categorize_errors(errors: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Categorize errors into schema, data quality, and other categories.

    Parameters:
    -----------
    errors : list
        List of error dictionaries from clifpy

    Returns:
    --------
    dict
        Errors grouped by category
    """
    schema_errors = []
    data_quality_issues = []
    other_errors = []

    for error in errors:
        category = error.get('category', 'other')

        if category == 'schema':
            schema_errors.append(error)
        elif category == 'data_quality':
            data_quality_issues.append(error)
        else:
            other_errors.append(error)

    return {
        'schema_errors': schema_errors,
        'data_quality_issues': data_quality_issues,
        'other_errors': other_errors
    }


def get_validation_summary(validation_results: Dict[str, Any]) -> str:
    """
    Create a text summary of validation results.

    Parameters:
    -----------
    validation_results : dict
        Results from table validation

    Returns:
    --------
    str
        Human-readable summary
    """
    status = validation_results.get('status', 'unknown')
    is_valid = validation_results.get('is_valid', False)
    errors = validation_results.get('errors', {})

    # Count errors by category
    schema_count = len(errors.get('schema_errors', []))
    quality_count = len(errors.get('data_quality_issues', []))
    other_count = len(errors.get('other_errors', []))
    total_count = schema_count + quality_count + other_count

    # Build summary
    summary_parts = []

    if status == 'complete':
        summary_parts.append("✅ Validation passed with no issues")
    elif status == 'partial':
        summary_parts.append(f"⚠️  Validation passed with {total_count} warning(s)")
    else:
        summary_parts.append(f"❌ Validation failed with {total_count} error(s)")

    if total_count > 0:
        error_breakdown = []
        if schema_count > 0:
            error_breakdown.append(f"{schema_count} schema")
        if quality_count > 0:
            error_breakdown.append(f"{quality_count} data quality")
        if other_count > 0:
            error_breakdown.append(f"{other_count} other")

        summary_parts.append(f"Issues found: {', '.join(error_breakdown)}")

    return " - ".join(summary_parts)


def format_error_for_display(error: Dict[str, Any]) -> str:
    """
    Format an error for display in the UI.

    Parameters:
    -----------
    error : dict
        Formatted error dictionary

    Returns:
    --------
    str
        Display-friendly error string
    """
    return f"**{error['type']}**: {error['description']}"