"""
Validation Utilities for CLIF Analysis

This module provides functions to format and categorize validation errors
from clifpy, similar to the clif_report_card.py implementation.
"""

from typing import Dict, Any, List


def format_clifpy_error(error: Dict[str, Any], row_count: int = 0, table_name: str = None) -> Dict[str, Any]:
    """
    Format a clifpy error object for display (from clif_report_card.py logic).

    This method filters and formats clifpy errors to show only relevant
    validation issues (schema, data quality, outliers), excluding low-level
    clifpy internals.

    Parameters:
    -----------
    error : dict
        Error dictionary from clifpy validation
    row_count : int
        Total number of rows in the table (for percentage calculations)
    table_name : str, optional
        Name of the table being validated

    Returns:
    --------
    dict
        Formatted error with type, description, category, and details
    """
    error_type = error.get('type', 'unknown')
    message = error.get('message', '')

    # Determine category based on error type
    category = 'other'
    display_type = error_type.replace('_', ' ').title()

    # Store original error details for PDF display
    details = {}

    # Schema-related errors
    if error_type == 'missing_columns':
        category = 'schema'
        display_type = 'Missing Required Columns'
        if not message:
            columns = error.get('columns', [])
            message = f"Required columns not found: {', '.join(columns)}"
        details['missing_columns'] = error.get('columns', [])

    elif error_type in ['datatype_mismatch', 'datatype_castable']:
        category = 'schema'
        display_type = 'Datatype Casting Error'
        if not message:
            column = error.get('column', 'unknown')
            expected = error.get('expected', 'unknown')
            actual = error.get('actual', 'unknown')
            message = f"Column '{column}' has type {actual} but expected {expected}"
        details['column'] = error.get('column')
        details['expected_type'] = error.get('expected')
        details['actual_type'] = error.get('actual')

    # Data quality errors
    elif error_type == 'null_values':
        category = 'data_quality'
        display_type = 'Missing Values'
        if not message:
            column = error.get('column', 'unknown')
            count = error.get('count', 0)
            percentage = (count / row_count * 100) if row_count > 0 else 0
            message = f"Column '{column}' has {count:,} missing values ({percentage:.1f}%)"
        details['column'] = error.get('column')
        details['missing_count'] = error.get('count', 0)
        details['total_rows'] = row_count

    elif error_type in ['invalid_category', 'invalid_categorical_values']:
        category = 'data_quality'
        display_type = 'Invalid Categories'
        if not message:
            column = error.get('column', 'unknown')
            invalid_values = error.get('invalid_values', error.get('values', []))
            truncated = invalid_values[:3]
            message = f"Column '{column}' contains invalid values: {', '.join(map(str, truncated))}"
            if len(invalid_values) > 3:
                message += f" (and {len(invalid_values) - 3} more)"
        details['column'] = error.get('column')
        details['invalid_values'] = error.get('invalid_values', error.get('values', []))

    elif error_type == 'missing_categorical_values':
        category = 'data_quality'
        display_type = 'Missing Categorical Values'
        if not message:
            column = error.get('column', 'unknown')
            missing_values = error.get('missing_values', [])
            total_missing = error.get('total_missing', len(missing_values))
            if missing_values:
                # Show all missing values, not just a summary
                values_str = str(missing_values) if len(missing_values) <= 10 else str(missing_values[:10]) + f'... ({len(missing_values) - 10} more)'
                message = f"Column '{column}' is missing {total_missing} expected category values: {values_str}"
            else:
                message = f"Column '{column}' is missing {total_missing} expected category values"
        details['column'] = error.get('column')
        details['missing_values'] = error.get('missing_values', [])
        details['total_missing'] = error.get('total_missing', len(error.get('missing_values', [])))

    elif error_type == 'duplicate_check':
        category = 'data_quality'
        display_type = 'Duplicate Check'
        if not message:
            duplicate_count = error.get('duplicate_rows', 0)
            total_rows = error.get('total_rows', row_count)
            keys = error.get('composite_keys', [])
            keys_str = ', '.join(keys) if keys else 'composite keys'
            message = f"Found {duplicate_count} duplicate rows out of {total_rows} total rows based on keys: {keys_str}"
        details['duplicate_rows'] = error.get('duplicate_rows', 0)
        details['composite_keys'] = error.get('composite_keys', [])

    elif error_type == 'unit_validation':
        category = 'data_quality'
        display_type = 'Unit Validation'
        if not message:
            cat = error.get('category', 'unknown')
            unexpected_units = error.get('unexpected_units', [])
            expected_units = error.get('expected_units', [])
            if unexpected_units and expected_units:
                message = f"Category '{cat}' has unexpected units: {', '.join(unexpected_units[:3])}, expected: {', '.join(expected_units)}"
            else:
                message = f"Unit validation issue for category '{cat}'"
        details['category'] = error.get('category')
        details['unexpected_units'] = error.get('unexpected_units', [])
        details['expected_units'] = error.get('expected_units', [])

    elif error_type in ['below_range', 'above_range', 'unknown_vital_category']:
        category = 'data_quality'
        display_type = 'Range Validation'
        if not message:
            vital_category = error.get('vital_category', 'unknown')
            if error_type == 'below_range':
                min_val = error.get('observed_min', 'N/A')
                expected_min = error.get('expected_min', 'N/A')
                message = f"Values below expected minimum for {vital_category} (found: {min_val}, expected: >={expected_min})"
            elif error_type == 'above_range':
                max_val = error.get('observed_max', 'N/A')
                expected_max = error.get('expected_max', 'N/A')
                message = f"Values above expected maximum for {vital_category} (found: {max_val}, expected: <={expected_max})"
            else:  # unknown_vital_category
                message = f"Unknown vital category '{vital_category}' found in data"
        details['vital_category'] = error.get('vital_category')
        if error_type == 'below_range':
            details['observed_min'] = error.get('observed_min')
            details['expected_min'] = error.get('expected_min')
        elif error_type == 'above_range':
            details['observed_max'] = error.get('observed_max')
            details['expected_max'] = error.get('expected_max')

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

    # Fallback: use error as-is or stringify
    if not message:
        message = str(error)

    result = {
        'type': display_type,
        'description': message,
        'category': category,
        'raw_type': error_type
    }

    # Add details if any were collected
    if details:
        result['details'] = details

    return result


def determine_validation_status(errors: List[Dict[str, Any]],
                               required_columns: List[str] = None,
                               table_name: str = None) -> str:
    """
    Determine overall validation status based on errors.

    Status Logic (aligned with CLIF Report Card):
    - INCOMPLETE (Red): Missing required columns OR non-castable datatype errors
                       OR 100% missing values in required columns
    - PARTIAL (Yellow): Has required columns but missing categorical values
    - COMPLETE (Green): All required columns present, all categorical values present

    Parameters:
    -----------
    errors : list
        List of formatted error dictionaries
    required_columns : list, optional
        List of required column names from schema
    table_name : str, optional
        Name of the table (for table-specific exceptions)

    Returns:
    --------
    str
        Status: 'complete', 'partial', or 'incomplete'
    """
    if not errors:
        return 'complete'

    # Separate errors by category
    schema_errors = [e for e in errors if e.get('category') == 'schema']
    data_quality_issues = [e for e in errors if e.get('category') == 'data_quality']

    # Red (incomplete): Missing required columns OR NON-CASTABLE datatype errors
    has_missing_columns = any(
        error.get('type') == 'Missing Required Columns'
        for error in schema_errors
    )

    # Only treat as INCOMPLETE if datatype CANNOT be cast
    # Errors that say "can be cast to" should not trigger INCOMPLETE
    has_datatype_errors = any(
        error.get('type') == 'Datatype Casting Error'
        and 'can be cast to' not in error.get('description', '')
        for error in schema_errors
    )

    # Check for 100% missing values in REQUIRED columns only (red condition)
    has_100_percent_missing_required = False

    # Define columns that should NOT trigger INCOMPLETE even if 100% null and required
    # Table-specific exceptions
    table_specific_exceptions = {
        'patient_assessments': {'numerical_value', 'categorical_value', 'text_value'},
        'crrt_therapy': {'pre_filter_replacement_fluid_rate', 'post_filter_replacement_fluid_rate'},
        'respiratory_support': {
            'device_category', 'mode_category', 'vent_brand_name', 'tracheostomy',
            'fio2_set', 'lpm_set', 'tidal_volume_set', 'resp_rate_set',
            'pressure_control_set', 'pressure_support_set', 'flow_rate_set',
            'peak_inspiratory_pressure_set', 'inspiratory_time_set', 'peep_set',
            'tidal_volume_obs', 'resp_rate_obs', 'plateau_pressure_obs',
            'peak_inspiratory_pressure_obs', 'peep_obs', 'minute_vent_obs',
            'mean_airway_pressure_obs'
        }
    }

    if required_columns:
        # Get exceptions for this specific table
        exceptions_for_table = table_specific_exceptions.get(table_name, set()) if table_name else set()

        for error in data_quality_issues:
            if error.get('type') == 'Missing Values':
                description = error.get('description', '')

                # Check if this is 100% missing (look for "100.0%" or "100.00%")
                if '100.0%' in description or '100.00%' in description:
                    # Extract column name from description
                    if "Column '" in description:
                        try:
                            column_name = description.split("Column '")[1].split("'")[0]

                            # Check if column should be excluded from INCOMPLETE trigger
                            # 1. Check if it ends with _type (pattern-based exception)
                            # 2. Check if it's in the table-specific exceptions
                            if column_name.endswith('_type'):
                                continue
                            elif column_name in exceptions_for_table:
                                continue

                            # Check if this column is in the required_columns list
                            if column_name in required_columns:
                                has_100_percent_missing_required = True
                                break
                        except Exception:
                            pass
    else:
        # Fallback: if no schema available, treat all 100% missing as problematic
        has_100_percent_missing_required = any(
            error.get('type') == 'Missing Values'
            and ('100.0%' in error.get('description', '') or '100.00%' in error.get('description', ''))
            for error in data_quality_issues
        )

    # Yellow (partial): Has required columns but missing categorical values
    has_categorical_issues = any(
        error.get('type') in ['Invalid Categories', 'Missing Categorical Values']
        for error in data_quality_issues
    )

    if has_missing_columns or has_datatype_errors or has_100_percent_missing_required:
        # Red: Missing required columns OR datatype casting problems OR 100% missing values in required columns
        return 'incomplete'
    elif has_categorical_issues:
        # Yellow: Has required columns but missing some required categorical values
        return 'partial'
    else:
        # Green: All required columns present, all categorical values present
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


def classify_errors_by_status_impact(
    errors: Dict[str, List[Dict[str, Any]]],
    required_columns: List[str] = None,
    table_name: str = None,
    config_timezone: str = None
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Classify errors into status-affecting vs informational categories.

    Status-Affecting Errors (require feedback):
    - Missing Required Columns (INCOMPLETE)
    - Non-castable Datatype Errors (INCOMPLETE)
    - 100% Missing Values in Required Columns (INCOMPLETE)
    - Missing Categorical Values (PARTIAL)
    - Invalid Categories (PARTIAL)

    Informational Issues (no feedback required):
    - Missing values in non-required columns
    - Outlier validation errors
    - Duplicate check warnings
    - Unit validation issues
    - Range validation errors

    Filtered Out Completely (not shown at all):
    - Missing values in columns ending with '_name'
    - Timezone errors when timezone is UTC or config["timezone"]

    Parameters:
    -----------
    errors : dict
        Categorized errors from format_errors()
    required_columns : list, optional
        List of required column names from schema
    table_name : str, optional
        Name of the table (for table-specific exceptions)
    config_timezone : str, optional
        Configured timezone from config (defaults to UTC if not provided)

    Returns:
    --------
    dict
        Dictionary with 'status_affecting' and 'informational' keys,
        each containing categorized error lists
    """
    # Define columns that should NOT trigger INCOMPLETE even if 100% null and required
    table_specific_exceptions = {
        'patient_assessments': {'numerical_value', 'categorical_value', 'text_value'},
        'crrt_therapy': {'pre_filter_replacement_fluid_rate', 'post_filter_replacement_fluid_rate'},
        'respiratory_support': {
            'device_category', 'mode_category', 'vent_brand_name', 'tracheostomy',
            'fio2_set', 'lpm_set', 'tidal_volume_set', 'resp_rate_set',
            'pressure_control_set', 'pressure_support_set', 'flow_rate_set',
            'peak_inspiratory_pressure_set', 'inspiratory_time_set', 'peep_set',
            'tidal_volume_obs', 'resp_rate_obs', 'plateau_pressure_obs',
            'peak_inspiratory_pressure_obs', 'peep_obs', 'minute_vent_obs',
            'mean_airway_pressure_obs'
        }
    }

    exceptions_for_table = table_specific_exceptions.get(table_name, set()) if table_name else set()

    # Set default timezone if not provided
    if config_timezone is None:
        config_timezone = 'UTC'

    def should_filter_out(error: Dict[str, Any]) -> bool:
        """
        Determine if an error should be completely filtered out (not shown at all).

        Returns True if the error should be filtered out.
        """
        error_type = error.get('type')
        description = error.get('description', '')
        details = error.get('details', {})

        # Filter out missing values in _name columns
        if error_type == 'Missing Values':
            # Extract column name from description
            if "Column '" in description:
                try:
                    column_name = description.split("Column '")[1].split("'")[0]
                    if column_name.endswith('_name'):
                        return True
                except Exception:
                    pass

            # Also check details if available
            column = details.get('column', '')
            if column.endswith('_name'):
                return True

        # Filter out timezone errors for UTC or config timezone
        # Check if the error mentions timezone issues
        if 'timezone' in description.lower() or 'timezone' in error.get('raw_type', ''):
            # Extract timezone from error if possible
            error_timezone = None

            # Try to find timezone in description (common patterns)
            # e.g., "timezone 'America/New_York'", "timezone: EST", etc.
            if 'UTC' in description or "'UTC'" in description or '"UTC"' in description:
                error_timezone = 'UTC'
            elif config_timezone in description:
                error_timezone = config_timezone

            # If the error is about UTC or the config timezone, filter it out
            if error_timezone in ['UTC', config_timezone]:
                return True

        return False

    status_affecting = {
        'schema_errors': [],
        'data_quality_issues': [],
        'other_errors': []
    }

    informational = {
        'schema_errors': [],
        'data_quality_issues': [],
        'other_errors': []
    }

    # Process schema errors
    for error in errors.get('schema_errors', []):
        # Skip errors that should be filtered out completely
        if should_filter_out(error):
            continue

        error_type = error.get('type')

        # Missing Required Columns - always status-affecting (INCOMPLETE)
        if error_type == 'Missing Required Columns':
            status_affecting['schema_errors'].append(error)

        # Datatype Casting Error - status-affecting if NON-CASTABLE
        elif error_type == 'Datatype Casting Error':
            if 'can be cast to' not in error.get('description', ''):
                status_affecting['schema_errors'].append(error)
            else:
                informational['schema_errors'].append(error)

        else:
            # Other schema errors (file_not_found, empty_table, etc.)
            informational['schema_errors'].append(error)

    # Process data quality issues
    for error in errors.get('data_quality_issues', []):
        # Skip errors that should be filtered out completely
        if should_filter_out(error):
            continue

        error_type = error.get('type')

        # Missing Categorical Values - always status-affecting (PARTIAL)
        if error_type == 'Missing Categorical Values':
            status_affecting['data_quality_issues'].append(error)

        # Invalid Categories - always status-affecting (PARTIAL)
        elif error_type == 'Invalid Categories':
            status_affecting['data_quality_issues'].append(error)

        # Missing Values - status-affecting ONLY if 100% missing in required column
        elif error_type == 'Missing Values':
            description = error.get('description', '')

            # Check if this is 100% missing
            if '100.0%' in description or '100.00%' in description:
                # Extract column name
                if "Column '" in description:
                    try:
                        column_name = description.split("Column '")[1].split("'")[0]

                        # Check if column should be excluded from status-affecting
                        if column_name.endswith('_type'):
                            informational['data_quality_issues'].append(error)
                        elif column_name in exceptions_for_table:
                            informational['data_quality_issues'].append(error)
                        elif required_columns and column_name in required_columns:
                            # 100% missing in required column - status-affecting (INCOMPLETE)
                            status_affecting['data_quality_issues'].append(error)
                        else:
                            # 100% missing but not required - informational
                            informational['data_quality_issues'].append(error)
                    except Exception:
                        informational['data_quality_issues'].append(error)
                else:
                    informational['data_quality_issues'].append(error)
            else:
                # Partial missingness - always informational
                informational['data_quality_issues'].append(error)

        # All other data quality issues are informational
        else:
            informational['data_quality_issues'].append(error)

    # All other errors are informational
    for error in errors.get('other_errors', []):
        # Skip errors that should be filtered out completely
        if should_filter_out(error):
            continue

        informational['other_errors'].append(error)

    return {
        'status_affecting': status_affecting,
        'informational': informational
    }