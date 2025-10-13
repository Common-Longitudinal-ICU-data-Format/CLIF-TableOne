"""
Shared validation formatter for CLIF tables.

This module provides consistent error formatting and status determination
aligned with the CLIF Report Card format.
"""

from typing import Dict, Any, List


class ValidationFormatter:
    """Format clifpy validation errors for display in reports."""

    def format_clifpy_error(self, error: Dict[str, Any], row_count: int, table_name: str) -> Dict[str, Any]:
        """
        Format a clifpy error object for display in reports.

        This method filters and formats clifpy errors to show only relevant
        validation issues (schema, data quality, outliers), excluding low-level
        clifpy internals.

        Args:
            error: Raw error object from clifpy validation
            row_count: Total number of rows in the table (for percentage calculations)
            table_name: Name of the table being validated

        Returns:
            Formatted error dictionary with type, description, category, and details
        """
        error_type = error.get('type', 'unknown')

        # Use clifpy's message if available, otherwise build one
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

        # Fallback: use error as-is or stringify
        if not message:
            message = str(error)

        result = {
            'type': display_type,
            'description': message,
            'category': category
        }

        # Add details if any were collected
        if details:
            result['details'] = details

        return result

    def categorize_errors(self, formatted_errors: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Categorize formatted errors into schema, data quality, and other categories.

        Args:
            formatted_errors: List of formatted error dictionaries

        Returns:
            Dictionary with categorized errors
        """
        schema_errors = []
        data_quality_issues = []
        other_errors = []

        for error in formatted_errors:
            if error['category'] == 'schema':
                schema_errors.append(error)
            elif error['category'] == 'data_quality':
                data_quality_issues.append(error)
            else:
                other_errors.append(error)

        return {
            'schema_errors': schema_errors,
            'data_quality_issues': data_quality_issues,
            'other_errors': other_errors
        }

    def determine_status(
        self,
        schema_errors: List[Dict[str, Any]],
        data_quality_issues: List[Dict[str, Any]],
        required_columns: List[str] = None,
        table_name: str = None
    ) -> str:
        """
        Determine validation status (complete/partial/incomplete) based on errors.

        Status Logic:
        - INCOMPLETE (Red): Missing required columns OR non-castable datatype errors
                           OR 100% missing values in required columns
        - PARTIAL (Yellow): Has required columns but missing categorical values
        - COMPLETE (Green): All required columns present, all categorical values present

        Args:
            schema_errors: List of schema-related errors
            data_quality_issues: List of data quality errors
            required_columns: List of required column names from schema (optional)
            table_name: Name of the table (for table-specific exceptions)

        Returns:
            Status string: 'complete', 'partial', or 'incomplete'
        """
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
