"""
Schema-based validation for CLIF tables.

This module provides validation functions using clifpy schemas
but with efficient Polars-based loading.
"""

import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional


# Data type mapping from schema to pandas
TYPE_MAPPING = {
    'VARCHAR': ['object', 'string'],
    'DOUBLE': ['float64', 'float32', 'int64', 'int32'],  # Allow numeric types
    'INTEGER': ['int64', 'int32', 'Int64', 'Int32'],
    'BIGINT': ['int64', 'Int64'],
    'BOOLEAN': ['bool', 'boolean'],
    'DATETIME': ['datetime64[ns]', 'datetime64[ns, UTC]', 'datetime64[ns, America/New_York]', 'datetime64']  # Allow any timezone
}


def load_schema(table_name: str) -> Dict[str, Any]:
    """
    Load YAML schema for a table.

    Parameters
    ----------
    table_name : str
        Name of the table (e.g., 'vitals', 'labs')

    Returns
    -------
    dict
        Schema dictionary
    """
    schema_dir = Path(__file__).parent / 'schemas'
    schema_file = schema_dir / f'{table_name}_schema.yaml'

    if not schema_file.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_file}")

    with open(schema_file, 'r') as f:
        schema = yaml.safe_load(f)

    return schema


def validate_dataframe(df: pd.DataFrame, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Validate a DataFrame against a schema.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    schema : dict
        Schema dictionary from load_schema()

    Returns
    -------
    list
        List of error dictionaries
    """
    errors = []

    # Validate required columns
    errors.extend(validate_required_columns(df, schema))

    # Validate data types
    errors.extend(validate_data_types(df, schema))

    # Validate categories (permissible values)
    errors.extend(validate_categories(df, schema))

    # Validate ranges (for tables with ranges like vitals)
    if 'vital_ranges' in schema:
        errors.extend(validate_ranges(df, schema))

    return errors


def validate_required_columns(df: pd.DataFrame, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Validate that all required columns are present.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    schema : dict
        Schema dictionary

    Returns
    -------
    list
        List of missing column errors
    """
    errors = []
    required_columns = schema.get('required_columns', [])

    for col in required_columns:
        if col not in df.columns:
            errors.append({
                'type': 'missing_required_column',
                'column': col,
                'message': f"Required column '{col}' is missing"
            })

    return errors


def validate_data_types(df: pd.DataFrame, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Validate that columns have correct data types.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    schema : dict
        Schema dictionary

    Returns
    -------
    list
        List of data type errors
    """
    errors = []

    for col_spec in schema.get('columns', []):
        col_name = col_spec['name']
        expected_type = col_spec.get('data_type')

        if col_name not in df.columns:
            continue  # Skip missing columns (handled by required_columns check)

        actual_dtype = str(df[col_name].dtype)

        # Check if actual type matches any acceptable type for this schema type
        acceptable_types = TYPE_MAPPING.get(expected_type, [])

        # Special handling for datetime columns - accept any timezone variant
        if expected_type == 'DATETIME':
            if not (actual_dtype.startswith('datetime64') or 'datetime' in actual_dtype.lower()):
                errors.append({
                    'type': 'incorrect_data_type',
                    'column': col_name,
                    'expected': expected_type,
                    'actual': actual_dtype,
                    'message': f"Column '{col_name}' has incorrect data type. Expected {expected_type}, got {actual_dtype}"
                })
        elif actual_dtype not in acceptable_types:
            # Check for string-like columns (VARCHAR) - be more lenient
            if expected_type == 'VARCHAR' and ('object' in actual_dtype or 'string' in actual_dtype.lower()):
                continue

            errors.append({
                'type': 'incorrect_data_type',
                'column': col_name,
                'expected': expected_type,
                'actual': actual_dtype,
                'message': f"Column '{col_name}' has incorrect data type. Expected {expected_type}, got {actual_dtype}"
            })

    return errors


def validate_categories(df: pd.DataFrame, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Validate that categorical columns only contain permissible values.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    schema : dict
        Schema dictionary

    Returns
    -------
    list
        List of category validation errors
    """
    errors = []

    for col_spec in schema.get('columns', []):
        col_name = col_spec['name']

        if col_name not in df.columns:
            continue

        if not col_spec.get('is_category_column', False):
            continue

        permissible_values = col_spec.get('permissible_values', [])
        if not permissible_values:
            continue

        # Get unique non-null values in the column
        actual_values = df[col_name].dropna().unique()

        # Find invalid values
        invalid_values = [v for v in actual_values if v not in permissible_values]

        if invalid_values:
            errors.append({
                'type': 'invalid_category_values',
                'column': col_name,
                'invalid_values': invalid_values[:10],  # Limit to first 10
                'count': len(df[df[col_name].isin(invalid_values)]),
                'permissible_values': permissible_values,
                'message': f"Column '{col_name}' contains {len(invalid_values)} invalid category value(s). Examples: {invalid_values[:5]}"
            })

    return errors


def validate_ranges(df: pd.DataFrame, schema: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Validate that numeric values fall within expected ranges.

    Specifically for vitals table with vital_ranges.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    schema : dict
        Schema dictionary

    Returns
    -------
    list
        List of range validation errors
    """
    errors = []

    vital_ranges = schema.get('vital_ranges', {})
    if not vital_ranges:
        return errors

    # Check required columns
    if 'vital_category' not in df.columns or 'vital_value' not in df.columns:
        return errors

    # For each vital category, check ranges
    for vital_cat, range_spec in vital_ranges.items():
        min_val = range_spec.get('min')
        max_val = range_spec.get('max')

        # Filter to this vital category
        vital_data = df[df['vital_category'] == vital_cat]
        if len(vital_data) == 0:
            continue

        # Check for out-of-range values
        if min_val is not None:
            below_min = vital_data[vital_data['vital_value'] < min_val]
            if len(below_min) > 0:
                errors.append({
                    'type': 'value_below_range',
                    'column': 'vital_value',
                    'vital_category': vital_cat,
                    'min': min_val,
                    'count': len(below_min),
                    'message': f"{vital_cat}: {len(below_min)} value(s) below minimum ({min_val})"
                })

        if max_val is not None:
            above_max = vital_data[vital_data['vital_value'] > max_val]
            if len(above_max) > 0:
                errors.append({
                    'type': 'value_above_range',
                    'column': 'vital_value',
                    'vital_category': vital_cat,
                    'max': max_val,
                    'count': len(above_max),
                    'message': f"{vital_cat}: {len(above_max)} value(s) above maximum ({max_val})"
                })

    return errors
