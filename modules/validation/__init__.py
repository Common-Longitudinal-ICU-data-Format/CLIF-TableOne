"""
Validation module for CLIF tables.

This module provides schema-based validation using Polars for memory efficiency.
It replaces clifpy's validation backend while maintaining the same output format.
"""

from .schema_validator import (
    load_schema,
    validate_dataframe,
    validate_required_columns,
    validate_data_types,
    validate_categories,
    validate_ranges
)

from .polars_loader import load_with_filter

__all__ = [
    'load_schema',
    'validate_dataframe',
    'validate_required_columns',
    'validate_data_types',
    'validate_categories',
    'validate_ranges',
    'load_with_filter'
]
