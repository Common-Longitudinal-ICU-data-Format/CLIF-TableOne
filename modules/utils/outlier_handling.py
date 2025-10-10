"""
Outlier Handling Utility for CLIF TableOne

This module provides utilities to handle outliers based on configuration file.
"""

import yaml
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import os


def load_outlier_config(config_path: str = 'config/outlier_config.yaml') -> Dict[str, Any]:
    """
    Load outlier configuration from YAML file.

    Parameters:
    -----------
    config_path : str
        Path to the outlier configuration file

    Returns:
    --------
    dict
        Outlier configuration dictionary (tables section)
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            # Extract the 'tables' section from the config
            return config.get('tables', {}) if config else {}
    except FileNotFoundError:
        print(f"Warning: Outlier config file not found at {config_path}")
        return {}
    except Exception as e:
        print(f"Error loading outlier config: {e}")
        return {}


def apply_outlier_ranges(
    df: pd.DataFrame,
    table_name: str,
    outlier_config: Optional[Dict[str, Any]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply outlier ranges to dataframe columns and replace outliers with NA.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    table_name : str
        Name of the table (e.g., 'crrt_therapy')
    outlier_config : dict, optional
        Outlier configuration dictionary. If None, loads from default path.

    Returns:
    --------
    tuple
        (cleaned_df, outlier_stats)
        - cleaned_df: DataFrame with outliers replaced by NA
        - outlier_stats: Dictionary with outlier statistics per column
    """
    if outlier_config is None:
        outlier_config = load_outlier_config()

    # Check if table has outlier config
    if table_name not in outlier_config:
        return df.copy(), {}

    table_config = outlier_config[table_name]
    cleaned_df = df.copy()
    outlier_stats = {}

    for column, ranges in table_config.items():
        if column not in cleaned_df.columns:
            continue

        min_val = ranges.get('min')
        max_val = ranges.get('max')

        # Get original value counts
        original_count = cleaned_df[column].notna().sum()

        # Create mask for outliers (excluding NaN values)
        outlier_mask = pd.Series([False] * len(cleaned_df), index=cleaned_df.index)

        if min_val is not None:
            outlier_mask |= (cleaned_df[column].notna() & (cleaned_df[column] < min_val))
        if max_val is not None:
            outlier_mask |= (cleaned_df[column].notna() & (cleaned_df[column] > max_val))

        # Count outliers
        outlier_count = outlier_mask.sum()

        # Replace outliers with NA
        cleaned_df.loc[outlier_mask, column] = pd.NA

        # Store statistics
        outlier_stats[column] = {
            'outlier_count': int(outlier_count),
            'original_valid_count': int(original_count),
            'outlier_percentage': round((outlier_count / original_count * 100) if original_count > 0 else 0, 2),
            'min_threshold': min_val,
            'max_threshold': max_val
        }

    return cleaned_df, outlier_stats


def get_outlier_info(
    df: pd.DataFrame,
    column: str,
    table_name: str,
    outlier_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Get outlier information for a specific column.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    column : str
        Column name
    table_name : str
        Name of the table
    outlier_config : dict, optional
        Outlier configuration dictionary

    Returns:
    --------
    dict
        Outlier information including thresholds and counts
    """
    if outlier_config is None:
        outlier_config = load_outlier_config()

    if table_name not in outlier_config or column not in outlier_config[table_name]:
        return {
            'has_outlier_config': False,
            'column': column
        }

    ranges = outlier_config[table_name][column]
    min_val = ranges.get('min')
    max_val = ranges.get('max')

    # Count outliers (excluding NaN values)
    outlier_mask = pd.Series([False] * len(df), index=df.index)
    if min_val is not None:
        outlier_mask |= (df[column].notna() & (df[column] < min_val))
    if max_val is not None:
        outlier_mask |= (df[column].notna() & (df[column] > max_val))

    outlier_count = outlier_mask.sum()
    valid_count = df[column].notna().sum()

    return {
        'has_outlier_config': True,
        'column': column,
        'min_threshold': min_val,
        'max_threshold': max_val,
        'outlier_count': int(outlier_count),
        'valid_count': int(valid_count),
        'outlier_percentage': round((outlier_count / valid_count * 100) if valid_count > 0 else 0, 2)
    }
