"""
Utility functions for EDA Streamlit app
Handles config loading, data filtering, binning, and analysis
"""

import json
import yaml
import os
from typing import Dict, List, Tuple, Any
import polars as pl
import pandas as pd
import numpy as np
from scipy import stats


# ============================================================================
# Configuration Loading
# ============================================================================

def load_clif_config(config_path: str) -> Dict[str, Any]:
    """
    Load CLIF configuration file.

    Args:
        config_path: Path to config.json

    Returns:
        Dictionary with tables_path, file_type, etc.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    required_keys = ['tables_path', 'file_type']
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    return config


def load_outlier_config(config_path: str = 'get_ecdf/ecdf_config/outlier_config.yaml') -> Dict[str, Any]:
    """
    Load outlier configuration for labs and vitals.

    Returns:
        Dictionary with outlier ranges for each category
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Outlier config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def load_lab_vital_config(config_path: str = 'get_ecdf/ecdf_config/lab_vital_config.yaml') -> Dict[str, Any]:
    """
    Load lab and vital binning configuration.

    Returns:
        Dictionary with normal ranges and bin counts for each category
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Lab/vital config not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


# ============================================================================
# ADT Processing - Get ICU Hospitalization IDs
# ============================================================================

def get_icu_hospitalization_ids(tables_path: str, file_type: str) -> List[str]:
    """
    Load ADT table and extract hospitalization IDs for ICU stays.
    Uses Polars streaming for memory efficiency.

    Args:
        tables_path: Path to CLIF data directory
        file_type: File type (e.g., 'parquet')

    Returns:
        List of unique hospitalization_ids with ICU stays
    """
    adt_path = os.path.join(tables_path, f'clif_adt.{file_type}')

    if not os.path.exists(adt_path):
        raise FileNotFoundError(f"ADT file not found: {adt_path}")

    # Scan ADT with Polars (lazy evaluation)
    adt_lazy = pl.scan_parquet(adt_path)

    # Filter to ICU locations only
    icu_stays = adt_lazy.filter(
        pl.col('location_category').str.to_lowercase() == 'icu'
    ).select('hospitalization_id').unique()

    # Collect with streaming
    icu_df = icu_stays.collect(streaming=True)

    # Convert to list
    hosp_ids = icu_df['hospitalization_id'].to_list()

    return hosp_ids


# ============================================================================
# Lab/Vital Data Loading
# ============================================================================

def load_lab_vital_data(
    table_type: str,
    category: str,
    icu_hosp_ids: List[str],
    tables_path: str,
    file_type: str
) -> pd.DataFrame:
    """
    Load lab or vital data for a specific category, filtered to ICU stays.

    Args:
        table_type: 'labs' or 'vitals'
        category: Specific lab or vital category
        icu_hosp_ids: List of ICU hospitalization IDs
        tables_path: Path to CLIF data directory
        file_type: File type (e.g., 'parquet')

    Returns:
        Pandas DataFrame with filtered data
    """
    # Determine file path and column names
    if table_type == 'labs':
        file_path = os.path.join(tables_path, f'clif_labs.{file_type}')
        category_col = 'lab_category'
        value_col = 'lab_value_numeric'
        datetime_col = 'lab_result_dttm'
    else:  # vitals
        file_path = os.path.join(tables_path, f'clif_vitals.{file_type}')
        category_col = 'vital_category'
        value_col = 'vital_value'
        datetime_col = 'recorded_dttm'

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Scan with Polars (lazy)
    data_lazy = pl.scan_parquet(file_path)

    # Filter to:
    # 1. ICU hospitalization IDs
    # 2. Selected category
    # 3. Select only needed columns
    filtered = data_lazy.filter(
        pl.col('hospitalization_id').is_in(icu_hosp_ids) &
        (pl.col(category_col) == category)
    ).select([
        'hospitalization_id',
        datetime_col,
        category_col,
        value_col
    ])

    # Collect with streaming
    df_pl = filtered.collect(streaming=True)

    # Convert to pandas
    df = df_pl.to_pandas()

    return df


# ============================================================================
# Outlier Removal
# ============================================================================

def remove_outliers(
    df: pd.DataFrame,
    value_col: str,
    outlier_min: float,
    outlier_max: float
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Remove values outside the outlier range.

    Args:
        df: DataFrame with values
        value_col: Name of value column
        outlier_min: Minimum acceptable value
        outlier_max: Maximum acceptable value

    Returns:
        Tuple of (filtered DataFrame, stats dict)
    """
    original_count = len(df)

    # Filter to outlier range
    df_filtered = df[
        (df[value_col] >= outlier_min) &
        (df[value_col] <= outlier_max)
    ].copy()

    removed_count = original_count - len(df_filtered)

    stats = {
        'original_count': original_count,
        'removed_count': removed_count,
        'remaining_count': len(df_filtered),
        'removal_percentage': (removed_count / original_count * 100) if original_count > 0 else 0
    }

    return df_filtered, stats


# ============================================================================
# Binning Functions
# ============================================================================

def create_bins_for_segment(
    data: pd.Series,
    segment_min: float,
    segment_max: float,
    num_bins: int,
    segment_name: str,
    extra_bins_last: int = 0,
    split_first: bool = False
) -> List[Dict[str, Any]]:
    """
    Create quantile-based bins for a segment.

    If extra_bins_last > 0:
    - split_first=True: Split FIRST bin (most extreme low - for below_normal)
    - split_first=False: Split LAST bin (most extreme high - for above_normal)

    Args:
        data: Data values
        segment_min: Minimum value for segment
        segment_max: Maximum value for segment
        num_bins: Number of bins
        segment_name: Name of segment ('below', 'normal', 'above')
        extra_bins_last: Number of quantile bins to split the extreme bin into
        split_first: If True, split first bin; if False, split last bin

    Returns:
        List of bin dictionaries with min, max, count, percentage
    """
    # Filter data to segment range
    segment_data = data[(data >= segment_min) & (data <= segment_max)]

    if len(segment_data) == 0:
        return []

    # If only 1 bin requested or very few values
    if num_bins == 1 or len(segment_data) < num_bins * 2:
        return [{
            'segment': segment_name,
            'bin_num': 1,
            'bin_min': float(segment_data.min()),
            'bin_max': float(segment_data.max()),
            'count': len(segment_data),
            'percentage': 100.0
        }]

    try:
        # STEP 1: Create regular quantile bins FIRST
        quantiles = np.linspace(0, 1, num_bins + 1)
        bins_cut, bin_edges = pd.qcut(segment_data, q=quantiles, retbins=True, duplicates='drop')

        # Count values in each bin
        bin_counts = bins_cut.value_counts().sort_index()

        # Create bin info
        bins = []
        for i, (interval, count) in enumerate(bin_counts.items(), 1):
            bins.append({
                'segment': segment_name,
                'bin_num': i,
                'bin_min': float(interval.left),
                'bin_max': float(interval.right),
                'count': int(count),
                'percentage': float(count / len(segment_data) * 100)
            })

        # STEP 2: If extra_bins_last > 0, split the extreme bin into quantiles
        if extra_bins_last > 0 and len(bins) > 0:
            if split_first:
                # Split FIRST bin (for below_normal - most extreme low values)
                extreme_bin = bins[0]

                # Get data in the first bin ONLY
                extreme_bin_data = segment_data[
                    (segment_data >= extreme_bin['bin_min']) &
                    (segment_data <= extreme_bin['bin_max'])
                ]

                if len(extreme_bin_data) >= extra_bins_last * 2:
                    # Split this first bin into extra_bins_last quantile bins
                    tail_quantiles = np.linspace(0, 1, extra_bins_last + 1)
                    tail_bins_cut, tail_edges = pd.qcut(
                        extreme_bin_data,
                        q=tail_quantiles,
                        retbins=True,
                        duplicates='drop'
                    )

                    # Remove the original first bin
                    bins = bins[1:]

                    # Add the new split bins at the beginning
                    tail_counts = tail_bins_cut.value_counts().sort_index()
                    new_bins = []
                    for j, (interval, count) in enumerate(tail_counts.items(), 1):
                        new_bins.append({
                            'segment': segment_name,
                            'bin_num': j,
                            'bin_min': float(interval.left),
                            'bin_max': float(interval.right),
                            'count': int(count),
                            'percentage': float(count / len(segment_data) * 100)
                        })

                    # Renumber remaining bins
                    for bin_info in bins:
                        bin_info['bin_num'] = len(new_bins) + 1
                        new_bins.append(bin_info)

                    bins = new_bins

            else:
                # Split LAST bin (for above_normal - most extreme high values)
                extreme_bin = bins[-1]

                # Get data in the last bin ONLY
                extreme_bin_data = segment_data[
                    (segment_data >= extreme_bin['bin_min']) &
                    (segment_data <= extreme_bin['bin_max'])
                ]

                if len(extreme_bin_data) >= extra_bins_last * 2:
                    # Split this last bin into extra_bins_last quantile bins
                    tail_quantiles = np.linspace(0, 1, extra_bins_last + 1)
                    tail_bins_cut, tail_edges = pd.qcut(
                        extreme_bin_data,
                        q=tail_quantiles,
                        retbins=True,
                        duplicates='drop'
                    )

                    # Remove the original last bin
                    bins = bins[:-1]

                    # Add the new split bins
                    tail_counts = tail_bins_cut.value_counts().sort_index()
                    for j, (interval, count) in enumerate(tail_counts.items(), 1):
                        bins.append({
                            'segment': segment_name,
                            'bin_num': len(bins) + 1,
                            'bin_min': float(interval.left),
                            'bin_max': float(interval.right),
                            'count': int(count),
                            'percentage': float(count / len(segment_data) * 100)
                        })

        return bins

    except Exception as e:
        # Fallback: single bin if qcut fails
        return [{
            'segment': segment_name,
            'bin_num': 1,
            'bin_min': float(segment_data.min()),
            'bin_max': float(segment_data.max()),
            'count': len(segment_data),
            'percentage': 100.0
        }]


def create_all_bins(
    data: pd.Series,
    normal_lower: float,
    normal_upper: float,
    outlier_min: float,
    outlier_max: float,
    bins_below: int,
    bins_normal: int,
    bins_above: int,
    extra_bins_below: int = 0,
    extra_bins_above: int = 0
) -> List[Dict[str, Any]]:
    """
    Create bins for all segments (below, normal, above).

    Args:
        data: Data values (already filtered to outlier range)
        normal_lower: Lower bound of normal range
        normal_upper: Upper bound of normal range
        outlier_min: Minimum outlier value
        outlier_max: Maximum outlier value
        bins_below: Number of bins below normal
        bins_normal: Number of bins in normal range
        bins_above: Number of bins above normal
        extra_bins_below: Extra bins for last below_normal bin
        extra_bins_above: Extra bins for last above_normal bin

    Returns:
        List of all bin dictionaries
    """
    all_bins = []

    # Below normal segment
    if bins_below > 0 and outlier_min < normal_lower:
        below_bins = create_bins_for_segment(
            data, outlier_min, normal_lower, bins_below, 'below',
            extra_bins_last=extra_bins_below,
            split_first=True  # Split FIRST bin (most extreme low)
        )
        all_bins.extend(below_bins)

    # Normal segment
    if bins_normal > 0:
        normal_bins = create_bins_for_segment(
            data, normal_lower, normal_upper, bins_normal, 'normal'
        )
        all_bins.extend(normal_bins)

    # Above normal segment
    if bins_above > 0 and normal_upper < outlier_max:
        above_bins = create_bins_for_segment(
            data, normal_upper, outlier_max, bins_above, 'above',
            extra_bins_last=extra_bins_above,
            split_first=False  # Split LAST bin (most extreme high)
        )
        all_bins.extend(above_bins)

    return all_bins


# ============================================================================
# Statistics Calculation
# ============================================================================

def calculate_segment_stats(df: pd.DataFrame, value_col: str, bins: List[Dict]) -> Dict[str, Dict]:
    """
    Calculate statistics for each segment.

    Args:
        df: DataFrame with values
        value_col: Name of value column
        bins: List of bin dictionaries

    Returns:
        Dictionary with stats per segment
    """
    stats = {}

    for segment_name in ['below', 'normal', 'above']:
        segment_bins = [b for b in bins if b['segment'] == segment_name]

        if not segment_bins:
            continue

        # Get min and max for this segment
        seg_min = min(b['bin_min'] for b in segment_bins)
        seg_max = max(b['bin_max'] for b in segment_bins)

        # Filter data to this segment
        seg_data = df[(df[value_col] >= seg_min) & (df[value_col] <= seg_max)][value_col]

        if len(seg_data) > 0:
            stats[segment_name] = {
                'count': len(seg_data),
                'mean': float(seg_data.mean()),
                'median': float(seg_data.median()),
                'std': float(seg_data.std()),
                'min': float(seg_data.min()),
                'max': float(seg_data.max())
            }

    return stats


def calculate_ecdf(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Empirical Cumulative Distribution Function (ECDF).

    Args:
        data: Array of values

    Returns:
        Tuple of (sorted x values, cumulative probabilities)
    """
    if len(data) < 2:
        return np.array([]), np.array([])

    # Sort data
    x_sorted = np.sort(data)

    # Calculate cumulative probabilities (from 0 to 1)
    y_ecdf = np.arange(1, len(x_sorted) + 1) / len(x_sorted)

    return x_sorted, y_ecdf
