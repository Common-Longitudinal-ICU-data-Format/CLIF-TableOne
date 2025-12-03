#!/usr/bin/env python3
"""
Pre-compute ECDF and Bins for ICU Lab/Vital Data

This script:
1. Extracts ICU time windows from ADT table
2. Filters labs/vitals to values during ICU stays only (temporal filtering)
3. For LABS: Discovers all (category, unit) combinations and matches against config
4. Computes ECDF (distinct value/probability pairs) for each category
5. Computes quantile bins with auto-extreme-splitting for each category
6. Saves results as parquet files
7. Logs unit mismatches to file

Auto-extreme-splitting:
- If bins_below > 1: Split FIRST bin (most extreme low) into 5 sub-bins
- If bins_above > 1: Split LAST bin (most extreme high) into 5 sub-bins

Usage:
    python get_ecdf/precompute_ecdf_bins.py

Output structure:
    output/final/
    ├── configs/
    │   ├── clif_config.json
    │   ├── lab_vital_config.yaml
    │   └── outlier_config.yaml
    ├── ecdf/
    │   ├── labs/{category}_{unit}.parquet
    │   ├── vitals/{category}.parquet
    │   └── respiratory_support/{column}.parquet
    ├── bins/
    │   ├── labs/{category}_{unit}.parquet
    │   ├── vitals/{category}.parquet
    │   └── respiratory_support/{column}.parquet
    └── unit_mismatches.log
"""

import json
import yaml
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple
import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime

# Import binning functions from the same module
from .utils import create_all_bins


# ============================================================================
# Configuration Loading
# ============================================================================

def load_configs(
    clif_config_path: str = None,
    outlier_config_path: str = None,
    lab_vital_config_path: str = None
) -> Tuple[Dict, Dict, Dict]:
    """
    Load all required configuration files.

    Returns:
        Tuple of (clif_config, outlier_config, lab_vital_config)
    """
    # Get project root and set default paths
    project_root = Path(__file__).parent.parent.parent

    if clif_config_path is None:
        clif_config_path = project_root / 'config' / 'config.json'
    if outlier_config_path is None:
        outlier_config_path = Path(__file__).parent / 'config' / 'outlier_config.yaml'
    if lab_vital_config_path is None:
        lab_vital_config_path = Path(__file__).parent / 'config' / 'lab_vital_config.yaml'

    # Load clif_config.json
    if not os.path.exists(clif_config_path):
        raise FileNotFoundError(f"CLIF config not found: {clif_config_path}")

    with open(clif_config_path, 'r') as f:
        clif_config = json.load(f)

    # Load outlier_config.yaml
    if not os.path.exists(outlier_config_path):
        raise FileNotFoundError(f"Outlier config not found: {outlier_config_path}")

    with open(outlier_config_path, 'r') as f:
        outlier_config = yaml.safe_load(f)

    # Load lab_vital_config.yaml
    if not os.path.exists(lab_vital_config_path):
        raise FileNotFoundError(f"Lab/vital config not found: {lab_vital_config_path}")

    with open(lab_vital_config_path, 'r') as f:
        lab_vital_config = yaml.safe_load(f)

    return clif_config, outlier_config, lab_vital_config


def copy_configs_to_output(
    output_dir: str,
    clif_config_path: str = None,
    outlier_config_path: str = None,
    lab_vital_config_path: str = None
):
    """Copy configuration files to output directory."""
    # Get project root and set default paths
    project_root = Path(__file__).parent.parent.parent

    if clif_config_path is None:
        clif_config_path = project_root / 'config' / 'config.json'
    if outlier_config_path is None:
        outlier_config_path = Path(__file__).parent / 'config' / 'outlier_config.yaml'
    if lab_vital_config_path is None:
        lab_vital_config_path = Path(__file__).parent / 'config' / 'lab_vital_config.yaml'

    config_dir = os.path.join(output_dir, 'configs')
    os.makedirs(config_dir, exist_ok=True)

    shutil.copy(str(clif_config_path), os.path.join(config_dir, 'config.json'))
    shutil.copy(str(outlier_config_path), os.path.join(config_dir, 'outlier_config.yaml'))
    shutil.copy(str(lab_vital_config_path), os.path.join(config_dir, 'lab_vital_config.yaml'))

    print(f"✓ Copied configs to {config_dir}")


# ============================================================================
# Lab Category-Unit Discovery
# ============================================================================

def discover_lab_category_units(
    tables_path: str,
    file_type: str
) -> pl.DataFrame:
    """
    Discover all unique (lab_category, reference_unit) combinations
    in the labs data.

    Returns:
        Polars DataFrame with columns:
        - lab_category: str
        - reference_unit: str
    """
    labs_path = os.path.join(tables_path, f'clif_labs.{file_type}')

    if not os.path.exists(labs_path):
        raise FileNotFoundError(f"Labs file not found: {labs_path}")

    print(f"Discovering lab category-unit combinations from {labs_path}...")

    # Scan labs with Polars (lazy evaluation)
    labs_lazy = pl.scan_parquet(labs_path)

    # Get distinct (lab_category, reference_unit) pairs
    category_units = labs_lazy.select([
        'lab_category',
        'reference_unit'
    ]).unique()

    # Collect with streaming
    category_units_df = category_units.collect(streaming=True)

    # Filter out null lab_category only - keep null reference_unit (will be matched with "(no units)")
    category_units_df = category_units_df.filter(
        pl.col('lab_category').is_not_null()
    )

    # Replace null reference_unit with "(no units)" sentinel value
    category_units_df = category_units_df.with_columns(
        pl.col('reference_unit').fill_null('(no units)')
    )

    print(f"✓ Found {len(category_units_df):,} unique (category, unit) combinations")

    return category_units_df


def sanitize_unit_for_filename(unit: str) -> str:
    """
    Sanitize unit string for use in filename.

    Examples:
        mmol/L -> mmol_L
        mg/dL -> mg_dL
        % -> pct
        °C -> degC
    """
    if unit is None:
        return "unknown"

    # Replace common characters
    sanitized = unit.replace('/', '_').replace('%', 'pct').replace('°', 'deg')
    # Remove other special characters
    sanitized = ''.join(c if c.isalnum() or c == '_' else '_' for c in sanitized)
    # Remove consecutive underscores
    while '__' in sanitized:
        sanitized = sanitized.replace('__', '_')
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')

    return sanitized


# ============================================================================
# ICU Time Window Extraction
# ============================================================================

def extract_icu_time_windows(
    tables_path: str,
    file_type: str
) -> pl.DataFrame:
    """
    Extract ICU time windows from ADT table.

    Returns:
        Polars DataFrame with columns:
        - hospitalization_id: str
        - in_dttm: datetime
        - out_dttm: datetime
    """
    adt_path = os.path.join(tables_path, f'clif_adt.{file_type}')

    if not os.path.exists(adt_path):
        raise FileNotFoundError(f"ADT file not found: {adt_path}")

    print(f"Loading ICU time windows from {adt_path}...")

    # Scan ADT with Polars (lazy evaluation)
    adt_lazy = pl.scan_parquet(adt_path)

    # Filter to ICU locations only, select relevant columns
    icu_windows = adt_lazy.filter(
        pl.col('location_category').str.to_lowercase() == 'icu'
    ).select([
        'hospitalization_id',
        'in_dttm',
        'out_dttm'
    ])

    # Collect with streaming
    icu_windows_df = icu_windows.collect(streaming=True)

    # Strip timezone from datetime columns to enable comparison with other datetime columns
    icu_windows_df = icu_windows_df.with_columns([
        pl.col('in_dttm').dt.replace_time_zone(None),
        pl.col('out_dttm').dt.replace_time_zone(None)
    ])

    print(f"✓ Found {len(icu_windows_df):,} ICU time windows")

    return icu_windows_df


# ============================================================================
# ECDF Computation
# ============================================================================

def compute_ecdf_compact(values: np.ndarray) -> pl.DataFrame:
    """
    Compute ECDF and return distinct (value, probability) pairs.

    Args:
        values: Array of numeric values

    Returns:
        Polars DataFrame with columns:
        - value: float
        - probability: float (0 to 1)
    """
    if len(values) == 0:
        return pl.DataFrame({
            'value': [],
            'probability': []
        })

    # Sort values
    sorted_values = np.sort(values)

    # Calculate cumulative probabilities
    n = len(sorted_values)
    probabilities = np.arange(1, n + 1) / n

    # Create dataframe
    ecdf_df = pl.DataFrame({
        'value': sorted_values,
        'probability': probabilities
    })

    # Group by value and keep max probability (prevents zigzag effect)
    # For duplicate values, we only keep the highest probability (final occurrence)
    ecdf_df = ecdf_df.group_by('value').agg(
        pl.col('probability').max()
    ).sort('value')

    return ecdf_df


# ============================================================================
# Binning Functions
# ============================================================================

def create_flat_bins(data: pd.Series, num_bins: int = 10) -> List[Dict[str, Any]]:
    """
    Create flat quantile bins (no segmentation).

    Used for respiratory_support where there's no normal range concept.

    Args:
        data: Pandas Series with values
        num_bins: Number of bins to create (default: 10)

    Returns:
        List of bin dictionaries with:
        - bin_num, bin_min, bin_max, count, percentage, interval
    """
    if len(data) == 0:
        return []

    # If too few values for requested bins, reduce bin count
    if len(data) < num_bins * 2:
        num_bins = max(1, len(data) // 2)

    try:
        # Create quantile bins
        bins_cut, bin_edges = pd.qcut(data, q=num_bins, retbins=True, duplicates='drop')
        bin_counts = bins_cut.value_counts().sort_index()

        bins = []
        for i, (interval, count) in enumerate(bin_counts.items(), 1):
            # First bin is closed on both sides, others are left-open, right-closed
            if i == 1:
                interval_str = f"[{interval.left:.2f}, {interval.right:.2f}]"
            else:
                interval_str = f"({interval.left:.2f}, {interval.right:.2f}]"

            bins.append({
                'bin_num': i,
                'bin_min': float(interval.left),
                'bin_max': float(interval.right),
                'count': int(count),
                'percentage': float(count / len(data) * 100),
                'interval': interval_str
            })

        return bins

    except Exception as e:
        # Fallback: single bin if qcut fails
        return [{
            'bin_num': 1,
            'bin_min': float(data.min()),
            'bin_max': float(data.max()),
            'count': len(data),
            'percentage': 100.0,
            'interval': f"[{data.min():.2f}, {data.max():.2f}]"
        }]


# ============================================================================
# Lab/Vital Data Processing
# ============================================================================

def process_category(
    table_type: str,
    category: str,
    unit: str,
    icu_windows: pl.DataFrame,
    tables_path: str,
    file_type: str,
    outlier_range: Dict[str, float],
    cat_config: Dict[str, Any],
    output_dir: str,
    extreme_bins_count: int = 5
) -> Dict[str, Any]:
    """
    Process a single lab/vital category:
    1. Load data filtered to ICU time windows
    2. Remove outliers
    3. Compute ECDF
    4. Compute bins with auto-extreme-splitting
    5. Save results

    Args:
        unit: Reference unit (for labs only, pass None for vitals)
        extreme_bins_count: Number of bins to split extreme bins into (default: 5)

    Returns:
        Dictionary with processing statistics
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

    # Display name includes unit for labs
    if table_type == 'labs' and unit:
        display_name = f"{category} ({unit})"
    else:
        display_name = category

    print(f"  Loading {display_name}...")

    # Scan data with Polars (lazy)
    data_lazy = pl.scan_parquet(file_path)

    # Filter to selected category (and unit for labs)
    if table_type == 'labs' and unit:
        if isinstance(unit, list):
            unit_filter = pl.col('reference_unit').is_in(unit)
            if '(no units)' in unit:
                unit_filter = unit_filter | pl.col('reference_unit').is_null()
        else:
            unit_filter = pl.col('reference_unit') == unit
            if '(no units)' in unit:
                unit_filter = unit_filter | pl.col('reference_unit').is_null()
        data_category = data_lazy.filter(
            (pl.col(category_col) == category) &
            unit_filter
        ).select([
            'hospitalization_id',
            datetime_col,
            value_col
        ])
    else:
        data_category = data_lazy.filter(
            pl.col(category_col) == category
        ).select([
            'hospitalization_id',
            datetime_col,
            value_col
        ])
    # if table_type == 'labs' and unit:
    #     data_category = data_lazy.filter(
    #         (pl.col(category_col) == category) &
    #         (pl.col('reference_unit') == unit)
    #     ).select([
    #         'hospitalization_id',
    #         datetime_col,
    #         value_col
    #     ])
    # else:
    #     data_category = data_lazy.filter(
    #         pl.col(category_col) == category
    #     ).select([
    #         'hospitalization_id',
    #         datetime_col,
    #         value_col
    #     ])

    # Join with ICU time windows
    data_icu = data_category.join(
        icu_windows.lazy(),
        on='hospitalization_id',
        how='inner'
    )

    # Filter to values during ICU stay only (temporal filtering)
    # Strip timezone from datetime column for comparison with timezone-naive ICU windows
    data_filtered = data_icu.filter(
        (pl.col(datetime_col).dt.replace_time_zone(None) >= pl.col('in_dttm')) &
        (pl.col(datetime_col).dt.replace_time_zone(None) <= pl.col('out_dttm'))
    ).select([value_col])

    # Collect with streaming
    values_df = data_filtered.collect(streaming=True)

    original_count = len(values_df)

    if original_count == 0:
        print(f"  ⚠️  No data found for {display_name} during ICU stays")
        return {
            'category': category,
            'unit': unit if table_type == 'labs' else None,
            'original_count': 0,
            'clean_count': 0,
            'ecdf_distinct_pairs': 0,
            'num_bins': 0
        }

    # Remove outliers
    values_clean = values_df.filter(
        (pl.col(value_col) >= outlier_range['min']) &
        (pl.col(value_col) <= outlier_range['max'])
    )

    clean_count = len(values_clean)

    if clean_count == 0:
        print(f"  ⚠️  No data remaining after outlier removal for {display_name}")
        return {
            'category': category,
            'unit': unit if table_type == 'labs' else None,
            'original_count': original_count,
            'clean_count': 0,
            'ecdf_distinct_pairs': 0,
            'num_bins': 0
        }

    # Extract values as numpy array
    values_array = values_clean[value_col].to_numpy()

    # ========================================================================
    # Compute ECDF
    # ========================================================================

    ecdf_df = compute_ecdf_compact(values_array)

    # Save ECDF
    ecdf_dir = os.path.join(output_dir, 'ecdf', table_type)
    os.makedirs(ecdf_dir, exist_ok=True)

    if table_type == 'labs' and unit:
        # Handle list of units - use first non-empty unit for filename
        if isinstance(unit, list):
            # Filter out "(no units)" and empty strings for filename, use first real unit
            real_units = [u for u in unit if u and u != '(no units)']
            unit_for_filename = real_units[0] if real_units else 'no_units'
        else:
            unit_for_filename = unit if unit != '(no units)' else 'no_units'
        unit_safe = sanitize_unit_for_filename(unit_for_filename)
        filename = f'{category}_{unit_safe}.parquet'
    else:
        filename = f'{category}.parquet'

    ecdf_path = os.path.join(ecdf_dir, filename)
    ecdf_df.write_parquet(ecdf_path)

    # ========================================================================
    # Compute Bins with Auto-Extreme-Splitting
    # ========================================================================

    bins_below = cat_config['bins']['below_normal']
    bins_normal = cat_config['bins']['normal']
    bins_above = cat_config['bins']['above_normal']

    # AUTO-SPLIT EXTREMES: Use extreme_bins_count parameter if segment has >1 bin
    extra_bins_below = extreme_bins_count if bins_below > 1 else 0
    extra_bins_above = extreme_bins_count if bins_above > 1 else 0

    # Create bins (using function from get_ecdf/utils.py)
    bins = create_all_bins(
        data=pd.Series(values_array),
        normal_lower=cat_config['normal_range']['lower'],
        normal_upper=cat_config['normal_range']['upper'],
        outlier_min=outlier_range['min'],
        outlier_max=outlier_range['max'],
        bins_below=bins_below,
        bins_normal=bins_normal,
        bins_above=bins_above,
        extra_bins_below=extra_bins_below,
        extra_bins_above=extra_bins_above
    )

    # Add interval notation column
    # First bin in each segment: [min, max] (closed both sides)
    # Other bins: (min, max] (left-open, right-closed)
    for bin_info in bins:
        if bin_info['bin_num'] == 1:
            # First bin is closed on both sides
            interval = f"[{bin_info['bin_min']:.2f}, {bin_info['bin_max']:.2f}]"
        else:
            # Other bins are left-open, right-closed
            interval = f"({bin_info['bin_min']:.2f}, {bin_info['bin_max']:.2f}]"
        bin_info['interval'] = interval

    # Convert to Polars DataFrame
    bins_df = pl.DataFrame(bins)

    # Save bins
    bins_dir = os.path.join(output_dir, 'bins', table_type)
    os.makedirs(bins_dir, exist_ok=True)
    bins_path = os.path.join(bins_dir, filename)
    bins_df.write_parquet(bins_path)

    # Return statistics
    return {
        'category': category,
        'unit': unit if table_type == 'labs' else None,
        'original_count': original_count,
        'clean_count': clean_count,
        'ecdf_distinct_pairs': len(ecdf_df),
        'num_bins': len(bins)
    }


# ============================================================================
# Respiratory Support Data Processing
# ============================================================================

def process_respiratory_column(
    column_name: str,
    icu_windows: pl.DataFrame,
    tables_path: str,
    file_type: str,
    outlier_range: Dict[str, float],
    output_dir: str,
    num_bins: int = 10
) -> Dict[str, Any]:
    """
    Process a single respiratory support column:
    1. Load respiratory_support data filtered to ICU time windows
    2. Remove outliers
    3. Compute ECDF
    4. Compute flat 10 bins (no segmentation)
    5. Save results

    Args:
        column_name: Column name (e.g., 'fio2_set', 'peep_obs')
        icu_windows: ICU time windows
        tables_path: Path to data
        file_type: File type
        outlier_range: Dict with 'min' and 'max'
        output_dir: Output directory
        num_bins: Number of bins (default: 10)

    Returns:
        Dictionary with processing statistics
    """
    file_path = os.path.join(tables_path, f'clif_respiratory_support.{file_type}')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Respiratory support file not found: {file_path}")

    print(f"  Loading {column_name}...")

    # Scan data with Polars (lazy)
    data_lazy = pl.scan_parquet(file_path)

    # Select only needed columns
    data_selected = data_lazy.select([
        'hospitalization_id',
        'recorded_dttm',
        column_name
    ])

    # Join with ICU time windows
    data_icu = data_selected.join(
        icu_windows.lazy(),
        on='hospitalization_id',
        how='inner'
    )

    # Filter to values during ICU stay only (temporal filtering)
    # Strip timezone from datetime column for comparison with timezone-naive ICU windows
    data_filtered = data_icu.filter(
        (pl.col('recorded_dttm').dt.replace_time_zone(None) >= pl.col('in_dttm')) &
        (pl.col('recorded_dttm').dt.replace_time_zone(None) <= pl.col('out_dttm'))
    ).select([column_name])

    # Collect with streaming
    values_df = data_filtered.collect(streaming=True)

    original_count = len(values_df)

    if original_count == 0:
        print(f"  ⚠️  No data found for {column_name} during ICU stays")
        return {
            'column': column_name,
            'original_count': 0,
            'clean_count': 0,
            'ecdf_distinct_pairs': 0,
            'num_bins': 0
        }

    # Remove outliers
    values_clean = values_df.filter(
        (pl.col(column_name) >= outlier_range['min']) &
        (pl.col(column_name) <= outlier_range['max'])
    )

    clean_count = len(values_clean)

    if clean_count == 0:
        print(f"  ⚠️  No data remaining after outlier removal for {column_name}")
        return {
            'column': column_name,
            'original_count': original_count,
            'clean_count': 0,
            'ecdf_distinct_pairs': 0,
            'num_bins': 0
        }

    # Extract values as numpy array
    values_array = values_clean[column_name].to_numpy()

    # ========================================================================
    # Compute ECDF
    # ========================================================================

    ecdf_df = compute_ecdf_compact(values_array)

    # Save ECDF
    ecdf_dir = os.path.join(output_dir, 'ecdf', 'respiratory_support')
    os.makedirs(ecdf_dir, exist_ok=True)
    ecdf_path = os.path.join(ecdf_dir, f'{column_name}.parquet')
    ecdf_df.write_parquet(ecdf_path)

    # ========================================================================
    # Compute Flat Bins (15 bins, no segmentation)
    # ========================================================================

    bins = create_flat_bins(pd.Series(values_array), num_bins=num_bins)

    # Convert to Polars DataFrame
    bins_df = pl.DataFrame(bins)

    # Save bins
    bins_dir = os.path.join(output_dir, 'bins', 'respiratory_support')
    os.makedirs(bins_dir, exist_ok=True)
    bins_path = os.path.join(bins_dir, f'{column_name}.parquet')
    bins_df.write_parquet(bins_path)

    # Return statistics
    return {
        'column': column_name,
        'original_count': original_count,
        'clean_count': clean_count,
        'ecdf_distinct_pairs': len(ecdf_df),
        'num_bins': len(bins)
    }


# ============================================================================
# Main Processing
# ============================================================================

def main():
    """Main processing pipeline."""

    print("\n" + "="*80)
    print("Pre-compute ECDF and Bins for ICU Lab/Vital Data")
    print("="*80 + "\n")

    start_time = datetime.now()

    # ========================================================================
    # Load Configurations
    # ========================================================================

    print("Loading configurations...")
    clif_config, outlier_config, lab_vital_config = load_configs()
    print("✓ Configs loaded\n")

    # ========================================================================
    # Setup Output Directory
    # ========================================================================

    output_dir = 'output/final'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Copy configs
    copy_configs_to_output(output_dir)
    print()

    # Setup log file
    log_file = os.path.join(output_dir, 'unit_mismatches.log')
    log_entries = []

    # ========================================================================
    # Extract ICU Time Windows
    # ========================================================================

    icu_windows = extract_icu_time_windows(
        clif_config['tables_path'],
        clif_config['file_type']
    )
    print()

    # ========================================================================
    # Discover Lab Category-Unit Combinations
    # ========================================================================

    print("="*80)
    print("Discovering Lab Category-Unit Combinations")
    print("="*80)

    lab_category_units = discover_lab_category_units(
        clif_config['tables_path'],
        clif_config['file_type']
    )
    print()

    # ========================================================================
    # Process Labs
    # ========================================================================

    print("="*80)
    print("Processing Labs")
    print("="*80)

    labs_config = lab_vital_config.get('labs', {})
    labs_outlier = outlier_config['tables']['labs']['lab_value_numeric']

    labs_stats = []
    processed_categories = set()  # Track processed categories to avoid duplicates

    # Iterate through discovered category-unit combinations
    for row in lab_category_units.iter_rows(named=True):
        category = row['lab_category']
        unit = row['reference_unit']

        # Skip if we already processed this category
        if category in processed_categories:
            continue

        # Check if category exists in config
        if category not in labs_config:
            log_entries.append(f"[SKIP] {category} ({unit}): Category not in lab_vital_config.yaml")
            continue

        # Check if category has outlier config
        if category not in labs_outlier:
            log_entries.append(f"[SKIP] {category} ({unit}): Category not in outlier_config.yaml")
            continue

        # Check if unit matches config's reference_unit (config stores as list)
        config_units = labs_config[category].get('reference_unit', [])
        # Handle both list and string config formats
        if isinstance(config_units, str):
            config_units = [config_units]
        if unit not in config_units:
            log_entries.append(
                f"[UNIT MISMATCH] {category}: Found unit '{unit}' in data, "
                f"but config expects one of {config_units}"
            )
            continue

        # Mark as processed before attempting
        processed_categories.add(category)

        # Process this category-unit combination
        # Pass the full config_units list so process_category can match all valid units
        # (including NULL handling when "(no units)" is in the list)
        try:
            stats = process_category(
                table_type='labs',
                category=category,
                unit=config_units,
                icu_windows=icu_windows,
                tables_path=clif_config['tables_path'],
                file_type=clif_config['file_type'],
                outlier_range=labs_outlier[category],
                cat_config=labs_config[category],
                output_dir=output_dir,
                extreme_bins_count=5  # Labs use 5 extreme bins
            )
            labs_stats.append(stats)

            print(f"  ✓ {category} ({config_units}): {stats['clean_count']:,} values → "
                  f"ECDF: {stats['ecdf_distinct_pairs']:,} pairs, "
                  f"Bins: {stats['num_bins']}")

        except Exception as e:
            print(f"  ❌ Error processing {category} ({config_units}): {e}")
            log_entries.append(f"[ERROR] {category} ({config_units}): {str(e)}")

    print()

    # ========================================================================
    # Process Vitals
    # ========================================================================

    print("="*80)
    print("Processing Vitals")
    print("="*80)

    vitals_config = lab_vital_config.get('vitals', {})
    vitals_outlier = outlier_config['tables']['vitals']['vital_value']

    vitals_stats = []

    for category in sorted(vitals_config.keys()):
        if category not in vitals_outlier:
            print(f"  ⚠️  Skipping {category} (no outlier config)")
            log_entries.append(f"[SKIP] {category}: Category not in outlier_config.yaml")
            continue

        # Determine extreme bins count: 5 for height/weight, 10 for others
        extreme_bins = 5 if category in ['height_cm', 'weight_kg'] else 10

        try:
            stats = process_category(
                table_type='vitals',
                category=category,
                unit=None,  # Vitals don't use unit differentiation
                icu_windows=icu_windows,
                tables_path=clif_config['tables_path'],
                file_type=clif_config['file_type'],
                outlier_range=vitals_outlier[category],
                cat_config=vitals_config[category],
                output_dir=output_dir,
                extreme_bins_count=extreme_bins  # 10 for most vitals, 5 for height/weight
            )
            vitals_stats.append(stats)

            print(f"  ✓ {category}: {stats['clean_count']:,} values → "
                  f"ECDF: {stats['ecdf_distinct_pairs']:,} pairs, "
                  f"Bins: {stats['num_bins']}")

        except Exception as e:
            print(f"  ❌ Error processing {category}: {e}")
            log_entries.append(f"[ERROR] {category}: {str(e)}")

    print()

    # ========================================================================
    # Process Respiratory Support
    # ========================================================================

    print("="*80)
    print("Processing Respiratory Support (17 columns)")
    print("="*80)

    resp_outlier = outlier_config['tables'].get('respiratory_support', {})

    # List of all 17 numerical columns in respiratory_support
    resp_columns = [
        'fio2_set', 'lpm_set', 'tidal_volume_set', 'resp_rate_set',
        'pressure_control_set', 'pressure_support_set', 'flow_rate_set',
        'peak_inspiratory_pressure_set', 'inspiratory_time_set', 'peep_set',
        'tidal_volume_obs', 'resp_rate_obs', 'plateau_pressure_obs',
        'peak_inspiratory_pressure_obs', 'peep_obs', 'minute_vent_obs',
        'mean_airway_pressure_obs'
    ]

    resp_stats = []

    for column in resp_columns:
        if column not in resp_outlier:
            print(f"  ⚠️  Skipping {column} (no outlier config)")
            log_entries.append(f"[SKIP] respiratory_support.{column}: Not in outlier_config.yaml")
            continue

        try:
            stats = process_respiratory_column(
                column_name=column,
                icu_windows=icu_windows,
                tables_path=clif_config['tables_path'],
                file_type=clif_config['file_type'],
                outlier_range=resp_outlier[column],
                output_dir=output_dir,
                num_bins=10
            )
            resp_stats.append(stats)

            print(f"  ✓ {column}: {stats['clean_count']:,} values → "
                  f"ECDF: {stats['ecdf_distinct_pairs']:,} pairs, "
                  f"Bins: {stats['num_bins']}")

        except Exception as e:
            print(f"  ❌ Error processing {column}: {e}")
            log_entries.append(f"[ERROR] respiratory_support.{column}: {str(e)}")

    print()

    # ========================================================================
    # Write Log File
    # ========================================================================

    if log_entries:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("Unit Mismatches and Processing Log\n")
            f.write("="*80 + "\n\n")
            for entry in log_entries:
                f.write(entry + "\n")
        print(f"✓ Log file written: {log_file} ({len(log_entries)} entries)")
        print()

    # ========================================================================
    # Summary Report
    # ========================================================================

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print("="*80)
    print("Processing Summary")
    print("="*80)
    print()

    print(f"Labs: {len(labs_stats)} category-unit combinations processed")
    for stat in labs_stats:
        if stat['clean_count'] > 0:
            compression_ratio = stat['clean_count'] / stat['ecdf_distinct_pairs'] if stat['ecdf_distinct_pairs'] > 0 else 0
            unit_display = f" ({stat['unit']})" if stat['unit'] else ""
            print(f"  - {stat['category']:20s}{unit_display:15s}: "
                  f"{stat['clean_count']:>10,} values → "
                  f"{stat['ecdf_distinct_pairs']:>8,} ECDF pairs "
                  f"({compression_ratio:.1f}x compression), "
                  f"{stat['num_bins']:>2} bins")

    print()
    print(f"Vitals: {len(vitals_stats)} categories processed")
    for stat in vitals_stats:
        if stat['clean_count'] > 0:
            compression_ratio = stat['clean_count'] / stat['ecdf_distinct_pairs'] if stat['ecdf_distinct_pairs'] > 0 else 0
            print(f"  - {stat['category']:20s}: "
                  f"{stat['clean_count']:>10,} values → "
                  f"{stat['ecdf_distinct_pairs']:>8,} ECDF pairs "
                  f"({compression_ratio:.1f}x compression), "
                  f"{stat['num_bins']:>2} bins")

    print()
    print(f"Total processing time: {duration:.1f} seconds")
    print()
    print(f"Output saved to: {output_dir}/")
    print("  - configs/")
    print("  - ecdf/labs/*.parquet (with unit in filename)")
    print("  - ecdf/vitals/*.parquet")
    print("  - bins/labs/*.parquet (with unit in filename)")
    print("  - bins/vitals/*.parquet")
    if log_entries:
        print(f"  - unit_mismatches.log ({len(log_entries)} entries)")
    print()
    print("="*80)
    print("✅ Done!")
    print("="*80)


if __name__ == '__main__':
    main()