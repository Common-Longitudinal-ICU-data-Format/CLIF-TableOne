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
    ├── overall/
    │   ├── ecdf/
    │   │   ├── labs/{category}_{unit}.parquet
    │   │   ├── vitals/{category}.parquet
    │   │   └── respiratory_support/{column}.parquet
    │   └── bins/
    │       ├── labs/{category}_{unit}.parquet
    │       ├── vitals/{category}.parquet
    │       └── respiratory_support/{column}.parquet
    ├── strata/
    │   ├── icu/{ecdf,bins}/...
    │   ├── advanced_resp/{ecdf,bins}/...
    │   ├── vaso/{ecdf,bins}/...
    │   └── deaths/{ecdf,bins}/...
    └── meta/
        ├── configs/
        │   ├── clif_config.json
        │   ├── lab_vital_config.yaml
        │   └── outlier_config.yaml
        └── unit_mismatches.log
"""

import json
import yaml
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

# Import binning functions from the same module
from .utils import create_all_bins
from modules.utils.output_paths import (
    CONFIGS,
    OVERALL,
    STRATA,
    META,
    cohort_dir,
    meta_dir,
    ensure_output_tree,
)
from modules.utils.clif_loader import ClifDB


# Canonical qualifying-event definitions — must match modules/tableone/generator.py
VASOACTIVE_MEDS = [
    'norepinephrine', 'epinephrine', 'phenylephrine',
    'vasopressin', 'dopamine', 'angiotensin',
]
ADVANCED_RESP_DEVICES = ['imv', 'nippv', 'cpap', 'high flow nc']
NIPPV_HFNC_DEVICES = ['nippv', 'high flow nc']

# Maps each stratum to its temporal window source.
# 'icu' = ICU ADT windows, 'vaso' = first vasopressor → discharge,
# 'resp' = first qualifying device → discharge,
# 'nippv_hfnc' = first NIPPV/HFNC device → discharge.
STRATUM_WINDOW_TYPE = {
    'icu': 'icu',
    'deaths': 'icu',
    'vaso': 'vaso',
    'vaso/icu': 'vaso',
    'vaso/no_icu': 'vaso',
    'vaso/ed_icu': 'vaso',
    'vaso/ed_ward': 'vaso',
    'advanced_resp': 'resp',
    'advanced_resp/icu': 'resp',
    'advanced_resp/no_icu': 'resp',
    'nippv_hfnc': 'nippv_hfnc',
    'nippv_hfnc/icu': 'nippv_hfnc',
    'nippv_hfnc/no_icu': 'nippv_hfnc',
    'no_imv': 'icu',
    'no_imv/icu': 'icu',
    'no_imv/no_icu': 'icu',
}


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

    with open(clif_config_path, 'r', encoding='utf-8') as f:
        clif_config = json.load(f)

    # Load outlier_config.yaml
    if not os.path.exists(outlier_config_path):
        raise FileNotFoundError(f"Outlier config not found: {outlier_config_path}")

    with open(outlier_config_path, 'r', encoding='utf-8') as f:
        outlier_config = yaml.safe_load(f)

    # Load lab_vital_config.yaml
    if not os.path.exists(lab_vital_config_path):
        raise FileNotFoundError(f"Lab/vital config not found: {lab_vital_config_path}")

    with open(lab_vital_config_path, 'r', encoding='utf-8') as f:
        lab_vital_config = yaml.safe_load(f)

    return clif_config, outlier_config, lab_vital_config


def copy_configs_to_output(
    output_dir: str = None,
    clif_config_path: str = None,
    outlier_config_path: str = None,
    lab_vital_config_path: str = None
):
    """Copy configuration files to output/final/meta/configs/.

    The ``output_dir`` argument is accepted for backwards compatibility but is
    ignored — configs are always written to the canonical CONFIGS directory
    so they can be discovered without knowing the cohort layout.
    """
    # Get project root and set default paths
    project_root = Path(__file__).parent.parent.parent

    if clif_config_path is None:
        clif_config_path = project_root / 'config' / 'config.json'
    if outlier_config_path is None:
        outlier_config_path = Path(__file__).parent / 'config' / 'outlier_config.yaml'
    if lab_vital_config_path is None:
        lab_vital_config_path = Path(__file__).parent / 'config' / 'lab_vital_config.yaml'

    config_dir = CONFIGS
    config_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(str(clif_config_path), str(config_dir / 'config.json'))
    shutil.copy(str(outlier_config_path), str(config_dir / 'outlier_config.yaml'))
    shutil.copy(str(lab_vital_config_path), str(config_dir / 'lab_vital_config.yaml'))

    print(f"✓ Copied configs to {config_dir}")


# ============================================================================
# Lab Category-Unit Discovery
# ============================================================================

def discover_lab_category_units(
    db: ClifDB,
) -> pd.DataFrame:
    """
    Discover all unique (lab_category, reference_unit) combinations
    in the labs data.

    Returns:
        pandas DataFrame with columns:
        - lab_category: str
        - reference_unit: str
    """
    labs_path = db.table_path('labs')

    print(f"Discovering lab category-unit combinations from {labs_path}...")

    df = db.query_df(
        """
        SELECT DISTINCT
            LOWER(TRIM(lab_category)) AS lab_category,
            COALESCE(
                NULLIF(LOWER(TRIM(reference_unit)), ''),
                '(no units)'
            ) AS reference_unit
        FROM read_parquet(?)
        WHERE lab_category IS NOT NULL
        """,
        [labs_path],
    )

    print(f"✓ Found {len(df):,} unique (category, unit) combinations")
    return df


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

def extract_icu_time_windows(db: ClifDB) -> pd.DataFrame:
    """
    Extract ICU time windows from ADT table.

    Returns:
        pandas DataFrame with columns:
        - hospitalization_id: str
        - in_dttm: datetime (timezone-naive)
        - out_dttm: datetime (timezone-naive)
    """
    adt_path = db.table_path('adt')

    print(f"Loading ICU time windows from {adt_path}...")

    df = db.query_df(
        """
        SELECT hospitalization_id,
               in_dttm::TIMESTAMP AS in_dttm,
               out_dttm::TIMESTAMP AS out_dttm
        FROM read_parquet(?)
        WHERE LOWER(location_category) = 'icu'
        """,
        [adt_path],
    )

    print(f"✓ Found {len(df):,} ICU time windows")
    return df


# ============================================================================
# Discharge Times & Event-Based Time Windows
# ============================================================================

def load_discharge_times(db: ClifDB) -> pd.DataFrame:
    """
    Load discharge_dttm from the hospitalization table.

    Returns:
        pandas DataFrame with columns:
        - hospitalization_id: str
        - discharge_dttm: datetime (timezone-naive)
    """
    hosp_path = db.table_path('hospitalization')

    print(f"Loading discharge times from {hosp_path}...")

    df = db.query_df(
        """
        SELECT hospitalization_id,
               discharge_dttm::TIMESTAMP AS discharge_dttm
        FROM read_parquet(?)
        WHERE discharge_dttm IS NOT NULL
        """,
        [hosp_path],
    )

    print(f"✓ Loaded discharge times for {len(df):,} hospitalizations")
    return df


def _extract_event_windows(
    db: ClifDB,
    file_path: str,
    category_col: str,
    datetime_col: str,
    qualifying_values: List[str],
    label: str,
) -> pd.DataFrame:
    """Shared implementation for vaso / resp / nippv-hfnc event windows.

    Finds the earliest qualifying event per hospitalization and bounds
    the window by discharge_dttm (which must already be registered as
    ``discharge_times`` on *db*).
    """
    placeholders = ', '.join(['?'] * len(qualifying_values))

    df = db.query_df(
        f"""
        SELECT e.hospitalization_id, e.in_dttm, d.discharge_dttm AS out_dttm
        FROM (
            SELECT hospitalization_id,
                   MIN({datetime_col}::TIMESTAMP) AS in_dttm
            FROM read_parquet(?)
            WHERE LOWER(TRIM({category_col})) IN ({placeholders})
            GROUP BY hospitalization_id
        ) e
        INNER JOIN discharge_times d USING (hospitalization_id)
        """,
        [file_path] + qualifying_values,
    )

    print(f"✓ Found {len(df):,} {label} event windows")
    return df


def extract_vaso_event_windows(db: ClifDB) -> pd.DataFrame:
    """Build time windows anchored to the first vasopressor administration."""
    meds_path = db.table_path('medication_admin_continuous')
    print(f"Extracting vasopressor event windows from {meds_path}...")
    return _extract_event_windows(
        db, meds_path, 'med_category', 'admin_dttm',
        VASOACTIVE_MEDS, 'vasopressor',
    )


def extract_advanced_resp_event_windows(db: ClifDB) -> pd.DataFrame:
    """Build time windows anchored to the first qualifying respiratory device."""
    resp_path = db.table_path('respiratory_support')
    print(f"Extracting advanced resp event windows from {resp_path}...")
    return _extract_event_windows(
        db, resp_path, 'device_category', 'recorded_dttm',
        ADVANCED_RESP_DEVICES, 'advanced resp',
    )


def extract_nippv_hfnc_event_windows(db: ClifDB) -> pd.DataFrame:
    """Build time windows anchored to the first NIPPV/HFNC device."""
    resp_path = db.table_path('respiratory_support')
    print(f"Extracting NIPPV/HFNC event windows from {resp_path}...")
    return _extract_event_windows(
        db, resp_path, 'device_category', 'recorded_dttm',
        NIPPV_HFNC_DEVICES, 'NIPPV/HFNC',
    )


# ============================================================================
# ECDF Computation
# ============================================================================

def compute_ecdf_compact(values: np.ndarray) -> pd.DataFrame:
    """
    Compute ECDF and return distinct (value, probability) pairs.

    Args:
        values: Array of numeric values

    Returns:
        pandas DataFrame with columns:
        - value: float
        - probability: float (0 to 1)
    """
    if len(values) == 0:
        return pd.DataFrame({'value': pd.array([], dtype='float64'),
                             'probability': pd.array([], dtype='float64')})

    sorted_values = np.sort(values)
    n = len(sorted_values)

    # For duplicate values, CDF = cumulative count / n
    unique_vals, counts = np.unique(sorted_values, return_counts=True)
    cumulative = np.cumsum(counts)
    probs = cumulative / n

    return pd.DataFrame({'value': unique_vals, 'probability': probs})


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
# SQL helpers
# ============================================================================

def _build_unit_filter(unit) -> Tuple[str, List]:
    """Build a SQL WHERE clause fragment for lab unit matching.

    Returns (sql_fragment, params) where *sql_fragment* is a string like
    ``"LOWER(TRIM(d.reference_unit)) = ?"`` and *params* are the bind values.
    """
    if unit is None:
        return "TRUE", []

    if isinstance(unit, list):
        no_units = [u for u in unit if u.lower().strip() == '(no units)']
        real_units = [u.lower().strip() for u in unit
                      if u.lower().strip() != '(no units)']

        parts = []
        params = []
        if real_units:
            placeholders = ', '.join(['?'] * len(real_units))
            parts.append(f"LOWER(TRIM(d.reference_unit)) IN ({placeholders})")
            params.extend(real_units)
        if no_units:
            parts.append(
                "(d.reference_unit IS NULL "
                "OR TRIM(d.reference_unit) = '' "
                "OR LOWER(TRIM(d.reference_unit)) = '(no units)')"
            )
        return '(' + ' OR '.join(parts) + ')', params

    unit_l = unit.lower().strip()
    if unit_l == '(no units)':
        return (
            "(d.reference_unit IS NULL "
            "OR TRIM(d.reference_unit) = '' "
            "OR LOWER(TRIM(d.reference_unit)) = '(no units)')"
        ), []

    return "LOWER(TRIM(d.reference_unit)) = ?", [unit_l]


# ============================================================================
# Lab / Vital Processing
# ============================================================================

def process_category(
    table_type: str,
    category: str,
    unit,
    db: ClifDB,
    outlier_range: Dict[str, float],
    cat_config: Dict,
    output_dir: str,
    extreme_bins_count: int = 5,
    suffix: str = ''
) -> Dict[str, Any]:
    """
    Process a single lab/vital category:
    1. Load data filtered to ICU time windows (via DuckDB join)
    2. Remove outliers
    3. Compute ECDF
    4. Compute quantile bins with auto-extreme-splitting
    5. Save results
    """
    if table_type == 'labs':
        file_path = db.table_path('labs')
        category_col = 'lab_category'
        value_col = 'lab_value_numeric'
        datetime_col = 'lab_result_dttm'
    else:  # vitals
        file_path = db.table_path('vitals')
        category_col = 'vital_category'
        value_col = 'vital_value'
        datetime_col = 'recorded_dttm'

    # Display name includes unit for labs
    if table_type == 'labs' and unit:
        display_name = f"{category} ({unit})"
    else:
        display_name = category

    print(f"  Loading {display_name}...")

    values = db.fetch_icu_values(
        parquet_path=file_path,
        value_col=value_col,
        datetime_col=datetime_col,
        category_col=category_col,
        category_value=category.lower().strip(),
        unit=unit if table_type == 'labs' else None,
    )

    original_count = len(values)

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

    # Remove outliers (numpy — fast, zero-copy)
    mask = (values >= outlier_range['min']) & (values <= outlier_range['max'])
    values_clean = values[mask]
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

    values_array = values_clean

    # ========================================================================
    # Compute ECDF
    # ========================================================================

    ecdf_df = compute_ecdf_compact(values_array)

    # Save ECDF
    ecdf_dir_path = os.path.join(output_dir, 'ecdf', table_type)
    os.makedirs(ecdf_dir_path, exist_ok=True)

    if table_type == 'labs' and unit:
        # Handle list of units - use first non-empty unit for filename
        if isinstance(unit, list):
            real_units = [u for u in unit if u and u != '(no units)']
            unit_for_filename = real_units[0] if real_units else 'no_units'
        else:
            unit_for_filename = unit
        safe_unit = sanitize_unit_for_filename(unit_for_filename)
        filename = f'{category}_{safe_unit}{suffix}.parquet'
    else:
        filename = f'{category}{suffix}.parquet'

    ecdf_path = os.path.join(ecdf_dir_path, filename)
    ecdf_df.to_parquet(ecdf_path, index=False)

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
    for bin_info in bins:
        if bin_info['bin_num'] == 1:
            interval = f"[{bin_info['bin_min']:.2f}, {bin_info['bin_max']:.2f}]"
        else:
            interval = f"({bin_info['bin_min']:.2f}, {bin_info['bin_max']:.2f}]"
        bin_info['interval'] = interval

    bins_df = pd.DataFrame(bins)

    # Save bins
    bins_dir_path = os.path.join(output_dir, 'bins', table_type)
    os.makedirs(bins_dir_path, exist_ok=True)
    bins_path = os.path.join(bins_dir_path, filename)
    bins_df.to_parquet(bins_path, index=False)

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
    db: ClifDB,
    outlier_range: Dict[str, float],
    output_dir: str,
    num_bins: int = 10,
    suffix: str = ''
) -> Dict[str, Any]:
    """
    Process a single respiratory support column:
    1. Load respiratory_support data filtered to ICU time windows
    2. Remove outliers
    3. Compute ECDF
    4. Compute flat 10 bins (no segmentation)
    5. Save results
    """
    file_path = db.table_path('respiratory_support')

    print(f"  Loading {column_name}...")

    values = db.fetch_icu_values(
        parquet_path=file_path,
        value_col=column_name,
        datetime_col='recorded_dttm',
    )
    original_count = len(values)

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
    mask = (values >= outlier_range['min']) & (values <= outlier_range['max'])
    values_clean = values[mask]
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

    values_array = values_clean

    # ========================================================================
    # Compute ECDF
    # ========================================================================

    ecdf_df = compute_ecdf_compact(values_array)

    # Save ECDF
    ecdf_dir_path = os.path.join(output_dir, 'ecdf', 'respiratory_support')
    os.makedirs(ecdf_dir_path, exist_ok=True)
    ecdf_path = os.path.join(ecdf_dir_path, f'{column_name}{suffix}.parquet')
    ecdf_df.to_parquet(ecdf_path, index=False)

    # ========================================================================
    # Compute Flat Bins (15 bins, no segmentation)
    # ========================================================================

    bins = create_flat_bins(pd.Series(values_array), num_bins=num_bins)

    bins_df = pd.DataFrame(bins)

    # Save bins
    bins_dir_path = os.path.join(output_dir, 'bins', 'respiratory_support')
    os.makedirs(bins_dir_path, exist_ok=True)
    bins_path = os.path.join(bins_dir_path, f'{column_name}{suffix}.parquet')
    bins_df.to_parquet(bins_path, index=False)

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

def run_ecdf_pipeline(
    icu_windows: pd.DataFrame,
    lab_category_units: pd.DataFrame,
    db: ClifDB,
    clif_config: Dict,
    outlier_config: Dict,
    lab_vital_config: Dict,
    output_dir: str,
    label: str = "overall",
    suffix: str = ''
) -> Tuple[List, List, List, List]:
    """
    Run the full ECDF/bins pipeline for a given set of ICU windows.

    This is the core processing loop extracted from main() so it can be
    called once for overall and once per encounter-type stratum.

    Returns:
        (labs_stats, vitals_stats, resp_stats, log_entries)
    """
    # Register the (possibly stratum-filtered) time windows for SQL joins
    db.register('icu_windows', icu_windows)

    log_entries = []

    # ========================================================================
    # Process Labs
    # ========================================================================

    print("="*80)
    print(f"Processing Labs [{label}]")
    print("="*80)

    labs_config = lab_vital_config.get('labs', {})
    labs_outlier = outlier_config['tables']['labs']['lab_value_numeric']

    labs_stats = []
    processed_categories = set()

    for row in lab_category_units.itertuples(index=False):
        category = row.lab_category
        unit = row.reference_unit

        if category in processed_categories:
            continue

        if category not in labs_config:
            log_entries.append(f"[SKIP] {category} ({unit}): Category not in lab_vital_config.yaml")
            continue

        if category not in labs_outlier:
            log_entries.append(f"[SKIP] {category} ({unit}): Category not in outlier_config.yaml")
            continue

        config_units = labs_config[category].get('reference_unit', [])
        if isinstance(config_units, str):
            config_units = [config_units]
        config_units_lower = [u.lower().strip() for u in config_units]
        if unit.lower().strip() not in config_units_lower:
            log_entries.append(
                f"[UNIT MISMATCH] {category}: Found unit '{unit}' in data, "
                f"but config expects one of {config_units}"
            )
            continue

        processed_categories.add(category)

        try:
            stats = process_category(
                table_type='labs',
                category=category,
                unit=config_units,
                db=db,
                outlier_range=labs_outlier[category],
                cat_config=labs_config[category],
                output_dir=output_dir,
                extreme_bins_count=5,
                suffix=suffix
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
    print(f"Processing Vitals [{label}]")
    print("="*80)

    vitals_config = lab_vital_config.get('vitals', {})
    vitals_outlier = outlier_config['tables']['vitals']['vital_value']

    vitals_stats = []

    for category in sorted(vitals_config.keys()):
        if category not in vitals_outlier:
            print(f"  ⚠️  Skipping {category} (no outlier config)")
            log_entries.append(f"[SKIP] {category}: Category not in outlier_config.yaml")
            continue

        extreme_bins = 5 if category in ['height_cm', 'weight_kg'] else 10

        try:
            stats = process_category(
                table_type='vitals',
                category=category,
                unit=None,
                db=db,
                outlier_range=vitals_outlier[category],
                cat_config=vitals_config[category],
                output_dir=output_dir,
                extreme_bins_count=extreme_bins,
                suffix=suffix
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
    print(f"Processing Respiratory Support (17 columns) [{label}]")
    print("="*80)

    resp_outlier = outlier_config['tables'].get('respiratory_support', {})

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
                db=db,
                outlier_range=resp_outlier[column],
                output_dir=output_dir,
                num_bins=10,
                suffix=suffix
            )
            resp_stats.append(stats)

            print(f"  ✓ {column}: {stats['clean_count']:,} values → "
                  f"ECDF: {stats['ecdf_distinct_pairs']:,} pairs, "
                  f"Bins: {stats['num_bins']}")

        except Exception as e:
            print(f"  ❌ Error processing {column}: {e}")
            log_entries.append(f"[ERROR] respiratory_support.{column}: {str(e)}")

    print()

    return labs_stats, vitals_stats, resp_stats, log_entries


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

    ensure_output_tree()

    output_dir = str(OVERALL)
    print(f"Output directory: {output_dir}\n")

    copy_configs_to_output()
    print()

    log_file = str(meta_dir() / 'unit_mismatches.log')

    # ========================================================================
    # Initialise shared DuckDB connection
    # ========================================================================

    db = ClifDB(clif_config['tables_path'], clif_config['file_type'])

    # ========================================================================
    # Extract ICU Time Windows
    # ========================================================================

    icu_windows = extract_icu_time_windows(db)
    print()

    # ========================================================================
    # Compute Event-Based Time Windows for Strata
    # ========================================================================

    discharge_times = load_discharge_times(db)

    # Register discharge_times so event-window functions can join against it
    db.register('discharge_times', discharge_times)

    vaso_windows = extract_vaso_event_windows(db)
    resp_windows = extract_advanced_resp_event_windows(db)
    nippv_hfnc_windows = extract_nippv_hfnc_event_windows(db)

    all_time_windows = {
        'icu': icu_windows,
        'vaso': vaso_windows,
        'resp': resp_windows,
        'nippv_hfnc': nippv_hfnc_windows,
    }
    print()

    # ========================================================================
    # Discover Lab Category-Unit Combinations
    # ========================================================================

    print("="*80)
    print("Discovering Lab Category-Unit Combinations")
    print("="*80)

    lab_category_units = discover_lab_category_units(db)
    print()

    # ========================================================================
    # Run overall ECDF pipeline
    # ========================================================================

    labs_stats, vitals_stats, resp_stats, log_entries = run_ecdf_pipeline(
        icu_windows=icu_windows,
        lab_category_units=lab_category_units,
        db=db,
        clif_config=clif_config,
        outlier_config=outlier_config,
        lab_vital_config=lab_vital_config,
        output_dir=output_dir,
        label="overall"
    )

    # ========================================================================
    # Run stratified ECDF pipelines
    # ========================================================================

    try:
        from modules.strata import load_strata_hospitalization_ids
        from modules.utils.output_paths import parse_stratum, cohort_dir

        strata_hosp_ids = load_strata_hospitalization_ids()

        for stratum_name, hosp_ids in strata_hosp_ids.items():
            # Select the right temporal windows for this stratum
            window_type = STRATUM_WINDOW_TYPE.get(stratum_name, 'icu')
            base_windows = all_time_windows[window_type]

            filtered_windows = base_windows[
                base_windows['hospitalization_id'].isin(hosp_ids)
            ]
            if len(filtered_windows) == 0:
                print(f"  ⚠️ Skipping {stratum_name}: no time windows")
                continue

            window_label = {'icu': 'ICU', 'vaso': 'vaso-onset→discharge', 'resp': 'resp-onset→discharge', 'nippv_hfnc': 'nippv/hfnc-onset→discharge'}
            print(f"\n{'='*80}")
            print(f"STRATIFIED ECDF: {stratum_name} "
                  f"({len(filtered_windows):,} {window_label[window_type]} windows)")
            print(f"{'='*80}\n")

            _, stratum_suffix = parse_stratum(stratum_name)
            stratum_output_dir = str(cohort_dir(stratum_name))
            os.makedirs(stratum_output_dir, exist_ok=True)

            s_labs, s_vitals, s_resp, s_log = run_ecdf_pipeline(
                icu_windows=filtered_windows,
                lab_category_units=lab_category_units,
                db=db,
                clif_config=clif_config,
                outlier_config=outlier_config,
                lab_vital_config=lab_vital_config,
                output_dir=stratum_output_dir,
                label=stratum_name,
                suffix=stratum_suffix
            )
            log_entries.extend(s_log)
            print(f"  ✅ {stratum_name}: {len(s_labs)} labs, {len(s_vitals)} vitals, {len(s_resp)} resp")

    except FileNotFoundError as e:
        print(f"\n  ⚠️ Skipping stratified ECDF: {e}")

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
    print(f"Vitals: {len(vitals_stats)} categories processed")
    print(f"Respiratory Support: {len(resp_stats)} columns processed")
    print()

    all_stats = labs_stats + vitals_stats + resp_stats
    total_original = sum(s.get('original_count', 0) for s in all_stats)
    total_clean = sum(s.get('clean_count', 0) for s in all_stats)

    if total_original > 0:
        pct = total_clean / total_original * 100
        print(f"Data Retention: {total_clean:,} / {total_original:,} ({pct:.1f}%)")
    print(f"Duration: {duration:.1f}s ({duration/60:.1f}m)")
    print()

    # Clean up
    db.close()
