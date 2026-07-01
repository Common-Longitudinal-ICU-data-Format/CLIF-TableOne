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
        ├── lab_category_units.csv   # (cat, unit) pairs in data + schema status
        ├── unit_mismatches.log      # data vs CLIF labs schema (site issues)
        └── ecdf_coverage_gaps.log   # schema-valid cats w/o bin config + errors
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
ADVANCED_RESP_DEVICES = ['imv', 'nippv', 'cpap', 'hfnc']
NIPPV_HFNC_DEVICES = ['nippv', 'hfnc']

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
    in the labs data, with a per-pair row count.

    Returns:
        pandas DataFrame with columns:
        - lab_category: str
        - reference_unit: str
        - n_rows: int (total rows in labs for this pair)
    """
    labs_path = db.table_path('labs')

    print(f"Discovering lab category-unit combinations from {labs_path}...")

    df = db.query_df(
        """
        SELECT
            LOWER(TRIM(lab_category)) AS lab_category,
            COALESCE(
                NULLIF(LOWER(TRIM(reference_unit)), ''),
                '(no units)'
            ) AS reference_unit,
            COUNT(*) AS n_rows
        FROM read_parquet(?)
        WHERE lab_category IS NOT NULL
        GROUP BY 1, 2
        """,
        [labs_path],
    )

    print(f"✓ Found {len(df):,} unique (category, unit) combinations")
    return df


def write_lab_category_units_csv(
    lab_category_units: pd.DataFrame,
    labs_schema: Dict[str, Any],
    output_path: str,
) -> str:
    """Write an audit CSV of every (lab_category, reference_unit) pair found
    in the labs data, classified against the CLIF labs schema.

    Columns:
      lab_category, reference_unit, n_rows, schema_status, canonical_unit

    schema_status is one of:
      - 'ok'            — category and unit both accepted by CLIF
      - 'unit_mismatch' — category accepted but unit spelling not in variants
      - 'not_in_spec'   — category is not in the CLIF labs vocabulary

    Sorted so 'ok' rows come first and most-frequent pairs float to the top
    within each status group.
    """
    rows = []
    for r in lab_category_units.itertuples(index=False):
        status, canonical = _classify_lab_unit(r.lab_category, r.reference_unit, labs_schema)
        rows.append({
            'lab_category': r.lab_category,
            'reference_unit': r.reference_unit,
            'n_rows': int(getattr(r, 'n_rows', 0)),
            'schema_status': status,
            'canonical_unit': canonical or '',
        })

    status_order = {'ok': 0, 'unit_mismatch': 1, 'not_in_spec': 2}
    df = pd.DataFrame(rows)
    df['_ord'] = df['schema_status'].map(status_order).fillna(99)
    df = df.sort_values(['_ord', 'n_rows'], ascending=[True, False]).drop(columns='_ord')
    df.to_csv(output_path, index=False)
    return output_path


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
    suffix: str = '',
    prefetched_values=None,
) -> Dict[str, Any]:
    """
    Process a single lab/vital category:
    1. Load data filtered to ICU time windows (via DuckDB join)
    2. Remove outliers
    3. Compute ECDF
    4. Compute quantile bins with auto-extreme-splitting
    5. Save results

    If prefetched_values is provided, skip the DB fetch (batch optimization).
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

    if prefetched_values is not None:
        values = prefetched_values
    else:
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

def _classify_lab_unit(
    category: str,
    unit: str,
    labs_schema: Dict[str, Any],
) -> Tuple[str, str | None]:
    """Classify a (category, unit) pair against the CLIF labs schema.

    The schema (from clifpy) is the authoritative source for which lab
    categories are legal CLIF vocabulary and which unit spellings are
    accepted for each category. Separated from the binning-config check
    so a category that's valid per CLIF but lacks an ECDF bin definition
    is classified as a coverage gap, not a unit mismatch.

    Returns a tuple (status, canonical):
        ('not_in_spec', None)       — category is not in CLIF's labs vocab
        ('unit_mismatch', canonical) — category is valid but unit spelling isn't
        ('ok', canonical)            — category and unit both accepted
    """
    ref_units = labs_schema.get('lab_reference_units', {}) or {}
    variants = labs_schema.get('allowed_unit_variants', {}) or {}
    canonical = ref_units.get(category)
    if canonical is None:
        return ('not_in_spec', None)
    allowed = variants.get(canonical) or [canonical]
    allowed_lower = {str(v).lower().strip() for v in allowed}
    if (unit or '').lower().strip() not in allowed_lower:
        return ('unit_mismatch', canonical)
    return ('ok', canonical)


def run_ecdf_pipeline(
    icu_windows: pd.DataFrame,
    lab_category_units: pd.DataFrame,
    db: ClifDB,
    clif_config: Dict,
    outlier_config: Dict,
    lab_vital_config: Dict,
    labs_schema: Dict[str, Any],
    output_dir: str,
    label: str = "overall",
    suffix: str = ''
) -> Tuple[List, List, List, List, List]:
    """
    Run the full ECDF/bins pipeline for a given set of ICU windows.

    This is the core processing loop extracted from main() so it can be
    called once for overall and once per encounter-type stratum.

    Returns:
        (labs_stats, vitals_stats, resp_stats, mismatch_entries, coverage_entries)

    Two log streams are returned separately:
      * mismatch_entries — data that doesn't comply with the CLIF labs schema
        (unknown category or unrecognized unit spelling). Written to
        unit_mismatches.log by the caller.
      * coverage_entries — valid CLIF categories that the ECDF pipeline
        skipped because of missing binning/outlier config, plus any runtime
        errors during bin generation. Written to ecdf_coverage_gaps.log.
    """
    # Register the (possibly stratum-filtered) time windows for SQL joins
    db.register('icu_windows', icu_windows)

    mismatch_entries: List[str] = []
    coverage_entries: List[str] = []

    # ── Batch pre-fetch: load all ICU-windowed values in one pass ──────
    # This avoids re-reading parquet metadata + re-joining with icu_windows
    # for each of the ~90 categories.  ~40-50% speedup on ECDF hot loop.
    import time as _time
    _batch_t0 = _time.time()
    try:
        _labs_view = db.preload_icu_joined_view(
            'labs', 'lab_value_numeric', 'lab_result_dttm', 'lab_category')
        _labs_batch = db.fetch_batch_categories(_labs_view, 'lab_value_numeric', [], 'labs')
        print(f"  Batch-loaded labs: {len(_labs_batch)} (category, unit) groups "
              f"in {_time.time() - _batch_t0:.1f}s")
    except Exception as e:
        print(f"  ⚠️ Labs batch pre-fetch failed ({e}), falling back to per-category queries")
        _labs_batch = {}

    _batch_t1 = _time.time()
    try:
        _vitals_view = db.preload_icu_joined_view(
            'vitals', 'vital_value', 'recorded_dttm', 'vital_category')
        _vitals_batch = db.fetch_batch_categories(_vitals_view, 'vital_value', [], 'vitals')
        print(f"  Batch-loaded vitals: {len(_vitals_batch)} categories "
              f"in {_time.time() - _batch_t1:.1f}s")
    except Exception as e:
        print(f"  ⚠️ Vitals batch pre-fetch failed ({e}), falling back to per-category queries")
        _vitals_batch = {}
    # ───────────────────────────────────────────────────────────────────

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

        # Schema-compliance check first: is this (category, unit) pair legal
        # CLIF vocabulary? Any failure here is a data-quality issue for the
        # site, not a coverage gap in this repo's ECDF configs.
        status, canonical = _classify_lab_unit(category, unit, labs_schema)
        if status == 'not_in_spec':
            mismatch_entries.append(f"[NOT IN CLIF SPEC] {category} ({unit})")
            continue
        if status == 'unit_mismatch':
            mismatch_entries.append(
                f"[UNIT MISMATCH] {category}: found '{unit}' in data, "
                f"CLIF canonical is '{canonical}'"
            )
            continue

        # Data is schema-valid; now check ECDF pipeline coverage.
        if category not in labs_config:
            coverage_entries.append(
                f"[ECDF SKIP] labs.{category} ({unit}): "
                f"no binning config in lab_vital_config.yaml"
            )
            continue

        if category not in labs_outlier:
            coverage_entries.append(
                f"[ECDF SKIP] labs.{category} ({unit}): "
                f"no outlier bounds in outlier_config.yaml"
            )
            continue

        config_units = labs_config[category].get('reference_unit', [])
        if isinstance(config_units, str):
            config_units = [config_units]

        processed_categories.add(category)

        try:
            # Look up pre-fetched values from batch
            _prefetched = None
            if _labs_batch:
                import numpy as _np
                # Match: batch key is (category_lower, unit_lower)
                # config_units may be a list of acceptable units
                _cat_lower = category.lower().strip()
                _matched_arrays = []
                for (_bc, _bu), _bv in _labs_batch.items():
                    if _bc == _cat_lower:
                        if isinstance(config_units, list):
                            if _bu in [u.lower().strip() for u in config_units]:
                                _matched_arrays.append(_bv)
                            elif _bu is None or _bu == '' or _bu == '(no units)':
                                if any(u.lower().strip() == '(no units)' for u in config_units):
                                    _matched_arrays.append(_bv)
                        elif _bu == config_units.lower().strip():
                            _matched_arrays.append(_bv)
                if _matched_arrays:
                    _prefetched = _np.concatenate(_matched_arrays) if len(_matched_arrays) > 1 else _matched_arrays[0]

            stats = process_category(
                table_type='labs',
                category=category,
                unit=config_units,
                db=db,
                outlier_range=labs_outlier[category],
                cat_config=labs_config[category],
                output_dir=output_dir,
                extreme_bins_count=5,
                suffix=suffix,
                prefetched_values=_prefetched,
            )
            labs_stats.append(stats)

            print(f"  ✓ {category} ({config_units}): {stats['clean_count']:,} values → "
                  f"ECDF: {stats['ecdf_distinct_pairs']:,} pairs, "
                  f"Bins: {stats['num_bins']}")

        except Exception as e:
            print(f"  ❌ Error processing {category} ({config_units}): {e}")
            coverage_entries.append(f"[ERROR] labs.{category} ({config_units}): {str(e)}")

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
            coverage_entries.append(
                f"[ECDF SKIP] vitals.{category}: no outlier bounds in outlier_config.yaml"
            )
            continue

        extreme_bins = 5 if category in ['height_cm', 'weight_kg'] else 10

        try:
            # Look up pre-fetched values from batch
            _prefetched_v = None
            if _vitals_batch:
                _cat_lower = category.lower().strip()
                _bv = _vitals_batch.get((_cat_lower, None))
                if _bv is not None and len(_bv) > 0:
                    _prefetched_v = _bv

            stats = process_category(
                table_type='vitals',
                category=category,
                unit=None,
                db=db,
                outlier_range=vitals_outlier[category],
                cat_config=vitals_config[category],
                output_dir=output_dir,
                extreme_bins_count=extreme_bins,
                prefetched_values=_prefetched_v,
                suffix=suffix
            )
            vitals_stats.append(stats)

            print(f"  ✓ {category}: {stats['clean_count']:,} values → "
                  f"ECDF: {stats['ecdf_distinct_pairs']:,} pairs, "
                  f"Bins: {stats['num_bins']}")

        except Exception as e:
            print(f"  ❌ Error processing {category}: {e}")
            coverage_entries.append(f"[ERROR] vitals.{category}: {str(e)}")

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
            coverage_entries.append(
                f"[ECDF SKIP] respiratory_support.{column}: "
                f"no outlier bounds in outlier_config.yaml"
            )
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
            coverage_entries.append(f"[ERROR] respiratory_support.{column}: {str(e)}")

    print()

    return labs_stats, vitals_stats, resp_stats, mismatch_entries, coverage_entries


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

    mismatches_log_file = str(meta_dir() / 'unit_mismatches.log')
    coverage_log_file = str(meta_dir() / 'ecdf_coverage_gaps.log')

    # Load the CLIF labs schema — authoritative source for legal lab_category
    # values and the accepted unit-spelling variants for each.
    from modules.utils.clif_loader import _load_schema
    labs_schema = _load_schema('labs') or {}

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

    # Audit CSV: every (lab_category, reference_unit) pair found in the data,
    # classified against the CLIF labs schema and sorted by row count. Useful
    # as a one-glance answer to "what lab vocabulary is in this dataset, and
    # how much of it complies with CLIF?" — independent of ECDF bin config.
    lab_cat_units_csv = str(meta_dir() / 'lab_category_units.csv')
    write_lab_category_units_csv(lab_category_units, labs_schema, lab_cat_units_csv)
    print(f"✓ Lab category/unit audit: {lab_cat_units_csv}")
    print()

    # ========================================================================
    # Run overall ECDF pipeline
    # ========================================================================

    labs_stats, vitals_stats, resp_stats, mismatch_entries, coverage_entries = run_ecdf_pipeline(
        icu_windows=icu_windows,
        lab_category_units=lab_category_units,
        db=db,
        clif_config=clif_config,
        outlier_config=outlier_config,
        lab_vital_config=lab_vital_config,
        labs_schema=labs_schema,
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

            s_labs, s_vitals, s_resp, s_mismatch, s_coverage = run_ecdf_pipeline(
                icu_windows=filtered_windows,
                lab_category_units=lab_category_units,
                db=db,
                clif_config=clif_config,
                outlier_config=outlier_config,
                lab_vital_config=lab_vital_config,
                labs_schema=labs_schema,
                output_dir=stratum_output_dir,
                label=stratum_name,
                suffix=stratum_suffix
            )
            mismatch_entries.extend(s_mismatch)
            coverage_entries.extend(s_coverage)
            print(f"  ✅ {stratum_name}: {len(s_labs)} labs, {len(s_vitals)} vitals, {len(s_resp)} resp")

    except FileNotFoundError as e:
        print(f"\n  ⚠️ Skipping stratified ECDF: {e}")

    # ========================================================================
    # Write Log File
    # ========================================================================

    if mismatch_entries:
        with open(mismatches_log_file, 'w', encoding='utf-8') as f:
            f.write("CLIF Schema Unit Mismatches\n")
            f.write("="*80 + "\n")
            f.write("Data rows whose lab_category or reference_unit isn't accepted\n")
            f.write("by the CLIF labs schema (clifpy/schemas/labs_schema.yaml).\n")
            f.write("These are data-quality issues for the site to investigate.\n\n")
            for entry in mismatch_entries:
                f.write(entry + "\n")
        print(f"✓ Mismatch log: {mismatches_log_file} ({len(mismatch_entries)} entries)")

    if coverage_entries:
        with open(coverage_log_file, 'w', encoding='utf-8') as f:
            f.write("ECDF Coverage Gaps and Processing Issues\n")
            f.write("="*80 + "\n")
            f.write("Categories that are valid per the CLIF schema but the ECDF\n")
            f.write("pipeline skipped because the binning or outlier config\n")
            f.write("doesn't define them — plus any runtime errors. These are\n")
            f.write("coverage gaps in this repo's configs, not data issues.\n\n")
            for entry in coverage_entries:
                f.write(entry + "\n")
        print(f"✓ Coverage log: {coverage_log_file} ({len(coverage_entries)} entries)")

    if mismatch_entries or coverage_entries:
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
