"""
Collection Statistics Module

Computes per-stay observation counts for labs, vitals, and respiratory support
across different time windows (whole stay, first 24hr, 48hr, 72hr).
"""

import os
from pathlib import Path
from typing import Dict, List, Any
import polars as pl
import pandas as pd
import numpy as np


def compute_per_stay_observation_counts(
    table_type: str,
    category: str,
    unit: str,
    icu_windows: pl.DataFrame,
    tables_path: str,
    file_type: str
) -> Dict[str, Any]:
    """
    Compute observation counts per stay for a single measurement type.

    Args:
        table_type: 'labs', 'vitals', or 'respiratory_support'
        category: lab_category, vital_category, or column_name
        unit: reference_unit (for labs only, None for others)
        icu_windows: ICU time windows with hospitalization_id, in_dttm, out_dttm
        tables_path: Path to CLIF data directory
        file_type: File type (e.g., 'parquet')

    Returns:
        Dictionary with statistics:
        - data_type, category, reference_unit
        - total_number_of_stays
        - total_observations (all rows)
        - total_distinct_observations (unique events: stay + timestamp + category)
        - mean_icu_los_days (mean ICU length of stay in days for stays with this measurement)
        - mean_icu_los_hours (mean ICU length of stay in hours for stays with this measurement)
        - whole_stay_mean/median/iqr
        - first_24hr_mean/median/iqr
        - first_48hr_mean/median/iqr
        - first_72hr_mean/median/iqr
    """
    # Determine file path and column names
    if table_type == 'labs':
        file_path = os.path.join(tables_path, f'clif_labs.{file_type}')
        category_col = 'lab_category'
        datetime_col = 'lab_result_dttm'
    elif table_type == 'vitals':
        file_path = os.path.join(tables_path, f'clif_vitals.{file_type}')
        category_col = 'vital_category'
        datetime_col = 'recorded_dttm'
    else:  # respiratory_support
        file_path = os.path.join(tables_path, f'clif_respiratory_support.{file_type}')
        category_col = None  # No category column, just column names
        datetime_col = 'recorded_dttm'

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    # Scan data with Polars (lazy)
    data_lazy = pl.scan_parquet(file_path)

    # Filter to selected category (and unit for labs)
    if table_type == 'labs' and unit:
        data_filtered = data_lazy.filter(
            (pl.col(category_col) == category) &
            (pl.col('reference_unit') == unit)
        ).select([
            'hospitalization_id',
            datetime_col,
            category_col  # Include category for distinct count
        ])
    elif table_type == 'respiratory_support':
        # For respiratory, we need to filter to non-null values in the specific column
        data_filtered = data_lazy.filter(
            pl.col(category).is_not_null()
        ).select([
            'hospitalization_id',
            datetime_col
        ])
    else:  # vitals
        data_filtered = data_lazy.filter(
            pl.col(category_col) == category
        ).select([
            'hospitalization_id',
            datetime_col,
            category_col  # Include category for distinct count
        ])

    # Join with ICU time windows
    data_with_windows = data_filtered.join(
        icu_windows.lazy(),
        on='hospitalization_id',
        how='inner'
    )

    # Strip timezone from ALL datetime columns for comparison (defensive)
    data_with_windows = data_with_windows.with_columns([
        pl.col(datetime_col).dt.replace_time_zone(None).alias('measurement_dttm_clean'),
        pl.col('in_dttm').dt.replace_time_zone(None).alias('icu_in_dttm'),
        pl.col('out_dttm').dt.replace_time_zone(None).alias('icu_out_dttm')
    ])

    # Calculate time since ICU admission in hours
    data_with_windows = data_with_windows.with_columns([
        ((pl.col('measurement_dttm_clean') - pl.col('icu_in_dttm')).dt.total_hours()).alias('hours_since_admission')
    ])

    # Filter to measurements during ICU stay only
    data_icu_stay = data_with_windows.filter(
        (pl.col('measurement_dttm_clean') >= pl.col('icu_in_dttm')) &
        (pl.col('measurement_dttm_clean') <= pl.col('icu_out_dttm'))
    )

    # Collect with streaming
    data_df = data_icu_stay.collect(streaming=True)

    # Calculate total number of observations
    total_observations = len(data_df)

    # Calculate distinct observation events
    if table_type == 'labs':
        # Distinct (hospitalization_id, lab_result_dttm, lab_category)
        distinct_cols = ['hospitalization_id', 'measurement_dttm_clean', category_col]
    elif table_type == 'vitals':
        # Distinct (hospitalization_id, recorded_dttm, vital_category)
        distinct_cols = ['hospitalization_id', 'measurement_dttm_clean', category_col]
    else:  # respiratory_support
        # Distinct (hospitalization_id, recorded_dttm)
        distinct_cols = ['hospitalization_id', 'measurement_dttm_clean']

    total_distinct_observations = data_df.select(distinct_cols).unique().height if len(data_df) > 0 else 0

    # Calculate mean ICU LOS for stays with this measurement
    if len(data_df) > 0:
        los_df = data_df.select([
            'hospitalization_id', 'icu_in_dttm', 'icu_out_dttm'
        ]).unique()

        los_df = los_df.with_columns([
            ((pl.col('icu_out_dttm') - pl.col('icu_in_dttm')).dt.total_hours()).alias('los_hours')
        ])

        mean_icu_los_hours = float(los_df['los_hours'].mean()) if los_df['los_hours'].len() > 0 else None
        mean_icu_los_days = mean_icu_los_hours / 24 if mean_icu_los_hours is not None else None
    else:
        mean_icu_los_hours = None
        mean_icu_los_days = None

    if len(data_df) == 0:
        return {
            'data_type': table_type,
            'category': category,
            'reference_unit': unit if table_type == 'labs' else None,
            'total_number_of_stays': 0,
            'total_observations': 0,
            'total_distinct_observations': 0,
            'mean_icu_los_days': None,
            'mean_icu_los_hours': None,
            'whole_stay_mean': None,
            'whole_stay_median': None,
            'whole_stay_iqr': None,
            'first_24hr_mean': None,
            'first_24hr_median': None,
            'first_24hr_iqr': None,
            'first_48hr_mean': None,
            'first_48hr_median': None,
            'first_48hr_iqr': None,
            'first_72hr_mean': None,
            'first_72hr_median': None,
            'first_72hr_iqr': None
        }

    # Group by hospitalization_id and count observations in each time window
    per_stay_counts = data_df.group_by('hospitalization_id').agg([
        # Whole stay: all measurements
        pl.count().alias('whole_stay_count'),
        # First 24 hours
        pl.col('hours_since_admission').filter(pl.col('hours_since_admission') <= 24).count().alias('first_24hr_count'),
        # First 48 hours
        pl.col('hours_since_admission').filter(pl.col('hours_since_admission') <= 48).count().alias('first_48hr_count'),
        # First 72 hours
        pl.col('hours_since_admission').filter(pl.col('hours_since_admission') <= 72).count().alias('first_72hr_count')
    ])

    # Convert to pandas for easier quantile calculations
    counts_pd = per_stay_counts.to_pandas()

    # Calculate statistics for each time window
    def calculate_stats(counts_series):
        """Calculate mean, median, and IQR for a series of counts."""
        if len(counts_series) == 0 or counts_series.isna().all():
            return None, None, None

        mean_val = float(counts_series.mean())
        median_val = float(counts_series.median())
        q1 = counts_series.quantile(0.25)
        q3 = counts_series.quantile(0.75)
        iqr_val = float(q3 - q1)

        return mean_val, median_val, iqr_val

    whole_stay_mean, whole_stay_median, whole_stay_iqr = calculate_stats(counts_pd['whole_stay_count'])
    first_24hr_mean, first_24hr_median, first_24hr_iqr = calculate_stats(counts_pd['first_24hr_count'])
    first_48hr_mean, first_48hr_median, first_48hr_iqr = calculate_stats(counts_pd['first_48hr_count'])
    first_72hr_mean, first_72hr_median, first_72hr_iqr = calculate_stats(counts_pd['first_72hr_count'])

    return {
        'data_type': table_type,
        'category': category,
        'reference_unit': unit if table_type == 'labs' else None,
        'total_number_of_stays': len(counts_pd),
        'total_observations': total_observations,
        'total_distinct_observations': total_distinct_observations,
        'mean_icu_los_days': mean_icu_los_days,
        'mean_icu_los_hours': mean_icu_los_hours,
        'whole_stay_mean': whole_stay_mean,
        'whole_stay_median': whole_stay_median,
        'whole_stay_iqr': whole_stay_iqr,
        'first_24hr_mean': first_24hr_mean,
        'first_24hr_median': first_24hr_median,
        'first_24hr_iqr': first_24hr_iqr,
        'first_48hr_mean': first_48hr_mean,
        'first_48hr_median': first_48hr_median,
        'first_48hr_iqr': first_48hr_iqr,
        'first_72hr_mean': first_72hr_mean,
        'first_72hr_median': first_72hr_median,
        'first_72hr_iqr': first_72hr_iqr
    }


def compute_collection_statistics(
    icu_windows: pl.DataFrame,
    tables_path: str,
    file_type: str,
    lab_category_units: pl.DataFrame,
    lab_vital_config: Dict[str, Any],
    output_dir: str
) -> str:
    """
    Compute collection statistics for all labs, vitals, and respiratory support.

    Args:
        icu_windows: ICU time windows
        tables_path: Path to CLIF data
        file_type: File type (e.g., 'parquet')
        lab_category_units: DataFrame with discovered lab category-unit combinations
        lab_vital_config: Configuration with vital categories
        output_dir: Output directory

    Returns:
        Path to saved CSV file
    """
    print("\n" + "="*80)
    print("Computing Collection Statistics")
    print("="*80)
    print()

    all_stats = []

    # ========================================================================
    # Process Labs
    # ========================================================================

    print("Processing Labs...")
    labs_config = lab_vital_config.get('labs', {})

    for row in lab_category_units.iter_rows(named=True):
        category = row['lab_category']
        unit = row['reference_unit']

        # Check if category exists in config
        if category not in labs_config:
            continue

        # Check if unit matches config
        config_unit = labs_config[category].get('reference_unit')
        if unit != config_unit:
            continue

        try:
            stats = compute_per_stay_observation_counts(
                table_type='labs',
                category=category,
                unit=unit,
                icu_windows=icu_windows,
                tables_path=tables_path,
                file_type=file_type
            )

            if stats['total_number_of_stays'] > 0:
                all_stats.append(stats)
                los_str = f", LOS={stats['mean_icu_los_days']:.1f}d" if stats['mean_icu_los_days'] else ""
                print(f"  ✓ {category} ({unit}): {stats['total_number_of_stays']} stays{los_str}, "
                      f"{stats['total_observations']:,} total obs, "
                      f"{stats['total_distinct_observations']:,} distinct, "
                      f"mean={stats['whole_stay_mean']:.1f} obs/stay")

        except Exception as e:
            print(f"  ❌ Error processing {category} ({unit}): {e}")

    print()

    # ========================================================================
    # Process Vitals
    # ========================================================================

    print("Processing Vitals...")
    vitals_config = lab_vital_config.get('vitals', {})

    for category in sorted(vitals_config.keys()):
        try:
            stats = compute_per_stay_observation_counts(
                table_type='vitals',
                category=category,
                unit=None,
                icu_windows=icu_windows,
                tables_path=tables_path,
                file_type=file_type
            )

            if stats['total_number_of_stays'] > 0:
                all_stats.append(stats)
                los_str = f", LOS={stats['mean_icu_los_days']:.1f}d" if stats['mean_icu_los_days'] else ""
                print(f"  ✓ {category}: {stats['total_number_of_stays']} stays{los_str}, "
                      f"{stats['total_observations']:,} total obs, "
                      f"{stats['total_distinct_observations']:,} distinct, "
                      f"mean={stats['whole_stay_mean']:.1f} obs/stay")

        except Exception as e:
            print(f"  ❌ Error processing {category}: {e}")

    print()

    # ========================================================================
    # Process Respiratory Support
    # ========================================================================

    print("Processing Respiratory Support (17 columns)...")

    resp_columns = [
        'fio2_set', 'lpm_set', 'tidal_volume_set', 'resp_rate_set',
        'pressure_control_set', 'pressure_support_set', 'flow_rate_set',
        'peak_inspiratory_pressure_set', 'inspiratory_time_set', 'peep_set',
        'tidal_volume_obs', 'resp_rate_obs', 'plateau_pressure_obs',
        'peak_inspiratory_pressure_obs', 'peep_obs', 'minute_vent_obs',
        'mean_airway_pressure_obs'
    ]

    for column in resp_columns:
        try:
            stats = compute_per_stay_observation_counts(
                table_type='respiratory_support',
                category=column,
                unit=None,
                icu_windows=icu_windows,
                tables_path=tables_path,
                file_type=file_type
            )

            if stats['total_number_of_stays'] > 0:
                all_stats.append(stats)
                los_str = f", LOS={stats['mean_icu_los_days']:.1f}d" if stats['mean_icu_los_days'] else ""
                print(f"  ✓ {column}: {stats['total_number_of_stays']} stays{los_str}, "
                      f"{stats['total_observations']:,} total obs, "
                      f"{stats['total_distinct_observations']:,} distinct, "
                      f"mean={stats['whole_stay_mean']:.1f} obs/stay")

        except Exception as e:
            print(f"  ❌ Error processing {column}: {e}")

    print()

    # ========================================================================
    # Save to CSV
    # ========================================================================

    if not all_stats:
        print("⚠️  No statistics computed")
        return None

    # Convert to DataFrame
    stats_df = pd.DataFrame(all_stats)

    # Round numerical columns to 2 decimal places (exclude integer count columns)
    exclude_cols = ['data_type', 'category', 'reference_unit', 'total_number_of_stays', 'total_observations', 'total_distinct_observations']
    numeric_cols = [col for col in stats_df.columns if col not in exclude_cols]
    for col in numeric_cols:
        stats_df[col] = stats_df[col].round(2)

    # Create stats directory
    stats_dir = os.path.join(output_dir, 'stats')
    os.makedirs(stats_dir, exist_ok=True)

    # Save to CSV
    output_path = os.path.join(stats_dir, 'collection_statistics.csv')
    stats_df.to_csv(output_path, index=False)

    print("="*80)
    print(f"✅ Collection statistics saved to: {output_path}")
    print(f"   Total measurements processed: {len(all_stats)}")
    print(f"   - Labs: {len([s for s in all_stats if s['data_type'] == 'labs'])}")
    print(f"   - Vitals: {len([s for s in all_stats if s['data_type'] == 'vitals'])}")
    print(f"   - Respiratory: {len([s for s in all_stats if s['data_type'] == 'respiratory_support'])}")
    print("="*80)

    return output_path
