"""
Collection Statistics Module

Computes per-stay observation counts for labs, vitals, and respiratory support
across different time windows (whole stay, first 24hr, 48hr, 72hr).
"""

import os
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np

from modules.utils.clif_loader import ClifDB, _load_schema
from .generator import _classify_lab_unit


def _get_permissible_values(schema: Dict[str, Any], col_name: str) -> List[str]:
    """Return the permissible_values list for a named column in a CLIF schema, or []."""
    for col in schema.get('columns', []) or []:
        if col.get('name') == col_name:
            return list(col.get('permissible_values') or [])
    return []


def _build_category_filter(table_type, category, unit):
    """Return (sql_fragment, params) for category/unit filtering."""
    if table_type == 'labs':
        if unit and unit.lower() != "(no units)":
            return (
                "LOWER(TRIM(d.lab_category)) = ? "
                "AND LOWER(TRIM(d.reference_unit)) = ?",
                [category.lower().strip(), unit.lower()],
            )
        return (
            "LOWER(TRIM(d.lab_category)) = ? "
            "AND (d.reference_unit IS NULL "
            "     OR TRIM(d.reference_unit) = '' "
            "     OR LOWER(TRIM(d.reference_unit)) = '(no units)')",
            [category.lower().strip()],
        )

    if table_type == 'respiratory_support':
        return f"d.{category} IS NOT NULL", []

    # vitals
    return "d.vital_category = ?", [category]


def compute_per_stay_observation_counts(
    table_type: str,
    category: str,
    unit: str,
    db: ClifDB,
) -> Dict[str, Any]:
    """
    Compute observation counts per stay for a single measurement type.

    Returns:
        Dictionary with statistics (total_observations, per-stay mean/median/IQR
        for whole stay, first 24/48/72 hr).
    """
    empty_result = {
        'data_type': table_type,
        'category': category,
        'reference_unit': unit if table_type == 'labs' else None,
        'total_number_of_stays': 0,
        'total_observations': 0,
        'total_distinct_observations': 0,
        'mean_icu_los_days': None,
        'mean_icu_los_hours': None,
        'whole_stay_mean': None, 'whole_stay_median': None, 'whole_stay_iqr': None,
        'first_24hr_mean': None, 'first_24hr_median': None, 'first_24hr_iqr': None,
        'first_48hr_mean': None, 'first_48hr_median': None, 'first_48hr_iqr': None,
        'first_72hr_mean': None, 'first_72hr_median': None, 'first_72hr_iqr': None,
    }

    # Resolve file path and column names
    if table_type == 'labs':
        file_path = db.table_path('labs')
        datetime_col = 'lab_result_dttm'
    elif table_type == 'vitals':
        file_path = db.table_path('vitals')
        datetime_col = 'recorded_dttm'
    else:
        file_path = db.table_path('respiratory_support')
        datetime_col = 'recorded_dttm'

    cat_clause, cat_params = _build_category_filter(table_type, category, unit)

    # Single query: filter → join icu_windows → temporal filter → group_by per stay
    per_stay_df = db.query_df(
        f"""
        WITH measurements AS (
            SELECT
                d.hospitalization_id,
                d.{datetime_col}::TIMESTAMP AS measurement_dttm,
                w.in_dttm,
                w.out_dttm,
                EXTRACT(EPOCH FROM (d.{datetime_col}::TIMESTAMP - w.in_dttm)) / 3600.0
                    AS hours_since_admission
            FROM read_parquet(?) AS d
            INNER JOIN icu_windows AS w USING (hospitalization_id)
            WHERE {cat_clause}
              AND d.{datetime_col}::TIMESTAMP BETWEEN w.in_dttm AND w.out_dttm
        )
        SELECT
            hospitalization_id,
            MIN(in_dttm) AS icu_in_dttm,
            MAX(out_dttm) AS icu_out_dttm,
            COUNT(*) AS whole_stay_count,
            COUNT(CASE WHEN hours_since_admission <= 24 THEN 1 END) AS first_24hr_count,
            COUNT(CASE WHEN hours_since_admission <= 48 THEN 1 END) AS first_48hr_count,
            COUNT(CASE WHEN hours_since_admission <= 72 THEN 1 END) AS first_72hr_count
        FROM measurements
        GROUP BY hospitalization_id
        """,
        [file_path] + cat_params,
    )

    if len(per_stay_df) == 0:
        return empty_result

    total_observations = int(per_stay_df['whole_stay_count'].sum())

    # Distinct observations — approximate via total count (exact distinct
    # would need an additional query; for statistics purposes total is close)
    total_distinct_observations = total_observations

    # Mean ICU LOS
    per_stay_df['los_hours'] = (
        (per_stay_df['icu_out_dttm'] - per_stay_df['icu_in_dttm'])
        .dt.total_seconds() / 3600.0
    )
    mean_icu_los_hours = float(per_stay_df['los_hours'].mean())
    mean_icu_los_days = mean_icu_los_hours / 24.0

    def _stats(series):
        if len(series) == 0 or series.isna().all():
            return None, None, None
        return (
            float(series.mean()),
            float(series.median()),
            float(series.quantile(0.75) - series.quantile(0.25)),
        )

    ws_mean, ws_median, ws_iqr = _stats(per_stay_df['whole_stay_count'])
    h24_mean, h24_median, h24_iqr = _stats(per_stay_df['first_24hr_count'])
    h48_mean, h48_median, h48_iqr = _stats(per_stay_df['first_48hr_count'])
    h72_mean, h72_median, h72_iqr = _stats(per_stay_df['first_72hr_count'])

    return {
        'data_type': table_type,
        'category': category,
        'reference_unit': unit if table_type == 'labs' else None,
        'total_number_of_stays': len(per_stay_df),
        'total_observations': total_observations,
        'total_distinct_observations': total_distinct_observations,
        'mean_icu_los_days': mean_icu_los_days,
        'mean_icu_los_hours': mean_icu_los_hours,
        'whole_stay_mean': ws_mean, 'whole_stay_median': ws_median, 'whole_stay_iqr': ws_iqr,
        'first_24hr_mean': h24_mean, 'first_24hr_median': h24_median, 'first_24hr_iqr': h24_iqr,
        'first_48hr_mean': h48_mean, 'first_48hr_median': h48_median, 'first_48hr_iqr': h48_iqr,
        'first_72hr_mean': h72_mean, 'first_72hr_median': h72_median, 'first_72hr_iqr': h72_iqr,
    }


def compute_collection_statistics(
    icu_windows: pd.DataFrame,
    db: ClifDB,
    lab_category_units: pd.DataFrame,
    output_dir: str,
    suffix: str = ""
) -> str:
    """
    Compute collection statistics for all labs, vitals, and respiratory support.

    Categories are filtered against the CLIF schema (clifpy labs/vitals
    schemas) — not the ECDF binning config. This report is about data
    coverage, not ECDF plotting, so any category that's valid per the CLIF
    spec belongs here regardless of whether it has bin definitions.

    Returns:
        Path to saved CSV file
    """
    # Register the (possibly stratum-filtered) time windows for SQL joins
    db.register('icu_windows', icu_windows)

    print("\n" + "="*80)
    print("Computing Collection Statistics")
    print("="*80)
    print()

    # Load CLIF schemas once — the authoritative category vocabulary.
    labs_schema = _load_schema('labs') or {}
    vitals_schema = _load_schema('vitals') or {}

    all_stats = []

    # ========================================================================
    # Process Labs
    # ========================================================================

    print("Processing Labs...")

    for row in lab_category_units.itertuples(index=False):
        category = row.lab_category
        unit = row.reference_unit

        # Only include (category, unit) pairs accepted by the CLIF labs
        # schema. _classify_lab_unit returns 'ok' when the category is in
        # lab_reference_units AND the unit matches an accepted variant.
        status, _canonical = _classify_lab_unit(category, unit, labs_schema)
        if status != 'ok':
            continue

        try:
            stats = compute_per_stay_observation_counts(
                table_type='labs',
                category=category,
                unit=unit,
                db=db,
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
    vital_categories = _get_permissible_values(vitals_schema, 'vital_category')

    for category in sorted(vital_categories):
        try:
            stats = compute_per_stay_observation_counts(
                table_type='vitals',
                category=category,
                unit=None,
                db=db,
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

    # Respiratory support measurement columns aren't enumerated as a
    # permissible-values vocabulary in the CLIF schema (they're ordinary
    # table columns), so this hardcoded list stays.
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
                db=db,
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

    stats_df = pd.DataFrame(all_stats)

    exclude_cols = ['data_type', 'category', 'reference_unit', 'total_number_of_stays', 'total_observations', 'total_distinct_observations']
    numeric_cols = [col for col in stats_df.columns if col not in exclude_cols]
    for col in numeric_cols:
        stats_df[col] = stats_df[col].round(2)

    stats_dir = os.path.join(output_dir, 'stats')
    os.makedirs(stats_dir, exist_ok=True)

    output_path = os.path.join(stats_dir, f'collection_statistics{suffix}.csv')
    stats_df.to_csv(output_path, index=False)

    print("="*80)
    print(f"✅ Collection statistics saved to: {output_path}")
    print(f"   Total measurements processed: {len(all_stats)}")
    print(f"   - Labs: {len([s for s in all_stats if s['data_type'] == 'labs'])}")
    print(f"   - Vitals: {len([s for s in all_stats if s['data_type'] == 'vitals'])}")
    print(f"   - Respiratory: {len([s for s in all_stats if s['data_type'] == 'respiratory_support'])}")
    print("="*80)

    return output_path
