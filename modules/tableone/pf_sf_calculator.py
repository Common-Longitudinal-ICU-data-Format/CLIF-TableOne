"""
PF/SF ratio calculator for advanced_resp strata.

Computes PaO2/FiO2 (PF) and SpO2/FiO2 (SF) ratios in the first 24 hours
after respiratory failure onset, following the CLIF PFvsSF Performance study
methodology.
"""

import polars as pl
import pandas as pd
import numpy as np
import logging
from typing import List, Optional
from pathlib import Path
from datetime import timedelta

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.sofa.calculator import (
    _load_labs,
    _load_vitals,
    _load_respiratory_support,
    _calculate_concurrent_pf_ratios,
    DEVICE_RANK_DICT,
)
from modules.utils.datetime_utils import standardize_datetime_columns

logger = logging.getLogger(__name__)

# Devices that qualify as respiratory failure onset
QUALIFYING_DEVICES_ALWAYS = ['imv', 'nippv', 'cpap']
HFNC_DEVICE = 'high flow nc'
HFNC_LPM_THRESHOLD = 30

# Outlier thresholds
PF_CAP = 600
SPO2_MIN, SPO2_MAX = 50, 100
PAO2_MIN, PAO2_MAX = 0, 700
FIO2_MIN, FIO2_MAX = 0.21, 1.0


def detect_respiratory_failure_onset(
    resp_df: pd.DataFrame,
    cohort_ids: np.ndarray,
    id_col: str = 'encounter_block'
) -> pd.DataFrame:
    """
    Find the earliest timestamp each encounter goes on a qualifying device.

    Qualifying: IMV, NIPPV, CPAP always; High Flow NC when lpm_set >= 30.

    Parameters
    ----------
    resp_df : pd.DataFrame
        Raw respiratory_support data with encounter_block already merged.
    cohort_ids : array-like
        Encounter IDs to restrict to.
    id_col : str
        Grouping column.

    Returns
    -------
    pd.DataFrame
        One row per qualifying encounter with columns:
        [id_col, 'hospitalization_id', 'onset_dttm', 'onset_device']
    """
    df = resp_df[resp_df[id_col].isin(cohort_ids)].copy()

    # Build qualifying mask
    device_lower = df['device_category'].str.lower()
    always_mask = device_lower.isin(QUALIFYING_DEVICES_ALWAYS)
    hfnc_mask = (device_lower == HFNC_DEVICE) & (df['lpm_set'] >= HFNC_LPM_THRESHOLD)
    qualifying = df[always_mask | hfnc_mask].copy()

    if qualifying.empty:
        logger.warning("No qualifying respiratory support rows found")
        return pd.DataFrame(columns=[id_col, 'hospitalization_id', 'onset_dttm', 'onset_device'])

    # For each encounter, find the earliest qualifying row
    idx = qualifying.groupby(id_col)['recorded_dttm'].idxmin()
    onset = qualifying.loc[idx, [id_col, 'hospitalization_id', 'recorded_dttm', 'device_category']].copy()
    onset = onset.rename(columns={'recorded_dttm': 'onset_dttm', 'device_category': 'onset_device'})

    # Keep one hospitalization_id per encounter (the one at onset)
    onset = onset.drop_duplicates(subset=[id_col])

    logger.info(f"Detected respiratory failure onset for {len(onset):,} encounters")
    return onset.reset_index(drop=True)


def _calculate_concurrent_sf_ratios(
    vitals_df: pl.DataFrame,
    resp_df: pl.DataFrame,
    time_tolerance_minutes: int = 240,
    id_cols: Optional[List[str]] = None
) -> pl.DataFrame:
    """
    Calculate S/F ratios from concurrent SpO2 and FiO2 measurements.

    Analogous to _calculate_concurrent_pf_ratios but for SpO2.
    Only eligible when SpO2 <= 97 (above this the O2 dissociation curve is flat).

    Parameters
    ----------
    vitals_df : pl.DataFrame
        Vitals data with spo2 values and recorded_dttm.
    resp_df : pl.DataFrame
        Respiratory support data with fio2_set (forward-filled) and recorded_dttm.
    time_tolerance_minutes : int
        Maximum lookback for FiO2 (default 240 = 4 hours).
    id_cols : list
        ID columns for joining.

    Returns
    -------
    pl.DataFrame
        Concurrent S/F ratios with device_category.
    """
    if id_cols is None:
        id_cols = ['hospitalization_id']

    # Filter to eligible SpO2: not null, within range, <= 97
    spo2_df = vitals_df.filter(
        pl.col('spo2').is_not_null() &
        (pl.col('spo2') >= SPO2_MIN) &
        (pl.col('spo2') <= SPO2_MAX) &
        (pl.col('spo2') <= 97)
    )

    if spo2_df.height == 0:
        logger.info("  No eligible SpO2 measurements (all > 97 or missing)")
        return pl.DataFrame(schema={
            **{c: pl.Utf8 for c in id_cols},
            'recorded_dttm': pl.Datetime('ns'),
            'spo2': pl.Float64,
            'fio2_set': pl.Float64,
            'device_category': pl.Utf8,
            'concurrent_sf': pl.Float64,
        })

    # Prepare respiratory data for join
    resp_for_join = resp_df.select([*id_cols, 'recorded_dttm', 'fio2_set', 'device_category'])

    # Sort for join_asof
    spo2_df = spo2_df.sort([*id_cols, 'recorded_dttm'])
    resp_for_join = resp_for_join.sort([*id_cols, 'recorded_dttm'])

    # Match each SpO2 with most recent FiO2 within tolerance
    spo2_with_fio2 = spo2_df.join_asof(
        resp_for_join,
        left_on='recorded_dttm',
        right_on='recorded_dttm',
        by=id_cols,
        tolerance=f'{time_tolerance_minutes}m',
        strategy='backward'
    )

    # Calculate SF ratio
    spo2_with_fio2 = spo2_with_fio2.with_columns([
        pl.when(
            (pl.col('spo2').is_not_null()) &
            (pl.col('fio2_set').is_not_null()) &
            (pl.col('fio2_set') >= FIO2_MIN) &
            (pl.col('fio2_set') <= FIO2_MAX)
        )
        .then(pl.col('spo2') / pl.col('fio2_set'))
        .otherwise(None)
        .alias('concurrent_sf')
    ])

    # Filter to successful matches
    sf_df = spo2_with_fio2.filter(pl.col('concurrent_sf').is_not_null())

    logger.info(f"  Calculated {len(sf_df)} concurrent S/F ratios from {len(spo2_df)} eligible SpO2 measurements")
    return sf_df


def _aggregate_pf_sf_per_encounter(
    pf_df: pl.DataFrame,
    sf_df: pl.DataFrame,
    onset_df: pd.DataFrame,
    id_col: str = 'encounter_block'
) -> pd.DataFrame:
    """
    Aggregate PF and SF measurements to one row per encounter.

    Returns worst (min), mean, median for each, plus device at worst,
    hours to worst, measurement counts, exposure group, and SF
    contemporary to worst PF.
    """
    onset_pl = pl.from_pandas(onset_df[[id_col, 'onset_dttm', 'onset_device']])
    # Standardize onset_dttm to match the precision/tz of the lab/vital
    # datetimes coming from the DuckDB-backed loaders (ns / site tz).
    # Without this, pandas-side μs/UTC sneaks through and the polars
    # subtraction below fails with "failed to determine supertype".
    from modules.utils.datetime_utils import canonical_tz_from_frame
    _target_tz = canonical_tz_from_frame(pf_df) or canonical_tz_from_frame(sf_df)
    if _target_tz is not None:
        onset_pl = standardize_datetime_columns(
            onset_pl,
            target_timezone=_target_tz,
            target_time_unit='ns',
            datetime_columns=['onset_dttm'],
        )

    # --- PF aggregation ---
    pf_agg = pl.DataFrame(schema={id_col: pl.Utf8})
    if pf_df.height > 0:
        # Add hours_from_onset
        pf_with_onset = pf_df.join(onset_pl, on=id_col, how='left')
        pf_with_onset = pf_with_onset.with_columns([
            ((pl.col('lab_result_dttm') - pl.col('onset_dttm')).dt.total_seconds() / 3600.0)
            .alias('hours_from_onset')
        ])

        # Cap PF at threshold
        pf_with_onset = pf_with_onset.with_columns([
            pl.when(pl.col('concurrent_pf') >= PF_CAP)
            .then(None)
            .otherwise(pl.col('concurrent_pf'))
            .alias('concurrent_pf')
        ]).filter(pl.col('concurrent_pf').is_not_null())

        if pf_with_onset.height > 0:
            pf_agg = pf_with_onset.group_by(id_col).agg([
                pl.col('concurrent_pf').min().alias('pf_24_min'),
                pl.col('concurrent_pf').mean().alias('pf_24_mean'),
                pl.col('concurrent_pf').median().alias('pf_24_median'),
                pl.col('concurrent_pf').count().alias('n_pf_measurements'),
                # Device at worst PF
                pl.col('device_category').sort_by('concurrent_pf').first().alias('device_at_worst_pf'),
                # Hours to worst PF
                pl.col('hours_from_onset').sort_by('concurrent_pf').first().alias('hours_to_worst_pf'),
                # Timestamp of worst PF (for SF contemporary lookup)
                pl.col('lab_result_dttm').sort_by('concurrent_pf').first().alias('worst_pf_dttm'),
            ])

    # --- SF aggregation ---
    sf_agg = pl.DataFrame(schema={id_col: pl.Utf8})
    if sf_df.height > 0:
        sf_with_onset = sf_df.join(onset_pl, on=id_col, how='left')
        sf_with_onset = sf_with_onset.with_columns([
            ((pl.col('recorded_dttm') - pl.col('onset_dttm')).dt.total_seconds() / 3600.0)
            .alias('hours_from_onset')
        ])

        sf_agg = sf_with_onset.group_by(id_col).agg([
            pl.col('concurrent_sf').min().alias('sf_24_min'),
            pl.col('concurrent_sf').mean().alias('sf_24_mean'),
            pl.col('concurrent_sf').median().alias('sf_24_median'),
            pl.col('concurrent_sf').count().alias('n_sf_measurements'),
            pl.col('device_category').sort_by('concurrent_sf').first().alias('device_at_worst_sf'),
            pl.col('hours_from_onset').sort_by('concurrent_sf').first().alias('hours_to_worst_sf'),
        ])

    # --- SF contemporary to worst PF ---
    sf_contemp = pl.DataFrame(schema={id_col: pl.Utf8})
    if pf_agg.height > 0 and sf_df.height > 0 and 'worst_pf_dttm' in pf_agg.columns:
        worst_pf_times = pf_agg.select([id_col, 'worst_pf_dttm'])
        sf_with_pf_time = sf_df.join(worst_pf_times, on=id_col, how='inner')
        sf_with_pf_time = sf_with_pf_time.with_columns([
            (pl.col('recorded_dttm') - pl.col('worst_pf_dttm')).dt.total_seconds().abs().alias('time_diff_secs')
        ])
        # Keep only SF within 4 hours of worst PF
        sf_with_pf_time = sf_with_pf_time.filter(
            pl.col('time_diff_secs') <= (4 * 3600)
        )
        if sf_with_pf_time.height > 0:
            # Take the closest SF to worst PF per encounter
            sf_contemp = sf_with_pf_time.sort([id_col, 'time_diff_secs']).group_by(id_col).first()
            sf_contemp = sf_contemp.select([
                id_col,
                pl.col('concurrent_sf').alias('sf_contemporary_to_pf'),
                pl.col('spo2').alias('sf_contemporary_spo2'),
                pl.col('fio2_set').alias('sf_contemporary_fio2'),
            ])

    # --- Combine all ---
    result = onset_pl.clone()
    if pf_agg.height > 0:
        pf_agg_out = pf_agg.drop('worst_pf_dttm') if 'worst_pf_dttm' in pf_agg.columns else pf_agg
        result = result.join(pf_agg_out, on=id_col, how='left')
    if sf_agg.height > 0:
        result = result.join(sf_agg, on=id_col, how='left')
    if sf_contemp.height > 0:
        result = result.join(sf_contemp, on=id_col, how='left')

    # Assign exposure group
    has_pf = pl.col('pf_24_min').is_not_null() if 'pf_24_min' in result.columns else pl.lit(False)
    has_sf = pl.col('sf_24_min').is_not_null() if 'sf_24_min' in result.columns else pl.lit(False)

    result = result.with_columns([
        pl.when(has_pf & has_sf).then(pl.lit('sf_pf'))
        .when(has_pf & ~has_sf).then(pl.lit('pf_only'))
        .when(~has_pf & has_sf).then(pl.lit('sf_only'))
        .otherwise(pl.lit('undefined_oxygenation'))
        .alias('exposure_group')
    ])

    return result.to_pandas()


def calculate_pf_sf_ratios(
    onset_df: pd.DataFrame,
    data_directory: str,
    filetype: str,
    timezone: str,
    id_col: str = 'encounter_block'
) -> pd.DataFrame:
    """
    Calculate PF and SF ratios for the first 24h after respiratory failure onset.

    Parameters
    ----------
    onset_df : pd.DataFrame
        From detect_respiratory_failure_onset(). Must have:
        [id_col, 'hospitalization_id', 'onset_dttm', 'onset_device']
    data_directory : str
        Path to CLIF data files.
    filetype : str
        File format (parquet, csv).
    timezone : str
        Site timezone.
    id_col : str
        Grouping column.

    Returns
    -------
    pd.DataFrame
        One row per encounter with PF/SF summary statistics.
    """
    if onset_df.empty:
        logger.warning("No onset data provided — returning empty DataFrame")
        return pd.DataFrame()

    logger.info(f"Computing PF/SF ratios for {len(onset_df):,} encounters...")

    # Build cohort_df for SOFA loaders: needs hospitalization_id, start_dttm, end_dttm
    cohort_pd = onset_df[['hospitalization_id', 'onset_dttm']].copy()
    cohort_pd['start_dttm'] = cohort_pd['onset_dttm']
    cohort_pd['end_dttm'] = cohort_pd['onset_dttm'] + timedelta(hours=24)
    cohort_pd = cohort_pd.drop(columns=['onset_dttm'])

    cohort_pl = pl.from_pandas(cohort_pd)
    # Standardize datetime columns
    cohort_pl = standardize_datetime_columns(
        cohort_pl,
        target_timezone=timezone,
        target_time_unit='ns',
        datetime_columns=['start_dttm', 'end_dttm']
    )

    hosp_ids = onset_df['hospitalization_id'].unique().tolist()

    # --- Load data for the 24h window ---
    logger.info("Loading respiratory support data...")
    resp_lf = _load_respiratory_support(
        data_directory, filetype, hosp_ids, cohort_pl,
        lookback_hours=24, timezone=timezone
    )
    resp_df = resp_lf.collect()
    logger.info(f"  Respiratory support: {resp_df.height:,} rows")

    logger.info("Loading labs data (po2_arterial)...")
    labs_lf = _load_labs(data_directory, filetype, hosp_ids, cohort_pl, timezone=timezone)
    labs_df = labs_lf.collect()
    logger.info(f"  Labs: {labs_df.height:,} rows")

    logger.info("Loading vitals data (spo2)...")
    vitals_lf = _load_vitals(data_directory, filetype, hosp_ids, cohort_pl, timezone=timezone)
    vitals_df = vitals_lf.collect()
    logger.info(f"  Vitals: {vitals_df.height:,} rows")

    # --- Map hospitalization_id to encounter_block for grouping ---
    h2e = pl.from_pandas(onset_df[['hospitalization_id', id_col]]).unique()

    # --- PF ratio ---
    logger.info("Computing PF ratios...")
    # Extract PO2 from labs (long format)
    labs_po2 = labs_df.filter(
        (pl.col('lab_category') == 'po2_arterial') &
        pl.col('lab_value_numeric').is_not_null() &
        (pl.col('lab_value_numeric') >= PAO2_MIN) &
        (pl.col('lab_value_numeric') <= PAO2_MAX)
    ).select([
        'hospitalization_id', 'lab_result_dttm',
        pl.col('lab_value_numeric').alias('po2_arterial')
    ])

    pf_df = pl.DataFrame()
    if labs_po2.height > 0 and resp_df.height > 0:
        pf_df = _calculate_concurrent_pf_ratios(
            labs_po2, resp_df,
            time_tolerance_minutes=240,
            id_cols=['hospitalization_id']
        )
        # Map to encounter_block
        if pf_df.height > 0:
            pf_df = pf_df.join(h2e, on='hospitalization_id', how='left')
    logger.info(f"  PF measurements: {pf_df.height:,}")

    # --- SF ratio ---
    logger.info("Computing SF ratios...")
    # Extract SpO2 from vitals (long format)
    vitals_spo2 = vitals_df.filter(
        (pl.col('vital_category') == 'spo2') &
        pl.col('vital_value').is_not_null()
    ).select([
        'hospitalization_id', 'recorded_dttm',
        pl.col('vital_value').alias('spo2')
    ])

    sf_df = pl.DataFrame()
    if vitals_spo2.height > 0 and resp_df.height > 0:
        sf_df = _calculate_concurrent_sf_ratios(
            vitals_spo2, resp_df,
            time_tolerance_minutes=240,
            id_cols=['hospitalization_id']
        )
        # Map to encounter_block
        if sf_df.height > 0:
            sf_df = sf_df.join(h2e, on='hospitalization_id', how='left')
    logger.info(f"  SF measurements: {sf_df.height:,}")

    # --- Aggregate per encounter ---
    logger.info("Aggregating per encounter...")
    result = _aggregate_pf_sf_per_encounter(pf_df, sf_df, onset_df, id_col=id_col)

    logger.info(f"PF/SF calculation complete: {len(result):,} encounters")
    exposure_counts = result['exposure_group'].value_counts()
    for grp, count in exposure_counts.items():
        logger.info(f"  {grp}: {count:,}")

    return result


def generate_aggregate_stats(
    per_encounter_df: pd.DataFrame,
    id_col: str = 'encounter_block'
) -> pd.DataFrame:
    """
    Compute cohort-level summary statistics from per-encounter PF/SF data.

    Returns a long-format DataFrame with one row per (variable, stratum) pair.
    """
    if per_encounter_df.empty:
        return pd.DataFrame(columns=['variable', 'stratum', 'n', 'mean', 'sd', 'median', 'q25', 'q75'])

    metrics = [
        'pf_24_min', 'pf_24_mean', 'pf_24_median',
        'sf_24_min', 'sf_24_mean', 'sf_24_median',
        'sf_contemporary_to_pf',
        'hours_to_worst_pf', 'hours_to_worst_sf',
    ]

    rows = []

    def _summarize(data, variable, stratum):
        vals = data[variable].dropna()
        if len(vals) == 0:
            return
        rows.append({
            'variable': variable,
            'stratum': stratum,
            'n': len(vals),
            'mean': vals.mean(),
            'sd': vals.std(),
            'median': vals.median(),
            'q25': vals.quantile(0.25),
            'q75': vals.quantile(0.75),
        })

    # Overall
    for m in metrics:
        if m in per_encounter_df.columns:
            _summarize(per_encounter_df, m, 'Overall')

    # By onset device
    if 'onset_device' in per_encounter_df.columns:
        for device, grp in per_encounter_df.groupby('onset_device'):
            for m in metrics:
                if m in grp.columns:
                    _summarize(grp, m, f'device:{device}')

    stats_df = pd.DataFrame(rows)

    # Add exposure group counts
    if 'exposure_group' in per_encounter_df.columns:
        counts = per_encounter_df['exposure_group'].value_counts()
        for grp, n in counts.items():
            stats_df = pd.concat([stats_df, pd.DataFrame([{
                'variable': 'exposure_group_count',
                'stratum': grp,
                'n': n, 'mean': np.nan, 'sd': np.nan,
                'median': np.nan, 'q25': np.nan, 'q75': np.nan,
            }])], ignore_index=True)

    return stats_df


def generate_pf_sf_comparison_figure(
    strata_data: dict,
    output_path: str,
) -> None:
    """
    Create a grouped box plot comparing PF and SF distributions across strata.

    Parameters
    ----------
    strata_data : dict
        Mapping of stratum label to per-encounter DataFrame.
        E.g. {'Overall': df_all, 'ICU': df_icu, 'No ICU': df_noicu}
    output_path : str
        Path to save the PNG figure.
    """
    import matplotlib.pyplot as plt
    import os

    metrics = [
        ('pf_24_min', 'Worst PF Ratio'),
        ('sf_24_min', 'Worst SF Ratio'),
        ('pf_24_median', 'Median PF Ratio'),
        ('sf_24_median', 'Median SF Ratio'),
    ]

    # Filter to metrics that have data in at least one stratum
    available = []
    for col, label in metrics:
        for df in strata_data.values():
            if col in df.columns and df[col].notna().sum() > 0:
                available.append((col, label))
                break

    if not available:
        logger.warning("No PF/SF data available for comparison figure")
        return

    n_metrics = len(available)
    strata_labels = list(strata_data.keys())
    colors = {'Overall': '#636EFA', 'ICU': '#EF553B', 'No ICU': '#00CC96'}

    fig, axes = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, 6), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    for ax, (metric_col, title) in zip(axes, available):
        box_data = []
        box_labels = []
        box_colors = []
        for label in strata_labels:
            df = strata_data[label]
            if metric_col in df.columns:
                vals = df[metric_col].dropna()
                if len(vals) > 0:
                    box_data.append(vals.values)
                    box_labels.append(f"{label}\n(n={len(vals):,})")
                    box_colors.append(colors.get(label, '#AB63FA'))

        if not box_data:
            ax.set_visible(False)
            continue

        bp = ax.boxplot(
            box_data,
            labels=box_labels,
            patch_artist=True,
            showmeans=True,
            meanprops=dict(marker='D', markerfacecolor='white', markeredgecolor='black', markersize=5),
            medianprops=dict(color='black', linewidth=1.5),
            whiskerprops=dict(color='gray'),
            capprops=dict(color='gray'),
            flierprops=dict(marker='.', markersize=2, alpha=0.3),
        )
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(axis='x', labelsize=9)

    fig.suptitle('PF/SF Ratio Comparison: Overall vs ICU vs No ICU\n(First 24h of Respiratory Failure)',
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved PF/SF comparison figure: {output_path}")
