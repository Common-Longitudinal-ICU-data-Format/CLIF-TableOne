"""
Plots for ventilated aggregates (CLIF-TableOne issue #10):
  1. Kaplan-Meier survival curve for time to extubation (2 panels:
     overall + stratified by ICU vs no-ICU).
  2. Min P/F and S/F ratio per day post-intubation (2 panels, 28-day
     horizon, median + shaded IQR band).

Each plot also writes an aggregate CSV (no PHI) with the exact numbers
used to render it, so sites can share the CSVs and a downstream step
can overlay multiple sites' curves on one plot.
"""

from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

MAX_DAYS_KM = 28


# ─────────────────────────────────────────────────────────────────────────
# KM plot + aggregate CSV
# ─────────────────────────────────────────────────────────────────────────
def plot_km_time_to_extubation(
    tableone_df: pd.DataFrame,
    figures_dir: str,
    csv_dir: str | None = None,
    max_days: int = MAX_DAYS_KM,
) -> tuple[Path | None, Path | None]:
    """
    Two-panel KM (overall + ICU vs no-ICU) + aggregate CSV.

    Event = extubation. Censoring:
      - discharged_on_imv → censored at discharge_dttm
      - death_on_imv      → censored at death_dttm (fallback discharge_dttm)
      - unknown           → censored at last known timestamp
    Pre-admission IMV encounters are excluded (true intubation time unknown).
    Observations beyond `max_days` are right-censored at the window boundary.

    CSV columns: stratum, timeline_days, survival_prob, ci_lower, ci_upper,
                 at_risk, observed_events.
    """
    from lifelines import KaplanMeierFitter

    required = [
        'intubation_start_dttm', 'extubation_end_dttm',
        'extubation_status', 'pre_admission_imv',
        'death_dttm', 'discharge_dttm',
    ]
    missing = [c for c in required if c not in tableone_df.columns]
    if missing:
        print(f"  ⚠️ KM plot skipped — missing columns: {missing}")
        return None, None

    df = tableone_df[
        tableone_df['intubation_start_dttm'].notna()
        & (tableone_df['pre_admission_imv'] == 0)
    ].copy()
    if len(df) == 0:
        print("  ⚠️ KM plot skipped — no eligible IMV encounters")
        return None, None

    df['_event'] = (df['extubation_status'] == 'extubated').astype(int)
    _extub_hrs = (
        (df['extubation_end_dttm'] - df['intubation_start_dttm'])
        .dt.total_seconds() / 3600.0
    )
    _censor_end = df['death_dttm'].fillna(df['discharge_dttm'])
    _censor_hrs = (
        (_censor_end - df['intubation_start_dttm'])
        .dt.total_seconds() / 3600.0
    )
    df['_duration_hrs'] = np.where(df['_event'] == 1, _extub_hrs, _censor_hrs)
    df = df[df['_duration_hrs'].notna() & (df['_duration_hrs'] >= 0)]
    if len(df) == 0:
        print("  ⚠️ KM plot skipped — no valid durations")
        return None, None

    max_hrs = max_days * 24
    beyond = df['_duration_hrs'] > max_hrs
    df.loc[beyond, '_event'] = 0
    df.loc[beyond, '_duration_hrs'] = max_hrs
    df['_duration_days'] = df['_duration_hrs'] / 24.0

    strata_masks = [('overall', pd.Series(True, index=df.index))]
    if 'high_support_icu_enc' in df.columns:
        strata_masks.append(('icu', df['high_support_icu_enc'] == 1))
    if 'high_support_no_icu_enc' in df.columns:
        strata_masks.append(('no_icu', df['high_support_no_icu_enc'] == 1))

    # Fit KM per stratum, collect agg rows, and prepare plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    agg_rows = []
    stratum_colors = {'overall': 'black', 'icu': 'tab:blue', 'no_icu': 'tab:orange'}

    for name, mask in strata_masks:
        sub = df[mask]
        if len(sub) == 0:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(sub['_duration_days'], sub['_event'],
                label=f'{name} (n={len(sub):,})')

        ax = axes[0] if name == 'overall' else axes[1]
        kmf.plot_survival_function(ax=ax, ci_show=True,
                                   color=stratum_colors.get(name, None))

        # Build aggregate rows from lifelines internals
        sf = kmf.survival_function_.iloc[:, 0]  # single column
        ci = kmf.confidence_interval_
        et = kmf.event_table  # columns: removed, observed, censored, at_risk, ...
        for t in sf.index:
            agg_rows.append({
                'stratum': name,
                'timeline_days': float(t),
                'survival_prob': float(sf.loc[t]),
                'ci_lower': float(ci.loc[t].iloc[0]) if t in ci.index else np.nan,
                'ci_upper': float(ci.loc[t].iloc[1]) if t in ci.index else np.nan,
                'at_risk': int(et.loc[t, 'at_risk']) if t in et.index else 0,
                'observed_events': int(et.loc[t, 'observed']) if t in et.index else 0,
            })

    _format_km_axis(axes[0], 'Time to extubation (overall)', max_days)
    _format_km_axis(axes[1], 'Time to extubation by ICU exposure', max_days)
    plt.tight_layout()

    png_path = Path(figures_dir) / 'km_time_to_extubation.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close('all')
    print(f"  ✅ KM plot saved: {png_path}")

    csv_path = None
    if agg_rows:
        csv_dir_p = Path(csv_dir) if csv_dir else png_path.parent
        csv_dir_p.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir_p / 'km_time_to_extubation.csv'
        pd.DataFrame(agg_rows).to_csv(csv_path, index=False)
        print(f"  ✅ KM aggregate CSV: {csv_path}")

    return png_path, csv_path


def _format_km_axis(ax, title, max_days):
    ax.set_title(title)
    ax.set_xlabel('Days since intubation')
    ax.set_ylabel('Probability still on IMV')
    ax.set_xlim(0, max_days)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)


# ─────────────────────────────────────────────────────────────────────────
# Daily min P/F + S/F plot + aggregate CSV
# ─────────────────────────────────────────────────────────────────────────
def plot_min_pf_sf_per_day_post_intubation(
    tableone_df: pd.DataFrame,
    data_directory: str,
    filetype: str,
    timezone: str,
    figures_dir: str,
    csv_dir: str | None = None,
    n_days: int = 28,
    resp_waterfall_df: pd.DataFrame | None = None,
) -> tuple[Path | None, Path | None]:
    """
    Two-panel plot of daily min P/F and S/F ratio for n_days post intubation,
    plus an aggregate CSV.

    For each IMV encounter with a detected intubation:
      1. Load labs (PaO2), vitals (SpO2), respiratory_support (FiO2) restricted
         to [intubation_start, intubation_start + n_days].
      2. Use concurrent PF/SF calculation (4-hour FiO2 lookback).
      3. Bucket observations by day-since-intubation.
      4. Take min ratio per encounter per day.
      5. Aggregate across encounters per day → median + Q1 + Q3 + n.

    CSV columns: ratio_type ∈ {PF, SF}, day, median, q1, q3, n_encounters.

    If ``resp_waterfall_df`` is provided (a pandas DataFrame of post-waterfall
    respiratory_support data), it is used directly and the per-year raw-parquet
    rescan + waterfall recompute is skipped. At large sites (JHU: 10 years,
    8.6M-row waterfall) this avoids ~10 full parquet scans + waterfall passes.
    """
    from modules.sofa.calculator import (
        _calculate_concurrent_pf_ratios,
        _load_labs, _load_vitals, _load_respiratory_support,
    )
    from modules.tableone.pf_sf_calculator import (
        _calculate_concurrent_sf_ratios,
        PAO2_MIN, PAO2_MAX, PF_CAP,
    )
    from modules.utils.datetime_utils import standardize_datetime_columns

    required = [
        'hospitalization_id', 'intubation_start_dttm',
        'pre_admission_imv', 'encounter_block',
    ]
    missing = [c for c in required if c not in tableone_df.columns]
    if missing:
        print(f"  ⚠️ Daily PF/SF plot skipped — missing columns: {missing}")
        return None, None

    intub = tableone_df[
        tableone_df['intubation_start_dttm'].notna()
        & (tableone_df['pre_admission_imv'] == 0)
    ][['hospitalization_id', 'encounter_block', 'intubation_start_dttm']].copy()
    if len(intub) == 0:
        print("  ⚠️ Daily PF/SF plot skipped — no eligible IMV encounters")
        return None, None

    cohort_pd = intub[['hospitalization_id', 'intubation_start_dttm']].copy()
    cohort_pd['start_dttm'] = cohort_pd['intubation_start_dttm']
    cohort_pd['end_dttm'] = (
        cohort_pd['intubation_start_dttm'] + timedelta(days=n_days)
    )
    cohort_pd = cohort_pd.drop(columns=['intubation_start_dttm'])

    cohort_pl = standardize_datetime_columns(
        pl.from_pandas(cohort_pd),
        target_timezone=timezone,
        target_time_unit='ns',
        datetime_columns=['start_dttm', 'end_dttm'],
    )
    hosp_ids = intub['hospitalization_id'].unique().tolist()

    print(f"  Loading data for {len(hosp_ids):,} hospitalizations "
          f"({n_days}d window)...")
    if resp_waterfall_df is not None and len(resp_waterfall_df) > 0:
        # Use the cached post-waterfall data instead of rescanning the raw
        # parquet + re-running waterfall. The cached df was produced earlier
        # by the same waterfall logic over the full cohort; filtering it to
        # this call's IMV cohort + [intubation, intubation + n_days] window
        # is bit-equivalent to re-running _load_respiratory_support.
        _hosp_set = set(str(h) for h in hosp_ids)
        _wf = resp_waterfall_df.copy()
        _wf['hospitalization_id'] = _wf['hospitalization_id'].astype(str)
        _wf = _wf[_wf['hospitalization_id'].isin(_hosp_set)].copy()
        # Normalize recorded_dttm to target tz so the time-window comparison
        # below is well-defined. The cached waterfall is typically UTC-aware
        # (DuckDB convention); intubation_start_dttm may be in site tz.
        _wf['recorded_dttm'] = pd.to_datetime(_wf['recorded_dttm'])
        if _wf['recorded_dttm'].dt.tz is None:
            _wf['recorded_dttm'] = _wf['recorded_dttm'].dt.tz_localize('UTC')
        _wf['recorded_dttm'] = _wf['recorded_dttm'].dt.tz_convert(timezone)

        _intub_window = intub[['hospitalization_id', 'intubation_start_dttm']].copy()
        _intub_window['hospitalization_id'] = _intub_window['hospitalization_id'].astype(str)
        _intub_window['intubation_start_dttm'] = pd.to_datetime(
            _intub_window['intubation_start_dttm']
        )
        if _intub_window['intubation_start_dttm'].dt.tz is None:
            _intub_window['intubation_start_dttm'] = (
                _intub_window['intubation_start_dttm'].dt.tz_localize(timezone)
            )
        else:
            _intub_window['intubation_start_dttm'] = (
                _intub_window['intubation_start_dttm'].dt.tz_convert(timezone)
            )
        _intub_window['start_dttm'] = _intub_window['intubation_start_dttm']
        _intub_window['end_dttm'] = (
            _intub_window['intubation_start_dttm'] + timedelta(days=n_days)
        )

        _filt = _wf.merge(
            _intub_window[['hospitalization_id', 'start_dttm', 'end_dttm']],
            on='hospitalization_id', how='inner',
        )
        _filt = _filt[
            (_filt['recorded_dttm'] >= _filt['start_dttm'])
            & (_filt['recorded_dttm'] <= _filt['end_dttm'])
        ].drop(columns=['start_dttm', 'end_dttm'])
        resp_df = standardize_datetime_columns(
            pl.from_pandas(_filt),
            target_timezone=timezone,
            target_time_unit='ns',
            datetime_columns=['recorded_dttm'],
        )
        print(f"  Using cached waterfall: {resp_df.height:,} rows in {n_days}d window")
        del _wf, _filt, _intub_window, _hosp_set
    else:
        resp_df = _load_respiratory_support(
            data_directory, filetype, hosp_ids, cohort_pl,
            lookback_hours=n_days * 24, timezone=timezone,
        ).collect()
    # Replace _load_labs / _load_vitals (polars scan_parquet + is_in filter on
    # 109M-row labs / 383M-row vitals) with a DuckDB-backed scan that pushes
    # the hosp_id AND category filter into the parquet read. The polars path
    # silently aborts at JHU after standardize_datetime_columns — same family
    # of polars-on-large-parquet crash we fixed for SOFA. The 28-day window
    # only needs po2_arterial (labs) and spo2 (vitals), so scope the read
    # tightly: a few MB instead of a full-file scan.
    from modules.utils.clif_loader import load_filtered_clif_table

    def _load_via_duckdb_in_window(
        file_path, columns, category_col, category_value,
        dttm_col, hosp_ids, intub_df, timezone, n_days,
    ):
        df_pd = load_filtered_clif_table(
            file_path,
            columns=columns,
            hosp_ids=hosp_ids,
            category_column=category_col,
            category_values=[category_value],
            return_as='pandas',
        )
        if len(df_pd) == 0:
            return pl.DataFrame(schema={c: pl.Utf8 for c in columns})
        df_pd['hospitalization_id'] = df_pd['hospitalization_id'].astype(str)
        df_pd[dttm_col] = pd.to_datetime(df_pd[dttm_col])
        if df_pd[dttm_col].dt.tz is None:
            df_pd[dttm_col] = df_pd[dttm_col].dt.tz_localize('UTC')
        df_pd[dttm_col] = df_pd[dttm_col].dt.tz_convert(timezone)

        win = intub_df[['hospitalization_id', 'intubation_start_dttm']].copy()
        win['hospitalization_id'] = win['hospitalization_id'].astype(str)
        win['intubation_start_dttm'] = pd.to_datetime(win['intubation_start_dttm'])
        if win['intubation_start_dttm'].dt.tz is None:
            win['intubation_start_dttm'] = win['intubation_start_dttm'].dt.tz_localize(timezone)
        else:
            win['intubation_start_dttm'] = win['intubation_start_dttm'].dt.tz_convert(timezone)
        win['start_dttm'] = win['intubation_start_dttm']
        win['end_dttm'] = win['intubation_start_dttm'] + timedelta(days=n_days)

        df_pd = df_pd.merge(
            win[['hospitalization_id', 'start_dttm', 'end_dttm']],
            on='hospitalization_id', how='inner',
        )
        df_pd = df_pd[
            (df_pd[dttm_col] >= df_pd['start_dttm'])
            & (df_pd[dttm_col] <= df_pd['end_dttm'])
        ].drop(columns=['start_dttm', 'end_dttm'])
        # Standardize datetime precision/tz to match the cached-waterfall
        # resp_df above (ns + site tz). Without this, DuckDB's μs/UTC sneaks
        # through and the downstream join_asof against resp_df fails with
        # "datatypes of join keys don't match".
        out = pl.from_pandas(df_pd)
        out = standardize_datetime_columns(
            out,
            target_timezone=timezone,
            target_time_unit='ns',
            datetime_columns=[dttm_col],
        )
        return out

    labs_df = _load_via_duckdb_in_window(
        str(Path(data_directory) / f"clif_labs.{filetype}"),
        columns=['hospitalization_id', 'lab_result_dttm', 'lab_category', 'lab_value_numeric'],
        category_col='lab_category', category_value='po2_arterial',
        dttm_col='lab_result_dttm', hosp_ids=hosp_ids,
        intub_df=intub, timezone=timezone, n_days=n_days,
    )
    vitals_df = _load_via_duckdb_in_window(
        str(Path(data_directory) / f"clif_vitals.{filetype}"),
        columns=['hospitalization_id', 'recorded_dttm', 'vital_category', 'vital_value'],
        category_col='vital_category', category_value='spo2',
        dttm_col='recorded_dttm', hosp_ids=hosp_ids,
        intub_df=intub, timezone=timezone, n_days=n_days,
    )
    print(f"    resp={resp_df.height:,}  labs={labs_df.height:,}  "
          f"vitals={vitals_df.height:,}")

    labs_po2 = labs_df.filter(
        (pl.col('lab_category') == 'po2_arterial')
        & pl.col('lab_value_numeric').is_not_null()
        & (pl.col('lab_value_numeric') >= PAO2_MIN)
        & (pl.col('lab_value_numeric') <= PAO2_MAX)
    ).select([
        'hospitalization_id', 'lab_result_dttm',
        pl.col('lab_value_numeric').alias('po2_arterial'),
    ])
    vitals_spo2 = vitals_df.filter(
        (pl.col('vital_category') == 'spo2')
        & pl.col('vital_value').is_not_null()
    ).select([
        'hospitalization_id', 'recorded_dttm',
        pl.col('vital_value').alias('spo2'),
    ])

    pf_df = (
        _calculate_concurrent_pf_ratios(
            labs_po2, resp_df, time_tolerance_minutes=240,
            id_cols=['hospitalization_id'],
        )
        if labs_po2.height > 0 and resp_df.height > 0
        else pl.DataFrame()
    )
    sf_df = (
        _calculate_concurrent_sf_ratios(
            vitals_spo2, resp_df, time_tolerance_minutes=240,
            id_cols=['hospitalization_id'],
        )
        if vitals_spo2.height > 0 and resp_df.height > 0
        else pl.DataFrame()
    )
    print(f"    PF measurements: {pf_df.height:,}  "
          f"SF measurements: {sf_df.height:,}")

    pf_agg = _daily_min_aggregate(
        pf_df, intub, ratio_col='concurrent_pf',
        obs_dttm_col='lab_result_dttm', cap=PF_CAP, n_days=n_days,
    )
    sf_agg = _daily_min_aggregate(
        sf_df, intub, ratio_col='concurrent_sf',
        obs_dttm_col='recorded_dttm', cap=None, n_days=n_days,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    _plot_daily_ratio(axes[0], pf_agg, 'P/F (PaO\u2082 / FiO\u2082)',
                      color='tab:blue', hline_refs=[100, 200, 300])
    _plot_daily_ratio(axes[1], sf_agg, 'S/F (SpO\u2082 / FiO\u2082)',
                      color='tab:green', hline_refs=[150, 235, 315])
    plt.suptitle(f'Daily minimum oxygenation post-intubation '
                 f'(median + IQR band, {n_days}-day window)',
                 fontsize=12, y=1.02)
    plt.tight_layout()

    png_path = Path(figures_dir) / 'min_pf_sf_per_day_post_intubation.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close('all')
    print(f"  ✅ Daily PF/SF plot saved: {png_path}")

    # Long-format aggregate CSV
    csv_dir_p = Path(csv_dir) if csv_dir else png_path.parent
    csv_dir_p.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir_p / 'min_pf_sf_per_day_post_intubation.csv'
    out_rows = []
    for label, agg in [('PF', pf_agg), ('SF', sf_agg)]:
        for _, r in agg.iterrows():
            out_rows.append({
                'ratio_type': label,
                'day': int(r['day']),
                'median': float(r['median']),
                'q1': float(r['q1']),
                'q3': float(r['q3']),
                'n_encounters': int(r['n']),
            })
    pd.DataFrame(out_rows).to_csv(csv_path, index=False)
    print(f"  ✅ Daily PF/SF aggregate CSV: {csv_path}")

    return png_path, csv_path


def _daily_min_aggregate(
    ratio_df: pl.DataFrame,
    intub_df: pd.DataFrame,
    ratio_col: str,
    obs_dttm_col: str,
    cap: float | None,
    n_days: int,
) -> pd.DataFrame:
    """Min ratio per encounter per day → median/Q1/Q3/n across encounters per day."""
    if ratio_df.height == 0:
        return pd.DataFrame(columns=['day', 'median', 'q1', 'q3', 'n'])

    r = ratio_df.to_pandas()
    cols = [c for c in ['hospitalization_id', obs_dttm_col, ratio_col] if c in r.columns]
    r = r[cols].rename(columns={obs_dttm_col: 'obs_dttm', ratio_col: 'ratio'})
    r = r[r['ratio'].notna()]
    if cap is not None:
        r['ratio'] = r['ratio'].clip(upper=cap)

    m = r.merge(
        intub_df[['hospitalization_id', 'encounter_block',
                  'intubation_start_dttm']],
        on='hospitalization_id', how='inner',
    )
    # Align tz before subtraction
    if (pd.api.types.is_datetime64_any_dtype(m['obs_dttm'])
            and pd.api.types.is_datetime64_any_dtype(m['intubation_start_dttm'])):
        obs_tz = getattr(m['obs_dttm'].dt, 'tz', None)
        intub_tz = getattr(m['intubation_start_dttm'].dt, 'tz', None)
        if obs_tz is None and intub_tz is not None:
            m['obs_dttm'] = m['obs_dttm'].dt.tz_localize(intub_tz)
        elif intub_tz is None and obs_tz is not None:
            m['intubation_start_dttm'] = (
                m['intubation_start_dttm'].dt.tz_localize(obs_tz)
            )

    delta_s = (m['obs_dttm'] - m['intubation_start_dttm']).dt.total_seconds()
    m['day'] = (delta_s // 86400).astype('Int64')
    m = m[(m['day'] >= 0) & (m['day'] < n_days)]
    if len(m) == 0:
        return pd.DataFrame(columns=['day', 'median', 'q1', 'q3', 'n'])

    daily = (
        m.groupby(['encounter_block', 'day'])['ratio']
        .min().reset_index()
    )
    agg = daily.groupby('day')['ratio'].agg(
        median='median',
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
        n='count',
    ).reset_index()
    agg['day'] = agg['day'].astype(int)
    return agg


def _plot_daily_ratio(ax, agg_df, ylabel, color, hline_refs):
    if agg_df.empty:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Days since intubation')
        return

    ax.plot(agg_df['day'], agg_df['median'], color=color,
            marker='o', linewidth=2, label='Median')
    ax.fill_between(agg_df['day'], agg_df['q1'], agg_df['q3'],
                    color=color, alpha=0.2, label='IQR (Q1–Q3)')

    for y in hline_refs:
        ax.axhline(y=y, color='gray', linestyle='--', linewidth=0.7, alpha=0.5)

    ax2 = ax.twinx()
    ax2.bar(agg_df['day'], agg_df['n'], alpha=0.15, color=color,
            width=0.9, label='n encounters')
    ax2.set_ylabel('n encounters contributing', fontsize=9)
    ax2.tick_params(axis='y', labelsize=8)

    ax.set_xlabel('Days since intubation')
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.5, agg_df['day'].max() + 0.5)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
