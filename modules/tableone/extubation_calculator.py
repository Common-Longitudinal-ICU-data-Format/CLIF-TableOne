"""
Intubation / extubation event detection from post-waterfall respiratory support.

Implements the two-lookback / two-lookforward pattern from clifpy issue #124:
https://github.com/Common-Longitudinal-ICU-data-Format/clifpy/issues/124

Within each encounter (sorted by recorded_dttm, with forward-filled
device_category from the waterfall), computes lag(1), lag(2), lead(1),
lead(2) on `is_imv`:

  Intubation: ~lag2 & ~lag1 & current & lead1 & lead2
  Extubation: lag2 & lag1 & current & ~lead1 & ~lead2

Intubation/extubation pairs are then assembled into per-encounter episodes.
Pre-admission IMV (first observation already on IMV) is flagged separately
and excluded from time-to-extubation.
"""

import pandas as pd
import numpy as np


def detect_intubation_extubation(
    resp_support_df: pd.DataFrame,
    encounter_df: pd.DataFrame,
    id_col: str = 'encounter_block',
    failed_attempt_threshold_min: float = 5.0,
    admit_proximity_hr: float = 24.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Detect intubation/extubation events via two-lookback/two-lookforward pattern.

    Args:
        resp_support_df: Post-waterfall respiratory support with columns
            [id_col, recorded_dttm, device_category]. Must already be
            forward-filled (waterfall applied).
        encounter_df: Encounter-level data with columns
            [id_col, admission_dttm, discharge_dttm, death_dttm, death_enc].
        id_col: Encounter identifier column.
        failed_attempt_threshold_min: Episodes shorter than this (minutes)
            are labeled as 'failed_attempt' and excluded from real-episode
            counts and medians.
        admit_proximity_hr: Window after admission_dttm within which a
            detected intubation counts as "intubated near admission".

    Returns:
        (per_encounter, per_episode):
          per_encounter: one row per IMV encounter with extubation metrics.
          per_episode: one row per (real) episode with timestamps + duration.
    """
    required = ['admission_dttm', 'discharge_dttm', 'death_dttm', 'death_enc']
    missing = [c for c in required if c not in encounter_df.columns]
    if missing:
        print(f"  ⚠️ Extubation detection skipped — missing columns: {missing}")
        return (pd.DataFrame(columns=[id_col]),
                pd.DataFrame(columns=[id_col, 'episode_n']))

    resp = resp_support_df[[id_col, 'recorded_dttm', 'device_category']].copy()
    resp = resp.sort_values([id_col, 'recorded_dttm']).reset_index(drop=True)
    resp['is_imv'] = resp['device_category'].str.contains(
        'imv', case=False, na=False
    )

    imv_enc_ids = resp.loc[resp['is_imv'], id_col].unique()
    if len(imv_enc_ids) == 0:
        return (pd.DataFrame(columns=[id_col]),
                pd.DataFrame(columns=[id_col, 'episode_n']))

    print(f"  Detecting intubation/extubation for "
          f"{len(imv_enc_ids):,} IMV encounters ...")

    resp = resp[resp[id_col].isin(imv_enc_ids)].reset_index(drop=True)

    # ── Lag/lead on is_imv within each encounter ─────────────────────
    grp_imv = resp.groupby(id_col, sort=False)['is_imv']
    resp['lag2'] = grp_imv.shift(2)
    resp['lag1'] = grp_imv.shift(1)
    resp['lead1'] = grp_imv.shift(-1)
    resp['lead2'] = grp_imv.shift(-2)
    resp['next_dttm'] = (
        resp.groupby(id_col, sort=False)['recorded_dttm'].shift(-1)
    )

    # NaN (window boundary) treated as NOT satisfying the pattern
    intub_mask = (
        (resp['lag2'] == False) & (resp['lag1'] == False)
        & (resp['is_imv'] == True)
        & (resp['lead1'] == True) & (resp['lead2'] == True)
    )
    extub_mask = (
        (resp['lag2'] == True) & (resp['lag1'] == True)
        & (resp['is_imv'] == True)
        & (resp['lead1'] == False) & (resp['lead2'] == False)
    )

    intubations = (
        resp.loc[intub_mask, [id_col, 'recorded_dttm']]
        .rename(columns={'recorded_dttm': 'intubation_start_dttm'})
        .copy()
    )
    intubations['is_synthetic'] = False

    extubations = (
        resp.loc[extub_mask, [id_col, 'next_dttm']]
        .rename(columns={'next_dttm': 'extubation_end_dttm'})
        .copy()
    )

    # ── Pre-admission IMV: first observation of encounter is already IMV ─
    enc_firsts = (
        resp.groupby(id_col, sort=False)
        .agg(first_dttm=('recorded_dttm', 'first'),
             first_is_imv=('is_imv', 'first'))
        .reset_index()
    )
    pre_admit_ids = set(enc_firsts.loc[enc_firsts['first_is_imv'], id_col])

    # Synthesize a pre-admit intubation at first_dttm when:
    #   - no real intubation was detected, OR
    #   - an extubation occurs before the first detected intubation
    # (so the first IMV run pairs with the first extubation)
    first_intub = intubations.groupby(id_col).agg(
        first_intub_dttm=('intubation_start_dttm', 'min'))
    first_extub = extubations.groupby(id_col).agg(
        first_extub_dttm=('extubation_end_dttm', 'min'))
    pre_admit_df = (
        enc_firsts[enc_firsts['first_is_imv']]
        .merge(first_intub, on=id_col, how='left')
        .merge(first_extub, on=id_col, how='left')
    )
    needs_synthetic = (
        pre_admit_df['first_intub_dttm'].isna()
        | (pre_admit_df['first_extub_dttm'].notna()
           & (pre_admit_df['first_extub_dttm']
              <= pre_admit_df['first_intub_dttm']))
    )
    synthetic_intubs = (
        pre_admit_df.loc[needs_synthetic, [id_col, 'first_dttm']]
        .rename(columns={'first_dttm': 'intubation_start_dttm'})
    )
    synthetic_intubs['is_synthetic'] = True

    # ── Pair intubations ↔ extubations sequentially per encounter ────
    all_intubs = pd.concat(
        [synthetic_intubs, intubations], ignore_index=True
    )
    all_intubs = all_intubs.sort_values(
        [id_col, 'intubation_start_dttm']
    ).reset_index(drop=True)
    all_intubs['episode_n'] = all_intubs.groupby(id_col).cumcount() + 1

    extub_sorted = extubations.sort_values(
        [id_col, 'extubation_end_dttm']
    ).reset_index(drop=True)
    extub_sorted['episode_n'] = extub_sorted.groupby(id_col).cumcount() + 1

    per_episode = all_intubs.merge(
        extub_sorted, on=[id_col, 'episode_n'], how='left'
    )
    per_episode['episode_duration_hours'] = (
        per_episode['extubation_end_dttm']
        - per_episode['intubation_start_dttm']
    ).dt.total_seconds() / 3600.0

    # Flag & exclude failed attempts (near-zero duration)
    per_episode['is_failed_attempt'] = (
        per_episode['episode_duration_hours'].notna()
        & (per_episode['episode_duration_hours'] * 60
           < failed_attempt_threshold_min)
    )

    real_eps = per_episode[~per_episode['is_failed_attempt']].copy()
    real_eps = real_eps.sort_values(
        [id_col, 'intubation_start_dttm']
    ).reset_index(drop=True)
    real_eps['episode_n'] = real_eps.groupby(id_col).cumcount() + 1

    # ── Per-encounter summary ────────────────────────────────────────
    first_ep = (
        real_eps[real_eps['episode_n'] == 1][[
            id_col, 'intubation_start_dttm', 'extubation_end_dttm']]
    )
    second_ep = (
        real_eps[real_eps['episode_n'] == 2][[
            id_col, 'intubation_start_dttm']]
        .rename(columns={'intubation_start_dttm': 'reintubation_dttm'})
    )
    ep_counts = (
        real_eps.groupby(id_col).size()
        .reset_index(name='imv_episodes_n')
    )

    per_enc = pd.DataFrame({id_col: imv_enc_ids})
    per_enc = per_enc.merge(first_ep, on=id_col, how='left')
    per_enc = per_enc.merge(second_ep, on=id_col, how='left')
    per_enc = per_enc.merge(ep_counts, on=id_col, how='left')
    per_enc['imv_episodes_n'] = (
        per_enc['imv_episodes_n'].fillna(0).astype(int)
    )

    per_enc['pre_admission_imv'] = (
        per_enc[id_col].isin(pre_admit_ids).astype(int)
    )

    # Time to extubation — exclude pre-admit (true intub time unknown)
    _delta_hours = (
        (per_enc['extubation_end_dttm']
         - per_enc['intubation_start_dttm'])
        .dt.total_seconds() / 3600.0
    )
    per_enc['time_to_extubation_hours'] = np.where(
        per_enc['pre_admission_imv'] == 1, np.nan, _delta_hours
    )

    # Time to reintubation & 48hr extubation-failure flag
    per_enc['time_to_reintubation_hours'] = (
        (per_enc['reintubation_dttm']
         - per_enc['extubation_end_dttm'])
        .dt.total_seconds() / 3600.0
    )
    per_enc['extubation_failure_48hr'] = np.where(
        per_enc['extubation_end_dttm'].notna(),
        (per_enc['time_to_reintubation_hours'].notna()
         & (per_enc['time_to_reintubation_hours'] <= 48)).astype(int),
        np.nan,
    )

    # Merge encounter info for status + admission-proximity flag
    enc_info = encounter_df[[id_col] + required].drop_duplicates(id_col)
    per_enc = per_enc.merge(enc_info, on=id_col, how='left')

    hrs_from_admit = (
        (per_enc['intubation_start_dttm']
         - per_enc['admission_dttm'])
        .dt.total_seconds() / 3600.0
    )
    per_enc['intubated_within_24hr_admit'] = (
        (per_enc['pre_admission_imv'] == 0)
        & hrs_from_admit.notna()
        & (hrs_from_admit <= admit_proximity_hr)
        & (hrs_from_admit >= -1)  # small slack for clock skew
    ).astype(int)

    # Extubation status
    per_enc['extubation_status'] = 'unknown'
    has_extub = per_enc['extubation_end_dttm'].notna()
    per_enc.loc[has_extub, 'extubation_status'] = 'extubated'

    no_extub = ~has_extub
    per_enc.loc[
        no_extub & (per_enc['death_enc'] == 1), 'extubation_status'
    ] = 'death_on_imv'
    per_enc.loc[
        no_extub & (per_enc['death_enc'] != 1)
        & per_enc['discharge_dttm'].notna(),
        'extubation_status'
    ] = 'discharged_on_imv'

    # Encounters whose only episode was a failed attempt → 'failed_attempt'
    failed_only_ids = set(
        per_episode.loc[per_episode['is_failed_attempt'], id_col]
    )
    failed_only_mask = (
        per_enc[id_col].isin(failed_only_ids)
        & (per_enc['imv_episodes_n'] == 0)
    )
    per_enc.loc[failed_only_mask, 'extubation_status'] = 'failed_attempt'

    # ── Diagnostics ──────────────────────────────────────────────────
    print("  Extubation status breakdown:")
    for k, v in per_enc['extubation_status'].value_counts().items():
        print(f"    {k}: {v:,}")
    print(f"  Pre-admit IMV: {per_enc['pre_admission_imv'].sum():,}")
    print(f"  Reintubation (≥2 real episodes): "
          f"{(per_enc['imv_episodes_n'] >= 2).sum():,}")
    _t = per_enc.loc[
        per_enc['extubation_status'] == 'extubated',
        'time_to_extubation_hours'
    ].dropna()
    if len(_t) > 0:
        print(f"  Time to extubation (hrs) median [Q1, Q3]: "
              f"{_t.median():.1f} "
              f"[{_t.quantile(.25):.1f}, {_t.quantile(.75):.1f}]")

    per_enc_out = per_enc[[
        id_col, 'intubation_start_dttm', 'extubation_end_dttm',
        'time_to_extubation_hours',
        'reintubation_dttm', 'time_to_reintubation_hours',
        'extubation_failure_48hr',
        'imv_episodes_n', 'pre_admission_imv',
        'intubated_within_24hr_admit', 'extubation_status',
    ]].copy()

    per_ep_out = real_eps[[
        id_col, 'episode_n',
        'intubation_start_dttm', 'extubation_end_dttm',
        'episode_duration_hours', 'is_synthetic',
    ]].copy()

    return per_enc_out, per_ep_out
