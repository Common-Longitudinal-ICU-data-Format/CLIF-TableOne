"""
28-day Ventilator-Free Days (VFD) calculator.

Computes VFDs for encounters with invasive mechanical ventilation (IMV).

Logic per encounter:
  window = [vent_start_dttm, vent_start_dttm + 27 days]
  1. Death within window → VFD = 0
     (uses death_dttm, falls back to discharge_dttm when death_enc == 1)
  2. Find last day on IMV within window (from episode boundaries)
  3. Still on IMV at day 28 → VFD = 0
  4. Otherwise: VFD = (window_end_day − last_imv_day)
  Reintubation: intermediate free days do NOT count.
"""

import pandas as pd
import numpy as np


def calculate_ventilator_free_days(
    resp_support_df: pd.DataFrame,
    encounter_df: pd.DataFrame,
    id_col: str = 'encounter_block',
) -> pd.DataFrame:
    """
    Calculate 28-day ventilator-free days for encounters with IMV.

    Args:
        resp_support_df: Respiratory support observations with columns:
            [id_col, recorded_dttm, device_category]
        encounter_df: Encounter-level data with columns:
            [id_col, on_vent, vent_start_dttm, death_dttm, death_enc,
             discharge_dttm]
        id_col: Encounter identifier column.

    Returns:
        DataFrame with columns [id_col, vfd_28].
        Only rows for encounters that had IMV (on_vent == 1).
    """
    required = ['on_vent', 'vent_start_dttm', 'death_dttm', 'death_enc',
                'discharge_dttm']
    missing = [c for c in required if c not in encounter_df.columns]
    if missing:
        print(f"  ⚠️ VFD skipped — missing columns: {missing}")
        return pd.DataFrame(columns=[id_col, 'vfd_28'])

    # Defensive: tzdata-broken hosts can present datetime columns as object
    # dtype. Coerce inputs so downstream .dt accessors work. Non-mutating —
    # uses assign() to avoid side effects on caller's DataFrames.
    _dt_cols = ('vent_start_dttm', 'death_dttm', 'discharge_dttm')
    if any(
        not pd.api.types.is_datetime64_any_dtype(encounter_df[c])
        for c in _dt_cols
    ):
        encounter_df = encounter_df.assign(**{
            c: pd.to_datetime(encounter_df[c], errors='coerce', utc=True)
            for c in _dt_cols
            if not pd.api.types.is_datetime64_any_dtype(encounter_df[c])
        })
    if not pd.api.types.is_datetime64_any_dtype(resp_support_df['recorded_dttm']):
        resp_support_df = resp_support_df.assign(
            recorded_dttm=pd.to_datetime(
                resp_support_df['recorded_dttm'], errors='coerce', utc=True
            )
        )

    # ── Filter to IMV encounters with a known vent start ─────────────
    imv_enc = encounter_df[
        (encounter_df['on_vent'] == 1) &
        encounter_df['vent_start_dttm'].notna()
    ][[id_col, 'vent_start_dttm', 'death_dttm', 'death_enc',
       'discharge_dttm']].copy()

    if len(imv_enc) == 0:
        return pd.DataFrame(columns=[id_col, 'vfd_28'])

    print(f"  Computing VFDs for {len(imv_enc):,} IMV encounters ...")

    imv_enc['window_end'] = imv_enc['vent_start_dttm'] + pd.Timedelta(days=27)

    # ── Identify IMV episode ends from observation data ──────────────
    resp = resp_support_df[[id_col, 'recorded_dttm', 'device_category']].copy()
    resp = resp.sort_values([id_col, 'recorded_dttm'])
    resp['is_imv'] = resp['device_category'].str.contains(
        'imv', case=False, na=False
    )
    resp['next_is_imv'] = resp.groupby(id_col)['is_imv'].shift(-1)
    resp['next_dttm'] = resp.groupby(id_col)['recorded_dttm'].shift(-1)

    # Episode end = IMV observation where the NEXT observation is non-IMV
    # (or it's the last observation for the encounter)
    episode_ends = resp[
        resp['is_imv'] & (~resp['next_is_imv'].fillna(False))
    ].copy()

    # Episode end time = next observation (transition away from IMV).
    # If last observation for encounter (next_dttm NaN), fall back to
    # discharge_dttm (assume IMV until discharge).
    episode_ends = episode_ends[[id_col, 'recorded_dttm', 'next_dttm']].merge(
        imv_enc[[id_col, 'discharge_dttm', 'window_end']],
        on=id_col, how='inner',
    )
    episode_ends['episode_end_dttm'] = (
        episode_ends['next_dttm']
        .fillna(episode_ends['discharge_dttm'])
        .fillna(episode_ends['recorded_dttm'])
    )

    # Keep episodes that start within the 28-day window
    episode_ends = episode_ends[
        episode_ends['recorded_dttm'] <= episode_ends['window_end']
    ]

    # Clamp episode end to window boundary
    episode_ends['clamped_end'] = episode_ends[
        ['episode_end_dttm', 'window_end']
    ].min(axis=1)

    # Last IMV end per encounter (maximum across all episodes in window)
    last_imv = (
        episode_ends
        .groupby(id_col)['clamped_end']
        .max()
        .reset_index()
        .rename(columns={'clamped_end': 'last_imv_dttm'})
    )

    # ── Compute VFDs ─────────────────────────────────────────────────
    result = imv_enc.merge(last_imv, on=id_col, how='left')
    result['vfd_28'] = np.nan

    # Effective death datetime:
    # Use death_dttm when available; fall back to discharge_dttm when
    # death_enc == 1 (discharge_category ∈ {expired, hospice}) to handle
    # missing death timestamps.
    result['effective_death_dttm'] = result['death_dttm'].copy()
    _died_no_dttm = result['death_dttm'].isna() & (result['death_enc'] == 1)
    result.loc[_died_no_dttm, 'effective_death_dttm'] = (
        result.loc[_died_no_dttm, 'discharge_dttm']
    )

    # Rule 1 — death within window → VFD = 0
    death_mask = (
        result['effective_death_dttm'].notna()
        & (result['effective_death_dttm'] <= result['window_end'])
    )
    result.loc[death_mask, 'vfd_28'] = 0

    # Rule 2 — no IMV observations found in window → VFD = 28
    no_imv = result['last_imv_dttm'].isna() & result['vfd_28'].isna()
    result.loc[no_imv, 'vfd_28'] = 28

    # Rule 3 — still on IMV at day 28 → VFD = 0
    still_vent = (
        result['last_imv_dttm'].notna()
        & (result['last_imv_dttm'] >= result['window_end'])
        & result['vfd_28'].isna()
    )
    result.loc[still_vent, 'vfd_28'] = 0

    # Rule 4 — free days = (window_end_day − last_imv_day)
    remaining = result['vfd_28'].isna()
    if remaining.any():
        result.loc[remaining, 'vfd_28'] = (
            result.loc[remaining, 'window_end'].dt.normalize()
            - result.loc[remaining, 'last_imv_dttm'].dt.normalize()
        ).dt.days

    result['vfd_28'] = result['vfd_28'].astype(int)

    # Summary
    vfd = result['vfd_28']
    _n_dead = death_mask.sum()
    _n_still = still_vent.sum()
    _n_free = (vfd > 0).sum()
    print(f"  VFD breakdown: {_n_dead:,} deaths in window, "
          f"{_n_still:,} still on IMV at day 28, "
          f"{_n_free:,} with free days > 0")

    return result[[id_col, 'vfd_28']]
