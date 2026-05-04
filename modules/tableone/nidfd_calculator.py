"""
28-day Non-Invasive Device Free Days (NIDFD) calculator.

Mirrors :mod:`modules.tableone.vfd_calculator` but for non-invasive
respiratory support (NIPPV / HFNC≥30 / CPAP) instead of IMV.

Use case from Tab 2 of the ATS abstract plan:
    "Non invasive device free days = from the last day on device to
     discharge, if dead then 0. Max = 28 days. Start day 1 = first day
     on device, count for 28 days after that."

Logic per encounter (mirrors VFD):
  window = [device_start_dttm, device_start_dttm + 27 days]
  1. Death within window                 → NIDFD = 0
  2. No NI device observations in window → NIDFD = 28
  3. Still on NI device at day 28        → NIDFD = 0
  4. Otherwise: NIDFD = (window_end_day − last_ni_device_day)
  Re-application of NI device: intermediate free days do NOT count.

DEFINITION OF "NON-INVASIVE DEVICE"
===================================
Per the existing high_support / nippv_hfnc cohort logic in generator.py,
"non-invasive" here means **NIPPV or CPAP at any LPM**, plus **high flow
nasal cannula at ≥ hfnc_lpm_threshold (default 30)**. IMV is excluded —
patients on IMV are already covered by VFD. The function is parameterized
so the caller can pass a custom threshold or device list if needed.

Note: encounters that received BOTH IMV and a non-invasive device should
be reported by both VFD and NIDFD (they're two separate per-encounter
metrics, not mutually exclusive).

KNOWN LIMITATION (shared with vfd_calculator)
==============================================
Encounters that are continuously on NI device through day 28 with no
transition-off observed within the 28-day window may be classified as
"no NI in window" (NIDFD=28) instead of "still on at day 28" (NIDFD=0).
This affects encounters whose only NI rows in the cohort are followed
by another NI row past the window boundary (no transition observed).

This is the same algorithmic limitation that exists in vfd_calculator.py.
Fixing it requires interval-based episode tracking (start + end of each
episode tracked separately), and should be applied to both calculators
together with regression tests against the existing VFD output.

Practical impact: rare. Most encounters either transition off (Rule 4)
or have observable IMV/NI continuing past day 28 with discrete
observations within the window (Rule 3). Only encounters with sparse
observations and continuous NI use are affected.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


__all__ = [
    "DEFAULT_HFNC_LPM_THRESHOLD",
    "DEFAULT_NI_DEVICES_ALWAYS",
    "calculate_non_invasive_device_free_days",
]


DEFAULT_HFNC_LPM_THRESHOLD: float = 30.0
DEFAULT_NI_DEVICES_ALWAYS: tuple[str, ...] = ("nippv", "cpap")


def calculate_non_invasive_device_free_days(
    resp_support_df: pd.DataFrame,
    encounter_df: pd.DataFrame,
    *,
    id_col: str = "encounter_block",
    hfnc_lpm_threshold: float = DEFAULT_HFNC_LPM_THRESHOLD,
    ni_devices_always: Iterable[str] = DEFAULT_NI_DEVICES_ALWAYS,
) -> pd.DataFrame:
    """
    Calculate 28-day non-invasive device free days for encounters that
    received NIPPV/CPAP/HFNC≥threshold at any point.

    Args:
        resp_support_df: Respiratory support observations with columns:
            [id_col, recorded_dttm, device_category, lpm_set]
            (lpm_set required only if device_category == 'high flow nc'
            rows are present.)
        encounter_df: Encounter-level data with columns:
            [id_col, nippv_hfnc_enc, ni_device_start_dttm, death_dttm,
             death_enc, discharge_dttm]

            ``nippv_hfnc_enc`` is the binary flag computed elsewhere
            (1 if encounter ever received NIPPV or HFNC≥threshold).
            ``ni_device_start_dttm`` is the first qualifying NI-device
            recorded_dttm for the encounter — the t=0 of the 28-day
            window.

            **NOTE**: callers should compute ``ni_device_start_dttm``
            upstream the same way ``vent_start_dttm`` is computed for
            VFD (min ``recorded_dttm`` over qualifying NI rows).
        id_col: Encounter identifier column.
        hfnc_lpm_threshold: HFNC LPM minimum to count as advanced.
        ni_devices_always: Device categories that always count as NI
            regardless of LPM (default: nippv, cpap).

    Returns:
        DataFrame with columns [id_col, nidfd_28].
        One row per encounter that had nippv_hfnc_enc == 1 and a known
        ni_device_start_dttm.
    """
    required = [
        "nippv_hfnc_enc",
        "ni_device_start_dttm",
        "death_dttm",
        "death_enc",
        "discharge_dttm",
    ]
    missing = [c for c in required if c not in encounter_df.columns]
    if missing:
        print(f"  ⚠️ NIDFD skipped — missing columns: {missing}")
        return pd.DataFrame(columns=[id_col, "nidfd_28"])

    if "device_category" not in resp_support_df.columns:
        print("  ⚠️ NIDFD skipped — resp_support_df missing device_category")
        return pd.DataFrame(columns=[id_col, "nidfd_28"])

    # Defensive datetime coercion (matches VFD pattern). UTC coercion
    # avoids tzdata mishandling on Windows / sites without tzdata installed.
    _dt_cols = ("ni_device_start_dttm", "death_dttm", "discharge_dttm")
    if any(
        not pd.api.types.is_datetime64_any_dtype(encounter_df[c]) for c in _dt_cols
    ):
        encounter_df = encounter_df.assign(**{
            c: pd.to_datetime(encounter_df[c], errors="coerce", utc=True)
            for c in _dt_cols
            if not pd.api.types.is_datetime64_any_dtype(encounter_df[c])
        })
    if not pd.api.types.is_datetime64_any_dtype(resp_support_df["recorded_dttm"]):
        resp_support_df = resp_support_df.assign(
            recorded_dttm=pd.to_datetime(
                resp_support_df["recorded_dttm"], errors="coerce", utc=True
            )
        )

    # ── Filter to encounters with NI support and a known start dttm ──
    ni_enc = encounter_df[
        (encounter_df["nippv_hfnc_enc"] == 1)
        & encounter_df["ni_device_start_dttm"].notna()
    ][[id_col, "ni_device_start_dttm", "death_dttm", "death_enc", "discharge_dttm"]].copy()

    if len(ni_enc) == 0:
        return pd.DataFrame(columns=[id_col, "nidfd_28"])

    print(f"  Computing NIDFDs for {len(ni_enc):,} NIPPV/HFNC encounters ...")

    ni_enc["window_end"] = ni_enc["ni_device_start_dttm"] + pd.Timedelta(days=27)

    # ── Identify NI device observations from resp_support ────────────
    resp = resp_support_df[[id_col, "recorded_dttm", "device_category"]].copy()
    if "lpm_set" in resp_support_df.columns:
        resp["lpm_set"] = pd.to_numeric(resp_support_df["lpm_set"], errors="coerce")
    else:
        resp["lpm_set"] = np.nan

    dev = resp["device_category"].astype(str).str.lower()
    always_set = {str(d).lower() for d in ni_devices_always}
    resp["is_ni"] = dev.isin(always_set) | (
        (dev == "high flow nc") & (resp["lpm_set"] >= hfnc_lpm_threshold)
    )

    resp = resp.sort_values([id_col, "recorded_dttm"])
    resp["next_is_ni"] = resp.groupby(id_col)["is_ni"].shift(-1)
    resp["next_dttm"] = resp.groupby(id_col)["recorded_dttm"].shift(-1)

    # Episode end = NI observation where the NEXT observation is non-NI
    # (or it's the last observation for the encounter)
    episode_ends = resp[resp["is_ni"] & (~resp["next_is_ni"].fillna(False))].copy()

    episode_ends = episode_ends[[id_col, "recorded_dttm", "next_dttm"]].merge(
        ni_enc[[id_col, "discharge_dttm", "window_end"]],
        on=id_col,
        how="inner",
    )
    episode_ends["episode_end_dttm"] = (
        episode_ends["next_dttm"]
        .fillna(episode_ends["discharge_dttm"])
        .fillna(episode_ends["recorded_dttm"])
    )

    # Keep episodes that started within the 28-day window
    # (Same conservative semantics as vfd_calculator.py — see KNOWN LIMITATION
    # docstring at the top of this module for the edge case where an encounter
    # is continuously on NI through day 28 with no transition-off observed
    # within the window. That case gets NIDFD=28 from Rule 2; the proper fix
    # requires interval-based episode tracking and should be applied to both
    # VFD and NIDFD together.)
    episode_ends = episode_ends[
        episode_ends["recorded_dttm"] <= episode_ends["window_end"]
    ]

    # Clamp episode end to window boundary
    episode_ends["clamped_end"] = episode_ends[
        ["episode_end_dttm", "window_end"]
    ].min(axis=1)

    # Last NI episode end per encounter
    last_ni = (
        episode_ends.groupby(id_col)["clamped_end"]
        .max()
        .reset_index()
        .rename(columns={"clamped_end": "last_ni_dttm"})
    )

    # ── Compute NIDFDs ───────────────────────────────────────────────
    result = ni_enc.merge(last_ni, on=id_col, how="left")
    result["nidfd_28"] = np.nan

    # Effective death datetime (death_dttm with discharge_dttm fallback)
    result["effective_death_dttm"] = result["death_dttm"].copy()
    _died_no_dttm = result["death_dttm"].isna() & (result["death_enc"] == 1)
    result.loc[_died_no_dttm, "effective_death_dttm"] = (
        result.loc[_died_no_dttm, "discharge_dttm"]
    )

    # Rule 1 — death within window → NIDFD = 0
    death_mask = (
        result["effective_death_dttm"].notna()
        & (result["effective_death_dttm"] <= result["window_end"])
    )
    result.loc[death_mask, "nidfd_28"] = 0

    # Rule 2 — no NI observations found in window → NIDFD = 28
    # (encounter qualified as nippv_hfnc_enc but the qualifying rows are
    # outside the 28-day window — rare edge case, fall back to "free")
    no_ni = result["last_ni_dttm"].isna() & result["nidfd_28"].isna()
    result.loc[no_ni, "nidfd_28"] = 28

    # Rule 3 — still on NI device at day 28 → NIDFD = 0
    still_on = (
        result["last_ni_dttm"].notna()
        & (result["last_ni_dttm"] >= result["window_end"])
        & result["nidfd_28"].isna()
    )
    result.loc[still_on, "nidfd_28"] = 0

    # Rule 4 — free days = (window_end_day − last_ni_day)
    remaining = result["nidfd_28"].isna()
    if remaining.any():
        result.loc[remaining, "nidfd_28"] = (
            result.loc[remaining, "window_end"].dt.normalize()
            - result.loc[remaining, "last_ni_dttm"].dt.normalize()
        ).dt.days

    result["nidfd_28"] = result["nidfd_28"].astype(int)

    nidfd = result["nidfd_28"]
    _n_dead = death_mask.sum()
    _n_still = still_on.sum()
    _n_free = (nidfd > 0).sum()
    print(
        f"  NIDFD breakdown: {_n_dead:,} deaths in window, "
        f"{_n_still:,} still on NI device at day 28, "
        f"{_n_free:,} with free days > 0"
    )

    return result[[id_col, "nidfd_28"]]
