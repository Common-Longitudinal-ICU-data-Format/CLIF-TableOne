"""Time-to-ICU-after-first-pressor metric.

Tab 2 of the ATS abstract plan asks for: "Time to ICU after pressor
administration" for the ED→ICU vasopressor sub-cohort. Specifically:
patients whose first vasoactive medication was administered in the ED
and who subsequently transferred to ICU.

This metric supplements the existing vaso/ed_icu vs vaso/ed_ward
stratification — the strata tell you WHO went where, this calculator
tells you HOW LONG it took to get to ICU.

USE
===
Called per-encounter on the vaso_ed_icu sub-stratum. Returns one row
per encounter with the time delta in hours.

DEFINITION
==========
For each encounter where ``vaso_ed_icu_enc == 1``:

    time_to_icu_hours = (first_icu_in_dttm - first_pressor_admin_dttm) / 3600s

Where:
  - ``first_pressor_admin_dttm`` = MIN(admin_dttm) of any qualifying
    vaso row in the ED location (computed upstream when the
    ``vaso_ed_icu_enc`` flag is set)
  - ``first_icu_in_dttm`` = MIN(in_dttm) of any ADT row with
    ``location_category == 'icu'`` for that encounter (computed
    upstream)

Encounters where either timestamp is missing are dropped from the
result (NaN time_to_icu_hours rows excluded).

Negative time deltas indicate the patient was already in ICU when
they got their "first" ED pressor — likely a data integrity issue.
The function returns these as-is so they show up as outliers; the
caller can choose to filter them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


__all__ = [
    "calculate_time_to_icu_after_pressor",
    "summarize_time_to_icu",
]


def calculate_time_to_icu_after_pressor(
    encounter_df: pd.DataFrame,
    *,
    id_col: str = "encounter_block",
    pressor_dttm_col: str = "first_pressor_admin_dttm",
    icu_dttm_col: str = "first_icu_in_dttm",
    flag_col: str = "vaso_ed_icu_enc",
) -> pd.DataFrame:
    """Compute time-to-ICU (hours) for the ED→ICU vasopressor cohort.

    Args:
        encounter_df: DataFrame with at least ``id_col``, ``flag_col``,
            ``pressor_dttm_col``, ``icu_dttm_col``. Other columns are
            ignored.
        id_col: encounter_block column name.
        pressor_dttm_col: column with first ED-pressor admin_dttm.
        icu_dttm_col: column with first ICU in_dttm.
        flag_col: binary flag identifying the ED→ICU vaso sub-cohort.

    Returns:
        DataFrame with columns ``[id_col, time_to_icu_hours]``. One row
        per encounter where flag_col == 1 AND both dttm columns are
        non-null. Encounters in the cohort but missing either timestamp
        are silently dropped (data quality issue at the site).
    """
    required = [id_col, flag_col, pressor_dttm_col, icu_dttm_col]
    missing = [c for c in required if c not in encounter_df.columns]
    if missing:
        return pd.DataFrame(columns=[id_col, "time_to_icu_hours"])

    # Filter to the sub-cohort (ED→ICU vasopressor flag).
    sub = encounter_df[encounter_df[flag_col] == 1][
        [id_col, pressor_dttm_col, icu_dttm_col]
    ].copy()

    if len(sub) == 0:
        return pd.DataFrame(columns=[id_col, "time_to_icu_hours"])

    # Defensive: coerce to tz-naive UTC to avoid mixed tz subtraction errors.
    for col in (pressor_dttm_col, icu_dttm_col):
        sub[col] = pd.to_datetime(sub[col], utc=True, errors="coerce").dt.tz_localize(None)

    # Drop rows missing either timestamp (data-quality drop).
    sub = sub[sub[pressor_dttm_col].notna() & sub[icu_dttm_col].notna()].copy()

    if len(sub) == 0:
        return pd.DataFrame(columns=[id_col, "time_to_icu_hours"])

    # Time delta in hours.
    delta = sub[icu_dttm_col] - sub[pressor_dttm_col]
    sub["time_to_icu_hours"] = delta.dt.total_seconds() / 3600.0

    return sub[[id_col, "time_to_icu_hours"]].reset_index(drop=True)


def summarize_time_to_icu(time_df: pd.DataFrame) -> dict:
    """Compute summary stats for the time_to_icu DataFrame.

    Returns a dict with median, q1, q3, mean, std, n. Callers typically
    embed these into a Table One row like
    ``"Time to ICU after first ED pressor (hr), median [Q1, Q3]"``.

    Negative time deltas (patient was already in ICU when they got their
    "first" ED pressor — likely data integrity issue at the site) are
    INCLUDED in the summary; the consortium aggregator can optionally
    filter them. Including them surfaces the data-quality signal.
    """
    if len(time_df) == 0:
        return {
            "n": 0,
            "n_negative": 0,
            "median": float("nan"),
            "q1":     float("nan"),
            "q3":     float("nan"),
            "mean":   float("nan"),
            "std":    float("nan"),
        }

    h = time_df["time_to_icu_hours"]
    return {
        "n":          int(len(h)),
        "n_negative": int((h < 0).sum()),
        "median":     float(h.median()),
        "q1":         float(h.quantile(0.25)),
        "q3":         float(h.quantile(0.75)),
        "mean":       float(h.mean()),
        "std":        float(h.std(ddof=1)) if len(h) > 1 else float("nan"),
    }
