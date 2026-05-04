"""Per-stratum medication summary CSVs (hourly + summary).

Extracted from `generator.py` as a pure refactor.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ._helpers import _suffixed


__all__ = [
    "generate_medications_hourly",
    "generate_medications_summary",
]


def generate_medications_hourly(meds_merged, total_icu_encounters, med_groups, output_dir, suffix=""):
    """Generate hourly medication usage data."""
    os.makedirs(output_dir, exist_ok=True)
    all_meds = [med for meds in med_groups.values() for med in meds]

    if len(meds_merged) == 0 or total_icu_encounters == 0:
        return

    meds_7d = meds_merged[
        (meds_merged["med_lower"].isin(all_meds))
        & (meds_merged["hour_bin"].notna())
        & (meds_merged["hour_bin"] >= 0)
        & (meds_merged["hour_bin"] <= 167)
    ]
    if len(meds_7d) == 0:
        return

    pivot = (
        meds_7d.groupby(["hour_bin", "med_lower"])["encounter_block"]
        .nunique()
        .unstack(fill_value=0)
        .reindex(index=np.arange(168), columns=all_meds, fill_value=0)
    )
    pct_pivot = (pivot / total_icu_encounters * 100) if total_icu_encounters > 0 else pivot * 0

    hourly_df = pd.DataFrame({"hour": np.arange(168)})
    hourly_df = pd.concat([hourly_df, pivot.add_suffix("_n"), pct_pivot.add_suffix("_pct")], axis=1)
    hourly_df.to_csv(
        os.path.join(output_dir, _suffixed("medications_hourly_data.csv", suffix)),
        index=False,
    )


def generate_medications_summary(meds_merged, total_icu_encounters, med_groups, output_dir, suffix=""):
    """Generate medication summary statistics."""
    os.makedirs(output_dir, exist_ok=True)
    all_meds = [med for meds in med_groups.values() for med in meds]

    if len(meds_merged) == 0 or total_icu_encounters == 0:
        return

    meds_7d = meds_merged[
        (meds_merged["med_lower"].isin(all_meds))
        & (meds_merged["hour_bin"].notna())
        & (meds_merged["hour_bin"] >= 0)
        & (meds_merged["hour_bin"] <= 167)
    ]
    if len(meds_7d) == 0:
        return

    summary_agg = (
        meds_7d.groupby("med_lower")["med_dose"]
        .agg(["count", "median", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
        .rename(columns={"count": "n_admin", "<lambda_0>": "q1_dose", "<lambda_1>": "q3_dose"})
    )
    encounter_counts = meds_7d.groupby("med_lower")["encounter_block"].nunique()
    dose_units = meds_7d.groupby("med_lower")["med_dose_unit"].agg(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else ""
    )
    med_to_group = {med: group for group, meds in med_groups.items() for med in meds}

    summary_df = pd.DataFrame({
        "medication": summary_agg.index,
        "n_encounters": encounter_counts.values,
        "pct_encounters": (encounter_counts / total_icu_encounters * 100).values,
        "median_dose": summary_agg["median"].values,
        "q1_dose": summary_agg["q1_dose"].values,
        "q3_dose": summary_agg["q3_dose"].values,
        "dose_unit": dose_units.values,
    })
    summary_df["group"] = summary_df["medication"].map(med_to_group)
    summary_df = summary_df[
        [
            "group", "medication", "n_encounters", "pct_encounters",
            "median_dose", "q1_dose", "q3_dose", "dose_unit",
        ]
    ]
    summary_df.to_csv(
        os.path.join(output_dir, _suffixed("medications_summary_stats.csv", suffix)),
        index=False,
    )
