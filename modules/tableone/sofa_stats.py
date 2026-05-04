"""SOFA mortality summary CSV.

Extracted from `generator.py` as a pure refactor.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats

from ._helpers import _suffixed


__all__ = ["generate_sofa_mortality"]


def generate_sofa_mortality(final_tableone_df, output_dir, suffix=""):
    """Generate SOFA-mortality summary."""
    os.makedirs(output_dir, exist_ok=True)
    if "sofa_total" not in final_tableone_df.columns or "death_enc" not in final_tableone_df.columns:
        return

    enc_deduped = final_tableone_df.drop_duplicates(subset=["encounter_block"])
    sofa_data = enc_deduped[enc_deduped["sofa_total"].notna()].copy()
    if len(sofa_data) == 0:
        return

    sofa_mortality = sofa_data.groupby("sofa_total").agg(
        count=("death_enc", "count"),
        mortality_rate=("death_enc", "mean"),
    ).reset_index()
    sofa_mortality.columns = ["sofa_score", "count", "mortality_rate"]
    sofa_mortality["mortality_rate"] = sofa_mortality["mortality_rate"] * 100
    sofa_mortality["deaths"] = (sofa_mortality["mortality_rate"] / 100) * sofa_mortality["count"]

    def wilson_ci(successes, n, confidence=0.95):
        z = stats.norm.ppf((1 + confidence) / 2)
        p_hat = successes / n
        denominator = 1 + z**2 / n
        center = (p_hat + z**2 / (2 * n)) / denominator
        margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denominator
        return center * 100, margin * 100

    ci_data = [
        wilson_ci(deaths, n) if n > 0 else (0, 0)
        for deaths, n in zip(sofa_mortality["deaths"], sofa_mortality["count"])
    ]
    sofa_mortality["ci_center"] = [x[0] for x in ci_data]
    sofa_mortality["ci_margin"] = [x[1] for x in ci_data]
    sofa_mortality["ci_lower"] = (sofa_mortality["mortality_rate"] - sofa_mortality["ci_margin"]).clip(lower=0)
    sofa_mortality["ci_upper"] = (sofa_mortality["mortality_rate"] + sofa_mortality["ci_margin"]).clip(upper=100)
    sofa_mortality["total_encounters"] = len(sofa_data)

    sofa_export = sofa_mortality[
        [
            "sofa_score", "total_encounters", "count", "deaths",
            "mortality_rate", "ci_lower", "ci_upper", "ci_margin",
        ]
    ].copy()
    sofa_export.columns = [
        "sofa_score", "total_encounters", "n_encounters", "n_deaths",
        "mortality_rate_percent", "ci_lower_95", "ci_upper_95", "ci_margin_95",
    ]
    sofa_export["n_encounters"] = sofa_export["n_encounters"].astype(int)
    sofa_export["total_encounters"] = sofa_export["total_encounters"].astype(int)
    sofa_export["n_deaths"] = sofa_export["n_deaths"].round(0).astype(int)
    for col in ["mortality_rate_percent", "ci_lower_95", "ci_upper_95", "ci_margin_95"]:
        sofa_export[col] = sofa_export[col].round(2)

    sofa_export.to_csv(
        os.path.join(output_dir, _suffixed("sofa_mortality_summary.csv", suffix)),
        index=False,
    )
