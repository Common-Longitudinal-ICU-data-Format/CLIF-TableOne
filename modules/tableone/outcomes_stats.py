"""Outcomes trends — hospice/mortality and CCI×year×category.

Extracted from `generator.py` as a pure refactor.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from ._helpers import _suffixed


__all__ = [
    "generate_hospice_trends",
    "generate_cci_hospice_mortality",
]


def generate_hospice_trends(final_tableone_df, output_dir, suffix=""):
    """Generate hospice vs mortality trends by year."""
    os.makedirs(output_dir, exist_ok=True)
    required_cols = [
        "encounter_block", "admission_year", "hospice_outcome",
        "expired_outcome", "hospice_or_expired",
    ]
    if not all(c in final_tableone_df.columns for c in required_cols):
        return

    enc_deduped = final_tableone_df.drop_duplicates(subset=["encounter_block"])
    if len(enc_deduped) == 0:
        return

    hospice_trends = enc_deduped.groupby("admission_year").agg({
        "encounter_block": "count",
        "hospice_outcome": "sum",
        "expired_outcome": "sum",
        "hospice_or_expired": "sum",
    }).reset_index()
    hospice_trends.columns = ["year", "total_encounters", "hospice", "expired", "hospice_or_expired"]

    hospice_trends["hospice_pct"] = hospice_trends["hospice"] / hospice_trends["total_encounters"] * 100
    hospice_trends["mortality_pct"] = hospice_trends["expired"] / hospice_trends["total_encounters"] * 100
    hospice_trends["hospice_among_eol_pct"] = (
        hospice_trends["hospice"] / hospice_trends["hospice_or_expired"] * 100
    )

    def calc_ci(successes, n, confidence=0.95):
        if n == 0:
            return 0, 0
        ci_low, ci_upp = proportion_confint(successes, n, alpha=1 - confidence, method="wilson")
        return ci_low * 100, ci_upp * 100

    ci_results = [
        calc_ci(row["hospice"], row["hospice_or_expired"])
        for _, row in hospice_trends.iterrows()
    ]
    hospice_trends["hospice_among_eol_ci_lower"] = [x[0] for x in ci_results]
    hospice_trends["hospice_among_eol_ci_upper"] = [x[1] for x in ci_results]

    hospice_trends.to_csv(
        os.path.join(output_dir, _suffixed("hospice_trends_summary.csv", suffix)),
        index=False,
    )


def generate_cci_hospice_mortality(final_tableone_df, output_dir, suffix=""):
    """Generate CCI-hospice-mortality comprehensive summary."""
    os.makedirs(output_dir, exist_ok=True)
    required_cols = [
        "encounter_block", "cci_score", "admission_year",
        "expired_outcome", "hospice_or_expired", "hospice_outcome",
    ]
    if not all(c in final_tableone_df.columns for c in required_cols):
        return

    tableone_with_cci = final_tableone_df[required_cols].copy()
    tableone_with_cci["cci_category"] = pd.cut(
        tableone_with_cci["cci_score"],
        bins=[-np.inf, 0, 2, 4, np.inf],
        labels=["0 (No comorbidity)", "1-2 (Mild)", "3-4 (Moderate)", "5+ (Severe)"],
    )

    cci_summary = tableone_with_cci.groupby(["admission_year", "cci_category"]).agg({
        "encounter_block": "count",
        "hospice_outcome": "sum",
        "expired_outcome": "sum",
        "hospice_or_expired": "sum",
    }).reset_index()
    cci_summary.columns = [
        "year", "cci_category", "total_encounters",
        "hospice_count", "expired_count", "hospice_or_expired_count",
    ]

    cci_summary["mortality_pct"] = cci_summary["expired_count"] / cci_summary["total_encounters"] * 100
    cci_summary["hospice_pct"] = cci_summary["hospice_count"] / cci_summary["total_encounters"] * 100
    cci_summary["combined_eol_pct"] = (
        cci_summary["hospice_or_expired_count"] / cci_summary["total_encounters"] * 100
    )
    cci_summary["hospice_among_eol_pct"] = (
        cci_summary["hospice_count"] / cci_summary["hospice_or_expired_count"] * 100
    )
    cci_summary["hospice_capture_rate"] = (
        cci_summary["hospice_count"] / cci_summary["expired_count"] * 100
    )

    def calc_ci(row):
        if row["hospice_or_expired_count"] == 0:
            return pd.Series({"hospice_eol_ci_lower": np.nan, "hospice_eol_ci_upper": np.nan})
        ci_low, ci_upp = proportion_confint(
            row["hospice_count"], row["hospice_or_expired_count"], alpha=0.05, method="wilson"
        )
        return pd.Series({
            "hospice_eol_ci_lower": ci_low * 100,
            "hospice_eol_ci_upper": ci_upp * 100,
        })

    cci_summary[["hospice_eol_ci_lower", "hospice_eol_ci_upper"]] = cci_summary.apply(calc_ci, axis=1)

    cci_summary = cci_summary[
        [
            "year", "cci_category", "total_encounters",
            "expired_count", "mortality_pct", "hospice_count", "hospice_pct",
            "hospice_or_expired_count", "combined_eol_pct",
            "hospice_among_eol_pct", "hospice_eol_ci_lower", "hospice_eol_ci_upper",
            "hospice_capture_rate",
        ]
    ]
    cci_summary.to_csv(
        os.path.join(output_dir, _suffixed("cci_hospice_mortality_comprehensive_summary.csv", suffix)),
        index=False,
    )

    # Plot data
    categories = ["0 (No comorbidity)", "1-2 (Mild)", "3-4 (Moderate)", "5+ (Severe)"]
    plot_data = []
    for category in categories:
        cat_data = cci_summary[cci_summary["cci_category"] == category].sort_values("year")
        plot_data.append(cat_data.assign(cci_category_label=category))
    plot_data_df = pd.concat(plot_data, axis=0)
    plot_data_df.to_csv(
        os.path.join(output_dir, _suffixed("cci_mortality_hospice_trends_by_year_category_plotdata.csv", suffix)),
        index=False,
    )
