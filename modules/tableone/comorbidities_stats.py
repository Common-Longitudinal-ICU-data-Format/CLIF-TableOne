"""Comorbidity prevalence CSVs (per-1000 hospitalizations + summary).

Extracted from `generator.py` as a pure refactor.
"""

from __future__ import annotations

import csv
import os

import pandas as pd

from ._helpers import _suffixed


__all__ = ["generate_comorbidities"]


def generate_comorbidities(cci_results, output_dir, suffix=""):
    """Generate comorbidity prevalence per 1000 hospitalizations."""
    os.makedirs(output_dir, exist_ok=True)
    if len(cci_results) == 0:
        return

    total_hospitalizations = cci_results["encounter_block"].nunique()
    exclude_columns = {"hospitalization_id", "encounter_block", "cci_score"}
    comorbidity_columns = [col for col in cci_results.columns if col not in exclude_columns]
    if not comorbidity_columns:
        return

    comorbidity_counts = cci_results[comorbidity_columns].sum()
    comorbidity_per_1000 = (comorbidity_counts / total_hospitalizations) * 1000
    prevalence_percent = (comorbidity_counts.values / total_hospitalizations * 100).round(2)

    comorbidity_summary = (
        pd.DataFrame({
            "comorbidity": comorbidity_columns,
            "n_patients": comorbidity_counts.values,
            "prevalence_percent": prevalence_percent,
            "per_1000_hospitalizations": comorbidity_per_1000.values.round(1),
        })
        .sort_values("per_1000_hospitalizations", ascending=False)
        .reset_index(drop=True)
    )

    comorbidity_summary.to_csv(
        os.path.join(output_dir, _suffixed("comorbidities_per_1000_hospitalizations.csv", suffix)),
        index=False,
    )

    total_comorbidities = int(comorbidity_counts.sum())
    avg_comorbidities_per_hosp = (
        total_comorbidities / total_hospitalizations if total_hospitalizations > 0 else 0
    )
    most_common_comorbidity = comorbidity_summary.iloc[0]["comorbidity"]
    most_common_per_1000 = comorbidity_summary.iloc[0]["per_1000_hospitalizations"]

    summary_stats_rows = [
        ["Total hospitalizations", total_hospitalizations],
        ["Total comorbidities across all patients", total_comorbidities],
        ["Average comorbidities per hospitalization", f"{avg_comorbidities_per_hosp:.2f}"],
        ["Most common comorbidity", most_common_comorbidity],
        ["Most common: per 1000 hospitalizations", f"{most_common_per_1000:.1f}"],
    ]
    with open(
        os.path.join(output_dir, _suffixed("comorbidities_per_1000_hospitalizations_summary.csv", suffix)),
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for row in summary_stats_rows:
            writer.writerow(row)
