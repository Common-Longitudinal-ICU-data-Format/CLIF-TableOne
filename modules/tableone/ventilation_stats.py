"""Ventilator settings, tidal volume, pressure-control, and mode-proportion CSVs.

These four functions are the per-stratum ventilation summary writers — they
consume already-prepared respiratory_support DataFrames and emit small
aggregate CSVs (no PHI). Extracted from `generator.py` as a pure refactor.

NOTE: the underlying *production* of `resp_valid` / `resp_imv_post_start`
(the +18.5 GB hotspot) lives in generator.py's `main()` and will be
rewritten in Polars/DuckDB during Phase 5. These four functions take that
DataFrame as input and are not themselves the memory bottleneck.
"""

from __future__ import annotations

import os

import pandas as pd

from ._helpers import _suffixed


__all__ = [
    "generate_ventilator_settings_summary",
    "generate_tidal_volume_stats",
    "generate_pressure_control_stats",
    "generate_mode_proportions",
]


def generate_ventilator_settings_summary(resp_valid, vent_settings, output_dir, suffix=""):
    """Generate ventilator settings summary CSVs."""
    os.makedirs(output_dir, exist_ok=True)
    existing_settings = [col for col in vent_settings if col in resp_valid.columns]
    if not existing_settings or len(resp_valid) == 0:
        return

    medians = resp_valid.groupby(["device_category", "mode_category"])[existing_settings].median()
    q1 = resp_valid.groupby(["device_category", "mode_category"])[existing_settings].quantile(0.25)
    q3 = resp_valid.groupby(["device_category", "mode_category"])[existing_settings].quantile(0.75)

    medians_reset = medians.reset_index()
    q1_reset = q1.reset_index()
    q3_reset = q3.reset_index()

    settings_summary = medians_reset[["device_category", "mode_category"]].copy()
    for setting in existing_settings:
        if setting in medians.columns:
            settings_summary[setting] = (
                medians_reset[setting].round(1).astype(str)
                + " ("
                + q1_reset[setting].round(1).astype(str)
                + "-"
                + q3_reset[setting].round(1).astype(str)
                + ")"
            )

    rename_dict = {}
    if "mode_category" in settings_summary.columns:
        rename_dict["mode_category"] = "ventilator_setting"
    column_mapping = {
        "fio2_set": "FiO2 Set", "lpm_set": "LPM Set",
        "tidal_volume_set": "Tidal Volume Set", "resp_rate_set": "Resp Rate Set",
        "pressure_control_set": "Pressure Control Set", "peep_set": "PEEP Set",
        "pressure_support_set": "Pressure Support Set", "flow_rate_set": "Flow Rate Set",
    }
    for old_name, new_name in column_mapping.items():
        if old_name in settings_summary.columns:
            rename_dict[old_name] = new_name
    settings_summary = settings_summary.rename(columns=rename_dict)
    sort_col = "ventilator_setting" if "ventilator_setting" in settings_summary.columns else "mode_category"
    settings_summary = settings_summary.sort_values(["device_category", sort_col])
    settings_summary.to_csv(
        os.path.join(output_dir, _suffixed("ventilator_settings_by_device_mode.csv", suffix)),
        index=False,
    )

    # Counts table
    counts_summary = resp_valid.groupby(["device_category", "mode_category"])[existing_settings].count().reset_index()
    counts_rename = {}
    if "mode_category" in counts_summary.columns:
        counts_rename["mode_category"] = "ventilator_setting"
    for old_name, new_name in column_mapping.items():
        col_n = old_name
        if col_n in counts_summary.columns:
            counts_rename[col_n] = new_name + " (N)"
    counts_summary = counts_summary.rename(columns=counts_rename)
    sort_col = "ventilator_setting" if "ventilator_setting" in counts_summary.columns else "mode_category"
    counts_summary = counts_summary.sort_values(["device_category", sort_col])
    counts_summary.to_csv(
        os.path.join(output_dir, _suffixed("ventilator_settings_counts_by_device_mode.csv", suffix)),
        index=False,
    )

    # Total observations
    total_obs_df = pd.DataFrame({
        "metric": ["total_respiratory_support_observations"],
        "value": [len(resp_valid)],
    })
    total_obs_df.to_csv(
        os.path.join(output_dir, _suffixed("ventilator_settings_total_observations.csv", suffix)),
        index=False,
    )


def generate_tidal_volume_stats(resp_imv_post_start, output_dir, suffix=""):
    """Generate tidal volume stats for volume control modes."""
    os.makedirs(output_dir, exist_ok=True)
    volume_control_modes = ["assist control-volume control", "pressure-regulated volume control"]
    volume_mode_data = resp_imv_post_start[
        resp_imv_post_start["mode_category"].isin(volume_control_modes)
    ].copy()
    if len(volume_mode_data) == 0 or "tidal_volume_set" not in volume_mode_data.columns:
        return

    volume_mode_data["hour_bin"] = volume_mode_data["hours_from_vent_start"].round(0).astype(int)
    volume_mode_data_7d = volume_mode_data[volume_mode_data["hour_bin"] <= 168].copy()

    tv_stats = volume_mode_data_7d.groupby("hour_bin")["tidal_volume_set"].agg([
        ("median", "median"),
        ("q25", lambda x: x.quantile(0.25)),
        ("q75", lambda x: x.quantile(0.75)),
        ("mean", "mean"),
        ("std", "std"),
        ("count", "count"),
    ]).reset_index()
    tv_stats = tv_stats[tv_stats["count"] >= 10]

    tv_stats.to_csv(
        os.path.join(output_dir, _suffixed("tidal_volume_volume_control_modes.csv", suffix)),
        index=False,
    )
    tv_stats[["hour_bin", "mean", "std", "count"]].to_csv(
        os.path.join(output_dir, _suffixed("tidal_volume_volume_control_modes_mean_sd.csv", suffix)),
        index=False,
    )


def generate_pressure_control_stats(resp_imv_post_start, output_dir, suffix=""):
    """Generate pressure control stats for pressure control mode."""
    os.makedirs(output_dir, exist_ok=True)
    pressure_mode_data = resp_imv_post_start[
        resp_imv_post_start["mode_category"].isin(["pressure control"])
    ].copy()
    if len(pressure_mode_data) == 0 or "pressure_control_set" not in pressure_mode_data.columns:
        return

    pressure_mode_data["hour_bin"] = pressure_mode_data["hours_from_vent_start"].round(0).astype(int)
    pressure_mode_data_7d = pressure_mode_data[pressure_mode_data["hour_bin"] <= 168].copy()

    pc_stats = pressure_mode_data_7d.groupby("hour_bin")["pressure_control_set"].agg([
        ("median", "median"),
        ("q25", lambda x: x.quantile(0.25)),
        ("q75", lambda x: x.quantile(0.75)),
        ("mean", "mean"),
        ("std", "std"),
        ("count", "count"),
    ]).reset_index()
    pc_stats = pc_stats[pc_stats["count"] >= 10]

    pc_stats.to_csv(
        os.path.join(output_dir, _suffixed("pressure_control_pressure_control_mode.csv", suffix)),
        index=False,
    )
    pc_stats[["hour_bin", "mean", "std", "count"]].to_csv(
        os.path.join(output_dir, _suffixed("pressure_control_pressure_control_mode_mean_sd.csv", suffix)),
        index=False,
    )


def generate_mode_proportions(resp_imv_post_start, output_dir, suffix=""):
    """Generate mode proportions for first 24 hours of IMV."""
    os.makedirs(output_dir, exist_ok=True)
    if len(resp_imv_post_start) == 0:
        return

    imv_first_24h = resp_imv_post_start[
        (resp_imv_post_start["hours_from_vent_start"] >= 0)
        & (resp_imv_post_start["hours_from_vent_start"] <= 24)
    ].copy()
    if len(imv_first_24h) == 0:
        return

    mode_mapping = {
        "assist control-volume control": "Assist Control-Volume Control",
        "pressure-regulated volume control": "Pressure-Regulated Volume Control",
        "simv": "SIMV",
        "pressure support/cpap": "Pressure Support/CPAP",
        "pressure support": "Pressure Support/CPAP",
        "cpap": "Pressure Support/CPAP",
        "pressure control": "Pressure Control",
    }
    imv_first_24h["mode_group"] = (
        imv_first_24h["mode_category"].str.lower().map(mode_mapping).fillna("Other")
    )

    mode_counts = imv_first_24h["mode_group"].value_counts()
    total_obs = len(imv_first_24h)
    mode_proportions = (mode_counts / total_obs).sort_values(ascending=False)

    plot_data = pd.DataFrame({
        "Mode": mode_proportions.index,
        "Proportion": mode_proportions.values,
        "Count": mode_counts[mode_proportions.index].values,
    })
    plot_data.to_csv(
        os.path.join(output_dir, _suffixed("mode_proportions_first_24h.csv", suffix)),
        index=False,
    )
