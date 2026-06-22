"""
ADT location coverage analysis for the overall (critical-illness) cohort.

Answers two questions:
1. Coverage — What % of labs/vitals/respiratory_support events were recorded
   at each ADT location_category?
2. Dwell time — How much cumulative time do patients spend at each location?

Adapted from dev/adt_location_coverage.py (standalone marimo notebook).
"""

import os
from typing import List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from modules.utils.clif_loader import load_filtered_clif_table

NO_MATCH = "__no_adt_match__"


# ── helpers ──────────────────────────────────────────────────────────────────


def _prepare_adt(adt_df: pd.DataFrame, hospitalization_ids: List[str]) -> pl.DataFrame:
    """Filter ADT to cohort, convert to polars, compute stay_hours."""
    adt = (
        pl.from_pandas(
            adt_df[adt_df["hospitalization_id"].isin(set(hospitalization_ids))]
            [["hospitalization_id", "in_dttm", "out_dttm", "location_category"]]
        )
        .drop_nulls(["in_dttm", "out_dttm", "location_category"])
        .filter(pl.col("out_dttm") > pl.col("in_dttm"))
        .with_columns(
            ((pl.col("out_dttm") - pl.col("in_dttm")).dt.total_seconds() / 3600.0)
            .alias("stay_hours")
        )
        .sort(["hospitalization_id", "in_dttm"])
    )
    return adt


def _compute_dwell_summary(adt_pl: pl.DataFrame) -> pl.DataFrame:
    return (
        adt_pl.group_by("location_category")
        .agg(
            pl.col("stay_hours").sum().alias("total_hours"),
            pl.col("stay_hours").median().alias("median_stay_hours"),
            pl.col("stay_hours").quantile(0.25).alias("q1_stay_hours"),
            pl.col("stay_hours").quantile(0.75).alias("q3_stay_hours"),
            pl.col("hospitalization_id").n_unique().alias("distinct_encounters"),
            pl.len().alias("n_stays"),
        )
        .with_columns((pl.col("total_hours") / 24.0).alias("total_days"))
        .sort("total_hours", descending=True)
    )


def _join_events_to_location(
    events: pl.DataFrame,
    ts_col: str,
    adt_for_join: pl.DataFrame,
) -> pl.DataFrame:
    """Backward asof-join events onto ADT intervals, flagging gaps."""
    # Normalize datetime precision (μs vs ns) to prevent join failures
    events = events.cast({ts_col: pl.Datetime("us", time_zone=events[ts_col].dtype.time_zone)})
    adt_for_join = adt_for_join.cast({
        "in_dttm": pl.Datetime("us", time_zone=adt_for_join["in_dttm"].dtype.time_zone),
        "out_dttm": pl.Datetime("us", time_zone=adt_for_join["out_dttm"].dtype.time_zone),
    })
    joined = events.join_asof(
        adt_for_join,
        left_on=ts_col,
        right_on="in_dttm",
        by="hospitalization_id",
        strategy="backward",
    )
    return joined.with_columns(
        pl.when(
            pl.col("out_dttm").is_not_null()
            & (pl.col(ts_col) < pl.col("out_dttm"))
        )
        .then(pl.col("location_category"))
        .otherwise(pl.lit(NO_MATCH))
        .alias("location_category")
    )


def _summarize_capture(
    joined: pl.DataFrame,
    table_label: str,
    dwell_summary: pl.DataFrame,
) -> pl.DataFrame:
    total = joined.height
    per_loc = (
        joined.group_by("location_category")
        .agg(pl.len().alias("n_events"))
        .with_columns(
            (pl.col("n_events") / max(total, 1) * 100).alias("pct_events"),
            pl.lit(table_label).alias("table"),
        )
    )
    per_loc = per_loc.join(
        dwell_summary.select(["location_category", "total_hours"]),
        on="location_category",
        how="left",
    ).with_columns(
        pl.when(
            (pl.col("location_category") != NO_MATCH)
            & pl.col("total_hours").is_not_null()
            & (pl.col("total_hours") > 0)
        )
        .then(pl.col("n_events") / pl.col("total_hours"))
        .otherwise(None)
        .alias("events_per_hour")
    )
    return per_loc.sort("n_events", descending=True)


# ── figures ──────────────────────────────────────────────────────────────────


def _create_dwell_bar(dwell_summary: pl.DataFrame) -> go.Figure:
    fig = px.bar(
        dwell_summary.to_pandas(),
        x="location_category",
        y="total_hours",
        text="distinct_encounters",
        labels={
            "location_category": "ADT location_category",
            "total_hours": "Cumulative dwell hours",
            "distinct_encounters": "Distinct encounters",
        },
        title="Total dwell hours per location_category (bar label = # distinct encounters)",
    )
    fig.update_traces(textposition="outside")
    _max_hours = dwell_summary.select(pl.col("total_hours").max()).item() or 0
    fig.update_layout(
        xaxis_tickangle=-30,
        height=450,
        yaxis_range=[0, _max_hours * 1.15],
    )
    return fig


def _create_los_box(adt_pl: pl.DataFrame) -> go.Figure:
    los_stats = (
        adt_pl.group_by("location_category")
        .agg(
            pl.col("stay_hours").quantile(0.25).alias("q1"),
            pl.col("stay_hours").median().alias("median"),
            pl.col("stay_hours").quantile(0.75).alias("q3"),
            pl.col("stay_hours").min().alias("min_h"),
            pl.col("stay_hours").max().alias("max_h"),
            pl.col("stay_hours").len().alias("n"),
        )
        .sort("median", descending=True)
        .to_dicts()
    )
    categories = [f"{r['location_category']}\n(n={r['n']:,})" for r in los_stats]
    q1 = [r["q1"] for r in los_stats]
    median = [r["median"] for r in los_stats]
    q3 = [r["q3"] for r in los_stats]
    lower = [max(r["min_h"], r["q1"] - 1.5 * (r["q3"] - r["q1"])) for r in los_stats]
    upper = [min(r["max_h"], r["q3"] + 1.5 * (r["q3"] - r["q1"])) for r in los_stats]

    fig = go.Figure(
        data=[
            go.Box(
                x=categories, q1=q1, median=median, q3=q3,
                lowerfence=lower, upperfence=upper,
                boxpoints=False, showlegend=False, marker_color="steelblue",
            )
        ]
    )
    fig.update_layout(
        title="Distribution of individual stay durations by location_category",
        yaxis_title="Per-stay length of stay (hours)",
        xaxis_title="ADT location_category",
        xaxis_tickangle=-45, height=480,
    )
    return fig


def _create_capture_bar(capture_long: pl.DataFrame) -> go.Figure:
    fig = px.bar(
        capture_long.to_pandas(),
        x="location_category", y="pct_events", color="table",
        barmode="group",
        labels={
            "location_category": "ADT location_category",
            "pct_events": "% of events in this table",
            "table": "Source table",
        },
        title="% of events captured in each ADT location_category",
    )
    fig.update_layout(xaxis_tickangle=-30, height=480)
    return fig


def _create_events_per_hour(capture_long: pl.DataFrame) -> go.Figure:
    table_order = ["labs", "vitals", "respiratory_support"]
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[t.replace("_", " ").title() for t in table_order],
        shared_yaxes=False,
    )
    for col_idx, table_label in enumerate(table_order, start=1):
        sub = (
            capture_long.filter(
                (pl.col("table") == table_label)
                & (pl.col("location_category") != NO_MATCH)
                & pl.col("events_per_hour").is_not_null()
            )
            .sort("events_per_hour", descending=True)
            .to_pandas()
        )
        fig.add_trace(
            go.Bar(
                x=sub["location_category"], y=sub["events_per_hour"],
                name=table_label, showlegend=False,
            ),
            row=1, col=col_idx,
        )
        fig.update_yaxes(title_text="events / location-hour", row=1, col=col_idx)
        fig.update_xaxes(tickangle=-30, row=1, col=col_idx)
    fig.update_layout(
        height=460,
        title_text="Instrumentation density: events per hour spent in each location",
    )
    return fig


# ── public entry point ───────────────────────────────────────────────────────


def run_adt_location_coverage(
    clif,
    hospitalization_ids: List[str],
    output_csv_dir: str,
    output_fig_dir: str,
) -> None:
    """Run ADT location coverage analysis for the overall cohort.

    Args:
        clif: ClifOrchestrator with adt and respiratory_support already loaded.
        hospitalization_ids: final_hosp_ids for the overall cohort.
        output_csv_dir: Path to write CSVs (e.g. output/final/overall/tableone/).
        output_fig_dir: Path to write HTML figures (e.g. output/final/overall/figures/).
    """
    print("\n" + "=" * 80)
    print("ADT LOCATION COVERAGE ANALYSIS")
    print("=" * 80)

    hosp_set = set(hospitalization_ids)

    # ── 1. Prepare ADT ──────────────────────────────────────────────────────
    adt_pl = _prepare_adt(clif.adt.df, hospitalization_ids)
    print(f"ADT intervals (cohort): {adt_pl.height:,}")

    adt_for_join = adt_pl.select(
        ["hospitalization_id", "in_dttm", "out_dttm", "location_category"]
    )

    # ── 2. Load event tables ────────────────────────────────────────────────
    # Labs and vitals are loaded via the DuckDB-backed helper instead of
    # clif.load_table. clifpy's load_table reads the full parquet into pandas
    # then filters — at large sites (JHU: 108M labs rows, 75M+ vitals rows)
    # that materializes 30+ GB of pandas per table and OOM-kills inside a
    # constrained SLURM cgroup. load_filtered_clif_table pushes the cohort
    # filter into the parquet scan so only matching rows are read.
    _labs_path = os.path.join(
        clif.data_directory, f"clif_labs.{clif.filetype}"
    )
    _vitals_path = os.path.join(
        clif.data_directory, f"clif_vitals.{clif.filetype}"
    )

    print("Loading labs for coverage analysis...")
    labs_pl = (
        load_filtered_clif_table(
            _labs_path,
            columns=["hospitalization_id", "lab_collect_dttm", "lab_category"],
            hosp_ids=list(hospitalization_ids),
        )
        .drop_nulls(["lab_collect_dttm"])
        .sort(["hospitalization_id", "lab_collect_dttm"])
    )
    print(f"  Labs rows: {labs_pl.height:,}")

    print("Loading vitals for coverage analysis...")
    vitals_pl = (
        load_filtered_clif_table(
            _vitals_path,
            columns=["hospitalization_id", "recorded_dttm", "vital_category"],
            hosp_ids=list(hospitalization_ids),
        )
        .drop_nulls(["recorded_dttm"])
        .sort(["hospitalization_id", "recorded_dttm"])
    )
    print(f"  Vitals rows: {vitals_pl.height:,}")

    # Respiratory support — already loaded + cohort-filtered in generator
    resp_pl = (
        pl.from_pandas(
            clif.respiratory_support.df[
                clif.respiratory_support.df["hospitalization_id"].isin(hosp_set)
            ][["hospitalization_id", "recorded_dttm", "device_category"]]
        )
        .drop_nulls(["recorded_dttm"])
        .sort(["hospitalization_id", "recorded_dttm"])
    )
    print(f"  Respiratory support rows: {resp_pl.height:,}")

    # ── 3. Join events to ADT locations ─────────────────────────────────────
    labs_with_loc = _join_events_to_location(labs_pl, "lab_collect_dttm", adt_for_join)
    vitals_with_loc = _join_events_to_location(vitals_pl, "recorded_dttm", adt_for_join)
    resp_with_loc = _join_events_to_location(resp_pl, "recorded_dttm", adt_for_join)

    # Free raw event frames
    del labs_pl, vitals_pl, resp_pl

    # ── 4. Compute summaries ────────────────────────────────────────────────
    dwell_summary = _compute_dwell_summary(adt_pl)

    capture_long = pl.concat([
        _summarize_capture(labs_with_loc, "labs", dwell_summary),
        _summarize_capture(vitals_with_loc, "vitals", dwell_summary),
        _summarize_capture(resp_with_loc, "respiratory_support", dwell_summary),
    ])

    # Free joined frames
    del labs_with_loc, vitals_with_loc, resp_with_loc

    # ── 5. Write CSVs ──────────────────────────────────────────────────────
    os.makedirs(output_csv_dir, exist_ok=True)
    dwell_path = os.path.join(output_csv_dir, "adt_dwell_summary.csv")
    dwell_summary.write_csv(dwell_path)
    print(f"Wrote {dwell_path}")

    capture_path = os.path.join(output_csv_dir, "adt_event_capture.csv")
    capture_long.write_csv(capture_path)
    print(f"Wrote {capture_path}")

    # ── 6. Write figures (static PNG) ───────────────────────────────────────
    os.makedirs(output_fig_dir, exist_ok=True)
    figures = {
        "adt_dwell_hours_by_location": _create_dwell_bar(dwell_summary),
        "adt_los_distribution_by_location": _create_los_box(adt_pl),
        "adt_event_capture_pct": _create_capture_bar(capture_long),
        "adt_events_per_location_hour": _create_events_per_hour(capture_long),
    }
    for name, fig in figures.items():
        path = os.path.join(output_fig_dir, f"{name}.png")
        fig.write_image(path, scale=2)
        print(f"Wrote {path}")

    # ── 7. Headline summary ─────────────────────────────────────────────────
    for tbl in ("labs", "vitals", "respiratory_support"):
        sub = capture_long.filter(pl.col("table") == tbl)
        total = int(sub["n_events"].sum())
        unmatched = int(
            sub.filter(pl.col("location_category") == NO_MATCH)["n_events"].sum()
        )
        pct = (total - unmatched) / max(total, 1) * 100
        print(f"  {tbl}: {total - unmatched:,}/{total:,} events matched ({pct:.1f}%)")

    print("ADT location coverage analysis complete.\n")
