"""ADT location coverage — standalone marimo notebook.

Answers two questions about a CLIF dataset:

1. **Coverage** — Of all rows in `labs`, `vitals`, and `respiratory_support`,
   what fraction were recorded while the patient was in each ADT
   `location_category` (icu, ward, ed, stepdown, ...)?
2. **Dwell time** — How much cumulative time do patients spend in each
   `location_category`?

The notebook is fully self-contained: it only depends on `clifpy`, `polars`,
and `plotly` (plus `marimo` for the interactive UI mode).

==============================================================================
HOW TO RUN
==============================================================================

  1. Edit the four constants in the FIRST CELL below (DATA_DIRECTORY,
     FILETYPE, TIMEZONE, OUTPUT_DIR).

  2a. As an interactive notebook:
        marimo edit dev/adt_location_coverage.py

  2b. As a plain script (no marimo UI needed — just writes figures to
      OUTPUT_DIR):
        python dev/adt_location_coverage.py

==============================================================================
"""

import marimo

__generated_with = "0.22.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    from plotly.subplots import make_subplots

    from clifpy.tables.adt import Adt
    from clifpy.tables.labs import Labs
    from clifpy.tables.respiratory_support import RespiratorySupport
    from clifpy.tables.vitals import Vitals

    return Adt, Labs, RespiratorySupport, Vitals, go, make_subplots, mo, pl, px


@app.cell
def _(mo):
    mo.md(r"""
    # ADT location coverage

    For every row in **labs**, **vitals**, and **respiratory_support**, find
    the ADT `location_category` the patient was in at the time of the event.
    Then summarize:

    - **% of events captured per location** (and how many fall outside any
      ADT interval — a data-quality signal).
    - **Total dwell hours per location** across the cohort.
    - **Events per location-hour** — a normalized "instrumentation density"
      metric that controls for how long patients spend in each unit.

    Edit the four constants in the cell below, then either re-run the
    notebook here or close it and run `python dev/adt_location_coverage.py`
    from the project root.
    """)
    return


@app.cell
def _():
    # =========================================================================
    # EDIT THESE FOUR LINES, then re-run.
    # =========================================================================
    DATA_DIRECTORY = "/Users/dema/Downloads/2.1.0"           # absolute path to your CLIF tables directory
    FILETYPE = "parquet"          # "parquet" | "csv" | "feather"
    TIMEZONE = "UTC"              # e.g. "UTC", "US/Central"
    OUTPUT_DIR = "dev/output/adt_location_coverage"  # where figures get saved
    # =========================================================================
    return DATA_DIRECTORY, FILETYPE, OUTPUT_DIR, TIMEZONE


@app.cell
def _(Adt, DATA_DIRECTORY, FILETYPE, TIMEZONE, mo, pl):
    if not DATA_DIRECTORY:
        mo.stop(
            True,
            mo.md(
                "⚠️ Set **`DATA_DIRECTORY`** at the top of the notebook to "
                "the absolute path of your CLIF tables directory, then re-run."
            ),
        )

    adt_pdf = Adt.from_file(
        data_directory=DATA_DIRECTORY,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        columns=[
            "hospitalization_id",
            "in_dttm",
            "out_dttm",
            "location_category",
        ],
    ).df

    active_source = f"{DATA_DIRECTORY} ({FILETYPE}, {TIMEZONE})"

    adt_pl = (
        pl.from_pandas(adt_pdf)
        .drop_nulls(["in_dttm", "out_dttm", "location_category"])
        .filter(pl.col("out_dttm") > pl.col("in_dttm"))
        .with_columns(
            ((pl.col("out_dttm") - pl.col("in_dttm")).dt.total_seconds() / 3600.0)
            .alias("stay_hours")
        )
        .sort(["hospitalization_id", "in_dttm"])
    )
    return active_source, adt_pl


@app.cell
def _(active_source, mo):
    mo.md(f"""
    **Active data source:** `{active_source}`
    """)
    return


@app.cell
def _(
    DATA_DIRECTORY,
    FILETYPE,
    Labs,
    RespiratorySupport,
    TIMEZONE,
    Vitals,
    adt_pl,
    pl,
):
    # Load the three event tables from the configured directory.
    labs_df_raw = Labs.from_file(
        data_directory=DATA_DIRECTORY,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        columns=["hospitalization_id", "lab_collect_dttm", "lab_category"],
    ).df
    vitals_df_raw = Vitals.from_file(
        data_directory=DATA_DIRECTORY,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        columns=["hospitalization_id", "recorded_dttm", "vital_category"],
    ).df
    resp_df_raw = RespiratorySupport.from_file(
        data_directory=DATA_DIRECTORY,
        filetype=FILETYPE,
        timezone=TIMEZONE,
        columns=["hospitalization_id", "recorded_dttm", "device_category"],
    ).df

    labs_pl = (
        pl.from_pandas(labs_df_raw)
        .drop_nulls(["lab_collect_dttm"])
        .sort(["hospitalization_id", "lab_collect_dttm"])
    )

    vitals_pl = (
        pl.from_pandas(vitals_df_raw)
        .drop_nulls(["recorded_dttm"])
        .sort(["hospitalization_id", "recorded_dttm"])
    )

    resp_pl = (
        pl.from_pandas(resp_df_raw)
        .drop_nulls(["recorded_dttm"])
        .sort(["hospitalization_id", "recorded_dttm"])
    )

    # Lightweight summary so the notebook reports load sizes up front.
    table_sizes = pl.DataFrame(
        {
            "table": ["adt", "labs", "vitals", "respiratory_support"],
            "rows": [adt_pl.height, labs_pl.height, vitals_pl.height, resp_pl.height],
            "distinct_hospitalizations": [
                adt_pl.select(pl.col("hospitalization_id").n_unique()).item(),
                labs_pl.select(pl.col("hospitalization_id").n_unique()).item(),
                vitals_pl.select(pl.col("hospitalization_id").n_unique()).item(),
                resp_pl.select(pl.col("hospitalization_id").n_unique()).item(),
            ],
        }
    )
    return labs_pl, resp_pl, table_sizes, vitals_pl


@app.cell
def _(mo, table_sizes):
    mo.vstack(
        [
            mo.md("## Tables loaded"),
            mo.ui.table(table_sizes.to_pandas(), selection=None),
        ]
    )
    return


@app.cell
def _(adt_pl, pl):
    # Per-location dwell-time summary.
    dwell_summary = (
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
    return (dwell_summary,)


@app.cell
def _(adt_pl, pl):
    # Reusable interval-join helper.
    # Backward asof finds the most recent stay starting at-or-before the
    # event timestamp; we then check the event still falls before that
    # stay's out_dttm. Events that fall in a gap or have no matching
    # hospitalization are bucketed as `__no_adt_match__` so they show up in
    # the summary as a data-quality flag.
    NO_MATCH = "__no_adt_match__"

    adt_for_join = adt_pl.select(
        ["hospitalization_id", "in_dttm", "out_dttm", "location_category"]
    )

    def join_to_location(events: "pl.DataFrame", ts_col: str) -> "pl.DataFrame":
        joined = events.join_asof(
            adt_for_join,
            left_on=ts_col,
            right_on="in_dttm",
            by="hospitalization_id",
            strategy="backward",
        )
        return joined.with_columns(
            pl.when(
                pl.col("out_dttm").is_not_null() & (pl.col(ts_col) < pl.col("out_dttm"))
            )
            .then(pl.col("location_category"))
            .otherwise(pl.lit(NO_MATCH))
            .alias("location_category")
        )

    return NO_MATCH, join_to_location


@app.cell
def _(join_to_location, labs_pl, resp_pl, vitals_pl):
    labs_with_loc = join_to_location(labs_pl, "lab_collect_dttm")
    vitals_with_loc = join_to_location(vitals_pl, "recorded_dttm")
    resp_with_loc = join_to_location(resp_pl, "recorded_dttm")
    return labs_with_loc, resp_with_loc, vitals_with_loc


@app.cell
def _(NO_MATCH, dwell_summary, pl):
    # Build a per-table capture-% + events-per-hour table.
    def summarize_capture(joined: "pl.DataFrame", table_label: str) -> "pl.DataFrame":
        total = joined.height
        per_loc = (
            joined.group_by("location_category")
            .agg(pl.len().alias("n_events"))
            .with_columns(
                (pl.col("n_events") / max(total, 1) * 100).alias("pct_events"),
                pl.lit(table_label).alias("table"),
            )
        )
        # Bring in dwell hours for matched locations and compute events/hour.
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

    return (summarize_capture,)


@app.cell
def _(labs_with_loc, pl, resp_with_loc, summarize_capture, vitals_with_loc):
    capture_long = pl.concat(
        [
            summarize_capture(labs_with_loc, "labs"),
            summarize_capture(vitals_with_loc, "vitals"),
            summarize_capture(resp_with_loc, "respiratory_support"),
        ]
    )
    return (capture_long,)


@app.cell
def _(NO_MATCH, capture_long, dwell_summary, mo, pl):
    # Headline numbers.
    def headline_for(table_label: str) -> dict:
        sub = capture_long.filter(pl.col("table") == table_label)
        total = int(sub["n_events"].sum())
        unmatched = int(
            sub.filter(pl.col("location_category") == NO_MATCH)["n_events"].sum()
        )
        matched = total - unmatched
        pct_matched = (matched / total * 100) if total else 0.0
        return {
            "table": table_label,
            "total_events": total,
            "matched_events": matched,
            "unmatched_events": unmatched,
            "pct_matched": pct_matched,
        }

    headlines = [headline_for(t) for t in ("labs", "vitals", "respiratory_support")]
    total_dwell_hours = float(dwell_summary["total_hours"].sum())
    total_dwell_days = total_dwell_hours / 24.0

    bullets = "\n".join(
        f"- **{h['table']}**: {h['matched_events']:,} / {h['total_events']:,} events "
        f"({h['pct_matched']:.1f}%) fall inside an ADT interval. "
        f"{h['unmatched_events']:,} unmatched."
        for h in headlines
    )

    mo.md(
        f"""
        ## Headline numbers

        - **Total cohort dwell time across all locations:** {total_dwell_hours:,.0f} hours ({total_dwell_days:,.1f} days)

        {bullets}
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Dwell time per ADT location
    """)
    return


@app.cell
def _(dwell_summary, px):
    fig_dwell = px.bar(
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
    fig_dwell.update_traces(textposition="outside")
    fig_dwell.update_layout(xaxis_tickangle=-30, height=450)
    fig_dwell
    return (fig_dwell,)


@app.cell
def _(adt_pl, go, pl):
    # Pre-aggregate stay_hours quantiles per location and feed a single
    # go.Box trace with parallel arrays. plotly's default px.box ships every
    # raw row to the browser, which can blow past marimo's 10 MB output cap
    # on real cohorts. Fences use the standard 1.5×IQR Tukey rule, clamped
    # to the observed min/max.
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

    _categories = [f"{r['location_category']}<br>(n={r['n']:,})" for r in los_stats]
    _q1 = [r["q1"] for r in los_stats]
    _median = [r["median"] for r in los_stats]
    _q3 = [r["q3"] for r in los_stats]
    _lower = [
        max(r["min_h"], r["q1"] - 1.5 * (r["q3"] - r["q1"])) for r in los_stats
    ]
    _upper = [
        min(r["max_h"], r["q3"] + 1.5 * (r["q3"] - r["q1"])) for r in los_stats
    ]

    fig_los_box = go.Figure(
        data=[
            go.Box(
                x=_categories,
                q1=_q1,
                median=_median,
                q3=_q3,
                lowerfence=_lower,
                upperfence=_upper,
                boxpoints=False,
                showlegend=False,
                marker_color="steelblue",
            )
        ]
    )
    fig_los_box.update_layout(
        title="Distribution of individual stay durations by location_category",
        yaxis_title="Per-stay length of stay (hours)",
        xaxis_title="ADT location_category",
        xaxis_tickangle=0,
        height=480,
    )
    fig_los_box
    return (fig_los_box,)


@app.cell
def _(dwell_summary, mo):
    mo.ui.table(
        dwell_summary.select(
            [
                "location_category",
                "n_stays",
                "distinct_encounters",
                "total_hours",
                "total_days",
                "median_stay_hours",
                "q1_stay_hours",
                "q3_stay_hours",
            ]
        ).to_pandas(),
        selection=None,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Event capture by ADT location
    """)
    return


@app.cell
def _(capture_long, px):
    fig_capture = px.bar(
        capture_long.to_pandas(),
        x="location_category",
        y="pct_events",
        color="table",
        barmode="group",
        labels={
            "location_category": "ADT location_category",
            "pct_events": "% of events in this table",
            "table": "Source table",
        },
        title="% of events captured in each ADT location_category",
    )
    fig_capture.update_layout(xaxis_tickangle=-30, height=480)
    fig_capture
    return (fig_capture,)


@app.cell
def _(NO_MATCH, capture_long, make_subplots, pl):
    import plotly.graph_objects as _go

    table_order = ["labs", "vitals", "respiratory_support"]
    fig_eph = make_subplots(
        rows=1,
        cols=3,
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
        fig_eph.add_trace(
            _go.Bar(
                x=sub["location_category"],
                y=sub["events_per_hour"],
                name=table_label,
                showlegend=False,
            ),
            row=1,
            col=col_idx,
        )
        fig_eph.update_yaxes(title_text="events / location-hour", row=1, col=col_idx)
        fig_eph.update_xaxes(tickangle=-30, row=1, col=col_idx)
    fig_eph.update_layout(
        height=460,
        title_text="Instrumentation density: events per hour spent in each location",
    )
    fig_eph
    return (fig_eph,)


@app.cell
def _(capture_long, mo):
    mo.ui.table(
        capture_long.select(
            ["table", "location_category", "n_events", "pct_events", "events_per_hour"]
        )
        .sort(["table", "n_events"], descending=[False, True])
        .to_pandas(),
        selection=None,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    ## Export figures to disk

    Charts are written to `OUTPUT_DIR` (set at the top) as interactive
    HTML. Static PNGs are written too if `kaleido` is installed
    (`pip install kaleido`).
    """)
    return


@app.cell
def _(
    OUTPUT_DIR,
    fig_capture,
    fig_dwell,
    fig_eph,
    fig_los_box,
    mo,
):
    from pathlib import Path as _Path

    out_dir = _Path(OUTPUT_DIR).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try to load kaleido for PNG export; fall back gracefully if missing.
    try:
        import kaleido  # noqa: F401

        png_available = True
        png_warning = ""
    except ImportError:
        png_available = False
        png_warning = (
            "\n\n> ⚠️ `kaleido` is not installed, so only HTML files were written. "
            "Run `pip install kaleido` (or `uv add kaleido`) to also export PNGs."
        )

    figs = [
        (fig_dwell, "dwell_hours_by_location"),
        (fig_los_box, "los_distribution_by_location"),
        (fig_capture, "event_capture_pct"),
        (fig_eph, "events_per_location_hour"),
    ]

    written_files = []
    for fig, base in figs:
        html_path = out_dir / f"{base}.html"
        fig.write_html(html_path, include_plotlyjs="cdn")
        written_files.append(html_path)
        if png_available:
            png_path = out_dir / f"{base}.png"
            fig.write_image(png_path, scale=2)
            written_files.append(png_path)

    file_list = "\n".join(f"- `{p}`" for p in written_files)
    mo.md(
        f"""
        **Wrote {len(written_files)} file(s) to** `{out_dir}`:

        {file_list}
        {png_warning}
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ---

    ### Notes on the join

    - Events are matched to ADT stays using a polars `join_asof` (backward
      strategy) by `hospitalization_id`, then filtered to require
      `event_dttm < out_dttm`. Events that fall in a gap between stays — or
      whose `hospitalization_id` is absent from ADT — get bucketed as
      `__no_adt_match__`.
    - Dwell hours per stay are computed as
      `(out_dttm - in_dttm).total_seconds() / 3600`. Stays with
      `out_dttm <= in_dttm` are dropped as malformed.
    - "Events per location-hour" divides the per-location event count by
      the cohort-wide cumulative dwell hours in that location, giving a
      rough instrumentation-density rate.
    """)
    return


if __name__ == "__main__":
    app.run()
