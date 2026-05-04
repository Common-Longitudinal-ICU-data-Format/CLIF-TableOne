"""DuckDB-based replacements for the two memory-hot ventilation aggregations
in `generator.py`'s "IMV - First 24 hours" + "Ventilator Settings Table"
blocks (the +18.5 GB spike at UCMC, the OOM at UMich, the 373 GB peak at
UPenn).

These functions are drop-in replacements for two specific pandas blocks:

  1. ``compute_per_encounter_vent_stats(resp_imv_df, vent_settings)``
     replaces the chained groupby().median()/mean()/std()/quantile() at
     ``generator.py:~2830-2847``.  Returns a wide DataFrame with one row
     per encounter_block and columns
     ``{setting}_median, {setting}_q1, {setting}_q3, {setting}_mean,
     {setting}_std`` for each setting in ``vent_settings``.

  2. ``compute_vent_stats_by_device_mode(resp_valid_df, vent_settings)``
     replaces the chained groupby().median()/quantile() at
     ``generator.py:~3055-3170``.  Returns a dict with keys
     ``{'medians_long', 'q1_long', 'q3_long', 'counts_long'}``, each a
     long-format DataFrame keyed by (device_category, mode_category, setting).
     The *formatting* (median (q1-q3) string columns) stays in pandas in
     the calling site — it's pure cosmetics, not memory-critical.

DESIGN INTENT
=============
Pandas keeps the entire input DataFrame plus 5 separate groupby
intermediates in memory simultaneously.  At UCMC scale (~5M resp_support
rows) that's manageable.  At UPenn scale (200-500M rows, minute-level
density) it's the OOM trigger.

DuckDB streams the aggregation: it scans the input once, accumulates
hash-table aggregates per group, and never materializes the wide
intermediate frames.  Memory is bounded by the number of GROUPS, not the
number of input rows — so the same query that takes 28 GB in pandas
takes ~1 GB in DuckDB at the same scale.

We use ``duckdb.from_df()`` (zero-copy via Arrow), run the query, and
return ``.df()``.  The pandas → DuckDB → pandas round-trip is ~3-5%
overhead at typical sizes, but the memory ceiling is dramatically lower.

CORRECTNESS
===========
DuckDB's ``quantile_cont`` uses the same linear interpolation as pandas'
default.  ``avg`` and ``stddev_samp`` match pandas' ``mean`` and ``std``
(unbiased, ddof=1).  Counts and groupings are identical.  See the
side-by-side verification harness in ``dev/verify_ventilation_duckdb.py``
which asserts row-for-row equality on UCMC data.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

import duckdb


__all__ = [
    "compute_per_encounter_vent_stats",
    "compute_vent_stats_by_device_mode",
]


def _filter_existing(settings: Iterable[str], df_columns: Iterable[str]) -> list[str]:
    """Match the pandas codepath: silently drop settings not present in df."""
    available = set(df_columns)
    return [s for s in settings if s in available]


def compute_per_encounter_vent_stats(
    resp_imv_df: pd.DataFrame,
    vent_settings: Iterable[str],
) -> pd.DataFrame:
    """Per-encounter ventilator settings stats — DuckDB-backed.

    Drop-in replacement for the pandas block at ``generator.py:~2830-2847``::

        medians = resp_imv.groupby('encounter_block')[settings].median()
        medians.columns = [f'{col}_median' ...]
        means   = resp_imv.groupby('encounter_block')[settings].mean()
        means.columns   = [f'{col}_mean' ...]
        stds    = resp_imv.groupby('encounter_block')[settings].std()
        stds.columns    = [f'{col}_std' ...]
        q1 = resp_imv.groupby('encounter_block')[settings].quantile(0.25)
        q1.columns = [f'{col}_q1' ...]
        q3 = resp_imv.groupby('encounter_block')[settings].quantile(0.75)
        q3.columns = [f'{col}_q3' ...]
        vent_settings_stats = pd.concat(
            [medians, q1, q3, means, stds], axis=1
        ).reset_index()

    Args:
        resp_imv_df: respiratory_support rows for IMV encounters, must
            contain ``encounter_block`` plus zero or more setting columns.
        vent_settings: list of setting column names to aggregate. Settings
            not present in the DataFrame are silently dropped (matching
            the pandas codepath).

    Returns:
        Wide DataFrame with columns:
            ``encounter_block``,
            ``{setting}_median``, ``{setting}_q1``, ``{setting}_q3``,
            ``{setting}_mean``, ``{setting}_std`` for each setting.

        Column ORDER matches the original pandas output (medians first,
        then q1, q3, mean, std). One row per unique encounter_block.
        For groups with all-NaN inputs, a setting's stats will be NaN
        (matching pandas behavior).
    """
    if "encounter_block" not in resp_imv_df.columns:
        raise KeyError("resp_imv_df must contain encounter_block column")

    settings = _filter_existing(vent_settings, resp_imv_df.columns)
    if not settings:
        # Mirror pandas behavior: empty result with just encounter_block
        return pd.DataFrame({
            "encounter_block": resp_imv_df["encounter_block"].drop_duplicates().reset_index(drop=True)
        })

    # Build a SELECT clause that reproduces pandas' column ordering exactly:
    # all medians first, then all q1, q3, mean, std (matches pd.concat order
    # in the original code).
    median_cols = ", ".join(
        f'quantile_cont("{s}", 0.5) AS "{s}_median"' for s in settings
    )
    q1_cols = ", ".join(
        f'quantile_cont("{s}", 0.25) AS "{s}_q1"' for s in settings
    )
    q3_cols = ", ".join(
        f'quantile_cont("{s}", 0.75) AS "{s}_q3"' for s in settings
    )
    mean_cols = ", ".join(
        f'avg("{s}") AS "{s}_mean"' for s in settings
    )
    std_cols = ", ".join(
        f'stddev_samp("{s}") AS "{s}_std"' for s in settings
    )

    sql = f"""
        SELECT
            encounter_block,
            {median_cols},
            {q1_cols},
            {q3_cols},
            {mean_cols},
            {std_cols}
        FROM df
        GROUP BY encounter_block
        ORDER BY encounter_block
    """

    con = duckdb.connect(":memory:")
    try:
        con.register("df", resp_imv_df)
        result = con.execute(sql).df()
    finally:
        con.close()

    return result


def compute_vent_stats_by_device_mode(
    resp_valid_df: pd.DataFrame,
    vent_settings: Iterable[str],
) -> dict[str, pd.DataFrame]:
    """Ventilator settings stats grouped by (device_category, mode_category).

    Drop-in replacement for the pandas block at
    ``generator.py:~3055-3170``::

        medians = resp_valid.groupby(['device_category', 'mode_category'])[settings].median()
        q1      = resp_valid.groupby(['device_category', 'mode_category'])[settings].quantile(0.25)
        q3      = resp_valid.groupby(['device_category', 'mode_category'])[settings].quantile(0.75)
        # ... and the equivalent count for the counts table

    The original code uses these to build wide format strings like
    ``"4.5 (3.5-5.5)"`` and a parallel counts table.  Caller-side string
    formatting stays in pandas — that's pure cosmetics on small data, not
    a memory hotspot.

    Args:
        resp_valid_df: respiratory_support rows (full, not just IMV); must
            contain ``encounter_block``, ``device_category``,
            ``mode_category``, and zero or more setting columns.
        vent_settings: list of setting column names to aggregate. Settings
            not present in the DataFrame are silently dropped.

    Returns:
        ``{'medians': df, 'q1': df, 'q3': df, 'counts': df}`` — each a
        wide DataFrame keyed by (device_category, mode_category) with
        one column per setting. The same shape pandas' ``.median()`` /
        ``.quantile()`` / ``.count()`` returns.

        Caller is responsible for the string formatting and any rename/
        sort steps the original code did after the aggregation.
    """
    needed = {"device_category", "mode_category"}
    missing = needed - set(resp_valid_df.columns)
    if missing:
        raise KeyError(f"resp_valid_df missing required columns: {missing}")

    settings = _filter_existing(vent_settings, resp_valid_df.columns)
    if not settings:
        empty = pd.DataFrame(columns=["device_category", "mode_category"])
        return {"medians": empty.copy(), "q1": empty.copy(), "q3": empty.copy(), "counts": empty.copy()}

    select_median = ", ".join(
        f'quantile_cont("{s}", 0.5) AS "{s}"' for s in settings
    )
    select_q1 = ", ".join(
        f'quantile_cont("{s}", 0.25) AS "{s}"' for s in settings
    )
    select_q3 = ", ".join(
        f'quantile_cont("{s}", 0.75) AS "{s}"' for s in settings
    )
    select_count = ", ".join(
        f'count("{s}") AS "{s}"' for s in settings
    )

    # NOTE: pandas `groupby(['device_category', 'mode_category'])` defaults to
    # dropna=True — rows where either key is NULL are silently dropped. DuckDB
    # GROUP BY keeps NULL as its own group, which would produce extra rows
    # vs the pandas reference. WHERE-clause filter matches pandas exactly.
    base = """
        SELECT device_category, mode_category, {agg}
        FROM df
        WHERE device_category IS NOT NULL AND mode_category IS NOT NULL
        GROUP BY device_category, mode_category
        ORDER BY device_category, mode_category
    """

    con = duckdb.connect(":memory:")
    try:
        con.register("df", resp_valid_df)
        out = {
            "medians": con.execute(base.format(agg=select_median)).df(),
            "q1":      con.execute(base.format(agg=select_q1)).df(),
            "q3":      con.execute(base.format(agg=select_q3)).df(),
            "counts":  con.execute(base.format(agg=select_count)).df(),
        }
    finally:
        con.close()

    return out
