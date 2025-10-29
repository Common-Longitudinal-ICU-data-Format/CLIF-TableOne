"""
Unified datetime handling utilities for consistent timezone and time unit conversions.

This module provides a single source of truth for all datetime conversions across the codebase,
ensuring consistent handling of timezones, time units, and ambiguous times during DST transitions.
"""

import polars as pl
from typing import Union, List, Optional
import logging

logger = logging.getLogger(__name__)


def standardize_datetime_columns(
    df: Union[pl.DataFrame, pl.LazyFrame],
    target_timezone: str,
    target_time_unit: str = 'ns',
    ambiguous: str = 'earliest',
    datetime_columns: Optional[List[str]] = None
) -> Union[pl.DataFrame, pl.LazyFrame]:
    """
    Standardize all datetime columns to consistent timezone and time unit.

    This function handles three cases for each datetime column:
    1. **UTC datetime** → Convert to target timezone
    2. **Naive datetime** → Assume it's in target timezone, make it timezone-aware
    3. **Already in target timezone** → No change needed

    Additionally, it standardizes the time unit (default: nanoseconds) to prevent
    precision mismatch errors during joins.

    Parameters
    ----------
    df : pl.DataFrame or pl.LazyFrame
        Input dataframe
    target_timezone : str
        Target timezone from config (e.g., 'US/Eastern', 'America/New_York')
    target_time_unit : str, default='ns'
        Target time unit for all datetime columns ('ms', 'us', 'ns')
        Recommended: 'ns' (nanoseconds) for maximum precision
    ambiguous : str, default='earliest'
        How to handle ambiguous times during DST transitions:
        - 'earliest': Use earlier occurrence (recommended)
        - 'latest': Use later occurrence
        - 'raise': Raise error
        - 'null': Set ambiguous times to null
    datetime_columns : list of str, optional
        Specific datetime columns to convert. If None, auto-detects all datetime columns.

    Returns
    -------
    pl.DataFrame or pl.LazyFrame
        Dataframe with standardized datetime columns

    Examples
    --------
    >>> # Standardize all datetime columns to US/Eastern timezone
    >>> df = standardize_datetime_columns(df, target_timezone='US/Eastern')

    >>> # Convert specific columns only
    >>> df = standardize_datetime_columns(
    ...     df,
    ...     target_timezone='US/Eastern',
    ...     datetime_columns=['recorded_dttm', 'admin_dttm']
    ... )
    """
    is_lazy = isinstance(df, pl.LazyFrame)

    # Get schema to identify datetime columns
    schema = df.schema if not is_lazy else df.collect_schema()

    # Auto-detect datetime columns if not specified
    if datetime_columns is None:
        datetime_columns = [
            col_name for col_name, dtype in schema.items()
            if isinstance(dtype, pl.Datetime) or (isinstance(dtype, type) and issubclass(dtype, pl.Datetime))
        ]

    if not datetime_columns:
        logger.debug("No datetime columns found to standardize")
        return df

    logger.info(f"Standardizing {len(datetime_columns)} datetime column(s) to {target_timezone} with time unit {target_time_unit}")

    # Build conversion expressions
    conversions = []

    for col_name in datetime_columns:
        if col_name not in schema:
            logger.warning(f"Column '{col_name}' not found in dataframe, skipping")
            continue

        dtype = schema[col_name]

        # Skip if not a datetime column
        if not (isinstance(dtype, pl.Datetime) or (isinstance(dtype, type) and issubclass(dtype, pl.Datetime))):
            logger.warning(f"Column '{col_name}' is not a datetime column, skipping")
            continue

        # Build conversion expression based on current timezone state
        conversion_expr = _build_datetime_conversion_expr(
            col_name,
            dtype,
            target_timezone,
            target_time_unit,
            ambiguous
        )

        conversions.append(conversion_expr)

    if conversions:
        df = df.with_columns(conversions)
        logger.info(f"Successfully standardized {len(conversions)} datetime column(s)")

    return df


def _build_datetime_conversion_expr(
    col_name: str,
    dtype: pl.Datetime,
    target_timezone: str,
    target_time_unit: str,
    ambiguous: str
) -> pl.Expr:
    """
    Build Polars expression for datetime conversion based on current state.

    Parameters
    ----------
    col_name : str
        Column name
    dtype : pl.Datetime
        Current datetime type
    target_timezone : str
        Target timezone
    target_time_unit : str
        Target time unit
    ambiguous : str
        Ambiguous time handling strategy

    Returns
    -------
    pl.Expr
        Polars expression for conversion
    """
    dtype_str = str(dtype)
    current_tz = None

    # Extract current timezone from dtype
    if 'time_zone=' in dtype_str:
        # Extract timezone value from dtype string like "Datetime(time_zone='UTC')"
        tz_start = dtype_str.find('time_zone=') + len('time_zone=')
        tz_value = dtype_str[tz_start:].split(',')[0].split(')')[0].strip('\'"')
        if tz_value != 'None':
            current_tz = tz_value

    # Case 1: Naive datetime (no timezone)
    if current_tz is None:
        logger.debug(f"{col_name}: Naive datetime → Assuming {target_timezone}, making timezone-aware")
        expr = (
            pl.col(col_name)
            .cast(pl.Datetime(target_time_unit))  # Standardize time unit first
            .dt.replace_time_zone(target_timezone, ambiguous=ambiguous)  # Localize to target timezone
            .alias(col_name)
        )

    # Case 2: Already in target timezone
    elif current_tz == target_timezone:
        logger.debug(f"{col_name}: Already in {target_timezone} → Standardizing time unit only")
        expr = (
            pl.col(col_name)
            .cast(pl.Datetime(target_time_unit, target_timezone))  # Just standardize time unit
            .alias(col_name)
        )

    # Case 3: Different timezone (e.g., UTC → target timezone)
    else:
        logger.debug(f"{col_name}: {current_tz} → Converting to {target_timezone}")
        expr = (
            pl.col(col_name)
            .cast(pl.Datetime(target_time_unit, current_tz))  # Standardize time unit first
            .dt.convert_time_zone(target_timezone)  # Convert to target timezone
            .alias(col_name)
        )

    return expr


def ensure_datetime_precision_match(
    df1: Union[pl.DataFrame, pl.LazyFrame],
    df2: Union[pl.DataFrame, pl.LazyFrame],
    df1_datetime_col: str,
    df2_datetime_col: str,
    target_timezone: str,
    target_time_unit: str = 'ns'
) -> tuple:
    """
    Ensure two dataframes have matching datetime precision for joins.

    This is specifically useful before join_asof operations which require
    exact datetime type matching.

    Parameters
    ----------
    df1 : pl.DataFrame or pl.LazyFrame
        First dataframe
    df2 : pl.DataFrame or pl.LazyFrame
        Second dataframe
    df1_datetime_col : str
        Datetime column name in df1
    df2_datetime_col : str
        Datetime column name in df2
    target_timezone : str
        Target timezone
    target_time_unit : str, default='ns'
        Target time unit

    Returns
    -------
    tuple
        (df1_standardized, df2_standardized) with matching datetime precision

    Examples
    --------
    >>> # Prepare dataframes for join_asof
    >>> labs, resp = ensure_datetime_precision_match(
    ...     labs, resp,
    ...     'lab_result_dttm', 'recorded_dttm',
    ...     target_timezone='US/Eastern'
    ... )
    >>> # Now safe to join
    >>> result = labs.join_asof(resp, left_on='lab_result_dttm', right_on='recorded_dttm')
    """
    logger.info(f"Ensuring datetime precision match: {df1_datetime_col} ↔ {df2_datetime_col}")

    # Standardize both dataframes
    df1 = standardize_datetime_columns(
        df1,
        target_timezone=target_timezone,
        target_time_unit=target_time_unit,
        datetime_columns=[df1_datetime_col]
    )

    df2 = standardize_datetime_columns(
        df2,
        target_timezone=target_timezone,
        target_time_unit=target_time_unit,
        datetime_columns=[df2_datetime_col]
    )

    logger.info(f"Datetime precision matched at {target_time_unit} in {target_timezone}")

    return df1, df2


def standardize_datetime_for_comparison(
    col: pl.Expr,
    target_timezone: str,
    target_time_unit: str = 'ns'
) -> pl.Expr:
    """
    Standardize a datetime column expression for comparisons in lazy operations.

    Useful for filtering or comparing datetime columns in lazy dataframes.

    Parameters
    ----------
    col : pl.Expr
        Datetime column expression
    target_timezone : str
        Target timezone
    target_time_unit : str, default='ns'
        Target time unit

    Returns
    -------
    pl.Expr
        Standardized datetime expression

    Examples
    --------
    >>> # In a lazy filter operation
    >>> lazy_df = lazy_df.filter(
    ...     standardize_datetime_for_comparison(pl.col('recorded_dttm'), 'US/Eastern')
    ...     > pl.datetime(2020, 1, 1)
    ... )
    """
    # This is a simplified version - for lazy operations,
    # we convert to target timezone for comparisons
    return col.dt.convert_time_zone(target_timezone)
