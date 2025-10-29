"""
Efficient data loading using Polars with lazy evaluation and streaming.

This module provides memory-efficient loading for CLIF tables,
especially when filtering to a sample of hospitalization IDs.
"""

import polars as pl
import pandas as pd
from pathlib import Path
from typing import List, Optional
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.utils.datetime_utils import standardize_datetime_columns


def load_with_filter(
    file_path: str,
    filetype: str,
    hospitalization_ids: List[str],
    timezone: Optional[str] = None,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load data with efficient filtering using Polars lazy loading + streaming.

    This function:
    1. Uses Polars scan (lazy) to avoid loading full file
    2. Applies hospitalization_id filter during scan
    3. Collects with streaming mode for memory efficiency
    4. Returns pandas DataFrame for compatibility

    Parameters
    ----------
    file_path : str
        Path to the data file
    filetype : str
        File type ('parquet' or 'csv')
    hospitalization_ids : list of str
        List of hospitalization IDs to filter to
    timezone : str, optional
        Timezone for datetime conversion
    columns : list of str, optional
        Specific columns to load

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame

    Examples
    --------
    >>> df = load_with_filter(
    ...     'data/clif_vitals.parquet',
    ...     'parquet',
    ...     ['hosp1', 'hosp2', 'hosp3'],
    ...     timezone='US/Central'
    ... )
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Normalize hospitalization IDs to strings
    hosp_ids_str = [str(hid) for hid in hospitalization_ids]

    # Scan the file (lazy - doesn't load data yet)
    if filetype.lower() == 'parquet':
        lazy_df = pl.scan_parquet(file_path)
    elif filetype.lower() == 'csv':
        lazy_df = pl.scan_csv(file_path)
    else:
        raise ValueError(f"Unsupported filetype: {filetype}. Use 'parquet' or 'csv'")

    # Select specific columns if requested
    if columns:
        lazy_df = lazy_df.select(columns)

    # Apply filter for hospitalization_ids
    # This gets pushed down to the file reader - only filtered rows are read!
    lazy_df = lazy_df.filter(
        pl.col('hospitalization_id').is_in(hosp_ids_str)
    )

    # Convert hospitalization_id to Utf8 for consistency
    lazy_df = lazy_df.with_columns([
        pl.col('hospitalization_id').cast(pl.Utf8)
    ])

    # Convert timezone for datetime columns if specified
    if timezone:
        # Standardize all datetime columns to consistent timezone and time unit
        lazy_df = standardize_datetime_columns(
            lazy_df,
            target_timezone=timezone,
            target_time_unit='ns',
            ambiguous='earliest'
        )

    # Collect with streaming mode
    # This processes data in chunks, never materializing the full dataset in RAM
    df_polars = lazy_df.collect(streaming=True)

    # Convert to pandas for compatibility with existing code
    df_pandas = df_polars.to_pandas()

    return df_pandas


def load_full_table(
    file_path: str,
    filetype: str,
    timezone: Optional[str] = None,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load full table without filtering (for non-sample mode).

    Uses streaming collection for memory efficiency even without filters.

    Parameters
    ----------
    file_path : str
        Path to the data file
    filetype : str
        File type ('parquet' or 'csv')
    timezone : str, optional
        Timezone for datetime conversion
    columns : list of str, optional
        Specific columns to load

    Returns
    -------
    pd.DataFrame
        Full DataFrame
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Scan the file (lazy)
    if filetype.lower() == 'parquet':
        lazy_df = pl.scan_parquet(file_path)
    elif filetype.lower() == 'csv':
        lazy_df = pl.scan_csv(file_path)
    else:
        raise ValueError(f"Unsupported filetype: {filetype}. Use 'parquet' or 'csv'")

    # Select specific columns if requested
    if columns:
        lazy_df = lazy_df.select(columns)

    # Convert hospitalization_id to Utf8 for consistency
    if 'hospitalization_id' in lazy_df.schema:
        lazy_df = lazy_df.with_columns([
            pl.col('hospitalization_id').cast(pl.Utf8)
        ])

    # Convert timezone for datetime columns if specified
    if timezone:
        # Standardize all datetime columns to consistent timezone and time unit
        lazy_df = standardize_datetime_columns(
            lazy_df,
            target_timezone=timezone,
            target_time_unit='ns',
            ambiguous='earliest'
        )

    # Collect with streaming mode
    df_polars = lazy_df.collect(streaming=True)

    # Convert to pandas
    df_pandas = df_polars.to_pandas()

    return df_pandas
