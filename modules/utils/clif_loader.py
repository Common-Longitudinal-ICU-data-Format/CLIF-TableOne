"""
Memory-efficient CLIF parquet loader.

Loads parquet files via DuckDB -> Arrow -> arrow-backed pandas. This produces
DataFrames that are ~4x lighter than the numpy-object pandas frames clifpy's
default ``BaseTable.from_file`` path produces, and avoids the polars
dependency entirely so the same code runs on Windows boxes where polars OOMs
faster than DuckDB.

Why DuckDB instead of polars
----------------------------
- DuckDB streams parquet decoding through its own arena allocator and returns
  buffers that pyarrow can wrap zero-copy.
- DuckDB's heap actually releases pages back to the OS between tables, unlike
  the pandas/numpy/polars Python allocators which fragment and hoard pages on
  long-running processes.
- Some Windows users have reported polars hitting OOM faster than DuckDB on
  the same parquet files; DuckDB has been the more consistent path for them.

Backend choice
--------------
clifpy's validator picks polars when polars is importable. We force it to
``duckdb`` here so validation runs without ever materializing a second
``pl.from_pandas`` copy of the dataframe. Set ``CLIF_BACKEND=polars`` in the
environment to opt back into polars on machines with plenty of RAM.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Set the clifpy validator backend before any validator code runs.
# Doing this at import time means every downstream `from clifpy.utils.validator
# import ...` sees the duckdb backend.
_BACKEND_OVERRIDE = os.environ.get('CLIF_BACKEND', 'duckdb').lower()
try:
    import clifpy.utils.validator as _cv
    if _BACKEND_OVERRIDE in ('duckdb', 'polars'):
        _cv._ACTIVE_BACKEND = _BACKEND_OVERRIDE
        logger.info("clifpy validator backend forced to %s", _BACKEND_OVERRIDE)
except Exception as _e:  # pragma: no cover
    logger.warning("Could not force clifpy backend: %s", _e)


def _load_schema(table_name: str) -> Optional[Dict[str, Any]]:
    """Load a CLIF schema YAML from the installed clifpy package."""
    import yaml
    import clifpy

    schema_path = (
        Path(clifpy.__file__).parent / 'schemas' / f'{table_name}_schema.yaml'
    )
    if not schema_path.exists():
        logger.warning("Schema file not found: %s", schema_path)
        return None
    with open(schema_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _resolve_parquet_path(data_dir: Path, table_name: str, filetype: str) -> Optional[Path]:
    """CLIF files exist as either ``clif_<name>.parquet`` or ``<name>.parquet``."""
    candidates = [
        data_dir / f"clif_{table_name}.{filetype}",
        data_dir / f"{table_name}.{filetype}",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_clif_table(
    table_name: str,
    data_dir: str,
    filetype: str = 'parquet',
    timezone: str = 'UTC',
    output_directory: Optional[str] = None,
    duckdb_memory_limit: str = '8GB',
) -> Optional[SimpleNamespace]:
    """
    Load a CLIF table via DuckDB into an arrow-backed pandas DataFrame.

    Returns
    -------
    SimpleNamespace or None
        Object with ``.df`` (arrow-backed pandas), ``.schema`` (dict), and
        ``.table_name`` (str) — matching clifpy's BaseTable interface so all
        downstream analyzer/validator code works unchanged. Returns ``None``
        if the file is not found or load fails.
    """
    data_path = Path(data_dir)
    file_path = _resolve_parquet_path(data_path, table_name, filetype)
    if file_path is None:
        logger.info("CLIF file not found for %s in %s", table_name, data_dir)
        return None

    schema = _load_schema(table_name) or {}

    try:
        import duckdb
        import pandas as pd
    except ImportError as e:
        logger.error("DuckDB/pandas import failed: %s", e)
        return None

    try:
        con = duckdb.connect()
        con.execute(f"SET memory_limit='{duckdb_memory_limit}'")
        con.execute("SET timezone='UTC'")

        # Discover columns to build a SELECT that casts _id columns to VARCHAR
        # (matches clifpy's _cast_id_cols_to_string) before materializing.
        col_rows = con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{file_path.as_posix()}')"
        ).fetchall()
        col_names = [row[0] for row in col_rows]

        select_clauses = []
        for col in col_names:
            if col.endswith('_id'):
                select_clauses.append(f'CAST("{col}" AS VARCHAR) AS "{col}"')
            else:
                select_clauses.append(f'"{col}"')

        rel = con.sql(
            f"SELECT {', '.join(select_clauses)} "
            f"FROM read_parquet('{file_path.as_posix()}')"
        )

        # fetch_arrow_table returns an Arrow table whose buffers we can wrap
        # zero-copy as arrow-backed pandas — much lighter than fetchdf()'s
        # numpy-object representation.
        arrow_tbl = rel.fetch_arrow_table()

        pandas_df = arrow_tbl.to_pandas(types_mapper=pd.ArrowDtype)

        # Localize naive datetime columns to the requested site tz, matching
        # clifpy's behavior in convert_datetime_columns_to_site_tz.
        if timezone:
            for col_def in schema.get('columns', []):
                cname = col_def.get('name')
                if not cname or cname not in pandas_df.columns:
                    continue
                if col_def.get('data_type', '').upper() not in ('DATETIME', 'TIMESTAMP'):
                    continue
                series = pandas_df[cname]
                # Skip non-datetime dtype (shouldn't happen for DATETIME schema cols)
                if not pd.api.types.is_datetime64_any_dtype(series):
                    continue
                tz = getattr(series.dt, 'tz', None)
                if tz is None:
                    pandas_df[cname] = series.dt.tz_localize(
                        timezone, ambiguous=True, nonexistent='shift_forward'
                    )
                elif str(tz) != str(timezone):
                    pandas_df[cname] = series.dt.tz_convert(timezone)

        # Release the Arrow holder; the pandas frame still wraps the same
        # buffers internally via the ArrowDtype extension arrays.
        del arrow_tbl
        con.close()
        del con, rel

    except Exception as e:
        logger.warning("DuckDB load failed for %s: %s", table_name, e)
        return None

    out_dir = output_directory or 'output'
    os.makedirs(out_dir, exist_ok=True)

    return SimpleNamespace(
        df=pandas_df,
        schema=schema,
        table_name=table_name,
        # mimic clifpy BaseTable's bookkeeping attributes so any downstream code
        # that prods at them doesn't AttributeError
        data_directory=data_dir,
        filetype=filetype,
        timezone=timezone,
        output_directory=out_dir,
        errors=[],
    )
