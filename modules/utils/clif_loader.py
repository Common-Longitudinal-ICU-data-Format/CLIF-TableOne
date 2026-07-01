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


def _get_available_ram() -> int:
    """Best-effort available RAM in bytes."""
    try:
        import psutil
        return psutil.virtual_memory().available
    except ImportError:
        pass
    try:
        pages = os.sysconf('SC_PHYS_PAGES')
        page_size = os.sysconf('SC_PAGE_SIZE')
        return pages * page_size
    except (AttributeError, ValueError):
        pass
    # Fallback: assume 8 GB
    return 8 * 1024 ** 3


class ClifDB:
    """Shared DuckDB connection for querying CLIF parquet files.

    One instance per pipeline run keeps parquet metadata cached across
    queries and allows in-memory DataFrames (e.g. ICU time windows) to
    be registered once and joined in every subsequent query.

    For the ECDF hot loop (called ~200× per run), the backend is auto-
    selected: **polars** when it is installed and data fits in RAM,
    **DuckDB** otherwise.  Override with ``CLIF_ECDF_BACKEND=polars``
    or ``CLIF_ECDF_BACKEND=duckdb``.
    """

    def __init__(
        self,
        tables_path: str,
        file_type: str = 'parquet',
        memory_limit: str = None,
    ):
        import duckdb as _duckdb

        self.tables_path = tables_path
        self.file_type = file_type
        self._con = _duckdb.connect()

        # Auto-size DuckDB memory limit: 50% of available RAM, floor 4 GB
        if memory_limit is None:
            available = _get_available_ram()
            auto_bytes = max(4 * 1024 ** 3, int(available * 0.5))
            memory_limit = f'{auto_bytes // (1024 ** 3)}GB'
            logger.info("DuckDB memory_limit auto-set to %s (available RAM: %.1f GB)",
                        memory_limit, available / 1e9)

        self._con.execute(f"SET memory_limit='{memory_limit}'")
        self._con.execute("SET timezone='UTC'")

        self._use_polars = self._should_use_polars()
        self._pl_icu_windows = None  # set by register() when polars active

    # -- backend detection ---------------------------------------------------

    def _should_use_polars(self) -> bool:
        override = os.environ.get('CLIF_ECDF_BACKEND', '').lower()
        if override == 'duckdb':
            logger.info("ECDF backend forced to DuckDB via CLIF_ECDF_BACKEND")
            return False
        if override == 'polars':
            logger.info("ECDF backend forced to polars via CLIF_ECDF_BACKEND")
            return True

        try:
            import polars  # noqa: F401
        except ImportError:
            logger.info("polars not installed; ECDF hot loop will use DuckDB")
            return False

        hot_tables = ['labs', 'vitals', 'respiratory_support']
        total_bytes = 0
        for t in hot_tables:
            try:
                total_bytes += os.path.getsize(self.table_path(t))
            except (FileNotFoundError, OSError):
                pass

        available = _get_available_ram()
        threshold = available * 0.50
        use_pl = total_bytes < threshold

        logger.info(
            "ECDF hot-loop backend: %s  (parquet %.1f GB, avail RAM %.1f GB, "
            "threshold %.1f GB)",
            'polars' if use_pl else 'DuckDB',
            total_bytes / 1e9, available / 1e9, threshold / 1e9,
        )
        return use_pl

    # -- path helpers --------------------------------------------------------

    def table_path(self, table_name: str) -> str:
        """Resolve parquet path for a CLIF table (e.g. ``'labs'`` ->
        ``'/data/clif_labs.parquet'``).  Raises ``FileNotFoundError``
        if neither ``clif_<name>.<ext>`` nor ``<name>.<ext>`` exists.
        """
        path = _resolve_parquet_path(
            Path(self.tables_path), table_name, self.file_type,
        )
        if path is None:
            raise FileNotFoundError(
                f"CLIF file not found for '{table_name}' in {self.tables_path}"
            )
        return path.as_posix()

    # -- query interface -----------------------------------------------------

    def register(self, name: str, df) -> None:
        """Register a pandas/Arrow DataFrame as a virtual table for joins."""
        self._con.register(name, df)
        if name == 'icu_windows' and self._use_polars:
            import polars as _pl
            self._pl_icu_windows = _pl.from_pandas(df)

    def query_df(self, sql: str, params=None):
        """Execute *sql*, return a **pandas** ``DataFrame``."""
        import pandas as _pd

        return self._con.execute(sql, params or []).fetchdf()

    def query_numpy(self, sql: str, params=None) -> dict:
        """Execute *sql*, return ``dict[str, np.ndarray]``."""
        return self._con.execute(sql, params or []).fetchnumpy()

    def query_arrow(self, sql: str, params=None):
        """Execute *sql*, return a ``pyarrow.Table``."""
        return self._con.execute(sql, params or []).fetch_arrow_table()

    # -- ECDF hot-loop helper ------------------------------------------------

    def fetch_icu_values(
        self,
        parquet_path: str,
        value_col: str,
        datetime_col: str,
        category_col: str = None,
        category_value: str = None,
        unit=None,
    ):
        """Fetch values joined with ``icu_windows`` + temporal filter.

        This is the ECDF hot-loop operation (called ~200× per run).
        Automatically uses polars or DuckDB based on the backend chosen
        at init time.

        Parameters
        ----------
        parquet_path : str
            Absolute path to the CLIF parquet file.
        value_col : str
            Column containing the numeric values to return.
        datetime_col : str
            Timestamp column for temporal filtering.
        category_col : str, optional
            Category column to filter on (e.g. ``'lab_category'``).
            When *None*, the filter is ``value_col IS NOT NULL``
            (used for respiratory support columns).
        category_value : str, optional
            Value to match in *category_col* (lowercase, stripped).
        unit : str or list or None
            Lab unit filter.  Ignored when *category_col* is None.

        Returns
        -------
        numpy.ndarray
            1-D array of numeric values.
        """
        import numpy as _np

        if self._use_polars:
            return self._fetch_via_polars(
                parquet_path, value_col, datetime_col,
                category_col, category_value, unit,
            )
        return self._fetch_via_duckdb(
            parquet_path, value_col, datetime_col,
            category_col, category_value, unit,
        )

    # -- polars implementation (fast, in-memory) -----------------------------

    def _fetch_via_polars(
        self, parquet_path, value_col, datetime_col,
        category_col, category_value, unit,
    ):
        import polars as pl

        lf = pl.scan_parquet(parquet_path)

        if category_col is not None:
            # Category filter
            lf = lf.filter(
                pl.col(category_col).str.to_lowercase().str.strip_chars()
                == category_value
            )
            # Unit filter (labs only)
            if unit is not None:
                lf = lf.filter(self._pl_unit_filter(unit))
        else:
            # Respiratory: just require non-null on the value column
            lf = lf.filter(pl.col(value_col).is_not_null())

        lf = lf.select(['hospitalization_id', datetime_col, value_col])

        # Join with ICU time windows
        lf = lf.join(
            self._pl_icu_windows.lazy(),
            on='hospitalization_id',
            how='inner',
        )

        # Temporal filter (strip timezone for comparison)
        lf = lf.filter(
            (pl.col(datetime_col).dt.replace_time_zone(None) >= pl.col('in_dttm'))
            & (pl.col(datetime_col).dt.replace_time_zone(None) <= pl.col('out_dttm'))
        ).select([value_col])

        return lf.collect(streaming=True)[value_col].to_numpy()

    @staticmethod
    def _pl_unit_filter(unit):
        """Build a polars filter expression for lab unit matching."""
        import polars as pl

        if isinstance(unit, list):
            no_units = [u for u in unit if u.lower().strip() == '(no units)']
            real_units = [u.lower().strip() for u in unit
                          if u.lower().strip() != '(no units)']
            parts = []
            if real_units:
                parts.append(
                    pl.col('reference_unit').str.to_lowercase()
                    .str.strip_chars().is_in(real_units)
                )
            if no_units:
                parts.append(
                    pl.col('reference_unit').is_null()
                    | (pl.col('reference_unit').str.strip_chars() == '')
                    | (pl.col('reference_unit').str.to_lowercase()
                       .str.strip_chars() == '(no units)')
                )
            expr = parts[0]
            for p in parts[1:]:
                expr = expr | p
            return expr

        unit_l = unit.lower().strip()
        if unit_l == '(no units)':
            return (
                pl.col('reference_unit').is_null()
                | (pl.col('reference_unit').str.strip_chars() == '')
                | (pl.col('reference_unit').str.to_lowercase()
                   .str.strip_chars() == '(no units)')
            )
        return pl.col('reference_unit').str.to_lowercase().str.strip_chars() == unit_l

    # -- DuckDB implementation (safe, disk-spilling) -------------------------

    def _fetch_via_duckdb(
        self, parquet_path, value_col, datetime_col,
        category_col, category_value, unit,
    ):
        if category_col is not None:
            unit_clause, unit_params = self._duckdb_unit_filter(unit)
            sql = f"""
                SELECT d.{value_col}
                FROM read_parquet(?) AS d
                INNER JOIN icu_windows AS w USING (hospitalization_id)
                WHERE LOWER(TRIM(d.{category_col})) = ?
                  AND {unit_clause}
                  AND d.{datetime_col}::TIMESTAMP BETWEEN w.in_dttm AND w.out_dttm
            """
            params = [parquet_path, category_value] + unit_params
        else:
            sql = f"""
                SELECT d.{value_col}
                FROM read_parquet(?) AS d
                INNER JOIN icu_windows AS w USING (hospitalization_id)
                WHERE d.{value_col} IS NOT NULL
                  AND d.{datetime_col}::TIMESTAMP BETWEEN w.in_dttm AND w.out_dttm
            """
            params = [parquet_path]

        result = self._con.execute(sql, params).fetchnumpy()
        return result[value_col]

    @staticmethod
    def _duckdb_unit_filter(unit):
        """Return ``(sql_fragment, params)`` for lab unit WHERE clause."""
        if unit is None:
            return 'TRUE', []

        if isinstance(unit, list):
            no_units = [u for u in unit if u.lower().strip() == '(no units)']
            real_units = [u.lower().strip() for u in unit
                          if u.lower().strip() != '(no units)']
            parts, params = [], []
            if real_units:
                ph = ', '.join(['?'] * len(real_units))
                parts.append(f"LOWER(TRIM(d.reference_unit)) IN ({ph})")
                params.extend(real_units)
            if no_units:
                parts.append(
                    "(d.reference_unit IS NULL "
                    "OR TRIM(d.reference_unit) = '' "
                    "OR LOWER(TRIM(d.reference_unit)) = '(no units)')"
                )
            return '(' + ' OR '.join(parts) + ')', params

        unit_l = unit.lower().strip()
        if unit_l == '(no units)':
            return (
                "(d.reference_unit IS NULL "
                "OR TRIM(d.reference_unit) = '' "
                "OR LOWER(TRIM(d.reference_unit)) = '(no units)')"
            ), []
        return "LOWER(TRIM(d.reference_unit)) = ?", [unit_l]

    # -- batch ECDF helpers (Phase: ECDF optimization) -------------------------

    def preload_icu_joined_view(self, table_name: str, value_col: str,
                                 datetime_col: str, category_col: str = None) -> str:
        """Create a DuckDB temp view pre-joined with icu_windows + temporal filter.

        Returns the view name.  Subsequent queries against this view avoid
        re-reading parquet metadata and re-joining on every call.
        """
        parquet_path = self.table_path(table_name)
        view_name = f"_icu_{table_name}"
        cols = f"d.hospitalization_id, d.{value_col}, d.{datetime_col}"
        if category_col:
            cols += f", LOWER(TRIM(d.{category_col})) AS _category"
        if table_name == 'labs':
            cols += ", LOWER(TRIM(d.reference_unit)) AS _unit"

        sql = f"""
            CREATE OR REPLACE TEMP VIEW {view_name} AS
            SELECT {cols}
            FROM read_parquet('{parquet_path}') AS d
            INNER JOIN icu_windows AS w USING (hospitalization_id)
            WHERE d.{datetime_col}::TIMESTAMP BETWEEN w.in_dttm AND w.out_dttm
        """
        if category_col is None:
            sql += f" AND d.{value_col} IS NOT NULL"
        self._con.execute(sql)
        logger.info("Registered ICU-joined view '%s' for %s", view_name, table_name)
        return view_name

    def fetch_batch_categories(self, view_name: str, value_col: str,
                                categories: list, table_type: str = 'labs') -> dict:
        """Fetch values for ALL categories in one query from a pre-joined view.

        Returns dict of ``{(category, unit_or_None): numpy.ndarray}``.
        """
        import numpy as _np

        if table_type == 'labs':
            sql = f"""
                SELECT _category, _unit, {value_col}
                FROM {view_name}
                WHERE _category IS NOT NULL
            """
            result = self._con.execute(sql).fetchdf()
            out = {}
            for (cat, unit), grp in result.groupby(['_category', '_unit']):
                vals = grp[value_col].dropna().to_numpy()
                if len(vals) > 0:
                    out[(cat, unit)] = vals
            return out
        else:
            # vitals — no unit column
            sql = f"""
                SELECT _category, {value_col}
                FROM {view_name}
                WHERE _category IS NOT NULL
            """
            result = self._con.execute(sql).fetchdf()
            out = {}
            for cat, grp in result.groupby('_category'):
                vals = grp[value_col].dropna().to_numpy()
                if len(vals) > 0:
                    out[(cat, None)] = vals
            return out

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        self._con.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _load_schema(table_name: str, clif_version: str = '3.0') -> Optional[Dict[str, Any]]:
    """Load a versioned CLIF schema YAML from the installed clifpy package."""
    import yaml
    import clifpy

    schema_path = (
        Path(clifpy.__file__).parent / 'schemas' / clif_version / f'{table_name}_schema.yaml'
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
    clif_version: str = '3.0',
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

    schema = _load_schema(table_name, clif_version) or {}

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
                # A DATETIME schema column should already be datetime-typed out
                # of DuckDB+Arrow, but on Windows hosts with broken tzdata the
                # column can sneak through as object/string. Coerce with
                # utc=True so we preserve the CLIF timezone convention (DuckDB
                # stores UTC). Fall back to tz-naive only if even utc=True
                # fails — at least the column is then datetime-typed and
                # downstream .dt math still works.
                if not pd.api.types.is_datetime64_any_dtype(series):
                    try:
                        series = pd.to_datetime(series, errors='coerce', utc=True)
                        pandas_df[cname] = series
                        logger.warning(
                            "Coerced non-datetime column %s.%s to UTC datetime — "
                            "likely a tzdata misconfiguration on the host.",
                            table_name, cname,
                        )
                    except Exception as e_utc:
                        try:
                            series = pd.to_datetime(series, errors='coerce')
                            pandas_df[cname] = series
                            logger.warning(
                                "Coerced %s.%s to tz-naive datetime (utc=True "
                                "failed: %s). Downstream tz-aware ops may need "
                                "their own guard.",
                                table_name, cname, e_utc,
                            )
                        except Exception as e:
                            logger.warning(
                                "Could not coerce %s.%s to datetime: %s — "
                                "downstream .dt ops may still fail.",
                                table_name, cname, e,
                            )
                            continue
                tz = getattr(series.dt, 'tz', None)
                try:
                    if tz is None:
                        pandas_df[cname] = series.dt.tz_localize(
                            timezone, ambiguous=True, nonexistent='shift_forward'
                        )
                    elif str(tz) != str(timezone):
                        pandas_df[cname] = series.dt.tz_convert(timezone)
                except Exception as e:
                    # On a broken-tzdata host, tz_localize/tz_convert can fail
                    # even when the column is datetime-typed. Leave the column
                    # as-is (tz-naive or in whatever tz it arrived in) rather
                    # than poisoning the whole load. Subtraction/.dt math still
                    # works for same-tz pairs.
                    logger.warning(
                        "Could not apply timezone %s to %s.%s: %s — "
                        "leaving column as-is. Install the 'tzdata' package "
                        "on Windows to resolve.",
                        timezone, table_name, cname, e,
                    )

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
