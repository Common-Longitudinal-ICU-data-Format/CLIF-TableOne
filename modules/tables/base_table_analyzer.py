"""
Base Table Analyzer for CLIF 2.1 Tables

This module provides the base class for all table-specific analyzers,
leveraging clifpy's validation capabilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import pandas as pd
import os


class BaseTableAnalyzer(ABC):
    """Base class for all table-specific analyzers."""

    def __init__(self, data_dir: str, filetype: str = 'parquet',
                 timezone: str = 'UTC', output_dir: str = None, **kwargs):
        """
        Initialize the base analyzer.

        Parameters:
        -----------
        data_dir : str
            Directory containing the CLIF data files
        filetype : str
            File format (parquet, csv, feather)
        timezone : str
            Timezone for datetime processing
        output_dir : str, optional
            Directory for output files
        """
        self.data_dir = data_dir
        self.filetype = filetype
        self.timezone = timezone
        self.output_dir = output_dir or 'output'
        self.table = None

        # Ensure output directories exist with new structure (overall/strata/validation/...)
        from modules.utils.output_paths import (
            ensure_output_tree,
            validation_json_reports_dir,
            validation_consolidated_dir,
            validation_feedback_dir,
            validation_monthly_trends_dir,
            PDF_REPORTS,
        )
        os.makedirs(os.path.join(self.output_dir, 'intermediate'), exist_ok=True)
        ensure_output_tree()
        # Concrete dirs we will write into below
        validation_json_reports_dir().mkdir(parents=True, exist_ok=True)
        validation_consolidated_dir().mkdir(parents=True, exist_ok=True)
        validation_feedback_dir().mkdir(parents=True, exist_ok=True)
        validation_monthly_trends_dir().mkdir(parents=True, exist_ok=True)
        PDF_REPORTS.mkdir(parents=True, exist_ok=True)

        # Memory-efficient primary loader: DuckDB -> Arrow -> arrow-backed pandas.
        # Cuts peak RSS roughly 4x vs clifpy's default numpy-object pandas path,
        # and (when CLIF_BACKEND=duckdb is set) routes clifpy's validator onto
        # its duckdb backend so we never spawn a polars copy of the dataframe
        # during validation. See modules/utils/clif_loader.py for details.
        # Set CLIF_LOADER=clifpy to bypass and use clifpy's loader directly.
        if self.filetype == 'parquet' and os.environ.get('CLIF_LOADER', 'duckdb').lower() == 'duckdb':
            try:
                from modules.utils.clif_loader import load_clif_table
                self.table = load_clif_table(
                    table_name=self.get_table_name(),
                    data_dir=self.data_dir,
                    filetype=self.filetype,
                    timezone=self.timezone,
                    output_directory=os.path.join(self.output_dir, 'logs'),
                )
            except Exception as e:
                import logging as _lg
                _lg.getLogger(__name__).warning(
                    "DuckDB loader failed for %s, falling back to clifpy: %s",
                    self.get_table_name(), e,
                )
                self.table = None

        # Fall back to clifpy's loader if the DuckDB path is disabled or failed
        if self.table is None:
            self.load_table()

        # Last-resort: clifpy failed too — try the polars streaming fallback
        if self.table is None and self.filetype == 'parquet':
            self._try_lazy_fallback()

    def _try_lazy_fallback(self):
        """
        Fallback loader for large parquet files that OOM during normal loading.

        Uses Polars streaming + arrow-backed pandas to bypass the
        DuckDB -> numpy-object-dtype pandas intermediate that causes OOM.

        Fallback chain:
          pl.scan_parquet() -> LazyFrame -> collect(streaming=True) -> Polars DF
          -> to_arrow() -> to_pandas(types_mapper=ArrowDtype) -> arrow-backed pandas

        Output is a SimpleNamespace with .df, .schema, .table_name matching
        clifpy's BaseTable interface so all downstream code works unchanged.
        """
        from pathlib import Path
        import logging

        logger = logging.getLogger(__name__)
        table_name = self.get_table_name()
        data_path = Path(self.data_dir)

        # Find the parquet file (both naming conventions)
        file_path = data_path / f"{table_name}.{self.filetype}"
        if not file_path.exists():
            file_path = data_path / f"clif_{table_name}.{self.filetype}"
        if not file_path.exists():
            return  # File not found — not an OOM issue

        try:
            import polars as pl
            import yaml
            import clifpy

            logger.info(f"Attempting streaming fallback for {table_name}...")
            print(f"  ℹ️  Normal loading failed for {table_name}, trying streaming fallback...")

            # Load schema from clifpy's installed schemas
            clifpy_root = os.path.dirname(clifpy.__file__)
            schema_path = os.path.join(clifpy_root, 'schemas', f'{table_name}_schema.yaml')
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = yaml.safe_load(f)

            # Scan parquet lazily
            lf = pl.scan_parquet(str(file_path))
            original_schema = lf.collect_schema()
            col_names = original_schema.names()

            # Apply timezone to naive datetime columns (matches clifpy behavior)
            for col_def in schema.get('columns', []):
                col_name = col_def['name']
                if col_name not in col_names:
                    continue
                if col_def.get('data_type', '').upper() not in ('DATETIME', 'TIMESTAMP'):
                    continue
                dtype = original_schema[col_name]
                if hasattr(dtype, 'time_zone') and dtype.time_zone is None:
                    lf = lf.with_columns(
                        pl.col(col_name).dt.replace_time_zone(
                            self.timezone,
                            non_existent='null',
                            ambiguous='earliest',
                        )
                    )

            # Cast _id columns to string (matches clifpy behavior)
            for col_name in col_names:
                if col_name.endswith('_id'):
                    lf = lf.with_columns(pl.col(col_name).cast(pl.Utf8))

            # Collect with streaming to manage peak memory
            polars_df = lf.collect(engine="streaming")

            # Convert to arrow-backed pandas (~50% less memory than numpy object dtype)
            pandas_df = polars_df.to_arrow().to_pandas(types_mapper=pd.ArrowDtype)
            del polars_df

            # Create table object matching clifpy's BaseTable interface
            from types import SimpleNamespace
            self.table = SimpleNamespace(
                df=pandas_df,
                schema=schema,
                table_name=table_name
            )

            row_count = len(pandas_df)
            print(f"  ✅ Loaded {table_name} via streaming fallback ({row_count:,} rows, arrow-backed)")
            logger.info(f"Loaded {table_name} via streaming fallback ({row_count:,} rows, arrow-backed)")

        except Exception as e:
            logger.warning(f"Streaming fallback failed for {table_name}: {e}")
            print(f"  ⚠️  Streaming fallback failed for {table_name}: {e}")
            self.table = None

    def validate(self, tables=None, hosp_years=None) -> Dict[str, Any]:
        """
        Run full DQA validation (conformance, completeness, plausibility, relational).

        Parameters:
        -----------
        tables : list, optional
            List of loaded BaseTable objects for cross-table checks
            (relational integrity, cross-table plausibility).
            If None, only single-table checks are run.
        hosp_years : set, optional
            Pre-extracted hospitalization years for P.6 temporal consistency.
            When provided, skips scanning the hospitalization table.

        Returns:
        --------
        dict
            Full DQA results with keys: table_name, backend,
            conformance, completeness, relational, plausibility
        """
        from clifpy.utils.validator import run_full_dqa

        table_name = getattr(self.table, 'table_name', None) or self.get_table_name()
        schema_name = table_name.replace('clif_', '')  # Removes 'clif_' prefix if present

        return run_full_dqa(
            df=self.table.df,
            schema=self.table.schema,
            table_name=schema_name,
            tables=tables,
            error_threshold=10.0,
            warning_threshold=1.0,
            hosp_years=hosp_years,
        )

    def extract_cross_table_cache(self) -> Dict[str, Any]:
        """
        Extract lightweight cache for cross-table checks.

        Returns a dict with FK ID sets, temporal subset, and
        hospitalization-specific caches (bounds, years) — used by
        the optimised pipeline to avoid keeping full DataFrames in memory.

        Returns:
        --------
        dict
            Cache dict with keys: table_name, fk_ids, schema_cols,
            temporal_df, hosp_bounds_df, hosp_years.
        """
        from clifpy.utils.validator import extract_cross_table_cache
        return extract_cross_table_cache(self.table)
        # if self.table is None:
        #     return {
        #         'is_valid': False,
        #         'errors': {'schema_errors': [{'type': 'Table Not Loaded',
        #                                      'description': 'Table could not be loaded',
        #                                      'category': 'schema'}]},
        #         'status': 'incomplete',
        #         'data_info': {}
        #     }

        # # Check if using new Polars backend (SimpleNamespace) or clifpy backend
        # from types import SimpleNamespace

        # if isinstance(self.table, SimpleNamespace):
        #     # New Polars backend: Use clifpy validation
        #     from clifpy.utils.validator import validate_dataframe

        #     errors = validate_dataframe(self.table.df, self.table.schema)
        #     is_valid = len(errors) == 0

        # else:
        #     # Clifpy backend: Use clifpy validation
        #     self.table.validate()
        #     errors = self.table.errors if hasattr(self.table, 'errors') else []
        #     is_valid = self.table.isvalid() if hasattr(self.table, 'isvalid') else False

        # return {
        #     'is_valid': is_valid,
        #     'errors': self.format_errors(errors),
        #     'status': self.determine_status(),
        #     'data_info': self.get_data_info()
        # }

    # def format_errors(self, errors: list) -> Dict[str, list]:
    #     """
    #     Format errors into categories like clif_report_card.py.

    #     Parameters:
    #     -----------
    #     errors : list
    #         List of error dictionaries from clifpy

    #     Returns:
    #     --------
    #     dict
    #         Categorized errors (schema_errors, data_quality_issues, other_errors)
    #     """
    #     schema_errors = []
    #     data_quality_issues = []
    #     other_errors = []

    #     for error in errors:
    #         formatted = self.format_single_error(error)

    #         if formatted['category'] == 'schema':
    #             schema_errors.append(formatted)
    #         elif formatted['category'] == 'data_quality':
    #             data_quality_issues.append(formatted)
    #         else:
    #             other_errors.append(formatted)

    #     return {
    #         'schema_errors': schema_errors,
    #         'data_quality_issues': data_quality_issues,
    #         'other_errors': other_errors
    #     }

    # def format_single_error(self, error: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     Format a single error for display.

    #     Parameters:
    #     -----------
    #     error : dict
    #         Error dictionary from clifpy

    #     Returns:
    #     --------
    #     dict
    #         Formatted error with type, description, and category
    #     """
    #     from clifpy.utils.validator import format_clifpy_error

    #     # Get row count if available
    #     row_count = len(self.table.df) if hasattr(self.table, 'df') and self.table.df is not None else 0

    #     # Get table name
    #     table_name = self.get_table_name()

    #     return format_clifpy_error(error, row_count, table_name)

    # def determine_status(self) -> str:
    #     """
    #     Determine validation status (complete/partial/incomplete).

    #     Returns:
    #     --------
    #     str
    #         Status: 'complete', 'partial', or 'incomplete'
    #     """
    #     if not hasattr(self.table, 'errors'):
    #         return 'complete' if self.table and hasattr(self.table, 'df') else 'incomplete'

    #     errors = self.table.errors

    #     if not errors:
    #         return 'complete'

    #     # Format errors to check their types
    #     formatted_errors = [self.format_single_error(e) for e in errors]

    #     # Get required columns from schema if available
    #     required_columns = []
    #     if hasattr(self.table, 'schema') and self.table.schema:
    #         required_columns = self.table.schema.get('required_columns', [])

    #     # Get table name
    #     table_name = self.get_table_name()

    #     from clifpy.utils.validator import determine_validation_status
    #     return determine_validation_status(formatted_errors, required_columns, table_name)

    @abstractmethod
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get table-specific data information.

        Returns:
        --------
        dict
            Information about the loaded data
        """
        pass

    @abstractmethod
    def analyze_distributions(self) -> Dict[str, Any]:
        """
        Analyze table-specific distributions.

        Returns:
        --------
        dict
            Distribution analysis results
        """
        pass

    def calculate_missingness(self) -> Dict[str, Any]:
        """
        Calculate missingness for all columns.

        Returns:
        --------
        dict
            Missingness statistics for the table
        """
        from ..utils.missingness import calculate_missingness

        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {
                'error': 'No data available',
                'total_rows': 0
            }

        return calculate_missingness(self.table.df)

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for the table.

        Returns:
        --------
        dict
            Summary statistics including distributions and missingness
        """
        return {
            'data_info': self.get_data_info(),
            'missingness': self.calculate_missingness(),
            'distributions': self.analyze_distributions()
        }

    def save_intermediate_data(self, df: pd.DataFrame, suffix: str = ''):
        """
        Save intermediate data to output directory.

        Parameters:
        -----------
        df : pd.DataFrame
            Data to save
        suffix : str
            Suffix for the filename
        """
        if df is None or df.empty:
            return

        table_name = self.get_table_name()
        filename = f"{table_name}{suffix}.{self.filetype}"
        filepath = os.path.join(self.output_dir, 'intermediate', filename)

        if self.filetype == 'parquet':
            df.to_parquet(filepath, index=False)
        elif self.filetype == 'csv':
            df.to_csv(filepath, index=False)
        elif self.filetype == 'feather':
            df.to_feather(filepath)

    def save_summary_data(self, summary: Dict[str, Any], suffix: str = ''):
        """Save summary data (e.g. summary statistics) to validation/consolidated/."""
        import json
        from modules.utils.output_paths import validation_consolidated_dir

        table_name = self.get_table_name()
        filename = f"{table_name}_summary{suffix}.json"
        out_dir = validation_consolidated_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        filepath = str(out_dir / filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)

    def save_validation_results(self, dqa_result: Dict[str, Any]):
        """Save DQA validation results (from run_full_dqa) to validation/json_reports/."""
        import json
        from modules.utils.output_paths import validation_json_reports_dir

        table_name = self.get_table_name()
        out_dir = validation_json_reports_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        filepath = str(out_dir / f"{table_name}_dqa.json")

        serializable = {
            'table_name': dqa_result.get('table_name', table_name),
            'backend': dqa_result.get('backend', ''),
        }
        for key in ('conformance', 'completeness', 'relational', 'plausibility'):
            checks = dqa_result.get(key, {})
            serializable[key] = dict(checks) if checks else {}

        if 'table_stats' in dqa_result:
            serializable['table_stats'] = dqa_result['table_stats']
        if 'total_rows' in dqa_result:
            serializable['total_rows'] = dqa_result['total_rows']

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2, default=str)

    def save_monthly_trend_csvs(self, dqa_result: Dict[str, Any]) -> Optional[str]:
        """Extract monthly trends from P.6 result and write CSVs in two places:

        - /intermediate/validation/monthly_trends/<table>_<col>_monthly.csv
          (per-month detail, raw N — local QA only, not shareable because
          single-digit (month, category) cells are a small-cell risk)
        - /final/validation/monthly_trends/<table>_<col>_overall.csv
          (per-category counts collapsed across all months — time dimension
          gone, so the small-cell risk that drives the policy is gone too)
        """
        from modules.utils.output_paths import (
            validation_monthly_trends_dir,
            validation_monthly_trends_raw_dir,
        )
        p6 = dqa_result.get('plausibility', {}).get('category_temporal_consistency', {})
        monthly_trends = p6.get('metrics', {}).get('monthly_trends', {})
        if not monthly_trends:
            return None

        table_name = self.get_table_name()
        raw_dir   = validation_monthly_trends_raw_dir()
        final_dir = validation_monthly_trends_dir()
        raw_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)

        for cat_col, records in monthly_trends.items():
            if not records:
                continue
            df = pd.DataFrame(records)

            # Per-month detail → /intermediate (raw, unsuppressed).
            df.to_csv(raw_dir / f"{table_name}_{cat_col}_monthly.csv", index=False)

            # All-time per-category aggregate → /final.
            # Sum n across months. Drop 'avg' from the overall — its meaning
            # is ambiguous when collapsing time, and the per-month file
            # already carries it for QA.
            if 'n' in df.columns:
                group_cols = [c for c in df.columns if c not in ('month_year', 'n', 'avg')]
                if group_cols:
                    df_overall = (
                        df.groupby(group_cols, dropna=False, as_index=False)
                          .agg(n=('n', 'sum'))
                    )
                    df_overall.to_csv(
                        final_dir / f"{table_name}_{cat_col}_overall.csv",
                        index=False,
                    )

        return str(final_dir)

    def get_table_name(self) -> str:
        """Get the table name from the analyzer class."""
        return self.__class__.__name__.replace('Analyzer', '').lower()