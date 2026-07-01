"""Generic schema-driven analyzer for CLIF 3.0 tables.

Many CLIF 3.0 tables (line, drain, airway, output, input, radiology, ...) do not
need bespoke analytics — validation flows through clifpy's ``run_full_dqa`` against
the versioned schema. This module provides a single ``GenericTableAnalyzer`` plus a
``make_generic_analyzer`` factory so those tables can be registered for validation
without a hand-written file each.

The clifpy table class is resolved dynamically from the orchestrator's
``TABLE_CLASSES`` registry, and loading falls back to the DuckDB/streaming loaders in
``BaseTableAnalyzer`` (which are schema-driven) when the primary path is used.
"""

import os
from pathlib import Path
from typing import Dict, Any

from .base_table_analyzer import BaseTableAnalyzer


class GenericTableAnalyzer(BaseTableAnalyzer):
    """Schema-driven analyzer for a CLIF 3.0 table with no bespoke analyzer.

    Subclasses set ``TABLE_NAME``; use :func:`make_generic_analyzer` to build one.
    """

    TABLE_NAME: str = None

    def get_table_name(self) -> str:
        if not self.TABLE_NAME:
            raise ValueError("GenericTableAnalyzer requires a TABLE_NAME")
        return self.TABLE_NAME

    def load_table(self):
        """Load the table via its clifpy class (fallback path; the DuckDB loader in
        BaseTableAnalyzer handles parquet by default)."""
        name = self.get_table_name()
        data_path = Path(self.data_dir)
        if not ((data_path / f"{name}.{self.filetype}").exists()
                or (data_path / f"clif_{name}.{self.filetype}").exists()):
            print(f"⚠️  No {name} file found in {self.data_dir}")
            self.table = None
            return

        try:
            from clifpy.clif_orchestrator import TABLE_CLASSES
        except Exception as e:  # pragma: no cover - clifpy always present
            print(f"⚠️  Could not import clifpy TABLE_CLASSES: {e}")
            self.table = None
            return

        table_class = TABLE_CLASSES.get(name)
        if table_class is None:
            print(f"⚠️  No clifpy table class registered for {name}")
            self.table = None
            return

        from modules.utils.output_paths import validation_json_reports_dir
        clifpy_output_dir = str(validation_json_reports_dir())
        os.makedirs(clifpy_output_dir, exist_ok=True)

        try:
            self.table = table_class.from_file(
                data_directory=self.data_dir,
                clif_version=self.clif_version,
                filetype=self.filetype,
                timezone=self.timezone,
                output_directory=clifpy_output_dir,
            )
        except Exception as e:
            print(f"⚠️  Error loading {name} table: {e}")
            self.table = None

    def get_data_info(self) -> Dict[str, Any]:
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}
        df = self.table.df
        info = {'row_count': len(df), 'column_count': len(df.columns)}
        if 'hospitalization_id' in df.columns:
            info['unique_hospitalizations'] = df['hospitalization_id'].nunique()
        return info

    def analyze_distributions(self) -> Dict[str, Any]:
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {}
        return {}


def make_generic_analyzer(table_name: str) -> type:
    """Build a GenericTableAnalyzer subclass bound to ``table_name``."""
    class_name = ''.join(part.capitalize() for part in table_name.split('_')) + 'Analyzer'
    return type(class_name, (GenericTableAnalyzer,), {'TABLE_NAME': table_name})
