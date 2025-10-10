"""Respiratory Support table analyzer using clifpy for CLIF 2.1."""

from clifpy.tables.respiratory_support import RespiratorySupport
from .base_table_analyzer import BaseTableAnalyzer
from typing import Dict, Any
from pathlib import Path


class RespiratorySupportAnalyzer(BaseTableAnalyzer):
    """Analyzer for Respiratory Support table using clifpy."""

    def get_table_name(self) -> str:
        return 'respiratory_support'

    def load_table(self):
        data_path = Path(self.data_dir)
        file_without_clif = data_path / f"respiratory_support.{self.filetype}"
        file_with_clif = data_path / f"clif_respiratory_support.{self.filetype}"

        if not (file_without_clif.exists() or file_with_clif.exists()):
            print(f"⚠️  No respiratory_support file found in {self.data_dir}")
            self.table = None
            return

        try:
            self.table = RespiratorySupport.from_file(
                data_directory=self.data_dir,
                filetype=self.filetype,
                timezone=self.timezone,
                output_directory=self.output_dir
            )
        except Exception as e:
            print(f"⚠️  Error loading respiratory_support table: {e}")
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

    def check_data_quality(self) -> Dict[str, Any]:
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}
        return {}
