"""MCS (mechanical circulatory support) table analyzer for CLIF 3.0.

CLIF 3.0 replaced the wide ``ecmo_mcs`` table with the long-format ``mcs`` table
(support_category / device_category / config_category / setting_category /
setting_value). Validation flows through clifpy's ``run_full_dqa`` against the 3.0
mcs schema, so this analyzer is a thin loader.
"""

from clifpy.tables.mcs import Mcs
from .base_table_analyzer import BaseTableAnalyzer
import os
from typing import Dict, Any
from pathlib import Path


class MCSAnalyzer(BaseTableAnalyzer):
    """Analyzer for the CLIF 3.0 MCS table using clifpy."""

    def get_table_name(self) -> str:
        return 'mcs'

    def load_table(self):
        """Load MCS table using clifpy."""
        data_path = Path(self.data_dir)
        file_without_clif = data_path / f"mcs.{self.filetype}"
        file_with_clif = data_path / f"clif_mcs.{self.filetype}"

        if not (file_without_clif.exists() or file_with_clif.exists()):
            print(f"⚠️  No mcs file found in {self.data_dir}")
            self.table = None
            return

        from modules.utils.output_paths import validation_json_reports_dir
        clifpy_output_dir = str(validation_json_reports_dir())
        os.makedirs(clifpy_output_dir, exist_ok=True)

        try:
            self.table = Mcs.from_file(
                data_directory=self.data_dir,
                clif_version=self.clif_version,
                filetype=self.filetype,
                timezone=self.timezone,
                output_directory=clifpy_output_dir
            )
        except Exception as e:
            print(f"⚠️  Error loading mcs table: {e}")
            self.table = None

    def get_data_info(self) -> Dict[str, Any]:
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}
        df = self.table.df
        info = {'row_count': len(df), 'column_count': len(df.columns)}
        if 'hospitalization_id' in df.columns:
            info['unique_hospitalizations'] = df['hospitalization_id'].nunique()
        if 'support_category' in df.columns:
            info['unique_support_categories'] = df['support_category'].nunique()
        return info

    def analyze_distributions(self) -> Dict[str, Any]:
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {}
        return {}
