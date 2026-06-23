"""Microbiology Non-Culture table analyzer using clifpy for CLIF 2.1."""

from clifpy.tables.microbiology_nonculture import MicrobiologyNonculture
from .base_table_analyzer import BaseTableAnalyzer
import os
from typing import Dict, Any
from pathlib import Path


class MicrobiologyNoncultureAnalyzer(BaseTableAnalyzer):
    """Analyzer for Microbiology Non-Culture table using clifpy."""

    def get_table_name(self) -> str:
        return 'microbiology_nonculture'

    def load_table(self):
        """Load Microbiology Non-Culture table using clifpy."""
        data_path = Path(self.data_dir)
        file_without_clif = data_path / f"microbiology_nonculture.{self.filetype}"
        file_with_clif = data_path / f"clif_microbiology_nonculture.{self.filetype}"

        if not (file_without_clif.exists() or file_with_clif.exists()):
            print(f"⚠️  No microbiology_nonculture file found in {self.data_dir}")
            self.table = None
            return

        from modules.utils.output_paths import validation_json_reports_dir
        clifpy_output_dir = str(validation_json_reports_dir())
        os.makedirs(clifpy_output_dir, exist_ok=True)

        try:
            self.table = MicrobiologyNonculture.from_file(
                data_directory=self.data_dir,
                filetype=self.filetype,
                timezone=self.timezone,
                output_directory=clifpy_output_dir,
                clif_version=self.clif_version,
            )
        except Exception as e:
            print(f"⚠️  Error loading microbiology_nonculture table: {e}")
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

        df = self.table.df
        quality_checks = {}

        # Check for duplicate nonculture entries (same hospitalization + result time + organism)
        if all(col in df.columns for col in ['hospitalization_id', 'result_dttm', 'organism_category']):
            duplicates_mask = df.duplicated(subset=['hospitalization_id', 'result_dttm', 'organism_category'], keep=False)
            duplicates = duplicates_mask.sum()

            examples = None
            if duplicates > 0:
                example_cols = ['hospitalization_id', 'result_dttm', 'organism_category', 'result_category']
                example_cols = [col for col in example_cols if col in df.columns]
                examples = df[duplicates_mask][example_cols].head(10)

            quality_checks['duplicate_nonculture_entries'] = {
                'count': int(duplicates),
                'percentage': round((duplicates / len(df) * 100) if len(df) > 0 else 0, 2),
                'status': 'pass' if duplicates == 0 else 'warning',
                'examples': examples
            }

        return quality_checks
