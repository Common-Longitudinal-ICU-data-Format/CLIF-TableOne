"""Microbiology Susceptibility table analyzer using clifpy for CLIF 2.1."""

from clifpy.tables.microbiology_susceptibility import MicrobiologySusceptibility
from .base_table_analyzer import BaseTableAnalyzer
import os
from typing import Dict, Any
from pathlib import Path


class MicrobiologySusceptibilityAnalyzer(BaseTableAnalyzer):
    """Analyzer for Microbiology Susceptibility table using clifpy."""

    def get_table_name(self) -> str:
        return 'microbiology_susceptibility'

    def load_table(self, sample_filter=None):
        """
        Load Microbiology Susceptibility table using clifpy.

        Note: Susceptibility table only has organism_id (no hospitalization_id),
        so sample_filter is not applicable. The table is loaded in full
        regardless of sample setting, similar to the patient table.

        Parameters:
        -----------
        sample_filter : list, optional
            List of hospitalization_ids (not applicable to susceptibility table)
        """
        data_path = Path(self.data_dir)
        file_without_clif = data_path / f"microbiology_susceptibility.{self.filetype}"
        file_with_clif = data_path / f"clif_microbiology_susceptibility.{self.filetype}"

        if not (file_without_clif.exists() or file_with_clif.exists()):
            print(f"⚠️  No microbiology_susceptibility file found in {self.data_dir}")
            self.table = None
            return

        clifpy_output_dir = os.path.join(self.output_dir, "final", "clifpy")
        os.makedirs(clifpy_output_dir, exist_ok=True)

        try:
            # Load full table without filters (susceptibility doesn't have hospitalization_id)
            self.table = MicrobiologySusceptibility.from_file(
                data_directory=self.data_dir,
                filetype=self.filetype,
                timezone=self.timezone,
                output_directory=clifpy_output_dir
            )
        except Exception as e:
            print(f"⚠️  Error loading microbiology_susceptibility table: {e}")
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
