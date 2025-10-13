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

        Parameters:
        -----------
        sample_filter : list, optional
            List of hospitalization_ids to filter to (uses clifpy filters)
        """
        data_path = Path(self.data_dir)
        file_without_clif = data_path / f"microbiology_susceptibility.{self.filetype}"
        file_with_clif = data_path / f"clif_microbiology_susceptibility.{self.filetype}"

        if not (file_without_clif.exists() or file_with_clif.exists()):
            print(f"⚠️  No microbiology_susceptibility file found in {self.data_dir}")
            self.table = None
            return

        # Clifpy saves files directly to output_directory, so pass the final/clifpy subdirectory


        clifpy_output_dir = os.path.join(self.output_dir, "final", "clifpy")


        os.makedirs(clifpy_output_dir, exist_ok=True)



        try:
            # Use filters parameter ONLY when sample is provided
            if sample_filter is not None:
                self.table = MicrobiologySusceptibility.from_file(
                    data_directory=self.data_dir,
                    filetype=self.filetype,
                    timezone=self.timezone,
                    output_directory=clifpy_output_dir,
                    filters={'hospitalization_id': list(sample_filter)}
                )
            else:
                # Normal load without filters
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
