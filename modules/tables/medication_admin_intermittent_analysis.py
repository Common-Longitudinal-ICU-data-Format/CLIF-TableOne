"""
Medication Admin Intermittent table analyzer using clifpy for CLIF 2.1.

This module provides basic analysis scaffolding for the medication_admin_intermittent table.
Table-specific distributions and quality checks to be added after data verification.
"""

from clifpy.tables.medication_admin_intermittent import MedicationAdminIntermittent
from .base_table_analyzer import BaseTableAnalyzer
from typing import Dict, Any
import pandas as pd
from pathlib import Path


class MedicationAdminIntermittentAnalyzer(BaseTableAnalyzer):
    """Analyzer for Medication Admin Intermittent table using clifpy."""

    def get_table_name(self) -> str:
        """Return the table name."""
        return 'medication_admin_intermittent'

    def load_table(self, sample_filter=None):
        """
        Load Medication Admin Intermittent table using clifpy.

        Parameters:
        -----------
        sample_filter : list, optional
            List of hospitalization_ids to filter to (uses clifpy filters)
        """
        data_path = Path(self.data_dir)
        filetype = self.filetype

        file_without_clif = data_path / f"medication_admin_intermittent.{filetype}"
        file_with_clif = data_path / f"clif_medication_admin_intermittent.{filetype}"

        if not (file_without_clif.exists() or file_with_clif.exists()):
            print(f"⚠️  No medication_admin_intermittent file found in {self.data_dir}")
            self.table = None
            return

        try:
            # Use filters parameter ONLY when sample is provided
            if sample_filter is not None:
                self.table = MedicationAdminIntermittent.from_file(
                    data_directory=self.data_dir,
                    filetype=self.filetype,
                    timezone=self.timezone,
                    output_directory=self.output_dir,
                    filters={'hospitalization_id': list(sample_filter)}
                )
            else:
                # Normal load without filters
                self.table = MedicationAdminIntermittent.from_file(
                    data_directory=self.data_dir,
                    filetype=self.filetype,
                    timezone=self.timezone,
                    output_directory=self.output_dir
                )
        except Exception as e:
            print(f"⚠️  Error loading medication_admin_intermittent table: {e}")
            self.table = None

    def get_data_info(self) -> Dict[str, Any]:
        """Get medication_admin_intermittent basic data information."""
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        df = self.table.df
        info = {'row_count': len(df), 'column_count': len(df.columns)}

        if 'hospitalization_id' in df.columns:
            info['unique_hospitalizations'] = df['hospitalization_id'].nunique()

        return info

    def analyze_distributions(self) -> Dict[str, Any]:
        """Analyze medication_admin_intermittent distributions."""
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {}
        return {}  # TODO: Implement after data verification

    def check_data_quality(self) -> Dict[str, Any]:
        """Perform medication_admin_intermittent data quality checks."""
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}
        return {}  # TODO: Implement after data verification
