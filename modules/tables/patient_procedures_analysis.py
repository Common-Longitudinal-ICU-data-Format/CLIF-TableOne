"""Patient Procedures table analyzer using clifpy for CLIF 2.1."""

from clifpy.tables.patient_procedures import PatientProcedures
from .base_table_analyzer import BaseTableAnalyzer
import os
from typing import Dict, Any
from pathlib import Path


class PatientProceduresAnalyzer(BaseTableAnalyzer):
    """Analyzer for Patient Procedures table using clifpy."""

    def get_table_name(self) -> str:
        return 'patient_procedures'

    def load_table(self, sample_filter=None):
        """
        Load Patient Procedures table using clifpy.

        Parameters:
        -----------
        sample_filter : list, optional
            List of hospitalization_ids to filter to (uses clifpy filters)
        """
        data_path = Path(self.data_dir)
        file_without_clif = data_path / f"patient_procedures.{self.filetype}"
        file_with_clif = data_path / f"clif_patient_procedures.{self.filetype}"

        if not (file_without_clif.exists() or file_with_clif.exists()):
            print(f"⚠️  No patient_procedures file found in {self.data_dir}")
            self.table = None
            return

        # Clifpy saves files directly to output_directory, so pass the final/clifpy subdirectory


        clifpy_output_dir = os.path.join(self.output_dir, "final", "clifpy")


        os.makedirs(clifpy_output_dir, exist_ok=True)



        try:
            # Use filters parameter ONLY when sample is provided
            if sample_filter is not None:
                self.table = PatientProcedures.from_file(
                    data_directory=self.data_dir,
                    filetype=self.filetype,
                    timezone=self.timezone,
                    output_directory=clifpy_output_dir,
                    filters={'hospitalization_id': list(sample_filter)}
                )
            else:
                # Normal load without filters
                self.table = PatientProcedures.from_file(
                    data_directory=self.data_dir,
                    filetype=self.filetype,
                    timezone=self.timezone,
                    output_directory=clifpy_output_dir
                )
        except Exception as e:
            print(f"⚠️  Error loading patient_procedures table: {e}")
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
