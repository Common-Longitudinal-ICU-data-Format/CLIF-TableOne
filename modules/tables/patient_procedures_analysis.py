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

    def load_table(self):
        """Load Patient Procedures table using clifpy."""
        data_path = Path(self.data_dir)
        file_without_clif = data_path / f"patient_procedures.{self.filetype}"
        file_with_clif = data_path / f"clif_patient_procedures.{self.filetype}"

        if not (file_without_clif.exists() or file_with_clif.exists()):
            print(f"⚠️  No patient_procedures file found in {self.data_dir}")
            self.table = None
            return

        clifpy_output_dir = os.path.join(self.output_dir, "final", "clifpy")
        os.makedirs(clifpy_output_dir, exist_ok=True)

        try:
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

        df = self.table.df
        quality_checks = {}

        # Check for duplicate procedure entries (same hospitalization + billed time + procedure code)
        if all(col in df.columns for col in ['hospitalization_id', 'procedure_billed_dttm', 'procedure_code']):
            duplicates_mask = df.duplicated(subset=['hospitalization_id', 'procedure_billed_dttm', 'procedure_code'], keep=False)
            duplicates = duplicates_mask.sum()

            examples = None
            if duplicates > 0:
                example_cols = ['hospitalization_id', 'procedure_billed_dttm', 'procedure_code', 'procedure_code_format']
                example_cols = [col for col in example_cols if col in df.columns]
                examples = df[duplicates_mask][example_cols].head(10)

            quality_checks['duplicate_procedure_entries'] = {
                'count': int(duplicates),
                'percentage': round((duplicates / len(df) * 100) if len(df) > 0 else 0, 2),
                'status': 'pass' if duplicates == 0 else 'warning',
                'examples': examples
            }

        return quality_checks
