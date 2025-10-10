"""
Hospital Diagnosis table analyzer using clifpy for CLIF 2.1.

This module provides basic analysis scaffolding for the hospital_diagnosis table.
Table-specific distributions and quality checks to be added after data verification.
"""

from clifpy.tables.hospital_diagnosis import HospitalDiagnosis
from .base_table_analyzer import BaseTableAnalyzer
from typing import Dict, Any
import pandas as pd
from pathlib import Path


class HospitalDiagnosisAnalyzer(BaseTableAnalyzer):
    """
    Analyzer for Hospital Diagnosis table using clifpy.

    Basic scaffolding implementation - table-specific logic to be added after
    verifying actual data structure.
    """

    def get_table_name(self) -> str:
        """Return the table name."""
        return 'hospital_diagnosis'

    def load_table(self):
        """
        Load Hospital Diagnosis table using clifpy.

        Handles both naming conventions:
        - hospital_diagnosis.parquet
        - clif_hospital_diagnosis.parquet
        """
        data_path = Path(self.data_dir)
        filetype = self.filetype

        # Check both file naming conventions
        file_without_clif = data_path / f"hospital_diagnosis.{filetype}"
        file_with_clif = data_path / f"clif_hospital_diagnosis.{filetype}"

        file_exists = file_without_clif.exists() or file_with_clif.exists()

        if not file_exists:
            print(f"⚠️  No hospital_diagnosis file found in {self.data_dir}")
            print(f"   Looking for: hospital_diagnosis.{filetype} or clif_hospital_diagnosis.{filetype}")
            self.table = None
            return

        try:
            self.table = HospitalDiagnosis.from_file(
                data_directory=self.data_dir,
                filetype=self.filetype,
                timezone=self.timezone,
                output_directory=self.output_dir
            )
        except FileNotFoundError:
            print(f"⚠️  hospital_diagnosis table file not found in {self.data_dir}")
            self.table = None
        except Exception as e:
            print(f"⚠️  Error loading hospital_diagnosis table: {e}")
            self.table = None

    def get_data_info(self) -> Dict[str, Any]:
        """
        Get hospital_diagnosis basic data information.

        Returns:
            Dictionary containing:
            - row_count: Total number of records
            - column_count: Number of columns
            - unique_patients: Number of unique patients (if patient_id exists)
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        df = self.table.df

        info = {
            'row_count': len(df),
            'column_count': len(df.columns),
        }

        # Add unique counts for any ID columns that exist
        if 'patient_id' in df.columns:
            info['unique_patients'] = df['patient_id'].nunique()

        if 'diagnostic_code' in df.columns:
            info['unique_diagnostic_codes'] = df['diagnostic_code'].nunique()

        return info

    def analyze_distributions(self) -> Dict[str, Any]:
        """
        Analyze hospital_diagnosis distributions.

        Currently returns empty - to be implemented after data structure verification.

        Returns:
            Dictionary containing distribution data (empty for now)
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {}

        distributions = {}

        # TODO: Add distributions after verifying actual data structure
        # Possible distributions:
        # - diagnosis_code_format (if exists)
        # - Primary vs secondary diagnoses (if column exists)

        return distributions

    def check_data_quality(self) -> Dict[str, Any]:
        """
        Perform hospital_diagnosis data quality checks.

        Currently returns empty - to be implemented after data structure verification.

        Returns:
            Dictionary of quality check results (empty for now)
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        quality_checks = {}

        # TODO: Add quality checks after verifying actual data structure
        # Possible checks:
        # - Duplicate diagnoses
        # - Invalid diagnostic codes
        # - Missing required fields

        return quality_checks
