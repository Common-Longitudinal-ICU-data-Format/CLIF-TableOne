"""
Labs table analyzer using clifpy for CLIF 2.1.

This module provides basic analysis scaffolding for the labs table.
Table-specific distributions and quality checks to be added after data verification.
"""

from clifpy.tables.labs import Labs
from .base_table_analyzer import BaseTableAnalyzer
from typing import Dict, Any
import pandas as pd
from pathlib import Path


class LabsAnalyzer(BaseTableAnalyzer):
    """
    Analyzer for Labs table using clifpy.

    Basic scaffolding implementation - table-specific logic to be added after
    verifying actual data structure.
    """

    def get_table_name(self) -> str:
        """Return the table name."""
        return 'labs'

    def load_table(self, sample_filter=None):
        """
        Load Labs table using clifpy.

        Parameters:
        -----------
        sample_filter : list, optional
            List of hospitalization_ids to filter to (uses clifpy filters)

        Handles both naming conventions:
        - labs.parquet
        - clif_labs.parquet
        """
        data_path = Path(self.data_dir)
        filetype = self.filetype

        # Check both file naming conventions
        file_without_clif = data_path / f"labs.{filetype}"
        file_with_clif = data_path / f"clif_labs.{filetype}"

        file_exists = file_without_clif.exists() or file_with_clif.exists()

        if not file_exists:
            print(f"⚠️  No labs file found in {self.data_dir}")
            print(f"   Looking for: labs.{filetype} or clif_labs.{filetype}")
            self.table = None
            return

        try:
            # Use filters parameter ONLY when sample is provided
            if sample_filter is not None:
                self.table = Labs.from_file(
                    data_directory=self.data_dir,
                    filetype=self.filetype,
                    timezone=self.timezone,
                    output_directory=self.output_dir,
                    filters={'hospitalization_id': list(sample_filter)}
                )
            else:
                # Normal load without filters
                self.table = Labs.from_file(
                    data_directory=self.data_dir,
                    filetype=self.filetype,
                    timezone=self.timezone,
                    output_directory=self.output_dir
                )
        except FileNotFoundError:
            print(f"⚠️  labs table file not found in {self.data_dir}")
            self.table = None
        except Exception as e:
            print(f"⚠️  Error loading labs table: {e}")
            self.table = None

    def get_data_info(self) -> Dict[str, Any]:
        """
        Get labs basic data information.

        Returns:
            Dictionary containing:
            - row_count: Total number of records
            - column_count: Number of columns
            - unique_hospitalizations: Number of unique hospitalizations (if column exists)
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        df = self.table.df

        info = {
            'row_count': len(df),
            'column_count': len(df.columns),
        }

        # Add unique counts for any ID columns that exist
        if 'hospitalization_id' in df.columns:
            info['unique_hospitalizations'] = df['hospitalization_id'].nunique()

        if 'lab_category' in df.columns:
            info['unique_lab_categories'] = df['lab_category'].nunique()

        if 'lab_order_category' in df.columns:
            info['unique_lab_order_categories'] = df['lab_order_category'].nunique()

        return info

    def analyze_distributions(self) -> Dict[str, Any]:
        """
        Analyze labs distributions.

        Currently returns empty - to be implemented after data structure verification.

        Returns:
            Dictionary containing distribution data (empty for now)
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {}

        distributions = {}

        # TODO: Add distributions after verifying actual data structure
        # Possible distributions:
        # - lab_order_category (8 categories: ABG, BMP, CBC, Coags, LFT, Lactic Acid, Misc, VBG)
        # - lab_category (54+ categories - may need top N)
        # - lab_specimen_category

        return distributions

    def check_data_quality(self) -> Dict[str, Any]:
        """
        Perform labs data quality checks.

        Currently returns empty - to be implemented after data structure verification.

        Returns:
            Dictionary of quality check results (empty for now)
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        quality_checks = {}

        # TODO: Add quality checks after verifying actual data structure
        # Possible checks:
        # - Future datetime checks (lab_order_dttm, lab_collect_dttm, lab_result_dttm)
        # - Invalid lab_order_category values
        # - Invalid lab_category values
        # - Datetime logic checks (e.g., result before collection)
        # - Negative or invalid lab_value_numeric ranges

        return quality_checks
