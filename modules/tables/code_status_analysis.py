"""
Code Status Table Analyzer for CLIF 2.1

This module provides specific analysis for the Code Status table using clifpy.
"""

from typing import Dict, Any, Optional
import pandas as pd
from .base_table_analyzer import BaseTableAnalyzer


class CodeStatusAnalyzer(BaseTableAnalyzer):
    """Analyzer for Code Status table using clifpy."""

    def get_table_name(self) -> str:
        """Return the correct table name with underscore."""
        return 'code_status'

    def load_table(self):
        """Load Code Status table using clifpy."""
        try:
            from clifpy.tables.code_status import CodeStatus
            import os

            # Check for both naming conventions
            possible_names = ['code_status', 'clif_code_status']
            file_found = False

            for name in possible_names:
                file_path = os.path.join(self.data_dir, f"{name}.{self.filetype}")
                if os.path.exists(file_path):
                    file_found = True
                    break

            if not file_found:
                print(f"Warning: Could not find code_status.{self.filetype} or clif_code_status.{self.filetype} in {self.data_dir}")
                self.table = None
                return

            # Clifpy saves files directly to output_directory, so pass the final subdirectory
            clifpy_output_dir = os.path.join(self.output_dir, 'final')
            os.makedirs(clifpy_output_dir, exist_ok=True)

            self.table = CodeStatus.from_file(
                data_directory=self.data_dir,
                filetype=self.filetype,
                timezone=self.timezone,
                output_directory=clifpy_output_dir
            )

            # Move any CSV files that clifpy created in parent directory to final/
            self._move_clifpy_csvs_to_final()

        except ImportError:
            print("Warning: clifpy not installed or CodeStatus table not available. Install with: pip install clifpy")
            self.table = None
        except Exception as e:
            print(f"Error loading Code Status table: {e}")
            self.table = None

    def _move_clifpy_csvs_to_final(self):
        """Move any CSV files created by clifpy from output/ to output/final/"""
        import os
        import shutil

        parent_dir = self.output_dir
        final_dir = os.path.join(parent_dir, 'final')

        if not os.path.exists(parent_dir):
            return

        clifpy_csv_patterns = ['missing_data_stats_', 'validation_errors_']

        for filename in os.listdir(parent_dir):
            if filename.endswith('.csv'):
                for pattern in clifpy_csv_patterns:
                    if pattern in filename:
                        source = os.path.join(parent_dir, filename)
                        dest = os.path.join(final_dir, filename)
                        try:
                            shutil.move(source, dest)
                            print(f"Moved {filename} to final/")
                        except Exception as e:
                            print(f"Could not move {filename}: {e}")
                        break

    def get_data_info(self) -> Dict[str, Any]:
        """
        Get code_status-specific data information.

        Returns:
        --------
        dict
            Information about the code status data
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {
                'error': 'No data available',
                'row_count': 0,
                'unique_patients': 0,
                'columns': [],
                'unique_code_statuses': 0
            }

        df = self.table.df

        # Get date range for code status changes
        date_range = {}
        if 'start_dttm' in df.columns:
            df['start_dttm'] = pd.to_datetime(df['start_dttm'])
            first_year = int(df['start_dttm'].dt.year.min()) if not df['start_dttm'].isna().all() else None
            last_year = int(df['start_dttm'].dt.year.max()) if not df['start_dttm'].isna().all() else None
            date_range = {
                'first_year': first_year,
                'last_year': last_year,
                'duration_years': (last_year - first_year + 1) if first_year and last_year else 0
            }

        return {
            'row_count': len(df),
            'unique_patients': df['patient_id'].nunique() if 'patient_id' in df.columns else 0,
            'unique_code_statuses': df['code_status_category'].nunique() if 'code_status_category' in df.columns else 0,
            'columns': list(df.columns),
            'column_count': len(df.columns),
            **date_range
        }

    def analyze_distributions(self) -> Dict[str, Any]:
        """
        Analyze code status distributions.

        Returns:
        --------
        dict
            Distribution analysis for code status categories
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        df = self.table.df
        distributions = {}

        # Analyze code status categories
        if 'code_status_category' in df.columns:
            distributions['code_status_category'] = self.get_categorical_distribution('code_status_category')

        # Analyze year distribution if we have date data
        if 'start_dttm' in df.columns:
            df['start_dttm'] = pd.to_datetime(df['start_dttm'])
            df['year'] = df['start_dttm'].dt.year
            year_counts = df['year'].value_counts().sort_index()

            distributions['year_distribution'] = {
                'years': year_counts.index.tolist(),
                'counts': year_counts.values.tolist(),
                'total': len(df[df['year'].notna()])
            }

        return distributions

    def get_categorical_distribution(self, column: str) -> Dict[str, Any]:
        """
        Get distribution for a categorical column.

        Parameters:
        -----------
        column : str
            Column name to analyze

        Returns:
        --------
        dict
            Distribution statistics for the column
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        df = self.table.df

        if column not in df.columns:
            return {'error': f'Column {column} not found'}

        # Get value counts
        value_counts = df[column].value_counts()

        # Calculate percentages
        total = len(df)
        percentages = (value_counts / total * 100).round(2)

        # Get missing count
        missing_count = df[column].isna().sum()

        return {
            'values': value_counts.index.tolist(),
            'counts': value_counts.values.tolist(),
            'percentages': percentages.values.tolist(),
            'missing_count': int(missing_count),
            'missing_percentage': round((missing_count / total * 100) if total > 0 else 0, 2),
            'unique_values': int(df[column].nunique()),
            'total_rows': total
        }

    def check_data_quality(self) -> Dict[str, Any]:
        """
        Perform data quality checks specific to Code Status table.

        Returns:
        --------
        dict
            Data quality check results
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        df = self.table.df
        quality_checks = {}

        # Check for future code status dates
        if 'start_dttm' in df.columns:
            df['start_dttm'] = pd.to_datetime(df['start_dttm'])
            future_dates_mask = df['start_dttm'] > pd.Timestamp.now(tz=self.timezone)

            examples = None
            if future_dates_mask.sum() > 0:
                example_cols = ['patient_id', 'start_dttm', 'code_status_category']
                example_cols = [col for col in example_cols if col in df.columns]
                examples = df[future_dates_mask][example_cols].head(10)

            quality_checks['future_code_status_dates'] = {
                'count': int(future_dates_mask.sum()),
                'percentage': round((future_dates_mask.sum() / len(df) * 100) if len(df) > 0 else 0, 2),
                'status': 'pass' if future_dates_mask.sum() == 0 else 'warning',
                'examples': examples
            }

        # Check for invalid code status categories
        if 'code_status_category' in df.columns:
            valid_categories = ['DNR', 'DNAR', 'UDNR', 'DNR/DNI', 'DNAR/DNI', 'AND', 'Full', 'Presume Full', 'Other']
            invalid_mask = ~df['code_status_category'].isin(valid_categories + [pd.NA, None])

            examples = None
            if invalid_mask.sum() > 0:
                example_cols = ['patient_id', 'code_status_category', 'start_dttm']
                example_cols = [col for col in example_cols if col in df.columns]
                examples = df[invalid_mask][example_cols].head(10)

            quality_checks['invalid_code_status_categories'] = {
                'count': int(invalid_mask.sum()),
                'percentage': round((invalid_mask.sum() / len(df) * 100) if len(df) > 0 else 0, 2),
                'status': 'pass' if invalid_mask.sum() == 0 else 'error',
                'examples': examples
            }

        return quality_checks
