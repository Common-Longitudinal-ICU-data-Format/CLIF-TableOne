"""
Hospitalization Table Analyzer for CLIF 2.1

This module provides specific analysis for the Hospitalization table using clifpy.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from .base_table_analyzer import BaseTableAnalyzer


class HospitalizationAnalyzer(BaseTableAnalyzer):
    """Analyzer for Hospitalization table using clifpy."""

    def load_table(self, sample_filter=None):
        """
        Load Hospitalization table using clifpy.

        Parameters:
        -----------
        sample_filter : list, optional
            List of hospitalization_ids to filter to (uses clifpy filters)
        """
        try:
            from clifpy.tables.hospitalization import Hospitalization
            import os

            # Check for both naming conventions
            possible_names = ['hospitalization', 'clif_hospitalization']
            file_found = False

            for name in possible_names:
                file_path = os.path.join(self.data_dir, f"{name}.{self.filetype}")
                if os.path.exists(file_path):
                    file_found = True
                    break

            if not file_found:
                print(f"Warning: Could not find hospitalization.{self.filetype} or clif_hospitalization.{self.filetype} in {self.data_dir}")
                self.table = None
                return

            # Clifpy saves files directly to output_directory, so pass the final subdirectory
            clifpy_output_dir = os.path.join(self.output_dir, "final", "clifpy")
            os.makedirs(clifpy_output_dir, exist_ok=True)

            # Use filters parameter ONLY when sample is provided
            if sample_filter is not None:
                self.table = Hospitalization.from_file(
                    data_directory=self.data_dir,
                    filetype=self.filetype,
                    timezone=self.timezone,
                    output_directory=clifpy_output_dir,
                    filters={'hospitalization_id': list(sample_filter)}
                )
            else:
                self.table = Hospitalization.from_file(
                    data_directory=self.data_dir,
                    filetype=self.filetype,
                    timezone=self.timezone,
                    output_directory=clifpy_output_dir
                )

            # Move any CSV files that clifpy created in parent directory to final/
            self._move_clifpy_csvs_to_final()

        except ImportError:
            print("Warning: clifpy not installed. Install with: pip install clifpy")
            self.table = None
        except Exception as e:
            print(f"Error loading Hospitalization table: {e}")
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
        Get hospitalization-specific data information.

        Returns:
        --------
        dict
            Information about the hospitalization data
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {
                'error': 'No data available',
                'row_count': 0,
                'unique_hospitalizations': 0,
                'unique_patients': 0,
                'columns': [],
                'first_admission_year': None,
                'last_admission_year': None
            }

        df = self.table.df

        # Get first and last admission years
        first_admission_year = None
        last_admission_year = None
        if 'admission_dttm' in df.columns:
            valid_admissions = df['admission_dttm'].dropna()
            if len(valid_admissions) > 0:
                first_admission_year = valid_admissions.min().year
                last_admission_year = valid_admissions.max().year

        return {
            'row_count': len(df),
            'unique_hospitalizations': df['hospitalization_id'].nunique() if 'hospitalization_id' in df.columns else 0,
            'unique_patients': df['patient_id'].nunique() if 'patient_id' in df.columns else 0,
            'columns': list(df.columns),
            'column_count': len(df.columns),
            'first_admission_year': first_admission_year,
            'last_admission_year': last_admission_year
        }

    def analyze_distributions(self) -> Dict[str, Any]:
        """
        Analyze hospitalization metrics distributions.

        Returns:
        --------
        dict
            Distribution analysis for hospitalization metrics
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        df = self.table.df
        distributions = {}

        # Analyze categorical columns (using CLIF 2.1 schema column names)
        categorical_columns = [
            'discharge_category',  # Standardized discharge disposition
            'admission_type_category'  # Standardized admission type
        ]

        for column in categorical_columns:
            if column in df.columns:
                dist_result = self.get_categorical_distribution(column)
                # Only add if valid distribution (has values and counts)
                if 'error' not in dist_result and 'values' in dist_result and len(dist_result['values']) > 0:
                    distributions[column] = dist_result

        # If no distributions found, check what columns are available
        if not distributions:
            print(f"Warning: No categorical distributions found in hospitalization table")
            print(f"Available columns: {list(df.columns)}")

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

    def analyze_length_of_stay(self) -> Dict[str, Any]:
        """
        Analyze hospital length of stay.

        Returns:
        --------
        dict
            Length of stay statistics
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        df = self.table.df

        # Check for required columns
        if 'admission_dttm' not in df.columns or 'discharge_dttm' not in df.columns:
            return {'error': 'Required datetime columns not found'}

        # Calculate LOS in days
        los_data = df[['admission_dttm', 'discharge_dttm']].copy()
        los_data = los_data.dropna()

        if len(los_data) == 0:
            return {'error': 'No valid datetime data available'}

        los_days = (los_data['discharge_dttm'] - los_data['admission_dttm']).dt.total_seconds() / (24 * 3600)

        return {
            'count': int(len(los_days)),
            'mean': float(los_days.mean()),
            'std': float(los_days.std()),
            'min': float(los_days.min()),
            'q1': float(los_days.quantile(0.25)),
            'median': float(los_days.median()),
            'q3': float(los_days.quantile(0.75)),
            'max': float(los_days.max())
        }

    def analyze_age_distribution(self) -> Dict[str, Any]:
        """
        Analyze age at admission distribution.

        Returns:
        --------
        dict
            Age distribution statistics
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        df = self.table.df

        if 'age_at_admission' not in df.columns:
            return {'error': 'age_at_admission column not found'}

        ages = df['age_at_admission'].dropna()

        if len(ages) == 0:
            return {'error': 'No age data available'}

        return {
            'count': int(len(ages)),
            'mean': float(ages.mean()),
            'std': float(ages.std()),
            'min': float(ages.min()),
            'q1': float(ages.quantile(0.25)),
            'median': float(ages.median()),
            'q3': float(ages.quantile(0.75)),
            'max': float(ages.max()),
            'adult_count': int((ages >= 18).sum()),
            'pediatric_count': int((ages < 18).sum())
        }

    def generate_hospitalization_summary(self) -> pd.DataFrame:
        """
        Generate a summary dataframe for the hospitalization table.

        Returns:
        --------
        pd.DataFrame
            Summary table with key metrics
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return pd.DataFrame({'Metric': ['No data available'], 'Value': [None]})

        df = self.table.df
        summary_data = []

        # Total hospitalizations
        summary_data.append({
            'Metric': 'Total Hospitalizations',
            'Value': f"{df['hospitalization_id'].nunique():,}" if 'hospitalization_id' in df.columns else 'N/A'
        })

        # Unique patients
        summary_data.append({
            'Metric': 'Unique Patients',
            'Value': f"{df['patient_id'].nunique():,}" if 'patient_id' in df.columns else 'N/A'
        })

        # Age statistics
        age_stats = self.analyze_age_distribution()
        if 'error' not in age_stats:
            summary_data.append({
                'Metric': 'Age (mean ± SD)',
                'Value': f"{age_stats['mean']:.1f} ± {age_stats['std']:.1f}"
            })
            summary_data.append({
                'Metric': 'Adult (≥18 years)',
                'Value': f"{age_stats['adult_count']:,} ({age_stats['adult_count']/age_stats['count']*100:.1f}%)"
            })

        # Length of stay statistics
        los_stats = self.analyze_length_of_stay()
        if 'error' not in los_stats:
            summary_data.append({
                'Metric': 'Hospital LOS (median [IQR])',
                'Value': f"{los_stats['median']:.1f} [{los_stats['q1']:.1f}-{los_stats['q3']:.1f}] days"
            })

        # Discharge disposition (using CLIF 2.1 schema name)
        if 'discharge_category' in df.columns:
            disp_counts = df['discharge_category'].value_counts().head(5)
            for disp, count in disp_counts.items():
                pct = (count / len(df) * 100)
                summary_data.append({
                    'Metric': f'Discharge - {disp}',
                    'Value': f"{count:,} ({pct:.1f}%)"
                })

        # Admission type (using CLIF 2.1 schema name)
        if 'admission_type_category' in df.columns:
            adm_counts = df['admission_type_category'].value_counts().head(3)
            for adm_type, count in adm_counts.items():
                pct = (count / len(df) * 100)
                summary_data.append({
                    'Metric': f'Admission Type - {adm_type}',
                    'Value': f"{count:,} ({pct:.1f}%)"
                })

        return pd.DataFrame(summary_data)

    def check_data_quality(self) -> Dict[str, Any]:
        """
        Perform data quality checks specific to Hospitalization table.

        Returns:
        --------
        dict
            Data quality check results
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        df = self.table.df
        quality_checks = {}

        # Check for duplicate hospitalization IDs
        if 'hospitalization_id' in df.columns:
            duplicates_mask = df['hospitalization_id'].duplicated(keep=False)
            duplicates = duplicates_mask.sum()

            # Get examples of duplicate records
            examples = None
            if duplicates > 0:
                dup_ids = df.loc[duplicates_mask, 'hospitalization_id'].unique()[:5]
                example_cols = ['hospitalization_id', 'patient_id', 'admission_dttm']
                example_cols = [col for col in example_cols if col in df.columns]
                examples = df[df['hospitalization_id'].isin(dup_ids)][example_cols].head(10)

            quality_checks['duplicate_hospitalization_ids'] = {
                'count': int(duplicates),
                'percentage': round((duplicates / len(df) * 100) if len(df) > 0 else 0, 2),
                'status': 'pass' if duplicates == 0 else 'warning',
                'examples': examples
            }

        # Check for invalid dates (discharge before admission)
        if 'admission_dttm' in df.columns and 'discharge_dttm' in df.columns:
            invalid_dates_mask = df['discharge_dttm'] < df['admission_dttm']

            # Get examples of invalid dates
            examples = None
            if invalid_dates_mask.sum() > 0:
                example_cols = ['hospitalization_id', 'patient_id', 'admission_dttm', 'discharge_dttm']
                example_cols = [col for col in example_cols if col in df.columns]
                examples = df[invalid_dates_mask][example_cols].head(10)

            quality_checks['invalid_discharge_dates'] = {
                'count': int(invalid_dates_mask.sum()),
                'percentage': round((invalid_dates_mask.sum() / len(df) * 100) if len(df) > 0 else 0, 2),
                'status': 'pass' if invalid_dates_mask.sum() == 0 else 'error',
                'examples': examples
            }

        # Check for negative ages
        if 'age_at_admission' in df.columns:
            negative_ages_mask = df['age_at_admission'] < 0

            # Get examples of negative ages
            examples = None
            if negative_ages_mask.sum() > 0:
                example_cols = ['hospitalization_id', 'patient_id', 'age_at_admission']
                example_cols = [col for col in example_cols if col in df.columns]
                examples = df[negative_ages_mask][example_cols].head(10)

            quality_checks['negative_ages'] = {
                'count': int(negative_ages_mask.sum()),
                'percentage': round((negative_ages_mask.sum() / len(df) * 100) if len(df) > 0 else 0, 2),
                'status': 'pass' if negative_ages_mask.sum() == 0 else 'error',
                'examples': examples
            }

        return quality_checks
