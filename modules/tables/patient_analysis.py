"""
Patient Table Analyzer for CLIF 2.1

This module provides specific analysis for the Patient table using clifpy.
"""

from typing import Dict, Any, Optional
import pandas as pd
from .base_table_analyzer import BaseTableAnalyzer


class PatientAnalyzer(BaseTableAnalyzer):
    """Analyzer for Patient table using clifpy."""

    def load_table(self, sample_filter=None):
        """
        Load Patient table using clifpy.

        Parameters:
        -----------
        sample_filter : list, optional
            List of hospitalization_ids to filter to (not applicable to patient table)
        """
        try:
            from clifpy.tables.patient import Patient
            import os

            # Note: Patient table doesn't have hospitalization_id, so sample_filter is not applicable
            # The patient table is loaded in full regardless of sample setting

            # Check for both naming conventions
            possible_names = ['patient', 'clif_patient']
            file_found = False

            for name in possible_names:
                file_path = os.path.join(self.data_dir, f"{name}.{self.filetype}")
                if os.path.exists(file_path):
                    file_found = True
                    # clifpy will automatically find the correct file
                    break

            if not file_found:
                print(f"Warning: Could not find patient.{self.filetype} or clif_patient.{self.filetype} in {self.data_dir}")
                self.table = None
                return

            # Clifpy saves files directly to output_directory, so pass the final subdirectory
            clifpy_output_dir = os.path.join(self.output_dir, 'final')
            os.makedirs(clifpy_output_dir, exist_ok=True)

            self.table = Patient.from_file(
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
            print(f"Error loading Patient table: {e}")
            self.table = None

    def _move_clifpy_csvs_to_final(self):
        """Move any CSV files created by clifpy from output/ to output/final/"""
        import os
        import shutil

        # Check parent output directory for CSV files
        parent_dir = self.output_dir
        final_dir = os.path.join(parent_dir, 'final')

        if not os.path.exists(parent_dir):
            return

        # Look for common clifpy CSV files
        clifpy_csv_patterns = ['missing_data_stats_', 'validation_errors_']

        for filename in os.listdir(parent_dir):
            if filename.endswith('.csv'):
                # Check if it matches clifpy patterns
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
        Get patient-specific data information.

        Returns:
        --------
        dict
            Information about the patient data
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {
                'error': 'No data available',
                'row_count': 0,
                'unique_patients': 0,
                'columns': [],
                'has_death_records': 0
            }

        df = self.table.df

        return {
            'row_count': len(df),
            'unique_patients': df['patient_id'].nunique() if 'patient_id' in df.columns else 0,
            'columns': list(df.columns),
            'has_death_records': df['death_dttm'].notna().sum() if 'death_dttm' in df.columns else 0,
            'column_count': len(df.columns)
        }

    def analyze_distributions(self) -> Dict[str, Any]:
        """
        Analyze patient demographics distributions.

        Returns:
        --------
        dict
            Distribution analysis for demographics
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        df = self.table.df
        distributions = {}

        # Analyze categorical demographics
        categorical_columns = [
            'sex_category',
            'race_category',
            'ethnicity_category',
            'language_category'
        ]

        for column in categorical_columns:
            if column in df.columns:
                distributions[column] = self.get_categorical_distribution(column)

        # Note: death_dttm missingness is shown in the missingness analysis section
        # No need to calculate mortality statistics separately

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

    def analyze_age_distribution(self, hospitalization_df: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Analyze age distribution if hospitalization data is provided.

        Parameters:
        -----------
        hospitalization_df : pd.DataFrame, optional
            Hospitalization data with age_at_admission

        Returns:
        --------
        dict
            Age distribution statistics
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No patient data available'}

        if hospitalization_df is None:
            return {'error': 'No hospitalization data provided'}

        # Merge with hospitalization to get age_at_admission
        merged = self.table.df.merge(
            hospitalization_df[['patient_id', 'age_at_admission']],
            on='patient_id',
            how='left'
        )

        # Remove missing ages
        ages = merged['age_at_admission'].dropna()

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

    def generate_patient_summary(self) -> pd.DataFrame:
        """
        Generate a summary dataframe for the patient table.

        Returns:
        --------
        pd.DataFrame
            Summary table with key metrics
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return pd.DataFrame({'Metric': ['No data available'], 'Value': [None]})

        df = self.table.df
        summary_data = []

        # Total patients
        summary_data.append({
            'Metric': 'Total Patients',
            'Value': f"{df['patient_id'].nunique():,}" if 'patient_id' in df.columns else 'N/A'
        })

        # Sex distribution
        if 'sex_category' in df.columns:
            sex_counts = df['sex_category'].value_counts()
            for sex, count in sex_counts.items():
                pct = (count / len(df) * 100)
                summary_data.append({
                    'Metric': f'Sex - {sex}',
                    'Value': f"{count:,} ({pct:.1f}%)"
                })

        # Race distribution (top 3)
        if 'race_category' in df.columns:
            race_counts = df['race_category'].value_counts().head(3)
            for race, count in race_counts.items():
                pct = (count / len(df) * 100)
                summary_data.append({
                    'Metric': f'Race - {race}',
                    'Value': f"{count:,} ({pct:.1f}%)"
                })

        # Ethnicity distribution
        if 'ethnicity_category' in df.columns:
            ethnicity_counts = df['ethnicity_category'].value_counts()
            for ethnicity, count in ethnicity_counts.items():
                pct = (count / len(df) * 100)
                summary_data.append({
                    'Metric': f'Ethnicity - {ethnicity}',
                    'Value': f"{count:,} ({pct:.1f}%)"
                })

        # Mortality
        if 'death_dttm' in df.columns:
            death_count = df['death_dttm'].notna().sum()
            mortality_rate = (death_count / len(df) * 100)
            summary_data.append({
                'Metric': 'Deaths',
                'Value': f"{death_count:,} ({mortality_rate:.1f}%)"
            })

        return pd.DataFrame(summary_data)

    def check_data_quality(self) -> Dict[str, Any]:
        """
        Perform data quality checks specific to Patient table.

        Returns:
        --------
        dict
            Data quality check results
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        df = self.table.df
        quality_checks = {}

        # Check for duplicate patient IDs
        if 'patient_id' in df.columns:
            duplicates_mask = df['patient_id'].duplicated(keep=False)
            duplicates = duplicates_mask.sum()

            # Get examples of duplicate records
            examples = None
            if duplicates > 0:
                dup_ids = df.loc[duplicates_mask, 'patient_id'].unique()[:5]
                example_cols = ['patient_id', 'sex_category', 'race_category']
                example_cols = [col for col in example_cols if col in df.columns]
                examples = df[df['patient_id'].isin(dup_ids)][example_cols].head(10)

            quality_checks['duplicate_patient_ids'] = {
                'count': int(duplicates),
                'percentage': round((duplicates / len(df) * 100) if len(df) > 0 else 0, 2),
                'status': 'pass' if duplicates == 0 else 'warning',
                'examples': examples
            }

        # Check for invalid sex categories
        if 'sex_category' in df.columns:
            valid_sex = ['Male', 'Female', 'Other', 'Unknown']
            invalid_sex_mask = ~df['sex_category'].isin(valid_sex + [pd.NA, None])

            # Get examples of invalid sex categories
            examples = None
            if invalid_sex_mask.sum() > 0:
                example_cols = ['patient_id', 'sex_category']
                example_cols = [col for col in example_cols if col in df.columns]
                examples = df[invalid_sex_mask][example_cols].head(10)

            quality_checks['invalid_sex_categories'] = {
                'count': int(invalid_sex_mask.sum()),
                'percentage': round((invalid_sex_mask.sum() / len(df) * 100) if len(df) > 0 else 0, 2),
                'status': 'pass' if invalid_sex_mask.sum() == 0 else 'warning',
                'examples': examples
            }

        # Check for future death dates
        if 'death_dttm' in df.columns:
            from datetime import datetime
            future_deaths_mask = df['death_dttm'] > pd.Timestamp.now(tz=self.timezone)

            # Get examples of future death dates
            examples = None
            if future_deaths_mask.sum() > 0:
                example_cols = ['patient_id', 'death_dttm']
                example_cols = [col for col in example_cols if col in df.columns]
                examples = df[future_deaths_mask][example_cols].head(10)

            quality_checks['future_death_dates'] = {
                'count': int(future_deaths_mask.sum()),
                'percentage': round((future_deaths_mask.sum() / len(df) * 100) if len(df) > 0 else 0, 2),
                'status': 'pass' if future_deaths_mask.sum() == 0 else 'error',
                'examples': examples
            }

        return quality_checks