"""
ADT Table Analyzer for CLIF 2.1

This module provides specific analysis for the ADT (Admission/Discharge/Transfer) table using clifpy.
"""

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from .base_table_analyzer import BaseTableAnalyzer


class ADTAnalyzer(BaseTableAnalyzer):
    """Analyzer for ADT table using clifpy."""

    def load_table(self, sample_filter=None):
        """
        Load ADT table using clifpy.

        Parameters:
        -----------
        sample_filter : list, optional
            List of hospitalization_ids to filter to (uses clifpy filters)
        """
        try:
            from clifpy.tables.adt import Adt
            import os

            # Check for both naming conventions
            possible_names = ['adt', 'clif_adt']
            file_found = False

            for name in possible_names:
                file_path = os.path.join(self.data_dir, f"{name}.{self.filetype}")
                if os.path.exists(file_path):
                    file_found = True
                    break

            if not file_found:
                print(f"Warning: Could not find adt.{self.filetype} or clif_adt.{self.filetype} in {self.data_dir}")
                self.table = None
                return

            # Clifpy saves files directly to output_directory, so pass the final subdirectory
            clifpy_output_dir = os.path.join(self.output_dir, "final", "clifpy")
            os.makedirs(clifpy_output_dir, exist_ok=True)

            # Use filters parameter ONLY when sample is provided
            if sample_filter is not None:
                self.table = Adt.from_file(
                    data_directory=self.data_dir,
                    filetype=self.filetype,
                    timezone=self.timezone,
                    output_directory=clifpy_output_dir,
                    filters={'hospitalization_id': list(sample_filter)}
                )
            else:
                self.table = Adt.from_file(
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
            print(f"Error loading ADT table: {e}")
            self.table = None

    def analyze_encounter_categories(self) -> Dict[str, Any]:
        """
        Categorize hospitalizations by location types visited.
        Uses DuckDB for optimized performance on large datasets.

        Returns:
        --------
        dict
            Hospitalization categories with counts
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {
                'icu_hospitalizations': 0,
                'icu_only_hospitalizations': 0,
                'ed_only_hospitalizations': 0,
                'ward_only_hospitalizations': 0,
                'total_hospitalizations': 0
            }

        df = self.table.df

        # Check required columns
        if 'hospitalization_id' not in df.columns or 'location_category' not in df.columns:
            return {
                'icu_hospitalizations': 0,
                'icu_only_hospitalizations': 0,
                'ed_only_hospitalizations': 0,
                'ward_only_hospitalizations': 0,
                'total_hospitalizations': 0
            }

        try:
            import duckdb

            # DuckDB query - optimized for large datasets
            result = duckdb.query("""
                WITH encounter_locations AS (
                    SELECT
                        hospitalization_id,
                        MAX(CASE WHEN location_category = 'icu' THEN 1 ELSE 0 END) AS has_icu,
                        MAX(CASE WHEN location_category = 'ed' THEN 1 ELSE 0 END) AS has_ed,
                        MAX(CASE WHEN location_category = 'ward' THEN 1 ELSE 0 END) AS has_ward,
                        MAX(CASE WHEN location_category NOT IN ('icu', 'ed', 'ward')
                            AND location_category IS NOT NULL THEN 1 ELSE 0 END) AS has_other
                    FROM df
                    WHERE location_category IS NOT NULL
                    GROUP BY hospitalization_id
                )
                SELECT
                    SUM(has_icu) AS icu_hospitalizations,
                    SUM(CASE WHEN has_icu = 1 AND has_ed = 0 AND has_ward = 0 AND has_other = 0
                        THEN 1 ELSE 0 END) AS icu_only_hospitalizations,
                    SUM(CASE WHEN has_ed = 1 AND has_icu = 0 AND has_ward = 0 AND has_other = 0
                        THEN 1 ELSE 0 END) AS ed_only_hospitalizations,
                    SUM(CASE WHEN has_ward = 1 AND has_icu = 0 AND has_ed = 0 AND has_other = 0
                        THEN 1 ELSE 0 END) AS ward_only_hospitalizations,
                    COUNT(*) AS total_hospitalizations
                FROM encounter_locations
            """).fetchone()

            return {
                'icu_hospitalizations': int(result[0] or 0),
                'icu_only_hospitalizations': int(result[1] or 0),
                'ed_only_hospitalizations': int(result[2] or 0),
                'ward_only_hospitalizations': int(result[3] or 0),
                'total_hospitalizations': int(result[4] or 0)
            }

        except ImportError:
            # Fallback to pandas if duckdb not available
            print("Warning: duckdb not installed. Using slower pandas implementation. Install with: pip install duckdb")

            # Group by hospitalization and get unique location categories
            encounter_locations = df.groupby('hospitalization_id')['location_category'].apply(
                lambda x: set(x.dropna())
            )

            # Categorize hospitalizations
            icu_hospitalizations = 0
            icu_only_hospitalizations = 0
            ed_only_hospitalizations = 0
            ward_only_hospitalizations = 0

            for hosp_id, locations in encounter_locations.items():
                # ICU hospitalizations: went to ICU at least once
                if 'icu' in locations:
                    icu_hospitalizations += 1
                    # ICU-only: only visited ICU, no other locations
                    if locations == {'icu'}:
                        icu_only_hospitalizations += 1
                # ED-only: only visited ED, no other locations
                elif locations == {'ed'}:
                    ed_only_hospitalizations += 1
                # Ward-only: only visited ward, no other locations
                elif locations == {'ward'}:
                    ward_only_hospitalizations += 1

            return {
                'icu_hospitalizations': int(icu_hospitalizations),
                'icu_only_hospitalizations': int(icu_only_hospitalizations),
                'ed_only_hospitalizations': int(ed_only_hospitalizations),
                'ward_only_hospitalizations': int(ward_only_hospitalizations),
                'total_hospitalizations': int(len(encounter_locations))
            }

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
        Get ADT-specific data information.

        Returns:
        --------
        dict
            Information about the ADT data
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {
                'error': 'No data available',
                'row_count': 0,
                'unique_hospitalizations': 0,
                'unique_locations': 0,
                'columns': [],
                'first_event_year': None,
                'last_event_year': None,
                'icu_hospitalizations': 0,
                'icu_only_hospitalizations': 0,
                'ed_only_hospitalizations': 0,
                'ward_only_hospitalizations': 0
            }

        df = self.table.df

        # Get first and last event years
        first_event_year = None
        last_event_year = None
        if 'in_dttm' in df.columns:
            valid_events = df['in_dttm'].dropna()
            if len(valid_events) > 0:
                first_event_year = valid_events.min().year
                last_event_year = valid_events.max().year

        # Get encounter categories
        encounter_categories = self.analyze_encounter_categories()

        return {
            'row_count': len(df),
            'unique_hospitalizations': df['hospitalization_id'].nunique() if 'hospitalization_id' in df.columns else 0,
            'unique_locations': df['location_category'].nunique() if 'location_category' in df.columns else 0,
            'columns': list(df.columns),
            'column_count': len(df.columns),
            'first_event_year': first_event_year,
            'last_event_year': last_event_year,
            'icu_hospitalizations': encounter_categories['icu_hospitalizations'],
            'icu_only_hospitalizations': encounter_categories['icu_only_hospitalizations'],
            'ed_only_hospitalizations': encounter_categories['ed_only_hospitalizations'],
            'ward_only_hospitalizations': encounter_categories['ward_only_hospitalizations']
        }

    def analyze_distributions(self) -> Dict[str, Any]:
        """
        Analyze ADT location distributions.

        Returns:
        --------
        dict
            Distribution analysis for ADT metrics
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        df = self.table.df
        distributions = {}

        # Analyze categorical columns based on CLIF 2.1 schema
        categorical_columns = [
            'location_category',  # Standardized location categories
            'location_type',  # ICU type categories
            'hospital_type'  # Hospital type categories
        ]

        for column in categorical_columns:
            if column in df.columns:
                dist_result = self.get_categorical_distribution(column)
                # Only add if valid distribution (has values and counts)
                if 'error' not in dist_result and 'values' in dist_result and len(dist_result['values']) > 0:
                    distributions[column] = dist_result

        # If no distributions found, check what columns are available
        if not distributions:
            print(f"Warning: No categorical distributions found in ADT table")
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

    def analyze_location_durations(self) -> Dict[str, Any]:
        """
        Analyze duration of stay in each location.

        Returns:
        --------
        dict
            Location duration statistics
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        df = self.table.df

        # Check for required columns
        if 'in_dttm' not in df.columns or 'out_dttm' not in df.columns:
            return {'error': 'Required datetime columns not found'}

        # Calculate duration in hours
        duration_data = df[['in_dttm', 'out_dttm', 'location_category']].copy()
        duration_data = duration_data.dropna(subset=['in_dttm', 'out_dttm'])

        if len(duration_data) == 0:
            return {'error': 'No valid datetime data available'}

        duration_hours = (duration_data['out_dttm'] - duration_data['in_dttm']).dt.total_seconds() / 3600

        return {
            'count': int(len(duration_hours)),
            'mean_hours': float(duration_hours.mean()),
            'std_hours': float(duration_hours.std()),
            'min_hours': float(duration_hours.min()),
            'q1_hours': float(duration_hours.quantile(0.25)),
            'median_hours': float(duration_hours.median()),
            'q3_hours': float(duration_hours.quantile(0.75)),
            'max_hours': float(duration_hours.max())
        }

    def analyze_icu_stays(self) -> Dict[str, Any]:
        """
        Analyze ICU-specific ADT events.

        Returns:
        --------
        dict
            ICU stay statistics
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        df = self.table.df

        if 'location_category' not in df.columns:
            return {'error': 'location_category column not found'}

        # Filter for ICU locations
        icu_mask = df['location_category'] == 'icu'
        icu_df = df[icu_mask]

        if len(icu_df) == 0:
            return {'error': 'No ICU location data available'}

        results = {
            'total_icu_events': int(len(icu_df)),
            'unique_icu_hospitalizations': int(icu_df['hospitalization_id'].nunique()) if 'hospitalization_id' in icu_df.columns else 0
        }

        # Analyze ICU types if available
        if 'location_type' in icu_df.columns:
            icu_type_counts = icu_df['location_type'].value_counts()
            results['icu_types'] = {
                'types': icu_type_counts.index.tolist(),
                'counts': icu_type_counts.values.tolist()
            }

        return results

    def generate_adt_summary(self) -> pd.DataFrame:
        """
        Generate a summary dataframe for the ADT table.

        Returns:
        --------
        pd.DataFrame
            Summary table with key metrics
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return pd.DataFrame({'Metric': ['No data available'], 'Value': [None]})

        df = self.table.df
        summary_data = []

        # Total ADT events
        summary_data.append({
            'Metric': 'Total ADT Events',
            'Value': f"{len(df):,}"
        })

        # Unique hospitalizations
        summary_data.append({
            'Metric': 'Unique Hospitalizations',
            'Value': f"{df['hospitalization_id'].nunique():,}" if 'hospitalization_id' in df.columns else 'N/A'
        })

        # Hospitalization categories
        hospitalization_categories = self.analyze_encounter_categories()
        summary_data.append({
            'Metric': 'ICU Hospitalizations (visited ICU at least once)',
            'Value': f"{hospitalization_categories['icu_hospitalizations']:,}"
        })
        summary_data.append({
            'Metric': 'ICU-Only Hospitalizations',
            'Value': f"{hospitalization_categories['icu_only_hospitalizations']:,}"
        })
        summary_data.append({
            'Metric': 'ED-Only Hospitalizations',
            'Value': f"{hospitalization_categories['ed_only_hospitalizations']:,}"
        })
        summary_data.append({
            'Metric': 'Ward-Only Hospitalizations',
            'Value': f"{hospitalization_categories['ward_only_hospitalizations']:,}"
        })

        # Location distribution (top 5)
        if 'location_category' in df.columns:
            loc_counts = df['location_category'].value_counts().head(5)
            for loc, count in loc_counts.items():
                pct = (count / len(df) * 100)
                summary_data.append({
                    'Metric': f'Location - {loc}',
                    'Value': f"{count:,} ({pct:.1f}%)"
                })

        # ICU statistics
        icu_stats = self.analyze_icu_stays()
        if 'error' not in icu_stats:
            summary_data.append({
                'Metric': 'ICU Events',
                'Value': f"{icu_stats['total_icu_events']:,}"
            })
            summary_data.append({
                'Metric': 'Unique ICU Hospitalizations',
                'Value': f"{icu_stats['unique_icu_hospitalizations']:,}"
            })

        # Duration statistics
        duration_stats = self.analyze_location_durations()
        if 'error' not in duration_stats:
            summary_data.append({
                'Metric': 'Location Duration (median hours [IQR])',
                'Value': f"{duration_stats['median_hours']:.1f} [{duration_stats['q1_hours']:.1f}-{duration_stats['q3_hours']:.1f}]"
            })

        # Hospital type distribution if available
        if 'hospital_type' in df.columns:
            hosp_type_counts = df['hospital_type'].value_counts()
            for hosp_type, count in hosp_type_counts.items():
                pct = (count / len(df) * 100)
                summary_data.append({
                    'Metric': f'Hospital Type - {hosp_type}',
                    'Value': f"{count:,} ({pct:.1f}%)"
                })

        return pd.DataFrame(summary_data)

    def check_data_quality(self) -> Dict[str, Any]:
        """
        Perform data quality checks specific to ADT table.

        Returns:
        --------
        dict
            Data quality check results
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        df = self.table.df
        quality_checks = {}

        # Check for invalid dates (out_dttm before in_dttm)
        if 'in_dttm' in df.columns and 'out_dttm' in df.columns:
            invalid_dates_mask = df['out_dttm'] < df['in_dttm']

            # Get examples of invalid dates
            examples = None
            if invalid_dates_mask.sum() > 0:
                example_cols = ['hospitalization_id', 'in_dttm', 'out_dttm', 'location_category']
                example_cols = [col for col in example_cols if col in df.columns]
                examples = df[invalid_dates_mask][example_cols].head(10)

            quality_checks['invalid_location_dates'] = {
                'count': int(invalid_dates_mask.sum()),
                'percentage': round((invalid_dates_mask.sum() / len(df) * 100) if len(df) > 0 else 0, 2),
                'status': 'pass' if invalid_dates_mask.sum() == 0 else 'error',
                'examples': examples
            }

        # Check for missing location categories
        if 'location_category' in df.columns:
            missing_loc_mask = df['location_category'].isna()

            # Get examples of missing locations
            examples = None
            if missing_loc_mask.sum() > 0:
                example_cols = ['hospitalization_id', 'in_dttm', 'location_name', 'location_category']
                example_cols = [col for col in example_cols if col in df.columns]
                examples = df[missing_loc_mask][example_cols].head(10)

            quality_checks['missing_location_category'] = {
                'count': int(missing_loc_mask.sum()),
                'percentage': round((missing_loc_mask.sum() / len(df) * 100) if len(df) > 0 else 0, 2),
                'status': 'pass' if missing_loc_mask.sum() == 0 else 'warning',
                'examples': examples
            }

        # Check for duplicate ADT events (same hospitalization_id, in_dttm, location)
        if all(col in df.columns for col in ['hospitalization_id', 'in_dttm', 'location_category']):
            duplicates_mask = df.duplicated(subset=['hospitalization_id', 'in_dttm', 'location_category'], keep=False)
            duplicates = duplicates_mask.sum()

            # Get examples of duplicate records
            examples = None
            if duplicates > 0:
                example_cols = ['hospitalization_id', 'in_dttm', 'out_dttm', 'location_category']
                example_cols = [col for col in example_cols if col in df.columns]
                examples = df[duplicates_mask][example_cols].head(10)

            quality_checks['duplicate_adt_events'] = {
                'count': int(duplicates),
                'percentage': round((duplicates / len(df) * 100) if len(df) > 0 else 0, 2),
                'status': 'pass' if duplicates == 0 else 'warning',
                'examples': examples
            }

        return quality_checks
