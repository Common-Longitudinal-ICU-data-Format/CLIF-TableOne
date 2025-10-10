"""
ECMO/MCS Table Analyzer for CLIF 2.1

This module provides specific analysis for the ECMO/MCS table using clifpy.
"""

from typing import Dict, Any, Optional
import pandas as pd
from .base_table_analyzer import BaseTableAnalyzer


class ECMOMCSAnalyzer(BaseTableAnalyzer):
    """Analyzer for ECMO/MCS table using clifpy."""

    def get_table_name(self) -> str:
        """Return the correct table name with underscore."""
        return 'ecmo_mcs'

    def load_table(self):
        """Load ECMO/MCS table using clifpy."""
        try:
            from clifpy.tables.ecmo_mcs import EcmoMcs
            import os

            # Check for both naming conventions
            possible_names = ['ecmo_mcs', 'clif_ecmo_mcs']
            file_found = False

            for name in possible_names:
                file_path = os.path.join(self.data_dir, f"{name}.{self.filetype}")
                if os.path.exists(file_path):
                    file_found = True
                    break

            if not file_found:
                print(f"Warning: Could not find ecmo_mcs.{self.filetype} or clif_ecmo_mcs.{self.filetype} in {self.data_dir}")
                self.table = None
                return

            # Clifpy saves files directly to output_directory, so pass the final subdirectory
            clifpy_output_dir = os.path.join(self.output_dir, 'final')
            os.makedirs(clifpy_output_dir, exist_ok=True)

            self.table = EcmoMcs.from_file(
                data_directory=self.data_dir,
                filetype=self.filetype,
                timezone=self.timezone,
                output_directory=clifpy_output_dir
            )

            # Move any CSV files that clifpy created in parent directory to final/
            self._move_clifpy_csvs_to_final()

        except ImportError as e:
            print(f"Warning: clifpy not installed or EcmoMcs table not available: {e}")
            print("Install with: pip install clifpy")
            self.table = None
        except Exception as e:
            print(f"Error loading ECMO/MCS table: {e}")
            import traceback
            traceback.print_exc()
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
        Get ECMO/MCS-specific data information.

        Returns:
        --------
        dict
            Information about the ECMO/MCS data
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {
                'error': 'No data available',
                'row_count': 0,
                'unique_hospitalizations': 0,
                'columns': [],
                'unique_devices': 0,
                'unique_device_categories': 0,
                'unique_mcs_groups': 0
            }

        df = self.table.df

        # Get date range for ECMO/MCS recordings
        date_range = {}
        if 'recorded_dttm' in df.columns:
            df['recorded_dttm'] = pd.to_datetime(df['recorded_dttm'])
            first_year = int(df['recorded_dttm'].dt.year.min()) if not df['recorded_dttm'].isna().all() else None
            last_year = int(df['recorded_dttm'].dt.year.max()) if not df['recorded_dttm'].isna().all() else None
            date_range = {
                'first_year': first_year,
                'last_year': last_year,
                'duration_years': (last_year - first_year + 1) if first_year and last_year else 0
            }

        return {
            'row_count': len(df),
            'unique_hospitalizations': df['hospitalization_id'].nunique() if 'hospitalization_id' in df.columns else 0,
            'unique_devices': df['device_name'].nunique() if 'device_name' in df.columns else 0,
            'unique_device_categories': df['device_category'].nunique() if 'device_category' in df.columns else 0,
            'unique_mcs_groups': df['mcs_group'].nunique() if 'mcs_group' in df.columns else 0,
            'columns': list(df.columns),
            'column_count': len(df.columns),
            **date_range
        }

    def analyze_distributions(self) -> Dict[str, Any]:
        """
        Analyze ECMO/MCS distributions.

        Returns:
        --------
        dict
            Distribution analysis for device categories and numeric parameters
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        df = self.table.df
        distributions = {}

        # Analyze device categories
        if 'device_category' in df.columns:
            distributions['device_category'] = self.get_categorical_distribution('device_category')

        # Analyze MCS groups
        if 'mcs_group' in df.columns:
            distributions['mcs_group'] = self.get_categorical_distribution('mcs_group')

        # Analyze numeric parameters
        numeric_cols = [
            'device_rate',
            'sweep',
            'flow',
            'fdO2'
        ]

        for col in numeric_cols:
            if col in df.columns:
                distributions[col] = self.get_numeric_distribution(col)

        # Analyze year distribution if we have date data
        if 'recorded_dttm' in df.columns:
            df['recorded_dttm'] = pd.to_datetime(df['recorded_dttm'])
            df['year'] = df['recorded_dttm'].dt.year
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

    def get_numeric_distribution(self, column: str) -> Dict[str, Any]:
        """
        Get distribution statistics for a numeric column.

        Parameters:
        -----------
        column : str
            Column name to analyze

        Returns:
        --------
        dict
            Numeric distribution statistics
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        df = self.table.df

        if column not in df.columns:
            return {'error': f'Column {column} not found'}

        # Get numeric statistics
        valid_values = df[column].dropna()
        missing_count = df[column].isna().sum()
        total = len(df)

        if len(valid_values) == 0:
            return {
                'missing_count': int(missing_count),
                'missing_percentage': 100.0,
                'error': 'No valid numeric values'
            }

        return {
            'mean': round(float(valid_values.mean()), 2),
            'median': round(float(valid_values.median()), 2),
            'std': round(float(valid_values.std()), 2),
            'min': round(float(valid_values.min()), 2),
            'max': round(float(valid_values.max()), 2),
            'q25': round(float(valid_values.quantile(0.25)), 2),
            'q75': round(float(valid_values.quantile(0.75)), 2),
            'missing_count': int(missing_count),
            'missing_percentage': round((missing_count / total * 100) if total > 0 else 0, 2),
            'total_rows': total
        }

    def get_numeric_distributions_with_outliers(self) -> Dict[str, Any]:
        """
        Get numeric distributions for ECMO/MCS parameters with outlier handling.

        Returns:
        --------
        dict
            Dictionary with numeric column distributions, including:
            - raw_data: Statistics from original data
            - cleaned_data: Statistics after outlier removal
            - outlier_info: Information about outliers
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        from modules.utils.outlier_handling import apply_outlier_ranges, load_outlier_config

        df = self.table.df
        table_name = self.get_table_name()

        # Load outlier config
        outlier_config = load_outlier_config()

        # Apply outlier ranges
        cleaned_df, outlier_stats = apply_outlier_ranges(df, table_name, outlier_config)

        # Numeric columns to analyze
        numeric_cols = [
            'device_rate',
            'sweep',
            'flow',
            'fdO2'
        ]

        distributions = {}

        for col in numeric_cols:
            if col not in df.columns:
                continue

            # Get raw data statistics
            raw_valid = df[col].dropna()
            raw_stats = {
                'mean': round(float(raw_valid.mean()), 2) if len(raw_valid) > 0 else None,
                'median': round(float(raw_valid.median()), 2) if len(raw_valid) > 0 else None,
                'std': round(float(raw_valid.std()), 2) if len(raw_valid) > 0 else None,
                'min': round(float(raw_valid.min()), 2) if len(raw_valid) > 0 else None,
                'max': round(float(raw_valid.max()), 2) if len(raw_valid) > 0 else None,
                'q25': round(float(raw_valid.quantile(0.25)), 2) if len(raw_valid) > 0 else None,
                'q75': round(float(raw_valid.quantile(0.75)), 2) if len(raw_valid) > 0 else None,
                'count': int(len(raw_valid)),
                'values': raw_valid.tolist()  # For plotting
            }

            # Get cleaned data statistics
            cleaned_valid = cleaned_df[col].dropna()
            cleaned_stats = {
                'mean': round(float(cleaned_valid.mean()), 2) if len(cleaned_valid) > 0 else None,
                'median': round(float(cleaned_valid.median()), 2) if len(cleaned_valid) > 0 else None,
                'std': round(float(cleaned_valid.std()), 2) if len(cleaned_valid) > 0 else None,
                'min': round(float(cleaned_valid.min()), 2) if len(cleaned_valid) > 0 else None,
                'max': round(float(cleaned_valid.max()), 2) if len(cleaned_valid) > 0 else None,
                'q25': round(float(cleaned_valid.quantile(0.25)), 2) if len(cleaned_valid) > 0 else None,
                'q75': round(float(cleaned_valid.quantile(0.75)), 2) if len(cleaned_valid) > 0 else None,
                'count': int(len(cleaned_valid)),
                'values': cleaned_valid.tolist()  # For plotting
            }

            distributions[col] = {
                'raw_data': raw_stats,
                'cleaned_data': cleaned_stats,
                'outlier_info': outlier_stats.get(col, {})
            }

        return distributions

    def save_numeric_distributions(self) -> Optional[str]:
        """
        Generate and save numeric distributions with outlier handling to JSON file.

        Returns:
        --------
        str or None
            Path to saved file, or None if save failed
        """
        import json
        import os

        numeric_dists = self.get_numeric_distributions_with_outliers()

        if 'error' in numeric_dists:
            print(f"Cannot save distributions: {numeric_dists['error']}")
            return None

        # Prepare data for JSON serialization (remove values arrays to reduce file size)
        serializable_dists = {}
        for col_name, dist_data in numeric_dists.items():
            serializable_dists[col_name] = {
                'raw_data': {k: v for k, v in dist_data.get('raw_data', {}).items() if k != 'values'},
                'cleaned_data': {k: v for k, v in dist_data.get('cleaned_data', {}).items() if k != 'values'},
                'outlier_info': dist_data.get('outlier_info', {})
            }

        # Save to file
        final_dir = os.path.join(self.output_dir, 'final')
        os.makedirs(final_dir, exist_ok=True)

        filepath = os.path.join(final_dir, f"{self.get_table_name()}_numeric_distributions.json")

        try:
            with open(filepath, 'w') as f:
                json.dump(serializable_dists, f, indent=2)
            print(f"Saved numeric distributions to {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving numeric distributions: {e}")
            return None

    def save_visualization_data(self) -> Optional[str]:
        """
        Pre-generate and save visualization data for ECMO/MCS categorical-numeric distributions.
        This includes outlier-handled data ready for visualization.

        Returns:
        --------
        str or None
            Path to saved file, or None if save failed
        """
        import json
        import os
        from modules.utils.outlier_handling import apply_outlier_ranges

        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            print("Cannot save visualization data: No data available")
            return None

        # Apply outlier handling to the dataframe
        cleaned_df, outlier_stats = apply_outlier_ranges(
            df=self.table.df,
            table_name=self.get_table_name()
        )

        # Define columns for visualization
        categorical_columns = ['device_category', 'mcs_group']
        numeric_columns = [
            'device_rate',
            'sweep',
            'flow',
            'fdO2'
        ]

        # Prepare visualization data
        viz_data = {
            'outlier_stats': outlier_stats,
            'total_outliers_replaced': sum(stats['outlier_count'] for stats in outlier_stats.values()),
            'categorical_columns': categorical_columns,
            'numeric_columns': numeric_columns,
            'categories': {},
            'numeric_distributions': {}
        }

        # Get category values and their distributions
        for cat_col in categorical_columns:
            if cat_col in cleaned_df.columns:
                unique_cats = sorted([str(x) for x in cleaned_df[cat_col].dropna().unique()])
                viz_data['categories'][cat_col] = unique_cats

                # For each category, get numeric distributions
                for category in unique_cats:
                    cat_data = cleaned_df[cleaned_df[cat_col].astype(str) == category]

                    for num_col in numeric_columns:
                        if num_col in cat_data.columns:
                            valid_data = cat_data[num_col].dropna()

                            if len(valid_data) > 0:
                                key = f"{cat_col}_{category}_{num_col}"
                                viz_data['numeric_distributions'][key] = {
                                    'category_column': cat_col,
                                    'category_value': category,
                                    'numeric_column': num_col,
                                    'count': int(len(valid_data)),
                                    'mean': round(float(valid_data.mean()), 2),
                                    'median': round(float(valid_data.median()), 2),
                                    'std': round(float(valid_data.std()), 2),
                                    'min': round(float(valid_data.min()), 2),
                                    'max': round(float(valid_data.max()), 2),
                                    'q25': round(float(valid_data.quantile(0.25)), 2),
                                    'q75': round(float(valid_data.quantile(0.75)), 2)
                                }

        # Save to file
        final_dir = os.path.join(self.output_dir, 'final')
        os.makedirs(final_dir, exist_ok=True)

        filepath = os.path.join(final_dir, f"{self.get_table_name()}_visualization_data.json")

        try:
            with open(filepath, 'w') as f:
                json.dump(viz_data, f, indent=2)
            print(f"Saved visualization data to {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving visualization data: {e}")
            return None

    def check_data_quality(self) -> Dict[str, Any]:
        """
        Perform data quality checks specific to ECMO/MCS table.

        Returns:
        --------
        dict
            Data quality check results
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        df = self.table.df
        quality_checks = {}

        # Check for future recorded dates
        if 'recorded_dttm' in df.columns:
            df['recorded_dttm'] = pd.to_datetime(df['recorded_dttm'])
            future_dates_mask = df['recorded_dttm'] > pd.Timestamp.now(tz=self.timezone)

            examples = None
            if future_dates_mask.sum() > 0:
                example_cols = ['hospitalization_id', 'recorded_dttm', 'device_category']
                example_cols = [col for col in example_cols if col in df.columns]
                examples = df[future_dates_mask][example_cols].head(10)

            quality_checks['future_recorded_dates'] = {
                'count': int(future_dates_mask.sum()),
                'percentage': round((future_dates_mask.sum() / len(df) * 100) if len(df) > 0 else 0, 2),
                'status': 'pass' if future_dates_mask.sum() == 0 else 'warning',
                'examples': examples
            }

        # Check for invalid device categories
        if 'device_category' in df.columns:
            valid_categories = ['Impella', 'Centrimag', 'TandemHeart', 'HeartMate', 'ECMO', 'Other']
            invalid_mask = ~df['device_category'].isin(valid_categories + [pd.NA, None])

            examples = None
            if invalid_mask.sum() > 0:
                example_cols = ['hospitalization_id', 'device_category', 'recorded_dttm']
                example_cols = [col for col in example_cols if col in df.columns]
                examples = df[invalid_mask][example_cols].head(10)

            quality_checks['invalid_device_categories'] = {
                'count': int(invalid_mask.sum()),
                'percentage': round((invalid_mask.sum() / len(df) * 100) if len(df) > 0 else 0, 2),
                'status': 'pass' if invalid_mask.sum() == 0 else 'error',
                'examples': examples
            }

        # Check for invalid MCS groups
        if 'mcs_group' in df.columns:
            valid_groups = [
                'Impella_2.5', 'Impella_CP', 'Impella_5.0', 'Impella_5.5', 'Impella_LD', 'Impella_RP', 'Impella_RP_Flex',
                'iVAC', 'TandemHeart_LV', 'TandemHeart_RV', 'CentriMag_LV', 'CentriMag_RV', 'IABP',
                'VA_ECMO', 'VV_ECMO', 'VAV_ECMO', 'LAVA_ECMO', 'HeartWare', 'HMII', 'HMIII'
            ]
            invalid_mask = ~df['mcs_group'].isin(valid_groups + [pd.NA, None])

            examples = None
            if invalid_mask.sum() > 0:
                example_cols = ['hospitalization_id', 'mcs_group', 'recorded_dttm']
                example_cols = [col for col in example_cols if col in df.columns]
                examples = df[invalid_mask][example_cols].head(10)

            quality_checks['invalid_mcs_groups'] = {
                'count': int(invalid_mask.sum()),
                'percentage': round((invalid_mask.sum() / len(df) * 100) if len(df) > 0 else 0, 2),
                'status': 'pass' if invalid_mask.sum() == 0 else 'error',
                'examples': examples
            }

        # Check for negative numeric values
        numeric_checks = [
            ('device_rate', 'negative_device_rate'),
            ('sweep', 'negative_sweep'),
            ('flow', 'negative_flow')
        ]

        for col, check_name in numeric_checks:
            if col in df.columns:
                negative_mask = df[col] < 0

                examples = None
                if negative_mask.sum() > 0:
                    example_cols = ['hospitalization_id', col, 'recorded_dttm']
                    example_cols = [col for col in example_cols if col in df.columns]
                    examples = df[negative_mask][example_cols].head(10)

                quality_checks[check_name] = {
                    'count': int(negative_mask.sum()),
                    'percentage': round((negative_mask.sum() / len(df) * 100) if len(df) > 0 else 0, 2),
                    'status': 'pass' if negative_mask.sum() == 0 else 'error',
                    'examples': examples
                }

        # Check for invalid fdO2 (should be between 0.21 and 1.0)
        if 'fdO2' in df.columns:
            invalid_mask = (df['fdO2'] < 0.21) | (df['fdO2'] > 1.0)

            examples = None
            if invalid_mask.sum() > 0:
                example_cols = ['hospitalization_id', 'fdO2', 'recorded_dttm']
                example_cols = [col for col in example_cols if col in df.columns]
                examples = df[invalid_mask][example_cols].head(10)

            quality_checks['invalid_fdO2'] = {
                'count': int(invalid_mask.sum()),
                'percentage': round((invalid_mask.sum() / len(df) * 100) if len(df) > 0 else 0, 2),
                'status': 'pass' if invalid_mask.sum() == 0 else 'error',
                'examples': examples
            }

        return quality_checks
