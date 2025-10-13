"""
Labs table analyzer using clifpy for CLIF 2.1.

This module provides basic analysis scaffolding for the labs table.
Table-specific distributions and quality checks to be added after data verification.
"""

from clifpy.tables.labs import Labs
from .base_table_analyzer import BaseTableAnalyzer
import os
from typing import Dict, Any
import pandas as pd
from pathlib import Path
import duckdb


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

        # Clifpy saves files directly to output_directory, so pass the final/clifpy subdirectory


        clifpy_output_dir = os.path.join(self.output_dir, "final", "clifpy")


        os.makedirs(clifpy_output_dir, exist_ok=True)



        try:
            # Use filters parameter ONLY when sample is provided
            if sample_filter is not None:
                self.table = Labs.from_file(
                    data_directory=self.data_dir,
                    filetype=self.filetype,
                    timezone=self.timezone,
                    output_directory=clifpy_output_dir,
                    filters={'hospitalization_id': list(sample_filter)}
                )
            else:
                # Normal load without filters
                self.table = Labs.from_file(
                    data_directory=self.data_dir,
                    filetype=self.filetype,
                    timezone=self.timezone,
                    output_directory=clifpy_output_dir
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

        # Get first and last lab order years
        if 'lab_order_dttm' in df.columns:
            valid_times = df['lab_order_dttm'].dropna()
            if len(valid_times) > 0:
                info['first_lab_year'] = valid_times.min().year
                info['last_lab_year'] = valid_times.max().year

        return info

    def analyze_distributions(self) -> Dict[str, Any]:
        """
        Analyze labs distributions.

        Returns:
        --------
        dict
            Distribution analysis for categorical columns
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {}

        df = self.table.df
        distributions = {}

        # Analyze categorical columns
        categorical_columns = ['lab_order_category', 'lab_category', 'lab_specimen_category']

        for column in categorical_columns:
            if column in df.columns:
                distributions[column] = self.get_categorical_distribution(column)

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

    def generate_labs_summary(self) -> pd.DataFrame:
        """
        Generate a summary dataframe for the labs table using clifpy results.

        Returns:
        --------
        pd.DataFrame
            Summary table with key metrics from clifpy validation
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return pd.DataFrame({'Metric': ['No data available'], 'Value': [None]})

        df = self.table.df
        conn = duckdb.connect(':memory:')
        conn.register('df', df)

        summary_data = []

        # Basic counts using clifpy's validated data
        basic_stats = conn.execute("""
            SELECT
                COUNT(*) as total_records,
                COUNT(DISTINCT hospitalization_id) as unique_hosps,
                COUNT(DISTINCT lab_category) as unique_lab_categories,
                COUNT(DISTINCT lab_order_category) as unique_order_categories
            FROM df
        """).fetchone()

        summary_data.append({'Metric': 'Total Lab Tests', 'Value': f"{basic_stats[0]:,}"})
        summary_data.append({'Metric': 'Unique Hospitalizations', 'Value': f"{basic_stats[1]:,}"})
        summary_data.append({'Metric': 'Unique Lab Categories', 'Value': f"{basic_stats[2]:,}"})
        summary_data.append({'Metric': 'Unique Order Categories', 'Value': f"{basic_stats[3]:,}"})

        # Lab order category distribution (8 standard categories)
        order_cats = conn.execute("""
            SELECT
                lab_order_category,
                COUNT(*) as count,
                COUNT(DISTINCT hospitalization_id) as hosp_count,
                ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM df), 2) as percentage
            FROM df
            WHERE lab_order_category IS NOT NULL
            GROUP BY lab_order_category
            ORDER BY count DESC
        """).fetchdf()

        if not order_cats.empty:
            summary_data.append({'Metric': '', 'Value': ''})  # Separator
            summary_data.append({'Metric': 'Lab Order Categories', 'Value': ''})

            for i, row in order_cats.iterrows():
                summary_data.append({
                    'Metric': f'  {row["lab_order_category"]}',
                    'Value': f'{row["count"]:,} tests ({row["percentage"]:.1f}%)'
                })

        # Top lab tests by frequency
        top_labs = conn.execute("""
            SELECT
                lab_category,
                COUNT(*) as count,
                COUNT(DISTINCT hospitalization_id) as hosp_count
            FROM df
            WHERE lab_category IS NOT NULL
            GROUP BY lab_category
            ORDER BY count DESC
            LIMIT 10
        """).fetchdf()

        if not top_labs.empty:
            summary_data.append({'Metric': '', 'Value': ''})  # Separator
            summary_data.append({'Metric': 'Top 10 Lab Tests', 'Value': ''})

            for i, row in top_labs.iterrows():
                summary_data.append({
                    'Metric': f'  {row["lab_category"]}',
                    'Value': f'{row["count"]:,} tests'
                })

        # Specimen type distribution (only if column exists)
        if 'lab_specimen_category' in df.columns:
            specimen_stats = conn.execute("""
                SELECT
                    COUNT(DISTINCT lab_specimen_category) as unique_specimens,
                    COUNT(CASE WHEN lab_specimen_category IS NOT NULL THEN 1 END) as with_specimen,
                    COUNT(CASE WHEN lab_specimen_category IS NULL THEN 1 END) as without_specimen
                FROM df
            """).fetchone()

            if specimen_stats[0] > 0:
                summary_data.append({'Metric': '', 'Value': ''})  # Separator
                summary_data.append({'Metric': 'Specimen Information', 'Value': ''})
                summary_data.append({'Metric': '  Unique Specimen Types', 'Value': f'{specimen_stats[0]:,}'})
                summary_data.append({'Metric': '  Tests with Specimen Info', 'Value': f'{specimen_stats[1]:,}'})

        # Result availability statistics using clifpy's validation
        # Note: lab_value is always required, lab_value_numeric is optional
        if 'lab_value' in df.columns:
            if 'lab_value_numeric' in df.columns:
                result_stats = conn.execute("""
                    SELECT
                        COUNT(CASE WHEN lab_value_numeric IS NOT NULL THEN 1 END) as numeric_results,
                        COUNT(CASE WHEN lab_value IS NOT NULL AND lab_value_numeric IS NULL THEN 1 END) as text_only_results,
                        COUNT(CASE WHEN lab_value IS NULL THEN 1 END) as no_results
                    FROM df
                """).fetchone()

                summary_data.append({'Metric': '', 'Value': ''})  # Separator
                summary_data.append({'Metric': 'Result Availability', 'Value': ''})
                summary_data.append({'Metric': '  Numeric Results', 'Value': f'{result_stats[0]:,}'})
                summary_data.append({'Metric': '  Text-Only Results', 'Value': f'{result_stats[1]:,}'})
                summary_data.append({'Metric': '  Missing Results', 'Value': f'{result_stats[2]:,}'})
            else:
                # Only lab_value column exists
                result_stats = conn.execute("""
                    SELECT
                        COUNT(CASE WHEN lab_value IS NOT NULL THEN 1 END) as with_results,
                        COUNT(CASE WHEN lab_value IS NULL THEN 1 END) as no_results
                    FROM df
                """).fetchone()

                summary_data.append({'Metric': '', 'Value': ''})  # Separator
                summary_data.append({'Metric': 'Result Availability', 'Value': ''})
                summary_data.append({'Metric': '  With Results', 'Value': f'{result_stats[0]:,}'})
                summary_data.append({'Metric': '  Missing Results', 'Value': f'{result_stats[1]:,}'})

        # Turnaround time statistics if datetime columns exist
        if all(col in df.columns for col in ['lab_order_dttm', 'lab_result_dttm']):
            tat_stats = conn.execute("""
                SELECT
                    AVG(EXTRACT(EPOCH FROM (lab_result_dttm - lab_order_dttm)) / 3600) as avg_tat_hours,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY
                        EXTRACT(EPOCH FROM (lab_result_dttm - lab_order_dttm)) / 3600) as median_tat_hours
                FROM df
                WHERE lab_order_dttm IS NOT NULL
                AND lab_result_dttm IS NOT NULL
                AND lab_result_dttm > lab_order_dttm
            """).fetchone()

            if tat_stats[0] is not None:
                summary_data.append({'Metric': '', 'Value': ''})  # Separator
                summary_data.append({'Metric': 'Turnaround Time', 'Value': ''})
                summary_data.append({'Metric': '  Mean TAT', 'Value': f'{tat_stats[0]:.1f} hours'})
                summary_data.append({'Metric': '  Median TAT', 'Value': f'{tat_stats[1]:.1f} hours'})

        conn.close()

        return pd.DataFrame(summary_data)

    def analyze_lab_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in lab testing.

        Returns:
        --------
        dict
            Analysis of lab testing patterns
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        df = self.table.df
        patterns = {}

        # Analyze lab frequency by category
        if 'lab_order_category' in df.columns:
            order_freq = df['lab_order_category'].value_counts()
            patterns['order_frequency'] = {
                'categories': order_freq.index.tolist(),
                'counts': order_freq.values.tolist()
            }

        # Analyze specimen collection patterns
        if 'lab_specimen_category' in df.columns:
            specimen_freq = df['lab_specimen_category'].value_counts()
            patterns['specimen_types'] = {
                'types': specimen_freq.index.tolist()[:10],
                'counts': specimen_freq.values.tolist()[:10],
                'total_types': len(specimen_freq)
            }

        # Analyze temporal patterns
        if 'lab_order_dttm' in df.columns:
            try:
                df_copy = df.copy()
                df_copy['hour'] = pd.to_datetime(df_copy['lab_order_dttm'], errors='coerce').dt.hour
                hourly_dist = df_copy['hour'].value_counts().sort_index()
                patterns['hourly_distribution'] = {
                    'hours': hourly_dist.index.tolist(),
                    'counts': hourly_dist.values.tolist()
                }
            except Exception:
                # Gracefully skip if datetime conversion fails
                pass

        return patterns
