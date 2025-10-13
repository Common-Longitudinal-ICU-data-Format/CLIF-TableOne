"""Vitals table analyzer using clifpy for CLIF 2.1."""

from clifpy.tables.vitals import Vitals
from .base_table_analyzer import BaseTableAnalyzer
import os
from typing import Dict, Any
from pathlib import Path
import pandas as pd
import duckdb


class VitalsAnalyzer(BaseTableAnalyzer):
    """Analyzer for Vitals table using clifpy."""

    def get_table_name(self) -> str:
        return 'vitals'

    def load_table(self, sample_filter=None):
        """
        Load Vitals table using clifpy.

        Parameters:
        -----------
        sample_filter : list, optional
            List of hospitalization_ids to filter to (uses clifpy filters)
        """
        data_path = Path(self.data_dir)
        file_without_clif = data_path / f"vitals.{self.filetype}"
        file_with_clif = data_path / f"clif_vitals.{self.filetype}"

        if not (file_without_clif.exists() or file_with_clif.exists()):
            print(f"⚠️  No vitals file found in {self.data_dir}")
            self.table = None
            return

        # Clifpy saves files directly to output_directory, so pass the final/clifpy subdirectory


        clifpy_output_dir = os.path.join(self.output_dir, "final", "clifpy")


        os.makedirs(clifpy_output_dir, exist_ok=True)



        try:
            # Use filters parameter ONLY when sample is provided
            if sample_filter is not None:
                self.table = Vitals.from_file(
                    data_directory=self.data_dir,
                    filetype=self.filetype,
                    timezone=self.timezone,
                    output_directory=clifpy_output_dir,
                    filters={'hospitalization_id': list(sample_filter)}
                )
            else:
                # Normal load without filters
                self.table = Vitals.from_file(
                    data_directory=self.data_dir,
                    filetype=self.filetype,
                    timezone=self.timezone,
                    output_directory=clifpy_output_dir
                )
        except Exception as e:
            print(f"⚠️  Error loading vitals table: {e}")
            self.table = None

    def get_data_info(self) -> Dict[str, Any]:
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}
        df = self.table.df
        info = {'row_count': len(df), 'column_count': len(df.columns)}

        if 'hospitalization_id' in df.columns:
            info['unique_hospitalizations'] = df['hospitalization_id'].nunique()

        # Get first and last recording years
        if 'recorded_dttm' in df.columns:
            valid_times = df['recorded_dttm'].dropna()
            if len(valid_times) > 0:
                info['first_recording_year'] = valid_times.min().year
                info['last_recording_year'] = valid_times.max().year

        # Get unique vital types
        if 'vital_category' in df.columns:
            info['unique_vital_types'] = df['vital_category'].nunique()

        return info

    def analyze_distributions(self) -> Dict[str, Any]:
        """
        Analyze vitals distributions.

        Returns:
        --------
        dict
            Distribution analysis for categorical columns
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {}

        df = self.table.df
        distributions = {}

        # Analyze categorical column: vital_category
        if 'vital_category' in df.columns:
            distributions['vital_category'] = self.get_categorical_distribution('vital_category')

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
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}
        return {}

    def generate_vitals_summary(self) -> pd.DataFrame:
        """
        Generate a summary dataframe for the vitals table using clifpy results.

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
                COUNT(DISTINCT vital_category) as unique_vital_types
            FROM df
        """).fetchone()

        summary_data.append({'Metric': 'Total Vital Sign Records', 'Value': f"{basic_stats[0]:,}"})
        summary_data.append({'Metric': 'Unique Hospitalizations', 'Value': f"{basic_stats[1]:,}"})
        summary_data.append({'Metric': 'Unique Vital Types', 'Value': f"{basic_stats[2]:,}"})

        # Top vital categories by frequency
        top_vitals = conn.execute("""
            SELECT
                vital_category,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM df), 2) as percentage
            FROM df
            WHERE vital_category IS NOT NULL
            GROUP BY vital_category
            ORDER BY count DESC
            LIMIT 5
        """).fetchdf()

        summary_data.append({'Metric': '', 'Value': ''})  # Separator
        summary_data.append({'Metric': 'Top Vital Signs', 'Value': ''})

        for i, row in top_vitals.iterrows():
            summary_data.append({
                'Metric': f'  {row["vital_category"]}',
                'Value': f'{row["count"]:,} ({row["percentage"]:.1f}%)'
            })

        # Recording frequency statistics
        recording_stats = conn.execute("""
            SELECT
                COUNT(DISTINCT CAST(recorded_dttm AS DATE)) as unique_days,
                MIN(recorded_dttm) as first_recording,
                MAX(recorded_dttm) as last_recording,
                COUNT(*) * 1.0 / COUNT(DISTINCT hospitalization_id) as avg_per_hosp
            FROM df
            WHERE recorded_dttm IS NOT NULL
        """).fetchone()

        if recording_stats[0] > 0:
            summary_data.append({'Metric': '', 'Value': ''})  # Separator
            summary_data.append({'Metric': 'Recording Statistics', 'Value': ''})
            summary_data.append({'Metric': '  Unique Recording Days', 'Value': f'{recording_stats[0]:,}'})
            summary_data.append({'Metric': '  Average per Hospitalization', 'Value': f'{recording_stats[3]:.1f}'})

        # Check for missing data from clifpy validation
        missing_vitals = conn.execute("""
            SELECT
                SUM(CASE WHEN vital_value IS NULL THEN 1 ELSE 0 END) as missing_values,
                COUNT(*) as total
            FROM df
        """).fetchone()

        if missing_vitals[1] > 0:
            missing_pct = (missing_vitals[0] / missing_vitals[1] * 100)
            summary_data.append({'Metric': '', 'Value': ''})  # Separator
            summary_data.append({'Metric': 'Data Completeness', 'Value': ''})
            summary_data.append({'Metric': '  Missing Vital Values', 'Value': f'{missing_vitals[0]:,} ({missing_pct:.1f}%)'})

        conn.close()

        return pd.DataFrame(summary_data)

    def analyze_vital_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in vital sign recordings.

        Returns:
        --------
        dict
            Analysis of vital sign recording patterns
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        df = self.table.df
        patterns = {}

        # Analyze recording frequency by vital type
        if 'vital_category' in df.columns:
            vital_freq = df['vital_category'].value_counts()
            patterns['vital_frequency'] = {
                'categories': vital_freq.index.tolist()[:10],
                'counts': vital_freq.values.tolist()[:10],
                'total_categories': len(vital_freq)
            }

        # Analyze temporal patterns if datetime column exists
        if 'recorded_dttm' in df.columns:
            try:
                df_copy = df.copy()
                df_copy['hour'] = pd.to_datetime(df_copy['recorded_dttm'], errors='coerce').dt.hour
                hourly_dist = df_copy['hour'].value_counts().sort_index()
                patterns['hourly_distribution'] = {
                    'hours': hourly_dist.index.tolist(),
                    'counts': hourly_dist.values.tolist()
                }
            except Exception:
                # Gracefully skip if datetime conversion fails
                pass

        return patterns
