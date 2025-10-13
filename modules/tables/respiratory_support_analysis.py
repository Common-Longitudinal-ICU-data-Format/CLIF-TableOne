"""Respiratory Support table analyzer using clifpy for CLIF 2.1."""

from clifpy.tables.respiratory_support import RespiratorySupport
from .base_table_analyzer import BaseTableAnalyzer
import os
from typing import Dict, Any
from pathlib import Path
import pandas as pd
import duckdb


class RespiratorySupportAnalyzer(BaseTableAnalyzer):
    """Analyzer for Respiratory Support table using clifpy."""

    def get_table_name(self) -> str:
        return 'respiratory_support'

    def load_table(self, sample_filter=None):
        """
        Load Respiratory Support table using clifpy.

        Parameters:
        -----------
        sample_filter : list, optional
            List of hospitalization_ids to filter to (uses clifpy filters)
        """
        data_path = Path(self.data_dir)
        file_without_clif = data_path / f"respiratory_support.{self.filetype}"
        file_with_clif = data_path / f"clif_respiratory_support.{self.filetype}"

        if not (file_without_clif.exists() or file_with_clif.exists()):
            print(f"⚠️  No respiratory_support file found in {self.data_dir}")
            self.table = None
            return

        # Clifpy saves files directly to output_directory, so pass the final/clifpy subdirectory


        clifpy_output_dir = os.path.join(self.output_dir, "final", "clifpy")


        os.makedirs(clifpy_output_dir, exist_ok=True)



        try:
            # Use filters parameter ONLY when sample is provided
            if sample_filter is not None:
                self.table = RespiratorySupport.from_file(
                    data_directory=self.data_dir,
                    filetype=self.filetype,
                    timezone=self.timezone,
                    output_directory=clifpy_output_dir,
                    filters={'hospitalization_id': list(sample_filter)}
                )
            else:
                # Normal load without filters
                self.table = RespiratorySupport.from_file(
                    data_directory=self.data_dir,
                    filetype=self.filetype,
                    timezone=self.timezone,
                    output_directory=clifpy_output_dir
                )
        except Exception as e:
            print(f"⚠️  Error loading respiratory_support table: {e}")
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

        # Get unique device and mode types
        if 'device_category' in df.columns:
            info['unique_device_types'] = df['device_category'].nunique()

        if 'mode_category' in df.columns:
            info['unique_ventilation_modes'] = df['mode_category'].nunique()

        return info

    def analyze_distributions(self) -> Dict[str, Any]:
        """
        Analyze respiratory support distributions.

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
        categorical_columns = ['device_category', 'mode_category', 'tracheostomy']

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
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}
        return {}

    def generate_respiratory_summary(self) -> pd.DataFrame:
        """
        Generate a summary dataframe for the respiratory support table using clifpy results.

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
                COUNT(DISTINCT device_category) as unique_devices,
                COUNT(DISTINCT mode_category) as unique_modes
            FROM df
        """).fetchone()

        summary_data.append({'Metric': 'Total Respiratory Events', 'Value': f"{basic_stats[0]:,}"})
        summary_data.append({'Metric': 'Unique Hospitalizations', 'Value': f"{basic_stats[1]:,}"})
        summary_data.append({'Metric': 'Unique Device Types', 'Value': f"{basic_stats[2]:,}"})
        summary_data.append({'Metric': 'Unique Ventilation Modes', 'Value': f"{basic_stats[3]:,}"})

        # Ventilation mode distribution
        vent_modes = conn.execute("""
            SELECT
                mode_category,
                COUNT(DISTINCT hospitalization_id) as hosp_count,
                ROUND(COUNT(DISTINCT hospitalization_id) * 100.0 /
                      (SELECT COUNT(DISTINCT hospitalization_id) FROM df), 2) as percentage
            FROM df
            WHERE mode_category IS NOT NULL
            GROUP BY mode_category
            ORDER BY hosp_count DESC
            LIMIT 5
        """).fetchdf()

        if not vent_modes.empty:
            summary_data.append({'Metric': '', 'Value': ''})  # Separator
            summary_data.append({'Metric': 'Ventilation Mode Distribution', 'Value': ''})

            for i, row in vent_modes.iterrows():
                summary_data.append({
                    'Metric': f'  {row["mode_category"]}',
                    'Value': f'{row["hosp_count"]:,} hospitalizations ({row["percentage"]:.1f}%)'
                })

        # Device category distribution
        device_cats = conn.execute("""
            SELECT
                device_category,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM df), 2) as percentage
            FROM df
            WHERE device_category IS NOT NULL
            GROUP BY device_category
            ORDER BY count DESC
            LIMIT 5
        """).fetchdf()

        if not device_cats.empty:
            summary_data.append({'Metric': '', 'Value': ''})  # Separator
            summary_data.append({'Metric': 'Device Categories', 'Value': ''})

            for i, row in device_cats.iterrows():
                summary_data.append({
                    'Metric': f'  {row["device_category"]}',
                    'Value': f'{row["count"]:,} ({row["percentage"]:.1f}%)'
                })

        # Check O2 flow and FiO2 statistics (using correct column names from schema)
        # Only query if columns exist
        has_lpm = 'lpm_set' in df.columns
        has_fio2 = 'fio2_set' in df.columns

        if has_lpm or has_fio2:
            # Build query dynamically based on available columns
            if has_lpm and has_fio2:
                o2_stats = conn.execute("""
                    SELECT
                        COUNT(CASE WHEN lpm_set IS NOT NULL THEN 1 END) as lpm_count,
                        AVG(lpm_set) as avg_lpm,
                        COUNT(CASE WHEN fio2_set IS NOT NULL THEN 1 END) as fio2_count,
                        AVG(fio2_set) as avg_fio2
                    FROM df
                """).fetchone()

                if o2_stats[0] > 0 or o2_stats[2] > 0:
                    summary_data.append({'Metric': '', 'Value': ''})  # Separator
                    summary_data.append({'Metric': 'Oxygen Parameters', 'Value': ''})

                    if o2_stats[0] > 0:
                        summary_data.append({'Metric': '  LPM (Liters Per Minute) Records', 'Value': f'{o2_stats[0]:,}'})
                        summary_data.append({'Metric': '  Mean LPM', 'Value': f'{o2_stats[1]:.1f} L/min'})

                    if o2_stats[2] > 0:
                        summary_data.append({'Metric': '  FiO2 Records', 'Value': f'{o2_stats[2]:,}'})
                        summary_data.append({'Metric': '  Mean FiO2', 'Value': f'{o2_stats[3]:.1f}%'})
            elif has_lpm:
                o2_stats = conn.execute("""
                    SELECT
                        COUNT(CASE WHEN lpm_set IS NOT NULL THEN 1 END) as lpm_count,
                        AVG(lpm_set) as avg_lpm
                    FROM df
                """).fetchone()

                if o2_stats[0] > 0:
                    summary_data.append({'Metric': '', 'Value': ''})  # Separator
                    summary_data.append({'Metric': 'Oxygen Parameters', 'Value': ''})
                    summary_data.append({'Metric': '  LPM (Liters Per Minute) Records', 'Value': f'{o2_stats[0]:,}'})
                    summary_data.append({'Metric': '  Mean LPM', 'Value': f'{o2_stats[1]:.1f} L/min'})
            elif has_fio2:
                o2_stats = conn.execute("""
                    SELECT
                        COUNT(CASE WHEN fio2_set IS NOT NULL THEN 1 END) as fio2_count,
                        AVG(fio2_set) as avg_fio2
                    FROM df
                """).fetchone()

                if o2_stats[0] > 0:
                    summary_data.append({'Metric': '', 'Value': ''})  # Separator
                    summary_data.append({'Metric': 'Oxygen Parameters', 'Value': ''})
                    summary_data.append({'Metric': '  FiO2 Records', 'Value': f'{o2_stats[0]:,}'})
                    summary_data.append({'Metric': '  Mean FiO2', 'Value': f'{o2_stats[1]:.1f}%'})

        conn.close()

        return pd.DataFrame(summary_data)

    def analyze_ventilation_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in ventilation support.

        Returns:
        --------
        dict
            Analysis of ventilation support patterns
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        df = self.table.df
        patterns = {}

        # Analyze ventilation mode transitions
        if 'mode_category' in df.columns and 'hospitalization_id' in df.columns:
            # Count unique modes per hospitalization
            modes_per_hosp = df.groupby('hospitalization_id')['mode_category'].nunique()
            patterns['mode_transitions'] = {
                'single_mode': int((modes_per_hosp == 1).sum()),
                'multiple_modes': int((modes_per_hosp > 1).sum()),
                'max_modes_per_patient': int(modes_per_hosp.max())
            }

        # Analyze device usage patterns
        if 'device_category' in df.columns:
            device_freq = df['device_category'].value_counts()
            patterns['device_frequency'] = {
                'categories': device_freq.index.tolist(),
                'counts': device_freq.values.tolist()
            }

        return patterns
