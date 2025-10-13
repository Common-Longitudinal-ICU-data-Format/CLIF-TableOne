"""Position table analyzer using clifpy for CLIF 2.1."""

from clifpy.tables.position import Position
from .base_table_analyzer import BaseTableAnalyzer
import os
from typing import Dict, Any
from pathlib import Path
import pandas as pd
import duckdb


class PositionAnalyzer(BaseTableAnalyzer):
    """Analyzer for Position table using clifpy."""

    def get_table_name(self) -> str:
        return 'position'

    def load_table(self, sample_filter=None):
        """
        Load Position table using clifpy.

        Parameters:
        -----------
        sample_filter : list, optional
            List of hospitalization_ids to filter to (uses clifpy filters)
        """
        data_path = Path(self.data_dir)
        file_without_clif = data_path / f"position.{self.filetype}"
        file_with_clif = data_path / f"clif_position.{self.filetype}"

        if not (file_without_clif.exists() or file_with_clif.exists()):
            print(f"⚠️  No position file found in {self.data_dir}")
            self.table = None
            return

        # Clifpy saves files directly to output_directory, so pass the final/clifpy subdirectory


        clifpy_output_dir = os.path.join(self.output_dir, "final", "clifpy")


        os.makedirs(clifpy_output_dir, exist_ok=True)



        try:
            # Use filters parameter ONLY when sample is provided
            if sample_filter is not None:
                self.table = Position.from_file(
                    data_directory=self.data_dir,
                    filetype=self.filetype,
                    timezone=self.timezone,
                    output_directory=clifpy_output_dir,
                    filters={'hospitalization_id': list(sample_filter)}
                )
            else:
                # Normal load without filters
                self.table = Position.from_file(
                    data_directory=self.data_dir,
                    filetype=self.filetype,
                    timezone=self.timezone,
                    output_directory=clifpy_output_dir
                )
        except Exception as e:
            print(f"⚠️  Error loading position table: {e}")
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

        # Get unique position types
        if 'position_category' in df.columns:
            info['unique_position_types'] = df['position_category'].nunique()

        return info

    def analyze_distributions(self) -> Dict[str, Any]:
        """
        Analyze position distributions.

        Returns:
        --------
        dict
            Distribution analysis for categorical columns
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {}

        df = self.table.df
        distributions = {}

        # Analyze categorical column: position_category
        if 'position_category' in df.columns:
            distributions['position_category'] = self.get_categorical_distribution('position_category')

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

    def generate_position_summary(self) -> pd.DataFrame:
        """
        Generate a summary dataframe for the position table using clifpy results.

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
                COUNT(DISTINCT position_category) as unique_positions
            FROM df
        """).fetchone()

        summary_data.append({'Metric': 'Total Position Records', 'Value': f"{basic_stats[0]:,}"})
        summary_data.append({'Metric': 'Unique Hospitalizations', 'Value': f"{basic_stats[1]:,}"})
        summary_data.append({'Metric': 'Unique Position Types', 'Value': f"{basic_stats[2]:,}"})

        # Position type distribution
        position_dist = conn.execute("""
            SELECT
                position_category,
                COUNT(*) as count,
                COUNT(DISTINCT hospitalization_id) as hosp_count,
                ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM df), 2) as percentage
            FROM df
            WHERE position_category IS NOT NULL
            GROUP BY position_category
            ORDER BY count DESC
        """).fetchdf()

        if not position_dist.empty:
            summary_data.append({'Metric': '', 'Value': ''})  # Separator
            summary_data.append({'Metric': 'Position Distribution', 'Value': ''})

            for i, row in position_dist.iterrows():
                summary_data.append({
                    'Metric': f'  {row["position_category"]}',
                    'Value': f'{row["count"]:,} records ({row["percentage"]:.1f}%)'
                })

        # Position changes per hospitalization
        position_changes = conn.execute("""
            SELECT
                AVG(position_count) as avg_positions,
                MAX(position_count) as max_positions,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY position_count) as median_positions
            FROM (
                SELECT hospitalization_id, COUNT(*) as position_count
                FROM df
                GROUP BY hospitalization_id
            ) t
        """).fetchone()

        if position_changes[0] is not None:
            summary_data.append({'Metric': '', 'Value': ''})  # Separator
            summary_data.append({'Metric': 'Position Changes per Hospitalization', 'Value': ''})
            summary_data.append({'Metric': '  Average', 'Value': f'{position_changes[0]:.1f}'})
            summary_data.append({'Metric': '  Median', 'Value': f'{position_changes[2]:.0f}'})
            summary_data.append({'Metric': '  Maximum', 'Value': f'{position_changes[1]:.0f}'})

        # Duration statistics if datetime columns exist
        if 'recorded_dttm' in df.columns and 'hospitalization_id' in df.columns:
            # Calculate time between position changes
            duration_stats = conn.execute("""
                WITH position_times AS (
                    SELECT
                        hospitalization_id,
                        position_category,
                        recorded_dttm,
                        LEAD(recorded_dttm) OVER (PARTITION BY hospitalization_id ORDER BY recorded_dttm) as next_time
                    FROM df
                    WHERE recorded_dttm IS NOT NULL
                )
                SELECT
                    AVG(EXTRACT(EPOCH FROM (next_time - recorded_dttm)) / 3600) as avg_duration_hours,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY
                        EXTRACT(EPOCH FROM (next_time - recorded_dttm)) / 3600) as median_duration_hours
                FROM position_times
                WHERE next_time IS NOT NULL
            """).fetchone()

            if duration_stats[0] is not None:
                summary_data.append({'Metric': '', 'Value': ''})  # Separator
                summary_data.append({'Metric': 'Position Duration', 'Value': ''})
                summary_data.append({'Metric': '  Mean Duration', 'Value': f'{duration_stats[0]:.1f} hours'})
                summary_data.append({'Metric': '  Median Duration', 'Value': f'{duration_stats[1]:.1f} hours'})

        # Prone positioning analysis (important for COVID and ARDS patients)
        # Note: position_category values are 'prone' and 'not_prone' (lowercase)
        prone_stats = conn.execute("""
            SELECT
                COUNT(DISTINCT CASE WHEN position_category = 'prone' THEN hospitalization_id END) as prone_hosps,
                COUNT(CASE WHEN position_category = 'prone' THEN 1 END) as prone_records,
                COUNT(DISTINCT hospitalization_id) as total_hosps
            FROM df
        """).fetchone()

        if prone_stats[0] > 0:
            prone_pct = (prone_stats[0] / prone_stats[2] * 100) if prone_stats[2] > 0 else 0
            summary_data.append({'Metric': '', 'Value': ''})  # Separator
            summary_data.append({'Metric': 'Prone Positioning', 'Value': ''})
            summary_data.append({'Metric': '  Hospitalizations with Prone', 'Value': f'{prone_stats[0]:,} ({prone_pct:.1f}%)'})
            summary_data.append({'Metric': '  Total Prone Records', 'Value': f'{prone_stats[1]:,}'})

        # Check for missing position data from clifpy validation
        missing_positions = conn.execute("""
            SELECT
                SUM(CASE WHEN position_category IS NULL THEN 1 ELSE 0 END) as missing_positions,
                COUNT(*) as total
            FROM df
        """).fetchone()

        if missing_positions[1] > 0:
            missing_pct = (missing_positions[0] / missing_positions[1] * 100)
            summary_data.append({'Metric': '', 'Value': ''})  # Separator
            summary_data.append({'Metric': 'Data Completeness', 'Value': ''})
            summary_data.append({'Metric': '  Missing Position Values', 'Value': f'{missing_positions[0]:,} ({missing_pct:.1f}%)'})

        conn.close()

        return pd.DataFrame(summary_data)

    def analyze_position_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in patient positioning.

        Returns:
        --------
        dict
            Analysis of position patterns
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        df = self.table.df
        patterns = {}

        # Analyze position frequency
        if 'position_category' in df.columns:
            position_freq = df['position_category'].value_counts()
            patterns['position_frequency'] = {
                'positions': position_freq.index.tolist(),
                'counts': position_freq.values.tolist()
            }

        # Analyze position transitions
        if 'position_category' in df.columns and 'hospitalization_id' in df.columns:
            # Count unique positions per hospitalization
            positions_per_hosp = df.groupby('hospitalization_id')['position_category'].nunique()
            patterns['position_diversity'] = {
                'single_position': int((positions_per_hosp == 1).sum()),
                'multiple_positions': int((positions_per_hosp > 1).sum()),
                'max_positions_per_patient': int(positions_per_hosp.max())
            }

        # Analyze temporal patterns
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
