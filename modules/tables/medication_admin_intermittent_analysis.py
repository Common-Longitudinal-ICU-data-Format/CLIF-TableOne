"""
Medication Admin Intermittent table analyzer using clifpy for CLIF 2.1.

This module provides comprehensive analysis for the medication_admin_intermittent table,
including medication group distributions, dose analysis with outlier handling,
and visualization of dose distributions.
"""

from clifpy.tables.medication_admin_intermittent import MedicationAdminIntermittent
from .base_table_analyzer import BaseTableAnalyzer
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
import duckdb
import json
import os
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class MedicationAdminIntermittentAnalyzer(BaseTableAnalyzer):
    """
    Analyzer for Medication Admin Intermittent table using clifpy.

    Provides comprehensive analysis including:
    - Medication distribution by group
    - Dose analysis with outlier handling
    - Statistical summaries for dose distributions
    - Name to category mappings
    - Hospitalization metrics by medication group
    """

    def get_table_name(self) -> str:
        """Return the table name."""
        return 'medication_admin_intermittent'

    def load_table(self, sample_filter=None):
        """
        Load Medication Admin Intermittent table using clifpy.

        Parameters:
        -----------
        sample_filter : list, optional
            List of hospitalization_ids to filter to (uses clifpy filters)

        Handles both naming conventions:
        - medication_admin_intermittent.parquet
        - clif_medication_admin_intermittent.parquet
        """
        data_path = Path(self.data_dir)
        filetype = self.filetype

        # Check both file naming conventions
        file_without_clif = data_path / f"medication_admin_intermittent.{filetype}"
        file_with_clif = data_path / f"clif_medication_admin_intermittent.{filetype}"

        file_exists = file_without_clif.exists() or file_with_clif.exists()

        if not file_exists:
            print(f"⚠️  No medication_admin_intermittent file found in {self.data_dir}")
            print(f"   Looking for: medication_admin_intermittent.{filetype} or clif_medication_admin_intermittent.{filetype}")
            self.table = None
            return

        try:
            # Use filters parameter ONLY when sample is provided
            if sample_filter is not None:
                self.table = MedicationAdminIntermittent.from_file(
                    data_directory=self.data_dir,
                    filetype=self.filetype,
                    timezone=self.timezone,
                    output_directory=self.output_dir,
                    filters={'hospitalization_id': list(sample_filter)}
                )
            else:
                # Normal load without filters
                self.table = MedicationAdminIntermittent.from_file(
                    data_directory=self.data_dir,
                    filetype=self.filetype,
                    timezone=self.timezone,
                    output_directory=self.output_dir
                )
        except FileNotFoundError:
            print(f"⚠️  medication_admin_intermittent table file not found in {self.data_dir}")
            self.table = None
        except Exception as e:
            print(f"⚠️  Error loading medication_admin_intermittent table: {e}")
            self.table = None

    def get_data_info(self) -> Dict[str, Any]:
        """
        Get medication_admin_intermittent data information using DuckDB for efficiency.

        Returns:
            Dictionary containing comprehensive metrics
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        df = self.table.df

        # Use DuckDB for efficient aggregation
        conn = duckdb.connect(':memory:')
        conn.register('df', df)

        # Get basic statistics
        basic_stats = conn.execute("""
            SELECT
                COUNT(*) as row_count,
                COUNT(DISTINCT hospitalization_id) as unique_hospitalizations,
                COUNT(DISTINCT med_category) as unique_med_categories,
                COUNT(DISTINCT med_group) as unique_groups,
                MIN(admin_dttm) as first_admin,
                MAX(admin_dttm) as last_admin
            FROM df
        """).fetchone()

        info = {
            'row_count': int(basic_stats[0]),
            'unique_hospitalizations': int(basic_stats[1]),
            'unique_med_categories': int(basic_stats[2]),
            'unique_groups': int(basic_stats[3]),
            'column_count': len(df.columns),
            'columns': list(df.columns)
        }

        # Add date range information
        if basic_stats[4] is not None and basic_stats[5] is not None:
            first_admin = pd.to_datetime(basic_stats[4])
            last_admin = pd.to_datetime(basic_stats[5])
            info['first_admin_year'] = int(first_admin.year)
            info['last_admin_year'] = int(last_admin.year)
            info['duration_years'] = info['last_admin_year'] - info['first_admin_year'] + 1

        # Get med_group distribution counts
        group_counts = conn.execute("""
            SELECT
                med_group,
                COUNT(DISTINCT hospitalization_id) as hosp_count
            FROM df
            WHERE med_group IS NOT NULL
            GROUP BY med_group
            ORDER BY hosp_count DESC
        """).fetchdf()

        if not group_counts.empty:
            info['med_group_distribution'] = {
                row['med_group']: int(row['hosp_count'])
                for _, row in group_counts.iterrows()
            }

        conn.close()
        return info

    def analyze_medication_by_group(self) -> Dict[str, Any]:
        """
        Analyze medication distribution by med_group using DuckDB.

        Returns:
            Dictionary containing medication group analysis with hospitalization metrics
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        df = self.table.df
        conn = duckdb.connect(':memory:')
        conn.register('df', df)

        # Get total hospitalizations
        total_hosps = conn.execute("""
            SELECT COUNT(DISTINCT hospitalization_id) as total
            FROM df
        """).fetchone()[0]

        # Get hospitalizations that received each med group
        group_metrics = conn.execute("""
            SELECT
                med_group,
                COUNT(DISTINCT hospitalization_id) as hosp_count,
                COUNT(*) as admin_count,
                ROUND(COUNT(DISTINCT hospitalization_id) * 100.0 / ?, 2) as percentage
            FROM df
            WHERE med_group IS NOT NULL
            GROUP BY med_group
            ORDER BY hosp_count DESC
        """, [total_hosps]).fetchdf()

        # Within each group, get medication distribution
        within_group_dist = conn.execute("""
            SELECT
                med_group,
                med_category,
                COUNT(DISTINCT hospitalization_id) as hosp_count,
                COUNT(*) as admin_count
            FROM df
            WHERE med_group IS NOT NULL AND med_category IS NOT NULL
            GROUP BY med_group, med_category
            ORDER BY med_group, hosp_count DESC
        """).fetchdf()

        # Structure the results
        results = {
            'total_hospitalizations': int(total_hosps),
            'groups': {}
        }

        for _, row in group_metrics.iterrows():
            group = row['med_group']
            results['groups'][group] = {
                'hospitalization_count': int(row['hosp_count']),
                'administration_count': int(row['admin_count']),
                'percentage_of_hospitalizations': float(row['percentage']),
                'medications': {}
            }

            # Add medication distribution within this group
            group_meds = within_group_dist[within_group_dist['med_group'] == group]
            for _, med_row in group_meds.iterrows():
                results['groups'][group]['medications'][med_row['med_category']] = {
                    'hospitalization_count': int(med_row['hosp_count']),
                    'administration_count': int(med_row['admin_count'])
                }

        conn.close()
        return results

    def get_dose_distributions_with_outliers(self) -> Dict[str, Any]:
        """
        Get dose distributions with outlier handling for each medication.

        Returns:
            Dictionary with raw and cleaned dose statistics per medication
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        from modules.utils.outlier_handling import apply_outlier_ranges, load_outlier_config

        df = self.table.df
        conn = duckdb.connect(':memory:')
        conn.register('df', df)

        # Get unique med_category + med_dose_unit combinations
        med_unit_combos = conn.execute("""
            SELECT DISTINCT med_category, med_dose_unit
            FROM df
            WHERE med_dose IS NOT NULL AND med_dose_unit IS NOT NULL
            ORDER BY med_category, med_dose_unit
        """).fetchdf()

        # Load outlier configuration
        outlier_config = load_outlier_config()

        distributions = {}

        for _, row in med_unit_combos.iterrows():
            med_cat = row['med_category']
            unit = row['med_dose_unit']
            key = f"{med_cat}_{unit}"

            # Get subset for this med/unit combo
            subset_query = """
                SELECT hospitalization_id, admin_dttm, med_dose
                FROM df
                WHERE med_category = ?
                AND med_dose_unit = ?
                AND med_dose IS NOT NULL
            """
            subset = conn.execute(subset_query, [med_cat, unit]).fetchdf()

            if len(subset) == 0:
                continue

            # Calculate raw statistics
            raw_doses = subset['med_dose']
            raw_stats = {
                'count': int(len(raw_doses)),
                'mean': float(raw_doses.mean()),
                'median': float(raw_doses.median()),
                'std': float(raw_doses.std()),
                'min': float(raw_doses.min()),
                'max': float(raw_doses.max()),
                'q25': float(raw_doses.quantile(0.25)),
                'q75': float(raw_doses.quantile(0.75))
            }

            # Apply outlier handling
            cleaned_subset, outlier_stats = apply_outlier_ranges(
                subset.copy(),
                table_name='medication_admin_intermittent',
                outlier_config=outlier_config
            )

            # Calculate cleaned statistics
            cleaned_doses = cleaned_subset['med_dose'].dropna()
            cleaned_stats = {
                'count': int(len(cleaned_doses)),
                'mean': float(cleaned_doses.mean()) if len(cleaned_doses) > 0 else None,
                'median': float(cleaned_doses.median()) if len(cleaned_doses) > 0 else None,
                'std': float(cleaned_doses.std()) if len(cleaned_doses) > 0 else None,
                'min': float(cleaned_doses.min()) if len(cleaned_doses) > 0 else None,
                'max': float(cleaned_doses.max()) if len(cleaned_doses) > 0 else None,
                'q25': float(cleaned_doses.quantile(0.25)) if len(cleaned_doses) > 0 else None,
                'q75': float(cleaned_doses.quantile(0.75)) if len(cleaned_doses) > 0 else None
            }

            distributions[key] = {
                'medication': med_cat,
                'unit': unit,
                'raw_data': raw_stats,
                'cleaned_data': cleaned_stats,
                'outlier_info': outlier_stats.get('med_dose', {})
            }

        conn.close()
        return distributions

    def get_dose_statistics_table(self) -> pd.DataFrame:
        """
        Get dose statistics as a DataFrame for display.

        Returns:
            DataFrame with statistics for each medication/unit combination
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return pd.DataFrame()

        df = self.table.df
        conn = duckdb.connect(':memory:')
        conn.register('df', df)

        # Calculate statistics for each med_category + med_dose_unit combination
        stats_query = """
            SELECT
                med_category as Medication,
                med_dose_unit as Unit,
                COUNT(*) as Count,
                ROUND(MIN(med_dose), 2) as Min,
                ROUND(MAX(med_dose), 2) as Max,
                ROUND(AVG(med_dose), 2) as Mean,
                ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY med_dose), 2) as Median,
                ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY med_dose), 2) as Q1,
                ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY med_dose), 2) as Q3,
                ROUND(STDDEV(med_dose), 2) as StdDev
            FROM df
            WHERE med_dose IS NOT NULL AND med_dose_unit IS NOT NULL
            GROUP BY med_category, med_dose_unit
            HAVING COUNT(*) >= 5
            ORDER BY med_category, med_dose_unit
        """

        stats_df = conn.execute(stats_query).fetchdf()
        conn.close()

        return stats_df

    def generate_distribution_plots(self, max_meds=20) -> Optional[str]:
        """
        Generate a grid of distribution plots for top medications using seaborn.

        Parameters:
        -----------
        max_meds : int
            Maximum number of medications to plot

        Returns:
            Path to saved plot file or None if failed
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return None

        df = self.table.df
        conn = duckdb.connect(':memory:')
        conn.register('df', df)

        # Get top medications by frequency
        top_meds_query = f"""
            SELECT
                med_category,
                med_dose_unit,
                COUNT(*) as count
            FROM df
            WHERE med_dose IS NOT NULL AND med_dose_unit IS NOT NULL
            GROUP BY med_category, med_dose_unit
            ORDER BY count DESC
            LIMIT {max_meds}
        """

        top_meds = conn.execute(top_meds_query).fetchdf()

        if top_meds.empty:
            conn.close()
            return None

        # Calculate grid dimensions
        n_plots = len(top_meds)
        n_cols = min(4, n_plots)  # Max 4 columns
        n_rows = (n_plots + n_cols - 1) // n_cols

        # Set up the plot grid
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))

        # Flatten axes for easier iteration
        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        # Plot each medication/unit combination
        for idx, (_, row) in enumerate(top_meds.iterrows()):
            if idx >= len(axes):
                break

            ax = axes[idx]
            med_cat = row['med_category']
            unit = row['med_dose_unit']

            # Get dose data for this combination
            doses_query = """
                SELECT med_dose
                FROM df
                WHERE med_category = ?
                AND med_dose_unit = ?
                AND med_dose IS NOT NULL
            """
            doses = conn.execute(doses_query, [med_cat, unit]).fetchdf()['med_dose'].values

            # Apply outlier removal for cleaner plots
            if len(doses) > 0:
                q1, q3 = np.percentile(doses, [25, 75])
                iqr = q3 - q1
                lower_bound = max(0, q1 - 1.5 * iqr)
                upper_bound = q3 + 1.5 * iqr
                clean_doses = doses[(doses >= lower_bound) & (doses <= upper_bound)]

                # Create histogram with KDE overlay
                sns.histplot(clean_doses, kde=True, ax=ax, color='skyblue', edgecolor='black', alpha=0.7)

                # Set title and labels
                title = f"{med_cat[:20]}..." if len(med_cat) > 20 else med_cat
                ax.set_title(f"{title}\n({unit})", fontsize=10)
                ax.set_xlabel(f"Dose ({unit})", fontsize=9)
                ax.set_ylabel("Count", fontsize=9)

                # Add sample size annotation
                ax.text(0.98, 0.98, f"n={len(clean_doses)}",
                       transform=ax.transAxes, ha='right', va='top', fontsize=8)

        # Remove empty subplots
        for idx in range(n_plots, len(axes)):
            fig.delaxes(axes[idx])

        # Adjust layout
        plt.tight_layout()

        # Save the plot
        intermediate_dir = os.path.join(self.output_dir, 'intermediate')
        os.makedirs(intermediate_dir, exist_ok=True)

        plot_path = os.path.join(intermediate_dir, 'medication_intermittent_dose_distributions.png')
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()

        conn.close()
        print(f"Saved distribution plots to {plot_path}")
        return plot_path

    def save_name_category_mappings(self) -> Optional[str]:
        """
        Save med_name to med_category mappings with frequency counts.

        Returns:
            Path to saved CSV file or None if failed
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return None

        df = self.table.df
        conn = duckdb.connect(':memory:')
        conn.register('df', df)

        # Get med_name to med_category mappings with counts
        mappings = conn.execute("""
            SELECT
                med_name,
                med_category,
                COUNT(*) as frequency,
                COUNT(DISTINCT hospitalization_id) as unique_hospitalizations
            FROM df
            WHERE med_name IS NOT NULL AND med_category IS NOT NULL
            GROUP BY med_name, med_category
            ORDER BY frequency DESC
        """).fetchdf()

        conn.close()

        if mappings.empty:
            print("No name-category mappings found")
            return None

        # Save to CSV
        intermediate_dir = os.path.join(self.output_dir, 'intermediate')
        os.makedirs(intermediate_dir, exist_ok=True)

        output_path = os.path.join(intermediate_dir, 'medication_intermittent_name_category_mappings.csv')
        mappings.to_csv(output_path, index=False)
        print(f"Saved name-category mappings to {output_path}")

        return output_path

    def save_hospitalization_metrics(self) -> Optional[str]:
        """
        Calculate and save hospitalization metrics by medication group.

        Returns:
            Path to saved CSV file or None if failed
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return None

        df = self.table.df
        conn = duckdb.connect(':memory:')
        conn.register('df', df)

        # Get total hospitalizations
        total_hosps = conn.execute("""
            SELECT COUNT(DISTINCT hospitalization_id) as total
            FROM df
        """).fetchone()[0]

        # Calculate which hospitalizations received which med groups
        hosp_metrics = conn.execute("""
            WITH group_exposure AS (
                SELECT
                    hospitalization_id,
                    med_group,
                    COUNT(*) as administrations,
                    MIN(admin_dttm) as first_admin,
                    MAX(admin_dttm) as last_admin
                FROM df
                WHERE med_group IS NOT NULL
                GROUP BY hospitalization_id, med_group
            )
            SELECT
                med_group,
                COUNT(DISTINCT hospitalization_id) as exposed_hosps,
                ROUND(COUNT(DISTINCT hospitalization_id) * 100.0 / ?, 2) as percentage,
                CAST(AVG(administrations) AS INTEGER) as avg_administrations,
                CAST(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY administrations) AS INTEGER) as median_administrations,
                MIN(administrations) as min_administrations,
                MAX(administrations) as max_administrations
            FROM group_exposure
            GROUP BY med_group
            ORDER BY exposed_hosps DESC
        """, [total_hosps]).fetchdf()

        conn.close()

        if hosp_metrics.empty:
            print("No hospitalization metrics found")
            return None

        # Add total hospitalizations row
        hosp_metrics.loc[len(hosp_metrics)] = {
            'med_group': 'TOTAL_HOSPITALIZATIONS',
            'exposed_hosps': total_hosps,
            'percentage': 100.0,
            'avg_administrations': None,
            'median_administrations': None,
            'min_administrations': None,
            'max_administrations': None
        }

        # Save to CSV
        intermediate_dir = os.path.join(self.output_dir, 'intermediate')
        os.makedirs(intermediate_dir, exist_ok=True)

        output_path = os.path.join(intermediate_dir, 'medication_intermittent_group_hospitalizations.csv')
        hosp_metrics.to_csv(output_path, index=False)
        print(f"Saved hospitalization metrics to {output_path}")

        return output_path

    def analyze_distributions(self) -> Dict[str, Any]:
        """
        Analyze medication distributions comprehensively.

        Returns:
            Dictionary containing all distribution analyses
        """
        distributions = {}

        # Get medication group analysis
        group_analysis = self.analyze_medication_by_group()
        if 'error' not in group_analysis:
            distributions['medication_groups'] = group_analysis

        # Get dose statistics table
        dose_stats = self.get_dose_statistics_table()
        if not dose_stats.empty:
            distributions['dose_statistics'] = dose_stats

        # Save auxiliary files
        self.save_name_category_mappings()
        self.save_hospitalization_metrics()

        # Generate distribution plots
        plot_path = self.generate_distribution_plots()
        if plot_path:
            distributions['plot_path'] = plot_path

        return distributions

    def generate_medication_summary(self) -> pd.DataFrame:
        """
        Generate a summary dataframe for the medication admin intermittent table.

        Returns:
            Summary DataFrame with key metrics
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return pd.DataFrame({'Metric': ['No data available'], 'Value': [None]})

        df = self.table.df
        conn = duckdb.connect(':memory:')
        conn.register('df', df)

        summary_data = []

        # Basic counts
        basic_stats = conn.execute("""
            SELECT
                COUNT(*) as total_records,
                COUNT(DISTINCT hospitalization_id) as unique_hosps,
                COUNT(DISTINCT med_category) as unique_med_categories,
                COUNT(DISTINCT med_group) as unique_groups
            FROM df
        """).fetchone()

        summary_data.append({'Metric': 'Total Administration Records', 'Value': f"{basic_stats[0]:,}"})
        summary_data.append({'Metric': 'Unique Hospitalizations', 'Value': f"{basic_stats[1]:,}"})
        summary_data.append({'Metric': 'Unique Medication Categories', 'Value': f"{basic_stats[2]:,}"})
        summary_data.append({'Metric': 'Unique Medication Groups', 'Value': f"{basic_stats[3]:,}"})

        # Top medication groups by hospitalization exposure
        top_groups = conn.execute("""
            SELECT
                med_group,
                COUNT(DISTINCT hospitalization_id) as hosp_count,
                ROUND(COUNT(DISTINCT hospitalization_id) * 100.0 /
                      (SELECT COUNT(DISTINCT hospitalization_id) FROM df), 2) as percentage
            FROM df
            WHERE med_group IS NOT NULL
            GROUP BY med_group
            ORDER BY hosp_count DESC
            LIMIT 3
        """).fetchdf()

        for _, row in top_groups.iterrows():
            summary_data.append({
                'Metric': f'{row["med_group"].title()} Exposure',
                'Value': f'{row["hosp_count"]:,} hospitalizations ({row["percentage"]:.1f}%)'
            })

        # Top medications by hospitalization count
        top_meds = conn.execute("""
            SELECT
                med_category,
                COUNT(DISTINCT hospitalization_id) as hosp_count
            FROM df
            WHERE med_category IS NOT NULL
            GROUP BY med_category
            ORDER BY hosp_count DESC
            LIMIT 5
        """).fetchdf()

        summary_data.append({'Metric': '', 'Value': ''})  # Separator
        summary_data.append({'Metric': 'Top Medications', 'Value': ''})

        for i, row in top_meds.iterrows():
            summary_data.append({
                'Metric': f'  #{i+1} {row["med_category"]}',
                'Value': f'{row["hosp_count"]:,} hospitalizations'
            })

        # Medications with multiple dose units
        multi_unit_count = conn.execute("""
            SELECT COUNT(*) as count
            FROM (
                SELECT med_category
                FROM df
                WHERE med_dose IS NOT NULL AND med_dose_unit IS NOT NULL
                GROUP BY med_category
                HAVING COUNT(DISTINCT med_dose_unit) > 1
            ) t
        """).fetchone()[0]

        if multi_unit_count > 0:
            summary_data.append({'Metric': '', 'Value': ''})  # Separator
            summary_data.append({'Metric': 'Multi-Unit Medications', 'Value': f'{multi_unit_count} medications'})

        conn.close()

        return pd.DataFrame(summary_data)

    def check_data_quality(self) -> Dict[str, Any]:
        """
        Perform data quality checks specific to medication admin intermittent.

        Returns:
            Dictionary of quality check results
        """
        if self.table is None or not hasattr(self.table, 'df') or self.table.df is None:
            return {'error': 'No data available'}

        df = self.table.df
        conn = duckdb.connect(':memory:')
        conn.register('df', df)

        quality_checks = {}

        # Check for negative doses
        negative_doses = conn.execute("""
            SELECT COUNT(*) as count
            FROM df
            WHERE med_dose < 0
        """).fetchone()[0]

        quality_checks['negative_doses'] = {
            'count': int(negative_doses),
            'percentage': round(negative_doses / len(df) * 100, 2) if len(df) > 0 else 0,
            'status': 'pass' if negative_doses == 0 else 'error',
            'definition': 'Checks for medication doses with negative values, which are clinically impossible and indicate data entry errors.'
        }

        # Check for extreme doses (likely data entry errors)
        extreme_doses = conn.execute("""
            SELECT COUNT(*) as count
            FROM df
            WHERE med_dose > 99999
        """).fetchone()[0]

        quality_checks['extreme_doses'] = {
            'count': int(extreme_doses),
            'percentage': round(extreme_doses / len(df) * 100, 2) if len(df) > 0 else 0,
            'status': 'pass' if extreme_doses == 0 else 'warning',
            'definition': 'Identifies doses exceeding 99,999 units, which are likely data entry errors or unit conversion issues. Such extreme values should be investigated.'
        }

        # Check for mismatched med_category to med_group
        # Load expected mappings from schema
        expected_mappings = {
            # Antibiotics
            'amikacin': 'antibiotics',
            'azithromycin': 'antibiotics',
            'cefepime': 'antibiotics',
            'ceftriaxone': 'antibiotics',
            'ciprofloxacin': 'antibiotics',
            'linezolid': 'antibiotics',
            'meropenem': 'antibiotics',
            'piperacillin-tazobactam': 'antibiotics',
            'vancomycin': 'antibiotics',
            # Anticoagulants
            'heparin': 'anticoagulants',
            'enoxaparin': 'anticoagulants',
            'warfarin': 'anticoagulants',
            # Other medications
            'furosemide': 'diuretics',
            'hydralazine': 'antihypertensives',
            'insulin': 'antidiabetics',
            'lorazepam': 'sedation',
            'morphine': 'analgesia',
            'haloperidol': 'antipsychotics'
        }

        mismatched_count = 0
        if 'med_category' in df.columns and 'med_group' in df.columns:
            for med_cat, expected_group in expected_mappings.items():
                wrong_group = conn.execute("""
                    SELECT COUNT(*) as count
                    FROM df
                    WHERE med_category = ?
                    AND med_group != ?
                    AND med_group IS NOT NULL
                """, [med_cat, expected_group]).fetchone()[0]
                mismatched_count += wrong_group

        quality_checks['mismatched_category_group'] = {
            'count': int(mismatched_count),
            'percentage': round(mismatched_count / len(df) * 100, 2) if len(df) > 0 else 0,
            'status': 'pass' if mismatched_count == 0 else 'warning',
            'definition': 'Validates that medications are correctly assigned to their expected medication groups (e.g., vancomycin should be in antibiotics group). Mismatches may indicate data mapping errors.'
        }

        # Check for missing dose units when dose is present
        missing_units = conn.execute("""
            SELECT COUNT(*) as count
            FROM df
            WHERE med_dose IS NOT NULL
            AND med_dose_unit IS NULL
        """).fetchone()[0]

        quality_checks['missing_dose_units'] = {
            'count': int(missing_units),
            'percentage': round(missing_units / len(df) * 100, 2) if len(df) > 0 else 0,
            'status': 'pass' if missing_units == 0 else 'warning',
            'definition': 'Identifies records where a dose value is present but the unit of measurement is missing. Units are essential for proper dose interpretation and clinical decision-making.'
        }

        conn.close()

        return quality_checks