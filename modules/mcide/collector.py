#!/usr/bin/env python3
"""
Generate MCIDE (Minimum Common Data Elements) and Summary Statistics
Using Polars for efficient scanning without loading full datasets into memory

This script runs independently after validation to collect:
1. MCIDE value counts for categorical variables
2. Summary statistics for numerical variables
"""

import polars as pl
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MCIDEStatsCollector:
    """Collect MCIDE and summary statistics using Polars lazy scanning"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the collector with configuration

        Parameters:
        -----------
        config : dict
            Configuration dictionary with tables_path, output_dir, and filetype
        """
        self.config = config
        self.tables_path = Path(config.get('tables_path', ''))
        self.file_type = config.get('filetype', 'parquet')

        # Set up output directories
        output_base = Path(config.get('output_dir', '../output'))
        self.output_dir = output_base / 'final' / 'tableone'
        self.mcide_dir = self.output_dir / 'mcide'
        self.stats_dir = self.output_dir / 'summary_stats'

        # Create directories if they don't exist
        self.mcide_dir.mkdir(parents=True, exist_ok=True)
        self.stats_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized with tables from: {self.tables_path}")
        logger.info(f"MCIDE output: {self.mcide_dir}")
        logger.info(f"Stats output: {self.stats_dir}")

    def get_table_path(self, table_name: str) -> Optional[Path]:
        """
        Find table path, trying both with and without clif_ prefix

        Parameters:
        -----------
        table_name : str
            Name of the table

        Returns:
        --------
        Path or None
            Path to the table file if found
        """
        # Try both with and without clif_ prefix
        table_paths = [
            self.tables_path / f"clif_{table_name}.{self.file_type}",
            self.tables_path / f"{table_name}.{self.file_type}"
        ]

        for path in table_paths:
            if path.exists():
                logger.info(f"Found table at: {path}")
                return path

        logger.warning(f"Table {table_name} not found. Tried: {table_paths}")
        return None

    def check_column_exists(self, lf: pl.LazyFrame, column: str) -> bool:
        """Check if a column exists in the lazy frame"""
        return column in lf.columns

    def check_columns_exist(self, lf: pl.LazyFrame, columns: List[str]) -> bool:
        """Check if all columns exist in the lazy frame"""
        return all(col in lf.columns for col in columns)

    def collect_mcide(self, lf: pl.LazyFrame, table_name: str, columns: List[str]):
        """
        Collect MCIDE value counts for specified columns

        Parameters:
        -----------
        lf : pl.LazyFrame
            Polars lazy frame
        table_name : str
            Name of the table
        columns : list
            List of column names to group by
        """
        # Check if all columns exist
        valid_columns = [col for col in columns if col in lf.columns]
        if not valid_columns:
            logger.warning(f"No valid columns found for {table_name}: {columns}")
            return

        if len(valid_columns) != len(columns):
            missing = set(columns) - set(valid_columns)
            logger.info(f"Skipping missing columns in {table_name}: {missing}")

        try:
            # Collect value counts
            result = (
                lf.select(valid_columns)
                .group_by(valid_columns, maintain_order=True)
                .agg(pl.count().alias("N"))
                .sort("N", descending=True)
                .collect()
            )

            # Save to CSV with new naming convention
            columns_str = '_'.join(valid_columns)
            filename = f"{table_name}_{columns_str}_mcide.csv"
            output_path = self.mcide_dir / filename

            result.write_csv(output_path)
            logger.info(f"✓ Saved MCIDE: {filename} ({len(result)} unique combinations)")

        except Exception as e:
            logger.error(f"Error collecting MCIDE for {table_name} {columns}: {e}")

    def save_summary_stats(self, stats: pl.DataFrame, name: str, format: str = 'both'):
        """
        Save summary statistics to JSON and/or CSV

        Parameters:
        -----------
        stats : pl.DataFrame
            Statistics dataframe
        name : str
            Name for the output file
        format : str
            Output format: 'json', 'csv', or 'both'
        """
        try:
            if format in ['json', 'both']:
                # Convert to dictionary for JSON
                stats_dict = stats.to_dicts()
                json_path = self.stats_dir / f"{name}.json"
                with open(json_path, 'w') as f:
                    json.dump(stats_dict, f, indent=2, default=str)
                logger.info(f"✓ Saved stats JSON: {name}.json")

            if format in ['csv', 'both']:
                csv_path = self.stats_dir / f"{name}.csv"
                stats.write_csv(csv_path)
                logger.info(f"✓ Saved stats CSV: {name}.csv")

        except Exception as e:
            logger.error(f"Error saving summary stats for {name}: {e}")

    def collect_patient(self):
        """Collect MCIDE for patient table"""
        logger.info("Processing patient table...")

        table_path = self.get_table_path('patient')
        if not table_path:
            return

        lf = pl.scan_parquet(table_path) if self.file_type == 'parquet' else pl.scan_csv(table_path)

        # MCIDE collections
        self.collect_mcide(lf, 'patient', ['race_name', 'race_category'])
        self.collect_mcide(lf, 'patient', ['ethnicity_name', 'ethnicity_category'])

    def collect_hospitalization(self):
        """Collect MCIDE for hospitalization table"""
        logger.info("Processing hospitalization table...")

        table_path = self.get_table_path('hospitalization')
        if not table_path:
            return

        lf = pl.scan_parquet(table_path) if self.file_type == 'parquet' else pl.scan_csv(table_path)

        # MCIDE collections
        self.collect_mcide(lf, 'hospitalization', ['admission_type_name', 'admission_type_category'])
        self.collect_mcide(lf, 'hospitalization', ['discharge_name', 'discharge_category'])

    def collect_adt(self):
        """Collect MCIDE for ADT table"""
        logger.info("Processing ADT table...")

        table_path = self.get_table_path('adt')
        if not table_path:
            return

        lf = pl.scan_parquet(table_path) if self.file_type == 'parquet' else pl.scan_csv(table_path)

        # MCIDE collections
        self.collect_mcide(lf, 'adt', ['location_name', 'location_category', 'location_type'])

    def collect_labs_stats(self):
        """Labs: MCIDE + summary stats for lab_value_numeric"""
        logger.info("Processing labs table...")

        table_path = self.get_table_path('labs')
        if not table_path:
            return

        lf = pl.scan_parquet(table_path) if self.file_type == 'parquet' else pl.scan_csv(table_path)

        # MCIDE collections
        self.collect_mcide(lf, 'labs', ['lab_name', 'lab_category', 'lab_loinc_code'])

        # Check for optional columns
        if self.check_columns_exist(lf, ['lab_specimen_name', 'lab_specimen_category']):
            self.collect_mcide(lf, 'labs', ['lab_specimen_name', 'lab_specimen_category'])

        if self.check_columns_exist(lf, ['lab_order_name', 'lab_order_category']):
            self.collect_mcide(lf, 'labs', ['lab_order_name', 'lab_order_category'])

        # Summary statistics for lab_value_numeric by category
        if self.check_column_exists(lf, 'lab_value_numeric'):
            logger.info("Calculating labs summary statistics...")
            try:
                stats = (
                    lf.group_by('lab_category')
                    .agg([
                        pl.count().alias('total_obs'),
                        pl.col('lab_value_numeric').count().alias('n'),
                        pl.col('lab_value_numeric').null_count().alias('missing'),
                        pl.col('lab_value_numeric').min().alias('min'),
                        pl.col('lab_value_numeric').max().alias('max'),
                        pl.col('lab_value_numeric').mean().alias('mean'),
                        pl.col('lab_value_numeric').median().alias('median'),
                        pl.col('lab_value_numeric').std().alias('sd'),
                        pl.col('lab_value_numeric').quantile(0.25).alias('q1'),
                        pl.col('lab_value_numeric').quantile(0.75).alias('q3')
                    ])
                    .collect()
                )
                self.save_summary_stats(stats, 'labs_summary_by_category')
            except Exception as e:
                logger.error(f"Error calculating labs summary stats: {e}")

    def collect_vitals_stats(self):
        """Vitals: MCIDE + summary stats for vital_value by category and name"""
        logger.info("Processing vitals table...")

        table_path = self.get_table_path('vitals')
        if not table_path:
            return

        lf = pl.scan_parquet(table_path) if self.file_type == 'parquet' else pl.scan_csv(table_path)

        # MCIDE collections
        self.collect_mcide(lf, 'vitals', ['vital_name', 'vital_category'])

        # Summary statistics for vital_value by category and name
        if self.check_columns_exist(lf, ['vital_value', 'vital_category', 'vital_name']):
            logger.info("Calculating vitals summary statistics...")
            try:
                # Calculate stats including missing count - grouped by category AND name
                stats = (
                    lf.group_by(['vital_category', 'vital_name'])
                    .agg([
                        pl.count().alias('total_obs'),
                        pl.col('vital_value').count().alias('n'),
                        pl.col('vital_value').null_count().alias('missing'),
                        pl.col('vital_value').min().alias('min'),
                        pl.col('vital_value').max().alias('max'),
                        pl.col('vital_value').mean().alias('mean'),
                        pl.col('vital_value').median().alias('median'),
                        pl.col('vital_value').std().alias('sd'),
                        pl.col('vital_value').quantile(0.25).alias('q1'),
                        pl.col('vital_value').quantile(0.75).alias('q3')
                    ])
                    .collect()
                )
                self.save_summary_stats(stats, 'vitals_summary_by_category_and_name')
            except Exception as e:
                logger.error(f"Error calculating vitals summary stats: {e}")

    def collect_medication_stats(self, table_type: str = 'continuous'):
        """Medications: MCIDE + summary stats for med_dose by med_category and med_dose_unit"""
        table_name = f'medication_admin_{table_type}'
        logger.info(f"Processing {table_name} table...")

        table_path = self.get_table_path(table_name)
        if not table_path:
            return

        # Keep the full name with possible prefix for output
        full_table_name = f'clif_{table_name}' if 'clif_' in str(table_path) else table_name
        lf = pl.scan_parquet(table_path) if self.file_type == 'parquet' else pl.scan_csv(table_path)

        # MCIDE collections
        self.collect_mcide(lf, table_name, ['med_name', 'med_category'])

        # Check for optional columns
        if self.check_columns_exist(lf, ['med_route_name', 'med_route_category']):
            self.collect_mcide(lf, table_name, ['med_route_name', 'med_route_category'])

        if self.check_columns_exist(lf, ['mar_action_name', 'mar_action_category']):
            self.collect_mcide(lf, table_name, ['mar_action_name', 'mar_action_category'])

        # Summary statistics for med_dose grouped by med_category and med_dose_unit
        if self.check_columns_exist(lf, ['med_dose', 'med_dose_unit', 'med_category']):
            logger.info(f"Calculating {table_name} dose statistics...")
            try:
                stats = (
                    lf.group_by(['med_category', 'med_dose_unit'])
                    .agg([
                        pl.count().alias('total_obs'),
                        pl.col('med_dose').count().alias('n'),
                        pl.col('med_dose').null_count().alias('missing'),
                        pl.col('med_dose').min().alias('min'),
                        pl.col('med_dose').max().alias('max'),
                        pl.col('med_dose').mean().alias('mean'),
                        pl.col('med_dose').median().alias('median'),
                        pl.col('med_dose').std().alias('sd'),
                        pl.col('med_dose').quantile(0.25).alias('q1'),
                        pl.col('med_dose').quantile(0.75).alias('q3')
                    ])
                    .collect()
                )
                self.save_summary_stats(stats, f'{table_name}_dose_by_category_and_unit')
            except Exception as e:
                logger.error(f"Error calculating {table_name} dose stats: {e}")

    def collect_crrt_stats(self):
        """CRRT: MCIDE + summary stats for numerical variables"""
        logger.info("Processing CRRT therapy table...")

        table_path = self.get_table_path('crrt_therapy')
        if not table_path:
            return

        lf = pl.scan_parquet(table_path) if self.file_type == 'parquet' else pl.scan_csv(table_path)

        # MCIDE collections
        self.collect_mcide(lf, 'clif_crrt_therapy', ['crrt_mode_name', 'crrt_mode_category'])

        # Summary statistics for CRRT numerical variables
        numeric_cols = [
            'blood_flow_rate',
            'pre_filter_replacement_fluid_rate',
            'post_filter_replacement_fluid_rate',
            'dialysate_flow_rate',
            'ultrafiltration_out'
        ]

        for col in numeric_cols:
            if self.check_column_exists(lf, col):
                logger.info(f"Calculating CRRT stats for {col}...")
                try:
                    # Overall stats with missing count
                    overall_stats = (
                        lf.select([
                            pl.lit(col).alias('variable'),
                            pl.count().alias('total_obs'),
                            pl.col(col).count().alias('n'),
                            pl.col(col).null_count().alias('missing'),
                            pl.col(col).min().alias('min'),
                            pl.col(col).max().alias('max'),
                            pl.col(col).mean().alias('mean'),
                            pl.col(col).median().alias('median'),
                            pl.col(col).std().alias('sd'),
                            pl.col(col).quantile(0.25).alias('q1'),
                            pl.col(col).quantile(0.75).alias('q3')
                        ])
                        .collect()
                    )
                    self.save_summary_stats(overall_stats, f'crrt_{col}_overall')

                    # Stats by mode category with missing count
                    if self.check_column_exists(lf, 'crrt_mode_category'):
                        mode_stats = (
                            lf.group_by('crrt_mode_category')
                            .agg([
                                pl.count().alias('total_obs'),
                                pl.col(col).count().alias('n'),
                                pl.col(col).null_count().alias('missing'),
                                pl.col(col).min().alias('min'),
                                pl.col(col).max().alias('max'),
                                pl.col(col).mean().alias('mean'),
                                pl.col(col).median().alias('median'),
                                pl.col(col).std().alias('sd'),
                                pl.col(col).quantile(0.25).alias('q1'),
                                pl.col(col).quantile(0.75).alias('q3')
                            ])
                            .collect()
                        )
                        self.save_summary_stats(mode_stats, f'crrt_{col}_by_mode')

                except Exception as e:
                    logger.error(f"Error calculating CRRT stats for {col}: {e}")

    def collect_ecmo_stats(self):
        """ECMO: MCIDE + summary stats for numerical variables"""
        logger.info("Processing ECMO/MCS table...")
        table_path = self.tables_path / f"clif_ecmo_mcs.{self.file_type}"

        if not table_path.exists():
            logger.warning(f"ECMO/MCS table not found: {table_path}")
            return

        lf = pl.scan_parquet(table_path) if self.file_type == 'parquet' else pl.scan_csv(table_path)

        # MCIDE collections
        self.collect_mcide(lf, 'ecmo_mcs', ['device_name', 'device_category'])

        # Summary statistics for ECMO numerical variables
        numeric_cols = ['device_rate', 'sweep', 'flow', 'fdO2']

        for col in numeric_cols:
            if self.check_column_exists(lf, col):
                logger.info(f"Calculating ECMO stats for {col}...")
                try:
                    # Overall stats with missing count
                    overall_stats = (
                        lf.select([
                            pl.lit(col).alias('variable'),
                            pl.count().alias('total_obs'),
                            pl.col(col).count().alias('n'),
                            pl.col(col).null_count().alias('missing'),
                            pl.col(col).min().alias('min'),
                            pl.col(col).max().alias('max'),
                            pl.col(col).mean().alias('mean'),
                            pl.col(col).median().alias('median'),
                            pl.col(col).std().alias('sd'),
                            pl.col(col).quantile(0.25).alias('q1'),
                            pl.col(col).quantile(0.75).alias('q3')
                        ])
                        .collect()
                    )
                    self.save_summary_stats(overall_stats, f'ecmo_{col}_overall')

                    # Stats by device category with missing count
                    if self.check_column_exists(lf, 'device_category'):
                        category_stats = (
                            lf.group_by('device_category')
                            .agg([
                                pl.count().alias('total_obs'),
                                pl.col(col).count().alias('n'),
                                pl.col(col).null_count().alias('missing'),
                                pl.col(col).min().alias('min'),
                                pl.col(col).max().alias('max'),
                                pl.col(col).mean().alias('mean'),
                                pl.col(col).median().alias('median'),
                                pl.col(col).std().alias('sd'),
                                pl.col(col).quantile(0.25).alias('q1'),
                                pl.col(col).quantile(0.75).alias('q3')
                            ])
                            .collect()
                        )
                        self.save_summary_stats(category_stats, f'ecmo_{col}_by_category')

                except Exception as e:
                    logger.error(f"Error calculating ECMO stats for {col}: {e}")

    def collect_code_status(self):
        """Collect MCIDE for code status table"""
        logger.info("Processing code status table...")
        table_path = self.tables_path / f"clif_code_status.{self.file_type}"

        if not table_path.exists():
            logger.warning(f"Code status table not found: {table_path}")
            return

        lf = pl.scan_parquet(table_path) if self.file_type == 'parquet' else pl.scan_csv(table_path)

        # MCIDE collections
        self.collect_mcide(lf, 'code_status', ['code_status_name', 'code_status_category'])

    def collect_microbiology(self, table_type: str = 'culture'):
        """Collect MCIDE for microbiology tables"""
        table_name = f'clif_microbiology_{table_type}'
        logger.info(f"Processing {table_name} table...")
        table_path = self.tables_path / f"{table_name}.{self.file_type}"

        if not table_path.exists():
            logger.warning(f"{table_name} table not found: {table_path}")
            return

        lf = pl.scan_parquet(table_path) if self.file_type == 'parquet' else pl.scan_csv(table_path)

        # MCIDE collections common to both culture and nonculture
        self.collect_mcide(lf, table_name, ['fluid_name', 'fluid_category', 'lab_loinc_code'])
        self.collect_mcide(lf, table_name, ['method_name', 'method_category'])
        self.collect_mcide(lf, table_name, ['organism_name', 'organism_category'])

        # Additional for nonculture
        if table_type == 'nonculture' and self.check_column_exists(lf, 'micro_order_name'):
            self.collect_mcide(lf, table_name, ['micro_order_name'])

    def collect_microbiology_susceptibility(self):
        """Collect MCIDE for microbiology susceptibility table"""
        logger.info("Processing microbiology susceptibility table...")
        table_path = self.tables_path / f"clif_microbiology_susceptibility.{self.file_type}"

        if not table_path.exists():
            logger.warning(f"Microbiology susceptibility table not found: {table_path}")
            return

        lf = pl.scan_parquet(table_path) if self.file_type == 'parquet' else pl.scan_csv(table_path)

        # MCIDE collections
        self.collect_mcide(lf, 'microbiology_susceptibility', ['organism_name', 'organism_category'])

    def collect_patient_assessments(self):
        """Collect MCIDE for patient assessments table"""
        logger.info("Processing patient assessments table...")
        table_path = self.tables_path / f"clif_patient_assessments.{self.file_type}"

        if not table_path.exists():
            logger.warning(f"Patient assessments table not found: {table_path}")
            return

        lf = pl.scan_parquet(table_path) if self.file_type == 'parquet' else pl.scan_csv(table_path)

        # MCIDE collections
        self.collect_mcide(lf, 'patient_assessments', ['assessment_name', 'assessment_category', 'assessment_group'])

    # def collect_patient_procedures(self):
    #     """Collect MCIDE for patient procedures table"""
    #     logger.info("Processing patient procedures table...")
    #     table_path = self.get_table_path('patient_procedures')

    #     if not table_path:
    #         logger.warning(f"Patient procedures table not found")
    #         return

    #     lf = pl.scan_parquet(table_path) if self.file_type == 'parquet' else pl.scan_csv(table_path)

    #     # MCIDE collections
    #     self.collect_mcide(lf, 'patient_procedures', ['procedure_name', 'procedure_category'])

    def collect_position(self):
        """Collect MCIDE for position table"""
        logger.info("Processing position table...")
        table_path = self.get_table_path('position')

        if not table_path:
            logger.warning(f"Position table not found")
            return

        lf = pl.scan_parquet(table_path) if self.file_type == 'parquet' else pl.scan_csv(table_path)

        # MCIDE collections
        self.collect_mcide(lf, 'position', ['position_name', 'position_category'])

    def collect_hospital_diagnosis(self):
        """Collect MCIDE for hospital diagnosis table"""
        logger.info("Processing hospital diagnosis table...")
        table_path = self.get_table_path('hospital_diagnosis')

        if not table_path:
            logger.warning(f"Hospital diagnosis table not found")
            return

        lf = pl.scan_parquet(table_path) if self.file_type == 'parquet' else pl.scan_csv(table_path)

        # MCIDE collections
        self.collect_mcide(lf, 'hospital_diagnosis', ['diagnosis_code', 'diagnosis_code_type', 'diagnosis_type', 'poc_flag'])

    def collect_respiratory_support(self):
        """Collect MCIDE for respiratory support table"""
        logger.info("Processing respiratory support table...")
        table_path = self.tables_path / f"clif_respiratory_support.{self.file_type}"

        if not table_path.exists():
            logger.warning(f"Respiratory support table not found: {table_path}")
            return

        lf = pl.scan_parquet(table_path) if self.file_type == 'parquet' else pl.scan_csv(table_path)

        # MCIDE collections
        self.collect_mcide(lf, 'respiratory_support', ['device_name', 'device_category'])
        self.collect_mcide(lf, 'respiratory_support', ['mode_name', 'mode_category'])

    def run_all(self):
        """Run all collections"""
        start_time = datetime.now()
        logger.info("="*80)
        logger.info("Starting MCIDE and Summary Statistics Collection")
        logger.info("="*80)

        # Tables without summary stats
        self.collect_patient()
        self.collect_hospitalization()
        self.collect_adt()
        self.collect_code_status()
        self.collect_patient_assessments()
        # self.collect_patient_procedures()
        self.collect_respiratory_support()

        # Microbiology tables
        self.collect_microbiology('culture')
        self.collect_microbiology('nonculture')
        self.collect_microbiology_susceptibility()

        # Tables with summary stats
        self.collect_labs_stats()
        self.collect_vitals_stats()
        self.collect_medication_stats('continuous')
        self.collect_medication_stats('intermittent')
        self.collect_crrt_stats()
        self.collect_ecmo_stats()

        # Calculate and log runtime
        runtime = datetime.now() - start_time
        logger.info("="*80)
        logger.info(f"✅ Collection complete! Total runtime: {runtime}")
        logger.info("="*80)

        # Summary of outputs
        mcide_count = len(list(self.mcide_dir.glob('*.csv')))
        stats_count = len(list(self.stats_dir.glob('*.*')))
        logger.info(f"Generated {mcide_count} MCIDE files and {stats_count} summary statistics files")
        logger.info(f"MCIDE directory: {self.mcide_dir}")
        logger.info(f"Stats directory: {self.stats_dir}")


def load_config(config_path: str = '../config/config.json') -> Dict[str, Any]:
    """Load configuration from JSON file"""
    config_path = Path(config_path)

    # Try multiple possible config locations
    if not config_path.exists():
        alt_paths = [
            Path('config/config.json'),
            Path('../config/config.json'),
            Path('../../config/config.json')
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                config_path = alt_path
                break

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found. Tried: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    logger.info(f"Loaded configuration from: {config_path}")
    return config


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Generate MCIDE and Summary Statistics')
    parser.add_argument(
        '--config',
        type=str,
        default='../config/config.json',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--tables',
        type=str,
        nargs='+',
        help='Specific tables to process (default: all tables)'
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config(args.config)

        # Initialize collector
        collector = MCIDEStatsCollector(config)

        if args.tables:
            # Process specific tables
            logger.info(f"Processing specific tables: {args.tables}")
            for table in args.tables:
                method_name = f"collect_{table}"
                if hasattr(collector, method_name):
                    getattr(collector, method_name)()
                else:
                    logger.warning(f"No collection method found for table: {table}")
        else:
            # Process all tables
            collector.run_all()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())