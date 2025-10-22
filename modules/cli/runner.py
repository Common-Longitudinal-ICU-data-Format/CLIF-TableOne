"""CLI Analysis Runner for CLIF tables."""

import os
import json
from typing import Dict, Any, List, Optional
from modules.tables import (
    PatientAnalyzer, HospitalizationAnalyzer, ADTAnalyzer, CodeStatusAnalyzer,
    CRRTTherapyAnalyzer, ECMOMCSAnalyzer, HospitalDiagnosisAnalyzer, LabsAnalyzer,
    MedicationAdminContinuousAnalyzer, MedicationAdminIntermittentAnalyzer,
    MicrobiologyCultureAnalyzer, MicrobiologyNoncultureAnalyzer,
    MicrobiologySusceptibilityAnalyzer, PatientAssessmentsAnalyzer,
    PatientProceduresAnalyzer, PositionAnalyzer, RespiratorySupportAnalyzer,
    VitalsAnalyzer
)
from modules.utils import load_sample_list, sample_exists
from .formatters import ConsoleFormatter
from .pdf_generator import ValidationPDFGenerator


class CLIAnalysisRunner:
    """Orchestrates CLI-based analysis of CLIF tables."""

    TABLE_ANALYZERS = {
        'patient': PatientAnalyzer,
        'hospitalization': HospitalizationAnalyzer,
        'adt': ADTAnalyzer,
        'code_status': CodeStatusAnalyzer,
        'crrt_therapy': CRRTTherapyAnalyzer,
        'ecmo_mcs': ECMOMCSAnalyzer,
        'hospital_diagnosis': HospitalDiagnosisAnalyzer,
        'labs': LabsAnalyzer,
        'medication_admin_continuous': MedicationAdminContinuousAnalyzer,
        'medication_admin_intermittent': MedicationAdminIntermittentAnalyzer,
        'microbiology_culture': MicrobiologyCultureAnalyzer,
        'microbiology_nonculture': MicrobiologyNoncultureAnalyzer,
        'microbiology_susceptibility': MicrobiologySusceptibilityAnalyzer,
        'patient_assessments': PatientAssessmentsAnalyzer,
        'patient_procedures': PatientProceduresAnalyzer,
        'position': PositionAnalyzer,
        'respiratory_support': RespiratorySupportAnalyzer,
        'vitals': VitalsAnalyzer
    }

    # Tables that support hospitalization_id filtering (can use 1k ICU sample)
    # Excludes: patient (uses patient_id), hospitalization (defines the sample),
    # adt (used to create the sample), code_status (uses patient_id)
    SAMPLE_ELIGIBLE_TABLES = [
        'labs',
        'medication_admin_continuous',
        'medication_admin_intermittent',
        'microbiology_culture',
        'microbiology_nonculture',
        'microbiology_susceptibility',
        'vitals',
        'patient_assessments',
        'respiratory_support',
        'position',
        'patient_procedures',
        'crrt_therapy',
        'ecmo_mcs',
        'hospital_diagnosis'
    ]

    def __init__(self, config: Dict[str, Any], verbose: bool = False, quiet: bool = False,
                 generate_pdf: bool = True, use_sample: bool = False):
        """
        Initialize the CLI runner.

        Parameters:
        -----------
        config : dict
            Configuration dictionary with data_dir, filetype, etc.
        verbose : bool
            Enable verbose output
        quiet : bool
            Minimize output
        generate_pdf : bool
            Generate PDF reports for validation results
        use_sample : bool
            Use 1k ICU sample for faster analysis
        """
        self.config = config
        self.verbose = verbose
        self.quiet = quiet
        self.generate_pdf = generate_pdf
        self.use_sample = use_sample
        self.formatter = ConsoleFormatter()
        self.pdf_generator = ValidationPDFGenerator()

        # Extract config values
        self.data_dir = config.get('tables_path', './data')
        # Support both 'filetype' and 'file_type' keys in config
        self.filetype = config.get('filetype') or config.get('file_type', 'parquet')
        self.timezone = config.get('timezone', 'UTC')
        self.output_dir = config.get('output_dir', 'output')
        self.site_name = config.get('site_name')

    def log(self, message: str, force: bool = False):
        """Log message to console if not in quiet mode."""
        if not self.quiet or force:
            print(message)

    def run_table_analysis(self, table_name: str, run_validation: bool, run_summary: bool) -> Dict[str, Any]:
        """
        Run analysis for a single table.

        Parameters:
        -----------
        table_name : str
            Name of the table to analyze
        run_validation : bool
            Whether to run validation
        run_summary : bool
            Whether to generate summary statistics

        Returns:
        --------
        dict
            Results with status, validation, and summary data
        """
        result = {
            'table': table_name,
            'success': False,
            'validation': None,
            'summary': None,
            'error': None
        }

        try:
            # Get analyzer class
            analyzer_class = self.TABLE_ANALYZERS.get(table_name)
            if not analyzer_class:
                result['error'] = f"Analyzer not implemented for {table_name}"
                self.log(self.formatter.error(f"Analyzer not available for {table_name}"))
                return result

            # Load sample if requested
            sample_filter = None
            if self.use_sample:
                if sample_exists(self.output_dir):
                    sample_filter = load_sample_list(self.output_dir)
                    if sample_filter:
                        self.log(self.formatter.info(f"[STATS] Using 1k ICU sample ({len(sample_filter):,} hospitalizations)"))
                    else:
                        self.log(self.formatter.warning("[WARNING] Sample file exists but could not be loaded. Loading full table."))
                else:
                    self.log(self.formatter.warning("[WARNING] Sample file not found. Loading full table."))
                    self.log(self.formatter.info("   Generate sample by running: python run_analysis.py --adt --validate --summary"))

            # Load table
            # Only pass sample_filter to tables that support hospitalization_id filtering
            if table_name in self.SAMPLE_ELIGIBLE_TABLES and sample_filter:
                self.log(self.formatter.progress(f"Loading {table_name} table with 1k sample"))
                analyzer = analyzer_class(self.data_dir, self.filetype, self.timezone, self.output_dir, sample_filter)
            else:
                if sample_filter and table_name not in self.SAMPLE_ELIGIBLE_TABLES:
                    self.log(self.formatter.info(f"Note: {table_name} does not support sampling, loading full dataset"))
                self.log(self.formatter.progress(f"Loading {table_name} table"))
                analyzer = analyzer_class(self.data_dir, self.filetype, self.timezone, self.output_dir, None)

            if analyzer.table is None:
                result['error'] = f"Failed to load {table_name} table"
                self.log(self.formatter.error(f"Could not load {table_name} table"))
                return result

            self.log(self.formatter.success(f"Loaded {table_name} table"))

            # Run validation
            if run_validation:
                self.log(self.formatter.progress(f"Running validation for {table_name}"))
                validation_results = analyzer.validate()
                result['validation'] = validation_results

                # Save validation results
                try:
                    analyzer.save_summary_data(validation_results, '_validation')
                    self.log(self.formatter.success(f"Validation results saved"))
                except Exception as e:
                    self.log(self.formatter.warning(f"Could not save validation results: {e}"))

                # Generate PDF report
                if self.generate_pdf:
                    try:
                        reports_dir = os.path.join(self.output_dir, 'final', 'reports')
                        os.makedirs(reports_dir, exist_ok=True)

                        pdf_path = os.path.join(reports_dir, f"{table_name}_validation_report.pdf")

                        if self.pdf_generator.is_available():
                            self.log(self.formatter.progress(f"Generating PDF report for {table_name}"))
                            self.pdf_generator.generate_validation_pdf(
                                validation_results,
                                table_name,
                                pdf_path,
                                self.site_name
                            )
                            self.log(self.formatter.success(f"Validation PDF report saved: {table_name}_validation_report.pdf"))
                        else:
                            # Fall back to text report
                            txt_path = os.path.join(reports_dir, f"{table_name}_validation_report.txt")
                            self.log(self.formatter.info(f"reportlab not available, generating text report instead"))
                            self.pdf_generator.generate_text_report(
                                validation_results,
                                table_name,
                                txt_path,
                                self.site_name
                            )
                            self.log(self.formatter.success(f"Validation text report saved: {table_name}_validation_report.txt"))
                    except Exception as e:
                        self.log(self.formatter.warning(f"Could not generate validation report: {e}"))
                        if self.verbose:
                            import traceback
                            traceback.print_exc()

                # Display validation summary
                if self.verbose:
                    self.log(self.formatter.format_validation_summary(validation_results, table_name))

            # Run summary
            if run_summary:
                self.log(self.formatter.progress(f"Calculating summary statistics for {table_name}"))
                summary_stats = analyzer.get_summary_statistics()
                result['summary'] = summary_stats

                # Save summary results
                try:
                    analyzer.save_summary_data(summary_stats, '_summary')
                    self.log(self.formatter.success(f"Summary statistics saved"))
                except Exception as e:
                    self.log(self.formatter.warning(f"Could not save summary statistics: {e}"))

                # Save summary CSVs
                try:
                    results_dir = os.path.join(self.output_dir, 'final', 'results')
                    os.makedirs(results_dir, exist_ok=True)

                    # Save patient demographics summary
                    if hasattr(analyzer, 'generate_patient_summary'):
                        patient_summary_df = analyzer.generate_patient_summary()
                        if not patient_summary_df.empty:
                            csv_filepath = os.path.join(results_dir, f"{table_name}_demographics_summary.csv")
                            patient_summary_df.to_csv(csv_filepath, index=False)
                            self.log(self.formatter.success(f"Patient demographics summary CSV saved"))

                    # Save hospitalization summary
                    if hasattr(analyzer, 'generate_hospitalization_summary'):
                        hosp_summary_df = analyzer.generate_hospitalization_summary()
                        if not hosp_summary_df.empty:
                            csv_filepath = os.path.join(results_dir, f"{table_name}_summary.csv")
                            hosp_summary_df.to_csv(csv_filepath, index=False)
                            self.log(self.formatter.success(f"Hospitalization summary CSV saved"))
                except Exception as e:
                    self.log(self.formatter.warning(f"Could not save summary CSV files: {e}"))

                # Display summary info
                if self.verbose:
                    self.log(self.formatter.format_summary_info(summary_stats, table_name))

            # Generate ICU sample after ADT analysis if --sample was requested and sample doesn't exist
            if table_name == 'adt' and self.use_sample and analyzer.table is not None:
                from modules.utils.sampling import (
                    get_icu_hospitalizations_from_adt,
                    generate_stratified_sample,
                    save_sample_list
                )

                # Only generate if sample doesn't already exist
                if not sample_exists(self.output_dir):
                    try:
                        self.log(self.formatter.progress("Generating 1k ICU sample for future analyses"))

                        # Step 1: Get ICU hospitalizations from ADT
                        icu_hosp_ids = get_icu_hospitalizations_from_adt(analyzer.table.df)

                        if len(icu_hosp_ids) > 0:
                            # Step 2: Load hospitalization table to get years
                            hosp_analyzer = HospitalizationAnalyzer(self.data_dir, self.filetype, self.timezone, self.output_dir)
                            if hosp_analyzer.table is not None:
                                # Step 3: Generate stratified sample
                                sample_ids = generate_stratified_sample(
                                    hosp_analyzer.table.df,
                                    icu_hosp_ids,
                                    sample_size=1000
                                )

                                # Step 4: Save for future use
                                save_sample_list(sample_ids, self.output_dir)
                                self.log(self.formatter.success(f"Generated 1k ICU sample (stratified by year) - {len(sample_ids):,} hospitalizations"))
                            else:
                                self.log(self.formatter.warning("Could not load hospitalization table for sampling"))
                        else:
                            self.log(self.formatter.warning("No ICU hospitalizations found in ADT table"))
                    except Exception as e:
                        self.log(self.formatter.warning(f"Could not generate sample: {e}"))
                        if self.verbose:
                            import traceback
                            traceback.print_exc()

            result['success'] = True
            self.log(self.formatter.success(f"Completed analysis for {table_name}"))

        except Exception as e:
            result['error'] = str(e)
            self.log(self.formatter.error(f"Error analyzing {table_name}: {e}"))
            if self.verbose:
                import traceback
                traceback.print_exc()

        return result

    def run_analysis(self, tables: List[str], run_validation: bool, run_summary: bool) -> Dict[str, Any]:
        """
        Run analysis for multiple tables.

        Parameters:
        -----------
        tables : list
            List of table names to analyze
        run_validation : bool
            Whether to run validation
        run_summary : bool
            Whether to generate summary statistics

        Returns:
        --------
        dict
            Overall results with per-table details
        """
        # If using sample mode, ensure ADT is processed early (after patient/hospitalization)
        # so the sample can be generated before other tables need it
        if self.use_sample and 'adt' in tables:
            # Reorder to ensure: patient, hospitalization, adt come first (in that order if present)
            priority_tables = ['patient', 'hospitalization', 'adt']
            ordered_tables = []

            # Add priority tables first (in order)
            for table in priority_tables:
                if table in tables:
                    ordered_tables.append(table)

            # Add remaining tables
            for table in tables:
                if table not in priority_tables:
                    ordered_tables.append(table)

            tables = ordered_tables

        # Header
        self.log(self.formatter.header("[HOSPITAL] CLIF TABLE ONE ANALYSIS"), force=True)
        self.log(f"{self.formatter.FOLDER} Data Directory: {self.data_dir}", force=True)
        self.log(f"{self.formatter.FILE} Output Directory: {os.path.join(self.output_dir, 'final', 'reports')} (reports), {os.path.join(self.output_dir, 'final', 'results')} (results)", force=True)
        self.log(f"[LIST] Tables: {', '.join(tables)}", force=True)
        self.log(f"[SEARCH] Validation: {'[OK]' if run_validation else '[X]'}", force=True)
        self.log(f"[STATS] Summary: {'[OK]' if run_summary else '[X]'}", force=True)
        self.log(f"[TARGET] Sample Mode: {'[OK] (1k ICU hospitalizations)' if self.use_sample else '[X]'}", force=True)
        self.log("", force=True)

        results = {
            'tables_analyzed': [],
            'tables_failed': [],
            'total_success': 0,
            'total_failed': 0,
            'details': {}
        }

        # Process each table
        for table_name in tables:
            self.log(f"\n{self.formatter.section(f'Processing {table_name.upper()} table')}")

            result = self.run_table_analysis(table_name, run_validation, run_summary)
            results['details'][table_name] = result

            if result['success']:
                results['tables_analyzed'].append(table_name)
                results['total_success'] += 1
            else:
                results['tables_failed'].append(table_name)
                results['total_failed'] += 1

        # Run MCIDE collection if validation was performed and we have successful tables
        if run_validation and results['tables_analyzed']:
            self.log(f"\n{self.formatter.section('MCIDE Statistics Collection')}", force=True)
            try:
                # Import from the new module location
                from modules.mcide import MCIDEStatsCollector
                
                # Create MCIDE collector with the same config
                mcide_config = {
                    'tables_path': self.data_dir,
                    'output_dir': self.output_dir,
                    'filetype': self.filetype
                }

                collector = MCIDEStatsCollector(mcide_config)
                
                # Track successful collections
                mcide_success = 0
                mcide_failed = []

                # Define collection methods for each table
                collection_methods = {
                    'patient': collector.collect_patient,
                    'hospitalization': collector.collect_hospitalization,
                    'adt': collector.collect_adt,
                    'labs': collector.collect_labs_stats,
                    'vitals': collector.collect_vitals_stats,
                    'medication_admin_continuous': lambda: collector.collect_medication_stats('continuous'),
                    'medication_admin_intermittent': lambda: collector.collect_medication_stats('intermittent'),
                    'respiratory_support': collector.collect_respiratory_support,
                    'microbiology_culture': lambda: collector.collect_microbiology('culture'),
                    'microbiology_nonculture': lambda: collector.collect_microbiology('nonculture'),
                    'microbiology_susceptibility': collector.collect_microbiology_susceptibility,
                    'patient_assessments': collector.collect_patient_assessments,
                    # 'patient_procedures': collector.collect_patient_procedures,  # Not collecting MCIDE for this
                    'position': collector.collect_position,
                    'crrt_therapy': collector.collect_crrt_stats,
                    'ecmo_mcs': collector.collect_ecmo_stats,
                    # 'hospital_diagnosis': collector.collect_hospital_diagnosis,  # Not collecting MCIDE for this
                    'code_status': collector.collect_code_status
                }

                # Collect MCIDE for successfully validated tables
                self.log(f"  Tables to process: {results['tables_analyzed']}", force=True)
                for table_name in results['tables_analyzed']:
                    if table_name in collection_methods:
                        try:
                            self.log(f"  Collecting MCIDE for {table_name}...", force=True)
                            collection_methods[table_name]()
                            mcide_success += 1
                            self.log(f"    ✓ Successfully collected MCIDE for {table_name}", force=True)
                        except Exception as e:
                            mcide_failed.append(table_name)
                            self.log(self.formatter.warning(f"    ✗ Could not collect MCIDE for {table_name}: {e}"))
                            if self.verbose:
                                import traceback
                                self.log(traceback.format_exc())
                    else:
                        self.log(f"    [WARNING]No MCIDE collection method for {table_name}")

                # Report MCIDE results
                if mcide_success > 0:
                    self.log(self.formatter.success(f"[SUCCESS] MCIDE statistics collected for {mcide_success} table(s)"))
                    self.log(f"  [LIST] MCIDE files: {os.path.join(self.output_dir, 'final', 'tableone', 'mcide')}")
                    self.log(f"  [STATS] Stats files: {os.path.join(self.output_dir, 'final', 'tableone', 'summary_stats')}")

                if mcide_failed:
                    self.log(self.formatter.warning(f"[WARNING] MCIDE collection failed for: {', '.join(mcide_failed)}"))

            except ImportError as e:
                self.log(self.formatter.warning(f"[WARNING] MCIDE collection module not found: {e}"))
                self.log("    Ensure the MCIDE module is properly installed in modules/mcide/")
                if self.verbose:
                    import traceback
                    traceback.print_exc()
            except Exception as e:
                self.log(self.formatter.warning(f"[WARNING] Error during MCIDE collection: {e}"))
                if self.verbose:
                    import traceback
                    traceback.print_exc()

        # Summary
        self.log("\n" + self.formatter.header("[STATS] ANALYSIS SUMMARY"), force=True)
        self.log(f"[SUCCESS] Successfully analyzed: {results['total_success']} table(s)", force=True)
        if results['total_failed'] > 0:
            self.log(f"[ERROR] Failed: {results['total_failed']} table(s)", force=True)
            for table in results['tables_failed']:
                error = results['details'][table].get('error', 'Unknown error')
                self.log(f"  - {table}: {error}", force=True)

        self.log(f"\n{self.formatter.FOLDER} Results saved to:", force=True)
        self.log(f"  [REPORT] Reports: {os.path.join(self.output_dir, 'final', 'reports')}", force=True)
        self.log(f"  [STATS] Results: {os.path.join(self.output_dir, 'final', 'results')}", force=True)

        # Generate combined report if multiple tables were analyzed and validation was run
        if len(results['tables_analyzed']) > 1 and run_validation and self.generate_pdf:
            self.log(f"\n{self.formatter.section('Generating Combined Validation Report')}")
            try:
                from modules.reports.combined_report_generator import generate_combined_report

                # Include all requested tables (both analyzed and failed)
                all_tables = tables

                pdf_path = generate_combined_report(
                    self.output_dir,
                    all_tables,
                    self.site_name,
                    self.timezone
                )

                if pdf_path:
                    self.log(self.formatter.success(f"Combined validation report saved: combined_validation_report.pdf"), force=True)
                    self.log(self.formatter.success(f"Consolidated CSV saved: consolidated_validation.csv"), force=True)
                else:
                    self.log(self.formatter.warning("Could not generate combined validation report"), force=True)

            except Exception as e:
                self.log(self.formatter.warning(f"Could not generate combined report: {e}"), force=True)
                if self.verbose:
                    import traceback
                    traceback.print_exc()

        return results
