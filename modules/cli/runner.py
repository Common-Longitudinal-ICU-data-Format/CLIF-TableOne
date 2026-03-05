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
        'hospital_diagnosis': HospitalDiagnosisAnalyzer,
        'labs': LabsAnalyzer,
        'medication_admin_continuous': MedicationAdminContinuousAnalyzer,
        'medication_admin_intermittent': MedicationAdminIntermittentAnalyzer,
        'microbiology_culture': MicrobiologyCultureAnalyzer,
        'microbiology_susceptibility': MicrobiologySusceptibilityAnalyzer,
        'patient_assessments': PatientAssessmentsAnalyzer,
        'patient_procedures': PatientProceduresAnalyzer,
        'position': PositionAnalyzer,
        'respiratory_support': RespiratorySupportAnalyzer,
        'vitals': VitalsAnalyzer
    }

    def __init__(self, config: Dict[str, Any], verbose: bool = False, quiet: bool = False,
                 generate_pdf: bool = True):
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
        """
        self.config = config
        self.verbose = verbose
        self.quiet = quiet
        self.generate_pdf = generate_pdf
        self.formatter = ConsoleFormatter()
        self.pdf_generator = ValidationPDFGenerator()
        self._loaded_tables = {}  # table_name -> BaseTable object (temporary, released after single-table checks)
        self._cross_table_caches = {}  # table_name -> lightweight cache dict
        self._hosp_years = None  # cached hosp_years for P.6 temporal context

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

            # Load table
            self.log(self.formatter.progress(f"Loading {table_name} table"))
            analyzer = analyzer_class(self.data_dir, self.filetype, self.timezone, self.output_dir)

            if analyzer.table is None:
                result['error'] = f"Failed to load {table_name} table"
                self.log(self.formatter.error(f"Could not load {table_name} table"))
                return result

            self.log(self.formatter.success(f"Loaded {table_name} table"))
            self._loaded_tables[table_name] = analyzer.table

            # Run validation (single-table only; cross-table checks run in post-processing)
            if run_validation:
                self.log(self.formatter.progress(f"Running validation for {table_name}"))
                validation_results = analyzer.validate(tables=None, hosp_years=self._hosp_years)

                # Inject per-column data profile stats
                if analyzer.table is not None and hasattr(analyzer.table, 'df') and analyzer.table.df is not None:
                    from clifpy.utils.report_generator import compute_table_stats
                    validation_results['total_rows'] = len(analyzer.table.df)
                    validation_results['table_stats'] = compute_table_stats(
                        analyzer.table.df, analyzer.table.schema
                    )

                result['validation'] = validation_results

                # Save validation results as JSON
                try:
                    analyzer.save_validation_results(validation_results)
                    self.log(self.formatter.success(f"Validation results saved"))
                except Exception as e:
                    self.log(self.formatter.warning(f"Could not save validation results: {e}"))

                # Save monthly trend CSVs from P.6 temporal consistency
                try:
                    trends_dir = analyzer.save_monthly_trend_csvs(validation_results)
                    if trends_dir:
                        self.log(self.formatter.success(f"Monthly trends saved to {trends_dir}"))
                except Exception as e:
                    self.log(self.formatter.warning(f"Could not save monthly trends: {e}"))

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

                # Display validation summary (condensed always, full in verbose)
                self.log(self.formatter.format_condensed_validation(validation_results, table_name))
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

            # Extract lightweight cache for cross-table checks, then release the full DataFrame
            if run_validation and analyzer.table is not None:
                try:
                    import gc as _gc
                    cache = analyzer.extract_cross_table_cache()
                    self._cross_table_caches[table_name] = cache
                    if cache.get('hosp_years'):
                        self._hosp_years = cache['hosp_years']
                    # Release full DataFrame to free memory
                    del self._loaded_tables[table_name]
                    _gc.collect()
                except Exception as cache_e:
                    self.log(self.formatter.warning(f"Could not extract cache for {table_name}: {cache_e}"))
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
        # Header
        self.log(self.formatter.header("[HOSPITAL] CLIF TABLE ONE ANALYSIS"), force=True)
        self.log(f"{self.formatter.FOLDER} Data Directory: {self.data_dir}", force=True)
        self.log(f"{self.formatter.FILE} Output Directory: {os.path.join(self.output_dir, 'final', 'reports')} (reports), {os.path.join(self.output_dir, 'final', 'results')} (results)", force=True)
        self.log(f"[LIST] Tables: {', '.join(tables)}", force=True)
        self.log(f"[SEARCH] Validation: {'[OK]' if run_validation else '[X]'}", force=True)
        self.log(f"[STATS] Summary: {'[OK]' if run_summary else '[X]'}", force=True)
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

        # Run cross-table checks from caches (relational + plausibility) — ONCE
        if run_validation and len(self._cross_table_caches) > 1:
            self.log(f"\n{self.formatter.section('Cross-Table Checks (from cache)')}")
            try:
                from clifpy.utils.validator import (
                    run_relational_integrity_checks_from_cache,
                    run_cross_table_plausibility_checks_from_cache,
                )

                # --- Relational integrity (cache-based) ---
                self.log(self.formatter.progress(
                    f"Running relational integrity checks across {len(self._cross_table_caches)} tables (from cache)"
                ))
                rel_results = run_relational_integrity_checks_from_cache(self._cross_table_caches)

                for tname, rel_checks in rel_results.items():
                    if not rel_checks:
                        continue
                    serialized_rel = {k: v.to_dict() for k, v in rel_checks.items()}

                    # Update in-memory validation results (merge into completeness)
                    if tname in results['details'] and results['details'][tname].get('validation'):
                        vr = results['details'][tname]['validation']
                        vr.setdefault('completeness', {}).update(serialized_rel)

                    # Update saved JSON file (merge into completeness)
                    json_path = os.path.join(self.output_dir, 'final', 'clifpy', f'{tname}_dqa.json')
                    if os.path.exists(json_path):
                        with open(json_path, 'r') as f:
                            saved = json.load(f)
                        saved.setdefault('completeness', {}).update(serialized_rel)
                        with open(json_path, 'w') as f:
                            json.dump(saved, f, indent=2, default=str)

                rel_count = sum(len(v) for v in rel_results.values())
                self.log(self.formatter.success(
                    f"Relational checks complete: {rel_count} checks across {len(rel_results)} tables"
                ))

                # --- Cross-table plausibility (cache-based) ---
                self.log(self.formatter.progress(
                    f"Running cross-table plausibility checks across {len(self._cross_table_caches)} tables (from cache)"
                ))
                plaus_results = run_cross_table_plausibility_checks_from_cache(self._cross_table_caches)

                for tname, plaus_checks in plaus_results.items():
                    if not plaus_checks:
                        continue
                    serialized_plaus = {k: v.to_dict() for k, v in plaus_checks.items()}

                    # Update in-memory validation results (merge into plausibility)
                    if tname in results['details'] and results['details'][tname].get('validation'):
                        vr = results['details'][tname]['validation']
                        vr.setdefault('plausibility', {}).update(serialized_plaus)

                    # Update saved JSON file (merge into plausibility)
                    json_path = os.path.join(self.output_dir, 'final', 'clifpy', f'{tname}_dqa.json')
                    if os.path.exists(json_path):
                        with open(json_path, 'r') as f:
                            saved = json.load(f)
                        saved.setdefault('plausibility', {}).update(serialized_plaus)
                        with open(json_path, 'w') as f:
                            json.dump(saved, f, indent=2, default=str)

                plaus_count = sum(len(v) for v in plaus_results.values())
                self.log(self.formatter.success(
                    f"Cross-table plausibility checks complete: {plaus_count} checks across {len(plaus_results)} tables"
                ))

                # Regenerate PDFs for all tables affected by cross-table results
                affected_tables = set(rel_results.keys()) | set(plaus_results.keys())
                if self.generate_pdf and affected_tables:
                    reports_dir = os.path.join(self.output_dir, 'final', 'reports')
                    for tname in affected_tables:
                        if tname in results['details'] and results['details'][tname].get('validation'):
                            pdf_path = os.path.join(reports_dir, f"{tname}_validation_report.pdf")
                            try:
                                if self.pdf_generator.is_available():
                                    self.pdf_generator.generate_validation_pdf(
                                        results['details'][tname]['validation'],
                                        tname, pdf_path, self.site_name
                                    )
                            except Exception as pdf_e:
                                self.log(self.formatter.warning(
                                    f"Could not regenerate PDF for {tname}: {pdf_e}"))

                # Clear caches to free memory
                self._cross_table_caches.clear()
                import gc as _gc
                _gc.collect()

            except Exception as e:
                self.log(self.formatter.warning(f"Could not run cross-table checks: {e}"))
                if self.verbose:
                    import traceback
                    traceback.print_exc()

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
