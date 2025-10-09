"""CLI Analysis Runner for CLIF tables."""

import os
import json
from typing import Dict, Any, List, Optional
from modules.tables import PatientAnalyzer, HospitalizationAnalyzer
from .formatters import ConsoleFormatter
from .pdf_generator import ValidationPDFGenerator


class CLIAnalysisRunner:
    """Orchestrates CLI-based analysis of CLIF tables."""

    TABLE_ANALYZERS = {
        'patient': PatientAnalyzer,
        'hospitalization': HospitalizationAnalyzer
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

        # Extract config values
        self.data_dir = config.get('tables_path', './data')
        self.filetype = config.get('filetype', 'parquet')
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
                        final_dir = os.path.join(self.output_dir, 'final')
                        os.makedirs(final_dir, exist_ok=True)

                        pdf_path = os.path.join(final_dir, f"{table_name}_validation_report.pdf")

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
                            txt_path = os.path.join(final_dir, f"{table_name}_validation_report.txt")
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
                    final_dir = os.path.join(self.output_dir, 'final')
                    os.makedirs(final_dir, exist_ok=True)

                    # Save patient demographics summary
                    if hasattr(analyzer, 'generate_patient_summary'):
                        patient_summary_df = analyzer.generate_patient_summary()
                        if not patient_summary_df.empty:
                            csv_filepath = os.path.join(final_dir, f"{table_name}_demographics_summary.csv")
                            patient_summary_df.to_csv(csv_filepath, index=False)
                            self.log(self.formatter.success(f"Patient demographics summary CSV saved"))

                    # Save hospitalization summary
                    if hasattr(analyzer, 'generate_hospitalization_summary'):
                        hosp_summary_df = analyzer.generate_hospitalization_summary()
                        if not hosp_summary_df.empty:
                            csv_filepath = os.path.join(final_dir, f"{table_name}_summary.csv")
                            hosp_summary_df.to_csv(csv_filepath, index=False)
                            self.log(self.formatter.success(f"Hospitalization summary CSV saved"))
                except Exception as e:
                    self.log(self.formatter.warning(f"Could not save summary CSV files: {e}"))

                # Display summary info
                if self.verbose:
                    self.log(self.formatter.format_summary_info(summary_stats, table_name))

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
        self.log(self.formatter.header("🏥 CLIF TABLE ONE ANALYSIS"), force=True)
        self.log(f"{self.formatter.FOLDER} Data Directory: {self.data_dir}", force=True)
        self.log(f"{self.formatter.FILE} Output Directory: {os.path.join(self.output_dir, 'final')}", force=True)
        self.log(f"📋 Tables: {', '.join(tables)}", force=True)
        self.log(f"🔍 Validation: {'✓' if run_validation else '✗'}", force=True)
        self.log(f"📊 Summary: {'✓' if run_summary else '✗'}", force=True)
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

        # Summary
        self.log("\n" + self.formatter.header("📊 ANALYSIS SUMMARY"), force=True)
        self.log(f"✅ Successfully analyzed: {results['total_success']} table(s)", force=True)
        if results['total_failed'] > 0:
            self.log(f"❌ Failed: {results['total_failed']} table(s)", force=True)
            for table in results['tables_failed']:
                error = results['details'][table].get('error', 'Unknown error')
                self.log(f"  - {table}: {error}", force=True)

        self.log(f"\n{self.formatter.FOLDER} Results saved to: {os.path.join(self.output_dir, 'final')}", force=True)

        return results
