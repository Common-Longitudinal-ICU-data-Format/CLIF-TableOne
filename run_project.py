#!/usr/bin/env python3
"""
CLIF Project Runner - Complete Workflow Automation

This script orchestrates the complete CLIF analysis workflow:
1. Validation of CLIF tables (with optional sampling)
2. Table One generation with memory optimization
3. Automatic Streamlit app launch (on successful completion)

Usage:
    python run_project.py                          # Full validation + table one + app
    python run_project.py --sample                 # Use 1k ICU sample + app
    python run_project.py --validate-only          # Only run validation
    python run_project.py --tableone-only          # Only run table one + app
    python run_project.py --tables patient adt     # Validate specific tables
    python run_project.py --no-launch-app          # Skip automatic app launch
"""

import os
import sys
import argparse
import subprocess
import json
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime


class ProjectRunner:
    """Orchestrates the complete CLIF analysis workflow."""

    def __init__(self, config_path='config/config.json'):
        self.config_path = config_path
        self.config = self.load_config()
        self.start_time = datetime.now()
        self.log_file = self.setup_logging()
        self.logger = logging.getLogger('workflow')
        self.results = {
            'validation': None,
            'tableone': None,
            'get_ecdf': None,
            'overall_success': False
        }

    def setup_logging(self):
        """Setup comprehensive logging to capture all workflow output."""
        # Create logs directory
        log_dir = Path('output/final/logs')
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'workflow_execution_{timestamp}.log'

        # Also create a 'latest' symlink/copy
        latest_log = log_dir / 'workflow_execution_latest.log'

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.FileHandler(latest_log, mode='w'),  # Overwrite latest
                logging.StreamHandler(sys.stdout)  # Also print to console
            ]
        )

        return log_file

    def load_config(self):
        """Load project configuration."""
        if not os.path.exists(self.config_path):
            print(f"❌ Configuration file not found: {self.config_path}")
            sys.exit(1)

        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
            sys.exit(1)

    def print_header(self, title):
        """Print formatted section header."""
        print(f"\n{'='*80}")
        print(f"{title}")
        print(f"{'='*80}\n")

    def check_critical_tables(self):
        """
        Check if critical tables have at least 'partial' status.

        Returns
        -------
        tuple
            (success: bool, message: str)
            success is True if all critical tables are at least partial
            message contains details about any failures
        """
        # Define critical tables required for Table One generation
        critical_tables = [
            'Patient',
            'Hospitalization',
            'ADT',
            'Labs',
            'Vitals',
            'Medication Admin Continuous',
            'Patient Assessments',
            'Hospital Diagnosis',
            'Respiratory Support'
        ]

        # Path to consolidated validation results
        consolidated_csv = Path('output/final/results/consolidated_validation.csv')

        if not consolidated_csv.exists():
            return False, "Validation results file not found"

        try:
            # Read validation results
            df = pd.read_csv(consolidated_csv)

            # Get unique status per table (first row for each table)
            table_statuses = df.groupby('table_name')['status'].first().to_dict()

            # Check each critical table
            failed_tables = []
            missing_tables = []

            for table in critical_tables:
                if table not in table_statuses:
                    missing_tables.append(table)
                else:
                    status = table_statuses[table].lower()
                    # Acceptable statuses: complete, partial, or "partial → complete"
                    if 'complete' in status or 'partial' in status:
                        continue
                    else:
                        # Unacceptable: missing, incomplete
                        failed_tables.append(f"{table} ({status})")

            # Build error message if any tables failed
            if failed_tables or missing_tables:
                error_parts = []
                if failed_tables:
                    error_parts.append(f"Failed tables: {', '.join(failed_tables)}")
                if missing_tables:
                    error_parts.append(f"Missing tables: {', '.join(missing_tables)}")

                message = (
                    f"❌ Critical tables do not meet minimum requirements:\n\n"
                    f"   {'; '.join(error_parts)}\n\n"
                    f"   Table One generation requires these tables to have at least 'partial' status.\n\n"
                    f"📄 Review the validation report for details:\n"
                    f"   output/final/reports/combined_validation_report.pdf\n\n"
                    f"   Use --continue-on-error to proceed anyway (not recommended)."
                )
                return False, message

            # All critical tables passed
            return True, "All critical tables have at least 'partial' status"

        except Exception as e:
            return False, f"Error checking table statuses: {e}"

    def run_validation(self, tables=None, use_sample=False, verbose=False, no_summary=False):
        """
        Run CLIF validation using run_analysis.py.

        Parameters
        ----------
        tables : list of str, optional
            Specific tables to validate. If None, validates all.
        use_sample : bool
            Use 1k ICU sample for faster analysis
        verbose : bool
            Enable verbose output
        no_summary : bool
            Skip summary statistics generation

        Returns
        -------
        bool
            True if validation succeeded
        """
        self.print_header("STEP 1: CLIF VALIDATION")

        # Build command - use sys.executable to ensure correct Python interpreter
        cmd = [sys.executable, 'run_analysis.py']

        # Add table flags
        if tables:
            for table in tables:
                cmd.append(f'--{table}')
        else:
            cmd.append('--all')

        # Add validation flag
        cmd.append('--validate')
        if not no_summary:
            cmd.append('--summary')

        # Add optional flags
        if use_sample:
            cmd.append('--sample')
        if verbose:
            cmd.append('--verbose')

        self.logger.info(f"Running: {' '.join(cmd)}")
        self.logger.info(f"Sample mode: {'✓' if use_sample else '✗'}")
        self.logger.info(f"Tables: {', '.join(tables) if tables else 'all'}")

        try:
            # Stream output in real-time while also logging
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )

            # Stream stdout in real-time
            for line in process.stdout:
                line = line.rstrip()
                if line:  # Skip empty lines
                    print(line)  # Show in terminal immediately
                    self.logger.info(line)  # Also log to file

            # Get any remaining stderr
            stderr_output = process.stderr.read()
            if stderr_output:
                for line in stderr_output.strip().split('\n'):
                    if line:
                        print(line, file=sys.stderr)
                        self.logger.error(line)

            # Wait for process to complete
            exit_code = process.wait()

            if exit_code == 0:
                self.logger.info("✅ Validation completed successfully")
                critical_tables_ok = True
                critical_tables_msg = None
            elif exit_code == 2:
                self.logger.warning("⚠️  Validation completed with exit code 2 (partial success)")
                self.logger.info("Checking critical tables...")

                # Check if critical tables meet minimum requirements
                critical_tables_ok, critical_tables_msg = self.check_critical_tables()

                if critical_tables_ok:
                    self.logger.info(f"✅ {critical_tables_msg}")
                    self.logger.info("Proceeding to Table One generation...")
                else:
                    self.logger.error(critical_tables_msg)
            else:
                self.logger.error(f"❌ Validation failed with exit code {exit_code}")
                critical_tables_ok = False
                critical_tables_msg = "Validation failed"

            self.results['validation'] = {
                'success': exit_code == 0,
                'exit_code': exit_code,
                'critical_tables_ok': critical_tables_ok,
                'critical_tables_msg': critical_tables_msg
            }

            return exit_code == 0, critical_tables_ok

        except Exception as e:
            self.logger.exception(f"❌ Validation failed: {e}")
            self.results['validation'] = {
                'success': False,
                'error': str(e),
                'critical_tables_ok': False
            }
            return False, False

    def run_tableone(self):
        """
        Run Table One generation using the new modular structure.

        Returns
        -------
        bool
            True if table one generation succeeded
        """
        self.print_header("STEP 2: TABLE ONE GENERATION")

        # Import and use the new TableOneRunner module
        from modules.tableone.runner import TableOneRunner

        self.logger.info("Running Table One generation with memory monitoring...")

        try:
            # Use the new TableOneRunner directly
            runner = TableOneRunner(self.config)
            success = runner.run(profile_mode=False)

            if success:
                self.logger.info("✅ Table One generation completed successfully")
            else:
                self.logger.warning("⚠️  Table One generation completed with warnings")

            self.results['tableone'] = {
                'success': success
            }

            return success

        except Exception as e:
            self.logger.exception(f"❌ Table One generation failed: {e}")
            self.results['tableone'] = {
                'success': False,
                'error': str(e)
            }
            return False

    def run_get_ecdf(self, visualize=False):
        """
        Run ECDF generation using the new modular structure.

        Parameters
        ----------
        visualize : bool
            Whether to generate HTML visualizations after ECDF generation

        Returns
        -------
        bool
            True if ECDF generation succeeded
        """
        self.print_header("STEP 3: GET ECDF BINS")

        # Import and use the new ECDFRunner module
        from modules.ecdf.runner import ECDFRunner

        self.logger.info("Running ECDF generation...")
        self.logger.info(f"Visualize: {'✓' if visualize else '✗'}")

        try:
            # Use the new ECDFRunner directly
            runner = ECDFRunner(self.config)
            success = runner.run(visualize=visualize)

            if success:
                self.logger.info("✅ ECDF generation completed successfully")
            else:
                self.logger.warning("⚠️  ECDF generation completed with warnings")

            self.results['get_ecdf'] = {
                'success': success,
                'visualize': visualize
            }

            return success

        except Exception as e:
            self.logger.exception(f"❌ ECDF generation failed: {e}")
            self.results['get_ecdf'] = {
                'success': False,
                'error': str(e)
            }
            return False

    def generate_summary_report(self):
        """Generate final summary report."""
        self.print_header("WORKFLOW SUMMARY")

        elapsed_time = (datetime.now() - self.start_time).total_seconds()

        print(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"End Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Duration:   {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)\n")

        # Validation results
        if self.results['validation']:
            val_status = "✅ SUCCESS" if self.results['validation']['success'] else "❌ FAILED"
            print(f"Validation:        {val_status}")

            # Show critical tables status if relevant
            if 'critical_tables_ok' in self.results['validation']:
                crit_status = "✅ PASS" if self.results['validation']['critical_tables_ok'] else "❌ FAIL"
                print(f"Critical Tables:   {crit_status}")

        # Table One results
        if self.results['tableone']:
            tbl_status = "✅ SUCCESS" if self.results['tableone']['success'] else "❌ FAILED"
            print(f"Table One:         {tbl_status}")

        # Get ECDF results
        if self.results['get_ecdf']:
            ecdf_status = "✅ SUCCESS" if self.results['get_ecdf']['success'] else "❌ FAILED"
            print(f"Get ECDF:          {ecdf_status}")

        # Overall success now only depends on validation critical tables
        # App should launch unless critical tables validation fails
        if self.results['validation']:
            # Success if critical tables passed (even if overall validation had some failures)
            self.results['overall_success'] = self.results['validation'].get('critical_tables_ok', False)
        else:
            # If validation wasn't run, consider it successful (for --tableone-only mode)
            self.results['overall_success'] = True

        print(f"\nOverall Status:    {'✅ SUCCESS' if self.results['overall_success'] else '❌ FAILED'}")

        # Output locations
        print(f"\n📂 Output Locations:")
        print(f"   📋 Workflow Log:    {self.log_file}")
        print(f"   📋 Latest Log:      output/final/logs/workflow_execution_latest.log")
        if self.results['validation']:
            print(f"   Validation Reports: output/final/reports/")
            print(f"   Combined Report:    output/final/reports/combined_validation_report.pdf")
            print(f"   Validation Results: output/final/results/")
        if self.results['tableone']:
            print(f"   Table One:          output/final/tableone/")
        if self.results['get_ecdf']:
            print(f"   ECDF Data:          output/final/ecdf/, output/final/bins/")

        self.logger.info("="*80)
        self.logger.info(f"Workflow execution log saved to: {self.log_file}")
        self.logger.info("="*80)

        return self.results['overall_success']

    def launch_app(self):
        """Launch the Streamlit web application."""
        self.print_header("LAUNCHING STREAMLIT APP")

        print("Starting Streamlit web application...")
        print("The app will open in your default browser at http://localhost:8501")
        print("\nPress Ctrl+C to stop the app\n")

        cmd = ['streamlit', 'run', 'app.py']

        try:
            # Run streamlit in foreground so user can interact
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            print("\n\n✅ Streamlit app stopped")
        except FileNotFoundError:
            print("\n❌ Error: streamlit not found. Install with: uv add streamlit")
        except Exception as e:
            print(f"\n❌ Error launching app: {e}")

    def run(self, validate=True, tableone=True, get_ecdf=False, **kwargs):
        """
        Run the complete workflow.

        Parameters
        ----------
        validate : bool
            Run validation step
        tableone : bool
            Run table one generation step
        get_ecdf : bool
            Run get ECDF bins step
        **kwargs : dict
            Additional arguments for validation (tables, use_sample, verbose, visualize)

        Returns
        -------
        bool
            True if all steps succeeded
        """
        self.print_header("🏥 CLIF PROJECT RUNNER")

        self.logger.info("="*80)
        self.logger.info("🏥 CLIF PROJECT RUNNER - WORKFLOW STARTING")
        self.logger.info("="*80)
        self.logger.info(f"Configuration: {self.config_path}")
        self.logger.info(f"Data Directory: {self.config.get('tables_path', 'NOT SET')}")
        self.logger.info(f"Site Name: {self.config.get('site_name', 'NOT SET')}")
        self.logger.info(f"Workflow Steps:")
        self.logger.info(f"  1. Validation: {'✓' if validate else '✗'}")
        self.logger.info(f"  2. Table One:  {'✓' if tableone else '✗'}")
        self.logger.info(f"  3. Get ECDF:   {'✓' if get_ecdf else '✗'}")
        self.logger.info("="*80)

        # Step 1: Validation
        if validate:
            val_success, critical_tables_ok = self.run_validation(
                tables=kwargs.get('tables'),
                use_sample=kwargs.get('use_sample', False),
                verbose=kwargs.get('verbose', False),
                no_summary=kwargs.get('no_summary', False)
            )

            # Decide whether to proceed based on validation results
            can_proceed = val_success or critical_tables_ok

            if not can_proceed and not kwargs.get('continue_on_error', False):
                if not critical_tables_ok:
                    # Critical tables failed - show specific error
                    pass  # Error message already logged by run_validation
                else:
                    # General validation failure
                    self.logger.warning("⚠️  Validation failed. Stopping workflow.")
                    self.logger.info("Use --continue-on-error to proceed anyway.")

                self.generate_summary_report()
                return False

        # Step 2: Table One Generation
        if tableone:
            tbl_success = self.run_tableone()

        # Step 3: Get ECDF Bins
        if get_ecdf:
            ecdf_success = self.run_get_ecdf(
                visualize=kwargs.get('visualize', False)
            )

        # Generate summary
        overall_success = self.generate_summary_report()

        # App launch logic - launch unless critical tables failed
        # Allow override with --continue-on-error
        should_launch = overall_success or kwargs.get('continue_on_error', False)

        if should_launch:
            # Check if user wants to skip app launch
            if not kwargs.get('no_launch_app', False):
                print("\n" + "="*80)
                print("🚀 Launching Streamlit App...")
                print("="*80)

                # Show warnings for any failed steps
                warnings_shown = False

                if self.results['tableone'] and not self.results['tableone'].get('success', False):
                    print("\n⚠️  Warning: Table One generation failed")
                    print("   The app will launch but Table One data will not be available")
                    warnings_shown = True

                if self.results['get_ecdf'] and not self.results['get_ecdf'].get('success', False):
                    print("\n⚠️  Warning: ECDF generation failed")
                    print("   The app will launch but ECDF data will not be available")
                    warnings_shown = True

                if not overall_success and kwargs.get('continue_on_error', False):
                    print("\n⚠️  Warning: Critical tables validation failed")
                    print("   Launching app anyway due to --continue-on-error flag")
                    warnings_shown = True

                if not warnings_shown:
                    print("\nWorkflow completed successfully!")

                print("\nStarting the interactive web application in 3 seconds...")
                print("(Press Ctrl+C now to skip)\n")

                try:
                    import time
                    time.sleep(3)
                    self.launch_app()
                except KeyboardInterrupt:
                    print("\n\n⏭️  App launch skipped by user")
                    print("   You can launch it manually with: uv run streamlit run app.py\n")
        else:
            # Only happens if critical tables validation failed and no override
            print("\n" + "="*80)
            print("❌ App Launch Blocked")
            print("="*80)
            print("\nCritical tables validation failed. Cannot launch app.")
            print("Review validation report: output/final/reports/combined_validation_report.pdf")
            print("\nUse --continue-on-error flag to bypass this check (not recommended)\n")

        return overall_success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='CLIF Project Runner - Complete Workflow Automation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Full workflow (validation + tableone)
  %(prog)s --sample                           # Use 1k ICU sample
  %(prog)s --validate-only                    # Only validation
  %(prog)s --tableone-only                    # Only table one
  %(prog)s --get-ecdf-only                    # Only get ECDF bins
  %(prog)s --get-ecdf-only --visualize        # Get ECDF + HTML visualizations
  %(prog)s --get-ecdf                         # Full workflow + get ECDF
  %(prog)s --tables patient adt               # Validate specific tables
  %(prog)s --sample --no-launch-app           # Skip automatic app launch
        """
    )

    # Workflow control
    workflow_group = parser.add_argument_group('Workflow Control')
    workflow_group.add_argument('--validate-only', action='store_true',
                                help='Only run validation step')
    workflow_group.add_argument('--tableone-only', action='store_true',
                                help='Only run table one generation step')
    workflow_group.add_argument('--get-ecdf-only', action='store_true',
                                help='Only run get ECDF bins step')
    workflow_group.add_argument('--get-ecdf', action='store_true',
                                help='Include get ECDF bins in workflow')
    workflow_group.add_argument('--visualize', action='store_true',
                                help='Generate HTML visualizations (for get ECDF)')
    workflow_group.add_argument('--continue-on-error', action='store_true',
                                help='Continue to next step even if previous step fails')
    workflow_group.add_argument('--no-launch-app', action='store_true',
                                help='Skip automatic Streamlit app launch after completion')

    # Validation options
    validation_group = parser.add_argument_group('Validation Options')
    validation_group.add_argument('--tables', nargs='+',
                                  choices=['patient', 'hospitalization', 'adt', 'code_status',
                                          'crrt_therapy', 'ecmo_mcs', 'hospital_diagnosis', 'labs',
                                          'medication_admin_continuous', 'medication_admin_intermittent',
                                          'microbiology_culture', 'microbiology_nonculture',
                                          'microbiology_susceptibility', 'patient_assessments',
                                          'patient_procedures', 'position', 'respiratory_support', 'vitals'],
                                  help='Specific tables to validate')
    validation_group.add_argument('--sample', action='store_true',
                                  help='Use 1k ICU sample for faster analysis')
    validation_group.add_argument('--no-summary', action='store_true',
                                  help='Skip summary statistics generation (only run validation)')
    validation_group.add_argument('--verbose', '-v', action='store_true',
                                  help='Enable verbose output')

    # Configuration
    config_group = parser.add_argument_group('Configuration')
    config_group.add_argument('--config', default='config/config.json',
                             help='Path to configuration file')

    args = parser.parse_args()

    # Determine which steps to run
    if args.validate_only:
        validate = True
        tableone = False
        get_ecdf = False
    elif args.tableone_only:
        validate = False
        tableone = True
        get_ecdf = False
    elif args.get_ecdf_only:
        validate = False
        tableone = False
        get_ecdf = True
    else:
        # Default: run validation and tableone
        validate = True
        tableone = True
        # Get ECDF only if explicitly requested
        get_ecdf = args.get_ecdf

    # Initialize runner
    runner = ProjectRunner(config_path=args.config)

    # Run workflow
    try:
        success = runner.run(
            validate=validate,
            tableone=tableone,
            get_ecdf=get_ecdf,
            tables=args.tables,
            use_sample=args.sample,
            no_summary=args.no_summary,
            verbose=args.verbose,
            visualize=args.visualize,
            continue_on_error=args.continue_on_error,
            no_launch_app=args.no_launch_app
        )

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Workflow interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
