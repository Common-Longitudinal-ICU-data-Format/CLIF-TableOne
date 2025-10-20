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
from pathlib import Path
from datetime import datetime


class ProjectRunner:
    """Orchestrates the complete CLIF analysis workflow."""

    def __init__(self, config_path='config/config.json'):
        self.config_path = config_path
        self.config = self.load_config()
        self.start_time = datetime.now()
        self.results = {
            'validation': None,
            'tableone': None,
            'get_ecdf': None,
            'overall_success': False
        }

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
            'Position',
            'Patient Assessments',
            'Hospital Diagnosis',
            'CRRT Therapy',
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

    def run_validation(self, tables=None, use_sample=False, verbose=False):
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

        Returns
        -------
        bool
            True if validation succeeded
        """
        self.print_header("STEP 1: CLIF VALIDATION")

        # Build command
        cmd = ['python', 'run_analysis.py']

        # Add table flags
        if tables:
            for table in tables:
                cmd.append(f'--{table}')
        else:
            cmd.append('--all')

        # Add validation flag
        cmd.append('--validate')
        cmd.append('--summary')

        # Add optional flags
        if use_sample:
            cmd.append('--sample')
        if verbose:
            cmd.append('--verbose')

        print(f"Running: {' '.join(cmd)}")
        print(f"Sample mode: {'✓' if use_sample else '✗'}")
        print(f"Tables: {', '.join(tables) if tables else 'all'}\n")

        try:
            result = subprocess.run(cmd, check=False)
            exit_code = result.returncode

            if exit_code == 0:
                print(f"\n✅ Validation completed successfully")
                critical_tables_ok = True
                critical_tables_msg = None
            elif exit_code == 2:
                print(f"\n⚠️  Validation completed with exit code 2 (partial success)")
                print(f"   Checking critical tables...")

                # Check if critical tables meet minimum requirements
                critical_tables_ok, critical_tables_msg = self.check_critical_tables()

                if critical_tables_ok:
                    print(f"\n✅ {critical_tables_msg}")
                    print(f"   Proceeding to Table One generation...")
                else:
                    print(f"\n{critical_tables_msg}")
            else:
                print(f"\n❌ Validation failed with exit code {exit_code}")
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
            print(f"\n❌ Validation failed: {e}")
            self.results['validation'] = {
                'success': False,
                'error': str(e),
                'critical_tables_ok': False
            }
            return False, False

    def run_tableone(self):
        """
        Run Table One generation using code/run_table_one.py.

        Returns
        -------
        bool
            True if table one generation succeeded
        """
        self.print_header("STEP 2: TABLE ONE GENERATION")

        cmd = ['python', 'code/run_table_one.py']

        print(f"Running: {' '.join(cmd)}\n")

        try:
            result = subprocess.run(cmd, check=False, cwd=os.getcwd())
            success = result.returncode == 0

            if success:
                print(f"\n✅ Table One generation completed successfully")
            else:
                print(f"\n⚠️  Table One generation completed with exit code {result.returncode}")

            self.results['tableone'] = {
                'success': success,
                'exit_code': result.returncode
            }

            return success

        except Exception as e:
            print(f"\n❌ Table One generation failed: {e}")
            self.results['tableone'] = {
                'success': False,
                'error': str(e)
            }
            return False

    def run_get_ecdf(self, visualize=False):
        """
        Run ECDF generation using code/run_get_ecdf.py.

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

        cmd = ['python', 'code/run_get_ecdf.py']

        if visualize:
            cmd.append('--visualize')

        print(f"Running: {' '.join(cmd)}")
        print(f"Visualize: {'✓' if visualize else '✗'}\n")

        try:
            result = subprocess.run(cmd, check=False, cwd=os.getcwd())
            success = result.returncode == 0

            if success:
                print(f"\n✅ ECDF generation completed successfully")
            else:
                print(f"\n⚠️  ECDF generation completed with exit code {result.returncode}")

            self.results['get_ecdf'] = {
                'success': success,
                'exit_code': result.returncode
            }

            return success

        except Exception as e:
            print(f"\n❌ ECDF generation failed: {e}")
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

        # Overall status
        val_ok = self.results['validation'] and self.results['validation']['success']
        tbl_ok = self.results['tableone'] and self.results['tableone']['success']
        ecdf_ok = self.results['get_ecdf'] and self.results['get_ecdf']['success']

        # Overall success depends on which steps were run
        steps_run = [self.results['validation'], self.results['tableone'], self.results['get_ecdf']]
        steps_run = [s for s in steps_run if s is not None]

        if steps_run:
            # Check each step - for validation, accept if critical_tables_ok is True
            success_checks = []
            for step in steps_run:
                # Check if this is the validation step
                if step == self.results['validation'] and 'critical_tables_ok' in step:
                    # For validation, accept if either validation succeeded OR critical tables are OK
                    success_checks.append(step.get('success', False) or step.get('critical_tables_ok', False))
                else:
                    # For other steps, just check success
                    success_checks.append(step.get('success', False))

            self.results['overall_success'] = all(success_checks)
        else:
            self.results['overall_success'] = False

        print(f"\nOverall Status:    {'✅ SUCCESS' if self.results['overall_success'] else '❌ FAILED'}")

        # Output locations
        print(f"\n📂 Output Locations:")
        if self.results['validation']:
            print(f"   Validation Reports: output/final/reports/")
            print(f"   Combined Report:    output/final/reports/combined_validation_report.pdf")
            print(f"   Validation Results: output/final/results/")
        if self.results['tableone']:
            print(f"   Table One:          output/final/tableone/")
        if self.results['get_ecdf']:
            print(f"   ECDF Data:          output/final/ecdf/, output/final/bins/")
            print(f"   Execution Report:   output/final/execution_report.txt")

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

        print(f"Configuration: {self.config_path}")
        print(f"Data Directory: {self.config.get('tables_path', 'NOT SET')}")
        print(f"Site Name: {self.config.get('site_name', 'NOT SET')}\n")

        print(f"Workflow Steps:")
        print(f"  1. Validation: {'✓' if validate else '✗'}")
        print(f"  2. Table One:  {'✓' if tableone else '✗'}")
        print(f"  3. Get ECDF:   {'✓' if get_ecdf else '✗'}")

        # Step 1: Validation
        if validate:
            val_success, critical_tables_ok = self.run_validation(
                tables=kwargs.get('tables'),
                use_sample=kwargs.get('use_sample', False),
                verbose=kwargs.get('verbose', False)
            )

            # Decide whether to proceed based on validation results
            can_proceed = val_success or critical_tables_ok

            if not can_proceed and not kwargs.get('continue_on_error', False):
                if not critical_tables_ok:
                    # Critical tables failed - show specific error
                    pass  # Error message already printed by run_validation
                else:
                    # General validation failure
                    print("\n⚠️  Validation failed. Stopping workflow.")
                    print("   Use --continue-on-error to proceed anyway.")

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

        # Step 3: Automatic app launch after successful completion
        if overall_success:
            # Check if user wants to skip app launch
            if not kwargs.get('no_launch_app', False):
                print("\n" + "="*80)
                print("🚀 Launching Streamlit App...")
                print("="*80)
                print("\nWorkflow completed successfully!")
                print("Starting the interactive web application in 3 seconds...")
                print("(Press Ctrl+C now to skip)\n")

                try:
                    import time
                    time.sleep(3)
                    self.launch_app()
                except KeyboardInterrupt:
                    print("\n\n⏭️  App launch skipped by user")
                    print("   You can launch it manually with: uv run streamlit run app.py\n")

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
