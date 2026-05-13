#!/usr/bin/env python3
"""
CLIF Project Runner - Complete Workflow Automation

This script orchestrates the complete CLIF analysis workflow:
1. Validation of CLIF tables
2. Table One generation with memory optimization
3. Automatic web app launch (on successful completion)

Usage:
    python run_project.py                          # Full: validation + CI + ward + ECDF + app
    python run_project.py --no-ward --no-ecdf      # CI table one only (fastest)
    python run_project.py --validate-only          # Only run validation
    python run_project.py --tableone-only          # CI + ward table one (no validation/ECDF)
    python run_project.py --tables patient adt     # Validate specific tables
    python run_project.py --no-launch-app          # Skip automatic app launch
"""

import os
import sys
import io
import warnings

# The legacy code/ directory shadows Python's stdlib 'code' module.
# Rename it so it stops interfering with imports.
_legacy = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code')
if os.path.isdir(_legacy):
    os.rename(_legacy, _legacy.replace('code', '_archived'))
    print("  Renamed legacy code/ → _archived/ (was shadowing Python stdlib)")
del _legacy

# Force the headless matplotlib backend BEFORE any matplotlib import in this
# process or any child subprocess. Without this, macOS picks the "MacOSX"
# GUI backend on first import, which registers Python as a foreground app and
# spawns a dock icon (the rainbow pinwheel) for the duration of the run.
# Setting MPLBACKEND in os.environ also propagates to subprocess.Popen children
# (run_tableone_ward.py, run_ecdf.py, run_analysis.py) so they're headless too.
os.environ.setdefault("MPLBACKEND", "Agg")

# Silence pandas FutureWarnings (and related deprecations) before anything else
# imports pandas.  Propagated to subprocesses below via PYTHONWARNINGS.
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

# Force UTF-8 encoding for all platforms.
# On Windows this prevents 'charmap' codec errors when printing unicode
# (arrows, emoji, etc.) and propagates to subprocesses (ward, ECDF).
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("PYTHONUTF8", "1")
if sys.platform == 'win32':
    try:
        os.system('chcp 65001 >nul 2>&1')
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, TypeError):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        except (AttributeError, TypeError):
            pass
import argparse
import subprocess
import json
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime


class ProjectRunner:
    """Orchestrates the complete CLIF analysis workflow."""

    def __init__(self, config_path='config/config.json', verbose=False, force_refresh=False):
        self.config_path = config_path
        self.verbose = verbose
        self.force_refresh = force_refresh
        self.config = self.load_config()
        self.start_time = datetime.now()
        self.log_file = self.setup_logging()
        self.logger = logging.getLogger('workflow')
        self.results = {
            'validation': None,
            'tableone': None,
            'tableone_ward': None,
            'get_ecdf': None,
            'overall_success': False
        }

    def setup_logging(self):
        """Setup comprehensive logging to capture all workflow output."""
        # Create logs directory under meta/
        from modules.utils.output_paths import workflow_logs_dir
        log_dir = workflow_logs_dir()
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'workflow_execution_{timestamp}.log'

        # Also create a 'latest' symlink/copy
        latest_log = log_dir / 'workflow_execution_latest.log'

        # Build handler list.  File handlers always on.  The stdout handler is
        # only added in --verbose mode; otherwise the status-line writer wrapping
        # sys.stdout (set up in __main__) is the terminal UX, and the logger
        # stays out of its way.  File capture is unaffected either way.
        handlers = [
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.FileHandler(latest_log, mode='w', encoding='utf-8'),
        ]
        if self.verbose:
            handlers.append(logging.StreamHandler(sys.stdout))

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=handlers,
        )

        return log_file

    def load_config(self):
        """Load project configuration."""
        if not os.path.exists(self.config_path):
            print(f"[ERROR] Configuration file not found: {self.config_path}")
            sys.exit(1)

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[ERROR] Error loading configuration: {e}")
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
        from modules.utils.output_paths import validation_consolidated_dir
        consolidated_csv = validation_consolidated_dir() / 'consolidated_validation.csv'

        if not consolidated_csv.exists():
            return False, "Validation results file not found"

        try:
            # Read validation results
            df = pd.read_csv(consolidated_csv)

            # Determine table status from the new DQA format
            # Tables present in CSV with actual check results (not just "Data file not found") are at least partial
            tables_in_csv = set(df['table_name'].unique())

            # Check each critical table
            failed_tables = []
            missing_tables = []

            for table in critical_tables:
                if table not in tables_in_csv:
                    missing_tables.append(table)
                else:
                    table_rows = df[df['table_name'] == table]
                    # If the only row is "Data file not found", the table is incomplete
                    if (len(table_rows) == 1 and
                            'not found' in str(table_rows.iloc[0].get('message', '')).lower()):
                        failed_tables.append(f"{table} (data file not found)")
                    # Otherwise, the table has been validated (at least partial)
                    # Continue to next table

            # Build error message if any tables failed
            if failed_tables or missing_tables:
                error_parts = []
                if failed_tables:
                    error_parts.append(f"Failed tables: {', '.join(failed_tables)}")
                if missing_tables:
                    error_parts.append(f"Missing tables: {', '.join(missing_tables)}")

                message = (
                    f"[ERROR] Critical tables do not meet minimum requirements:\n\n"
                    f"   {'; '.join(error_parts)}\n\n"
                    f"   Table One generation requires these tables to have at least 'partial' status.\n\n"
                    f"[REPORT] Review the validation report for details:\n"
                    f"   output/final/validation/pdf_reports/combined_validation_report.pdf\n\n"
                    f"   Use --continue-on-error to proceed anyway (not recommended)."
                )
                return False, message

            # All critical tables passed
            return True, "All critical tables have at least 'partial' status"

        except Exception as e:
            return False, f"Error checking table statuses: {e}"

    def run_validation(self, tables=None, verbose=False, no_summary=False):
        """
        Run CLIF validation using run_analysis.py.

        Parameters
        ----------
        tables : list of str, optional
            Specific tables to validate. If None, validates all.
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
        if verbose:
            cmd.append('--verbose')

        self.logger.info(f"Running: {' '.join(cmd)}")
        self.logger.info(f"Tables: {', '.join(tables) if tables else 'all'}")

        try:
            # Ensure subprocess uses UTF-8 encoding (especially important on Windows)
            env = os.environ.copy()
            env['PYTHONUTF8'] = '1'
            env['PYTHONIOENCODING'] = 'utf-8'
            # Force unbuffered child stdout so per-table progress lines reach
            # ProgressDisplay in real time. Without this, pipe-destination
            # stdout is block-buffered (~8 KB) and the live status line sits
            # on "starting…" for minutes until validation flushes.
            env['PYTHONUNBUFFERED'] = '1'

            # Use different approach for Windows vs Unix-like systems
            if sys.platform == 'win32':
                # On Windows, use communicate() to avoid deadlocks
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # Merge stderr into stdout
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    env=env
                )

                # Read output in chunks to show progress
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        line = output.rstrip()
                        if line:  # Skip empty lines
                            print(line)  # Show in terminal immediately
                            self.logger.info(line)  # Also log to file

                # Get exit code
                exit_code = process.poll()
            else:
                # Original Unix/Linux/MacOS approach
                # Merge stderr into stdout so child log records (which Python's
                # logging module writes to stderr by default) are logged at the
                # parent's INFO level in chronological order, instead of being
                # drained at the end and re-logged as ERROR.
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,  # Line buffered
                    universal_newlines=True,
                    env=env
                )

                # Stream stdout in real-time
                for line in process.stdout:
                    line = line.rstrip()
                    if line:  # Skip empty lines
                        print(line)  # Show in terminal immediately
                        self.logger.info(line)  # Also log to file

                # Wait for process to complete
                exit_code = process.wait()

            if exit_code == 0:
                print("\n[SUCCESS] Validation completed successfully")
                self.logger.info("[SUCCESS] Validation completed successfully")
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
                self.logger.error(f"[ERROR] Validation failed with exit code {exit_code}")
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
            self.logger.exception(f"[ERROR] Validation failed: {e}")
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
        from io import StringIO

        self.logger.info("Running Table One generation with memory monitoring...")

        # Create a Tee class to write to both console and capture buffer
        class TeeOutput:
            """Write to multiple outputs simultaneously."""
            def __init__(self, *outputs):
                self.outputs = outputs

            def write(self, data):
                for output in self.outputs:
                    output.write(data)
                    output.flush()  # Ensure immediate output

            def flush(self):
                for output in self.outputs:
                    output.flush()

        try:
            # Capture stdout during execution
            stdout_capture = StringIO()
            old_stdout = sys.stdout

            try:
                # Redirect stdout to both console and capture buffer
                sys.stdout = TeeOutput(old_stdout, stdout_capture)

                # Use the new TableOneRunner directly
                runner = TableOneRunner(self.config, force_refresh=self.force_refresh)
                success = runner.run(profile_mode=False)

            finally:
                # Restore original stdout
                sys.stdout = old_stdout

                # Log all captured output to file
                captured_output = stdout_capture.getvalue()
                for line in captured_output.split('\n'):
                    if line.strip():  # Skip empty lines
                        self.logger.info(line)

            if success:
                self.logger.info("✅ Table One generation completed successfully")
            else:
                self.logger.warning("⚠️  Table One generation completed with warnings")

            self.results['tableone'] = {
                'success': success
            }

            return success

        except Exception as e:
            self.logger.exception(f"[ERROR] Table One generation failed: {e}")
            self.results['tableone'] = {
                'success': False,
                'error': str(e)
            }
            return False

    def run_tableone_ward(self):
        """
        Run Ward Table One generation in an isolated subprocess.

        Subprocess isolation guarantees memory from the critical-illness run is
        fully released before the ward run starts, so peak memory equals the
        larger of the two cohorts (not the sum). Required for 16GB systems.

        Returns
        -------
        bool
            True if ward table one generation succeeded
        """
        self.print_header("STEP 2b: WARD TABLE ONE GENERATION")

        self.logger.info("Running Ward Table One generation in isolated subprocess...")
        self.logger.info("(memory from any prior in-process Table One run is released before this starts)")

        cmd = [sys.executable, 'run_tableone_ward.py']
        if self.force_refresh:
            cmd.append('--force-refresh')
        self.logger.info(f"Running: {' '.join(cmd)}")

        try:
            env = os.environ.copy()
            env['PYTHONUTF8'] = '1'
            env['PYTHONIOENCODING'] = 'utf-8'
            # Force unbuffered child stdout so per-table progress lines reach
            # ProgressDisplay in real time. Without this, pipe-destination
            # stdout is block-buffered (~8 KB) and the live status line sits
            # on "starting…" for minutes until validation flushes.
            env['PYTHONUNBUFFERED'] = '1'

            last_output = ''
            if sys.platform == 'win32':
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    env=env
                )
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        line = output.rstrip()
                        if line:
                            print(line)
                            self.logger.info(line)
                            last_output = line
                exit_code = process.poll()
            else:
                # Merge stderr into stdout: see validation subprocess above.
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    env=env
                )
                for line in process.stdout:
                    line = line.rstrip()
                    if line:
                        print(line)
                        self.logger.info(line)
                        last_output = line
                exit_code = process.wait()

            success = (exit_code == 0)

            if success:
                self.logger.info("✅ Ward Table One generation completed successfully")
            else:
                self.logger.warning(f"⚠️  Ward Table One generation failed (exit code {exit_code})")

            self.results['tableone_ward'] = {
                'success': success,
                'exit_code': exit_code,
                'error': last_output if not success else None,
            }

            return success

        except Exception as e:
            self.logger.exception(f"[ERROR] Ward Table One generation failed: {e}")
            self.results['tableone_ward'] = {
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
            Whether to generate interactive HTML distribution viewers

        Returns
        -------
        bool
            True if ECDF generation succeeded
        """
        self.print_header("STEP 3: GET ECDF BINS")

        # Import and use the new ECDFRunner module
        from modules.ecdf.runner import ECDFRunner
        from io import StringIO

        self.logger.info("Running ECDF generation...")
        self.logger.info(f"Visualize: {'[OK]' if visualize else '[X]'}")

        # Create a Tee class to write to both console and capture buffer
        class TeeOutput:
            """Write to multiple outputs simultaneously."""
            def __init__(self, *outputs):
                self.outputs = outputs

            def write(self, data):
                for output in self.outputs:
                    output.write(data)
                    output.flush()  # Ensure immediate output

            def flush(self):
                for output in self.outputs:
                    output.flush()

        try:
            # Capture stdout during execution
            stdout_capture = StringIO()
            old_stdout = sys.stdout

            try:
                # Redirect stdout to both console and capture buffer
                sys.stdout = TeeOutput(old_stdout, stdout_capture)

                # Use the new ECDFRunner directly
                runner = ECDFRunner(self.config)
                success = runner.run(visualize=visualize)

            finally:
                # Restore original stdout
                sys.stdout = old_stdout

                # Log all captured output to file
                captured_output = stdout_capture.getvalue()
                for line in captured_output.split('\n'):
                    if line.strip():  # Skip empty lines
                        self.logger.info(line)

            if success:
                self.logger.info("[SUCCESS] ECDF generation completed successfully")
            else:
                self.logger.warning("[WARNING] ECDF generation completed with warnings")

            self.results['get_ecdf'] = {
                'success': success,
                'visualize': visualize
            }

            return success

        except Exception as e:
            self.logger.exception(f"[ERROR] ECDF generation failed: {e}")
            self.results['get_ecdf'] = {
                'success': False,
                'error': str(e)
            }
            return False

    def _record_step_timing(self, step_key, start_time):
        """Stamp start/end/duration on self.results[step_key] after a step returns.

        Safe no-op if the step's result dict was never created (e.g. the runner
        exited early). Parent calls this right after each self.run_* call.
        """
        if self.results.get(step_key) is None:
            self.results[step_key] = {}
        end_time = datetime.now()
        self.results[step_key]['start_time'] = start_time
        self.results[step_key]['end_time'] = end_time
        self.results[step_key]['duration_sec'] = (end_time - start_time).total_seconds()

    def _parse_peak_rss(self, report_path):
        """Extract 'Peak Memory: NNNN.N MB' from a step execution report. Returns float MB or None."""
        import re
        if not report_path.exists():
            return None
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                match = re.search(r'Peak Memory:\s+([\d.]+)\s+MB', f.read())
            return float(match.group(1)) if match else None
        except Exception:
            return None

    def generate_workflow_report(self):
        """Write a unified workflow-level execution report stitching all step reports together.

        Each step's wall-clock window and peak RSS are captured: duration from self.results
        (stamped by _record_step_timing), peak RSS parsed from per-step report files.
        """
        from modules.utils.output_paths import meta_dir
        report_path = meta_dir() / 'workflow_execution_report.txt'
        report_path.parent.mkdir(parents=True, exist_ok=True)

        end_time = datetime.now()
        total_sec = (end_time - self.start_time).total_seconds()

        def _fmt_duration(sec):
            if sec is None:
                return '—'
            m, s = divmod(int(sec), 60)
            h, m = divmod(m, 60)
            return f"{h:02d}:{m:02d}:{s:02d}"

        def _fmt_offset(ts):
            if ts is None:
                return '—'
            return _fmt_duration((ts - self.start_time).total_seconds())

        def _fmt_rss(mb):
            if mb is None:
                return '—'
            return f"{mb:>10,.1f} MB"

        md = meta_dir()
        step_rows = [
            ('Step 1: Validation',       self.results.get('validation'),     None),
            ('Step 2: CI Table One',     self.results.get('tableone'),       md / 'tableone_execution_report.txt'),
            ('Step 2b: Ward Table One',  self.results.get('tableone_ward'),  md / 'tableone_ward_execution_report.txt'),
            ('Step 3: ECDF',             self.results.get('get_ecdf'),       md / 'ecdf_execution_report.txt'),
        ]

        peaks = {}
        for label, r, report_file in step_rows:
            if r is None:
                continue
            peaks[label] = self._parse_peak_rss(report_file) if report_file else None

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("WORKFLOW EXECUTION REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Command: {' '.join(sys.argv)}\n")
            f.write(f"Start:   {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"End:     {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total:   {_fmt_duration(total_sec)}\n\n")

            f.write("="*80 + "\n")
            f.write("STEP TIMELINE\n")
            f.write("="*80 + "\n\n")

            header = f"{'Step':<28} {'Start':>10} {'End':>10} {'Duration':>10} {'Peak RSS':>14}   Report\n"
            f.write(header)
            f.write('-' * (len(header) - 1) + '\n')

            for label, r, report_file in step_rows:
                if r is None:
                    continue
                start = r.get('start_time')
                end = r.get('end_time')
                dur = r.get('duration_sec')
                peak = peaks.get(label)
                report_name = report_file.name if report_file is not None else '(no report — subprocess)'
                f.write(
                    f"{label:<28} {_fmt_offset(start):>10} {_fmt_offset(end):>10} "
                    f"{_fmt_duration(dur):>10} {_fmt_rss(peak):>14}   {report_name}\n"
                )

            f.write("\n")
            f.write("="*80 + "\n")
            f.write("PEAK ACROSS ALL STEPS\n")
            f.write("="*80 + "\n\n")

            valid_peaks = {k: v for k, v in peaks.items() if v is not None}
            if valid_peaks:
                winner = max(valid_peaks, key=valid_peaks.get)
                f.write(f"Peak RSS:      {valid_peaks[winner]:,.1f} MB (during {winner})\n")
            else:
                f.write("Peak RSS:      — (no step reports available)\n")
            f.write(f"Total Runtime: {_fmt_duration(total_sec)}\n")

        print(f"\n📊 Workflow report saved: {report_path}")

    def generate_summary_report(self):
        """Generate final summary report."""
        self.print_header("WORKFLOW SUMMARY")

        elapsed_time = (datetime.now() - self.start_time).total_seconds()

        # Helper function to both print and log
        def print_and_log(message):
            print(message)
            self.logger.info(message)

        # ⏱️ Total runtime — surfaced FIRST in the summary block and also
        # mirrored to stdout via plain print() so it's visible on the terminal
        # even when the user has redirected the workflow log to a file.
        _h = int(elapsed_time // 3600)
        _m = int((elapsed_time % 3600) // 60)
        _s = elapsed_time % 60
        if _h:
            _hms = f"{_h}h {_m}m {_s:.1f}s"
        elif _m:
            _hms = f"{_m}m {_s:.1f}s"
        else:
            _hms = f"{_s:.1f}s"
        total_time_line = (
            f"⏱️  TOTAL RUNTIME: {_hms}  "
            f"({elapsed_time:.1f} s / {elapsed_time/60:.1f} min)"
        )
        # Print both ways: stdout for terminal visibility, logger for the log file.
        print("\n" + "=" * len(total_time_line))
        print(total_time_line)
        print("=" * len(total_time_line) + "\n")
        self.logger.info(total_time_line)

        print_and_log(f"Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print_and_log(f"End Time:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print_and_log(f"Duration:   {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)\n")

        # Per-step status. Distinguish:
        #   exit_code == 0       → [SUCCESS]
        #   exit_code == 2       → [PARTIAL] (e.g. validation: pipeline ran but
        #                          some tables had errors / one was absent)
        #   any other non-zero   → [FAILED]
        def _status(result):
            if not result:
                return None
            if result.get('success'):
                return "[SUCCESS]"
            if result.get('exit_code') == 2:
                return "[PARTIAL]"
            return "[FAILED]"

        step_statuses = []  # for overall rollup

        # Validation results
        if self.results['validation']:
            val_status = _status(self.results['validation'])
            print_and_log(f"Validation:        {val_status}")
            if val_status == '[FAILED]' and self.results['validation'].get('error'):
                print_and_log(f"  Error: {self.results['validation']['error']}")
            step_statuses.append(val_status)

            # Show critical tables status if relevant
            if 'critical_tables_ok' in self.results['validation']:
                crit_status = "[PASS]" if self.results['validation']['critical_tables_ok'] else "[FAIL]"
                print_and_log(f"Critical Tables:   {crit_status}")

        # Table One results
        if self.results['tableone']:
            tbl_status = _status(self.results['tableone'])
            print_and_log(f"Table One (CI):    {tbl_status}")
            if tbl_status == '[FAILED]' and self.results['tableone'].get('error'):
                print_and_log(f"  Error: {self.results['tableone']['error']}")
            step_statuses.append(tbl_status)

        # Ward Table One results
        if self.results['tableone_ward']:
            ward_status = _status(self.results['tableone_ward'])
            print_and_log(f"Table One (Ward):  {ward_status}")
            if ward_status == '[FAILED]' and self.results['tableone_ward'].get('error'):
                print_and_log(f"  Error: {self.results['tableone_ward']['error']}")
            step_statuses.append(ward_status)

        # Get ECDF results
        if self.results['get_ecdf']:
            ecdf_status = _status(self.results['get_ecdf'])
            print_and_log(f"Get ECDF:          {ecdf_status}")
            if ecdf_status == '[FAILED]' and self.results['get_ecdf'].get('error'):
                print_and_log(f"  Error: {self.results['get_ecdf']['error']}")
            step_statuses.append(ecdf_status)

        # Overall rollup: SUCCESS only if every step was SUCCESS; PARTIAL if any
        # step is PARTIAL and none are FAILED; FAILED if any step is FAILED.
        # Critical-tables gate still controls whether the app may launch.
        if "[FAILED]" in step_statuses:
            overall = "[FAILED]"
        elif "[PARTIAL]" in step_statuses:
            overall = "[PARTIAL]"
        else:
            overall = "[SUCCESS]"

        if self.results['validation']:
            # App-launch gate: critical tables must pass; PARTIAL on validation
            # is still allowed to proceed downstream.
            self.results['overall_success'] = (
                self.results['validation'].get('critical_tables_ok', False)
                and overall != "[FAILED]"
            )
        else:
            self.results['overall_success'] = (overall != "[FAILED]")

        print_and_log(f"\nOverall Status:    {overall}")

        # Output locations
        print_and_log(f"\n[FOLDER] Output Locations:")
        print_and_log(f"   [LOG] Workflow Log:    {self.log_file}")
        print_and_log(f"   [LOG] Latest Log:      output/final/meta/workflow_logs/workflow_execution_latest.log")
        if self.results['validation']:
            print_and_log(f"   Validation Reports: output/final/validation/pdf_reports/")
            print_and_log(f"   Combined Report:    output/final/validation/pdf_reports/combined_validation_report.pdf")
            print_and_log(f"   Validation Results: output/final/validation/consolidated/")
        if self.results['tableone']:
            print_and_log(f"   Table One (overall): output/final/overall/tableone/")
            print_and_log(f"   Table One (strata):  output/final/strata/<icu|advanced_resp|vaso|deaths>/tableone/")
        if self.results['tableone_ward']:
            print_and_log(f"   Ward Table One:      output/final/overall_ward/tableone/")
            print_and_log(f"   Ward strata:         output/final/overall_ward/strata/<icu|advanced_resp|vaso|deaths>/tableone/")
        if self.results['get_ecdf']:
            print_and_log(f"   ECDF Data (overall): output/final/overall/{{ecdf,bins}}/")
            print_and_log(f"   ECDF Data (strata):  output/final/strata/<...>/{{ecdf,bins}}/")

        self.logger.info("="*80)
        self.logger.info(f"Workflow execution log saved to: {self.log_file}")
        self.logger.info("="*80)

        print_and_log("")
        print_and_log("📋 If reporting an issue, share these logs:")
        print_and_log(f"   1. output/final/meta/workflow_logs/workflow_execution_latest.log")
        print_and_log(f"      (pipeline: validation + tableone + ward + ecdf)")
        print_and_log(f"   2. output/final/meta/workflow_logs/webapp_execution_latest.log")
        print_and_log(f"      (webapp: uvicorn + FastAPI)")

        return self.results['overall_success']

    def _reclaim_port(self, port=8000):
        """Return a usable port, self-healing from orphaned uvicorns.

        If `port` is free, returns it unchanged. If a prior CLIF uvicorn
        (identified by `server.main:app` in its command) is still bound,
        terminates it and reuses the port. Otherwise, falls back to a
        free ephemeral port so the launch still succeeds.
        """
        import socket
        import shutil
        import signal
        import time

        def _is_free(p):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(('127.0.0.1', p))
                    return True
                except OSError:
                    return False

        if _is_free(port):
            return port

        holder_pid = None
        if shutil.which('lsof'):
            try:
                out = subprocess.check_output(
                    ['lsof', f'-iTCP:{port}', '-sTCP:LISTEN', '-n', '-P', '-Fpc'],
                    text=True, stderr=subprocess.DEVNULL,
                )
                pid = cmd_name = None
                for line in out.splitlines():
                    if line.startswith('p'):
                        pid = int(line[1:])
                    elif line.startswith('c'):
                        cmd_name = line[1:]
                if pid and cmd_name and 'python' in cmd_name.lower():
                    ps_out = subprocess.check_output(
                        ['ps', '-p', str(pid), '-o', 'command='], text=True,
                    )
                    if 'server.main:app' in ps_out:
                        holder_pid = pid
            except (subprocess.CalledProcessError, ValueError):
                pass

        if holder_pid is not None:
            print(f"⚠️  Port {port} held by orphaned CLIF uvicorn (PID {holder_pid}) — cleaning up…")
            try:
                os.kill(holder_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            for _ in range(20):
                time.sleep(0.1)
                if _is_free(port):
                    return port
            try:
                os.kill(holder_pid, signal.SIGKILL)
                time.sleep(0.3)
            except ProcessLookupError:
                pass
            if _is_free(port):
                return port

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', 0))
            alt = s.getsockname()[1]
        print(f"⚠️  Port {port} is in use by another process — falling back to port {alt}")
        return alt

    def launch_app(self):
        """Launch the FastAPI web application."""
        self.print_header("LAUNCHING WEB APP")

        # Capture webapp stdout+stderr to a log file so users can share it
        # for debugging.  Written to the same workflow_logs/ directory as
        # the main execution log; timestamped copy + a 'latest' symlink-style
        # file that's overwritten each run.
        from modules.utils.output_paths import workflow_logs_dir
        log_dir = workflow_logs_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        webapp_log_file = log_dir / f'webapp_execution_{timestamp}.log'
        webapp_latest_log = log_dir / 'webapp_execution_latest.log'

        port = self._reclaim_port(8000)
        url = f"http://localhost:{port}"

        print("Starting CLIF web application...")
        print(f"The app will open in your default browser at {url}")
        print(f"Webapp logs → {webapp_latest_log}")
        print("\nPress Ctrl+C to stop the app\n")

        # Open browser after a short delay
        import threading
        import webbrowser

        def open_browser():
            import time
            time.sleep(1.5)
            webbrowser.open(url)

        threading.Thread(target=open_browser, daemon=True).start()

        cmd = [sys.executable, '-m', 'uvicorn', 'server.main:app', '--host', '127.0.0.1', '--port', str(port), '--no-access-log']

        # Send uvicorn's stdout+stderr to both log files simultaneously.
        # `tee` keeps the workflow log pattern consistent (timestamped + latest)
        # and leaves a clean terminal so the "All steps complete" final frame
        # stays visible while the app runs.
        try:
            with open(webapp_log_file, 'w', encoding='utf-8') as fp_ts, \
                 open(webapp_latest_log, 'w', encoding='utf-8') as fp_latest:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                try:
                    for line in process.stdout:
                        fp_ts.write(line)
                        fp_latest.write(line)
                        fp_ts.flush()
                        fp_latest.flush()
                    process.wait()
                    if process.returncode not in (0, None):
                        raise subprocess.CalledProcessError(process.returncode, cmd)
                except KeyboardInterrupt:
                    process.terminate()
                    process.wait()
                    raise
        except KeyboardInterrupt:
            print("\n\n[SUCCESS] Web app stopped")
        except FileNotFoundError:
            print("\n[ERROR] Error: uvicorn not found. Install with: uv add uvicorn[standard]")
        except Exception as e:
            print(f"\n[ERROR] Error launching app: {e}")
            print(f"         Check webapp log: {webapp_latest_log}")

        # Remind users what to share if anything went wrong — visible after
        # they've exited uvicorn and the progress bar has stopped.
        print("")
        print("📋 If reporting an issue, share these logs:")
        print("   1. output/final/meta/workflow_logs/workflow_execution_latest.log")
        print("      (pipeline: validation + tableone + ward + ecdf)")
        print("   2. output/final/meta/workflow_logs/webapp_execution_latest.log")
        print("      (webapp: uvicorn + FastAPI)")
        print("")

    def run(self, validate=True, tableone=True, ward_tableone=False, get_ecdf=False, **kwargs):
        """
        Run the complete workflow.

        Parameters
        ----------
        validate : bool
            Run validation step
        tableone : bool
            Run critical-illness table one generation step
        ward_tableone : bool
            Run ward table one generation step (in isolated subprocess for memory)
        get_ecdf : bool
            Run get ECDF bins step
        **kwargs : dict
            Additional arguments for validation (tables, verbose, visualize)

        Returns
        -------
        bool
            True if all steps succeeded
        """
        self.print_header("[HOSPITAL] CLIF PROJECT RUNNER")


        self.logger.info("="*80)
        self.logger.info("[HOSPITAL] CLIF PROJECT RUNNER - WORKFLOW STARTING")
        self.logger.info("="*80)
        self.logger.info(f"Configuration: {self.config_path}")
        self.logger.info(f"Data Directory: {self.config.get('tables_path', 'NOT SET')}")
        self.logger.info(f"Site Name: {self.config.get('site_name', 'NOT SET')}")
        self.logger.info(f"Workflow Steps:")
        self.logger.info(f"  1. Validation:        {'[OK]' if validate else '[X]'}")
        self.logger.info(f"  2. Table One (CI):    {'[OK]' if tableone else '[X]'}")
        self.logger.info(f"  2b. Table One (Ward): {'[OK]' if ward_tableone else '[X]'}")
        self.logger.info(f"  3. Get ECDF:          {'[OK]' if get_ecdf else '[X]'}")
        self.logger.info("="*80)

        # Step 1: Validation
        if validate:
            print("\n[DEBUG] Starting validation step...")
            _t0 = datetime.now()
            val_success, critical_tables_ok = self.run_validation(
                tables=kwargs.get('tables'),
                verbose=kwargs.get('verbose', False),
                no_summary=kwargs.get('no_summary', False)
            )
            self._record_step_timing('validation', _t0)
            print(f"[DEBUG] Validation completed. val_success={val_success}, critical_tables_ok={critical_tables_ok}")

            # Decide whether to proceed based on validation results
            can_proceed = val_success or critical_tables_ok
            print(f"[DEBUG] Can proceed to next step: {can_proceed}")

            if not can_proceed and not kwargs.get('continue_on_error', False):
                if not critical_tables_ok:
                    # Critical tables failed - show specific error
                    pass  # Error message already logged by run_validation
                else:
                    # General validation failure
                    self.logger.warning("[WARNING] Validation failed. Stopping workflow.")
                    self.logger.info("Use --continue-on-error to proceed anyway.")

                self.generate_summary_report()
                self.generate_workflow_report()
                return False

        # Step 2: Critical-illness Table One Generation
        if tableone:
            print("\n[DEBUG] Starting Table One generation (critical-illness)...")
            _t0 = datetime.now()
            tbl_success = self.run_tableone()
            self._record_step_timing('tableone', _t0)

        # Step 2b: Ward Table One Generation (isolated subprocess)
        # Independent of critical-illness Table One — has its own cohort and writes
        # to output/final/overall_ward/ + final_tableone_ward_df.parquet.
        if ward_tableone:
            print("\n[DEBUG] Starting Ward Table One generation (isolated subprocess)...")
            _t0 = datetime.now()
            self.run_tableone_ward()
            self._record_step_timing('tableone_ward', _t0)

        # Step 3: Get ECDF Bins
        if get_ecdf:
            _t0 = datetime.now()
            ecdf_success = self.run_get_ecdf(
                visualize=kwargs.get('visualize', False)
            )
            self._record_step_timing('get_ecdf', _t0)

        # Generate summary
        overall_success = self.generate_summary_report()
        self.generate_workflow_report()

        # Total wall-clock time for the whole workflow (validation + tableone +
        # ward + ECDF + summary), printed before app-launch logic so it always
        # shows regardless of whether the web app launches or is skipped.
        total_sec = (datetime.now() - self.start_time).total_seconds()
        m, s = divmod(int(total_sec), 60)
        h, m = divmod(m, 60)
        print(f"\n⏱️  Total workflow time: {h:02d}:{m:02d}:{s:02d} (HH:MM:SS) — {total_sec:,.0f}s")

        # App launch logic - launch unless critical tables failed
        # Allow override with --continue-on-error
        should_launch = overall_success or kwargs.get('continue_on_error', False)

        if should_launch:
            # Check if user wants to skip app launch
            if not kwargs.get('no_launch_app', False):
                print("\n" + "="*80)
                print("🚀 Launching Web App...")
                print("="*80)

                # Show warnings for any failed steps
                warnings_shown = False

                if self.results['tableone'] and not self.results['tableone'].get('success', False):
                    print("\n⚠️  Warning: Table One (critical-illness) generation failed")
                    print("   The app will launch but Table One data will not be available")
                    warnings_shown = True

                if self.results['tableone_ward'] and not self.results['tableone_ward'].get('success', False):
                    print("\n⚠️  Warning: Ward Table One generation failed")
                    print("   The app will launch but Ward Table One data will not be available")
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

                print("")
                print("📋 If reporting an issue, share these logs:")
                print("   1. output/final/meta/workflow_logs/workflow_execution_latest.log")
                print("      (pipeline: validation + tableone + ward + ecdf)")
                print("   2. output/final/meta/workflow_logs/webapp_execution_latest.log")
                print("      (webapp: uvicorn + FastAPI)")

                print("\nStarting the interactive web application in 3 seconds...")
                print("(Press Ctrl+C now to skip)\n")

                try:
                    import time
                    time.sleep(3)
                    self.launch_app()
                except KeyboardInterrupt:
                    print("\n\n⏭️  App launch skipped by user")
                    print("   You can launch it manually with: uv run uvicorn server.main:app\n")
        else:
            # Only happens if critical tables validation failed and no override
            print("\n" + "="*80)
            print("[ERROR] App Launch Blocked")
            print("="*80)
            print("\nCritical tables validation failed. Cannot launch app.")
            print("Review validation report: output/final/validation/pdf_reports/combined_validation_report.pdf")
            print("\nUse --continue-on-error flag to bypass this check (not recommended)\n")

        # Zip output/final for easy upload to Box
        import shutil
        _final_dir = Path(self.config_path).resolve().parent.parent / 'output' / 'final'
        if _final_dir.exists():
            _zip_path = _final_dir.parent / 'final'
            shutil.make_archive(str(_zip_path), 'zip', str(_final_dir.parent), 'final')
            print(f"\n📦 Zipped results: output/final.zip (upload this to Box)")

        return overall_success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='CLIF Project Runner - Complete Workflow Automation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Full workflow (validation + tableone)
  %(prog)s --validate-only                    # Only validation
  %(prog)s --tableone-only                    # Only critical-illness table one
  %(prog)s --ward-only                        # Only ward table one (isolated subprocess)
  %(prog)s --get-ecdf-only                    # Only get ECDF bins
  %(prog)s --get-ecdf-only --visualize        # Get ECDF + interactive HTML viewers
  %(prog)s --no-ward                          # Full workflow without ward table one
  %(prog)s --no-ecdf                          # Full workflow without ECDF generation
  %(prog)s --no-ward --no-ecdf                # CI table one only (fastest full run)
  %(prog)s --tables patient adt               # Validate specific tables
        """
    )

    # Workflow control
    workflow_group = parser.add_argument_group('Workflow Control')
    workflow_group.add_argument('--validate-only', action='store_true',
                                help='Only run validation step')
    workflow_group.add_argument('--tableone-only', action='store_true',
                                help='Only run critical-illness table one generation step')
    workflow_group.add_argument('--ward-only', action='store_true',
                                help='Only run ward table one generation step (isolated subprocess)')
    workflow_group.add_argument('--ward', action='store_true', default=None,
                                help='(deprecated, now default) Include ward table one generation')
    workflow_group.add_argument('--no-ward', action='store_true',
                                help='Skip ward table one generation')
    workflow_group.add_argument('--get-ecdf-only', action='store_true',
                                help='Only run get ECDF bins step')
    workflow_group.add_argument('--get-ecdf', action='store_true', default=None,
                                help='(deprecated, now default) Include get ECDF bins')
    workflow_group.add_argument('--no-ecdf', action='store_true',
                                help='Skip ECDF generation')
    workflow_group.add_argument('--visualize', action='store_true',
                                help='Generate interactive HTML distribution viewers (for get ECDF)')
    workflow_group.add_argument('--continue-on-error', action='store_true',
                                help='Continue to next step even if previous step fails')
    workflow_group.add_argument('--no-launch-app', action='store_true',
                                help='Skip automatic web app launch after completion')
    workflow_group.add_argument('--force-refresh', action='store_true',
                                help='Bypass filtered-CLIF-table cache and rebuild from raw source parquets')

    # Validation options
    validation_group = parser.add_argument_group('Validation Options')
    validation_group.add_argument('--tables', nargs='+',
                                  choices=['patient', 'hospitalization', 'adt', 'code_status',
                                          'crrt_therapy', 'hospital_diagnosis', 'labs',
                                          'medication_admin_continuous', 'medication_admin_intermittent',
                                          'microbiology_culture', 
                                          'microbiology_susceptibility', 'patient_assessments',
                                          'patient_procedures', 'position', 'respiratory_support', 'vitals'],
                                  help='Specific tables to validate')
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
        ward_tableone = False
        get_ecdf = False
    elif args.tableone_only:
        validate = False
        tableone = True
        ward_tableone = not args.no_ward
        get_ecdf = False
    elif args.ward_only:
        validate = False
        tableone = False
        ward_tableone = True
        get_ecdf = False
    elif args.get_ecdf_only:
        validate = False
        tableone = False
        ward_tableone = False
        get_ecdf = True
    else:
        # Default: run validation + tableone + ward + ecdf (opt-out via --no-ward / --no-ecdf)
        validate = True
        tableone = True
        ward_tableone = not args.no_ward
        get_ecdf = not args.no_ecdf

    # Default terminal UX: live rich progress bar.  --verbose keeps the
    # full scroll behavior.  File logs are captured the same way in both modes.
    progress_display = None
    _orig_stderr = None
    _saved_stderr_fd = None
    _devnull_fd = None
    if not args.verbose:
        from modules.utils.status_line import ProgressDisplay
        total_steps = sum([validate, tableone, ward_tableone, get_ecdf])
        os.environ["TQDM_DISABLE"] = "1"
        progress_display = ProgressDisplay(
            total_steps=max(total_steps, 1),
            out_stream=sys.stdout,
        )
        sys.stdout = progress_display
        # Silence stderr at both the Python level AND the OS fd level.
        # Replacing sys.stderr alone doesn't help — tqdm, polars, pyarrow and
        # other C-extension progress bars write to fd=2 directly.  We dup the
        # devnull fd over fd=2 so those writes go to /dev/null, then restore
        # on teardown.  Traceback.print_exc, warnings, and Python-level
        # stderr writes are covered by the sys.stderr swap above.
        _orig_stderr = sys.stderr
        sys.stderr = open(os.devnull, "w")
        try:
            _saved_stderr_fd = os.dup(2)
            _devnull_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(_devnull_fd, 2)
        except OSError:
            _saved_stderr_fd = None
            _devnull_fd = None
        progress_display.start()

    def _teardown_display():
        """Stop live display and restore stderr.  Idempotent."""
        nonlocal progress_display, _orig_stderr, _saved_stderr_fd, _devnull_fd
        if progress_display is not None:
            try:
                progress_display.stop()
            except Exception:
                pass
            progress_display = None
        # Restore fd=2 before closing our devnull fd.
        if _saved_stderr_fd is not None:
            try:
                os.dup2(_saved_stderr_fd, 2)
                os.close(_saved_stderr_fd)
            except OSError:
                pass
            _saved_stderr_fd = None
        if _devnull_fd is not None:
            try:
                os.close(_devnull_fd)
            except OSError:
                pass
            _devnull_fd = None
        if _orig_stderr is not None:
            try:
                sys.stderr.close()
            except Exception:
                pass
            sys.stderr = _orig_stderr
            _orig_stderr = None

    # Initialize runner
    runner = ProjectRunner(config_path=args.config, verbose=args.verbose, force_refresh=args.force_refresh)

    # Run workflow
    try:
        success = runner.run(
            validate=validate,
            tableone=tableone,
            ward_tableone=ward_tableone,
            get_ecdf=get_ecdf,
            tables=args.tables,
            no_summary=args.no_summary,
            verbose=args.verbose,
            visualize=args.visualize,
            continue_on_error=args.continue_on_error,
            no_launch_app=args.no_launch_app
        )

        # Exit with appropriate code
        _teardown_display()
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        _teardown_display()
        print("\n\n[WARNING] Workflow interrupted by user")
        print("")
        print("📋 If reporting an issue, share these logs:")
        print("   1. output/final/meta/workflow_logs/workflow_execution_latest.log")
        print("   2. output/final/meta/workflow_logs/webapp_execution_latest.log")
        sys.exit(130)
    except Exception as e:
        _teardown_display()
        print(f"\n[ERROR] Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        print("")
        print("📋 If reporting an issue, share these logs:")
        print("   1. output/final/meta/workflow_logs/workflow_execution_latest.log")
        print("   2. output/final/meta/workflow_logs/webapp_execution_latest.log")
        sys.exit(1)


if __name__ == "__main__":
    main()
