#!/usr/bin/env python3
"""
Execution script for generating Table One with memory monitoring and validation.

This script:
- Monitors memory usage throughout execution
- Validates all output files
- Provides progress tracking
- Generates a summary report

Usage:
    python run_table_one.py
    python run_table_one.py --profile  # For detailed memory profiling
"""

import os
import sys
import json
import time
import traceback
import psutil
from pathlib import Path
from datetime import datetime


class MemoryMonitor:
    """Monitor memory usage during script execution."""

    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()
        self.start_memory = self.get_memory_mb()
        self.peak_memory = self.start_memory
        self.checkpoints = []

    def get_memory_mb(self):
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def checkpoint(self, label):
        """Record a memory checkpoint."""
        current_memory = self.get_memory_mb()
        elapsed_time = time.time() - self.start_time

        if current_memory > self.peak_memory:
            self.peak_memory = current_memory

        self.checkpoints.append({
            'label': label,
            'memory_mb': current_memory,
            'peak_mb': self.peak_memory,
            'elapsed_sec': elapsed_time
        })

        print(f"  [{label}] Memory: {current_memory:.1f} MB | Peak: {self.peak_memory:.1f} MB | Time: {elapsed_time:.1f}s")

    def get_summary(self):
        """Get memory usage summary."""
        end_memory = self.get_memory_mb()
        total_time = time.time() - self.start_time

        return {
            'start_memory_mb': self.start_memory,
            'end_memory_mb': end_memory,
            'peak_memory_mb': self.peak_memory,
            'memory_increase_mb': end_memory - self.start_memory,
            'total_time_sec': total_time,
            'checkpoints': self.checkpoints
        }


def validate_config():
    """Validate configuration file exists and is correct."""
    config_path = Path(__file__).parent.parent / 'config' / 'config.json'

    if not config_path.exists():
        print(f"❌ Error: Configuration file not found at {config_path}")
        return None

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Validate required fields
        required_fields = ['tables_path', 'file_type', 'timezone']
        missing_fields = [f for f in required_fields if f not in config]

        if missing_fields:
            print(f"❌ Error: Missing required fields in config: {missing_fields}")
            return None

        # Validate data directory exists
        data_dir = Path(config['tables_path'])
        if not data_dir.exists():
            print(f"❌ Error: Data directory not found: {data_dir}")
            return None

        print(f"✅ Configuration validated")
        print(f"   Data directory: {config['tables_path']}")
        print(f"   File type: {config['file_type']}")
        print(f"   Timezone: {config['timezone']}")

        return config

    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in configuration file: {e}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def validate_outputs():
    """Validate that expected output files were created."""
    expected_files = [
        '../output/final/tableone/table_one_overall.csv',
        '../output/final/tableone/table_one_by_year.csv',
        '../output/final/tableone/consort_flow_diagram.png',
        '../output/final/tableone/cohort_intersect_upset_plot.png',
        '../output/final/tableone/venn_all_4_groups.png',
        '../output/final/tableone/medications_summary_stats.csv',
        '../output/final/tableone/comorbidities_per_1000_hospitalizations.csv',
        '../output/intermediate/final_tableone_df.parquet'
    ]

    missing_files = []
    existing_files = []

    base_dir = Path(__file__).parent

    for file_path in expected_files:
        full_path = base_dir / file_path
        if full_path.exists():
            size_mb = full_path.stat().st_size / 1024 / 1024
            existing_files.append((str(full_path), size_mb))
        else:
            missing_files.append(str(full_path))

    print(f"\n{'='*80}")
    print("OUTPUT VALIDATION")
    print(f"{'='*80}")

    if existing_files:
        print(f"\n✅ Generated files ({len(existing_files)}):")
        for file_path, size_mb in existing_files:
            print(f"   {Path(file_path).name} ({size_mb:.2f} MB)")

    if missing_files:
        print(f"\n⚠️  Missing expected files ({len(missing_files)}):")
        for file_path in missing_files:
            print(f"   {Path(file_path).name}")

    return len(missing_files) == 0


def run_table_one_script(memory_monitor):
    """Execute the main table one generation script."""
    script_path = Path(__file__).parent / 'generate_table_one_2_1.py'

    if not script_path.exists():
        print(f"❌ Error: Script not found at {script_path}")
        return False

    print(f"\n{'='*80}")
    print("EXECUTING TABLE ONE GENERATION")
    print(f"{'='*80}")
    print(f"Script: {script_path}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    memory_monitor.checkpoint("Script Start")

    try:
        # Change to code directory so relative paths work correctly
        original_dir = os.getcwd()
        code_dir = script_path.parent
        os.chdir(code_dir)

        # Execute the script
        with open(script_path, 'r') as f:
            code = f.read()

        # Execute in current namespace to capture outputs
        exec(code, {'__name__': '__main__', '__file__': str(script_path)})

        # Restore original directory
        os.chdir(original_dir)

        memory_monitor.checkpoint("Script Complete")
        print(f"\n{'='*80}")
        print("✅ TABLE ONE GENERATION SUCCESSFUL")
        print(f"{'='*80}")
        return True

    except Exception as e:
        # Restore original directory on error
        os.chdir(original_dir)

        memory_monitor.checkpoint("Script Failed")
        print(f"\n{'='*80}")
        print("❌ TABLE ONE GENERATION FAILED")
        print(f"{'='*80}")
        print(f"\nError: {e}")
        print(f"\nTraceback:")
        traceback.print_exc()
        return False


def generate_report(memory_monitor, success, validation_passed):
    """Generate a summary report."""
    summary = memory_monitor.get_summary()

    report_path = Path(__file__).parent.parent / 'output' / 'final' / 'tableone' / 'execution_report.txt'
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TABLE ONE GENERATION EXECUTION REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Status: {'✅ SUCCESS' if success else '❌ FAILED'}\n")
        f.write(f"Validation: {'✅ PASSED' if validation_passed else '⚠️  INCOMPLETE'}\n\n")

        f.write("="*80 + "\n")
        f.write("MEMORY USAGE SUMMARY\n")
        f.write("="*80 + "\n\n")

        f.write(f"Start Memory:     {summary['start_memory_mb']:.1f} MB\n")
        f.write(f"End Memory:       {summary['end_memory_mb']:.1f} MB\n")
        f.write(f"Peak Memory:      {summary['peak_memory_mb']:.1f} MB\n")
        f.write(f"Memory Increase:  {summary['memory_increase_mb']:.1f} MB\n")
        f.write(f"Total Time:       {summary['total_time_sec']:.1f} seconds ({summary['total_time_sec']/60:.1f} minutes)\n\n")

        if summary['checkpoints']:
            f.write("="*80 + "\n")
            f.write("MEMORY CHECKPOINTS\n")
            f.write("="*80 + "\n\n")

            for cp in summary['checkpoints']:
                f.write(f"{cp['label']:<30} Memory: {cp['memory_mb']:>8.1f} MB | Peak: {cp['peak_mb']:>8.1f} MB | Time: {cp['elapsed_sec']:>8.1f}s\n")

    print(f"\n📊 Execution report saved: {report_path}")

    # Print summary to console
    print(f"\n{'='*80}")
    print("MEMORY USAGE SUMMARY")
    print(f"{'='*80}")
    print(f"Start Memory:     {summary['start_memory_mb']:.1f} MB")
    print(f"End Memory:       {summary['end_memory_mb']:.1f} MB")
    print(f"Peak Memory:      {summary['peak_memory_mb']:.1f} MB")
    print(f"Memory Increase:  {summary['memory_increase_mb']:.1f} MB")
    print(f"Total Time:       {summary['total_time_sec']:.1f} seconds ({summary['total_time_sec']/60:.1f} minutes)")


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("TABLE ONE GENERATION WITH MEMORY MONITORING")
    print("="*80 + "\n")

    # Check for profiling flag
    profile_mode = '--profile' in sys.argv
    if profile_mode:
        print("📊 Memory profiling mode enabled (requires memory_profiler)")
        print("   Note: This will significantly slow down execution\n")

    # Initialize memory monitor
    memory_monitor = MemoryMonitor()
    memory_monitor.checkpoint("Initialization")

    # Step 1: Validate configuration
    print(f"\n{'='*80}")
    print("STEP 1: VALIDATING CONFIGURATION")
    print(f"{'='*80}\n")

    config = validate_config()
    if config is None:
        print("\n❌ Validation failed. Exiting.")
        sys.exit(1)

    memory_monitor.checkpoint("Config Validated")

    # Step 2: Run the table one generation script
    success = run_table_one_script(memory_monitor)

    # Step 3: Validate outputs
    if success:
        print(f"\n{'='*80}")
        print("STEP 3: VALIDATING OUTPUTS")
        print(f"{'='*80}")

        validation_passed = validate_outputs()
    else:
        validation_passed = False

    memory_monitor.checkpoint("Validation Complete")

    # Step 4: Generate report
    generate_report(memory_monitor, success, validation_passed)

    # Exit with appropriate code
    if success and validation_passed:
        print(f"\n{'='*80}")
        print("✅ ALL STEPS COMPLETED SUCCESSFULLY")
        print(f"{'='*80}\n")
        sys.exit(0)
    elif success:
        print(f"\n{'='*80}")
        print("⚠️  GENERATION SUCCEEDED BUT VALIDATION INCOMPLETE")
        print(f"{'='*80}\n")
        sys.exit(2)
    else:
        print(f"\n{'='*80}")
        print("❌ GENERATION FAILED")
        print(f"{'='*80}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
