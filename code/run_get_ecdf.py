#!/usr/bin/env python3
"""
Execution script for getting ECDF and bins data with validation.

This script:
- Validates configuration file exists
- Runs ECDF/bins generation (ECDF and bins for labs/vitals/respiratory)
- Optionally generates HTML visualizations
- Validates output files
- Provides progress tracking
- Generates a summary report

Usage:
    python code/run_get_ecdf.py
    python code/run_get_ecdf.py --visualize  # Also generate HTML plots
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime


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


def check_dependencies():
    """Check if required dependencies and config files exist."""
    print(f"\n{'='*80}")
    print("CHECKING DEPENDENCIES")
    print(f"{'='*80}\n")

    missing = []

    # Check for config files
    outlier_config = Path('get_ecdf/ecdf_config/outlier_config.yaml')
    lab_vital_config = Path('get_ecdf/ecdf_config/lab_vital_config.yaml')
    utils = Path('get_ecdf/utils.py')

    if not outlier_config.exists():
        missing.append(f"get_ecdf/ecdf_config/outlier_config.yaml - Required for outlier filtering")
    else:
        print(f"✓ Found: get_ecdf/ecdf_config/outlier_config.yaml")

    if not lab_vital_config.exists():
        missing.append(f"get_ecdf/ecdf_config/lab_vital_config.yaml - Required for binning configuration")
    else:
        print(f"✓ Found: get_ecdf/ecdf_config/lab_vital_config.yaml")

    if not utils.exists():
        missing.append(f"get_ecdf/utils.py - Required for create_all_bins() function")
    else:
        print(f"✓ Found: get_ecdf/utils.py")

    if missing:
        print(f"\n❌ Missing dependencies:")
        for item in missing:
            print(f"   - {item}")
        print(f"\n   ECDF generation requires these files to run.")
        print(f"   Please ensure all configuration files are in place.")
        return False

    print(f"\n✅ All dependencies found")
    return True


def validate_outputs():
    """Validate that expected output files were created."""
    expected_dirs = [
        'output/final/configs',
        'output/final/ecdf/labs',
        'output/final/ecdf/vitals',
        'output/final/ecdf/respiratory_support',
        'output/final/bins/labs',
        'output/final/bins/vitals',
        'output/final/bins/respiratory_support',
    ]

    missing_dirs = []
    existing_dirs = []

    for dir_path in expected_dirs:
        full_path = Path(dir_path)
        if full_path.exists():
            # Count files in directory
            file_count = len(list(full_path.glob('*.parquet')))
            existing_dirs.append((str(full_path), file_count))
        else:
            missing_dirs.append(str(full_path))

    print(f"\n{'='*80}")
    print("OUTPUT VALIDATION")
    print(f"{'='*80}")

    if existing_dirs:
        print(f"\n✅ Generated directories ({len(existing_dirs)}):")
        for dir_path, file_count in existing_dirs:
            print(f"   {Path(dir_path).name:30s} ({file_count} files)")

    if missing_dirs:
        print(f"\n⚠️  Missing expected directories ({len(missing_dirs)}):")
        for dir_path in missing_dirs:
            print(f"   {Path(dir_path).name}")

    # Check for log file
    log_file = Path('output/final/unit_mismatches.log')
    if log_file.exists():
        print(f"\n📋 Log file: {log_file}")

    return len(missing_dirs) == 0


def run_get_ecdf_script():
    """Execute the get ECDF bins script."""
    script_path = Path(__file__).parent.parent / 'get_ecdf' / 'precompute_ecdf_bins.py'

    if not script_path.exists():
        print(f"❌ Error: Script not found at {script_path}")
        return False

    print(f"\n{'='*80}")
    print("EXECUTING GET ECDF BINS")
    print(f"{'='*80}")
    print(f"Script: {script_path}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    start_time = time.time()

    try:
        # Change to project root so relative paths work correctly
        original_dir = os.getcwd()
        project_root = script_path.parent.parent
        os.chdir(project_root)

        # Import and execute the main function
        sys.path.insert(0, str(script_path.parent))
        from precompute_ecdf_bins import main
        main()

        # Restore original directory
        os.chdir(original_dir)

        elapsed = time.time() - start_time

        print(f"\n{'='*80}")
        print("✅ GET ECDF SUCCESSFUL")
        print(f"{'='*80}")
        print(f"Duration: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")

        return True

    except Exception as e:
        # Restore original directory on error
        os.chdir(original_dir)

        print(f"\n{'='*80}")
        print("❌ GET ECDF FAILED")
        print(f"{'='*80}")
        print(f"\nError: {e}")
        print(f"\nTraceback:")
        traceback.print_exc()
        return False


def run_visualization_script():
    """Execute the visualization script to generate HTML plots."""
    script_path = Path(__file__).parent.parent / 'get_ecdf' / 'visualize_bins_ecdf.py'

    if not script_path.exists():
        print(f"❌ Error: Visualization script not found at {script_path}")
        return False

    print(f"\n{'='*80}")
    print("GENERATING HTML VISUALIZATIONS")
    print(f"{'='*80}")
    print(f"Script: {script_path}")
    print(f"{'='*80}\n")

    start_time = time.time()

    try:
        # Change to project root so relative paths work correctly
        original_dir = os.getcwd()
        project_root = script_path.parent.parent
        os.chdir(project_root)

        # Import and execute the main function
        sys.path.insert(0, str(script_path.parent))
        from visualize_bins_ecdf import main
        main()

        # Restore original directory
        os.chdir(original_dir)

        elapsed = time.time() - start_time

        print(f"\n{'='*80}")
        print("✅ VISUALIZATION SUCCESSFUL")
        print(f"{'='*80}")
        print(f"Duration: {elapsed:.1f} seconds")

        return True

    except Exception as e:
        # Restore original directory on error
        os.chdir(original_dir)

        print(f"\n{'='*80}")
        print("❌ VISUALIZATION FAILED")
        print(f"{'='*80}")
        print(f"\nError: {e}")
        print(f"\nTraceback:")
        traceback.print_exc()
        return False


def generate_report(success, validation_passed, total_time):
    """Generate a summary report."""
    report_path = Path('output/final/execution_report.txt')
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("GET ECDF BINS EXECUTION REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Status: {'✅ SUCCESS' if success else '❌ FAILED'}\n")
        f.write(f"Validation: {'✅ PASSED' if validation_passed else '⚠️  INCOMPLETE'}\n\n")

        f.write(f"Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)\n\n")

        f.write("="*80 + "\n")
        f.write("OUTPUT STRUCTURE\n")
        f.write("="*80 + "\n\n")

        f.write("output/final/\n")
        f.write("├── configs/           # Configuration files\n")
        f.write("├── ecdf/             # ECDF parquet files\n")
        f.write("│   ├── labs/\n")
        f.write("│   ├── vitals/\n")
        f.write("│   └── respiratory_support/\n")
        f.write("├── bins/             # Bin parquet files\n")
        f.write("│   ├── labs/\n")
        f.write("│   ├── vitals/\n")
        f.write("│   └── respiratory_support/\n")
        f.write("└── unit_mismatches.log\n")

    print(f"\n📊 Execution report saved: {report_path}")


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("GET ECDF AND BINS")
    print("="*80 + "\n")

    # Check for visualization flag
    visualize = '--visualize' in sys.argv
    if visualize:
        print("📊 Visualization mode enabled - will generate HTML plots after ECDF generation\n")

    start_time = time.time()

    # Step 1: Validate configuration
    print(f"{'='*80}")
    print("STEP 1: VALIDATING CONFIGURATION")
    print(f"{'='*80}\n")

    config = validate_config()
    if config is None:
        print("\n❌ Validation failed. Exiting.")
        sys.exit(1)

    # Step 2: Check dependencies
    deps_ok = check_dependencies()
    if not deps_ok:
        print("\n❌ Dependency check failed. Exiting.")
        sys.exit(1)

    # Step 3: Run ECDF generation
    success = run_get_ecdf_script()

    # Step 4: Run visualization (optional)
    if success and visualize:
        viz_success = run_visualization_script()
        if not viz_success:
            print("\n⚠️  ECDF generation succeeded but visualization failed")

    # Step 5: Validate outputs
    if success:
        validation_passed = validate_outputs()
    else:
        validation_passed = False

    total_time = time.time() - start_time

    # Step 6: Generate report
    generate_report(success, validation_passed, total_time)

    # Print summary
    print(f"\n{'='*80}")
    print("EXECUTION SUMMARY")
    print(f"{'='*80}")
    print(f"Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
    print(f"Validation: {'✅ PASSED' if validation_passed else '⚠️  INCOMPLETE'}")

    # Exit with appropriate code
    if success and validation_passed:
        print(f"\n{'='*80}")
        print("✅ ALL STEPS COMPLETED SUCCESSFULLY")
        print(f"{'='*80}\n")
        sys.exit(0)
    elif success:
        print(f"\n{'='*80}")
        print("⚠️  GET ECDF SUCCEEDED BUT VALIDATION INCOMPLETE")
        print(f"{'='*80}\n")
        sys.exit(2)
    else:
        print(f"\n{'='*80}")
        print("❌ GET ECDF FAILED")
        print(f"{'='*80}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
