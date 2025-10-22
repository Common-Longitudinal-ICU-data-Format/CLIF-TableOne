"""
ECDF Runner Module

Provides execution and validation for ECDF/bins generation with
progress tracking and output validation.
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime


class ECDFRunner:
    """Runner for ECDF and bins data generation with validation."""

    def __init__(self, config=None):
        """Initialize ECDF runner with configuration."""
        self.config = config or self.load_config()
        self.start_time = None
        self.project_root = Path(__file__).parent.parent.parent

    def load_config(self):
        """Load configuration from config.json."""
        config_path = Path(__file__).parent.parent.parent / 'config' / 'config.json'

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        with open(config_path, 'r') as f:
            return json.load(f)

    def validate_config(self):
        """Validate configuration file exists and is correct."""
        try:
            # Validate required fields
            required_fields = ['tables_path', 'file_type', 'timezone']
            missing_fields = [f for f in required_fields if f not in self.config]

            if missing_fields:
                print(f"❌ Error: Missing required fields in config: {missing_fields}")
                return False

            # Validate data directory exists
            data_dir = Path(self.config['tables_path'])
            if not data_dir.exists():
                print(f"❌ Error: Data directory not found: {data_dir}")
                return False

            print(f"✅ Configuration validated")
            print(f"   Data directory: {self.config['tables_path']}")
            print(f"   File type: {self.config['file_type']}")
            print(f"   Timezone: {self.config['timezone']}")

            return True

        except Exception as e:
            print(f"❌ Error validating config: {e}")
            return False

    def check_dependencies(self):
        """Check if required dependencies and config files exist."""
        print(f"\n{'='*80}")
        print("CHECKING DEPENDENCIES")
        print(f"{'='*80}\n")

        missing = []

        # Check for config files in new location
        outlier_config = self.project_root / 'modules' / 'ecdf' / 'config' / 'outlier_config.yaml'
        lab_vital_config = self.project_root / 'modules' / 'ecdf' / 'config' / 'lab_vital_config.yaml'
        utils = self.project_root / 'modules' / 'ecdf' / 'utils.py'

        if not outlier_config.exists():
            missing.append(f"modules/ecdf/config/outlier_config.yaml - Required for outlier filtering")
        else:
            print(f"✓ Found: modules/ecdf/config/outlier_config.yaml")

        if not lab_vital_config.exists():
            missing.append(f"modules/ecdf/config/lab_vital_config.yaml - Required for binning configuration")
        else:
            print(f"✓ Found: modules/ecdf/config/lab_vital_config.yaml")

        if not utils.exists():
            missing.append(f"modules/ecdf/utils.py - Required for create_all_bins() function")
        else:
            print(f"✓ Found: modules/ecdf/utils.py")

        if missing:
            print(f"\n❌ Missing dependencies:")
            for item in missing:
                print(f"   - {item}")
            print(f"\n   ECDF generation requires these files to run.")
            print(f"   Please ensure all configuration files are in place.")
            return False

        print(f"\n✅ All dependencies found")
        return True

    def validate_outputs(self):
        """Validate that expected output files were created."""
        expected_dirs = [
            self.project_root / 'output/final/configs',
            self.project_root / 'output/final/ecdf/labs',
            self.project_root / 'output/final/ecdf/vitals',
            self.project_root / 'output/final/ecdf/respiratory_support',
            self.project_root / 'output/final/bins/labs',
            self.project_root / 'output/final/bins/vitals',
            self.project_root / 'output/final/bins/respiratory_support',
            self.project_root / 'output/final/stats',
        ]

        missing_dirs = []
        existing_dirs = []

        for dir_path in expected_dirs:
            if dir_path.exists():
                # Count files in directory
                if dir_path.name == 'stats':
                    file_count = len(list(dir_path.glob('*.csv')))
                else:
                    file_count = len(list(dir_path.glob('*.parquet')))
                existing_dirs.append((str(dir_path), file_count))
            else:
                missing_dirs.append(str(dir_path))

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
        log_file = self.project_root / 'output/final/unit_mismatches.log'
        if log_file.exists():
            print(f"\n📋 Log file: {log_file}")

        return len(missing_dirs) == 0

    def execute_ecdf_generation(self):
        """Execute the main ECDF/bins generation."""
        print(f"\n{'='*80}")
        print("EXECUTING ECDF/BINS GENERATION")
        print(f"{'='*80}")
        print(f"Module: modules.ecdf.generator")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")

        try:
            # Import and execute the main function
            from .generator import main
            main()

            print(f"\n{'='*80}")
            print("✅ ECDF GENERATION SUCCESSFUL")
            print(f"{'='*80}")

            return True

        except Exception as e:
            print(f"\n{'='*80}")
            print("❌ ECDF GENERATION FAILED")
            print(f"{'='*80}")
            print(f"\nError: {e}")
            print(f"\nTraceback:")
            traceback.print_exc()
            return False

    def execute_visualization(self):
        """Execute the visualization script to generate HTML plots."""
        print(f"\n{'='*80}")
        print("GENERATING HTML VISUALIZATIONS")
        print(f"{'='*80}")
        print(f"Module: modules.ecdf.visualizer")
        print(f"{'='*80}\n")

        try:
            # Import and execute the visualization function
            from .visualizer import main
            main()

            print(f"\n{'='*80}")
            print("✅ VISUALIZATION SUCCESSFUL")
            print(f"{'='*80}")

            return True

        except Exception as e:
            print(f"\n{'='*80}")
            print("❌ VISUALIZATION FAILED")
            print(f"{'='*80}")
            print(f"\nError: {e}")
            print(f"\nTraceback:")
            traceback.print_exc()
            return False

    def execute_statistics_generation(self):
        """Execute collection statistics generation."""
        print(f"\n{'='*80}")
        print("GENERATING COLLECTION STATISTICS")
        print(f"{'='*80}")
        print(f"Module: modules.ecdf.statistics")
        print(f"{'='*80}\n")

        try:
            # Import necessary functions
            from .generator import load_configs, extract_icu_time_windows, discover_lab_category_units
            from .statistics import compute_collection_statistics

            # Load configurations
            clif_config, outlier_config, lab_vital_config = load_configs()

            # Extract ICU time windows
            icu_windows = extract_icu_time_windows(
                clif_config['tables_path'],
                clif_config['file_type']
            )

            # Discover lab category-unit combinations
            lab_category_units = discover_lab_category_units(
                clif_config['tables_path'],
                clif_config['file_type']
            )

            # Compute statistics
            output_dir = self.project_root / 'output/final'
            stats_path = compute_collection_statistics(
                icu_windows=icu_windows,
                tables_path=clif_config['tables_path'],
                file_type=clif_config['file_type'],
                lab_category_units=lab_category_units,
                lab_vital_config=lab_vital_config,
                output_dir=str(output_dir)
            )

            if stats_path:
                print(f"\n{'='*80}")
                print("✅ COLLECTION STATISTICS SUCCESSFUL")
                print(f"{'='*80}")
                return True
            else:
                print(f"\n{'='*80}")
                print("⚠️  COLLECTION STATISTICS GENERATED NO DATA")
                print(f"{'='*80}")
                return False

        except Exception as e:
            print(f"\n{'='*80}")
            print("❌ COLLECTION STATISTICS FAILED")
            print(f"{'='*80}")
            print(f"\nError: {e}")
            print(f"\nTraceback:")
            traceback.print_exc()
            return False

    def generate_report(self, success, validation_passed, total_time):
        """Generate a summary report."""
        report_path = self.project_root / 'output/final/ecdf_execution_report.txt'
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ECDF/BINS GENERATION EXECUTION REPORT\n")
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
            f.write("├── stats/            # Collection statistics CSV\n")
            f.write("│   └── collection_statistics.csv\n")
            f.write("└── unit_mismatches.log\n")

        print(f"\n📊 Execution report saved: {report_path}")

    def run(self, visualize=False):
        """
        Main execution method.

        Args:
            visualize (bool): Whether to generate HTML visualizations

        Returns:
            bool: True if successful, False otherwise
        """
        print("\n" + "="*80)
        print("ECDF AND BINS GENERATION")
        print("="*80 + "\n")

        if visualize:
            print("📊 Visualization mode enabled - will generate HTML plots after ECDF generation\n")

        self.start_time = time.time()

        # Step 1: Validate configuration
        print(f"{'='*80}")
        print("STEP 1: VALIDATING CONFIGURATION")
        print(f"{'='*80}\n")

        if not self.validate_config():
            print("\n❌ Configuration validation failed. Exiting.")
            return False

        # Step 2: Check dependencies
        if not self.check_dependencies():
            print("\n❌ Dependency check failed. Exiting.")
            return False

        # Step 3: Run ECDF generation
        success = self.execute_ecdf_generation()

        # Step 4: Run collection statistics generation
        stats_success = False
        if success:
            stats_success = self.execute_statistics_generation()
            if not stats_success:
                print("\n⚠️  ECDF generation succeeded but statistics generation failed")

        # Step 5: Run visualization (optional)
        if success and visualize:
            viz_success = self.execute_visualization()
            if not viz_success:
                print("\n⚠️  ECDF generation succeeded but visualization failed")

        # Step 6: Validate outputs
        validation_passed = False
        if success:
            validation_passed = self.validate_outputs()

        total_time = time.time() - self.start_time

        # Step 6: Generate report
        self.generate_report(success, validation_passed, total_time)

        # Print summary
        print(f"\n{'='*80}")
        print("EXECUTION SUMMARY")
        print(f"{'='*80}")
        print(f"Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Status: {'✅ SUCCESS' if success else '❌ FAILED'}")
        print(f"Validation: {'✅ PASSED' if validation_passed else '⚠️  INCOMPLETE'}")

        if success and validation_passed:
            print(f"\n{'='*80}")
            print("✅ ALL STEPS COMPLETED SUCCESSFULLY")
            print(f"{'='*80}\n")
            return True
        elif success:
            print(f"\n{'='*80}")
            print("⚠️  ECDF GENERATION SUCCEEDED BUT VALIDATION INCOMPLETE")
            print(f"{'='*80}\n")
            return True
        else:
            print(f"\n{'='*80}")
            print("❌ ECDF GENERATION FAILED")
            print(f"{'='*80}\n")
            return False


def main():
    """Command-line entry point for ECDF generation."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate ECDF and bins data')
    parser.add_argument('--visualize', action='store_true',
                        help='Also generate HTML visualizations')
    args = parser.parse_args()

    runner = ECDFRunner()
    success = runner.run(visualize=args.visualize)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()