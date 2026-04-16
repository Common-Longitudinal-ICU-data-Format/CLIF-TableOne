"""
ECDF Runner Module

Provides execution and validation for ECDF/bins generation with
progress tracking and output validation.
"""

import os
import sys
import json
import traceback
from pathlib import Path
from datetime import datetime

from modules.utils.memory_monitor import MemoryMonitor


class ECDFRunner:
    """Runner for ECDF and bins data generation with validation."""

    def __init__(self, config=None):
        """Initialize ECDF runner with configuration."""
        self.config = config or self.load_config()
        self.memory_monitor = None
        self.project_root = Path(__file__).parent.parent.parent

    def load_config(self):
        """Load configuration from config.json."""
        config_path = Path(__file__).parent.parent.parent / 'config' / 'config.json'

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
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
        from modules.utils.output_paths import (
            CONFIGS, STATS, ecdf_dir, bins_dir, meta_dir, STRATA_NAMES,
        )

        expected_dirs = [
            CONFIGS,
            ecdf_dir(table_type='labs'),
            ecdf_dir(table_type='vitals'),
            ecdf_dir(table_type='respiratory_support'),
            bins_dir(table_type='labs'),
            bins_dir(table_type='vitals'),
            bins_dir(table_type='respiratory_support'),
            STATS,
        ]
        for stratum in STRATA_NAMES:
            expected_dirs.extend([
                ecdf_dir(stratum, 'labs'),
                ecdf_dir(stratum, 'vitals'),
                ecdf_dir(stratum, 'respiratory_support'),
                bins_dir(stratum, 'labs'),
                bins_dir(stratum, 'vitals'),
                bins_dir(stratum, 'respiratory_support'),
            ])

        # Deduplicate: sub-strata now resolve to the same parent directories
        seen = set()
        deduped_dirs = []
        for d in expected_dirs:
            if d not in seen:
                seen.add(d)
                deduped_dirs.append(d)
        expected_dirs = deduped_dirs

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
                # Show last two path parts for clarity (e.g. "overall/labs", "icu/ecdf")
                short = '/'.join(Path(dir_path).parts[-3:])
                print(f"   {short:40s} ({file_count} files)")

        if missing_dirs:
            print(f"\n⚠️  Missing expected directories ({len(missing_dirs)}):")
            for dir_path in missing_dirs:
                short = '/'.join(Path(dir_path).parts[-3:])
                print(f"   {short}")

        # Check for log file
        log_file = meta_dir() / 'unit_mismatches.log'
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
        """Generate interactive HTML distribution viewers."""
        print(f"\n{'='*80}")
        print("GENERATING INTERACTIVE DISTRIBUTION VIEWERS")
        print(f"{'='*80}")
        print(f"Module: modules.ecdf.visualizer")
        print(f"{'='*80}\n")

        try:
            from .visualizer import generate_interactive_html
            from modules.utils.output_paths import (
                OVERALL, cohort_dir, figures_dir,
            )

            # Simple strata: single-panel HTML
            simple = [
                (str(OVERALL), str(figures_dir() / 'distributions.html'), 'Overall'),
                (str(cohort_dir('icu')), str(figures_dir('icu') / 'distributions.html'), 'ICU'),
                (str(cohort_dir('deaths')), str(figures_dir('deaths') / 'distributions.html'), 'Deaths'),
            ]
            for base, out, label in simple:
                if not os.path.isdir(os.path.join(base, 'bins')):
                    continue
                print(f"  Generating {label} viewer → {out}")
                generate_interactive_html(base, out, f'{label} ECDF Distributions')

            # Complex strata: 3-panel (Overall / ICU / No ICU)
            complex_strata = [
                ('advanced_resp', 'Advanced Respiratory Support'),
                ('vaso', 'Vasopressor Support'),
            ]
            for stratum, label in complex_strata:
                base = str(cohort_dir(stratum))
                if not os.path.isdir(os.path.join(base, 'bins')):
                    continue
                out = str(figures_dir(stratum) / 'distributions.html')
                print(f"  Generating {label} viewer (3-panel) → {out}")
                generate_interactive_html(
                    base, out, f'{label} ECDF Distributions',
                    sub_strata_suffixes=['_icu', '_no_icu'],
                    panel_labels=['Overall', 'ICU', 'No ICU'],
                )

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
            from .generator import (
                load_configs, extract_icu_time_windows, discover_lab_category_units,
                load_discharge_times, extract_vaso_event_windows,
                extract_advanced_resp_event_windows, extract_nippv_hfnc_event_windows,
                STRATUM_WINDOW_TYPE,
            )
            from .statistics import compute_collection_statistics
            from modules.utils.clif_loader import ClifDB

            # Load configurations
            clif_config, outlier_config, lab_vital_config = load_configs()

            # Initialise shared DuckDB connection
            db = ClifDB(clif_config['tables_path'], clif_config['file_type'])

            # Extract ICU time windows
            icu_windows = extract_icu_time_windows(db)

            # Compute event-based time windows
            discharge_times = load_discharge_times(db)
            db.register('discharge_times', discharge_times)

            vaso_windows = extract_vaso_event_windows(db)
            resp_windows = extract_advanced_resp_event_windows(db)
            nippv_hfnc_windows = extract_nippv_hfnc_event_windows(db)
            all_time_windows = {
                'icu': icu_windows,
                'vaso': vaso_windows,
                'resp': resp_windows,
                'nippv_hfnc': nippv_hfnc_windows,
            }

            # Discover lab category-unit combinations
            lab_category_units = discover_lab_category_units(db)

            if self.memory_monitor is not None:
                self.memory_monitor.checkpoint("Time Windows Extracted")

            # Compute statistics
            output_dir = self.project_root / 'output' / 'final'
            stats_path = compute_collection_statistics(
                icu_windows=icu_windows,
                db=db,
                lab_category_units=lab_category_units,
                lab_vital_config=lab_vital_config,
                output_dir=str(output_dir)
            )

            if stats_path:
                print(f"\n{'='*80}")
                print("✅ COLLECTION STATISTICS SUCCESSFUL")
                print(f"{'='*80}")
            else:
                print(f"\n{'='*80}")
                print("⚠️  COLLECTION STATISTICS GENERATED NO DATA")
                print(f"{'='*80}")

            if self.memory_monitor is not None:
                self.memory_monitor.checkpoint("Overall Collection Stats Complete")

            # Stratified collection statistics
            try:
                from modules.strata import load_strata_hospitalization_ids

                strata_hosp_ids = load_strata_hospitalization_ids()

                for stratum_name, hosp_ids in strata_hosp_ids.items():
                    window_type = STRATUM_WINDOW_TYPE.get(stratum_name, 'icu')
                    base_windows = all_time_windows[window_type]
                    filtered_windows = base_windows[
                        base_windows['hospitalization_id'].isin(hosp_ids)
                    ]
                    if len(filtered_windows) == 0:
                        print(f"  ⚠️ Skipping collection statistics for {stratum_name}: no time windows")
                        continue

                    print(f"\n  Computing collection statistics for {stratum_name} "
                          f"({len(filtered_windows):,} time windows)...")
                    compute_collection_statistics(
                        icu_windows=filtered_windows,
                        db=db,
                        lab_category_units=lab_category_units,
                        lab_vital_config=lab_vital_config,
                        output_dir=str(output_dir),
                        suffix=f"_{stratum_name.replace('/', '_')}"
                    )
            except FileNotFoundError as e:
                print(f"\n  ⚠️ Skipping stratified collection statistics: {e}")

            if self.memory_monitor is not None:
                self.memory_monitor.checkpoint("Stratified Collection Stats Complete")

            db.close()

            if self.memory_monitor is not None:
                self.memory_monitor.checkpoint("DuckDB Closed")

            return True

        except Exception as e:
            print(f"\n{'='*80}")
            print("❌ COLLECTION STATISTICS FAILED")
            print(f"{'='*80}")
            print(f"\nError: {e}")
            print(f"\nTraceback:")
            traceback.print_exc()
            return False

    def generate_report(self, success, validation_passed):
        """Generate a summary report."""
        from modules.utils.output_paths import meta_dir
        report_path = meta_dir() / 'ecdf_execution_report.txt'
        report_path.parent.mkdir(parents=True, exist_ok=True)

        summary = self.memory_monitor.get_summary() if self.memory_monitor is not None else None

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("ECDF/BINS GENERATION EXECUTION REPORT\n")
            f.write("="*80 + "\n\n")

            f.write(f"Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Status: {'✅ SUCCESS' if success else '❌ FAILED'}\n")
            f.write(f"Validation: {'✅ PASSED' if validation_passed else '⚠️  INCOMPLETE'}\n\n")

            if summary is not None:
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
                        f.write(f"{cp['label']:<40} Memory: {cp['memory_mb']:>8.1f} MB | Peak: {cp['peak_mb']:>8.1f} MB | Time: {cp['elapsed_sec']:>8.1f}s\n")
                    f.write("\n")

            f.write("="*80 + "\n")
            f.write("OUTPUT STRUCTURE\n")
            f.write("="*80 + "\n\n")

            f.write("output/final/\n")
            f.write("├── overall/\n")
            f.write("│   ├── ecdf/{labs,vitals,respiratory_support}/   # Overall ECDF parquets\n")
            f.write("│   └── bins/{labs,vitals,respiratory_support}/   # Overall bin parquets\n")
            f.write("├── strata/\n")
            f.write("│   ├── icu/{ecdf,bins}/...        # Stratified ECDF/bins\n")
            f.write("│   ├── advanced_resp/{ecdf,bins}/...\n")
            f.write("│   ├── vaso/{ecdf,bins}/...\n")
            f.write("│   └── deaths/{ecdf,bins}/...\n")
            f.write("├── stats/                         # Collection statistics CSV\n")
            f.write("│   └── collection_statistics.csv\n")
            f.write("└── meta/\n")
            f.write("    ├── configs/                   # Configuration snapshots\n")
            f.write("    ├── ecdf_execution_report.txt\n")
            f.write("    └── unit_mismatches.log\n")

        print(f"\n📊 Execution report saved: {report_path}")

    def run(self, visualize=False):
        """
        Main execution method.

        Args:
            visualize (bool): Whether to generate interactive HTML viewers

        Returns:
            bool: True if successful, False otherwise
        """
        print("\n" + "="*80)
        print("ECDF AND BINS GENERATION")
        print("="*80 + "\n")

        self.memory_monitor = MemoryMonitor()
        self.memory_monitor.checkpoint("Initialization")

        # Step 1: Validate configuration
        print(f"{'='*80}")
        print("STEP 1: VALIDATING CONFIGURATION")
        print(f"{'='*80}\n")

        if not self.validate_config():
            print("\n❌ Configuration validation failed. Exiting.")
            return False

        self.memory_monitor.checkpoint("Config Validated")

        # Step 2: Check dependencies
        if not self.check_dependencies():
            print("\n❌ Dependency check failed. Exiting.")
            return False

        self.memory_monitor.checkpoint("Dependencies Checked")

        # Step 3: Run ECDF generation
        self.memory_monitor.checkpoint("Script Start")
        success = self.execute_ecdf_generation()
        self.memory_monitor.checkpoint("Overall ECDF/Bins Computed")

        # Step 4: Run collection statistics generation
        stats_success = False
        if success:
            stats_success = self.execute_statistics_generation()
            if not stats_success:
                print("\n⚠️  ECDF generation succeeded but statistics generation failed")

        # Step 5: Run visualization (optional)
        if success and visualize:
            self.memory_monitor.checkpoint("Starting Visualization")
            viz_success = self.execute_visualization()
            self.memory_monitor.checkpoint("Visualization Complete")
            if not viz_success:
                print("\n⚠️  ECDF generation succeeded but visualization failed")

        # Step 6: Validate outputs
        validation_passed = False
        if success:
            validation_passed = self.validate_outputs()

        self.memory_monitor.checkpoint("ECDF Generation Complete")
        total_time = self.memory_monitor.get_summary()['total_time_sec']

        # Step 6: Generate report
        self.generate_report(success, validation_passed)

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