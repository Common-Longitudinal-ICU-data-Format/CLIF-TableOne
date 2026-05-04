"""
Table One Runner Module

Provides execution with memory monitoring, validation, and reporting
for Table One generation.
"""

import os
import sys
import json
import traceback
from pathlib import Path
from datetime import datetime

from modules.utils.memory_monitor import MemoryMonitor


class TableOneRunner:
    """Runner for Table One generation with memory monitoring and validation."""

    def __init__(self, config=None, cohort_mode='critical_illness', force_refresh=False):
        """Initialize Table One runner with configuration.

        Args:
            config: Optional config dict. Loaded from config/config.json if None.
            cohort_mode: 'critical_illness' (default) or 'ward'. Drives which Table
                One pipeline runs and where outputs are written.
            force_refresh: If True, bypass the filtered-CLIF-table cache and
                rebuild from raw source parquets.
        """
        self.config = config or self.load_config()
        self.memory_monitor = None
        self.project_root = Path(__file__).parent.parent.parent
        self.cohort_mode = cohort_mode
        self.force_refresh = force_refresh

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

    def validate_outputs(self):
        """Validate that expected output files were created."""
        # Build cohort-aware expected file list using output_paths helpers so the
        # paths stay in sync with the layout in modules/utils/output_paths.py.
        if self.cohort_mode == 'ward':
            from modules.utils.output_paths import (
                ward_tableone_dir as _t1_dir,
                ward_figures_dir as _fig_dir,
            )
            parquet_name = 'final_tableone_ward_df.parquet'
        else:
            from modules.utils.output_paths import (
                tableone_dir as _t1_dir,
                figures_dir as _fig_dir,
            )
            parquet_name = 'final_tableone_df.parquet'

        # Compute paths relative to project_root for the existence check below.
        def _rel(p):
            return str(p.relative_to(self.project_root))

        expected_files = [
            _rel(_t1_dir() / 'table_one_overall.csv'),
            _rel(_t1_dir() / 'table_one_by_year.csv'),
            _rel(_fig_dir() / 'consort_flow_diagram.png'),
            _rel(_fig_dir() / 'cohort_intersect_upset_plot.png'),
            _rel(_fig_dir() / 'venn_all_4_groups.png'),
            _rel(_t1_dir() / 'comorbidities_per_1000_hospitalizations.csv'),
            f'output/intermediate/{parquet_name}',
        ]
        # medications_summary_stats.csv is only generated in critical-illness mode
        # (the medication-from-ICU plot block is skipped in ward mode per Decision 3).
        if self.cohort_mode != 'ward':
            expected_files.insert(
                5, _rel(_t1_dir() / 'medications_summary_stats.csv')
            )

        missing_files = []
        existing_files = []

        for file_path in expected_files:
            full_path = self.project_root / file_path
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

    def execute_table_one_generation(self):
        """Execute the main table one generation script."""
        print(f"\n{'='*80}")
        print("EXECUTING TABLE ONE GENERATION")
        print(f"{'='*80}")
        print(f"Module: modules.tableone.generator")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")

        self.memory_monitor.checkpoint("Script Start")

        try:
            # Import and execute the main function with memory monitoring
            from .generator import main
            success = main(memory_monitor=self.memory_monitor, cohort_mode=self.cohort_mode, force_refresh=self.force_refresh)

            self.memory_monitor.checkpoint("Script Complete")

            if success:
                print(f"\n{'='*80}")
                print("✅ TABLE ONE GENERATION SUCCESSFUL")
                print(f"{'='*80}")
            else:
                print(f"\n{'='*80}")
                print("⚠️  TABLE ONE GENERATION COMPLETED WITH WARNINGS")
                print(f"{'='*80}")

            return success

        except Exception as e:
            self.memory_monitor.checkpoint("Script Failed")
            print(f"\n{'='*80}")
            print("❌ TABLE ONE GENERATION FAILED")
            print(f"{'='*80}")
            print(f"\nError: {e}")
            print(f"\nTraceback:")
            traceback.print_exc()
            return False

    def execute_suppression(self):
        """Apply small-cell suppression: read raw table_one_*.csv from
        intermediate, write consortium-safe copies to final per
        config/tableone_merge_rules.yaml. Returns True on success.
        """
        print(f"\n{'='*80}")
        print("APPLYING SMALL-CELL SUPPRESSION (intermediate → final)")
        print(f"{'='*80}\n")

        from .suppression import apply_suppression_to_tree, MergeRules
        from modules.utils.output_paths import (
            TABLEONE_INTERMEDIATE,
            tableone_raw_dir, tableone_final_dir,
            ward_tableone_raw_dir, ward_tableone_final_dir,
        )

        rules_path = self.project_root / 'config' / 'tableone_merge_rules.yaml'
        rules = MergeRules.from_yaml(rules_path)
        print(f"   Rules: {rules_path.relative_to(self.project_root)} "
              f"(threshold={rules.suppression.threshold}, token={rules.suppression.token!r}, "
              f"complementary={rules.suppression.apply_complementary})")

        try:
            total = 0

            def _run(label, src, dst):
                nonlocal total
                if not src.exists():
                    return
                written = apply_suppression_to_tree(src, dst, rules)
                print(f"   {label}: {len(written)} CSV(s) → {dst.relative_to(self.project_root)}")
                total += len(written)

            if self.cohort_mode == 'ward':
                _run('ward overall', ward_tableone_raw_dir(), ward_tableone_final_dir())
            else:
                _run('overall', tableone_raw_dir(), tableone_final_dir())
                strata_root = TABLEONE_INTERMEDIATE / 'strata'
                if strata_root.exists():
                    for parent_dir in sorted(strata_root.iterdir()):
                        if parent_dir.is_dir():
                            parent = parent_dir.name
                            _run(f'strata/{parent}',
                                 tableone_raw_dir(stratum=parent),
                                 tableone_final_dir(stratum=parent))

            print(f"\n✅ Suppression complete: {total} file(s) written to final/")
            return True

        except Exception as e:
            print(f"\n❌ Suppression failed: {e}")
            traceback.print_exc()
            print("   Raw intermediate files are preserved for debugging.")
            return False

    def generate_report(self, success, validation_passed):
        """Generate a summary report."""
        summary = self.memory_monitor.get_summary()

        from modules.utils.output_paths import meta_dir
        # Cohort-aware execution report filename so the ward run doesn't overwrite
        # the critical-illness execution report (or vice versa).
        report_filename = (
            'tableone_ward_execution_report.txt'
            if self.cohort_mode == 'ward'
            else 'tableone_execution_report.txt'
        )
        report_path = meta_dir() / report_filename
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
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

    def run(self, profile_mode=False):
        """
        Main execution method.

        Args:
            profile_mode (bool): Whether to enable memory profiling

        Returns:
            bool: True if successful, False otherwise
        """
        print("\n" + "="*80)
        cohort_label = "WARD" if self.cohort_mode == 'ward' else "CRITICAL ILLNESS"
        print(f"TABLE ONE GENERATION WITH MEMORY MONITORING — {cohort_label} COHORT")
        print("="*80 + "\n")

        if profile_mode:
            print("📊 Memory profiling mode enabled")
            print("   Note: This will show detailed memory usage\n")

        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor()
        self.memory_monitor.checkpoint("Initialization")

        # Step 1: Validate configuration
        print(f"\n{'='*80}")
        print("STEP 1: VALIDATING CONFIGURATION")
        print(f"{'='*80}\n")

        if not self.validate_config():
            print("\n❌ Configuration validation failed. Exiting.")
            return False

        self.memory_monitor.checkpoint("Config Validated")

        # Step 2: Run the table one generation script
        success = self.execute_table_one_generation()

        # Step 3: Apply small-cell suppression (intermediate → final)
        if success:
            suppression_ok = self.execute_suppression()
            self.memory_monitor.checkpoint("Suppression Complete")
            if not suppression_ok:
                success = False

        # Step 4: Validate outputs
        validation_passed = False
        if success:
            print(f"\n{'='*80}")
            print("STEP 4: VALIDATING OUTPUTS")
            print(f"{'='*80}")

            validation_passed = self.validate_outputs()

        self.memory_monitor.checkpoint("Validation Complete")

        # Step 4: Generate report
        self.generate_report(success, validation_passed)

        if success and validation_passed:
            print(f"\n{'='*80}")
            print("✅ ALL STEPS COMPLETED SUCCESSFULLY")
            print(f"{'='*80}\n")
            return True
        elif success:
            print(f"\n{'='*80}")
            print("⚠️  GENERATION SUCCEEDED BUT VALIDATION INCOMPLETE")
            print(f"{'='*80}\n")
            return True
        else:
            print(f"\n{'='*80}")
            print("❌ GENERATION FAILED")
            print(f"{'='*80}\n")
            return False


def main():
    """Command-line entry point for Table One generation."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate Table One with memory monitoring')
    parser.add_argument('--profile', action='store_true',
                        help='Enable detailed memory profiling')
    args = parser.parse_args()

    runner = TableOneRunner()
    success = runner.run(profile_mode=args.profile)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()