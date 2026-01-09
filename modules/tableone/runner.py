"""
Table One Runner Module

Provides execution with memory monitoring, validation, and reporting
for Table One generation.
"""

import os
import sys
import json
import time
import traceback
import platform
import psutil
from pathlib import Path
from datetime import datetime


# All 18 CLIF input tables
CLIF_TABLES = [
    'patient', 'hospitalization', 'adt', 'labs', 'vitals',
    'medication_admin_continuous', 'medication_admin_intermittent',
    'patient_assessments', 'respiratory_support', 'position',
    'hospital_diagnosis', 'microbiology_culture', 'crrt_therapy',
    'patient_procedures', 'microbiology_susceptibility', 'ecmo_mcs',
    'microbiology_nonculture', 'code_status'
]


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


class TableOneRunner:
    """Runner for Table One generation with memory monitoring and validation."""

    def __init__(self, config=None):
        """Initialize Table One runner with configuration."""
        self.config = config or self.load_config()
        self.memory_monitor = None
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

    def get_system_resources(self):
        """Get system memory info (RAM and VRAM if available)."""
        resources = {
            'total_ram_gb': psutil.virtual_memory().total / (1024**3),
            'available_ram_gb': psutil.virtual_memory().available / (1024**3),
            'gpu_type': None,
            'gpu_name': None,
            'total_vram_gb': None,
            'available_vram_gb': None
        }

        # Try NVIDIA GPU (pynvml) - most common for data centers
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            resources['gpu_type'] = 'NVIDIA'
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            resources['gpu_name'] = gpu_name.decode('utf-8') if isinstance(gpu_name, bytes) else gpu_name
            resources['total_vram_gb'] = info.total / (1024**3)
            resources['available_vram_gb'] = info.free / (1024**3)
            pynvml.nvmlShutdown()
            return resources
        except Exception:
            pass

        # Try AMD GPU (pyamdgpuinfo) - ROCm systems
        try:
            import pyamdgpuinfo
            gpu = pyamdgpuinfo.get_gpu(0)
            resources['gpu_type'] = 'AMD'
            resources['gpu_name'] = gpu.name
            resources['total_vram_gb'] = gpu.memory_info['vram_size'] / (1024**3)
            resources['available_vram_gb'] = None  # Not always available
            return resources
        except Exception:
            pass

        # Check for Apple Silicon (unified memory architecture)
        if platform.system() == 'Darwin' and platform.processor() == 'arm':
            resources['gpu_type'] = 'Apple Silicon'
            resources['gpu_name'] = 'Unified Memory (GPU shares RAM)'
            return resources

        # No GPU detected
        resources['gpu_type'] = 'None detected'
        return resources

    def scan_input_tables(self):
        """Scan all 18 CLIF input tables and collect size metrics."""
        import duckdb

        tables_path = self.config['tables_path']
        file_type = self.config['file_type']
        results = []

        print(f"\n{'='*80}")
        print("SCANNING INPUT TABLES")
        print(f"{'='*80}\n")

        for table_name in CLIF_TABLES:
            file_path = Path(tables_path) / f'clif_{table_name}.{file_type}'
            info = {'name': table_name, 'exists': False}

            if file_path.exists():
                info['exists'] = True
                info['file_size_mb'] = file_path.stat().st_size / (1024**2)

                # Get row/col count via DuckDB (fast, doesn't load full data)
                try:
                    if file_type == 'parquet':
                        df = duckdb.query(f"SELECT * FROM '{file_path}' LIMIT 0").df()
                        info['cols'] = len(df.columns)
                        info['rows'] = duckdb.query(f"SELECT COUNT(*) FROM '{file_path}'").fetchone()[0]
                    else:  # csv
                        df = duckdb.query(f"SELECT * FROM read_csv_auto('{file_path}') LIMIT 0").df()
                        info['cols'] = len(df.columns)
                        info['rows'] = duckdb.query(f"SELECT COUNT(*) FROM read_csv_auto('{file_path}')").fetchone()[0]
                    print(f"   ✓ clif_{table_name}: {info['rows']:,} rows, {info['cols']} cols, {info['file_size_mb']:.2f} MB")
                except Exception as e:
                    info['error'] = str(e)
                    print(f"   ⚠ clif_{table_name}: Error reading - {e}")
            else:
                print(f"   ✗ clif_{table_name}: Not found")

            results.append(info)

        return results

    def validate_outputs(self):
        """Validate that expected output files were created."""
        expected_files = [
            'output/final/tableone/table_one_overall.csv',
            'output/final/tableone/table_one_by_year.csv',
            'output/final/tableone/consort_flow_diagram.png',
            'output/final/tableone/cohort_intersect_upset_plot.png',
            'output/final/tableone/venn_all_4_groups.png',
            'output/final/tableone/medications_summary_stats.csv',
            'output/final/tableone/comorbidities_per_1000_hospitalizations.csv',
            'output/intermediate/final_tableone_df.parquet'
        ]

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
            success = main(memory_monitor=self.memory_monitor)

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

    def generate_report(self, success, validation_passed):
        """Generate a summary report."""
        summary = self.memory_monitor.get_summary()

        report_path = self.project_root / 'output' / 'final' / 'tableone' / 'execution_report.txt'
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

            # System Resources section
            if hasattr(self, 'system_resources') and self.system_resources:
                res = self.system_resources
                f.write("\n" + "="*80 + "\n")
                f.write("SYSTEM RESOURCES (at start)\n")
                f.write("="*80 + "\n\n")

                ram_pct = (res['available_ram_gb'] / res['total_ram_gb']) * 100
                f.write(f"RAM:    Total: {res['total_ram_gb']:.1f} GB | Available: {res['available_ram_gb']:.1f} GB ({ram_pct:.1f}%)\n")

                if res['gpu_type'] == 'NVIDIA' and res['total_vram_gb']:
                    vram_pct = (res['available_vram_gb'] / res['total_vram_gb']) * 100
                    f.write(f"\nGPU:    {res['gpu_name']}\n")
                    f.write(f"VRAM:   Total: {res['total_vram_gb']:.1f} GB | Available: {res['available_vram_gb']:.1f} GB ({vram_pct:.1f}%)\n")
                elif res['gpu_type'] == 'AMD' and res['total_vram_gb']:
                    f.write(f"\nGPU:    {res['gpu_name']} (AMD)\n")
                    f.write(f"VRAM:   Total: {res['total_vram_gb']:.1f} GB\n")
                elif res['gpu_type'] == 'Apple Silicon':
                    f.write(f"\nGPU:    {res['gpu_type']} ({res['gpu_name']})\n")
                else:
                    f.write(f"\nGPU:    {res['gpu_type']}\n")

            # Input Data Size Summary section
            if hasattr(self, 'input_table_sizes') and self.input_table_sizes:
                f.write("\n" + "="*80 + "\n")
                f.write("INPUT DATA SIZE SUMMARY\n")
                f.write("="*80 + "\n\n")

                f.write(f"{'Table':<40} {'Rows':>12} {'Cols':>8} {'File Size':>12}\n")
                f.write("-"*80 + "\n")

                total_rows = 0
                total_size = 0
                tables_found = 0

                for table in self.input_table_sizes:
                    if table['exists']:
                        tables_found += 1
                        rows = table.get('rows', 0)
                        cols = table.get('cols', 0)
                        size = table.get('file_size_mb', 0)
                        total_rows += rows
                        total_size += size
                        f.write(f"clif_{table['name']:<36} {rows:>12,} {cols:>8} {size:>10.2f} MB\n")
                    else:
                        f.write(f"clif_{table['name']:<36} {'Not found':>12} {'-':>8} {'-':>12}\n")

                f.write("-"*80 + "\n")
                f.write(f"{'Total (' + str(tables_found) + ' tables)':<40} {total_rows:>12,} {'---':>8} {total_size:>10.2f} MB\n")

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
        print("TABLE ONE GENERATION WITH MEMORY MONITORING")
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

        # Capture system resources and scan input tables upfront
        self.system_resources = self.get_system_resources()
        self.input_table_sizes = self.scan_input_tables()

        self.memory_monitor.checkpoint("Input Tables Scanned")

        # Step 2: Run the table one generation script
        success = self.execute_table_one_generation()

        # Step 3: Validate outputs
        validation_passed = False
        if success:
            print(f"\n{'='*80}")
            print("STEP 3: VALIDATING OUTPUTS")
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