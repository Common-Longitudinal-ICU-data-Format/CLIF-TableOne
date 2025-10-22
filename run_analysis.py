#!/usr/bin/env python3
"""
CLIF Table One Analysis CLI

Command-line interface for running validation and summary analysis on CLIF tables.

Usage Examples:
    # Single table with both validation and summary
    python run_analysis.py --patient --validate --summary

    # Multiple tables with validation only
    python run_analysis.py --patient --hospitalization --validate

    # All implemented tables
    python run_analysis.py --all --validate --summary

    # Use 1k ICU sample for faster analysis
    python run_analysis.py --labs --validate --summary --sample

    # Specify custom config file
    python run_analysis.py --config path/to/config.json --patient --validate

    # Verbose output for debugging
    python run_analysis.py --patient --validate --summary --verbose

    # Quiet mode (minimal output)
    python run_analysis.py --all --validate --summary --quiet
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add modules to path
sys.path.insert(0, os.path.dirname(__file__))
# Add code directory for MCIDE import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

from modules.cli import CLIAnalysisRunner, ConsoleFormatter


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        print("\nPlease ensure the config file exists and contains:")
        print("""{
    "site_name": "Your Hospital Name",
    "site_id": "YOUR_ID",
    "tables_path": "/path/to/clif/tables",
    "filetype": "parquet",
    "timezone": "UTC",
    "output_dir": "output"
}""")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in configuration file: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='CLIF Table One Analysis - CLI Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --patient --validate --summary
  %(prog)s --patient --hospitalization --validate
  %(prog)s --all --validate --summary --verbose
  %(prog)s --config custom/config.json --patient --validate
        """
    )

    # Configuration
    parser.add_argument('--config', default='config/config.json',
                       help='Path to configuration JSON file (default: config/config.json)')
    parser.add_argument('--output-dir', help='Override output directory from config')

    # Table selection
    table_group = parser.add_argument_group('Table Selection')
    table_group.add_argument('--patient', action='store_true',
                            help='Analyze patient table')
    table_group.add_argument('--hospitalization', action='store_true',
                            help='Analyze hospitalization table')
    table_group.add_argument('--adt', action='store_true',
                            help='Analyze ADT table')
    table_group.add_argument('--code_status', action='store_true',
                            help='Analyze code status table')
    table_group.add_argument('--crrt_therapy', action='store_true',
                            help='Analyze CRRT therapy table')
    table_group.add_argument('--ecmo_mcs', action='store_true',
                            help='Analyze ECMO/MCS table')
    table_group.add_argument('--hospital_diagnosis', action='store_true',
                            help='Analyze hospital diagnosis table')
    table_group.add_argument('--labs', action='store_true',
                            help='Analyze labs table')
    table_group.add_argument('--medication_admin_continuous', action='store_true',
                            help='Analyze medication admin continuous table')
    table_group.add_argument('--medication_admin_intermittent', action='store_true',
                            help='Analyze medication admin intermittent table')
    table_group.add_argument('--microbiology_culture', action='store_true',
                            help='Analyze microbiology culture table')
    table_group.add_argument('--microbiology_nonculture', action='store_true',
                            help='Analyze microbiology non-culture table')
    table_group.add_argument('--microbiology_susceptibility', action='store_true',
                            help='Analyze microbiology susceptibility table')
    table_group.add_argument('--patient_assessments', action='store_true',
                            help='Analyze patient assessments table')
    table_group.add_argument('--patient_procedures', action='store_true',
                            help='Analyze patient procedures table')
    table_group.add_argument('--position', action='store_true',
                            help='Analyze position table')
    table_group.add_argument('--respiratory_support', action='store_true',
                            help='Analyze respiratory support table')
    table_group.add_argument('--vitals', action='store_true',
                            help='Analyze vitals table')
    table_group.add_argument('--all', action='store_true',
                            help='Analyze all implemented tables')

    # Operations
    ops_group = parser.add_argument_group('Operations')
    ops_group.add_argument('--validate', action='store_true',
                          help='Run validation using clifpy')
    ops_group.add_argument('--summary', action='store_true',
                          help='Generate summary statistics')

    # Output control
    output_group = parser.add_argument_group('Output Control')
    output_group.add_argument('--verbose', '-v', action='store_true',
                             help='Enable verbose output')
    output_group.add_argument('--quiet', '-q', action='store_true',
                             help='Minimize output (only show errors and final summary)')
    output_group.add_argument('--no-pdf', action='store_true',
                             help='Disable PDF report generation (only generate JSON)')
    output_group.add_argument('--sample', action='store_true',
                             help='Use 1k ICU sample for faster analysis (requires sample file from ADT analysis)')

    args = parser.parse_args()

    # Validate arguments
    has_table = (args.patient or args.hospitalization or args.adt or args.code_status or args.crrt_therapy or
                 args.ecmo_mcs or args.hospital_diagnosis or args.labs or args.medication_admin_continuous or
                 args.medication_admin_intermittent or args.microbiology_culture or args.microbiology_nonculture or
                 args.microbiology_susceptibility or args.patient_assessments or args.patient_procedures or
                 args.position or args.respiratory_support or args.vitals or args.all)
    if not has_table:
        parser.error('Please specify at least one table or use --all for all tables')

    if not (args.validate or args.summary):
        parser.error('Please specify at least one operation: --validate and/or --summary')

    # Determine which tables to analyze
    tables = []
    if args.all:
        tables = ['patient', 'hospitalization', 'adt', 'code_status', 'crrt_therapy', 'ecmo_mcs',
                  'hospital_diagnosis', 'labs', 'medication_admin_continuous', 'medication_admin_intermittent',
                  'microbiology_culture', 'microbiology_nonculture', 'microbiology_susceptibility',
                  'patient_assessments', 'patient_procedures', 'position', 'respiratory_support', 'vitals']
    else:
        if args.patient:
            tables.append('patient')
        if args.hospitalization:
            tables.append('hospitalization')
        if args.adt:
            tables.append('adt')
        if args.code_status:
            tables.append('code_status')
        if args.crrt_therapy:
            tables.append('crrt_therapy')
        if args.ecmo_mcs:
            tables.append('ecmo_mcs')
        if args.hospital_diagnosis:
            tables.append('hospital_diagnosis')
        if args.labs:
            tables.append('labs')
        if args.medication_admin_continuous:
            tables.append('medication_admin_continuous')
        if args.medication_admin_intermittent:
            tables.append('medication_admin_intermittent')
        if args.microbiology_culture:
            tables.append('microbiology_culture')
        if args.microbiology_nonculture:
            tables.append('microbiology_nonculture')
        if args.microbiology_susceptibility:
            tables.append('microbiology_susceptibility')
        if args.patient_assessments:
            tables.append('patient_assessments')
        if args.patient_procedures:
            tables.append('patient_procedures')
        if args.position:
            tables.append('position')
        if args.respiratory_support:
            tables.append('respiratory_support')
        if args.vitals:
            tables.append('vitals')

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        sys.exit(1)

    # Override output directory if specified
    if args.output_dir:
        config['output_dir'] = args.output_dir

    # Validate configuration
    if 'tables_path' not in config:
        print("❌ Error: 'tables_path' not found in configuration file")
        sys.exit(1)

    data_path = config.get('tables_path', '')
    if not os.path.exists(data_path):
        print(f"❌ Error: Data directory not found: {data_path}")
        print("Please check your configuration and ensure the tables_path is correct.")
        sys.exit(1)

    # Initialize runner
    generate_pdf = not args.no_pdf
    runner = CLIAnalysisRunner(config, verbose=args.verbose, quiet=args.quiet, generate_pdf=generate_pdf, use_sample=args.sample)

    # Run analysis
    try:
        results = runner.run_analysis(tables, args.validate, args.summary)

        # Exit with appropriate code
        if results['total_failed'] == 0:
            sys.exit(0)  # Success
        elif results['total_success'] > 0:
            sys.exit(2)  # Partial success
        else:
            sys.exit(1)  # Complete failure

    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
