#!/usr/bin/env python3
"""
Entry point for Ward Table One generation with memory monitoring.

Generates a Table One whose cohort is every adult hospitalization that touched
a ward at any point (location_category == 'ward'). Outputs go to
output/final/overall_ward/{tableone,figures,...}/ and the intermediate parquet
to output/intermediate/final_tableone_ward_df.parquet, leaving the
critical-illness pipeline outputs untouched.

Usage:
    uv run run_tableone_ward.py
    uv run run_tableone_ward.py --profile  # For detailed memory profiling
"""

import sys
from modules.tableone.runner import TableOneRunner


def main():
    """Main entry point for Ward Table One generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate Ward Table One statistics with memory monitoring'
    )
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable detailed memory profiling'
    )
    parser.add_argument(
        '--force-refresh',
        action='store_true',
        help='Bypass filtered-CLIF-table cache and rebuild from raw source parquets'
    )
    args = parser.parse_args()

    try:
        # Initialize and run the Ward Table One generator
        runner = TableOneRunner(cohort_mode='ward', force_refresh=args.force_refresh)
        success = runner.run(profile_mode=args.profile)

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
