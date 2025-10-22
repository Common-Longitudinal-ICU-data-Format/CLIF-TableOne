#!/usr/bin/env python3
"""
Entry point for Table One generation with memory monitoring.

This script provides a clean interface for running Table One generation
with comprehensive memory monitoring and validation.

Usage:
    uv run run_tableone.py
    uv run run_tableone.py --profile  # For detailed memory profiling
"""

import sys
from modules.tableone.runner import TableOneRunner


def main():
    """Main entry point for Table One generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate Table One statistics with memory monitoring'
    )
    parser.add_argument(
        '--profile',
        action='store_true',
        help='Enable detailed memory profiling'
    )
    args = parser.parse_args()

    try:
        # Initialize and run the Table One generator
        runner = TableOneRunner()
        success = runner.run(profile_mode=args.profile)

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()