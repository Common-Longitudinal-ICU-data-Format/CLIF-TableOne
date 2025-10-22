#!/usr/bin/env python3
"""
Entry point for ECDF and bins generation.

This script provides a clean interface for generating ECDF
(Empirical Cumulative Distribution Function) and bin data
for labs, vitals, and respiratory support measurements.

Usage:
    uv run run_ecdf.py
    uv run run_ecdf.py --visualize  # Also generate HTML visualizations
"""

import sys
from modules.ecdf.runner import ECDFRunner


def main():
    """Main entry point for ECDF generation."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate ECDF and bins data for labs, vitals, and respiratory support'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Also generate HTML visualizations'
    )
    args = parser.parse_args()

    try:
        # Initialize and run the ECDF generator
        runner = ECDFRunner()
        success = runner.run(visualize=args.visualize)

        # Exit with appropriate code
        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()