#!/usr/bin/env python3
"""
CLIF Table Report Card Generator

This script uses the clifpy package to validate CLIF tables and generates
a user-friendly report card in PDF format (or text format as fallback).

Usage:
    python generate_clif_report_card.py [--config CONFIG_PATH] [--output OUTPUT_PATH] [--format FORMAT]

Example:
    python generate_clif_report_card.py --config config/config.json --output clif_report_card.pdf --format pdf
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add the code directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

from clif_report_card import ClifReportCardGenerator


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")


def main():
    """Main function to generate CLIF report card."""
    parser = argparse.ArgumentParser(description='Generate CLIF table validation report card')
    parser.add_argument('--config', default='config/config.json',
                       help='Path to configuration JSON file (default: config/config.json)')
    parser.add_argument('--output', default='clif_report_card.pdf',
                       help='Output file path (default: clif_report_card.pdf)')
    parser.add_argument('--format', choices=['pdf', 'txt'], default='pdf',
                       help='Output format: pdf or txt (default: pdf)')
    parser.add_argument('--tables', nargs='+',
                       default=['adt', 'hospitalization', 'labs', 'medication_admin_continuous',
                               'patient', 'patient_assessments', 'position', 'respiratory_support', 'vitals'],
                       help='List of tables to check (default: all 9 CLIF tables)')

    args = parser.parse_args()

    print("="*60)
    print("🏥 CLIF TABLE VALIDATION REPORT CARD GENERATOR")
    print("="*60)
    print(f"Config file: {args.config}")
    print(f"Output file: {args.output}")
    print(f"Output format: {args.format.upper()}")
    print(f"Tables to check: {', '.join(args.tables)}")
    print()

    try:
        # Load configuration
        site_config = load_config(args.config)
        print(f"✅ Loaded configuration for site: {site_config.get('site_name', 'Unknown')}")

        # Initialize the generator
        data_dir = site_config.get('tables_path', 'data')

        # If only filename provided, place it in output/final directory
        if not os.path.dirname(args.output):
            args.output = os.path.join('output', 'final', args.output)

        output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else '.'

        # Ensure paths are absolute
        if not os.path.isabs(data_dir):
            data_dir = os.path.abspath(data_dir)
        if not os.path.isabs(args.output):
            args.output = os.path.abspath(args.output)

        print(f"📁 Data directory: {data_dir}")
        print(f"📄 Output file: {args.output}")
        print()

        # Check if data directory exists
        if not os.path.exists(data_dir):
            print(f"❌ Error: Data directory not found: {data_dir}")
            print("Please check your configuration and ensure the tables_path is correct.")
            return 1

        # Initialize generator
        generator = ClifReportCardGenerator(data_dir, output_dir, site_config)

        # Generate report based on format
        if args.format == 'pdf':
            try:
                report_data = generator.generate_pdf_report_card(args.tables, args.output)
                print(f"\n🎉 PDF Report Card Generated Successfully!")
            except ImportError:
                print(f"\n⚠️  PDF generation not available. Falling back to text format...")
                # Change extension to .txt
                txt_output = args.output.replace('.pdf', '.txt')
                report_data = generator.generate_simple_text_report(args.tables, txt_output)
                args.output = txt_output
                print(f"📝 Text Report Card Generated!")
                print(f"💡 Install 'reportlab' for PDF generation: pip install reportlab")
        else:
            report_data = generator.generate_simple_text_report(args.tables, args.output)
            print(f"\n📝 Text Report Card Generated Successfully!")

        # Print summary
        print(f"\n📋 VALIDATION SUMMARY")
        print(f"{'='*40}")
        print(f"Site: {report_data['site_name']}")
        print(f"Generated: {report_data['timestamp']}")
        print()

        # Overall status with visual indicator
        status_display = {
            'complete': '✅ COMPLETE - All tables validated successfully!',
            'partial': '⚠️  PARTIAL - Some issues found but data is usable',
            'noinformation': '❓ NO INFORMATION - Critical issues prevent validation'
        }

        # Table-by-table summary with counts
        results = report_data['table_results']
        complete_count = sum(1 for r in results.values() if r['status'] == 'complete')
        partial_count = sum(1 for r in results.values() if r['status'] in ['partial', 'incomplete'])
        missing_count = sum(1 for r in results.values() if r['status'] in ['missing', 'error'])
        total_rows = sum(r.get('data_info', {}).get('row_count', 0) for r in results.values())

        print(f"📊 Table Summary:")
        print(f"   ✅ Complete: {complete_count} tables")
        print(f"   ⚠️  Partial:   {partial_count} tables")
        print(f"   ❌ Incomplete:   {missing_count} tables")
        print()

        # Quick status overview
        print(f"🔍 Quick Status Overview:")
        for table_name, result in results.items():
            status = result['status']
            icons = {'complete': '✅', 'partial': '⚠️', 'incomplete': '❌'}
            icon = icons.get(status, '❓')

            data_info = result.get('data_info', {})
            rows = data_info.get('row_count', 0)
            row_info = f" ({rows:,} rows)" if rows > 0 else ""

            table_display = generator.table_display_names.get(table_name, table_name.title())
            print(f"   {icon} {table_display:<35} {status.upper()}{row_info}")

        print(f"\n📁 Full report available at: {args.output}")
        return 0

    except FileNotFoundError as e:
        print(f"❌ Configuration Error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        import traceback
        print(f"\n🔍 Debug Information:")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())