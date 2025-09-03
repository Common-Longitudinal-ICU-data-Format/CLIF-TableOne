#!/usr/bin/env python3
"""
CLIF Table Completeness Generator

This script uses the clifpy package to check the completeness status of CLIF tables
and generates a report in the format matching sample.json.

Usage:
    python generate_clif_completeness.py [--config CONFIG_PATH] [--output OUTPUT_PATH]

Example:
    python generate_clif_completeness.py --config config/config.json --output clif_completeness_report.json
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add the code directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'code'))

from clif_completeness_checker import ClifTableOneChecker


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")


def main():
    """Main function to generate CLIF completeness report."""
    parser = argparse.ArgumentParser(description='Generate CLIF table completeness report')
    parser.add_argument('--config', default='config/config.json', 
                       help='Path to configuration JSON file (default: config/config.json)')
    parser.add_argument('--output', default='clif_completeness_report.json',
                       help='Output JSON file path (default: clif_completeness_report.json)')
    parser.add_argument('--tables', nargs='+', 
                       default=['adt', 'hospitalization', 'labs', 'medication_admin_continuous',
                               'patient', 'patient_assessments', 'position', 'respiratory_support', 'vitals'],
                       help='List of tables to check (default: all 9 CLIF tables)')
    
    args = parser.parse_args()
    
    print("=== CLIF Table Completeness Generator ===")
    print(f"Config file: {args.config}")
    print(f"Output file: {args.output}")
    print(f"Tables to check: {', '.join(args.tables)}")
    print()
    
    try:
        # Load configuration
        site_config = load_config(args.config)
        print(f"✅ Loaded configuration for site: {site_config.get('site_name', 'Unknown')}")
        
        # Initialize the checker
        data_dir = site_config.get('tables_path', 'data')
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
        
        # Initialize checker
        checker = ClifTableOneChecker(data_dir, output_dir, site_config)
        
        # Generate report
        report = checker.generate_report(args.tables, args.output)
        
        # Print detailed summary
        print("\n=== CLIF Table Completeness Summary ===")
        print(f"Site: {report['site_name']} ({report['site_id']})")
        print(f"Contact: {report['contact']}")
        print(f"Generated: {report['last_updated']}")
        print()
        
        # Table-by-table summary
        total_issues = 0
        for table_name, status in report['tables'].items():
            missing_cols = status['missing_columns']
            missing_cats = status['missing_categories']
            
            # Count actual issues (exclude error markers)
            col_issues = len([c for c in missing_cols if c not in ['ERROR', 'TABLE_NOT_FOUND']])
            cat_issues = len([c for c in missing_cats if c not in ['ERROR', 'TABLE_NOT_FOUND']])
            
            # Status indicator
            if 'ERROR' in missing_cols or 'TABLE_NOT_FOUND' in missing_cols:
                status_icon = "❌"
            elif col_issues == 0 and cat_issues == 0:
                status_icon = "✅"
            else:
                status_icon = "⚠️"
                total_issues += col_issues + cat_issues
            
            print(f"{status_icon} {table_name.ljust(25)}: {col_issues} missing columns, {cat_issues} missing categories")
            
            # Show details if there are issues
            if status['details'] and status['details'] != f"Table: {table_name}":
                print(f"    Details: {status['details']}")
        
        print()
        if total_issues == 0:
            print("🎉 All tables are complete! No missing columns or categories found.")
        else:
            print(f"📊 Total issues found: {total_issues}")
            print("See the generated JSON file for detailed information.")
        
        print(f"\n📄 Full report saved to: {args.output}")
        return 0
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())