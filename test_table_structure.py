#!/usr/bin/env python3
"""Test script to verify table structure before implementation."""

import sys
import os

# Add modules to path
sys.path.insert(0, os.path.dirname(__file__))

def test_table_structure(table_name, table_class_name):
    """Load a table and print its structure."""
    try:
        # Dynamic import
        module = __import__(f'clifpy.tables.{table_name}', fromlist=[table_class_name])
        TableClass = getattr(module, table_class_name)

        # Load from config
        import json
        with open('config/config.json', 'r') as f:
            config = json.load(f)

        data_dir = config['tables_path']
        filetype = config.get('filetype', 'parquet')
        timezone = config.get('timezone', 'UTC')

        print(f"\n{'='*60}")
        print(f"Testing: {table_name}")
        print(f"{'='*60}")

        # Try to load table
        table = TableClass.from_file(
            data_directory=data_dir,
            filetype=filetype,
            timezone=timezone,
            output_directory='output/test'
        )

        if table.df is None:
            print("❌ Table loaded but df is None")
            return False

        print(f"✅ Table loaded successfully!")
        print(f"\n📊 Basic Info:")
        print(f"  - Rows: {len(table.df):,}")
        print(f"  - Columns: {len(table.df.columns)}")

        print(f"\n📋 Columns:")
        for col in table.df.columns:
            print(f"  - {col}: {table.df[col].dtype}")

        print(f"\n🔍 Sample Data:")
        print(table.df.head(2))

        print(f"\n✅ SUCCESS: {table_name} structure verified\n")
        return True

    except FileNotFoundError:
        print(f"⚠️  File not found for {table_name}")
        return False
    except Exception as e:
        print(f"❌ Error loading {table_name}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test hospital_diagnosis
    test_table_structure('hospital_diagnosis', 'HospitalDiagnosis')
