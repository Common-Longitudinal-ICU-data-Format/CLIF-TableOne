#!/usr/bin/env python3
"""
Create sample test data to demonstrate the CLIF completeness checker.
This creates minimal parquet files with some missing columns and invalid categories
to show how the validation system works.
"""

import pandas as pd
import os
from datetime import datetime, timedelta
import numpy as np

def create_test_data():
    """Create sample test data files."""
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    print("Creating sample test data...")
    
    # Sample patient data - complete and valid
    patient_data = pd.DataFrame({
        'patient_id': ['pat_001', 'pat_002', 'pat_003'],
        'race_category': ['White', 'Black or African American', 'Asian'],
        'ethnicity_category': ['Not Hispanic or Latino', 'Hispanic or Latino', 'Not Hispanic or Latino'],
        'sex_category': ['M', 'F', 'M'],
        'birth_date': pd.to_datetime(['1980-01-15', '1990-05-20', '1975-12-10'])
    })
    patient_data.to_parquet(os.path.join(data_dir, 'patient.parquet'), index=False)
    print("✅ Created patient.parquet (complete)")
    
    # Sample ADT data - missing some required columns and has invalid categories
    adt_data = pd.DataFrame({
        'hospitalization_id': ['hosp_001', 'hosp_002', 'hosp_003'],
        'hospital_id': ['hosp_a', 'hosp_b', 'hosp_a'],
        'hospital_type': ['academic', 'community', 'invalid_type'],  # One invalid value
        'in_dttm': pd.to_datetime(['2023-01-01 08:00:00', '2023-01-02 14:30:00', '2023-01-03 10:15:00']),
        # Missing: out_dttm (required), location_category (required)
        'location_category': ['icu', 'ward', 'invalid_location']  # One invalid value
    })
    adt_data.to_parquet(os.path.join(data_dir, 'adt.parquet'), index=False)
    print("⚠️ Created adt.parquet (missing out_dttm, has invalid categories)")
    
    # Sample Labs data - complete structure but with invalid lab categories
    labs_data = pd.DataFrame({
        'hospitalization_id': ['hosp_001', 'hosp_002', 'hosp_003'],
        'lab_collect_dttm': pd.to_datetime(['2023-01-01 09:00:00', '2023-01-02 15:30:00', '2023-01-03 11:15:00']),
        'lab_order_category': ['abc', 'bmp', 'invalid_lab_order'],  # One invalid value
        'lab_name': ['albumin', 'sodium', 'invalid_lab']
    })
    labs_data.to_parquet(os.path.join(data_dir, 'labs.parquet'), index=False)
    print("⚠️ Created labs.parquet (has invalid categories)")
    
    # Sample Vitals data - missing required columns
    vitals_data = pd.DataFrame({
        'hospitalization_id': ['hosp_001', 'hosp_002', 'hosp_003'],
        # Missing: recorded_dttm (required), vital_category (required)
        'vital_value': [98.6, 120, 80]
    })
    vitals_data.to_parquet(os.path.join(data_dir, 'vitals.parquet'), index=False)
    print("⚠️ Created vitals.parquet (missing required columns)")
    
    # Create empty files for other tables to show "TABLE_NOT_FOUND" vs actual validation
    empty_tables = ['hospitalization', 'medication_admin_continuous', 'patient_assessments', 'position', 'respiratory_support']
    for table in empty_tables:
        # Create empty DataFrame with no columns to simulate missing data structure
        empty_df = pd.DataFrame()
        empty_df.to_parquet(os.path.join(data_dir, f'{table}.parquet'), index=False)
        print(f"📝 Created empty {table}.parquet")
    
    print(f"\n✅ Test data created in {data_dir}/ directory")
    print("This will demonstrate:")
    print("- Complete valid tables (patient)")
    print("- Missing required columns (adt, vitals)")
    print("- Invalid categorical values (adt, labs)")
    print("- Empty table structures (hospitalization, etc.)")

if __name__ == "__main__":
    create_test_data()