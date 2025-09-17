import os
import json
import yaml
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import importlib

from clifpy.clif_orchestrator import ClifOrchestrator
from clifpy.tables.adt import Adt
from clifpy.tables.hospitalization import Hospitalization
from clifpy.tables.labs import Labs
from clifpy.tables.medication_admin_continuous import MedicationAdminContinuous
from clifpy.tables.patient import Patient
from clifpy.tables.patient_assessments import PatientAssessments
from clifpy.tables.position import Position
from clifpy.tables.respiratory_support import RespiratorySupport
from clifpy.tables.vitals import Vitals


class ClifTableOneChecker:
    """
    A class to check CLIF table completeness using the clifpy package.
    Generates completeness reports in the format matching sample.json.
    """
    
    def __init__(self, data_dir: str, output_dir: str, site_config: Dict[str, Any]):
        """
        Initialize the CLIF TableOne completeness checker.
        
        Args:
            data_dir: Directory containing CLIF data files
            output_dir: Directory for output files
            site_config: Site configuration dictionary
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.site_config = site_config
        
        # Initialize clifpy orchestrator
        self.orchestrator = ClifOrchestrator(
            data_directory=data_dir,
            filetype=site_config.get('file_type'),
            timezone=site_config.get('timezone'),
            output_directory=output_dir
        )
        
        # Mapping of table names to clifpy table classes
        self.table_mapping = {
            'adt': Adt,
            'hospitalization': Hospitalization,
            'labs': Labs,
            'medication_admin_continuous': MedicationAdminContinuous,
            'patient': Patient,
            'patient_assessments': PatientAssessments,
            'position': Position,
            'respiratory_support': RespiratorySupport,
            'vitals': Vitals
        }
    
    def validate_table_with_clifpy(self, table_name: str) -> Dict[str, Any]:
        """
        Use clifpy's built-in validation to check table completeness.

        Args:
            table_name: Name of the table to validate

        Returns:
            Dictionary with validation results in the expected format
        """
        table_class = self.table_mapping.get(table_name)
        if not table_class:
            return {
                'missing_columns': ['UNKNOWN_TABLE'],
                'missing_categories': [],
                'details': f'Unknown table type: {table_name}'
            }

        try:
            # Use from_file method to load the table data
            table_instance = table_class.from_file(
                self.data_dir,
                self.site_config.get('file_type', 'parquet')
            )

            # Run validation if the method exists
            if hasattr(table_instance, 'validate'):
                table_instance.validate()

            # Check if table loaded successfully
            if not hasattr(table_instance, 'df') or table_instance.df is None:
                return {
                    'missing_columns': ['TABLE_NOT_FOUND'],
                    'missing_categories': [],
                    'details': f'Data file for {table_name} not found or could not be loaded'
                }

            # Use clifpy's validation - check if table is valid
            is_valid = table_instance.isvalid() if hasattr(table_instance, 'isvalid') else True

            # Get validation results from clifpy
            missing_columns = []
            missing_categories = []
            details_parts = []

            # Add basic table info
            details_parts.append(f"Table: {table_name}")
            details_parts.append(f"Rows: {len(table_instance.df):,}")
            details_parts.append(f"Columns: {len(table_instance.df.columns)}")

            # Check for validation errors if available
            if hasattr(table_instance, 'errors') and table_instance.errors:
                # Parse clifpy validation errors into our format
                for error in table_instance.errors:
                    error_type = error.get('type', '')

                    if error_type == 'missing_columns':
                        # Completely missing columns (not in data structure)
                        missing_cols = error.get('columns', [])
                        if missing_cols:
                            missing_columns.extend(missing_cols)
                        details_parts.append(f"Validation error ({error_type}): {error}")

                    elif error_type == 'null_values':
                        # Missing required columns (all null)
                        column = error.get('column')
                        count = error.get('count', 0)
                        if column and count == len(table_instance.df):
                            missing_columns.append(column)
                        details_parts.append(f"Column '{column}' has {count} null values")

                    elif error_type in ['invalid_category', 'invalid_categorical_values']:
                        # Invalid categorical values
                        column = error.get('column')
                        if 'invalid_values' in error:
                            invalid_vals = error['invalid_values']
                            missing_categories.extend(invalid_vals)
                            details_parts.append(f"Column '{column}' has invalid values: {', '.join(map(str, invalid_vals))}")
                        elif 'values' in error:
                            vals = error['values']
                            missing_categories.extend(vals)
                            details_parts.append(f"Column '{column}' has invalid values: {', '.join(map(str, vals))}")

                    else:
                        # Other validation issues
                        details_parts.append(f"Validation error ({error_type}): {error}")

            # If table is not valid but no specific errors categorized, add general message
            if not is_valid and not missing_columns and not missing_categories:
                details_parts.append("Table validation failed - check data quality")
                missing_columns.append('VALIDATION_FAILED')

            return {
                'missing_columns': missing_columns,
                'missing_categories': missing_categories,
                'details': '; '.join(details_parts) if details_parts else ""
            }

        except FileNotFoundError:
            return {
                'missing_columns': ['TABLE_NOT_FOUND'],
                'missing_categories': [],
                'details': f'Data file for {table_name} not found'
            }
        except Exception as e:
            return {
                'missing_columns': ['ERROR'],
                'missing_categories': ['ERROR'],
                'details': f'Validation failed: {str(e)}'
            }
    
    
    
    def check_completeness_status(self, table_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Check completeness status for multiple tables using clifpy validation.

        Args:
            table_names: List of table names to check

        Returns:
            Dictionary mapping table names to their completeness status
        """
        completeness_status = {}

        for table_name in table_names:
            print(f"Checking completeness for table: {table_name}")

            # Use clifpy validation instead of manual parsing
            table_status = self.validate_table_with_clifpy(table_name)
            completeness_status[table_name] = table_status

        return completeness_status
    
    def generate_report(self, table_names: List[str], output_file: str) -> Dict[str, Any]:
        """
        Generate a completeness report in YAML format (more human-readable than JSON).
        
        Args:
            table_names: List of table names to check
            output_file: Path to output YAML file
            
        Returns:
            Dictionary containing the complete report
        """
        print("Generating CLIF table completeness report...")
        
        # Check completeness for all tables
        completeness_status = self.check_completeness_status(table_names)
        
        # Generate report in sample.json format structure
        report = {
            'site_name': self.site_config.get('site_name', 'Unknown Site'),
            'site_id': self.site_config.get('site_id', 'unknown_site'),
            'last_updated': datetime.now().isoformat() + 'Z',
            'contact': self.site_config.get('contact', ''),
            'tables': completeness_status
        }
        
        # Save to file as YAML with improved formatting
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            yaml.dump(report, f, 
                     default_flow_style=False,  # Multi-line format
                     indent=2,                   # 2-space indentation
                     allow_unicode=True,         # Allow unicode characters
                     width=float('inf'),         # Don't wrap long lines
                     sort_keys=False)            # Preserve order
        
        # Comment out JSON generation but keep code for reference
        # json_file = output_file.replace('.yaml', '.json')
        # with open(json_file, 'w') as f:
        #     json.dump(report, f, indent=2)
        
        print(f"Completeness report saved to: {output_file}")
        return report


def main():
    """Main function to run the completeness checker."""
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.json')
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            site_config = json.load(f)
    else:
        # Default configuration
        site_config = {
            'site_name': 'Test Site',
            'site_id': 'test_site',
            'contact': 'test@example.com',
            'tables_path': '../data',
            'file_type': 'parquet'
        }
    
    # Define target tables
    target_tables = [
        'adt', 'hospitalization', 'labs', 'medication_admin_continuous',
        'patient', 'patient_assessments', 'position', 'respiratory_support', 'vitals'
    ]
    
    # Initialize checker
    data_dir = site_config.get('tables_path', '../data')
    output_dir = '../output'
    
    checker = ClifTableOneChecker(data_dir, output_dir, site_config)
    
    # Generate report
    output_file = '../output/clif_completeness_report.json'
    report = checker.generate_report(target_tables, output_file)
    
    # Print summary
    print("\n=== CLIF Table Completeness Summary ===")
    for table_name, status in report['tables'].items():
        missing_cols = len(status['missing_columns'])
        missing_cats = len(status['missing_categories'])
        print(f"{table_name}: {missing_cols} missing columns, {missing_cats} missing categories")


if __name__ == "__main__":
    main()