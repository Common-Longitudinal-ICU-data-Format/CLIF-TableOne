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
            filetype=site_config.get('file_type', 'parquet'),
            timezone=site_config.get('timezone', 'UTC'),
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
    
    def get_clifpy_schemas_path(self) -> str:
        """Get the path to clifpy schemas directory."""
        try:
            import clifpy
            clifpy_path = os.path.dirname(clifpy.__file__)
            return os.path.join(clifpy_path, 'schemas')
        except ImportError:
            raise ImportError("clifpy package not found. Please install it first.")
    
    def parse_schema(self, table_name: str) -> Tuple[List[str], Dict[str, List[str]]]:
        """
        Parse a CLIF schema YAML file to extract required columns and categorical constraints.
        
        Args:
            table_name: Name of the table (e.g., 'adt', 'patient')
            
        Returns:
            Tuple of (required_columns, categorical_constraints)
        """
        schemas_path = self.get_clifpy_schemas_path()
        schema_file = os.path.join(schemas_path, f"{table_name}_schema.yaml")
        
        if not os.path.exists(schema_file):
            raise FileNotFoundError(f"Schema file not found: {schema_file}")
        
        with open(schema_file, 'r') as file:
            schema = yaml.safe_load(file)
        
        required_columns = []
        categorical_constraints = {}
        
        columns = schema.get('columns', [])
        for col_spec in columns:
            col_name = col_spec.get('name')
            if not col_name:
                continue
                
            # Check if column is required
            if col_spec.get('required', False):
                required_columns.append(col_name)
            
            # Check if column is categorical with permissible values
            if col_spec.get('is_category_column', False):
                permissible_values = col_spec.get('permissible_values', [])
                if permissible_values:
                    categorical_constraints[col_name] = permissible_values
        
        return required_columns, categorical_constraints
    
    def load_table_data(self, table_name: str) -> Optional[pd.DataFrame]:
        """
        Load data for a specific table.
        
        Args:
            table_name: Name of the table to load
            
        Returns:
            DataFrame with table data or None if file doesn't exist
        """
        file_type = self.site_config.get('file_type', 'parquet')
        
        # Try different filename patterns
        possible_names = [
            f"{table_name}.{file_type}",           # table_name.parquet
            f"clif_{table_name}.{file_type}",      # clif_table_name.parquet
        ]
        
        file_path = None
        for file_name in possible_names:
            potential_path = os.path.join(self.data_dir, file_name)
            if os.path.exists(potential_path):
                file_path = potential_path
                break
        
        if file_path is None:
            return None
        
        try:
            if file_type == 'parquet':
                df = pd.read_parquet(file_path)
            elif file_type == 'csv':
                df = pd.read_csv(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            return df
        except Exception as e:
            print(f"Error loading {table_name}: {str(e)}")
            return None
    
    def validate_table_completeness(self, table_name: str, data_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate table completeness against schema requirements.
        
        Args:
            table_name: Name of the table
            data_df: DataFrame containing the table data
            
        Returns:
            Dictionary with completeness status
        """
        try:
            # Parse schema to get requirements
            required_columns, categorical_constraints = self.parse_schema(table_name)
            
            # Check missing required columns
            missing_columns = []
            for col in required_columns:
                if col not in data_df.columns:
                    missing_columns.append(col)
            
            # Check missing categorical values
            missing_categories = []
            for col, allowed_values in categorical_constraints.items():
                if col in data_df.columns:
                    # Get unique non-null values from the data
                    actual_values = set(data_df[col].dropna().unique())
                    allowed_values_set = set(allowed_values)
                    
                    # Check if all actual values are in the allowed set
                    if not actual_values.issubset(allowed_values_set):
                        # Find values that are not allowed
                        invalid_values = actual_values - allowed_values_set
                        if invalid_values:
                            # Add the actual invalid values to missing_categories
                            missing_categories.extend(list(invalid_values))
                else:
                    # Column doesn't exist but has categorical constraints
                    # In this case, we consider all allowed values as "missing"
                    if col not in missing_columns:  # Don't double-count if column is already missing
                        missing_categories.extend(allowed_values)
            
            # Generate details - pass the original constraints for detail generation
            details = self.generate_details(table_name, data_df, missing_columns, missing_categories, categorical_constraints)
            
            return {
                'missing_columns': missing_columns,
                'missing_categories': missing_categories,
                'details': details
            }
            
        except Exception as e:
            return {
                'missing_columns': ['ERROR'],
                'missing_categories': ['ERROR'],
                'details': f"Schema validation failed: {str(e)}"
            }
    
    def generate_details(self, table_name: str, data_df: pd.DataFrame, 
                        missing_columns: List[str], missing_categories: List[str],
                        categorical_constraints: Dict[str, List[str]]) -> str:
        """
        Generate detailed information about validation results.
        
        Args:
            table_name: Name of the table
            data_df: DataFrame containing the table data
            missing_columns: List of missing required columns
            missing_categories: List of invalid categorical values (actual values, not column names)
            categorical_constraints: Dictionary of categorical constraints
            
        Returns:
            Detailed description string
        """
        details = []
        
        # Add data summary
        details.append(f"Table: {table_name}")
        details.append(f"Rows: {len(data_df):,}")
        details.append(f"Columns: {len(data_df.columns)}")
        
        # Add missing columns details
        if missing_columns:
            details.append(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Add categorical issues details
        if missing_categories:
            # Find which columns have invalid values and what those values are
            for col, allowed_values in categorical_constraints.items():
                if col in data_df.columns:
                    actual_values = set(data_df[col].dropna().unique())
                    allowed_values_set = set(allowed_values)
                    invalid_values = actual_values - allowed_values_set
                    
                    # Check if any of the missing_categories are in the invalid_values for this column
                    column_invalid_values = invalid_values.intersection(set(missing_categories))
                    if column_invalid_values:
                        details.append(f"Column '{col}' has invalid values: {', '.join(map(str, column_invalid_values))}")
        
        return '; '.join(details) if details else ""
    
    def check_completeness_status(self, table_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Check completeness status for multiple tables.
        
        Args:
            table_names: List of table names to check
            
        Returns:
            Dictionary mapping table names to their completeness status
        """
        completeness_status = {}
        
        for table_name in table_names:
            print(f"Checking completeness for table: {table_name}")
            
            # Load table data
            data_df = self.load_table_data(table_name)
            
            if data_df is None:
                completeness_status[table_name] = {
                    'missing_columns': ['TABLE_NOT_FOUND'],
                    'missing_categories': [],
                    'details': f"Data file for {table_name} not found"
                }
                continue
            
            # Validate completeness
            table_status = self.validate_table_completeness(table_name, data_df)
            completeness_status[table_name] = table_status
        
        return completeness_status
    
    def generate_report(self, table_names: List[str], output_file: str) -> Dict[str, Any]:
        """
        Generate a completeness report in the sample.json format.
        
        Args:
            table_names: List of table names to check
            output_file: Path to output JSON file
            
        Returns:
            Dictionary containing the complete report
        """
        print("Generating CLIF table completeness report...")
        
        # Check completeness for all tables
        completeness_status = self.check_completeness_status(table_names)
        
        # Generate report in sample.json format
        report = {
            'site_name': self.site_config.get('site_name', 'Unknown Site'),
            'site_id': self.site_config.get('site_id', 'unknown_site'),
            'last_updated': datetime.now().isoformat() + 'Z',
            'contact': self.site_config.get('contact', ''),
            'tables': completeness_status
        }
        
        # Save to file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
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