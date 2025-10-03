import os
import json
import logging
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd

# Add local clifpy development version to path
sys.path.insert(0, '/Users/dema/WD/clifpy')

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("WARNING: reportlab not installed. Install with: pip install reportlab")

from clifpy.clif_orchestrator import ClifOrchestrator
from clifpy.utils import validator
from clifpy.tables.adt import Adt
from clifpy.tables.code_status import CodeStatus
from clifpy.tables.crrt_therapy import CrrtTherapy
from clifpy.tables.ecmo_mcs import EcmoMcs
from clifpy.tables.hospital_diagnosis import HospitalDiagnosis
from clifpy.tables.hospitalization import Hospitalization
from clifpy.tables.labs import Labs
from clifpy.tables.medication_admin_continuous import MedicationAdminContinuous
from clifpy.tables.medication_admin_intermittent import MedicationAdminIntermittent
from clifpy.tables.microbiology_culture import MicrobiologyCulture
from clifpy.tables.microbiology_nonculture import MicrobiologyNonculture
from clifpy.tables.microbiology_susceptibility import MicrobiologySusceptibility
from clifpy.tables.patient import Patient
from clifpy.tables.patient_assessments import PatientAssessments
from clifpy.tables.patient_procedures import PatientProcedures
from clifpy.tables.position import Position
from clifpy.tables.respiratory_support import RespiratorySupport
from clifpy.tables.vitals import Vitals


class ClifReportCardGenerator:
    """
    Generate a user-friendly PDF report card for CLIF table validation results.
    """

    def __init__(self, data_dir: str, output_dir: str, site_config: Dict[str, Any]):
        """
        Initialize the CLIF Report Card generator.

        Args:
            data_dir: Directory containing CLIF data files
            output_dir: Directory for output files
            site_config: Site configuration dictionary
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.site_config = site_config

        # Setup logging to .logs folder
        self._setup_logging()

        # Mapping of table names to clifpy table classes (including all available tables)
        self.table_mapping = {
            'adt': Adt,
            'code_status': CodeStatus,
            'crrt_therapy': CrrtTherapy,
            'ecmo_mcs': EcmoMcs,
            'hospital_diagnosis': HospitalDiagnosis,
            'hospitalization': Hospitalization,
            'labs': Labs,
            'medication_admin_continuous': MedicationAdminContinuous,
            'medication_admin_intermittent': MedicationAdminIntermittent,
            'microbiology_culture': MicrobiologyCulture,
            'microbiology_nonculture': MicrobiologyNonculture,
            'microbiology_susceptibility': MicrobiologySusceptibility,
            'patient': Patient,
            'patient_assessments': PatientAssessments,
            'patient_procedures': PatientProcedures,
            'position': Position,
            'respiratory_support': RespiratorySupport,
            'vitals': Vitals
        }

        # Human-readable table names (including all available tables)
        self.table_display_names = {
            'adt': 'ADT',
            'code_status': 'Code Status',
            'crrt_therapy': 'CRRT Therapy',
            'ecmo_mcs': 'ECMO/MCS',
            'hospital_diagnosis': 'Hospital Diagnosis',
            'hospitalization': 'Hospitalization',
            'labs': 'Labs',
            'medication_admin_continuous': 'Medication Admin Continuous',
            'medication_admin_intermittent': 'Medication Admin Intermittent',
            'microbiology_culture': 'Microbiology Culture',
            'microbiology_nonculture': 'Microbiology Nonculture',
            'microbiology_susceptibility': 'Microbiology Susceptibility',
            'patient': 'Patient',
            'patient_assessments': 'Patient Assessments',
            'patient_procedures': 'Patient Procedures',
            'position': 'Position',
            'respiratory_support': 'Respiratory Support',
            'vitals': 'Vitals'
        }

    def _setup_logging(self):
        """Setup logging to redirect clifpy output to .logs folder."""
        # Create .logs directory if it doesn't exist
        logs_dir = Path('.logs')
        logs_dir.mkdir(exist_ok=True)

        # Create timestamp for log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = logs_dir / f'clif_validation_{timestamp}.log'

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stderr)  # Keep some output visible
            ]
        )

        # Store log file path for reference
        self.log_file = str(log_file)

        # Store logs directory for clifpy output redirection
        self.logs_dir = str(logs_dir)

        # Create intermediate directory for data files like overlapping_admissions.csv
        # Go up one level from output_dir if it ends with 'final'
        output_path = Path(self.output_dir)
        if output_path.name == 'final':
            base_output_dir = output_path.parent
        else:
            base_output_dir = output_path

        self.intermediate_dir = base_output_dir / 'intermediate'
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)

        # Path for consolidated validation report (goes in final directory)
        site_name_clean = self.site_config.get('site_name', 'Unknown_Site').replace(' ', '_').replace('-', '_')
        self.consolidated_report_path = Path(self.output_dir) / f'{site_name_clean}_consolidated_validation_report.csv'

    def _format_clifpy_error(self, error: Dict[str, Any], row_count: int, table_name: str) -> Dict[str, Any]:
        """
        Format a clifpy error object for display in reports.

        Args:
            error: Raw error object from clifpy validation
            row_count: Total number of rows in the table (for percentage calculations)
            table_name: Name of the table being validated

        Returns:
            Formatted error dictionary with type, description, and category
        """
        error_type = error.get('type', 'unknown')

        # Use clifpy's message if available, otherwise build one
        message = error.get('message', '')

        # Determine category based on error type
        category = 'other'
        display_type = error_type.replace('_', ' ').title()

        # Schema-related errors
        if error_type == 'missing_columns':
            category = 'schema'
            display_type = 'Missing Required Columns'
            if not message:
                columns = error.get('columns', [])
                message = f"Required columns not found: {', '.join(columns)}"

        elif error_type in ['datatype_mismatch', 'datatype_castable']:
            category = 'schema'
            display_type = 'Datatype Casting Error'
            if not message:
                column = error.get('column', 'unknown')
                expected = error.get('expected', 'unknown')
                actual = error.get('actual', 'unknown')
                message = f"Column '{column}' has type {actual} but expected {expected}"

        # Data quality errors
        elif error_type == 'null_values':
            category = 'data_quality'
            display_type = 'Missing Values'
            if not message:
                column = error.get('column', 'unknown')
                count = error.get('count', 0)
                percentage = (count / row_count * 100) if row_count > 0 else 0
                message = f"Column '{column}' has {count:,} missing values ({percentage:.1f}%)"

        elif error_type in ['invalid_category', 'invalid_categorical_values']:
            category = 'data_quality'
            display_type = 'Invalid Categories'
            if not message:
                column = error.get('column', 'unknown')
                invalid_values = error.get('invalid_values', error.get('values', []))
                truncated = invalid_values[:3]
                message = f"Column '{column}' contains invalid values: {', '.join(map(str, truncated))}"
                if len(invalid_values) > 3:
                    message += f" (and {len(invalid_values) - 3} more)"

        elif error_type == 'missing_categorical_values':
            category = 'data_quality'
            display_type = 'Missing Categorical Values'
            if not message:
                column = error.get('column', 'unknown')
                missing_values = error.get('missing_values', [])
                total_missing = error.get('total_missing', len(missing_values))
                if missing_values:
                    # Show all missing values, not just a summary
                    values_str = str(missing_values) if len(missing_values) <= 10 else str(missing_values[:10]) + f'... ({len(missing_values) - 10} more)'
                    message = f"Column '{column}' is missing {total_missing} expected category values: {values_str}"
                else:
                    message = f"Column '{column}' is missing {total_missing} expected category values"

        elif error_type == 'duplicate_check':
            category = 'data_quality'
            display_type = 'Duplicate Check'
            if not message:
                duplicate_count = error.get('duplicate_rows', 0)
                total_rows = error.get('total_rows', row_count)
                keys = error.get('composite_keys', [])
                keys_str = ', '.join(keys) if keys else 'composite keys'
                message = f"Found {duplicate_count} duplicate rows out of {total_rows} total rows based on keys: {keys_str}"

        elif error_type == 'unit_validation':
            category = 'data_quality'
            display_type = 'Unit Validation'
            if not message:
                cat = error.get('category', 'unknown')
                unexpected_units = error.get('unexpected_units', [])
                expected_units = error.get('expected_units', [])
                if unexpected_units and expected_units:
                    message = f"Category '{cat}' has unexpected units: {', '.join(unexpected_units[:3])}, expected: {', '.join(expected_units)}"
                else:
                    message = f"Unit validation issue for category '{cat}'"

        elif error_type in ['below_range', 'above_range', 'unknown_vital_category']:
            category = 'data_quality'
            display_type = 'Range Validation'
            if not message:
                vital_category = error.get('vital_category', 'unknown')
                if error_type == 'below_range':
                    min_val = error.get('observed_min', 'N/A')
                    expected_min = error.get('expected_min', 'N/A')
                    message = f"Values below expected minimum for {vital_category} (found: {min_val}, expected: >={expected_min})"
                elif error_type == 'above_range':
                    max_val = error.get('observed_max', 'N/A')
                    expected_max = error.get('expected_max', 'N/A')
                    message = f"Values above expected maximum for {vital_category} (found: {max_val}, expected: <={expected_max})"
                else:  # unknown_vital_category
                    message = f"Unknown vital category '{vital_category}' found in data"

        # Fallback: use error as-is or stringify
        if not message:
            message = str(error)

        return {
            'type': display_type,
            'description': message,
            'category': category
        }

    def _extract_outlier_summary(
        self,
        table_name: str,
        errors: List[Dict[str, Any]],
        df: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Extract outlier summary from validation errors and outlier config.
        Returns summary even for tables with 0 outliers if they have outlier checks configured.

        Args:
            table_name: Name of the table
            errors: List of validation error objects from clifpy
            df: DataFrame (used to check which columns exist)

        Returns:
            Dictionary with outlier summary information
        """
        summary = {
            'has_outlier_config': False,
            'columns_checked': [],
            'total_outliers': 0,
            'outlier_details': []
        }

        try:
            # Load outlier config to see what was checked
            outlier_config = validator.load_outlier_config()
            if not outlier_config:
                return summary

            table_config = outlier_config.get('tables', {}).get(table_name, {})
            if not table_config:
                return summary

            summary['has_outlier_config'] = True

            # Extract outlier validation errors
            outlier_errors = [e for e in errors if e.get('type') == 'outlier_validation']

            # Build a map of what was found in errors
            error_map = {}
            for error in outlier_errors:
                # Create unique key based on error structure
                if 'category' in error:
                    # Single category pattern
                    key = (error.get('column'), error.get('category'))
                elif 'primary_category' in error:
                    # Double category pattern
                    key = (error.get('column'), error.get('primary_category'), error.get('secondary_category'))
                else:
                    # Simple pattern
                    key = (error.get('column'),)
                error_map[key] = error

            # Go through config and create entries for all checked columns
            for col_name, col_config in table_config.items():
                # Skip if column doesn't exist in dataframe
                if df is not None and col_name not in df.columns:
                    continue

                summary['columns_checked'].append(col_name)

                # Pattern 1: Simple range {min: X, max: Y}
                if 'min' in col_config and 'max' in col_config:
                    key = (col_name,)
                    if key in error_map:
                        error = error_map[key]
                        summary['total_outliers'] += error.get('total_outliers', 0)
                        summary['outlier_details'].append({
                            'column': col_name,
                            'category': None,
                            'total_values': error.get('total_values', 0),
                            'total_outliers': error.get('total_outliers', 0),
                            'outlier_percent': error.get('outlier_percent', 0.0),
                            'below_min_count': error.get('below_min_count', 0),
                            'above_max_count': error.get('above_max_count', 0),
                            'min_expected': error.get('min_expected'),
                            'max_expected': error.get('max_expected')
                        })
                    else:
                        # No outliers found for this column
                        summary['outlier_details'].append({
                            'column': col_name,
                            'category': None,
                            'total_values': int(df[col_name].notna().sum()) if df is not None else 0,
                            'total_outliers': 0,
                            'outlier_percent': 0.0,
                            'below_min_count': 0,
                            'above_max_count': 0,
                            'min_expected': col_config['min'],
                            'max_expected': col_config['max']
                        })

                # Pattern 2 & 3: Category-dependent (just count from errors, don't enumerate all categories)
                else:
                    # Check if any errors exist for this column (any category)
                    col_errors = [e for e in outlier_errors if e.get('column') == col_name]
                    if col_errors:
                        for error in col_errors:
                            summary['total_outliers'] += error.get('total_outliers', 0)
                            category_label = error.get('category')
                            if 'primary_category' in error:
                                category_label = f"{error.get('primary_category')} / {error.get('secondary_category')}"

                            summary['outlier_details'].append({
                                'column': col_name,
                                'category': category_label,
                                'total_values': error.get('total_values', 0),
                                'total_outliers': error.get('total_outliers', 0),
                                'outlier_percent': error.get('outlier_percent', 0.0),
                                'below_min_count': error.get('below_min_count', 0),
                                'above_max_count': error.get('above_max_count', 0),
                                'min_expected': error.get('min_expected'),
                                'max_expected': error.get('max_expected')
                            })
                    else:
                        # No outliers found for any category in this column
                        summary['outlier_details'].append({
                            'column': col_name,
                            'category': 'All categories',
                            'total_values': int(df[col_name].notna().sum()) if df is not None else 0,
                            'total_outliers': 0,
                            'outlier_percent': 0.0,
                            'below_min_count': 0,
                            'above_max_count': 0,
                            'min_expected': None,
                            'max_expected': None
                        })

        except Exception as e:
            logging.error(f"Error extracting outlier summary for {table_name}: {str(e)}")

        return summary

    def validate_table(self, table_name: str) -> Dict[str, Any]:
        """
        Run clifpy validation on a table and return user-friendly results.

        Args:
            table_name: Name of the table to validate

        Returns:
            Dictionary with validation results and user-friendly information
        """
        table_class = self.table_mapping.get(table_name)
        if not table_class:
            return {
                'status': 'error',
                'error': f'Unknown table type: {table_name}',
                'validation_results': {},
                'data_info': {}
            }

        # Create temp directory and initialize variables
        temp_dir = None
        try:
            # Redirect stdout to capture clifpy's verbose output
            from io import StringIO
            import contextlib
            import tempfile
            import shutil

            captured_output = StringIO()
            temp_dir = tempfile.mkdtemp()

            with contextlib.redirect_stdout(captured_output):
                # Initialize table instance with temp directory
                table_instance = table_class.from_file(
                    data_directory=self.data_dir,
                    filetype=self.site_config.get('file_type'),
                    timezone=self.site_config.get('timezone'),
                    output_directory=temp_dir
                )

            # Log the captured output
            captured_text = captured_output.getvalue()
            if captured_text.strip():
                logging.info(f"Table {table_name} loading output:\n{captured_text}")

            # Check if table loaded successfully
            if not hasattr(table_instance, 'df') or table_instance.df is None:
                return {
                    'status': 'missing',
                    'error': f'Data file for {table_name} not found',
                    # 'validation_results': {},
                    # 'data_info': {}
                }

            # Get basic data information
            data_info = {
                'row_count': len(table_instance.df),
                'column_count': len(table_instance.df.columns),
                'columns': list(table_instance.df.columns)
            }

            # Add unique IDs based on what columns are available
            if 'hospitalization_id' in table_instance.df.columns:
                data_info['unique_hospitalizations'] = table_instance.df["hospitalization_id"].nunique()
            else:
                data_info['unique_hospitalizations'] = None

            if 'patient_id' in table_instance.df.columns:
                data_info['unique_patients'] = table_instance.df["patient_id"].nunique()
            else:
                data_info['unique_patients'] = None

            # ADT-specific check for overlapping admissions
            if table_name == 'adt':
                # Check for various possible method names for overlapping admissions
                overlap_methods = ['check_overlapping_admissions', 'overlapping_admissions', 'check_overlaps', 'find_overlapping_admissions']
                overlap_method = None

                for method_name in overlap_methods:
                    if hasattr(table_instance, method_name):
                        overlap_method = method_name
                        break

                if overlap_method:
                    try:
                        logging.info(f"Running overlapping admissions check using method '{overlap_method}', saving to: {self.intermediate_dir}")
                        method = getattr(table_instance, overlap_method)

                        # Try calling with different parameter patterns
                        overlapping_count = None
                        try:
                            overlapping_count = method(save_overlaps=True, overlaps_output_directory=str(self.intermediate_dir))
                        except TypeError:
                            try:
                                overlapping_count = method(output_directory=str(self.intermediate_dir))
                            except TypeError:
                                overlapping_count = method()

                        data_info['overlapping_admissions'] = overlapping_count if overlapping_count is not None else 0
                        logging.info(f"ADT overlapping admissions check completed: {overlapping_count} hospitalizations with overlaps")
                    except Exception as e:
                        logging.error(f"ADT overlapping admissions check failed: {str(e)}")
                        data_info['overlapping_admissions'] = None
                else:
                    # List available methods for debugging
                    methods = [method for method in dir(table_instance) if not method.startswith('_') and callable(getattr(table_instance, method))]
                    logging.warning(f"ADT table instance does not have overlapping admissions method. Available methods: {methods}")
                    data_info['overlapping_admissions'] = None
            else:
                data_info['overlapping_admissions'] = None

            # Run validation
            validation_results = {}
            if hasattr(table_instance, 'validate'):
                try:
                    # Capture validation output
                    validation_output = StringIO()
                    with contextlib.redirect_stdout(validation_output):
                        table_instance.validate()

                    # Log validation output
                    validation_text = validation_output.getvalue()
                    if validation_text.strip():
                        logging.info(f"Table {table_name} validation output:\n{validation_text}")

                    validation_results['validation_ran'] = True
                except Exception as e:
                    logging.error(f"Table {table_name} validation failed: {str(e)}")
                    validation_results['validation_error'] = str(e)
                    validation_results['validation_ran'] = False

            # Check if table is valid
            is_valid = table_instance.isvalid() if hasattr(table_instance, 'isvalid') else True
            validation_results['is_valid'] = is_valid

            # Format validation errors using clifpy's error objects
            schema_errors = []
            data_quality_issues = []
            other_errors = []

            if hasattr(table_instance, 'errors') and table_instance.errors:
                for error in table_instance.errors:
                    formatted_error = self._format_clifpy_error(error, data_info.get('row_count', 0), table_name)

                    # Categorize errors
                    if formatted_error['category'] == 'schema':
                        schema_errors.append(formatted_error)
                    elif formatted_error['category'] == 'data_quality':
                        data_quality_issues.append(formatted_error)
                    else:
                        other_errors.append(formatted_error)

            validation_results.update({
                'schema_errors': schema_errors,
                'data_quality_issues': data_quality_issues,
                'other_errors': other_errors
            })

            # Determine overall status based on new requirements
            # Red (incomplete): Missing required columns OR NON-CASTABLE datatype errors OR 100% missing values
            has_missing_columns = any(error.get('type') == 'Missing Required Columns' for error in schema_errors)

            # Only treat as INCOMPLETE if datatype CANNOT be cast
            # Errors that say "can be cast to" should not trigger INCOMPLETE
            has_datatype_errors = any(
                error.get('type') == 'Datatype Casting Error' and 'can be cast to' not in error.get('description', '')
                for error in schema_errors
            )

            # Check for 100% missing values in REQUIRED columns only (red condition)
            has_100_percent_missing_required = False

            # Define columns that should NOT trigger INCOMPLETE even if 100% null and required
            # Table-specific exceptions
            table_specific_exceptions = {
                'patient_assessments': {'numerical_value', 'categorical_value', 'text_value'},
                'crrt_therapy': {'pre_filter_replacement_fluid_rate', 'post_filter_replacement_fluid_rate'},
                'respiratory_support': {
                    'device_category', 'mode_category', 'vent_brand_name', 'tracheostomy',
                    'fio2_set', 'lpm_set', 'tidal_volume_set', 'resp_rate_set',
                    'pressure_control_set', 'pressure_support_set', 'flow_rate_set',
                    'peak_inspiratory_pressure_set', 'inspiratory_time_set', 'peep_set',
                    'tidal_volume_obs', 'resp_rate_obs', 'plateau_pressure_obs',
                    'peak_inspiratory_pressure_obs', 'peep_obs', 'minute_vent_obs',
                    'mean_airway_pressure_obs'
                }
            }

            # Get required columns from schema
            if hasattr(table_instance, 'schema') and table_instance.schema:
                required_columns = table_instance.schema.get('required_columns', [])
                logging.info(f"Table {table_name} required columns from schema: {required_columns}")

                # Get exceptions for this specific table
                exceptions_for_table = table_specific_exceptions.get(table_name, set())

                for error in data_quality_issues:
                    if error.get('type') == 'Missing Values':
                        description = error.get('description', '')
                        logging.info(f"Checking missing values error: {description}")

                        # Check if this is 100% missing (look for "100.0%" or "100.00%")
                        if '100.0%' in description or '100.00%' in description:
                            # Extract column name from description
                            if "Column '" in description:
                                try:
                                    column_name = description.split("Column '")[1].split("'")[0]

                                    # Check if column should be excluded from INCOMPLETE trigger
                                    # 1. Check if it ends with _type (pattern-based exception)
                                    # 2. Check if it's in the table-specific exceptions
                                    if column_name.endswith('_type'):
                                        logging.info(f"Column '{column_name}' ends with '_type', excluding from INCOMPLETE trigger")
                                        continue
                                    elif column_name in exceptions_for_table:
                                        logging.info(f"Column '{column_name}' is in exception list for {table_name}, excluding from INCOMPLETE trigger")
                                        continue

                                    # Check if this column is in the required_columns list
                                    if column_name in required_columns:
                                        has_100_percent_missing_required = True
                                        logging.warning(f"INCOMPLETE: Found 100% missing values in required column '{column_name}' for table {table_name}")
                                        break
                                    else:
                                        logging.info(f"Column '{column_name}' has 100% missing but is not required")
                                except Exception as e:
                                    logging.warning(f"Error extracting column name from description: {str(e)}")
            else:
                # Fallback: if no schema available, treat all 100% missing as problematic
                logging.warning(f"No schema found for table {table_name}, treating all 100% missing as problematic")
                has_100_percent_missing_required = any(
                    error.get('type') == 'Missing Values' and ('100.0%' in error.get('description', '') or '100.00%' in error.get('description', ''))
                    for error in data_quality_issues
                )

            # Yellow (partial): Has required columns but missing categorical values
            has_categorical_issues = any(
                error.get('type') in ['Invalid Categories', 'Missing Categorical Values']
                for error in data_quality_issues
            )

            if has_missing_columns or has_datatype_errors or has_100_percent_missing_required:
                # Red: Missing required columns OR datatype casting problems OR 100% missing values in required columns
                status = 'incomplete'
            elif has_categorical_issues:
                # Yellow: Has required columns but missing some required categorical values
                status = 'partial'
            else:
                # Green: All required columns present, all categorical values present
                status = 'complete'

            # Generate outlier plot if table has outlier configuration
            outlier_plot_paths = []
            try:
                outlier_config = validator.load_outlier_config()
                if outlier_config and hasattr(table_instance, 'schema') and table_instance.schema:
                    # Check if this table has outlier configuration
                    if table_name in outlier_config.get('tables', {}):
                        # Create plots directory in intermediate folder
                        plots_dir = self.intermediate_dir / 'outlier_plots'
                        plots_dir.mkdir(exist_ok=True)

                        # Generate plot
                        plot_path = plots_dir / f'{table_name}_outliers.png'
                        logging.info(f"Generating outlier plot for {table_name}")

                        fig = validator.plot_outlier_distribution(
                            df=table_instance.df,
                            table_name=table_name,
                            schema=table_instance.schema,
                            outlier_config=outlier_config,
                            save_path=str(plot_path),
                            show_plot=False,  # Don't display in non-interactive mode
                            figsize=(14, 8)
                        )

                        if fig is not None:
                            # Close the figure(s) to free memory
                            import matplotlib.pyplot as plt
                            if isinstance(fig, list):
                                # Multiple figures returned
                                for f in fig:
                                    plt.close(f)
                            else:
                                # Single figure
                                plt.close(fig)

                            # Check for multi-part plots (e.g., labs_outliers_part1.png, labs_outliers_part2.png)
                            import glob
                            part_pattern = str(plots_dir / f'{table_name}_outliers_part*.png')
                            part_files = sorted(glob.glob(part_pattern))

                            if part_files:
                                # Multi-part plots exist, use them instead
                                outlier_plot_paths = part_files
                                logging.info(f"Found {len(part_files)} multi-part outlier plots for {table_name}")
                            elif os.path.exists(plot_path):
                                # Single plot
                                outlier_plot_paths = [str(plot_path)]
                                logging.info(f"Outlier plot saved to: {plot_path}")
            except Exception as e:
                logging.warning(f"Could not generate outlier plot for {table_name}: {str(e)}")

            # Extract outlier summary from errors and config
            outlier_summary = self._extract_outlier_summary(
                table_name,
                table_instance.errors if hasattr(table_instance, 'errors') else [],
                table_instance.df if hasattr(table_instance, 'df') else None
            )

            # Extract missingness summary using clifpy's function
            missingness_summary = None
            if hasattr(table_instance, 'df') and table_instance.df is not None:
                try:
                    missingness_summary = validator.report_missing_data_summary(table_instance.df)
                except Exception as e:
                    logging.warning(f"Could not generate missingness summary for {table_name}: {str(e)}")

            return {
                'status': status,
                'data_info': data_info,
                'validation_results': validation_results,
                'outlier_plots': outlier_plot_paths,  # Changed to plural and list
                'outlier_summary': outlier_summary,
                'missingness_summary': missingness_summary
            }

        except FileNotFoundError:
            return {
                'status': 'missing',
                'error': f'Data file for {table_name} not found',
                'validation_results': {},
                'data_info': {}
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': f'Validation failed: {str(e)}',
                'validation_results': {},
                'data_info': {}
            }
        finally:
            # Clean up temporary directory in all cases
            if temp_dir is not None:
                shutil.rmtree(temp_dir, ignore_errors=True)

    def generate_pdf_report_card(self, table_names: List[str], output_file: str) -> Dict[str, Any]:
        """
        Generate a user-friendly PDF report card.

        Args:
            table_names: List of table names to validate
            output_file: Path to output PDF file

        Returns:
            Dictionary containing the validation results
        """
        if not REPORTLAB_AVAILABLE:
            raise ImportError("reportlab package is required for PDF generation. Install with: pip install reportlab")

        # Start runtime tracking
        start_time = datetime.now()

        print("Generating CLIF Data Validation Report Card...")
        print(f"📝 Detailed logs will be saved to: {self.log_file}")

        # Validate all tables
        table_results = {}
        for table_name in table_names:
            print(f"Validating table: {table_name}")
            logging.info(f"Starting validation for table: {table_name}")
            table_results[table_name] = self.validate_table(table_name)

        # Generate consolidated CSV report
        self._create_consolidated_csv_report(table_results)

        # Calculate runtime
        end_time = datetime.now()
        runtime_seconds = (end_time - start_time).total_seconds()
        runtime_str = f"{runtime_seconds:.2f} seconds"

        # Generate timestamp
        timestamp = end_time

        # Create report data
        report_data = {
            'site_name': self.site_config.get('site_name', 'Unknown Site'),
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'runtime': runtime_str,
            'table_results': table_results
        }

        # Generate PDF
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        self._generate_pdf_content(report_data, output_file)

        print(f"Report card saved to: {output_file}")
        print(f"⏱️  Total runtime: {runtime_str}")
        return report_data

    def _generate_pdf_content(self, report_data: Dict[str, Any], output_file: str):
        """Generate PDF content for the report card."""
        doc = SimpleDocTemplate(output_file, pagesize=letter,
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=18)

        # Container for the 'Flowable' objects
        story = []

        # Define professional styles
        styles = getSampleStyleSheet()

        # Professional color palette
        primary_color = colors.HexColor('#1F4E79')      # Deep blue
        secondary_color = colors.HexColor('#2F5F8F')    # Medium blue
        accent_color = colors.HexColor('#4A90A4')       # Teal blue
        text_dark = colors.HexColor('#2C3E50')          # Dark gray
        text_medium = colors.HexColor('#5D6D7E')        # Medium gray

        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=22,
            spaceAfter=24,
            alignment=TA_CENTER,
            textColor=primary_color,
            fontName='Helvetica-Bold'
        )

        header_style = ParagraphStyle(
            'CustomHeader',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=10,
            spaceBefore=16,
            textColor=text_dark,
            fontName='Helvetica-Bold'
        )

        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=9,
            textColor=text_medium,
            fontName='Helvetica'
        )

        # Add timestamp in top right corner as separate element
        timestamp_style = ParagraphStyle(
            'TimestampStyle',
            parent=styles['Normal'],
            fontSize=8,
            textColor=text_medium,
            alignment=TA_RIGHT,
            fontName='Helvetica'
        )

        # Create a single-cell table for just the timestamp in top right
        timestamp_table = Table([[Paragraph(f"Generated: {report_data['timestamp']}", timestamp_style)]],
                               colWidths=[7.5*inch])
        timestamp_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (0, 0), 'RIGHT'),
            ('VALIGN', (0, 0), (0, 0), 'TOP'),
            ('TOPPADDING', (0, 0), (0, 0), -24),  # Negative padding to move up
            ('BOTTOMPADDING', (0, 0), (0, 0), 2),
            ('LEFTPADDING', (0, 0), (0, 0), 0),
            ('RIGHTPADDING', (0, 0), (0, 0), 0),
        ]))
        story.append(timestamp_table)

        # Add centered title separately
        title = Paragraph(f"{report_data['site_name']} CLIF Data Report Card", title_style)
        story.append(title)
        story.append(Spacer(1, 12))

        # Check if outlier plots will be included and add note after title
        outlier_plots = {name: result.get('outlier_plots', [])
                        for name, result in report_data['table_results'].items()
                        if result.get('outlier_plots')}

        if outlier_plots:
            # Add note about appendix right after title
            appendix_note = Paragraph(
                f"<i>Outlier distribution plots for {len(outlier_plots)} table(s) included in Appendix</i>",
                ParagraphStyle(
                    'AppendixNote',
                    parent=normal_style,
                    fontSize=10,
                    textColor=text_medium,
                    alignment=TA_CENTER,
                    spaceAfter=6
                )
            )
            story.append(appendix_note)

        story.append(Spacer(1, 18))

        # Overall status
        status_colors = {
            'complete': colors.green,
            'partial': colors.orange,
            'noinformation': colors.grey
        }

        status_labels = {
            'complete': 'COMPLETE',
            'partial': 'PARTIAL',
            'noinformation': 'NO INFORMATION'
        }

        # Calculate summary statistics first
        results = report_data['table_results']
        total_tables = len(results)
        complete_tables = sum(1 for r in results.values() if r['status'] == 'complete')
        partial_tables = sum(1 for r in results.values() if r['status'] == 'partial')
        incomplete_tables = sum(1 for r in results.values() if r['status'] in ['missing', 'incomplete', 'error'])
        total_rows = sum(r.get('data_info', {}).get('row_count', 0) for r in results.values())

        # Combined Status Overview with creative layout
        combined_header = Paragraph("Status Report Overview", header_style)
        story.append(combined_header)

        # Professional status colors
        status_complete = colors.HexColor('#E8F5E8')    # Soft green background
        status_partial = colors.HexColor('#FFF4E6')     # Soft orange background
        status_incomplete = colors.HexColor('#FFEAEA')  # Soft red background
        header_bg = colors.HexColor('#F5F6FA')          # Light gray header

        # Create combined table with site name and description
        site_name = report_data['site_name']
        combined_data = [
            ['Status', site_name, 'Criteria'],
            ['COMPLETE', str(complete_tables), Paragraph('All required columns present, all categorical values present', normal_style)],
            ['PARTIAL', str(partial_tables), Paragraph('All required columns present, but missing some categorical values', normal_style)],
            ['INCOMPLETE', str(incomplete_tables), Paragraph('One or more of: (1) Missing required columns, (2) Non-castable datatype errors, (3) Required columns with no data (100% null count), (4) Table file not found', normal_style)]
        ]

        # Add a summary row if there are multiple tables
        if total_tables > 1:
            combined_data.append(['TOTAL', str(total_tables), Paragraph(f'<b>Total tables analyzed across {total_rows:,} data rows</b>', normal_style)])

        combined_table = Table(combined_data, colWidths=[1.2*inch, 0.8*inch, 4.5*inch])

        # Dynamic styling based on number of rows
        num_rows = len(combined_data)
        table_style = [
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (-1, 0), text_dark),
            ('TEXTCOLOR', (0, 1), (-1, -1), text_medium),
            ('BACKGROUND', (0, 0), (-1, 0), header_bg),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DADADA')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('LEFTPADDING', (0, 0), (-1, -1), 12),
            ('RIGHTPADDING', (0, 0), (-1, -1), 12),
            # Color-code the status column
            ('BACKGROUND', (0, 1), (0, 1), status_complete),
            ('BACKGROUND', (0, 2), (0, 2), status_partial),
            ('BACKGROUND', (0, 3), (0, 3), status_incomplete),
        ]

        # Add total row styling if it exists
        if num_rows > 4:  # Has total row
            table_style.extend([
                ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor('#F0F0F0')),
                ('FONTNAME', (0, 4), (-1, 4), 'Helvetica-Bold'),
                ('LINEABOVE', (0, 4), (-1, 4), 1, colors.HexColor('#DADADA')),
            ])

        combined_table.setStyle(TableStyle(table_style))
        story.append(combined_table)
        story.append(Spacer(1, 30))

        # Table validation results
        table_header = Paragraph("Table Validation Results", header_style)
        story.append(table_header)

        for table_name, result in report_data['table_results'].items():
            story.append(self._create_table_section(table_name, result))
            story.append(Spacer(1, 20))

        # Add Appendix section with outlier summary and plots
        # Check if any table has outlier configuration
        has_outlier_data = any(
            result.get('outlier_summary', {}).get('has_outlier_config', False)
            for result in report_data['table_results'].values()
        )

        outlier_plots = {name: result.get('outlier_plots', [])
                        for name, result in report_data['table_results'].items()
                        if result.get('outlier_plots')}

        if has_outlier_data or outlier_plots:
            story.append(PageBreak())
            appendix_header = Paragraph("Appendix: Outlier Validation Summary", header_style)
            story.append(appendix_header)
            story.append(Spacer(1, 12))

            # Add outlier distribution plots FIRST
            if outlier_plots:
                plot_section_header = Paragraph(
                    "Outlier Distribution Plots",
                    ParagraphStyle('PlotSectionHeader',
                                 parent=header_style,
                                 fontSize=11,
                                 spaceAfter=8)
                )
                story.append(plot_section_header)
                story.append(Spacer(1, 8))

                for table_name, plot_paths in outlier_plots.items():
                    if plot_paths:  # plot_paths is now a list
                        display_name = self.table_display_names.get(table_name, table_name.title())

                        # Add table name subheader
                        plot_header = Paragraph(f"{display_name} Table",
                                               ParagraphStyle('PlotHeader',
                                                            parent=header_style,
                                                            fontSize=10,
                                                            spaceAfter=8))
                        story.append(plot_header)

                        # Add each plot image (handles multi-part plots)
                        for i, plot_path in enumerate(plot_paths):
                            if os.path.exists(plot_path):
                                try:
                                    img = Image(plot_path, width=7*inch, height=4*inch)
                                    story.append(img)
                                    story.append(Spacer(1, 12))
                                except Exception as e:
                                    logging.warning(f"Could not add outlier plot {plot_path} for {table_name}: {str(e)}")

                        story.append(Spacer(1, 8))  # Extra space between tables

            # Add outlier summary table AFTER plots
            outlier_summary_table = self._create_outlier_summary_table(report_data['table_results'])
            if outlier_summary_table:
                # Add section header for summary table
                summary_subheader = Paragraph(
                    "Outlier Summary by Column",
                    ParagraphStyle('SummarySubheader',
                                 parent=header_style,
                                 fontSize=11,
                                 spaceAfter=8,
                                 spaceBefore=0)
                )
                story.append(summary_subheader)

                # Add explanatory note
                note_style = ParagraphStyle(
                    'NoteStyle',
                    parent=normal_style,
                    fontSize=8,
                    textColor=text_medium,
                    spaceAfter=8
                )
                note_text = Paragraph(
                    "<i>Highlighted rows indicate outliers found.</i>",
                    note_style
                )
                story.append(note_text)

                story.append(outlier_summary_table)
                story.append(Spacer(1, 20))

            # Add missingness summary table AFTER outlier summary
            missingness_summary_table = self._create_missingness_summary_table(report_data['table_results'])
            if missingness_summary_table:
                # Add section header for missingness table
                missingness_subheader = Paragraph(
                    "Missingness Summary by Column",
                    ParagraphStyle('MissingnessSubheader',
                                 parent=header_style,
                                 fontSize=11,
                                 spaceAfter=8,
                                 spaceBefore=0)
                )
                story.append(missingness_subheader)

                # Add explanatory note
                note_style = ParagraphStyle(
                    'NoteStyle',
                    parent=normal_style,
                    fontSize=8,
                    textColor=text_medium,
                    spaceAfter=8
                )
                note_text = Paragraph(
                    "<i>Highlighted rows indicate >50% missing.</i>",
                    note_style
                )
                story.append(note_text)

                story.append(missingness_summary_table)
                story.append(Spacer(1, 20))

        # Build PDF
        doc.build(story)

    def _create_table_section(self, table_name: str, result: Dict[str, Any]):
        """Create a section for a single table's results."""
        # Get styles for text wrapping
        from reportlab.lib.styles import getSampleStyleSheet
        styles = getSampleStyleSheet()

        # Define color palette (same as in main method)
        text_dark = colors.HexColor('#2C3E50')          # Dark gray
        text_medium = colors.HexColor('#5D6D7E')        # Medium gray

        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=9,
            textColor=text_medium,
            fontName='Helvetica'
        )
        display_name = self.table_display_names.get(table_name, table_name.title())
        status = result['status']

        # Use the exact same background colors as the Status Overview table
        status_colors_exact = {
            'complete': colors.HexColor('#E8F5E8'),      # Same soft green background
            'partial': colors.HexColor('#FFF4E6'),       # Same soft orange background
            'incomplete': colors.HexColor('#FFEAEA'),    # Same soft red background
            'missing': colors.HexColor('#F5F6FA'),       # Same light gray background
            'error': colors.HexColor('#FFEAEA')          # Same soft red background
        }

        color = status_colors_exact.get(status, colors.HexColor('#F5F6FA'))

        # Create table data
        table_data = [[display_name, status.upper()]]

        if status in ['missing', 'error']:
            error_msg = result.get('error', 'Unknown issue')
            table_data.append(['Issue:', error_msg])
        else:
            # Add data statistics - show unique IDs instead of rows/columns
            data_info = result.get('data_info', {})
            if data_info:
                # Show unique hospitalizations if available
                if data_info.get('unique_hospitalizations') is not None:
                    table_data.append(['Unique Hospitalizations:', f"{data_info.get('unique_hospitalizations', 0):,}"])

                # Show unique patients if available
                if data_info.get('unique_patients') is not None:
                    table_data.append(['Unique Patients:', f"{data_info.get('unique_patients', 0):,}"])

                # ADT-specific: show overlapping admissions count
                if table_name == 'adt' and data_info.get('overlapping_admissions') is not None:
                    overlapping_count = data_info.get('overlapping_admissions', 0)
                    if overlapping_count > 0:
                        table_data.append(['Overlapping Admissions', f"{overlapping_count:,} hospitalizations"])
                    else:
                        table_data.append(['Overlapping Admissions', "None found"])

                # If neither unique ID is available, fall back to total rows
                if data_info.get('unique_hospitalizations') is None and data_info.get('unique_patients') is None:
                    table_data.append(['Total Rows:', f"{data_info.get('row_count', 0):,}"])

            # Add validation issues
            validation_results = result.get('validation_results', {})
            all_issues = []
            all_issues.extend(validation_results.get('schema_errors', []))
            all_issues.extend(validation_results.get('data_quality_issues', []))
            all_issues.extend(validation_results.get('other_errors', []))

            if all_issues:
                # Check if this table has outlier plots in appendix
                has_outlier_plots = bool(result.get('outlier_plots'))

                # Separate outlier validation issues from other issues
                outlier_issues = []
                non_outlier_issues = []
                for issue in all_issues:
                    issue_type = issue.get('type', 'Issue')
                    if 'Outlier' in issue_type or issue_type == 'Range Validation':
                        outlier_issues.append(issue)
                    else:
                        non_outlier_issues.append(issue)

                # Count total issues (for display)
                table_data.append(['Issues Found:', str(len(all_issues))])

                # Group non-outlier issues by type for better display
                issue_groups = {}
                for issue in non_outlier_issues:
                    issue_type = issue.get('type', 'Issue')
                    if issue_type not in issue_groups:
                        issue_groups[issue_type] = []
                    issue_groups[issue_type].append(issue.get('description', 'No description'))

                # Display grouped issues with truncation to prevent overflow
                counter = 1
                MAX_ISSUES_TO_DISPLAY = 10  # Limit to prevent page overflow

                for issue_type, descriptions in issue_groups.items():
                    if len(descriptions) == 1:
                        # Single issue - display normally
                        issue_desc_paragraph = Paragraph(descriptions[0], normal_style)
                        table_data.append([f"{counter}. {issue_type}:", issue_desc_paragraph])
                        counter += 1
                    else:
                        # Multiple issues of same type - use bullet points with truncation
                        bullet_style = ParagraphStyle(
                            'BulletStyle',
                            parent=normal_style,
                            fontSize=8,
                            leftIndent=12,
                            bulletIndent=6,
                            spaceAfter=2
                        )

                        # Truncate if too many issues
                        display_descriptions = descriptions[:MAX_ISSUES_TO_DISPLAY]
                        truncated_count = len(descriptions) - len(display_descriptions)

                        bullet_points = [f"• {desc}" for desc in display_descriptions]
                        if truncated_count > 0:
                            bullet_points.append(f"• <i>...and {truncated_count} more (see CSV report for details)</i>")

                        bullet_text = "<br/>".join(bullet_points)
                        issue_desc_paragraph = Paragraph(bullet_text, bullet_style)
                        table_data.append([f"{counter}. {issue_type}:", issue_desc_paragraph])
                        counter += 1

                # Add outlier validation reference if applicable
                if outlier_issues and has_outlier_plots:
                    outlier_ref = Paragraph(
                        f"<i>See Appendix for {len(outlier_issues)} outlier distribution plot(s)</i>",
                        normal_style
                    )
                    table_data.append([f"{counter}. Outlier Validation:", outlier_ref])

            elif status == 'complete':
                # Check if there are outlier plots even for complete tables
                has_outlier_plots = bool(result.get('outlier_plots'))
                if has_outlier_plots:
                    # Check if there are any outlier validation issues
                    all_errors = []
                    all_errors.extend(validation_results.get('schema_errors', []))
                    all_errors.extend(validation_results.get('data_quality_issues', []))
                    all_errors.extend(validation_results.get('other_errors', []))

                    outlier_count = sum(1 for issue in all_errors
                                       if 'Outlier' in issue.get('type', '') or issue.get('type') == 'Range Validation')

                    if outlier_count > 0:
                        table_data.append(['Issues Found:', str(outlier_count)])
                        outlier_ref = Paragraph(
                            f"<i>See Appendix for {outlier_count} outlier distribution plot(s)</i>",
                            normal_style
                        )
                        table_data.append(['1. Outlier Validation:', outlier_ref])
                    else:
                        table_data.append(['Status:', 'All validation checks passed!'])
                else:
                    table_data.append(['Status:', 'All validation checks passed!'])

        # Create table with professional styling
        table = Table(table_data, colWidths=[2.5*inch, 4*inch])
        table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (-1, 0), color),
            ('TEXTCOLOR', (0, 0), (-1, 0), text_dark),  # Use dark text on light backgrounds
            ('TEXTCOLOR', (0, 1), (-1, -1), text_medium),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DADADA')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            # Alternate row background for better readability
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ]))

        return table

    def _create_outlier_summary_table(self, table_results: Dict[str, Any]) -> Optional[Table]:
        """
        Create a comprehensive outlier summary table for all tables.

        Args:
            table_results: Dictionary of validation results for all tables

        Returns:
            ReportLab Table object or None if no outlier data
        """
        from reportlab.lib.styles import getSampleStyleSheet
        styles = getSampleStyleSheet()

        # Define color palette
        text_dark = colors.HexColor('#2C3E50')
        text_medium = colors.HexColor('#5D6D7E')
        header_bg = colors.HexColor('#F5F6FA')
        no_outliers_bg = colors.HexColor('#E8F5E8')  # Light green
        has_outliers_bg = colors.HexColor('#FFF4E6')  # Light orange

        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=8,
            textColor=text_medium,
            fontName='Helvetica'
        )

        # Collect outlier data from all tables
        all_outlier_data = []

        for table_name, result in table_results.items():
            outlier_summary = result.get('outlier_summary', {})
            if not outlier_summary.get('has_outlier_config'):
                continue

            display_name = self.table_display_names.get(table_name, table_name.title())
            outlier_details = outlier_summary.get('outlier_details', [])

            if not outlier_details:
                # Table has config but no details (shouldn't happen, but handle it)
                all_outlier_data.append({
                    'table': display_name,
                    'column': 'N/A',
                    'category': '',
                    'total_values': 0,
                    'total_outliers': 0,
                    'outlier_percent': 0.0,
                    'range': 'N/A'
                })
            else:
                for detail in outlier_details:
                    # Skip entries where no valid range exists (both min and max are None)
                    if detail.get('min_expected') is None and detail.get('max_expected') is None:
                        continue

                    range_str = 'N/A'
                    if detail.get('min_expected') is not None and detail.get('max_expected') is not None:
                        range_str = f"{detail['min_expected']} - {detail['max_expected']}"

                    category_str = detail.get('category', '')
                    if category_str:
                        column_display = f"{detail['column']}\n({category_str})"
                    else:
                        column_display = detail['column']

                    all_outlier_data.append({
                        'table': display_name,
                        'column': column_display,
                        'category': category_str or '-',
                        'total_values': detail.get('total_values', 0),
                        'total_outliers': detail.get('total_outliers', 0),
                        'outlier_percent': detail.get('outlier_percent', 0.0),
                        'range': range_str,
                        'below_count': detail.get('below_min_count', 0),
                        'above_count': detail.get('above_max_count', 0)
                    })

        if not all_outlier_data:
            return None

        # Create table data
        table_data = [
            ['Table', 'Column/Category', 'Total Values', 'Outliers', 'Outlier %', 'Expected Range']
        ]

        for data in all_outlier_data:
            table_data.append([
                Paragraph(data['table'], normal_style),
                Paragraph(data['column'], normal_style),
                f"{data['total_values']:,}",
                f"{data['total_outliers']:,}",
                f"{data['outlier_percent']:.2f}%",
                Paragraph(data['range'], normal_style)
            ])

        # Create table with appropriate column widths
        col_widths = [1.2*inch, 1.8*inch, 1.0*inch, 0.8*inch, 0.8*inch, 1.2*inch]
        summary_table = Table(table_data, colWidths=col_widths)

        # Build style with alternating row colors based on outlier status
        table_style_list = [
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (2, 1), (4, -1), 'CENTER'),  # Center numeric columns
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('TEXTCOLOR', (0, 0), (-1, 0), text_dark),
            ('TEXTCOLOR', (0, 1), (-1, -1), text_medium),
            ('BACKGROUND', (0, 0), (-1, 0), header_bg),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DADADA')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]

        # Color rows based on outlier presence (only highlight rows with outliers)
        for idx, data in enumerate(all_outlier_data, start=1):
            if data['total_outliers'] > 0:
                table_style_list.append(('BACKGROUND', (0, idx), (-1, idx), has_outliers_bg))

        summary_table.setStyle(TableStyle(table_style_list))

        return summary_table

    def _create_missingness_summary_table(self, table_results: Dict[str, Any]) -> Optional[Table]:
        """
        Create a comprehensive missingness summary table for all tables.

        Args:
            table_results: Dictionary of validation results for all tables

        Returns:
            ReportLab Table object or None if no missingness data
        """
        from reportlab.lib.styles import getSampleStyleSheet
        styles = getSampleStyleSheet()

        # Define color palette
        text_dark = colors.HexColor('#2C3E50')
        text_medium = colors.HexColor('#5D6D7E')
        header_bg = colors.HexColor('#F5F6FA')
        high_missing_bg = colors.HexColor('#FFF4E6')  # Light orange for >50% missing
        complete_bg = colors.HexColor('#E8F5E8')  # Light green for 0% missing

        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=8,
            textColor=text_medium,
            fontName='Helvetica'
        )

        # Collect missingness data from all tables
        all_missing_data = []

        for table_name, result in table_results.items():
            if result.get('status') in ['missing', 'error']:
                continue

            missingness_summary = result.get('missingness_summary')
            if not missingness_summary or 'error' in missingness_summary:
                continue

            display_name = self.table_display_names.get(table_name, table_name.title())
            columns_with_missing = missingness_summary.get('columns_with_missing', [])

            # Add columns with missing data
            for col_info in columns_with_missing:
                all_missing_data.append({
                    'table': display_name,
                    'column': col_info['column'],
                    'missing_count': col_info['missing_count'],
                    'missing_percent': col_info['missing_percent'],
                    'total_rows': missingness_summary.get('total_rows', 0)
                })

            # Optionally add columns with 0% missing (can be commented out if too verbose)
            # for col_name in missingness_summary.get('complete_columns', []):
            #     all_missing_data.append({
            #         'table': display_name,
            #         'column': col_name,
            #         'missing_count': 0,
            #         'missing_percent': 0.0,
            #         'total_rows': missingness_summary.get('total_rows', 0)
            #     })

        if not all_missing_data:
            return None

        # Sort by table name (alphabetically), then by missing percentage (descending)
        all_missing_data.sort(key=lambda x: (x['table'], -x['missing_percent']))

        # Create table data
        table_data = [
            ['Table', 'Column', 'Missing Count', 'Missing %', 'Total Rows']
        ]

        for data in all_missing_data:
            table_data.append([
                Paragraph(data['table'], normal_style),
                Paragraph(data['column'], normal_style),
                f"{data['missing_count']:,}",
                f"{data['missing_percent']:.2f}%",
                f"{data['total_rows']:,}"
            ])

        # Create table with appropriate column widths
        col_widths = [1.5*inch, 2.0*inch, 1.2*inch, 1.0*inch, 1.0*inch]
        summary_table = Table(table_data, colWidths=col_widths)

        # Build style with conditional row colors based on missing percentage
        table_style_list = [
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (2, 1), (4, -1), 'CENTER'),  # Center numeric columns
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('TEXTCOLOR', (0, 0), (-1, 0), text_dark),
            ('TEXTCOLOR', (0, 1), (-1, -1), text_medium),
            ('BACKGROUND', (0, 0), (-1, 0), header_bg),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DADADA')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]

        # Color rows based on missing percentage (only highlight high missingness >50%)
        for idx, data in enumerate(all_missing_data, start=1):
            if data['missing_percent'] > 50.0:
                table_style_list.append(('BACKGROUND', (0, idx), (-1, idx), high_missing_bg))

        summary_table.setStyle(TableStyle(table_style_list))

        return summary_table

    def _create_consolidated_csv_report(self, table_results: Dict[str, Any]):
        """Create a single consolidated CSV file with all validation information."""
        import csv

        csv_data = []

        for table_name, result in table_results.items():
            if result['status'] in ['missing', 'error']:
                # For missing/error tables, just add basic info
                csv_data.append({
                    'table_name': self.table_display_names.get(table_name, table_name.title()),
                    'status': result['status'],
                    'issue_type': 'Table Status',
                    'issue_category': 'Critical',
                    'column_name': '',
                    'issue_description': result.get('error', 'Table missing or error'),
                    'unique_hospitalizations': '',
                    'unique_patients': '',
                    'total_rows': ''
                })
                continue

            data_info = result.get('data_info', {})
            validation_results = result.get('validation_results', {})

            # Add basic table information row
            csv_data.append({
                'table_name': self.table_display_names.get(table_name, table_name.title()),
                'status': result['status'],
                'issue_type': 'Table Summary',
                'issue_category': 'Info',
                'column_name': '',
                'issue_description': f"Table loaded successfully",
                'unique_hospitalizations': data_info.get('unique_hospitalizations', ''),
                'unique_patients': data_info.get('unique_patients', ''),
                'total_rows': data_info.get('row_count', '')
            })

            # Add all validation issues
            all_issues = []
            all_issues.extend(validation_results.get('schema_errors', []))
            all_issues.extend(validation_results.get('data_quality_issues', []))
            all_issues.extend(validation_results.get('other_errors', []))

            for issue in all_issues:
                issue_type = issue.get('type', 'Unknown')
                description = issue.get('description', 'No description')

                # Extract column name from description if possible
                column_name = ''
                if 'Column \'' in description:
                    try:
                        column_name = description.split("Column '")[1].split("'")[0]
                    except:
                        pass

                # Determine issue category
                if issue_type in ['Missing Required Columns', 'Datatype Casting Error']:
                    category = 'Schema'
                elif issue_type in ['Missing Categorical Values', 'Invalid Categories']:
                    category = 'Categorical'
                elif issue_type == 'Missing Values':
                    category = 'Data Quality'
                else:
                    category = 'Other'

                csv_data.append({
                    'table_name': self.table_display_names.get(table_name, table_name.title()),
                    'status': result['status'],
                    'issue_type': issue_type,
                    'issue_category': category,
                    'column_name': column_name,
                    'issue_description': description,
                    'unique_hospitalizations': data_info.get('unique_hospitalizations', ''),
                    'unique_patients': data_info.get('unique_patients', ''),
                    'total_rows': data_info.get('row_count', '')
                })

        # Write to CSV
        if csv_data:
            fieldnames = [
                'table_name', 'status', 'issue_type', 'issue_category',
                'column_name', 'issue_description',
                'unique_hospitalizations', 'unique_patients', 'total_rows'
            ]

            with open(self.consolidated_report_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)

            print(f"📊 Consolidated validation report saved to: {self.consolidated_report_path}")

    def generate_simple_text_report(self, table_names: List[str], output_file: str) -> Dict[str, Any]:
        """
        Generate a simple text-based report card (fallback if PDF not available).
        """
        print("Generating CLIF Data Validation Text Report...")
        print(f"📝 Detailed logs will be saved to: {self.log_file}")

        # Validate all tables
        table_results = {}
        for table_name in table_names:
            print(f"Validating table: {table_name}")
            logging.info(f"Starting validation for table: {table_name}")
            table_results[table_name] = self.validate_table(table_name)
        timestamp = datetime.now()

        # Create report data
        report_data = {
            'site_name': self.site_config.get('site_name', 'Unknown Site'),
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'table_results': table_results
        }

        # Generate text content
        content = []
        content.append("="*60)
        content.append(f"🏥 {report_data["site_name"]} CLIF REPORT CARD")
        content.append("="*60)
        content.append("")
        content.append(f"Generated: {report_data['timestamp']}")
        content.append("")

        # Overall status
        status_symbols = {
            'complete': '✅ COMPLETE',
            'partial': '⚠️  PARTIAL',
            'noinformation': '❓ NO INFORMATION'
        }

        # Summary statistics
        results = table_results
        complete_tables = sum(1 for r in results.values() if r['status'] == 'complete')
        partial_tables = sum(1 for r in results.values() if r['status'] == 'partial')
        missing_tables = sum(1 for r in results.values() if r['status'] == 'missing')

        # Calculate total unique hospitalizations and patients across all tables
        total_unique_hospitalizations = sum(
            r.get('data_info', {}).get('unique_hospitalizations', 0) or 0
            for r in results.values()
        )
        total_unique_patients = sum(
            r.get('data_info', {}).get('unique_patients', 0) or 0
            for r in results.values()
        )

        content.append("📊 SUMMARY STATISTICS")
        content.append("-" * 30)
        content.append(f"Complete Tables: {complete_tables}")
        content.append(f"Partial Tables: {partial_tables}")
        content.append(f"Missing Tables: {missing_tables}")

        # Add unique counts if available
        if total_unique_hospitalizations > 0:
            content.append(f"Unique Hospitalizations {total_unique_hospitalizations:,}")
        if total_unique_patients > 0:
            content.append(f"Unique Patients {total_unique_patients:,}")

        content.append("")

        # Table details
        content.append("📋 TABLE VALIDATION RESULTS")
        content.append("-" * 40)

        for table_name, result in table_results.items():
            display_name = self.table_display_names.get(table_name, table_name.title())
            status = result['status']

            symbols = {'complete': '✅', 'partial': '⚠️', 'incomplete': '❌', 'missing': '📋', 'error': '🚫'}
            symbol = symbols.get(status, '❓')

            content.append(f"\n{symbol} {display_name}")
            content.append(f"Status: {status.upper()}")

            if status in ['missing', 'error']:
                content.append(f"Issue: {result.get('error', 'Unknown issue')}")
            else:
                data_info = result.get('data_info', {})
                if data_info:
                    # Show unique IDs instead of rows/columns
                    if data_info.get('unique_hospitalizations') is not None:
                        content.append(f"Unique Hospitalizations {data_info.get('unique_hospitalizations', 0):,}")

                    if data_info.get('unique_patients') is not None:
                        content.append(f"Unique Patients {data_info.get('unique_patients', 0):,}")

                    # ADT-specific: show overlapping admissions count
                    if table_name == 'adt' and data_info.get('overlapping_admissions') is not None:
                        overlapping_count = data_info.get('overlapping_admissions', 0)
                        if overlapping_count > 0:
                            content.append(f"⚠️  Overlapping Admissions {overlapping_count:,} hospitalizations")
                        else:
                            content.append("✅ No Overlapping Admissions Found")

                    # If neither unique ID is available, fall back to total rows
                    if data_info.get('unique_hospitalizations') is None and data_info.get('unique_patients') is None:
                        content.append(f"Total Rows: {data_info.get('row_count', 0):,}")

                # Show validation issues
                validation_results = result.get('validation_results', {})
                all_issues = []
                all_issues.extend(validation_results.get('schema_errors', []))
                all_issues.extend(validation_results.get('data_quality_issues', []))
                all_issues.extend(validation_results.get('other_errors', []))

                if all_issues:
                    content.append(f"Issues Found: {len(all_issues)}")
                    for i, issue in enumerate(all_issues[:3]):
                        content.append(f"  {i+1}. {issue.get('type', 'Issue')}: {issue.get('description', 'No description')}")

                    if len(all_issues) > 3:
                        content.append(f"  ... and {len(all_issues) - 3} more issues")
                elif status == 'complete':
                    content.append("All validation checks passed!")

        # Save to file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))

        print(f"Report card saved to: {output_file}")
        return report_data