import os
import json
import logging
import sys
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Add local clifpy development version to path
sys.path.insert(0, '/Users/dema/WD/clifpy')

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("WARNING: reportlab not installed. Install with: pip install reportlab")

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

        # Human-readable table names
        self.table_display_names = {
            'adt': 'ADT',
            'hospitalization': 'Hospitalization',
            'labs': 'Labs',
            'medication_admin_continuous': 'Medication Admin Continuous',
            'patient': 'Patient',
            'patient_assessments': 'Patient Assessments',
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

            # Parse validation errors
            validation_errors = []
            schema_errors = []
            data_quality_issues = []

            if hasattr(table_instance, 'errors') and table_instance.errors:
                for error in table_instance.errors:
                    error_type = error.get('type', 'unknown')

                    if error_type == 'missing_columns':
                        schema_errors.append({
                            'type': 'Missing Required Columns',
                            'description': f"Required columns not found: {', '.join(error.get('columns', []))}",
                            'severity': 'high'
                        })

                    elif error_type == 'null_values':
                        column = error.get('column')
                        count = error.get('count', 0)
                        percentage = (count / data_info['row_count'] * 100) if data_info['row_count'] > 0 else 0

                        data_quality_issues.append({
                            'type': 'Missing Values',
                            'description': f"Column '{column}' has {count:,} missing values ({percentage:.1f}%)",
                            'severity': 'high' if percentage > 50 else 'medium' if percentage > 10 else 'low'
                        })

                    elif error_type in ['invalid_category', 'invalid_categorical_values']:
                        column = error.get('column')
                        invalid_values = error.get('invalid_values', error.get('values', []))

                        data_quality_issues.append({
                            'type': 'Invalid Categories',
                            'description': f"Column '{column}' contains invalid values: {', '.join(map(str, invalid_values[:3]))}{'...' if len(invalid_values) > 3 else ''}",
                            'severity': 'medium'
                        })

                    elif error_type == 'missing_categorical_values':
                        column = error.get('column')
                        missing_values = error.get('missing_values', [])
                        total_missing = error.get('total_missing', len(missing_values))
                        message = error.get('message', f"Column '{column}' is missing {total_missing} expected category values")

                        data_quality_issues.append({
                            'type': 'Missing Categorical Values',
                            'description': message,
                            'severity': 'medium'
                        })

                    elif error_type == 'datatype_castable':
                        column = error.get('column')
                        expected = error.get('expected', 'unknown')
                        actual = error.get('actual', 'unknown')
                        message = error.get('message', f"Column '{column}' has type {actual} but cannot be cast to {expected}")

                        schema_errors.append({
                            'type': 'Datatype Casting Error',
                            'description': message,
                            'severity': 'high'
                        })

                    elif error_type == 'duplicate_check':
                        # Handle duplicate check errors specially
                        message = error.get('message', '')
                        if not message and 'duplicate_rows' in error:
                            duplicate_count = error.get('duplicate_rows', 0)
                            total_rows = error.get('total_rows', 0)
                            keys = error.get('composite_keys', [])
                            keys_str = ', '.join(keys) if keys else 'composite keys'
                            message = f"Found {duplicate_count} duplicate rows out of {total_rows} total rows based on keys: {keys_str}"

                        data_quality_issues.append({
                            'type': 'Duplicate Check',
                            'description': message,
                            'severity': 'medium'
                        })

                    elif error_type == 'unit_validation':
                        # Handle unit validation errors
                        category = error.get('category', 'unknown')
                        expected_units = error.get('expected_units', [])
                        unexpected_units = error.get('unexpected_units', [])
                        message = error.get('message', '')

                        if not message:
                            if unexpected_units and expected_units:
                                message = f"Table '{table_name}' category '{category}' has unexpected units: {', '.join(unexpected_units[:3])}, expected: {', '.join(expected_units)}"
                            else:
                                message = f"Unit validation issue for category '{category}'"

                        data_quality_issues.append({
                            'type': 'Unit Validation',
                            'description': message,
                            'severity': 'medium'
                        })

                    else:
                        # For other error types, try to extract a user-friendly message
                        message = error.get('message', str(error))

                        # If message is still JSON-like, try to extract a better description
                        if isinstance(message, str) and message.startswith('{') and 'message' in message:
                            try:
                                import re
                                # Try to extract the message field from JSON-like string
                                match = re.search(r"'message':\s*'([^']*)'", message)
                                if match:
                                    message = match.group(1)
                                else:
                                    # Fallback: clean up the string
                                    message = message.replace("'", "").replace("{", "").replace("}", "")
                                    if len(message) > 200:
                                        message = message[:200] + "..."
                            except:
                                # If parsing fails, use a generic message
                                message = f"Validation issue found in table '{table_name}'"

                        validation_errors.append({
                            'type': error_type.replace('_', ' ').title(),
                            'description': message,
                            'severity': 'medium'
                        })

            validation_results.update({
                'schema_errors': schema_errors,
                'data_quality_issues': data_quality_issues,
                'other_errors': validation_errors
            })

            # Determine overall status based on new requirements
            # Red (incomplete): Missing required columns OR datatype casting errors OR 100% missing values
            has_missing_columns = any(error.get('type') == 'Missing Required Columns' for error in schema_errors)
            has_datatype_errors = any(error.get('type') == 'Datatype Casting Error' for error in schema_errors)

            # Check for 100% missing values in REQUIRED columns only (red condition)
            has_100_percent_missing_required = False

            # Get required columns from schema
            if hasattr(table_instance, 'schema') and table_instance.schema:
                required_columns = table_instance.schema.get('required_columns', [])

                for error in data_quality_issues:
                    if error.get('type') == 'Missing Values' and '(100.0%)' in error.get('description', ''):
                        # Extract column name from description
                        description = error.get('description', '')
                        if "Column '" in description:
                            try:
                                column_name = description.split("Column '")[1].split("'")[0]
                                # Check if this column is in the required_columns list
                                if column_name in required_columns:
                                    has_100_percent_missing_required = True
                                    logging.info(f"Found 100% missing values in required column: {column_name}")
                                    break
                            except Exception as e:
                                logging.warning(f"Error checking if column is required: {str(e)}")
            else:
                # Fallback: if no schema available, treat all 100% missing as problematic
                has_100_percent_missing_required = any(
                    error.get('type') == 'Missing Values' and '(100.0%)' in error.get('description', '')
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

            return {
                'status': status,
                'data_info': data_info,
                'validation_results': validation_results
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

        # Generate timestamp
        timestamp = datetime.now()

        # Create report data
        report_data = {
            'site_name': self.site_config.get('site_name', 'Unknown Site'),
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'table_results': table_results
        }

        # Generate PDF
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        self._generate_pdf_content(report_data, output_file)

        print(f"Report card saved to: {output_file}")
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
            ['INCOMPLETE', str(incomplete_tables), Paragraph('Missing required columns OR Datatype casting errors OR 100% Missing values for required columns OR Missing Table', normal_style)]
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
                table_data.append(['Issues Found:', str(len(all_issues))])

                # Group issues by type for better display
                issue_groups = {}
                for issue in all_issues:
                    issue_type = issue.get('type', 'Issue')
                    if issue_type not in issue_groups:
                        issue_groups[issue_type] = []
                    issue_groups[issue_type].append(issue.get('description', 'No description'))

                # Display grouped issues
                counter = 1
                for issue_type, descriptions in issue_groups.items():
                    if len(descriptions) == 1:
                        # Single issue - display normally
                        issue_desc_paragraph = Paragraph(descriptions[0], normal_style)
                        table_data.append([f"{counter}. {issue_type}:", issue_desc_paragraph])
                        counter += 1
                    else:
                        # Multiple issues of same type - use bullet points with professional styling
                        bullet_style = ParagraphStyle(
                            'BulletStyle',
                            parent=normal_style,
                            fontSize=8,
                            leftIndent=12,
                            bulletIndent=6,
                            spaceAfter=2
                        )
                        bullet_text = "<br/>".join([f"• {desc}" for desc in descriptions])
                        issue_desc_paragraph = Paragraph(bullet_text, bullet_style)
                        table_data.append([f"{counter}. {issue_type}:", issue_desc_paragraph])
                        counter += 1

            elif status == 'complete':
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
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#FAFBFC')),
        ]))

        return table

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
                    'severity': 'high',
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
                'severity': 'info',
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
                severity = issue.get('severity', 'medium')
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
                    'severity': severity,
                    'unique_hospitalizations': data_info.get('unique_hospitalizations', ''),
                    'unique_patients': data_info.get('unique_patients', ''),
                    'total_rows': data_info.get('row_count', '')
                })

        # Write to CSV
        if csv_data:
            fieldnames = [
                'table_name', 'status', 'issue_type', 'issue_category',
                'column_name', 'issue_description', 'severity',
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


def main():
    """Main function to generate the report card."""
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

    # Initialize generator
    data_dir = site_config.get('tables_path', '../data')
    output_dir = '../output'

    generator = ClifReportCardGenerator(data_dir, output_dir, site_config)

    try:
        # Generate PDF report card
        output_file = '../output/final/clif_report_card.pdf'
        report_data = generator.generate_pdf_report_card(target_tables, output_file)
        print(f"\n🎉 PDF Report Card Generated!")
        print(f"📁 File: {output_file}")

    except ImportError:
        # Fallback to HTML report
        output_file = '../output/final/clif_report_card.html'
        report_data = generator.generate_html_report_card(target_tables, output_file)
        print(f"\n🎉 HTML Report Card Generated!")
        print(f"📁 File: {output_file}")
        print("\nNote: Install 'reportlab' package for PDF generation: pip install reportlab")


if __name__ == "__main__":
    main()