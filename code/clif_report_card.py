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

        try:
            # Redirect stdout to capture clifpy's verbose output
            from io import StringIO
            import contextlib

            captured_output = StringIO()

            with contextlib.redirect_stdout(captured_output):
                # Initialize table instance with proper parameters - redirect output to .logs
                table_instance = table_class.from_file(
                    data_directory=self.data_dir,
                    filetype=self.site_config.get('file_type'),
                    timezone=self.site_config.get('timezone'),
                    output_directory=self.logs_dir  # Redirect clifpy output files to .logs
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
            if table_name == 'adt' and hasattr(table_instance, 'check_overlapping_admissions'):
                try:
                    overlapping_count = table_instance.check_overlapping_admissions(
                        save_overlaps=True,
                        overlaps_output_directory=self.logs_dir
                    )
                    data_info['overlapping_admissions'] = overlapping_count
                    logging.info(f"ADT overlapping admissions check: {overlapping_count} hospitalizations with overlaps")
                except Exception as e:
                    logging.warning(f"ADT overlapping admissions check failed: {str(e)}")
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

                    else:
                        validation_errors.append({
                            'type': error_type.replace('_', ' ').title(),
                            'description': str(error),
                            'severity': 'medium'
                        })

            validation_results.update({
                'schema_errors': schema_errors,
                'data_quality_issues': data_quality_issues,
                'other_errors': validation_errors
            })

            # Determine overall status
            if schema_errors:
                status = 'incomplete'
            elif data_quality_issues:
                high_severity_issues = [issue for issue in data_quality_issues if issue['severity'] == 'high']
                status = 'incomplete' if high_severity_issues else 'partial'
            elif validation_errors:
                status = 'partial'
            else:
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

    def calculate_overall_site_status(self, table_results: Dict[str, Dict[str, Any]]) -> str:
        """
        Calculate overall site status based on individual table results.

        Args:
            table_results: Dictionary mapping table names to validation results

        Returns:
            Overall status: 'complete', 'partial', or 'noinformation'
        """
        statuses = [result['status'] for result in table_results.values()]

        # If any tables are missing or have errors, we have incomplete information
        if 'missing' in statuses or 'error' in statuses:
            return 'noinformation'

        # If all tables are complete
        if all(status == 'complete' for status in statuses):
            return 'complete'

        # Otherwise, partial completeness
        return 'partial'

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

        # Calculate overall status
        overall_status = self.calculate_overall_site_status(table_results)

        # Generate timestamp
        timestamp = datetime.now()

        # Create report data
        report_data = {
            'site_name': self.site_config.get('site_name', 'Unknown Site'),
            'site_id': self.site_config.get('site_id', 'unknown_site'),
            'contact': self.site_config.get('contact', ''),
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'overall_status': overall_status,
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

        # Define styles
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#2C3E50')
        )

        header_style = ParagraphStyle(
            'CustomHeader',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.HexColor('#34495E')
        )

        normal_style = styles['Normal']

        # Title
        title = Paragraph("🏥 CLIF Data Validation Report Card", title_style)
        story.append(title)
        story.append(Spacer(1, 12))

        # Site information
        site_info = [
            ['Site Name:', report_data['site_name']],
            ['Site ID:', report_data['site_id']],
            ['Contact:', report_data['contact']],
            ['Generated:', report_data['timestamp']]
        ]

        site_table = Table(site_info, colWidths=[2*inch, 4*inch])
        site_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(site_table)
        story.append(Spacer(1, 20))

        # Overall status
        status_colors = {
            'complete': colors.green,
            'partial': colors.orange,
            'noinformation': colors.grey
        }

        status_labels = {
            'complete': 'COMPLETE ✅',
            'partial': 'PARTIAL ⚠️',
            'noinformation': 'NO INFORMATION ❓'
        }

        status_color = status_colors.get(report_data['overall_status'], colors.grey)
        status_label = status_labels.get(report_data['overall_status'], 'UNKNOWN')

        status_table = Table([[f'Overall Status: {status_label}']], colWidths=[6*inch])
        status_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 16),
            ('BACKGROUND', (0, 0), (-1, -1), status_color),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
            ('BOX', (0, 0), (-1, -1), 2, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ]))
        story.append(status_table)
        story.append(Spacer(1, 30))

        # Summary statistics
        results = report_data['table_results']
        total_tables = len(results)
        complete_tables = sum(1 for r in results.values() if r['status'] == 'complete')
        partial_tables = sum(1 for r in results.values() if r['status'] == 'partial')
        missing_tables = sum(1 for r in results.values() if r['status'] == 'missing')
        total_rows = sum(r.get('data_info', {}).get('row_count', 0) for r in results.values())

        summary_header = Paragraph("📊 Summary Statistics", header_style)
        story.append(summary_header)

        # Calculate total unique hospitalizations and patients across all tables
        total_unique_hospitalizations = sum(
            r.get('data_info', {}).get('unique_hospitalizations', 0) or 0
            for r in results.values()
        )
        total_unique_patients = sum(
            r.get('data_info', {}).get('unique_patients', 0) or 0
            for r in results.values()
        )

        summary_data = [
            ['Complete Tables', str(complete_tables)],
            ['Partial Tables', str(partial_tables)],
            ['Missing Tables', str(missing_tables)]
        ]

        # Add unique counts if available
        if total_unique_hospitalizations > 0:
            summary_data.append(['Unique Hospitalizations', f'{total_unique_hospitalizations:,}'])
        if total_unique_patients > 0:
            summary_data.append(['Unique Patients', f'{total_unique_patients:,}'])

        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 30))

        # Table validation results
        table_header = Paragraph("📋 Table Validation Results", header_style)
        story.append(table_header)

        for table_name, result in report_data['table_results'].items():
            story.append(self._create_table_section(table_name, result))
            story.append(Spacer(1, 20))

        # Build PDF
        doc.build(story)

    def _create_table_section(self, table_name: str, result: Dict[str, Any]):
        """Create a section for a single table's results."""
        display_name = self.table_display_names.get(table_name, table_name.title())
        status = result['status']

        # Status symbols and colors
        status_info = {
            'complete': ('✅', colors.green),
            'partial': ('⚠️', colors.orange),
            'incomplete': ('❌', colors.red),
            'missing': ('📋', colors.grey),
            'error': ('🚫', colors.red)
        }

        symbol, color = status_info.get(status, ('❓', colors.grey))

        # Create table data
        table_data = [[f"{symbol} {display_name}", status.upper()]]

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
                        table_data.append(['⚠️ Overlapping Admissions', f"{overlapping_count:,} hospitalizations"])
                    else:
                        table_data.append(['✅ Overlapping Admissions', "None found"])

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
                for i, issue in enumerate(all_issues[:3]):  # Show max 3 issues
                    issue_desc = f"{issue.get('type', 'Issue')}: {issue.get('description', 'No description')}"
                    # Truncate long descriptions
                    if len(issue_desc) > 80:
                        issue_desc = issue_desc[:77] + "..."
                    table_data.append([f"Issue {i+1}:", issue_desc])

                if len(all_issues) > 3:
                    table_data.append(['...', f"and {len(all_issues) - 3} more issues"])

            elif status == 'complete':
                table_data.append(['Status:', 'All validation checks passed!'])

        # Create table
        table = Table(table_data, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 0), (-1, 0), color),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))

        return table

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

        # Calculate overall status
        overall_status = self.calculate_overall_site_status(table_results)
        timestamp = datetime.now()

        # Create report data
        report_data = {
            'site_name': self.site_config.get('site_name', 'Unknown Site'),
            'site_id': self.site_config.get('site_id', 'unknown_site'),
            'contact': self.site_config.get('contact', ''),
            'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'overall_status': overall_status,
            'table_results': table_results
        }

        # Generate text content
        content = []
        content.append("="*60)
        content.append("🏥 CLIF DATA VALIDATION REPORT CARD")
        content.append("="*60)
        content.append("")
        content.append(f"Site Name: {report_data['site_name']}")
        content.append(f"Site ID: {report_data['site_id']}")
        content.append(f"Contact: {report_data['contact']}")
        content.append(f"Generated: {report_data['timestamp']}")
        content.append("")

        # Overall status
        status_symbols = {
            'complete': '✅ COMPLETE',
            'partial': '⚠️  PARTIAL',
            'noinformation': '❓ NO INFORMATION'
        }
        status_display = status_symbols.get(overall_status, '❓ UNKNOWN')
        content.append(f"Overall Status: {status_display}")
        content.append("")

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

    # Print summary
    print(f"🏥 Site: {report_data['site_name']}")
    print(f"📊 Overall Status: {report_data['overall_status'].upper()}")


if __name__ == "__main__":
    main()