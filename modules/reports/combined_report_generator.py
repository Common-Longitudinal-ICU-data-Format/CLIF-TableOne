"""
Combined report generator for multi-table validation results.

This module generates a single combined PDF report from multiple table validation results.
"""

import os
import json
import csv
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path


def collect_table_results(output_dir: str, table_names: List[str]) -> Dict[str, Any]:
    """
    Collect validation results from multiple tables, including user feedback if available.

    Parameters:
    -----------
    output_dir : str
        Output directory containing validation JSON files
    table_names : list
        List of table names to collect results for

    Returns:
    --------
    dict
        Dictionary mapping table names to their validation results with feedback merged
    """
    from ..utils.feedback import load_feedback

    results_dir = os.path.join(output_dir, 'final', 'results')
    results = {}

    for table_name in table_names:
        # First, load the base validation results from results subdirectory
        json_path = os.path.join(results_dir, f'{table_name}_summary_validation.json')

        validation_data = None
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    validation_data = json.load(f)
            except Exception as e:
                print(f"⚠️  Warning: Could not load validation results for {table_name}: {e}")

        # Check for user feedback file
        feedback = load_feedback(output_dir, table_name)

        if validation_data and feedback:
            # Merge feedback into validation data
            validation_data['original_status'] = validation_data.get('status', 'unknown')
            validation_data['status'] = feedback.get('adjusted_status', validation_data.get('status', 'unknown'))
            validation_data['has_feedback'] = True
            validation_data['feedback'] = feedback

            # Add feedback summary info
            validation_data['feedback_summary'] = {
                'total_errors': feedback.get('total_errors', 0),
                'accepted': feedback.get('accepted_count', 0),
                'rejected': feedback.get('rejected_count', 0),
                'pending': feedback.get('pending_count', 0),
                'status_changed': feedback.get('original_status') != feedback.get('adjusted_status'),
                'user_decisions': feedback.get('user_decisions', {})
            }

            results[table_name] = validation_data
        elif validation_data:
            # No feedback file, use original validation
            validation_data['has_feedback'] = False
            results[table_name] = validation_data
        else:
            # Table not analyzed yet
            results[table_name] = None

    return results


def aggregate_table_status(table_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aggregate status counts from multiple tables.

    Parameters:
    -----------
    table_results : dict
        Dictionary mapping table names to validation results

    Returns:
    --------
    dict
        Aggregated status counts and table lists
    """
    status_counts = {
        'complete': 0,
        'partial': 0,
        'incomplete': 0,
        'not_analyzed': 0
    }

    tables_by_status = {
        'complete': [],
        'partial': [],
        'incomplete': [],
        'not_analyzed': []
    }

    for table_name, result in table_results.items():
        if result is None:
            status_counts['not_analyzed'] += 1
            tables_by_status['not_analyzed'].append(table_name)
        else:
            status = result.get('status', 'unknown')
            if status in status_counts:
                status_counts[status] += 1
                tables_by_status[status].append(table_name)

    return {
        'status_counts': status_counts,
        'tables_by_status': tables_by_status,
        'total_tables': len(table_results),
        'analyzed_tables': len(table_results) - status_counts['not_analyzed']
    }


# Table display names mapping (matching legacy report card)
TABLE_DISPLAY_NAMES = {
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


def _create_table_section(table_name: str, result: Dict[str, Any], timezone: Optional[str] = None) -> 'Table':
    """
    Create a detailed section for a single table's results (matching legacy format).

    Parameters:
    -----------
    table_name : str
        Name of the table
    result : dict
        Validation result dictionary for the table
    timezone : str, optional
        Configured timezone for filtering errors

    Returns:
    --------
    Table
        ReportLab Table object with detailed validation information
    """
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import Table, TableStyle, Paragraph
    from reportlab.lib import colors
    from ..utils.validation import classify_errors_by_status_impact

    # Get styles for text wrapping
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

    display_name = TABLE_DISPLAY_NAMES.get(table_name, table_name.replace('_', ' ').title())
    status = result.get('status', 'unknown')
    original_status = result.get('original_status')
    has_feedback = result.get('has_feedback', False)
    feedback_summary = result.get('feedback_summary', {})

    # Use the exact same background colors as the Status Overview table
    status_colors_exact = {
        'complete': colors.HexColor('#E8F5E8'),      # Same soft green background
        'partial': colors.HexColor('#FFF4E6'),       # Same soft orange background
        'incomplete': colors.HexColor('#FFEAEA'),    # Same soft red background
        'missing': colors.HexColor('#F5F6FA'),       # Same light gray background
        'error': colors.HexColor('#FFEAEA'),         # Same soft red background
        'not_analyzed': colors.HexColor('#F5F6FA')   # Same light gray background
    }

    color = status_colors_exact.get(status, colors.HexColor('#F5F6FA'))

    # Create table data - show status change if feedback was provided
    if has_feedback and feedback_summary.get('status_changed'):
        status_display = f"{original_status.upper()} → {status.upper()}"
    else:
        status_display = status.upper()

    table_data = [[display_name, status_display]]

    if status in ['missing', 'error', 'not_analyzed']:
        error_msg = result.get('error', 'Table not analyzed or data not available')
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

            # If neither unique ID is available, fall back to total rows
            if data_info.get('unique_hospitalizations') is None and data_info.get('unique_patients') is None:
                table_data.append(['Total Rows:', f"{data_info.get('row_count', 0):,}"])

        # Classify and add validation issues
        errors = result.get('errors', {})

        # Get required columns if available (for proper classification)
        required_columns = []
        # TODO: Get required columns from schema if available

        # Classify errors into status-affecting and informational
        classified_errors = classify_errors_by_status_impact(errors, required_columns, table_name, timezone)
        status_affecting = classified_errors['status_affecting']
        informational = classified_errors['informational']

        # Count errors
        status_affecting_count = sum([
            len(status_affecting.get('schema_errors', [])),
            len(status_affecting.get('data_quality_issues', [])),
            len(status_affecting.get('other_errors', []))
        ])

        informational_count = sum([
            len(informational.get('schema_errors', [])),
            len(informational.get('data_quality_issues', [])),
            len(informational.get('other_errors', []))
        ])

        total_issues = status_affecting_count + informational_count

        if total_issues > 0:
            # Show total issues found
            table_data.append(['Issues Found:', f"{total_issues} ({status_affecting_count} status-affecting, {informational_count} informational)"])

            # Display status-affecting errors first (these determine the validation status)
            if status_affecting_count > 0:
                # Combine all status-affecting errors for display
                all_status_affecting = []
                all_status_affecting.extend(status_affecting.get('schema_errors', []))
                all_status_affecting.extend(status_affecting.get('data_quality_issues', []))
                all_status_affecting.extend(status_affecting.get('other_errors', []))

                # Group by type
                issue_groups = {}
                for issue in all_status_affecting:
                    issue_type = issue.get('type', 'Issue')
                    if issue_type not in issue_groups:
                        issue_groups[issue_type] = []
                    issue_groups[issue_type].append(issue.get('description', 'No description'))

                # Display grouped issues
                counter = 1
                MAX_ISSUES_TO_DISPLAY = 5  # Reduced to prevent overflow
                MAX_DESC_LENGTH = 500  # Max characters per description
                MAX_TOTAL_ISSUES = 8  # Total issues to display across all types

                total_issues_displayed = 0
                for issue_type, descriptions in list(issue_groups.items())[:MAX_TOTAL_ISSUES]:
                    if total_issues_displayed >= MAX_TOTAL_ISSUES:
                        # Add note about remaining issues
                        remaining_types = len(issue_groups) - counter + 1
                        if remaining_types > 0:
                            table_data.append(['Note:', Paragraph(f"<i>Plus {remaining_types} more issue types...</i>", normal_style)])
                        break

                    # Truncate long descriptions (e.g., lists of missing categorical values)
                    truncated_descriptions = []
                    for desc in descriptions:
                        if len(desc) > MAX_DESC_LENGTH:
                            # For lists of missing values, show first few and count
                            if 'Missing categorical values' in desc or 'Expected categorical' in desc:
                                # Extract the list portion if present
                                parts = desc.split(':', 1)
                                if len(parts) == 2:
                                    prefix = parts[0]
                                    values = parts[1].strip()
                                    # Count items
                                    value_list = [v.strip() for v in values.split(',')]
                                    if len(value_list) > 10:
                                        shown = ', '.join(value_list[:10])
                                        truncated_desc = f"{prefix}: {shown}, ... and {len(value_list)-10} more values"
                                    else:
                                        truncated_desc = desc[:MAX_DESC_LENGTH] + "..."
                                else:
                                    truncated_desc = desc[:MAX_DESC_LENGTH] + "..."
                            else:
                                truncated_desc = desc[:MAX_DESC_LENGTH] + "..."
                            truncated_descriptions.append(truncated_desc)
                        else:
                            truncated_descriptions.append(desc)

                    if len(truncated_descriptions) == 1:
                        # Single issue - display normally
                        issue_desc_paragraph = Paragraph(truncated_descriptions[0], normal_style)
                        table_data.append([f"{counter}. {issue_type}:", issue_desc_paragraph])
                        counter += 1
                        total_issues_displayed += 1
                    else:
                        # Multiple issues of same type - use bullet points
                        bullet_style = ParagraphStyle(
                            'BulletStyle',
                            parent=normal_style,
                            fontSize=8,
                            leftIndent=12,
                            bulletIndent=6,
                            spaceAfter=2
                        )

                        # Limit number of bullets displayed
                        display_descriptions = truncated_descriptions[:MAX_ISSUES_TO_DISPLAY]
                        truncated_count = len(descriptions) - len(display_descriptions)

                        bullet_points = []
                        for desc in display_descriptions:
                            # Further truncate bullets if needed
                            if len(desc) > 200:
                                desc = desc[:200] + "..."
                            bullet_points.append(f"• {desc}")

                        if truncated_count > 0:
                            bullet_points.append(f"• <i>...and {truncated_count} more similar issues</i>")

                        bullet_text = "<br/>".join(bullet_points)
                        issue_desc_paragraph = Paragraph(bullet_text, bullet_style)
                        table_data.append([f"{counter}. {issue_type}:", issue_desc_paragraph])
                        counter += 1
                        total_issues_displayed += len(display_descriptions)

            # If there are informational issues, add a note about them
            if informational_count > 0:
                info_text = f"<i>Plus {informational_count} informational issue(s) that do not affect status</i>"
                table_data.append(['Note:', Paragraph(info_text, normal_style)])

        elif status == 'complete':
            table_data.append(['Status:', 'All validation checks passed!'])

        # Add feedback information if available
        if has_feedback and feedback_summary.get('status_changed'):
            feedback_text = f"Status updated based on user feedback: {feedback_summary['rejected']} error(s) marked as not applicable"
            table_data.append(['Note:', Paragraph(f"<i>{feedback_text}</i>", normal_style)])
        elif has_feedback and feedback_summary.get('rejected', 0) > 0:
            feedback_text = f"User feedback: {feedback_summary['rejected']} error(s) marked as not applicable"
            table_data.append(['Note:', Paragraph(f"<i>{feedback_text}</i>", normal_style)])

    # Create table with professional styling
    from reportlab.lib.units import inch
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


def generate_combined_pdf(table_results: Dict[str, Any], output_path: str,
                          site_name: Optional[str] = None, timezone: Optional[str] = 'UTC',
                          used_sampling: bool = False) -> str:
    """
    Generate a combined PDF report from multiple table validation results.

    Parameters:
    -----------
    table_results : dict
        Dictionary mapping table names to validation results
    output_path : str
        Path where PDF should be saved
    site_name : str, optional
        Name of the site/hospital
    timezone : str, optional
        Configured timezone (defaults to UTC)
    used_sampling : bool, optional
        Whether a 1k ICU sample was used (defaults to False)

    Returns:
    --------
    str
        Path to generated PDF file
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        raise ImportError("reportlab is required for PDF generation. Install with: pip install reportlab")

    # Aggregate status information
    aggregated = aggregate_table_status(table_results)

    # Create document
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # Professional color palette (matching individual table reports)
    primary_color = colors.HexColor('#1F4E79')      # Deep blue
    text_dark = colors.HexColor('#2C3E50')          # Dark gray
    text_medium = colors.HexColor('#5D6D7E')        # Medium gray
    header_bg = colors.HexColor('#F5F6FA')          # Light gray header
    status_complete = colors.HexColor('#E8F5E8')    # Soft green background
    status_partial = colors.HexColor('#FFF4E6')     # Soft orange background
    status_incomplete = colors.HexColor('#FFEAEA')  # Soft red background

    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=22,
        textColor=primary_color,
        spaceAfter=24,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )

    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=text_dark,
        spaceAfter=10,
        spaceBefore=16,
        fontName='Helvetica-Bold'
    )

    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=9,
        textColor=text_medium,
        fontName='Helvetica'
    )

    # Timestamp in top right corner
    timestamp_style = ParagraphStyle(
        'TimestampStyle',
        parent=styles['Normal'],
        fontSize=8,
        textColor=text_medium,
        alignment=1,  # TA_RIGHT
        fontName='Helvetica'
    )

    timestamp_table = Table([[Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", timestamp_style)]],
                           colWidths=[7.5*inch])
    timestamp_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (0, 0), 'RIGHT'),
        ('VALIGN', (0, 0), (0, 0), 'TOP'),
        ('TOPPADDING', (0, 0), (0, 0), -24),
        ('BOTTOMPADDING', (0, 0), (0, 0), 2),
    ]))
    story.append(timestamp_table)

    # Title
    title_text = f"{site_name + ' ' if site_name else ''}CLIF Data Report Card"
    story.append(Paragraph(title_text, title_style))
    story.append(Paragraph("Combined Validation Report", heading_style))
    story.append(Spacer(1, 0.2 * inch))

    # Add sampling note if applicable
    if used_sampling:
        sample_note_style = ParagraphStyle(
            'SampleNote',
            parent=normal_style,
            fontSize=10,
            textColor=colors.HexColor('#007ACC'),  # Blue color for emphasis
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        story.append(Paragraph("📊 This report was generated using a 1k ICU sample (stratified by year) for eligible tables", sample_note_style))
        story.append(Spacer(1, 0.15 * inch))

    # Status Report Overview
    story.append(Paragraph("Status Report Overview", heading_style))

    status_counts = aggregated['status_counts']

    # Create professional status overview table
    status_data = [
        ['Status', 'Table Count', 'Criteria'],
        ['COMPLETE', str(status_counts['complete']),
         Paragraph('All required columns present, all categorical values present', normal_style)],
        ['PARTIAL', str(status_counts['partial']),
         Paragraph('All required columns present, but missing some categorical values', normal_style)],
        ['INCOMPLETE', str(status_counts['incomplete']),
         Paragraph('One or more of: (1) Missing required columns, (2) Non-castable datatype errors, (3) Required columns with no data (100% null count)', normal_style)]
    ]

    status_table = Table(status_data, colWidths=[1.2*inch, 0.8*inch, 4.5*inch])

    # Build table style with color-coded status column
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

    status_table.setStyle(TableStyle(table_style))
    story.append(status_table)
    story.append(Spacer(1, 0.3 * inch))

    # Table-by-table validation results (detailed sections matching legacy format)
    story.append(Paragraph("Table Validation Results", heading_style))

    # Sort tables by status (complete, partial, incomplete, not_analyzed)
    sorted_tables = []
    tables_by_status = aggregated['tables_by_status']
    for status in ['complete', 'partial', 'incomplete']:
        for table_name in tables_by_status[status]:
            sorted_tables.append((table_name, status, table_results[table_name]))

    # Add not analyzed tables at the end
    for table_name in tables_by_status['not_analyzed']:
        sorted_tables.append((table_name, 'not_analyzed', None))

    # Create detailed sections for each table (matching legacy format)
    for table_name, status, result in sorted_tables:
        # Handle not analyzed tables by creating a minimal result dict
        if result is None:
            result = {
                'status': 'not_analyzed',
                'error': 'Table not analyzed'
            }

        story.append(_create_table_section(table_name, result, timezone))
        story.append(Spacer(1, 20))

    # Build PDF
    doc.build(story)
    return output_path


def generate_consolidated_csv(table_results: Dict[str, Any], output_path: str,
                               timezone: Optional[str] = 'UTC') -> str:
    """
    Generate a consolidated CSV report from multiple table validation results.

    Parameters:
    -----------
    table_results : dict
        Dictionary mapping table names to validation results
    output_path : str
        Path where CSV should be saved
    timezone : str, optional
        Configured timezone for filtering errors

    Returns:
    --------
    str
        Path to generated CSV file
    """
    from ..utils.validation import classify_errors_by_status_impact

    rows = []

    # Process each table
    for table_name, result in table_results.items():
        display_name = TABLE_DISPLAY_NAMES.get(table_name, table_name.replace('_', ' ').title())

        if result is None:
            # Table not analyzed
            rows.append({
                'table_name': display_name,
                'status': 'missing',
                'issue_type': 'Table Status',
                'issue_category': 'Critical',
                'column_name': '',
                'issue_description': 'Data file not found or table not analyzed',
                'user_feedback': '',
                'unique_hospitalizations': '',
                'unique_patients': '',
                'total_rows': ''
            })
            continue

        status = result.get('status', 'unknown')
        original_status = result.get('original_status')
        has_feedback = result.get('has_feedback', False)
        feedback_summary = result.get('feedback_summary', {})
        data_info = result.get('data_info', {})
        unique_hosp = data_info.get('unique_hospitalizations', '')
        unique_patients = data_info.get('unique_patients', '')
        total_rows = data_info.get('row_count', '')

        # Show status change if feedback was provided
        status_display = status
        table_description = 'Table loaded successfully'
        if has_feedback and feedback_summary.get('status_changed'):
            status_display = f"{original_status} → {status}"
            table_description = f"Table loaded successfully (Status updated based on user feedback)"

        # Add table summary row
        rows.append({
            'table_name': display_name,
            'status': status_display,
            'issue_type': 'Table Summary',
            'issue_category': 'Info',
            'column_name': '',
            'issue_description': table_description,
            'user_feedback': '',
            'unique_hospitalizations': unique_hosp if unique_hosp else '',
            'unique_patients': unique_patients if unique_patients else '',
            'total_rows': total_rows
        })

        # Get and classify errors
        errors = result.get('errors', {})

        # Get required columns if available
        required_columns = []  # TODO: Get from schema if available

        # Classify errors
        classified_errors = classify_errors_by_status_impact(errors, required_columns, table_name, timezone)

        # Process all errors (both status-affecting and informational)
        all_errors = []

        # Add status-affecting errors with proper categorization
        for error in classified_errors['status_affecting'].get('schema_errors', []):
            all_errors.append({
                'type': error.get('type', 'Schema Error'),
                'category': 'Schema',
                'description': error.get('description', ''),
                'details': error.get('details', {})
            })

        for error in classified_errors['status_affecting'].get('data_quality_issues', []):
            all_errors.append({
                'type': error.get('type', 'Data Quality'),
                'category': error.get('category', 'Data Quality'),
                'description': error.get('description', ''),
                'details': error.get('details', {})
            })

        for error in classified_errors['status_affecting'].get('other_errors', []):
            all_errors.append({
                'type': error.get('type', 'Other'),
                'category': error.get('category', 'Other'),
                'description': error.get('description', ''),
                'details': error.get('details', {})
            })

        # Add informational errors
        for error in classified_errors['informational'].get('schema_errors', []):
            all_errors.append({
                'type': error.get('type', 'Schema Information'),
                'category': 'Schema',
                'description': error.get('description', ''),
                'details': error.get('details', {})
            })

        for error in classified_errors['informational'].get('data_quality_issues', []):
            all_errors.append({
                'type': error.get('type', 'Data Quality'),
                'category': error.get('category', 'Data Quality'),
                'description': error.get('description', ''),
                'details': error.get('details', {})
            })

        for error in classified_errors['informational'].get('other_errors', []):
            all_errors.append({
                'type': error.get('type', 'Other'),
                'category': error.get('category', 'Other'),
                'description': error.get('description', ''),
                'details': error.get('details', {})
            })

        # Add each error as a row
        for error in all_errors:
            # Extract column name from details if available
            details = error.get('details', {})
            column_name = details.get('column', '')

            # Map raw_type to category if category is not already set
            if error['category'] == 'Data Quality' and 'raw_type' in error:
                raw_type = error.get('raw_type', '')
                if 'categorical' in raw_type.lower():
                    error['category'] = 'Categorical'

            # Check if this error has user feedback
            user_feedback_status = ''
            if has_feedback and feedback_summary.get('user_decisions'):
                # Try to find this error in user decisions
                from ..utils.feedback import create_error_id
                error_id = create_error_id(error)
                if error_id in feedback_summary['user_decisions']:
                    decision = feedback_summary['user_decisions'][error_id].get('decision', '')
                    if decision == 'rejected':
                        user_feedback_status = 'Not Applicable'
                    elif decision == 'accepted':
                        user_feedback_status = 'Confirmed'

            rows.append({
                'table_name': display_name,
                'status': status_display,
                'issue_type': error['type'],
                'issue_category': error['category'],
                'column_name': column_name,
                'issue_description': error['description'],
                'user_feedback': user_feedback_status,
                'unique_hospitalizations': unique_hosp if unique_hosp else '',
                'unique_patients': unique_patients if unique_patients else '',
                'total_rows': total_rows
            })

    # Write CSV file
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'table_name', 'status', 'issue_type', 'issue_category',
            'column_name', 'issue_description', 'user_feedback',
            'unique_hospitalizations', 'unique_patients', 'total_rows'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(rows)

    return output_path


def generate_combined_report(output_dir: str, table_names: List[str],
                            site_name: Optional[str] = None,
                            timezone: Optional[str] = 'UTC',
                            used_sampling: bool = False) -> Optional[str]:
    """
    High-level function to generate a combined validation report (PDF and CSV).

    Parameters:
    -----------
    output_dir : str
        Output directory containing validation JSON files
    table_names : list
        List of table names to include in the report
    site_name : str, optional
        Name of the site/hospital
    timezone : str, optional
        Configured timezone (defaults to UTC)
    used_sampling : bool, optional
        Whether a 1k ICU sample was used for validation (defaults to False)

    Returns:
    --------
    str or None
        Path to generated PDF file, or None if generation failed
    """
    try:
        # Collect results
        table_results = collect_table_results(output_dir, table_names)

        # Check if any tables were analyzed
        analyzed_count = sum(1 for result in table_results.values() if result is not None)
        if analyzed_count == 0:
            print("⚠️  No analyzed tables found. Run validation first.")
            return None

        # Generate PDF
        reports_dir = os.path.join(output_dir, 'final', 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        pdf_path = os.path.join(reports_dir, 'combined_validation_report.pdf')

        generate_combined_pdf(table_results, pdf_path, site_name, timezone, used_sampling)

        # Generate consolidated CSV
        results_dir = os.path.join(output_dir, 'final', 'results')
        os.makedirs(results_dir, exist_ok=True)
        csv_path = os.path.join(results_dir, 'consolidated_validation.csv')
        generate_consolidated_csv(table_results, csv_path, timezone)
        print(f"✅ Consolidated CSV saved: consolidated_validation.csv")

        return pdf_path

    except Exception as e:
        print(f"❌ Error generating combined report: {e}")
        import traceback
        traceback.print_exc()
        return None
