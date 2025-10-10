"""
Combined report generator for multi-table validation results.

This module generates a single combined PDF report from multiple table validation results.
"""

import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path


def collect_table_results(output_dir: str, table_names: List[str]) -> Dict[str, Any]:
    """
    Collect validation results from multiple tables.

    Parameters:
    -----------
    output_dir : str
        Output directory containing validation JSON files
    table_names : list
        List of table names to collect results for

    Returns:
    --------
    dict
        Dictionary mapping table names to their validation results
    """
    final_dir = os.path.join(output_dir, 'final')
    results = {}

    for table_name in table_names:
        json_path = os.path.join(final_dir, f'{table_name}_summary_validation.json')

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    validation_data = json.load(f)
                    results[table_name] = validation_data
            except Exception as e:
                print(f"⚠️  Warning: Could not load validation results for {table_name}: {e}")
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


def _create_table_section(table_name: str, result: Dict[str, Any]) -> 'Table':
    """
    Create a detailed section for a single table's results (matching legacy format).

    Parameters:
    -----------
    table_name : str
        Name of the table
    result : dict
        Validation result dictionary for the table

    Returns:
    --------
    Table
        ReportLab Table object with detailed validation information
    """
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import Table, TableStyle, Paragraph
    from reportlab.lib import colors

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

    # Create table data
    table_data = [[display_name, status.upper()]]

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

        # Add validation issues
        validation_results = result.get('validation_results', {})
        all_issues = []
        all_issues.extend(validation_results.get('schema_errors', []))
        all_issues.extend(validation_results.get('data_quality_issues', []))
        all_issues.extend(validation_results.get('other_errors', []))

        if all_issues:
            # Count total issues (for display)
            table_data.append(['Issues Found:', str(len(all_issues))])

            # Group issues by type for better display
            issue_groups = {}
            for issue in all_issues:
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

        elif status == 'complete':
            table_data.append(['Status:', 'All validation checks passed!'])

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
                          site_name: Optional[str] = None, timezone: Optional[str] = 'UTC') -> str:
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

        story.append(_create_table_section(table_name, result))
        story.append(Spacer(1, 20))

    # Build PDF
    doc.build(story)
    return output_path


def generate_combined_report(output_dir: str, table_names: List[str],
                            site_name: Optional[str] = None,
                            timezone: Optional[str] = 'UTC') -> Optional[str]:
    """
    High-level function to generate a combined validation report.

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
        final_dir = os.path.join(output_dir, 'final')
        os.makedirs(final_dir, exist_ok=True)
        pdf_path = os.path.join(final_dir, 'combined_validation_report.pdf')

        generate_combined_pdf(table_results, pdf_path, site_name, timezone)

        return pdf_path

    except Exception as e:
        print(f"❌ Error generating combined report: {e}")
        import traceback
        traceback.print_exc()
        return None
