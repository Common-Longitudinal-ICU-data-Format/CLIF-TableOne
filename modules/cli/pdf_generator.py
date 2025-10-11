"""PDF report generator for validation results."""

import os
from typing import Dict, Any, Optional
from datetime import datetime


class ValidationPDFGenerator:
    """Generate PDF reports from validation JSON results."""

    def __init__(self):
        """Initialize the PDF generator."""
        try:
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
            from reportlab.lib import colors
            from reportlab.lib.enums import TA_CENTER, TA_LEFT

            self.reportlab_available = True
            self.letter = letter
            self.A4 = A4
            self.getSampleStyleSheet = getSampleStyleSheet
            self.ParagraphStyle = ParagraphStyle
            self.inch = inch
            self.SimpleDocTemplate = SimpleDocTemplate
            self.Paragraph = Paragraph
            self.Spacer = Spacer
            self.Table = Table
            self.TableStyle = TableStyle
            self.PageBreak = PageBreak
            self.colors = colors
            self.TA_CENTER = TA_CENTER
            self.TA_LEFT = TA_LEFT
        except ImportError:
            self.reportlab_available = False

    def is_available(self) -> bool:
        """Check if reportlab is available."""
        return self.reportlab_available

    def generate_validation_pdf(self, validation_data: Dict[str, Any],
                                table_name: str, output_path: str,
                                site_name: Optional[str] = None,
                                timezone: Optional[str] = None,
                                feedback: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a PDF report from validation JSON data.

        Parameters:
        -----------
        validation_data : dict
            Validation results from JSON
        table_name : str
            Name of the table
        output_path : str
            Path where PDF should be saved
        site_name : str, optional
            Name of the site/hospital
        timezone : str, optional
            Configured timezone for filtering errors (defaults to UTC)
        feedback : dict, optional
            User feedback on validation errors with adjusted status

        Returns:
        --------
        str
            Path to generated PDF file
        """
        if not self.reportlab_available:
            raise ImportError("reportlab is required for PDF generation. Install with: pip install reportlab")

        # Create document
        doc = self.SimpleDocTemplate(output_path, pagesize=self.letter)
        story = []
        styles = self.getSampleStyleSheet()

        # Professional color palette (matching CLIF Report Card)
        primary_color = self.colors.HexColor('#1F4E79')      # Deep blue
        text_dark = self.colors.HexColor('#2C3E50')          # Dark gray
        text_medium = self.colors.HexColor('#5D6D7E')        # Medium gray
        header_bg = self.colors.HexColor('#F5F6FA')          # Light gray header
        status_complete = self.colors.HexColor('#E8F5E8')    # Soft green background
        status_partial = self.colors.HexColor('#FFF4E6')     # Soft orange background
        status_incomplete = self.colors.HexColor('#FFEAEA')  # Soft red background

        # Custom styles
        title_style = self.ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=22,
            textColor=primary_color,
            spaceAfter=24,
            alignment=self.TA_CENTER,
            fontName='Helvetica-Bold'
        )

        heading_style = self.ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=text_dark,
            spaceAfter=10,
            spaceBefore=16,
            fontName='Helvetica-Bold'
        )

        normal_style = self.ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=9,
            textColor=text_medium,
            fontName='Helvetica'
        )

        # Timestamp in top right corner
        timestamp_style = self.ParagraphStyle(
            'TimestampStyle',
            parent=styles['Normal'],
            fontSize=8,
            textColor=text_medium,
            alignment=1,  # TA_RIGHT
            fontName='Helvetica'
        )

        timestamp_table = self.Table([[self.Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", timestamp_style)]],
                                     colWidths=[7.5*self.inch])
        timestamp_table.setStyle(self.TableStyle([
            ('ALIGN', (0, 0), (0, 0), 'RIGHT'),
            ('VALIGN', (0, 0), (0, 0), 'TOP'),
            ('TOPPADDING', (0, 0), (0, 0), -24),
            ('BOTTOMPADDING', (0, 0), (0, 0), 2),
        ]))
        story.append(timestamp_table)

        # Title
        title_text = f"{site_name + ' ' if site_name else ''}CLIF Data Report Card"
        story.append(self.Paragraph(title_text, title_style))
        story.append(self.Paragraph(f"{table_name.title()} Table", heading_style))
        story.append(self.Spacer(1, 0.2 * self.inch))

        # Status Report Overview (matching CLIF Report Card)
        story.append(self.Paragraph("Status Report Overview", heading_style))

        status = validation_data.get('status', 'unknown')
        original_status = None

        # Check if feedback was provided and status was adjusted
        if feedback:
            original_status = feedback.get('original_status')
            adjusted_status = feedback.get('adjusted_status')
            if adjusted_status and adjusted_status != original_status:
                status = adjusted_status  # Use the adjusted status

        # Determine which row should be highlighted
        status_counts = {'complete': 0, 'partial': 0, 'incomplete': 0}
        if status in status_counts:
            status_counts[status] = 1

        # Create professional status overview table
        status_data = [
            ['Status', 'Table Count', 'Criteria'],
            ['COMPLETE', str(status_counts['complete']),
             self.Paragraph('All required columns present, all categorical values present', normal_style)],
            ['PARTIAL', str(status_counts['partial']),
             self.Paragraph('All required columns present, but missing some categorical values', normal_style)],
            ['INCOMPLETE', str(status_counts['incomplete']),
             self.Paragraph('One or more of: (1) Missing required columns, (2) Non-castable datatype errors, (3) Required columns with no data (100% null count)', normal_style)]
        ]

        status_table = self.Table(status_data, colWidths=[1.2*self.inch, 0.8*self.inch, 4.5*self.inch])

        # Build table style with color-coded status column
        table_style = [
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (-1, 0), text_dark),
            ('TEXTCOLOR', (0, 1), (-1, -1), text_medium),
            ('BACKGROUND', (0, 0), (-1, 0), header_bg),
            ('GRID', (0, 0), (-1, -1), 0.5, self.colors.HexColor('#DADADA')),
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

        status_table.setStyle(self.TableStyle(table_style))
        story.append(status_table)
        story.append(self.Spacer(1, 0.3 * self.inch))

        # Table Validation Results (matching CLIF Report Card format)
        story.append(self.Paragraph("Table Validation Results", heading_style))
        data_info = validation_data.get('data_info', {})

        if 'error' not in data_info:
            # Show unique IDs instead of rows/columns (matching CLIF Report Card)
            # Show status change if feedback was provided
            status_display = status.upper()
            if feedback and original_status and original_status != status:
                status_display = f"{original_status.upper()} → {status.upper()}"

            data_rows = [
                [table_name.title(), status_display]
            ]

            # Add unique hospitalizations if available
            if data_info.get('unique_hospitalizations') is not None:
                data_rows.append(['Unique Hospitalizations:', f"{data_info.get('unique_hospitalizations', 0):,}"])

            # Add unique patients if available
            if data_info.get('unique_patients') is not None:
                data_rows.append(['Unique Patients:', f"{data_info.get('unique_patients', 0):,}"])

            # If neither unique ID is available, fall back to total rows
            if data_info.get('unique_hospitalizations') is None and data_info.get('unique_patients') is None:
                data_rows.append(['Total Rows:', f"{data_info.get('row_count', 0):,}"])

            # Use exact same colors as Status Overview table
            status_color_exact = {
                'complete': status_complete,
                'partial': status_partial,
                'incomplete': status_incomplete
            }.get(status, header_bg)

            data_table = self.Table(data_rows, colWidths=[2.5*self.inch, 4*self.inch])
            data_table.setStyle(self.TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
                ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('BACKGROUND', (0, 0), (-1, 0), status_color_exact),
                ('TEXTCOLOR', (0, 0), (-1, 0), text_dark),
                ('TEXTCOLOR', (0, 1), (-1, -1), text_medium),
                ('GRID', (0, 0), (-1, -1), 0.5, self.colors.HexColor('#DADADA')),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                ('BACKGROUND', (0, 1), (-1, -1), self.colors.white),
            ]))
            story.append(data_table)
            story.append(self.Spacer(1, 0.3 * self.inch))

        # Classify errors by status impact
        from modules.utils.validation import classify_errors_by_status_impact

        errors = validation_data.get('errors', {})

        # Get required columns if available
        required_columns = []
        if 'data_info' in validation_data and 'columns' in data_info:
            # Try to get required columns from schema in data_info
            # This would need to be passed from the analyzer
            pass

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

        # Add issue count to the table validation results section
        if total_issues > 0:
            # Status-Affecting Errors Section
            if status_affecting_count > 0:
                story.append(self.PageBreak())
                story.append(self.Paragraph(f"Status-Affecting Errors ({status_affecting_count})", heading_style))
                story.append(self.Paragraph("These errors affect the validation status and require review.", normal_style))
                story.append(self.Spacer(1, 0.2 * self.inch))

                # Schema Errors
                schema_errors = status_affecting.get('schema_errors', [])
                if schema_errors:
                    story.append(self.Paragraph(f"Critical Schema Issues ({len(schema_errors)})", heading_style))
                    for idx, error in enumerate(schema_errors, 1):
                        self._add_error_section(story, error, idx, styles)

                # Data Quality Issues
                quality_issues = status_affecting.get('data_quality_issues', [])
                if quality_issues:
                    story.append(self.Paragraph(f"Critical Data Quality Issues ({len(quality_issues)})", heading_style))
                    for idx, error in enumerate(quality_issues, 1):
                        self._add_error_section(story, error, idx, styles)

                # Other Errors
                other_errors = status_affecting.get('other_errors', [])
                if other_errors:
                    story.append(self.Paragraph(f"Other Critical Issues ({len(other_errors)})", heading_style))
                    for idx, error in enumerate(other_errors, 1):
                        self._add_error_section(story, error, idx, styles)

            # Informational Issues Section
            if informational_count > 0:
                story.append(self.PageBreak())
                story.append(self.Paragraph(f"Informational Issues ({informational_count})", heading_style))
                story.append(self.Paragraph("These issues are for awareness but do not affect the validation status.", normal_style))
                story.append(self.Spacer(1, 0.2 * self.inch))

                # Schema Info
                schema_info = informational.get('schema_errors', [])
                if schema_info:
                    story.append(self.Paragraph(f"Schema Information ({len(schema_info)})", heading_style))
                    for idx, error in enumerate(schema_info, 1):
                        self._add_error_section(story, error, idx, styles)

                # Data Quality Observations
                quality_obs = informational.get('data_quality_issues', [])
                if quality_obs:
                    story.append(self.Paragraph(f"Data Quality Observations ({len(quality_obs)})", heading_style))
                    for idx, error in enumerate(quality_obs, 1):
                        self._add_error_section(story, error, idx, styles)

                # Other Observations
                other_obs = informational.get('other_errors', [])
                if other_obs:
                    story.append(self.Paragraph(f"Other Observations ({len(other_obs)})", heading_style))
                    for idx, error in enumerate(other_obs, 1):
                        self._add_error_section(story, error, idx, styles)
        else:
            story.append(self.Paragraph("✓ No validation issues found!", styles['Normal']))

        # Add User Feedback Section if feedback was provided
        if feedback:
            rejected_count = feedback.get('rejected_count', 0)
            accepted_count = feedback.get('accepted_count', 0)

            if rejected_count > 0 or accepted_count > 0:
                story.append(self.PageBreak())
                story.append(self.Paragraph("User Feedback Summary", heading_style))

                # Show status change if applicable
                if original_status and original_status != status:
                    status_change_text = f"Status updated from <b>{original_status.upper()}</b> to <b>{status.upper()}</b> based on user feedback"
                    story.append(self.Paragraph(status_change_text, normal_style))
                    story.append(self.Spacer(1, 0.2 * self.inch))

                # Show feedback counts
                feedback_counts = []
                if rejected_count > 0:
                    feedback_counts.append(f"{rejected_count} error(s) marked as not applicable")
                if accepted_count > 0:
                    feedback_counts.append(f"{accepted_count} error(s) confirmed as valid")

                if feedback_counts:
                    story.append(self.Paragraph(" | ".join(feedback_counts), normal_style))
                    story.append(self.Spacer(1, 0.2 * self.inch))

                # Show rejected errors with reasons
                if rejected_count > 0 and 'user_decisions' in feedback:
                    story.append(self.Paragraph(f"Errors Marked as Not Applicable ({rejected_count})", heading_style))

                    for error_id, decision_info in feedback['user_decisions'].items():
                        if decision_info.get('decision') == 'rejected':
                            error_type = decision_info.get('error_type', 'Unknown')
                            description = decision_info.get('description', 'No description')
                            reason = decision_info.get('reason', '')

                            story.append(self.Paragraph(f"• <b>{error_type}</b>", styles['Normal']))

                            # Escape any HTML characters in description
                            from html import escape
                            description_escaped = escape(description)
                            story.append(self.Paragraph(f"  {description_escaped}", styles['Normal']))

                            if reason:
                                reason_escaped = escape(reason)
                                story.append(self.Paragraph(f"  <i>Reason: {reason_escaped}</i>", styles['Normal']))

                            story.append(self.Spacer(1, 0.1 * self.inch))

        # Build PDF
        doc.build(story)
        return output_path

    def _add_error_section(self, story, error: Dict[str, Any], index: int, styles):
        """Add an error section to the story."""
        # Error number and type
        error_title = f"{index}. {error.get('type', 'Unknown Error')}"
        story.append(self.Paragraph(error_title, styles['Heading3']))

        # Error description
        description = error.get('description', 'No description available')
        # Escape any HTML characters in description
        from html import escape
        description_escaped = escape(description).replace('\n', '<br/>')
        story.append(self.Paragraph(description_escaped, styles['Normal']))

        # Category
        category = error.get('category', 'unknown')
        category_text = f"<b>Category:</b> {category.replace('_', ' ').title()}"
        story.append(self.Paragraph(category_text, styles['Normal']))

        # Add details if available
        if 'details' in error and error['details']:
            details = error['details']
            if isinstance(details, dict):
                # Show details as key-value pairs
                for key, value in details.items():
                    detail_text = f"<b>{key.replace('_', ' ').title()}:</b> {escape(str(value))}"
                    story.append(self.Paragraph(detail_text, styles['Normal']))
            elif isinstance(details, list):
                # Show details as list
                for item in details:
                    story.append(self.Paragraph(f"• {escape(str(item))}", styles['Normal']))
            else:
                detail_text = f"<b>Details:</b> {escape(str(details))}"
                story.append(self.Paragraph(detail_text, styles['Normal']))

        story.append(self.Spacer(1, 0.2 * self.inch))

    def _get_status_color(self, status: str):
        """Get color based on validation status."""
        status_colors = {
            'complete': self.colors.HexColor('#d4edda'),  # Light green
            'partial': self.colors.HexColor('#fff3cd'),   # Light yellow
            'incomplete': self.colors.HexColor('#f8d7da')  # Light red
        }
        return status_colors.get(status.lower(), self.colors.white)

    def generate_text_report(self, validation_data: Dict[str, Any],
                            table_name: str, output_path: str,
                            site_name: Optional[str] = None,
                            timezone: Optional[str] = None,
                            feedback: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a text report as fallback when reportlab is not available.

        Parameters:
        -----------
        validation_data : dict
            Validation results from JSON
        table_name : str
            Name of the table
        output_path : str
            Path where text file should be saved
        site_name : str, optional
            Name of the site/hospital
        timezone : str, optional
            Configured timezone for filtering errors (defaults to UTC)
        feedback : dict, optional
            User feedback on validation errors with adjusted status

        Returns:
        --------
        str
            Path to generated text file
        """
        lines = []
        lines.append("=" * 80)
        lines.append("CLIF 2.1 VALIDATION REPORT")
        lines.append(f"{table_name.upper()} TABLE")
        lines.append("=" * 80)
        lines.append("")

        if site_name:
            lines.append(f"Site: {site_name}")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Validation Status
        lines.append("-" * 80)
        lines.append("VALIDATION STATUS")
        lines.append("-" * 80)
        status = validation_data.get('status', 'unknown').upper()
        is_valid = validation_data.get('is_valid', False)

        # Check for feedback and adjusted status
        if feedback:
            original_status = feedback.get('original_status', '').upper()
            adjusted_status = feedback.get('adjusted_status', '').upper()
            if adjusted_status and adjusted_status != original_status:
                lines.append(f"Status: {original_status} → {adjusted_status} (Updated based on user feedback)")
                status = adjusted_status
            else:
                lines.append(f"Status: {status}")
        else:
            lines.append(f"Status: {status}")

        lines.append(f"Valid: {'Yes' if is_valid else 'No'}")
        lines.append("")

        # Data Overview
        lines.append("-" * 80)
        lines.append("DATA OVERVIEW")
        lines.append("-" * 80)
        data_info = validation_data.get('data_info', {})

        if 'error' not in data_info:
            lines.append(f"Total Rows: {data_info.get('row_count', 0):,}")
            lines.append(f"Total Columns: {data_info.get('column_count', 0)}")

            if 'unique_patients' in data_info:
                lines.append(f"Unique Patients: {data_info.get('unique_patients', 0):,}")

            if 'unique_hospitalizations' in data_info:
                lines.append(f"Unique Hospitalizations: {data_info.get('unique_hospitalizations', 0):,}")
        lines.append("")

        # Classify errors by status impact
        from modules.utils.validation import classify_errors_by_status_impact

        errors = validation_data.get('errors', {})
        classified_errors = classify_errors_by_status_impact(errors, [], table_name, timezone)
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

        # Validation Issues Summary
        lines.append("-" * 80)
        lines.append("VALIDATION ISSUES SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Status-Affecting Errors: {status_affecting_count}")
        lines.append(f"Informational Issues: {informational_count}")
        lines.append(f"Total Issues: {total_issues}")
        lines.append("")

        # Detailed Issues
        if total_issues > 0:
            # Status-Affecting Errors
            if status_affecting_count > 0:
                lines.append("=" * 80)
                lines.append(f"STATUS-AFFECTING ERRORS ({status_affecting_count})")
                lines.append("These errors affect the validation status and require review.")
                lines.append("=" * 80)

                schema_errors = status_affecting.get('schema_errors', [])
                if schema_errors:
                    lines.append("")
                    lines.append(f"-- Critical Schema Issues ({len(schema_errors)}) --")
                    for idx, error in enumerate(schema_errors, 1):
                        lines.append("")
                        lines.append(f"{idx}. {error.get('type', 'Unknown Error')}")
                        lines.append(f"   Description: {error.get('description', 'No description')}")
                        lines.append(f"   Category: {error.get('category', 'unknown')}")
                        if 'details' in error and error['details']:
                            self._add_error_details_text(lines, error['details'])

                quality_issues = status_affecting.get('data_quality_issues', [])
                if quality_issues:
                    lines.append("")
                    lines.append(f"-- Critical Data Quality Issues ({len(quality_issues)}) --")
                    for idx, error in enumerate(quality_issues, 1):
                        lines.append("")
                        lines.append(f"{idx}. {error.get('type', 'Unknown Error')}")
                        lines.append(f"   Description: {error.get('description', 'No description')}")
                        lines.append(f"   Category: {error.get('category', 'unknown')}")
                        if 'details' in error and error['details']:
                            self._add_error_details_text(lines, error['details'])

                other_errors = status_affecting.get('other_errors', [])
                if other_errors:
                    lines.append("")
                    lines.append(f"-- Other Critical Issues ({len(other_errors)}) --")
                    for idx, error in enumerate(other_errors, 1):
                        lines.append("")
                        lines.append(f"{idx}. {error.get('type', 'Unknown Error')}")
                        lines.append(f"   Description: {error.get('description', 'No description')}")
                        lines.append(f"   Category: {error.get('category', 'unknown')}")
                        if 'details' in error and error['details']:
                            self._add_error_details_text(lines, error['details'])

            # Informational Issues
            if informational_count > 0:
                lines.append("")
                lines.append("=" * 80)
                lines.append(f"INFORMATIONAL ISSUES ({informational_count})")
                lines.append("These issues are for awareness but do not affect validation status.")
                lines.append("=" * 80)

                schema_info = informational.get('schema_errors', [])
                if schema_info:
                    lines.append("")
                    lines.append(f"-- Schema Information ({len(schema_info)}) --")
                    for idx, error in enumerate(schema_info, 1):
                        lines.append("")
                        lines.append(f"{idx}. {error.get('type', 'Unknown Error')}")
                        lines.append(f"   Description: {error.get('description', 'No description')}")
                        lines.append(f"   Category: {error.get('category', 'unknown')}")
                        if 'details' in error and error['details']:
                            self._add_error_details_text(lines, error['details'])

                quality_obs = informational.get('data_quality_issues', [])
                if quality_obs:
                    lines.append("")
                    lines.append(f"-- Data Quality Observations ({len(quality_obs)}) --")
                    for idx, error in enumerate(quality_obs, 1):
                        lines.append("")
                        lines.append(f"{idx}. {error.get('type', 'Unknown Error')}")
                        lines.append(f"   Description: {error.get('description', 'No description')}")
                        lines.append(f"   Category: {error.get('category', 'unknown')}")
                        if 'details' in error and error['details']:
                            self._add_error_details_text(lines, error['details'])

                other_obs = informational.get('other_errors', [])
                if other_obs:
                    lines.append("")
                    lines.append(f"-- Other Observations ({len(other_obs)}) --")
                    for idx, error in enumerate(other_obs, 1):
                        lines.append("")
                        lines.append(f"{idx}. {error.get('type', 'Unknown Error')}")
                        lines.append(f"   Description: {error.get('description', 'No description')}")
                        lines.append(f"   Category: {error.get('category', 'unknown')}")
                        if 'details' in error and error['details']:
                            self._add_error_details_text(lines, error['details'])
        else:
            lines.append("✓ No validation issues found!")

        # Add User Feedback Section if feedback was provided
        if feedback:
            rejected_count = feedback.get('rejected_count', 0)
            accepted_count = feedback.get('accepted_count', 0)

            if rejected_count > 0 or accepted_count > 0:
                lines.append("")
                lines.append("=" * 80)
                lines.append("USER FEEDBACK SUMMARY")
                lines.append("=" * 80)

                # Show feedback counts
                lines.append(f"Errors marked as not applicable: {rejected_count}")
                lines.append(f"Errors confirmed as valid: {accepted_count}")
                lines.append("")

                # Show rejected errors with reasons
                if rejected_count > 0 and 'user_decisions' in feedback:
                    lines.append("ERRORS MARKED AS NOT APPLICABLE:")
                    lines.append("-" * 80)

                    for error_id, decision_info in feedback['user_decisions'].items():
                        if decision_info.get('decision') == 'rejected':
                            error_type = decision_info.get('error_type', 'Unknown')
                            description = decision_info.get('description', 'No description')
                            reason = decision_info.get('reason', '')

                            lines.append(f"• {error_type}")
                            lines.append(f"  {description}")
                            if reason:
                                lines.append(f"  Reason: {reason}")
                            lines.append("")

        lines.append("")
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)

        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        return output_path

    def _add_error_details_text(self, lines: list, details: Any):
        """Add error details to text report lines."""
        if isinstance(details, dict):
            # Show details as key-value pairs
            for key, value in details.items():
                lines.append(f"   {key.replace('_', ' ').title()}: {value}")
        elif isinstance(details, list):
            # Show details as list
            for item in details:
                lines.append(f"   • {item}")
        else:
            # Show as single detail
            lines.append(f"   Details: {details}")
