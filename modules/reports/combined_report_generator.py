"""
Combined report generator for multi-table DQA validation results.

Generates a single combined PDF + CSV report from multiple table DQA results.
"""

import os
import json
import csv
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from modules.cli.pdf_generator import _collect_dqa_issues, DQA_CATEGORIES
from clifpy.utils.report_generator import _make_error_id
from clifpy.utils.rule_codes import enrich_issue

# Table display names mapping
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


def collect_table_results(output_dir: str, table_names: List[str]):
    """
    Collect DQA validation results and feedback from JSON files.

    Returns (results, feedback_map) where:
    - results: dict mapping table_name -> serialized DQA result dict (or None)
    - feedback_map: dict mapping table_name -> feedback dict (or None)
    """
    from modules.utils.output_paths import validation_json_reports_dir, validation_feedback_dir
    clifpy_dir = str(validation_json_reports_dir())
    results_dir = str(validation_feedback_dir())
    results = {}
    feedback_map = {}

    for table_name in table_names:
        json_path = os.path.join(clifpy_dir, f'{table_name}_dqa.json')
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    results[table_name] = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load DQA results for {table_name}: {e}")
                results[table_name] = None
        else:
            results[table_name] = None

        # Load feedback if available
        fb_path = os.path.join(results_dir, f'{table_name}_validation_response.json')
        if os.path.exists(fb_path):
            try:
                with open(fb_path, 'r', encoding='utf-8') as f:
                    feedback_map[table_name] = json.load(f)
            except Exception:
                feedback_map[table_name] = None
        else:
            feedback_map[table_name] = None

    return results, feedback_map


def _score_from_serialized(dqa_data: Dict[str, Any]):
    """Compute category scores from a serialized (JSON-loaded) DQA result dict.

    Unlike _collect_dqa_issues, this works on dicts that have already been
    serialized via .to_dict() (no result objects).

    Returns (category_scores, all_issues) matching the same shape as
    _collect_dqa_issues but from plain dicts.
    """
    category_scores = {}
    all_issues = []

    for category in DQA_CATEGORIES:
        checks = dqa_data.get(category, {})
        if not checks:
            continue
        cat_issues = []
        for check_name, d in checks.items():
            for err in d.get('errors', []):
                issue = {
                    'category': category,
                    'check_type': d.get('check_type', check_name),
                    'severity': 'error',
                    'message': err.get('message', ''),
                    'details': err.get('details', {}),
                }
                enriched = enrich_issue(issue, check_key=check_name)
                if enriched is not None:
                    cat_issues.append(enriched)
            for warn in d.get('warnings', []):
                issue = {
                    'category': category,
                    'check_type': d.get('check_type', check_name),
                    'severity': 'warning',
                    'message': warn.get('message', ''),
                    'details': warn.get('details', {}),
                }
                enriched = enrich_issue(issue, check_key=check_name)
                if enriched is not None:
                    cat_issues.append(enriched)
            for info_msg in d.get('info', []):
                issue = {
                    'category': category,
                    'check_type': d.get('check_type', check_name),
                    'severity': 'info',
                    'message': info_msg.get('message', ''),
                    'details': info_msg.get('details', {}),
                }
                enriched = enrich_issue(issue, check_key=check_name)
                if enriched is not None:
                    cat_issues.append(enriched)
        all_issues.extend(cat_issues)
        if cat_issues:
            cat_passed = sum(1 for i in cat_issues if i['severity'] in ('info', 'warning'))
            category_scores[category] = (cat_passed, len(cat_issues))

    return category_scores, all_issues


def generate_combined_pdf(table_results: Dict[str, Any], output_path: str,
                          site_name: Optional[str] = None, timezone: Optional[str] = 'UTC',
                          feedback_map: Optional[Dict[str, Any]] = None) -> str:
    """Generate a combined PDF report from multiple table DQA results."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
    except ImportError:
        raise ImportError("reportlab is required for PDF generation. Install with: pip install reportlab")

    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    primary_color = colors.HexColor('#1F4E79')
    text_dark = colors.HexColor('#2C3E50')
    text_medium = colors.HexColor('#5D6D7E')
    header_bg = colors.HexColor('#F5F6FA')
    pass_bg = colors.HexColor('#E8F5E8')
    fail_bg = colors.HexColor('#FFEAEA')

    title_style = ParagraphStyle(
        'CustomTitle', parent=styles['Heading1'],
        fontSize=22, textColor=primary_color, spaceAfter=24,
        alignment=TA_CENTER, fontName='Helvetica-Bold',
    )
    heading_style = ParagraphStyle(
        'CustomHeading', parent=styles['Heading2'],
        fontSize=14, textColor=text_dark, spaceAfter=10,
        spaceBefore=16, fontName='Helvetica-Bold',
    )
    normal_style = ParagraphStyle(
        'CustomNormal', parent=styles['Normal'],
        fontSize=9, textColor=text_medium, fontName='Helvetica',
    )
    timestamp_style = ParagraphStyle(
        'TimestampStyle', parent=styles['Normal'],
        fontSize=8, textColor=text_medium, alignment=1, fontName='Helvetica',
    )

    # Timestamp
    ts = Table(
        [[Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", timestamp_style)]],
        colWidths=[7.5 * inch],
    )
    ts.setStyle(TableStyle([
        ('ALIGN', (0, 0), (0, 0), 'RIGHT'),
        ('TOPPADDING', (0, 0), (0, 0), -24),
        ('BOTTOMPADDING', (0, 0), (0, 0), 2),
    ]))
    story.append(ts)

    # Title
    title_text = f"{site_name + ' ' if site_name else ''}CLIF DQA Report Card"
    story.append(Paragraph(title_text, title_style))
    story.append(Paragraph("Combined Validation Report", heading_style))
    story.append(Spacer(1, 0.2 * inch))

    # Check if any table has feedback
    if feedback_map is None:
        feedback_map = {}
    has_any_feedback = any(
        fb and any(d.get('decision') in ('accepted', 'rejected')
                   for d in fb.get('user_decisions', {}).values())
        for fb in feedback_map.values() if fb
    )

    # Feedback banner
    if has_any_feedback:
        fb_banner_style = ParagraphStyle(
            'FbBanner', parent=normal_style, fontSize=8,
            textColor=colors.HexColor('#1F4E79'), fontName='Helvetica',
        )
        fb_banner = Table(
            [[Paragraph(
                "<i>This report includes user feedback decisions. "
                "See individual table reports for details.</i>",
                fb_banner_style,
            )]],
            colWidths=[7.5 * inch],
        )
        fb_banner.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, 0), colors.HexColor('#E8F0FE')),
            ('TOPPADDING', (0, 0), (0, 0), 6),
            ('BOTTOMPADDING', (0, 0), (0, 0), 6),
            ('LEFTPADDING', (0, 0), (0, 0), 10),
            ('RIGHTPADDING', (0, 0), (0, 0), 10),
        ]))
        story.append(fb_banner)
        story.append(Spacer(1, 0.15 * inch))

    # --- Overview table: one row per table ---
    story.append(Paragraph("DQA Overview", heading_style))
    cat_labels = [c.title() for c in DQA_CATEGORIES]
    overview_header = ['Table'] + cat_labels + ['Overall']
    if has_any_feedback:
        overview_header.append('Feedback')
    overview_rows = [overview_header]

    # Number of score columns (categories + Overall) used for _row_overall
    n_score_cols = len(DQA_CATEGORIES) + 1

    def _row_overall(row):
        """Sum p/t across category columns in a row (cols 1..len(DQA_CATEGORIES))."""
        total_p, total_t = 0, 0
        for cell in row[1:1 + len(DQA_CATEGORIES)]:
            if cell == 'N/A':
                continue
            parts = cell.split('/')
            if len(parts) == 2:
                total_p += int(parts[0])
                total_t += int(parts[1])
        return f"{total_p}/{total_t}" if total_t > 0 else 'N/A'

    def _feedback_summary(fb):
        """Compact feedback summary like '2R/1A'."""
        if not fb or not fb.get('user_decisions'):
            return ''
        n_a = sum(1 for d in fb['user_decisions'].values() if d.get('decision') == 'accepted')
        n_r = sum(1 for d in fb['user_decisions'].values() if d.get('decision') == 'rejected')
        parts = []
        if n_r:
            parts.append(f"{n_r}R")
        if n_a:
            parts.append(f"{n_a}A")
        return '/'.join(parts)

    for table_name, dqa_data in table_results.items():
        display_name = TABLE_DISPLAY_NAMES.get(table_name, table_name.replace('_', ' ').title())
        if dqa_data is None:
            row = [display_name] + ['N/A'] * n_score_cols
            if has_any_feedback:
                row.append('')
            overview_rows.append(row)
            continue
        scores, all_issues = _score_from_serialized(dqa_data)
        # Adjust scores for rejected feedback
        fb = feedback_map.get(table_name)
        rejected_ids: set = set()
        if fb and fb.get('user_decisions'):
            rejected_ids = {
                eid for eid, d in fb['user_decisions'].items()
                if d.get('decision') == 'rejected'
            }
        if rejected_ids:
            for cat in list(scores.keys()):
                cat_rejected = sum(
                    1 for i in all_issues
                    if i['category'] == cat and i['severity'] == 'error'
                    and _make_error_id(i) in rejected_ids
                )
                if cat_rejected:
                    p, t = scores[cat]
                    scores[cat] = (p + cat_rejected, t)
        row = [display_name]
        for cat in DQA_CATEGORIES:
            if cat in scores:
                p, t = scores[cat]
                row.append(f"{p}/{t}")
            else:
                row.append('N/A')
        row.append(_row_overall(row))
        if has_any_feedback:
            row.append(_feedback_summary(feedback_map.get(table_name)))
        overview_rows.append(row)

    # Add totals row
    totals_row = ['Total']
    for col_idx in range(1, len(DQA_CATEGORIES) + 1):
        total_passed = 0
        total_count = 0
        for row in overview_rows[1:]:  # skip header
            cell_val = row[col_idx]
            if cell_val == 'N/A':
                continue
            parts = cell_val.split('/')
            if len(parts) == 2:
                total_passed += int(parts[0])
                total_count += int(parts[1])
        totals_row.append(f"{total_passed}/{total_count}" if total_count > 0 else 'N/A')
    totals_row.append(_row_overall(totals_row))
    if has_any_feedback:
        totals_row.append('')
    overview_rows.append(totals_row)

    n_display_cols = len(overview_header)
    col_widths = [2.2 * inch] + [1.2 * inch] * (len(DQA_CATEGORIES) + 1)
    if has_any_feedback:
        col_widths.append(0.7 * inch)
    overview_tbl = Table(overview_rows, colWidths=col_widths)
    totals_row_idx = len(overview_rows) - 1
    tbl_style = [
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTNAME', (0, totals_row_idx), (-1, totals_row_idx), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BACKGROUND', (0, 0), (-1, 0), header_bg),
        ('BACKGROUND', (0, totals_row_idx), (-1, totals_row_idx), header_bg),
        ('TEXTCOLOR', (0, 0), (-1, -1), text_dark),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DADADA')),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]
    # Color-code pass/fail cells (score columns only: 1..n_score_cols)
    for row_idx in range(1, len(overview_rows)):
        for col_idx in range(1, n_score_cols + 1):
            cell_val = overview_rows[row_idx][col_idx]
            if cell_val == 'N/A':
                continue
            parts = cell_val.split('/')
            if len(parts) == 2 and parts[0] == parts[1]:
                tbl_style.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), pass_bg))
            else:
                tbl_style.append(('BACKGROUND', (col_idx, row_idx), (col_idx, row_idx), fail_bg))

    overview_tbl.setStyle(TableStyle(tbl_style))
    story.append(overview_tbl)
    story.append(Spacer(1, 0.3 * inch))

    doc.build(story)
    return output_path


def generate_consolidated_csv(table_results: Dict[str, Any], output_path: str,
                               timezone: Optional[str] = 'UTC',
                               feedback_map: Optional[Dict[str, Any]] = None) -> str:
    """Generate a consolidated CSV report from multiple table DQA results."""
    if feedback_map is None:
        feedback_map = {}
    rows = []

    for table_name, dqa_data in table_results.items():
        display_name = TABLE_DISPLAY_NAMES.get(table_name, table_name.replace('_', ' ').title())
        fb = feedback_map.get(table_name)
        fb_decisions = fb.get('user_decisions', {}) if fb else {}

        if dqa_data is None:
            rows.append({
                'table_name': display_name,
                'category': '',
                'rule_code': '',
                'rule_description': '',
                'check_type': 'Table Status',
                'column_field': 'NA',
                'severity': 'error',
                'passed': False,
                'message': 'Data file not found or table not analyzed',
                'decision': '',
                'reason': '',
            })
            continue

        _, all_issues = _score_from_serialized(dqa_data)

        if not all_issues:
            rows.append({
                'table_name': display_name,
                'category': '',
                'rule_code': '',
                'rule_description': '',
                'check_type': 'Summary',
                'column_field': 'NA',
                'severity': 'info',
                'passed': True,
                'message': 'All DQA checks passed',
                'decision': '',
                'reason': '',
            })
            continue

        for issue in all_issues:
            error_id = _make_error_id(issue)
            decision_info = fb_decisions.get(error_id, {})
            rows.append({
                'table_name': display_name,
                'category': issue['category'],
                'rule_code': issue.get('rule_code', ''),
                'rule_description': issue.get('rule_description', ''),
                'check_type': issue['check_type'],
                'column_field': issue.get('column_field', 'NA'),
                'severity': issue['severity'],
                'passed': False,
                'message': issue.get('finding', issue['message']),
                'decision': decision_info.get('decision', '') if issue['severity'] == 'error' else '',
                'reason': decision_info.get('reason', '') if issue['severity'] == 'error' else '',
            })

    fieldnames = ['table_name', 'category', 'rule_code', 'rule_description',
                  'check_type', 'column_field', 'severity', 'passed', 'message',
                  'decision', 'reason']
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_path


def generate_combined_report(output_dir: str, table_names: List[str],
                            site_name: Optional[str] = None,
                            timezone: Optional[str] = 'UTC') -> Optional[str]:
    """
    High-level function to generate a combined DQA report (PDF and CSV).

    Parameters
    ----------
    output_dir : str
        Output directory containing DQA JSON files.
    table_names : list
        List of table names to include.
    site_name : str, optional
        Name of the site/hospital.
    timezone : str, optional
        Configured timezone (defaults to UTC).

    Returns
    -------
    str or None
        Path to generated PDF, or None if generation failed.
    """
    try:
        table_results, feedback_map = collect_table_results(output_dir, table_names)

        analyzed_count = sum(1 for r in table_results.values() if r is not None)
        if analyzed_count == 0:
            print("Warning: No analyzed tables found. Run validation first.")
            return None

        from modules.utils.output_paths import PDF_REPORTS, validation_consolidated_dir
        reports_dir = PDF_REPORTS
        reports_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = str(reports_dir / 'combined_validation_report.pdf')
        generate_combined_pdf(table_results, pdf_path, site_name, timezone,
                              feedback_map=feedback_map)

        results_dir = validation_consolidated_dir()
        results_dir.mkdir(parents=True, exist_ok=True)
        csv_path = str(results_dir / 'consolidated_validation.csv')
        generate_consolidated_csv(table_results, csv_path, timezone, feedback_map=feedback_map)

        return pdf_path

    except Exception as e:
        print(f"Error generating combined report: {e}")
        import traceback
        traceback.print_exc()
        return None
