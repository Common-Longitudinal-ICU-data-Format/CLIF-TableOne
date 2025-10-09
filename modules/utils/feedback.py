"""
User Feedback Utilities for Validation Errors

This module provides functions to handle user feedback on validation errors,
allowing sites to accept or reject errors based on their specific data context.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import os


def create_error_id(error: Dict[str, Any]) -> str:
    """
    Create a unique identifier for an error.

    Parameters:
    -----------
    error : dict
        Error dictionary with type and description

    Returns:
    --------
    str
        Unique error identifier
    """
    # Use type and a hash of description for uniqueness
    import hashlib
    desc_hash = hashlib.md5(error.get('description', '').encode()).hexdigest()[:8]
    error_type = error.get('raw_type', error.get('type', 'unknown')).replace(' ', '_').lower()
    return f"{error_type}_{desc_hash}"


def create_feedback_structure(validation_results: Dict[str, Any],
                              table_name: str) -> Dict[str, Any]:
    """
    Create initial feedback structure for a validation result.

    Parameters:
    -----------
    validation_results : dict
        Validation results from analyzer
    table_name : str
        Name of the table

    Returns:
    --------
    dict
        Feedback structure with all errors initialized as 'pending'
    """
    errors = validation_results.get('errors', {})
    all_errors = (
        errors.get('schema_errors', []) +
        errors.get('data_quality_issues', []) +
        errors.get('other_errors', [])
    )

    feedback = {
        'table': table_name,
        'timestamp': datetime.now().isoformat(),
        'original_status': validation_results.get('status', 'unknown'),
        'adjusted_status': validation_results.get('status', 'unknown'),
        'total_errors': len(all_errors),
        'accepted_count': 0,
        'rejected_count': 0,
        'pending_count': len(all_errors),
        'user_decisions': {}
    }

    # Initialize all errors as pending
    for error in all_errors:
        error_id = create_error_id(error)
        feedback['user_decisions'][error_id] = {
            'error_type': error.get('type', 'Unknown'),
            'raw_type': error.get('raw_type', ''),
            'category': error.get('category', 'other'),
            'description': error.get('description', ''),
            'decision': 'pending',
            'reason': '',
            'timestamp': None
        }

    return feedback


def update_user_decision(feedback: Dict[str, Any], error_id: str,
                        decision: str, reason: str = '') -> Dict[str, Any]:
    """
    Update user decision for a specific error.

    Parameters:
    -----------
    feedback : dict
        Feedback structure
    error_id : str
        Error identifier
    decision : str
        'accepted', 'rejected', or 'pending'
    reason : str, optional
        Reason for the decision

    Returns:
    --------
    dict
        Updated feedback structure
    """
    if error_id not in feedback['user_decisions']:
        return feedback

    old_decision = feedback['user_decisions'][error_id]['decision']

    # Update counts
    if old_decision == 'accepted':
        feedback['accepted_count'] -= 1
    elif old_decision == 'rejected':
        feedback['rejected_count'] -= 1
    elif old_decision == 'pending':
        feedback['pending_count'] -= 1

    if decision == 'accepted':
        feedback['accepted_count'] += 1
    elif decision == 'rejected':
        feedback['rejected_count'] += 1
    elif decision == 'pending':
        feedback['pending_count'] += 1

    # Update decision
    feedback['user_decisions'][error_id]['decision'] = decision
    feedback['user_decisions'][error_id]['reason'] = reason
    feedback['user_decisions'][error_id]['timestamp'] = datetime.now().isoformat()

    return feedback


def get_accepted_errors(feedback: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get list of accepted errors from feedback.

    Parameters:
    -----------
    feedback : dict
        Feedback structure

    Returns:
    --------
    list
        List of accepted error details
    """
    accepted = []
    for error_id, error_info in feedback['user_decisions'].items():
        if error_info['decision'] == 'accepted':
            accepted.append({
                'error_id': error_id,
                **error_info
            })
    return accepted


def get_rejected_errors(feedback: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Get list of rejected errors from feedback.

    Parameters:
    -----------
    feedback : dict
        Feedback structure

    Returns:
    --------
    list
        List of rejected error details
    """
    rejected = []
    for error_id, error_info in feedback['user_decisions'].items():
        if error_info['decision'] == 'rejected':
            rejected.append({
                'error_id': error_id,
                **error_info
            })
    return rejected


def recalculate_status(original_status: str, feedback: Dict[str, Any]) -> str:
    """
    Recalculate validation status based on user feedback.

    Logic:
    - Status becomes 'complete' ONLY if ALL errors are explicitly rejected
    - If any errors are accepted OR pending, keep the original status
    - Accepted errors = real issues that need fixing
    - Pending errors = not yet reviewed, so assume they're real issues
    - Rejected errors = not applicable to this site

    Parameters:
    -----------
    original_status : str
        Original validation status
    feedback : dict
        Feedback structure with user decisions

    Returns:
    --------
    str
        Adjusted validation status
    """
    accepted_errors = get_accepted_errors(feedback)
    total_errors = feedback.get('total_errors', 0)
    rejected_count = feedback.get('rejected_count', 0)

    # If there are any accepted or pending errors, keep original status
    # because these represent real issues (accepted) or unreviewed issues (pending)
    if accepted_errors or rejected_count < total_errors:
        # Has accepted or pending errors - keep original status
        return original_status

    # All errors were explicitly rejected - table is complete
    # (rejected = site-specific, not applicable)
    return 'complete'


def save_feedback(feedback: Dict[str, Any], output_dir: str, table_name: str):
    """
    Save feedback to JSON file.

    Parameters:
    -----------
    feedback : dict
        Feedback structure
    output_dir : str
        Output directory
    table_name : str
        Name of the table
    """
    # Update adjusted status before saving
    feedback['adjusted_status'] = recalculate_status(
        feedback['original_status'],
        feedback
    )

    # Ensure output directory exists
    final_dir = os.path.join(output_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)

    # Save to file
    filename = f"{table_name}_validation_response.json"
    filepath = os.path.join(final_dir, filename)

    with open(filepath, 'w') as f:
        json.dump(feedback, f, indent=2, default=str)

    return filepath


def load_feedback(output_dir: str, table_name: str) -> Optional[Dict[str, Any]]:
    """
    Load existing feedback from file.

    Parameters:
    -----------
    output_dir : str
        Output directory
    table_name : str
        Name of the table

    Returns:
    --------
    dict or None
        Loaded feedback structure or None if file doesn't exist
    """
    filename = f"{table_name}_validation_response.json"
    filepath = os.path.join(output_dir, 'final', filename)

    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def get_feedback_summary(feedback: Dict[str, Any]) -> str:
    """
    Generate a text summary of feedback.

    Parameters:
    -----------
    feedback : dict
        Feedback structure

    Returns:
    --------
    str
        Human-readable summary
    """
    total = feedback['total_errors']
    accepted = feedback['accepted_count']
    rejected = feedback['rejected_count']
    pending = feedback['pending_count']

    original_status = feedback['original_status']
    adjusted_status = feedback['adjusted_status']

    summary_parts = []

    # Status comparison
    if original_status == adjusted_status:
        summary_parts.append(f"Status: {adjusted_status.upper()}")
    else:
        summary_parts.append(f"Status: {original_status.upper()} → {adjusted_status.upper()}")

    # Error breakdown
    summary_parts.append(f"Total errors: {total}")
    summary_parts.append(f"Accepted: {accepted}, Rejected: {rejected}, Pending: {pending}")

    return " | ".join(summary_parts)


def export_feedback_report(feedback: Dict[str, Any], output_path: str = None) -> str:
    """
    Export a detailed feedback report.

    Parameters:
    -----------
    feedback : dict
        Feedback structure
    output_path : str, optional
        Path to save report (if None, returns string)

    Returns:
    --------
    str
        Report content or file path
    """
    report_lines = []

    report_lines.append("=" * 80)
    report_lines.append(f"VALIDATION FEEDBACK REPORT - {feedback['table'].upper()}")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Original Analysis: {feedback.get('timestamp', 'Unknown')}")
    report_lines.append("")

    report_lines.append("STATUS SUMMARY")
    report_lines.append("-" * 80)
    report_lines.append(f"Original Status: {feedback['original_status'].upper()}")
    report_lines.append(f"Adjusted Status: {feedback['adjusted_status'].upper()}")
    report_lines.append("")

    report_lines.append("ERROR BREAKDOWN")
    report_lines.append("-" * 80)
    report_lines.append(f"Total Errors: {feedback['total_errors']}")
    report_lines.append(f"Accepted: {feedback['accepted_count']}")
    report_lines.append(f"Rejected: {feedback['rejected_count']}")
    report_lines.append(f"Pending Review: {feedback['pending_count']}")
    report_lines.append("")

    # Accepted errors
    accepted = get_accepted_errors(feedback)
    if accepted:
        report_lines.append("ACCEPTED ERRORS (These are considered valid issues)")
        report_lines.append("-" * 80)
        for error in accepted:
            report_lines.append(f"  • {error['error_type']}")
            report_lines.append(f"    {error['description']}")
            if error.get('reason'):
                report_lines.append(f"    Reason: {error['reason']}")
            report_lines.append("")

    # Rejected errors
    rejected = get_rejected_errors(feedback)
    if rejected:
        report_lines.append("REJECTED ERRORS (Site-specific, not considered issues)")
        report_lines.append("-" * 80)
        for error in rejected:
            report_lines.append(f"  • {error['error_type']}")
            report_lines.append(f"    {error['description']}")
            if error.get('reason'):
                report_lines.append(f"    Reason: {error['reason']}")
            report_lines.append("")

    report_content = "\n".join(report_lines)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_content)
        return output_path

    return report_content