"""
User Feedback Utilities for Validation Errors

This module provides functions to handle user feedback on validation errors,
allowing sites to accept or reject errors based on their specific data context.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import os


def _is_multi_value_mcide(issue: Dict[str, Any]) -> bool:
    """True if this is an MCIDE coverage error with >1 missing values."""
    if issue.get('check_type') != 'mcide_value_coverage':
        return False
    details = issue.get('details') or {}
    missing = details.get('missing_values') or []
    return isinstance(missing, list) and len(missing) > 1


def create_error_id(error: Dict[str, Any]) -> str:
    """
    Create a unique identifier for a DQA issue (or legacy error).

    Parameters:
    -----------
    error : dict
        DQA issue dict (category, check_type, message) or legacy error dict

    Returns:
    --------
    str
        Unique error identifier
    """
    import hashlib
    # DQA format: hash on 'message'; legacy fallback: hash on 'description'
    msg = error.get('message', error.get('description', ''))
    desc_hash = hashlib.md5(msg.encode()).hexdigest()[:8]
    category = error.get('category', '')
    check_type = error.get('check_type', error.get('raw_type', error.get('type', 'unknown')))
    prefix = f"{category}_{check_type}" if category else check_type
    prefix = prefix.replace(' ', '_').lower()
    return f"{prefix}_{desc_hash}"


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
    from modules.cli.pdf_generator import _collect_dqa_issues
    category_scores, all_issues = _collect_dqa_issues(validation_results)
    reviewable = [i for i in all_issues if i['severity'] == 'error']

    # Compute status: use legacy top-level 'status' if present,
    # otherwise derive from DQA category scores
    if 'status' in validation_results:
        computed_status = validation_results['status']
    else:
        total_passed = sum(p for p, _ in category_scores.values())
        total_checks = sum(t for _, t in category_scores.values())
        error_count = sum(1 for i in all_issues if i['severity'] == 'error')
        if total_checks == 0:
            computed_status = 'unknown'
        elif total_passed == total_checks:
            computed_status = 'complete'
        elif error_count == 0:
            computed_status = 'partial'
        else:
            computed_status = 'incomplete'

    feedback = {
        'table': table_name,
        'timestamp': datetime.now().isoformat(),
        'original_status': computed_status,
        'adjusted_status': computed_status,
        'total_errors': 0,
        'accepted_count': 0,
        'rejected_count': 0,
        'pending_count': 0,
        'user_decisions': {}
    }

    # Initialize all reviewable issues as pending. For multi-value MCIDE
    # coverage errors (e.g. "Missing 21 mCIDE values: German, Yiddish, …"),
    # the parent decision stays 'pending' and we attach a value_decisions
    # sub-dict so the site can reject individual missing values rather than
    # the whole bundled row.
    for issue in reviewable:
        error_id = create_error_id(issue)
        entry = {
            'error_type': issue.get('check_type', 'Unknown'),
            'raw_type': issue.get('check_type', ''),
            'category': issue.get('category', 'other'),
            'severity': issue.get('severity', 'error'),
            'description': issue.get('message', ''),
            'decision': 'pending',
            'reason': '',
            'timestamp': None,
        }
        if _is_multi_value_mcide(issue):
            details = issue.get('details') or {}
            entry['mcide_column'] = details.get('column')
            entry['missing_values'] = list(details.get('missing_values') or [])
            entry['value_decisions'] = {
                v: {'decision': 'pending', 'reason': '', 'timestamp': None}
                for v in entry['missing_values']
            }
        feedback['user_decisions'][error_id] = entry

    _recompute_counts(feedback)
    return feedback


def _recompute_counts(feedback: Dict[str, Any]) -> None:
    """Recompute accepted/rejected/pending/total counts from user_decisions.

    For entries with a ``value_decisions`` sub-dict (multi-value MCIDE),
    each sub-value contributes 1 atom to the totals; the parent entry itself
    does not count. For regular entries, the parent counts as 1 atom. This
    matches how the UI presents the review status bar.
    """
    accepted = rejected = pending = 0
    for entry in feedback.get('user_decisions', {}).values():
        subs = entry.get('value_decisions')
        if isinstance(subs, dict) and subs:
            for sub in subs.values():
                d = sub.get('decision', 'pending')
                if d == 'accepted':
                    accepted += 1
                elif d == 'rejected':
                    rejected += 1
                else:
                    pending += 1
        else:
            d = entry.get('decision', 'pending')
            if d == 'accepted':
                accepted += 1
            elif d == 'rejected':
                rejected += 1
            else:
                pending += 1
    feedback['accepted_count'] = accepted
    feedback['rejected_count'] = rejected
    feedback['pending_count'] = pending
    feedback['total_errors'] = accepted + rejected + pending


def ensure_mcide_subdecisions(feedback: Dict[str, Any],
                              issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Migrate an on-disk feedback dict to the per-value MCIDE schema.

    Called on every feedback GET so older files auto-upgrade:
    * For each issue that qualifies as multi-value MCIDE, ensure the
      corresponding parent entry has a ``value_decisions`` sub-dict with
      one slot per missing value.
    * If the parent entry previously had ``decision ∈ {accepted, rejected}``,
      propagate that decision to every sub-value (with the same reason),
      then reset the parent decision to 'pending' — the parent is now a
      pure aggregate of its children.
    * Always recompute counts at the end.
    """
    if not feedback or 'user_decisions' not in feedback:
        return feedback
    decisions = feedback['user_decisions']
    for issue in issues or []:
        if not _is_multi_value_mcide(issue):
            continue
        eid = create_error_id(issue)
        details = issue.get('details') or {}
        values = list(details.get('missing_values') or [])
        entry = decisions.get(eid)
        if entry is None:
            # Parent was never in the dict (e.g. new error vs. stale file)
            decisions[eid] = {
                'error_type': 'mcide_value_coverage',
                'raw_type': 'mcide_value_coverage',
                'category': issue.get('category', 'completeness'),
                'severity': 'error',
                'description': issue.get('message', ''),
                'decision': 'pending',
                'reason': '',
                'timestamp': None,
                'mcide_column': details.get('column'),
                'missing_values': values,
                'value_decisions': {
                    v: {'decision': 'pending', 'reason': '', 'timestamp': None}
                    for v in values
                },
            }
            continue
        entry.setdefault('mcide_column', details.get('column'))
        entry.setdefault('missing_values', values)
        subs = entry.get('value_decisions')
        if not isinstance(subs, dict):
            subs = {}
        parent_decision = entry.get('decision', 'pending')
        parent_reason = entry.get('reason', '') or ''
        parent_ts = entry.get('timestamp')
        inherit = parent_decision in ('accepted', 'rejected') and not subs
        for v in values:
            if v not in subs:
                subs[v] = {
                    'decision': parent_decision if inherit else 'pending',
                    'reason': parent_reason if inherit else '',
                    'timestamp': parent_ts if inherit else None,
                }
        entry['value_decisions'] = subs
        # Parent is now a pure aggregate; reset its own decision so the
        # review-status math doesn't double-count.
        if inherit:
            entry['decision'] = 'pending'
            entry['reason'] = ''
            entry['timestamp'] = None
    _recompute_counts(feedback)
    return feedback


def prune_orphan_decisions(feedback: Dict[str, Any],
                           issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Drop ``user_decisions`` entries that no longer match a current error.

    ``error_id`` hashes the issue *message* (see :func:`create_error_id`), so
    when validation re-runs and a message text changes, the previously-saved
    decision becomes orphaned: it is never rendered (the UI only draws current
    issues) yet is still tallied by :func:`_recompute_counts`. That produces a
    phantom ``pending`` atom the user can never clear — e.g. a permanent
    "+1 pending" against an otherwise fully-reviewed table.

    Removes any decision whose key is absent from the current error set, then
    recomputes counts. No-op when ``issues`` is empty (we can't tell orphans
    from a transiently-unavailable validation result, so we keep everything).
    """
    if not feedback or 'user_decisions' not in feedback:
        return feedback
    error_issues = [i for i in (issues or []) if i.get('severity') == 'error']
    if not error_issues:
        return feedback
    valid_ids = {create_error_id(i) for i in error_issues}
    decisions = feedback['user_decisions']
    orphans = [eid for eid in decisions if eid not in valid_ids]
    if orphans:
        for eid in orphans:
            del decisions[eid]
        _recompute_counts(feedback)
    return feedback


def update_user_decision(feedback: Dict[str, Any], error_id: str,
                        decision: str, reason: str = '',
                        value_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Update user decision for a specific error.

    When ``value_key`` is provided, updates a single sub-value decision
    under ``user_decisions[error_id].value_decisions[value_key]`` (used for
    multi-value MCIDE coverage errors). Otherwise updates the top-level
    decision.

    Parameters:
    -----------
    feedback : dict
        Feedback structure
    error_id : str
        Error identifier (parent error)
    decision : str
        'accepted', 'rejected', or 'pending'
    reason : str, optional
        Reason for the decision
    value_key : str, optional
        A missing-value name to update inside ``value_decisions``.

    Returns:
    --------
    dict
        Updated feedback structure
    """
    entry = feedback.get('user_decisions', {}).get(error_id)
    if entry is None:
        return feedback

    now = datetime.now().isoformat()
    if value_key is None:
        entry['decision'] = decision
        entry['reason'] = reason
        entry['timestamp'] = now
    else:
        subs = entry.setdefault('value_decisions', {})
        sub = subs.setdefault(
            value_key,
            {'decision': 'pending', 'reason': '', 'timestamp': None},
        )
        sub['decision'] = decision
        sub['reason'] = reason
        sub['timestamp'] = now
    _recompute_counts(feedback)
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
    decisions = feedback.get('user_decisions', {})

    # Walk all decision atoms: sub-value decisions (for multi-value MCIDE)
    # count individually; for all other errors, the parent entry is the atom.
    atoms = []
    for entry in decisions.values():
        if entry.get('severity') != 'error':
            continue
        subs = entry.get('value_decisions')
        if isinstance(subs, dict) and subs:
            for sub in subs.values():
                atoms.append(sub.get('decision', 'pending'))
        else:
            atoms.append(entry.get('decision', 'pending'))

    if not atoms:
        # No errors at all — warnings alone don't make it incomplete
        return 'complete'

    # If any atoms are accepted (confirmed) or pending (unreviewed), keep
    # original status. Only when every atom is rejected is the table complete.
    if any(a in ('accepted', 'pending') for a in atoms):
        return original_status
    return 'complete'


def count_rejected_atoms(issue: Dict[str, Any],
                         feedback: Optional[Dict[str, Any]]) -> int:
    """How many atomic checks of this issue were rejected by the user.

    For multi-value MCIDE errors with ``value_decisions``, each rejected
    sub-value contributes 1 atom. For all other issues, the whole issue
    counts as ``atomic_count`` atoms when its top-level decision is
    'rejected', else 0.
    """
    if not feedback or not feedback.get('user_decisions'):
        return 0
    eid = create_error_id(issue)
    entry = feedback['user_decisions'].get(eid)
    if not entry:
        return 0
    subs = entry.get('value_decisions')
    details = issue.get('details') or {}
    missing = details.get('missing_values') or []
    is_multi = (
        issue.get('check_type') == 'mcide_value_coverage'
        and isinstance(subs, dict) and subs
        and isinstance(missing, list) and len(missing) > 1
    )
    if is_multi:
        return sum(
            1 for v in missing
            if isinstance(subs.get(v), dict)
            and subs[v].get('decision') == 'rejected'
        )
    return issue.get('atomic_count', 1) if entry.get('decision') == 'rejected' else 0


def flatten_mcide_for_report(validation_data: Dict[str, Any],
                             feedback: Optional[Dict[str, Any]]):
    """Preprocess a (validation, feedback) tuple so downstream reporting code
    that only understands top-level ``decision`` fields produces correct
    partial-rejection atom counts.

    For each multi-value MCIDE error whose ``value_decisions`` include at
    least one rejected or accepted sub-value, split the original error into
    up to three synthetic errors (one per decision bucket) with distinct
    messages and ``details.atomic_count`` matching each bucket's size.
    Then write matching top-level decisions into a copy of the feedback
    dict. The parent entry is removed (its sub-decisions are now expressed
    via the child entries).

    Single-value MCIDE errors, non-MCIDE errors, and multi-value MCIDE
    errors that are still all-pending pass through unchanged.

    Returns ``(adjusted_validation, adjusted_feedback)`` — deep copies; the
    inputs are not mutated.
    """
    import copy as _copy
    adj_validation = _copy.deepcopy(validation_data)
    adj_feedback = _copy.deepcopy(feedback) if feedback else {'user_decisions': {}}
    decisions = adj_feedback.setdefault('user_decisions', {})

    _DQA_CATEGORIES = ('conformance', 'completeness', 'plausibility')
    for category in _DQA_CATEGORIES:
        checks = adj_validation.get(category) or {}
        for check_name, d in checks.items():
            if d.get('check_type') != 'mcide_value_coverage':
                continue
            new_errors = []
            for err in (d.get('errors') or []):
                details = err.get('details') or {}
                missing = details.get('missing_values') or []
                if not isinstance(missing, list) or len(missing) <= 1:
                    new_errors.append(err)
                    continue
                # Recompute the parent error_id exactly how clifpy does it.
                parent_eid = create_error_id({
                    'category': category,
                    'check_type': 'mcide_value_coverage',
                    'message': err.get('message', ''),
                })
                parent = decisions.get(parent_eid)
                vd = parent.get('value_decisions') if parent else None
                if not (isinstance(vd, dict) and vd):
                    new_errors.append(err)
                    continue
                rejected_vals, accepted_vals, pending_vals = [], [], []
                for v in missing:
                    sub = vd.get(v) if isinstance(vd.get(v), dict) else None
                    decision = (sub or {}).get('decision', 'pending')
                    if decision == 'rejected':
                        rejected_vals.append(v)
                    elif decision == 'accepted':
                        accepted_vals.append(v)
                    else:
                        pending_vals.append(v)
                if not rejected_vals and not accepted_vals:
                    new_errors.append(err)
                    continue  # all pending — no split needed

                column = details.get('column') or ''

                def _mk(values, label, decision_val):
                    if not values:
                        return
                    values_str = ', '.join(str(v) for v in values)
                    msg = f"{label} {len(values)} mCIDE values: {values_str}"
                    syn_details = dict(details)
                    syn_details['missing_values'] = list(values)
                    new_errors.append({
                        'message': msg,
                        'details': syn_details,
                    })
                    syn_eid = create_error_id({
                        'category': category,
                        'check_type': 'mcide_value_coverage',
                        'message': msg,
                    })
                    # Merge per-value reasons into one combined reason string
                    reasons = []
                    for v in values:
                        r = (vd.get(v) or {}).get('reason') or ''
                        if r and r not in reasons:
                            reasons.append(r)
                    decisions[syn_eid] = {
                        'error_type': 'mcide_value_coverage',
                        'raw_type': 'mcide_value_coverage',
                        'category': category,
                        'severity': 'error',
                        'description': msg,
                        'decision': decision_val,
                        'reason': '; '.join(reasons),
                        'timestamp': None,
                    }

                _mk(rejected_vals, 'Rejected', 'rejected')
                _mk(accepted_vals, 'Acknowledged', 'accepted')
                _mk(pending_vals, 'Missing', 'pending')

                # Parent entry is now represented by its child buckets.
                decisions.pop(parent_eid, None)
            d['errors'] = new_errors
    return adj_validation, adj_feedback


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
    # Update adjusted status and timestamp before saving
    feedback['adjusted_status'] = recalculate_status(
        feedback['original_status'],
        feedback
    )
    feedback['timestamp'] = datetime.now().isoformat()

    # Ensure output directory exists (feedback now lives under validation/feedback/)
    from modules.utils.output_paths import validation_feedback_dir
    final_dir = str(validation_feedback_dir())
    os.makedirs(final_dir, exist_ok=True)

    # Save to file
    filename = f"{table_name}_validation_response.json"
    filepath = os.path.join(final_dir, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
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
    from modules.utils.output_paths import validation_feedback_dir
    filename = f"{table_name}_validation_response.json"
    filepath = str(validation_feedback_dir() / filename)

    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
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
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        return output_path

    return report_content