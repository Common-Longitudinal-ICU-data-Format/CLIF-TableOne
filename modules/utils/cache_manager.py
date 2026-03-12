"""
Cache Manager for File-Based State

This module provides functions to manage persistent table analysis state
by checking for files in output/final directory, enabling state to persist
across sessions and allowing users to start fresh by deleting files.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import os
import json


def _get_output_dir(config: dict) -> str:
    """Get the output directory from config."""
    if config:
        return os.path.join(config.get('output_dir', 'output'), 'final')
    return os.path.join('output', 'final')


def _get_file_timestamp(filepath: str) -> Optional[str]:
    """Get file modification timestamp as ISO string."""
    if os.path.exists(filepath):
        mtime = os.path.getmtime(filepath)
        return datetime.fromtimestamp(mtime).isoformat()
    return None


def _file_exists(table_name: str, file_suffix: str, config: dict) -> bool:
    """Check if a file exists for the given table."""
    output_dir = _get_output_dir(config)
    filepath = os.path.join(output_dir, f"{table_name}{file_suffix}")
    return os.path.exists(filepath)


def _load_json_file(table_name: str, file_suffix: str, config: dict) -> Optional[Dict[str, Any]]:
    """Load a JSON file for the given table."""
    output_dir = _get_output_dir(config)
    filepath = os.path.join(output_dir, f"{table_name}{file_suffix}")

    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def initialize_cache(store: dict):
    """Initialize the cache in session state if it doesn't exist."""
    if 'analyzed_tables' not in store:
        store['analyzed_tables'] = {}

    if 'cache_metadata' not in store:
        store['cache_metadata'] = {
            'initialized': datetime.now().isoformat(),
            'last_cleared': None
        }


def cache_analysis(table_name: str, analyzer: Any, validation_results: Dict[str, Any],
                   summary_stats: Dict[str, Any], feedback: Optional[Dict[str, Any]] = None,
                   store: dict = None, config: dict = None):
    """
    Cache analysis results for a table.

    Parameters:
    -----------
    table_name : str
        Name of the table
    analyzer : BaseTableAnalyzer
        The analyzer instance (note: storing the full object)
    validation_results : dict
        Validation results
    summary_stats : dict
        Summary statistics
    feedback : dict, optional
        User feedback on validation errors
    store : dict
        Session store dictionary
    config : dict
        Configuration dictionary
    """
    if store is None:
        store = {}
    if config is None:
        config = {}

    initialize_cache(store)

    store['analyzed_tables'][table_name] = {
        'analyzer': analyzer,
        'validation': validation_results,
        'summary': summary_stats,
        'feedback': feedback,
        'timestamp': datetime.now().isoformat(),
        'config_hash': _get_config_hash(config),
        'validation_complete': validation_results is not None,
        'summary_complete': summary_stats is not None,
        'validation_timestamp': datetime.now().isoformat() if validation_results else None,
        'summary_timestamp': datetime.now().isoformat() if summary_stats else None
    }


def get_cached_analysis(table_name: str, store: dict, config: dict) -> Optional[Dict[str, Any]]:
    """
    Get cached analysis for a table by checking files in output/final.

    Parameters:
    -----------
    table_name : str
        Name of the table
    store : dict
        Session store dictionary
    config : dict
        Configuration dictionary

    Returns:
    --------
    dict or None
        Cached analysis structure or None if not found
    """
    initialize_cache(store)

    output_dir = _get_output_dir(config)
    results_dir = os.path.join(output_dir, 'results')

    # Check for validation response file (has feedback and adjusted status) in results subdirectory
    feedback = None
    feedback_path = os.path.join(results_dir, f"{table_name}_validation_response.json")
    if os.path.exists(feedback_path):
        try:
            with open(feedback_path, 'r', encoding='utf-8') as f:
                feedback = json.load(f)
        except Exception as e:
            print(f"Error loading feedback: {e}")

    # Check for raw validation file in results subdirectory, then fallback to clifpy/ DQA file
    validation = None
    validation_path = os.path.join(results_dir, f"{table_name}_summary_validation.json")
    if os.path.exists(validation_path):
        try:
            with open(validation_path, 'r', encoding='utf-8') as f:
                validation = json.load(f)
        except Exception as e:
            print(f"Error loading validation: {e}")

    if validation is None:
        clifpy_path = os.path.join(output_dir, 'clifpy', f"{table_name}_dqa.json")
        if os.path.exists(clifpy_path):
            try:
                with open(clifpy_path, 'r', encoding='utf-8') as f:
                    validation = json.load(f)
            except Exception as e:
                print(f"Error loading clifpy DQA: {e}")

    # Check for summary file in results subdirectory
    summary = None
    summary_path = os.path.join(results_dir, f"{table_name}_summary_summary.json")
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
        except Exception as e:
            print(f"Error loading summary: {e}")

    # Check for CSV validation artifacts
    validation_csv_exists = _file_exists(table_name, '.csv', config) or \
                           os.path.exists(os.path.join(output_dir, f'validation_errors_{table_name}.csv'))
    missing_csv_exists = os.path.exists(os.path.join(output_dir, f'missing_data_stats_{table_name}.csv'))

    # Determine if anything exists
    has_validation = validation is not None or validation_csv_exists
    has_summary = summary is not None

    if not has_validation and not has_summary and feedback is None:
        return None

    # Get timestamp from most recent file in results subdirectory
    timestamps = []
    for suffix in ['_validation_response.json', '_summary_validation.json', '_summary_summary.json']:
        filepath = os.path.join(results_dir, f"{table_name}{suffix}")
        ts = _get_file_timestamp(filepath)
        if ts:
            timestamps.append(ts)

    # Add clifpy/ DQA file timestamp
    clifpy_ts = _get_file_timestamp(os.path.join(output_dir, 'clifpy', f"{table_name}_dqa.json"))
    if clifpy_ts:
        timestamps.append(clifpy_ts)

    # Add CSV file timestamps
    for csv_file in [f'validation_errors_{table_name}.csv', f'missing_data_stats_{table_name}.csv']:
        filepath = os.path.join(output_dir, csv_file)
        ts = _get_file_timestamp(filepath)
        if ts:
            timestamps.append(ts)

    timestamp = max(timestamps) if timestamps else datetime.now().isoformat()

    # Return cached structure
    return {
        'analyzer': store.get('analyzed_tables', {}).get(table_name, {}).get('analyzer'),  # Keep analyzer in memory
        'validation': validation,
        'summary': summary,
        'feedback': feedback,
        'timestamp': timestamp,
        'validation_complete': has_validation,
        'summary_complete': has_summary,
        'validation_timestamp': timestamp if has_validation else None,
        'summary_timestamp': timestamp if has_summary else None
    }


def is_table_cached(table_name: str, store: dict, config: dict) -> bool:
    """
    Check if a table has cached analysis.

    Parameters:
    -----------
    table_name : str
        Name of the table
    store : dict
        Session store dictionary
    config : dict
        Configuration dictionary

    Returns:
    --------
    bool
        True if table is cached, False otherwise
    """
    return get_cached_analysis(table_name, store, config) is not None


def clear_table_cache(table_name: str, store: dict):
    """
    Clear cache for a specific table.

    Parameters:
    -----------
    table_name : str
        Name of the table
    store : dict
        Session store dictionary
    """
    initialize_cache(store)

    if table_name in store['analyzed_tables']:
        del store['analyzed_tables'][table_name]


def clear_all_cache(store: dict):
    """Clear all cached analyses."""
    store['analyzed_tables'] = {}
    if 'cache_metadata' not in store:
        store['cache_metadata'] = {}
    store['cache_metadata']['last_cleared'] = datetime.now().isoformat()


def get_cache_summary(store: dict) -> Dict[str, Any]:
    """
    Get summary of cache status.

    Parameters:
    -----------
    store : dict
        Session store dictionary

    Returns:
    --------
    dict
        Cache summary with counts and timestamps
    """
    initialize_cache(store)

    cached_tables = list(store['analyzed_tables'].keys())

    summary = {
        'total_cached': len(cached_tables),
        'cached_tables': cached_tables,
        'initialized': store['cache_metadata'].get('initialized'),
        'last_cleared': store['cache_metadata'].get('last_cleared')
    }

    # Add timestamps for each cached table
    table_timestamps = {}
    for table_name in cached_tables:
        table_timestamps[table_name] = store['analyzed_tables'][table_name]['timestamp']

    summary['table_timestamps'] = table_timestamps

    return summary


def update_feedback_in_cache(table_name: str, feedback: Dict[str, Any], store: dict):
    """
    Update feedback for a cached table.

    Parameters:
    -----------
    table_name : str
        Name of the table
    feedback : dict
        Updated feedback structure
    store : dict
        Session store dictionary
    """
    initialize_cache(store)

    if table_name in store['analyzed_tables']:
        store['analyzed_tables'][table_name]['feedback'] = feedback
        store['analyzed_tables'][table_name]['feedback_updated'] = datetime.now().isoformat()


def get_table_status(table_name: str, store: dict, config: dict) -> str:
    """
    Get the current status of a table (considering feedback from files).

    Parameters:
    -----------
    table_name : str
        Name of the table
    store : dict
        Session store dictionary
    config : dict
        Configuration dictionary

    Returns:
    --------
    str
        Status: 'not_analyzed', 'complete', 'partial', 'incomplete', or adjusted status
    """
    cached = get_cached_analysis(table_name, store, config)

    if not cached:
        return 'not_analyzed'

    # Check if there's feedback file with adjusted status (priority)
    if cached.get('feedback'):
        return cached['feedback'].get('adjusted_status', cached['feedback'].get('original_status', 'unknown'))

    # Check validation results
    if cached.get('validation'):
        validation = cached['validation']
        # Legacy format: top-level 'status' key
        if 'status' in validation:
            return validation['status']

        # New DQA format: derive status from per-category check results
        try:
            from modules.cli.pdf_generator import _collect_dqa_issues
            category_scores, all_issues = _collect_dqa_issues(validation)
            total_passed = sum(p for p, _ in category_scores.values())
            total_checks = sum(t for _, t in category_scores.values())
            error_count = sum(1 for i in all_issues if i['severity'] == 'error')

            if total_checks == 0:
                # Check if DQA categories exist but all checks passed/not-applicable
                has_categories = any(
                    validation.get(cat) for cat in ('conformance', 'completeness', 'plausibility')
                )
                return 'complete' if has_categories else 'unknown'
            if total_passed == total_checks:
                return 'complete'
            if error_count == 0:
                return 'partial'
            return 'incomplete'
        except Exception:
            return 'unknown'

    return 'unknown'


def get_completion_status(table_name: str, store: dict, config: dict) -> Dict[str, bool]:
    """
    Get completion status for validation and summarization separately.

    Parameters:
    -----------
    table_name : str
        Name of the table
    store : dict
        Session store dictionary
    config : dict
        Configuration dictionary

    Returns:
    --------
    dict
        Dictionary with 'validation_complete' and 'summary_complete' booleans
    """
    cached = get_cached_analysis(table_name, store, config)

    if not cached:
        return {
            'validation_complete': False,
            'summary_complete': False
        }

    return {
        'validation_complete': cached.get('validation_complete', False),
        'summary_complete': cached.get('summary_complete', False)
    }


def get_status_display(table_name: str, store: dict, config: dict) -> str:
    """
    Get formatted status display for sidebar.

    Parameters:
    -----------
    table_name : str
        Name of the table
    store : dict
        Session store dictionary
    config : dict
        Configuration dictionary

    Returns:
    --------
    str
        Formatted status string
    """
    cached = get_cached_analysis(table_name, store, config)

    if not cached:
        return "Not analyzed"

    completion = get_completion_status(table_name, store, config)
    val_complete = completion['validation_complete']
    sum_complete = completion['summary_complete']

    if val_complete and sum_complete:
        status = get_table_status(table_name, store, config)
        return f"{status.upper()}"
    elif val_complete:
        status = get_table_status(table_name, store, config)
        return f"{status.upper()}"
    elif sum_complete:
        return "SUMMARY ONLY"
    else:
        return "Loaded (no analysis)"


def format_cache_timestamp(timestamp_iso: str) -> str:
    """
    Format ISO timestamp for display.

    Parameters:
    -----------
    timestamp_iso : str
        ISO format timestamp

    Returns:
    --------
    str
        Formatted timestamp
    """
    try:
        dt = datetime.fromisoformat(timestamp_iso)

        # Check if today
        now = datetime.now()
        if dt.date() == now.date():
            return dt.strftime("Today at %I:%M %p")

        # Check if yesterday
        from datetime import timedelta
        yesterday = now - timedelta(days=1)
        if dt.date() == yesterday.date():
            return dt.strftime("Yesterday at %I:%M %p")

        # Otherwise show date
        return dt.strftime("%Y-%m-%d %I:%M %p")
    except:
        return "Unknown"


def _get_config_hash(config: dict) -> str:
    """
    Get a hash of the current config to detect changes.

    Parameters:
    -----------
    config : dict
        Configuration dictionary

    Returns:
    --------
    str
        Hash of config
    """
    import hashlib
    import json

    if not config:
        return ''

    # Create hash of key config values
    config_str = json.dumps({
        'tables_path': config.get('tables_path'),
        'filetype': config.get('filetype'),
        'timezone': config.get('timezone')
    }, sort_keys=True)

    return hashlib.md5(config_str.encode()).hexdigest()


def get_cache_statistics(store: dict, config: dict) -> Dict[str, Any]:
    """
    Get detailed statistics about the cache.

    Parameters:
    -----------
    store : dict
        Session store dictionary
    config : dict
        Configuration dictionary

    Returns:
    --------
    dict
        Cache statistics
    """
    initialize_cache(store)

    stats = {
        'total_tables': len(store['analyzed_tables']),
        'tables_with_feedback': 0,
        'tables_with_errors': 0,
        'tables_complete': 0,
        'tables_partial': 0,
        'tables_incomplete': 0
    }

    for table_name, cached in store['analyzed_tables'].items():
        # Count feedback
        if cached.get('feedback'):
            stats['tables_with_feedback'] += 1

        # Count errors using DQA issue format
        validation = cached.get('validation', {})
        try:
            from modules.cli.pdf_generator import _collect_dqa_issues
            _, issues = _collect_dqa_issues(validation)
            error_count = sum(1 for i in issues if i['severity'] in ('error', 'warning'))
        except Exception:
            error_count = 0
        if error_count > 0:
            stats['tables_with_errors'] += 1

        # Count by status
        status = get_table_status(table_name, store, config)
        if status == 'complete':
            stats['tables_complete'] += 1
        elif status == 'partial':
            stats['tables_partial'] += 1
        elif status == 'incomplete':
            stats['tables_incomplete'] += 1

    return stats
