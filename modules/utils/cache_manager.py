"""
Cache Manager for File-Based State

This module provides functions to manage persistent table analysis state
by checking for files in output/final directory, enabling state to persist
across sessions and allowing users to start fresh by deleting files.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import streamlit as st
import os
import json


def _get_output_dir() -> str:
    """Get the output directory from config."""
    if 'config' in st.session_state:
        return os.path.join(st.session_state.config.get('output_dir', 'output'), 'final')
    return os.path.join('output', 'final')


def _get_file_timestamp(filepath: str) -> Optional[str]:
    """Get file modification timestamp as ISO string."""
    if os.path.exists(filepath):
        mtime = os.path.getmtime(filepath)
        return datetime.fromtimestamp(mtime).isoformat()
    return None


def _file_exists(table_name: str, file_suffix: str) -> bool:
    """Check if a file exists for the given table."""
    output_dir = _get_output_dir()
    filepath = os.path.join(output_dir, f"{table_name}{file_suffix}")
    return os.path.exists(filepath)


def _load_json_file(table_name: str, file_suffix: str) -> Optional[Dict[str, Any]]:
    """Load a JSON file for the given table."""
    output_dir = _get_output_dir()
    filepath = os.path.join(output_dir, f"{table_name}{file_suffix}")

    if not os.path.exists(filepath):
        return None

    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def initialize_cache():
    """Initialize the cache in session state if it doesn't exist."""
    if 'analyzed_tables' not in st.session_state:
        st.session_state.analyzed_tables = {}

    if 'cache_metadata' not in st.session_state:
        st.session_state.cache_metadata = {
            'initialized': datetime.now().isoformat(),
            'last_cleared': None
        }


def cache_analysis(table_name: str, analyzer: Any, validation_results: Dict[str, Any],
                   summary_stats: Dict[str, Any], feedback: Optional[Dict[str, Any]] = None):
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
    """
    initialize_cache()

    st.session_state.analyzed_tables[table_name] = {
        'analyzer': analyzer,
        'validation': validation_results,
        'summary': summary_stats,
        'feedback': feedback,
        'timestamp': datetime.now().isoformat(),
        'config_hash': _get_config_hash(),
        'validation_complete': validation_results is not None,
        'summary_complete': summary_stats is not None,
        'validation_timestamp': datetime.now().isoformat() if validation_results else None,
        'summary_timestamp': datetime.now().isoformat() if summary_stats else None
    }


def get_cached_analysis(table_name: str) -> Optional[Dict[str, Any]]:
    """
    Get cached analysis for a table by checking files in output/final.

    Parameters:
    -----------
    table_name : str
        Name of the table

    Returns:
    --------
    dict or None
        Cached analysis structure or None if not found
    """
    initialize_cache()

    output_dir = _get_output_dir()

    # Check for validation response file (has feedback and adjusted status)
    feedback = _load_json_file(table_name, '_validation_response.json')

    # Check for raw validation file
    validation = _load_json_file(table_name, '_summary_validation.json')

    # Check for summary file
    summary = _load_json_file(table_name, '_summary_summary.json')

    # Check for CSV validation artifacts
    validation_csv_exists = _file_exists(table_name, '.csv') or \
                           os.path.exists(os.path.join(output_dir, f'validation_errors_{table_name}.csv'))
    missing_csv_exists = os.path.exists(os.path.join(output_dir, f'missing_data_stats_{table_name}.csv'))

    # Determine if anything exists
    has_validation = validation is not None or validation_csv_exists
    has_summary = summary is not None

    if not has_validation and not has_summary and feedback is None:
        return None

    # Get timestamp from most recent file
    timestamps = []
    for suffix in ['_validation_response.json', '_summary_validation.json', '_summary_summary.json']:
        filepath = os.path.join(output_dir, f"{table_name}{suffix}")
        ts = _get_file_timestamp(filepath)
        if ts:
            timestamps.append(ts)

    # Add CSV file timestamps
    for csv_file in [f'validation_errors_{table_name}.csv', f'missing_data_stats_{table_name}.csv']:
        filepath = os.path.join(output_dir, csv_file)
        ts = _get_file_timestamp(filepath)
        if ts:
            timestamps.append(ts)

    timestamp = max(timestamps) if timestamps else datetime.now().isoformat()

    # Return cached structure
    return {
        'analyzer': st.session_state.analyzed_tables.get(table_name, {}).get('analyzer'),  # Keep analyzer in memory
        'validation': validation,
        'summary': summary,
        'feedback': feedback,
        'timestamp': timestamp,
        'validation_complete': has_validation,
        'summary_complete': has_summary,
        'validation_timestamp': timestamp if has_validation else None,
        'summary_timestamp': timestamp if has_summary else None
    }


def is_table_cached(table_name: str) -> bool:
    """
    Check if a table has cached analysis.

    Parameters:
    -----------
    table_name : str
        Name of the table

    Returns:
    --------
    bool
        True if table is cached, False otherwise
    """
    return get_cached_analysis(table_name) is not None


def clear_table_cache(table_name: str):
    """
    Clear cache for a specific table.

    Parameters:
    -----------
    table_name : str
        Name of the table
    """
    initialize_cache()

    if table_name in st.session_state.analyzed_tables:
        del st.session_state.analyzed_tables[table_name]


def clear_all_cache():
    """Clear all cached analyses."""
    st.session_state.analyzed_tables = {}
    st.session_state.cache_metadata['last_cleared'] = datetime.now().isoformat()


def get_cache_summary() -> Dict[str, Any]:
    """
    Get summary of cache status.

    Returns:
    --------
    dict
        Cache summary with counts and timestamps
    """
    initialize_cache()

    cached_tables = list(st.session_state.analyzed_tables.keys())

    summary = {
        'total_cached': len(cached_tables),
        'cached_tables': cached_tables,
        'initialized': st.session_state.cache_metadata.get('initialized'),
        'last_cleared': st.session_state.cache_metadata.get('last_cleared')
    }

    # Add timestamps for each cached table
    table_timestamps = {}
    for table_name in cached_tables:
        table_timestamps[table_name] = st.session_state.analyzed_tables[table_name]['timestamp']

    summary['table_timestamps'] = table_timestamps

    return summary


def update_feedback_in_cache(table_name: str, feedback: Dict[str, Any]):
    """
    Update feedback for a cached table.

    Parameters:
    -----------
    table_name : str
        Name of the table
    feedback : dict
        Updated feedback structure
    """
    initialize_cache()

    if table_name in st.session_state.analyzed_tables:
        st.session_state.analyzed_tables[table_name]['feedback'] = feedback
        st.session_state.analyzed_tables[table_name]['feedback_updated'] = datetime.now().isoformat()


def get_table_status(table_name: str) -> str:
    """
    Get the current status of a table (considering feedback from files).

    Parameters:
    -----------
    table_name : str
        Name of the table

    Returns:
    --------
    str
        Status: 'not_analyzed', 'complete', 'partial', 'incomplete', or adjusted status
    """
    cached = get_cached_analysis(table_name)

    if not cached:
        return 'not_analyzed'

    # Check if there's feedback file with adjusted status (priority)
    if cached.get('feedback'):
        return cached['feedback'].get('adjusted_status', cached['feedback'].get('original_status', 'unknown'))

    # Check validation results
    if cached.get('validation'):
        return cached['validation'].get('status', 'unknown')

    return 'unknown'


def get_completion_status(table_name: str) -> Dict[str, bool]:
    """
    Get completion status for validation and summarization separately.

    Parameters:
    -----------
    table_name : str
        Name of the table

    Returns:
    --------
    dict
        Dictionary with 'validation_complete' and 'summary_complete' booleans
    """
    cached = get_cached_analysis(table_name)

    if not cached:
        return {
            'validation_complete': False,
            'summary_complete': False
        }

    return {
        'validation_complete': cached.get('validation_complete', False),
        'summary_complete': cached.get('summary_complete', False)
    }


def get_status_display(table_name: str) -> str:
    """
    Get formatted status display for sidebar.

    Parameters:
    -----------
    table_name : str
        Name of the table

    Returns:
    --------
    str
        Formatted status string
    """
    cached = get_cached_analysis(table_name)

    if not cached:
        return "Not analyzed"

    completion = get_completion_status(table_name)
    val_complete = completion['validation_complete']
    sum_complete = completion['summary_complete']

    if val_complete and sum_complete:
        status = get_table_status(table_name)
        return f"{status.upper()}"
    elif val_complete:
        status = get_table_status(table_name)
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


def _get_config_hash() -> str:
    """
    Get a hash of the current config to detect changes.

    Returns:
    --------
    str
        Hash of config
    """
    import hashlib
    import json

    if 'config' not in st.session_state:
        return ''

    config = st.session_state.config
    # Create hash of key config values
    config_str = json.dumps({
        'tables_path': config.get('tables_path'),
        'filetype': config.get('filetype'),
        'timezone': config.get('timezone')
    }, sort_keys=True)

    return hashlib.md5(config_str.encode()).hexdigest()


def get_cache_statistics() -> Dict[str, Any]:
    """
    Get detailed statistics about the cache.

    Returns:
    --------
    dict
        Cache statistics
    """
    initialize_cache()

    stats = {
        'total_tables': len(st.session_state.analyzed_tables),
        'tables_with_feedback': 0,
        'tables_with_errors': 0,
        'tables_complete': 0,
        'tables_partial': 0,
        'tables_incomplete': 0
    }

    for table_name, cached in st.session_state.analyzed_tables.items():
        # Count feedback
        if cached.get('feedback'):
            stats['tables_with_feedback'] += 1

        # Count errors
        validation = cached.get('validation', {})
        errors = validation.get('errors', {})
        error_count = sum([
            len(errors.get('schema_errors', [])),
            len(errors.get('data_quality_issues', [])),
            len(errors.get('other_errors', []))
        ])
        if error_count > 0:
            stats['tables_with_errors'] += 1

        # Count by status
        status = get_table_status(table_name)
        if status == 'complete':
            stats['tables_complete'] += 1
        elif status == 'partial':
            stats['tables_partial'] += 1
        elif status == 'incomplete':
            stats['tables_incomplete'] += 1

    return stats