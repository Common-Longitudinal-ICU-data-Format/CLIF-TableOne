"""Bridge between server session and cache_manager functions."""
from server import session
from modules.utils.cache_manager import (
    initialize_cache, cache_analysis, get_cached_analysis,
    is_table_cached, clear_all_cache, get_table_status,
    get_completion_status, get_status_display, format_cache_timestamp,
    update_feedback_in_cache,
)

def _store():
    return session.get_store()

def _config():
    return session.get("config") or {}

def init():
    initialize_cache(_store())

def cache(table_name, analyzer, validation, summary, feedback=None):
    cache_analysis(table_name, analyzer, validation, summary, feedback, _store(), _config())

def get(table_name):
    return get_cached_analysis(table_name, _store(), _config())

def is_cached(table_name):
    return is_table_cached(table_name, _store(), _config())

def status(table_name):
    return get_table_status(table_name, _store(), _config())

def completion(table_name):
    return get_completion_status(table_name, _store(), _config())

def status_display(table_name):
    return get_status_display(table_name, _store(), _config())

def update_feedback(table_name, feedback):
    update_feedback_in_cache(table_name, feedback, _store())

def clear():
    clear_all_cache(_store())
