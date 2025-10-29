"""Utility modules for CLIF analysis"""

from .validation import (
    format_clifpy_error,
    determine_validation_status,
    get_validation_summary
)
from .missingness import (
    calculate_missingness,
    get_high_missingness_columns,
    get_missingness_summary,
    create_missingness_report
)
from .distributions import (
    generate_ecdf,
    get_categorical_distribution,
    get_numeric_distribution
)
from .feedback import (
    create_error_id,
    create_feedback_structure,
    update_user_decision,
    get_accepted_errors,
    get_rejected_errors,
    recalculate_status,
    save_feedback,
    load_feedback,
    get_feedback_summary,
    export_feedback_report
)
from .cache_manager import (
    initialize_cache,
    cache_analysis,
    get_cached_analysis,
    is_table_cached,
    clear_table_cache,
    clear_all_cache,
    get_cache_summary,
    update_feedback_in_cache,
    get_table_status,
    format_cache_timestamp,
    get_cache_statistics,
    get_completion_status,
    get_status_display
)
from .categorical_numeric_viz import (
    show_categorical_numeric_distribution
)
from .sampling import (
    get_icu_hospitalizations_from_adt,
    generate_stratified_sample,
    save_sample_list,
    load_sample_list,
    sample_exists
)
from .datetime_utils import (
    standardize_datetime_columns,
    ensure_datetime_precision_match,
    standardize_datetime_for_comparison
)

__all__ = [
    # Validation
    'format_clifpy_error',
    'determine_validation_status',
    'get_validation_summary',
    # Missingness
    'calculate_missingness',
    'get_high_missingness_columns',
    'get_missingness_summary',
    'create_missingness_report',
    # Distributions
    'generate_ecdf',
    'get_categorical_distribution',
    'get_numeric_distribution',
    # Feedback
    'create_error_id',
    'create_feedback_structure',
    'update_user_decision',
    'get_accepted_errors',
    'get_rejected_errors',
    'recalculate_status',
    'save_feedback',
    'load_feedback',
    'get_feedback_summary',
    'export_feedback_report',
    # Cache
    'initialize_cache',
    'cache_analysis',
    'get_cached_analysis',
    'is_table_cached',
    'clear_table_cache',
    'clear_all_cache',
    'get_cache_summary',
    'update_feedback_in_cache',
    'get_table_status',
    'format_cache_timestamp',
    'get_cache_statistics',
    'get_completion_status',
    'get_status_display',
    # Categorical-Numeric Visualization
    'show_categorical_numeric_distribution',
    # Sampling
    'get_icu_hospitalizations_from_adt',
    'generate_stratified_sample',
    'save_sample_list',
    'load_sample_list',
    'sample_exists',
    # Datetime utilities
    'standardize_datetime_columns',
    'ensure_datetime_precision_match',
    'standardize_datetime_for_comparison'
]