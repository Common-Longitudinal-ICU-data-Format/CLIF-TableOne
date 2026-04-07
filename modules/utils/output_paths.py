"""
Centralized output path construction for the CLIF-TableOne pipeline.

The output/final/ directory is organized by cohort first, artifact type second:

    output/final/
    ├── overall/
    │   ├── tableone/        CSVs from the overall cohort
    │   ├── figures/         PNG/HTML/PDF visualizations from the overall cohort
    │   ├── ecdf/            ECDF parquets ({labs,vitals,respiratory_support})
    │   ├── bins/            binned distributions
    │   ├── summary_stats/   summary stats JSONs/CSVs
    │   └── mcide/           MCIDE value counts
    ├── strata/
    │   ├── icu/{tableone,figures,ecdf,bins,summary_stats}/
    │   ├── advanced_resp/{...}
    │   ├── vaso/{...}
    │   └── deaths/{...}
    ├── validation/          data quality assessment (merges old clifpy/ + results/)
    │   ├── json_reports/    <table>_dqa.json + missing_data_stats / validation_errors CSVs
    │   ├── consolidated/    consolidated_validation.csv + *_summary*.json
    │   ├── feedback/        *_validation_response.json
    │   ├── monthly_trends/  monthly trend CSVs
    │   └── pdf_reports/     validation report PDFs (per-table + combined)
    ├── configs/             config snapshots used for the run
    ├── meta/                run metadata, logs, status files
    │   └── workflow_logs/   timestamped pipeline execution logs
    └── stats/               collection_statistics.csv (ECDF coverage)

All call sites that write to output/final/ should use the helpers below
instead of hard-coding paths.
"""

from pathlib import Path
from typing import Optional, Union

# Project root is two levels up from this file (modules/utils/output_paths.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Top-level directories
FINAL = PROJECT_ROOT / 'output' / 'final'
OVERALL = FINAL / 'overall'
STRATA = FINAL / 'strata'
VALIDATION = FINAL / 'validation'
PDF_REPORTS = VALIDATION / 'pdf_reports'
CONFIGS = FINAL / 'configs'
META = FINAL / 'meta'
STATS = FINAL / 'stats'


def cohort_dir(stratum: Optional[str] = None) -> Path:
    """Return the cohort root directory.

    Args:
        stratum: stratum name (e.g. 'icu', 'deaths'). None means the overall cohort.
    """
    return OVERALL if stratum is None else STRATA / stratum


def tableone_dir(stratum: Optional[str] = None) -> Path:
    """Directory for tableone CSVs (table_one_overall, mortality_rates, etc.)."""
    return cohort_dir(stratum) / 'tableone'


def figures_dir(stratum: Optional[str] = None) -> Path:
    """Directory for tableone figures (PNG/HTML/PDF visualizations)."""
    return cohort_dir(stratum) / 'figures'


def ecdf_dir(stratum: Optional[str] = None, table_type: Optional[str] = None) -> Path:
    """Directory for ECDF parquets.

    Args:
        stratum: stratum name, or None for the overall cohort.
        table_type: one of 'labs', 'vitals', 'respiratory_support'. If None,
            returns the parent ecdf/ directory.
    """
    base = cohort_dir(stratum) / 'ecdf'
    return base if table_type is None else base / table_type


def bins_dir(stratum: Optional[str] = None, table_type: Optional[str] = None) -> Path:
    """Directory for binned distribution parquets."""
    base = cohort_dir(stratum) / 'bins'
    return base if table_type is None else base / table_type


def summary_stats_dir(stratum: Optional[str] = None) -> Path:
    """Directory for summary statistics JSONs/CSVs."""
    return cohort_dir(stratum) / 'summary_stats'


def mcide_dir() -> Path:
    """Directory for MCIDE value counts. Only the overall cohort has MCIDE outputs."""
    return OVERALL / 'mcide'


def validation_json_reports_dir() -> Path:
    """Directory for per-table DQA JSONs and clifpy CSVs (validation/json_reports/)."""
    return VALIDATION / 'json_reports'


def validation_consolidated_dir() -> Path:
    """Directory for consolidated DQA outputs (consolidated_validation.csv, summary JSONs)."""
    return VALIDATION / 'consolidated'


def validation_feedback_dir() -> Path:
    """Directory for user feedback on validation errors."""
    return VALIDATION / 'feedback'


def validation_monthly_trends_dir() -> Path:
    """Directory for monthly trend CSVs."""
    return VALIDATION / 'monthly_trends'


def validation_pdf_reports_dir() -> Path:
    """Directory for validation report PDFs (per-table + combined)."""
    return PDF_REPORTS


def configs_dir() -> Path:
    """Directory for config snapshots used during the run."""
    return CONFIGS


def meta_dir() -> Path:
    """Directory for run metadata (file_metadata.json, execution reports, log files)."""
    return META


def workflow_logs_dir() -> Path:
    """Directory for timestamped pipeline execution logs."""
    return META / 'workflow_logs'


def stats_dir() -> Path:
    """Directory for collection_statistics.csv (ECDF coverage)."""
    return STATS


# Strata names known to the pipeline. Mirrors modules.strata.ENCOUNTER_TYPE_STRATA keys.
STRATA_NAMES = ('icu', 'advanced_resp', 'vaso', 'deaths')


def ensure_output_tree() -> None:
    """Create the full output/final/ directory tree.

    Idempotent — safe to call from any pipeline entry point. Creates every
    directory in the target layout (overall + each stratum + validation + meta + ...).
    """
    dirs = [
        # Overall cohort
        tableone_dir(),
        figures_dir(),
        ecdf_dir(table_type='labs'),
        ecdf_dir(table_type='vitals'),
        ecdf_dir(table_type='respiratory_support'),
        bins_dir(table_type='labs'),
        bins_dir(table_type='vitals'),
        bins_dir(table_type='respiratory_support'),
        summary_stats_dir(),
        mcide_dir(),
        # Validation (DQA)
        validation_json_reports_dir(),
        validation_consolidated_dir(),
        validation_feedback_dir(),
        validation_monthly_trends_dir(),
        PDF_REPORTS,
        # Top-level
        CONFIGS,
        META,
        workflow_logs_dir(),
        STATS,
    ]
    # Strata
    for stratum in STRATA_NAMES:
        dirs.extend([
            tableone_dir(stratum),
            figures_dir(stratum),
            ecdf_dir(stratum, 'labs'),
            ecdf_dir(stratum, 'vitals'),
            ecdf_dir(stratum, 'respiratory_support'),
            bins_dir(stratum, 'labs'),
            bins_dir(stratum, 'vitals'),
            bins_dir(stratum, 'respiratory_support'),
            summary_stats_dir(stratum),
        ])

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
