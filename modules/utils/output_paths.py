"""
Centralized output path construction for the CLIF-TableOne pipeline.

The output/final/ directory is organized by cohort first, artifact type second:

    output/final/
    ├── overall/             critical-illness cohort (ICU stay OR died/hospice)
    │   ├── tableone/                CSVs from the overall cohort
    │   ├── figures/                 PNG/HTML/PDF visualizations from the overall cohort
    │   ├── ecdf/                    ECDF parquets ({labs,vitals,respiratory_support})
    │   ├── bins/                    binned distributions
    │   ├── summary_stats/           summary stats JSONs/CSVs
    │   ├── mcide/                   MCIDE value counts
    │   └── ventilated_aggregates/   cross-site KM + daily P/F|S/F aggregates (no PHI)
    ├── overall_ward/        ward cohort (every adult encounter that touched a ward)
    │   ├── tableone/        ward Table One CSVs
    │   ├── figures/         ward CONSORT, sankey, code status, etc.
    │   ├── summary_stats/   ward summary stats
    │   └── strata/          stratified ward outputs (icu, advanced_resp, vaso, deaths
    │       ├── icu/{tableone,figures,summary_stats}/    subsets within the ward cohort)
    │       ├── advanced_resp/{...}
    │       ├── vaso/{...}
    │       └── deaths/{...}
    ├── strata/              stratified critical-illness outputs
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
    ├── meta/                run metadata, logs, status files
    │   ├── configs/         config snapshots used for the run
    │   └── workflow_logs/   timestamped pipeline execution logs
    └── stats/               collection_statistics.csv (ECDF coverage)

All call sites that write to output/final/ should use the helpers below
instead of hard-coding paths.

Cohort note: ward outputs are written under overall_ward/ via the parallel
ward_*_dir() helpers. The ward Table One pipeline (run_tableone_ward.py) drops
SOFA, ICU LOS, ICU episodes, IMV/ventilator settings, and the medication-from-ICU
plot, so ward output has fewer files than the critical-illness output.
"""

from pathlib import Path
from typing import Optional, Union

# Project root is two levels up from this file (modules/utils/output_paths.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Top-level directories
FINAL = PROJECT_ROOT / 'output' / 'final'
OVERALL = FINAL / 'overall'
OVERALL_WARD = FINAL / 'overall_ward'
STRATA = FINAL / 'strata'
STRATA_WARD = OVERALL_WARD / 'strata'
VALIDATION = FINAL / 'validation'
PDF_REPORTS = VALIDATION / 'pdf_reports'
META = FINAL / 'meta'
CONFIGS = META / 'configs'
STATS = FINAL / 'stats'

# Intermediate tree — unsuppressed Table One CSVs. The generator writes raw
# counts here; the small-cell suppression step reads from here and writes
# shareable (merged + cell-suppressed) CSVs under FINAL/... . The app and any
# local analysis should read from here to see full-fidelity counts.
INTERMEDIATE = PROJECT_ROOT / 'output' / 'intermediate'
TABLEONE_INTERMEDIATE = INTERMEDIATE / 'tableone'


def parse_stratum(stratum: str) -> tuple:
    """Decompose a stratum key into (parent_dir_key, filename_suffix).

    Sub-strata keys contain a slash (e.g. 'advanced_resp/icu').  The part
    before the slash is the parent directory key; the part after becomes a
    filename suffix so that all outputs land in the parent directory.

    Examples:
        'advanced_resp/icu'    -> ('advanced_resp', '_icu')
        'advanced_resp/no_icu' -> ('advanced_resp', '_no_icu')
        'vaso/icu'             -> ('vaso', '_icu')
        'icu'                  -> ('icu', '')
        'deaths'               -> ('deaths', '')
    """
    if '/' in stratum:
        parent, sub = stratum.rsplit('/', 1)
        return parent, f'_{sub}'
    return stratum, ''


def cohort_dir(stratum: Optional[str] = None) -> Path:
    """Return the cohort root directory.

    Args:
        stratum: stratum name (e.g. 'icu', 'deaths', 'advanced_resp/icu').
            Sub-strata (containing '/') resolve to the parent directory so
            that outputs are flattened with filename suffixes instead of
            nested subdirectories.
    """
    if stratum is None:
        return OVERALL
    parent, _suffix = parse_stratum(stratum)
    return STRATA / parent


def tableone_dir(stratum: Optional[str] = None) -> Path:
    """Directory for tableone analytical CSVs (under ``output/final/``).

    Mortality rates, comorbidities, code-status summaries, strobe counts,
    demographic crosstabs, ventilator settings, etc. all land here. The
    literal ``table_one_*.csv`` result files do NOT — those go through
    ``tableone_raw_dir()`` (unsuppressed, intermediate) and the small-cell
    suppression step writes their safe copy back under this directory.
    """
    return cohort_dir(stratum) / 'tableone'


def tableone_raw_dir(stratum: Optional[str] = None) -> Path:
    """Directory for **raw** (unsuppressed) Table One result CSVs.

    This is where the generator writes ``table_one_overall.csv``,
    ``table_one_by_year.csv``, and the stratified ``table_one_*.csv``
    variants. The suppression step reads from here and writes the
    consortium-safe copy to ``tableone_final_dir(stratum)``. The app and
    any local analysis read from here to see full-fidelity counts.
    """
    if stratum is None:
        return TABLEONE_INTERMEDIATE / 'overall'
    parent, _suffix = parse_stratum(stratum)
    return TABLEONE_INTERMEDIATE / 'strata' / parent


def tableone_final_dir(stratum: Optional[str] = None) -> Path:
    """Alias of ``tableone_dir(stratum)`` — destination for suppressed
    Table One result CSVs (``table_one_*.csv`` family).

    Kept as a distinct name so the suppression step's write target reads
    semantically different from ``tableone_raw_dir(stratum)`` (its source).
    """
    return tableone_dir(stratum)


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


def ventilated_aggregates_dir() -> Path:
    """Directory for cross-site shareable ventilated aggregates (KM + daily P/F|S/F CSVs).

    Only the overall critical-illness cohort produces these; skipped in ward mode.
    """
    return OVERALL / 'ventilated_aggregates'


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
    """Directory for **all-time** category counts aggregated from the monthly
    trend data. One row per category — safe under /final because the time
    dimension is collapsed away. Writers should produce per-month detail
    under ``validation_monthly_trends_raw_dir()`` and an aggregate here.
    """
    return VALIDATION / 'monthly_trends'


def validation_monthly_trends_raw_dir() -> Path:
    """Directory for **per-month** category trend CSVs (raw).

    Per-month counts can include very small N rows (single-digit patient
    counts in a specific category in a specific month), which is the same
    small-cell risk the table_one suppression policy addresses. We park
    the per-month detail in /intermediate so it's available locally for
    QA but never shipped to the consortium.
    """
    return INTERMEDIATE / 'validation' / 'monthly_trends'


def validation_pdf_reports_dir() -> Path:
    """Directory for validation report PDFs (per-table + combined)."""
    return PDF_REPORTS


def configs_dir() -> Path:
    """Directory for config snapshots (nested under meta/)."""
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


# ============================================================================
# Ward cohort helpers (parallel to overall cohort, write under overall_ward/)
# ============================================================================
#
# These mirror the overall cohort helpers above. Generator code can do a
# conditional import like:
#
#     if cohort_mode == 'ward':
#         from modules.utils.output_paths import (
#             ward_tableone_dir as _tableone_dir,
#             ward_figures_dir as _figures_dir,
#             ...
#         )
#     else:
#         from modules.utils.output_paths import (
#             tableone_dir as _tableone_dir,
#             figures_dir as _figures_dir,
#             ...
#         )
#
# so all downstream call sites stay unchanged and resolve to the right cohort
# tree depending on cohort_mode.

def ward_cohort_dir(stratum: Optional[str] = None) -> Path:
    """Return the ward cohort root directory.

    Args:
        stratum: stratum name (e.g. 'icu', 'deaths') for a subset within the ward
            cohort. None means the full ward cohort.
    """
    if stratum is None:
        return OVERALL_WARD
    parent, _suffix = parse_stratum(stratum)
    return STRATA_WARD / parent


def ward_tableone_dir(stratum: Optional[str] = None) -> Path:
    """Directory for ward tableone analytical CSVs (under ``output/final/``).

    Parallel to ``tableone_dir()`` for the ward cohort. Holds
    mortality, comorbidities, code-status summaries, etc. The literal
    ``table_one_*.csv`` result files go through ``ward_tableone_raw_dir()``.
    """
    return ward_cohort_dir(stratum) / 'tableone'


def ward_tableone_raw_dir(stratum: Optional[str] = None) -> Path:
    """Directory for **raw** ward Table One result CSVs (unsuppressed,
    local-only). Parallel to ``tableone_raw_dir()`` for the ward cohort."""
    if stratum is None:
        return TABLEONE_INTERMEDIATE / 'overall_ward'
    parent, _suffix = parse_stratum(stratum)
    return TABLEONE_INTERMEDIATE / 'overall_ward' / 'strata' / parent


def ward_tableone_final_dir(stratum: Optional[str] = None) -> Path:
    """Alias of ``ward_tableone_dir(stratum)`` — destination for suppressed
    ward Table One result CSVs."""
    return ward_tableone_dir(stratum)


def ward_figures_dir(stratum: Optional[str] = None) -> Path:
    """Directory for ward Table One figures (parallel to figures_dir())."""
    return ward_cohort_dir(stratum) / 'figures'


def ward_summary_stats_dir(stratum: Optional[str] = None) -> Path:
    """Directory for ward summary statistics JSONs/CSVs."""
    return ward_cohort_dir(stratum) / 'summary_stats'


def ward_ecdf_dir(stratum: Optional[str] = None, table_type: Optional[str] = None) -> Path:
    """Directory for ward ECDF parquets (parallel to ecdf_dir()).

    Note: the ward Table One pipeline does not currently produce ECDF outputs,
    but this helper exists so the directory layout stays symmetrical with the
    overall cohort.
    """
    base = ward_cohort_dir(stratum) / 'ecdf'
    return base if table_type is None else base / table_type


def ward_bins_dir(stratum: Optional[str] = None, table_type: Optional[str] = None) -> Path:
    """Directory for ward binned distribution parquets."""
    base = ward_cohort_dir(stratum) / 'bins'
    return base if table_type is None else base / table_type


def ward_mcide_dir() -> Path:
    """Directory for ward MCIDE value counts (only the overall ward cohort)."""
    return OVERALL_WARD / 'mcide'


def ward_validation_json_reports_dir() -> Path:
    """Re-export of the shared validation/json_reports/ directory.

    Validation outputs are not cohort-specific, so the ward pipeline shares the
    same directory as the overall cohort. This alias exists so generator code
    that imports the validation helper through the ward import block resolves
    to the correct (shared) directory.
    """
    return VALIDATION / 'json_reports'


# Strata names known to the pipeline. Mirrors modules.strata.ENCOUNTER_TYPE_STRATA keys.
# Slash-prefixed entries are sub-strata that resolve to the *parent* directory
# (e.g. 'vaso/icu' → output/final/strata/vaso/) with filename suffix '_icu'.
# See parse_stratum() for the decomposition logic.
STRATA_NAMES = ('icu',
                'advanced_resp', 'advanced_resp/icu', 'advanced_resp/no_icu',
                'nippv_hfnc', 'nippv_hfnc/icu', 'nippv_hfnc/no_icu',
                'vaso', 'vaso/icu', 'vaso/no_icu', 'vaso/ed_icu', 'vaso/ed_ward',
                'deaths')


def ensure_output_tree() -> None:
    """Create the full output/final/ directory tree.

    Idempotent — safe to call from any pipeline entry point. Creates every
    directory in the target layout (overall + each stratum + validation + meta + ...).
    """
    dirs = [
        # Overall cohort (critical-illness)
        tableone_dir(),          # final/overall/tableone/ — analytical CSVs + suppressed table_one_*.csv
        tableone_raw_dir(),      # intermediate/tableone/overall/ — raw table_one_*.csv
        figures_dir(),
        ecdf_dir(table_type='labs'),
        ecdf_dir(table_type='vitals'),
        ecdf_dir(table_type='respiratory_support'),
        bins_dir(table_type='labs'),
        bins_dir(table_type='vitals'),
        bins_dir(table_type='respiratory_support'),
        summary_stats_dir(),
        mcide_dir(),
        ventilated_aggregates_dir(),
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
            tableone_dir(stratum),      # final (analytical + suppressed table_one_*)
            tableone_raw_dir(stratum),  # intermediate (raw table_one_*)
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


def ensure_ward_output_tree() -> None:
    """Create the ward cohort output subtree under output/final/overall_ward/.

    Idempotent. Mirrors ensure_output_tree() but only creates the directories
    that the ward pipeline actually writes to. Validation, meta, configs, and
    stats directories are shared with the overall cohort and are created by
    ensure_output_tree(); call that too if running the ward pipeline standalone
    (run_tableone_ward.py does this).

    Note: the following directories are intentionally NOT created in the ward
    tree, even though parallel ward_*_dir() helpers exist for API symmetry:
    - mcide/ — populated by a separate script (modules/mcide/collector.py)
      that does not run for the ward cohort
    - summary_stats/ — same lifecycle as mcide
    - strata/{icu,advanced_resp,vaso,deaths}/* — both stratified loops in
      generator.py are skipped in ward mode (Phase 2 Decision P2-2), so no code
      writes to these subdirs. Creating them empty only adds noise.

    The ward_mcide_dir() / ward_summary_stats_dir() / ward_tableone_dir(stratum)
    / ward_figures_dir(stratum) helpers stay defined for API symmetry and
    because generator.py imports them (where they're vestigial in ward mode).
    """
    dirs = [
        ward_tableone_dir(),      # final (analytical + suppressed table_one_*)
        ward_tableone_raw_dir(),  # intermediate (raw table_one_*)
        ward_figures_dir(),
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
