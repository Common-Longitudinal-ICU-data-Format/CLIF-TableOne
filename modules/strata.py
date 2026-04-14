"""
Shared strata definitions and filtering utilities.

Single source of truth for encounter-type strata used across
tableone, ECDF, collection statistics, and MCIDE summary stats pipelines.

Depends on output/intermediate/final_tableone_df.parquet being generated
by the tableone pipeline first.
"""

from pathlib import Path
from typing import Dict, Optional, Set

import polars as pl


# Canonical strata definition — used by all pipelines.
#
# Slash-prefixed keys ('advanced_resp/icu', 'advanced_resp/no_icu', 'vaso/icu',
# 'vaso/no_icu') are sub-strata that split a parent stratum by ICU touch.
# parse_stratum() in output_paths.py decomposes 'advanced_resp/icu' into parent
# dir key 'advanced_resp' and filename suffix '_icu', so sub-strata outputs land
# in the parent directory with suffixed filenames (no nested subdirectories).
ENCOUNTER_TYPE_STRATA = {
    'icu': 'icu_enc',
    'advanced_resp': 'high_support_enc',
    'advanced_resp/icu': 'high_support_icu_enc',
    'advanced_resp/no_icu': 'high_support_no_icu_enc',
    'nippv_hfnc': 'nippv_hfnc_enc',
    'nippv_hfnc/icu': 'nippv_hfnc_icu_enc',
    'nippv_hfnc/no_icu': 'nippv_hfnc_no_icu_enc',
    'vaso': 'vaso_support_enc',
    'vaso/icu': 'vaso_icu_enc',
    'vaso/no_icu': 'vaso_no_icu_enc',
    'deaths': 'death_enc',
}


def load_strata_hospitalization_ids(
    intermediate_dir: str = None,
) -> Dict[str, Set[str]]:
    """
    Load final_tableone_df.parquet and return hospitalization_ids per stratum.

    Args:
        intermediate_dir: Path to intermediate output directory.
            Defaults to <project_root>/output/intermediate.

    Returns:
        Dict mapping stratum name to set of hospitalization_ids.
        e.g. {'icu': {'h1', 'h2', ...}, 'advanced_resp': {...}, ...}

    Raises:
        FileNotFoundError: If final_tableone_df.parquet doesn't exist.
    """
    if intermediate_dir is None:
        project_root = Path(__file__).parent.parent
        intermediate_dir = project_root / 'output' / 'intermediate'
    else:
        intermediate_dir = Path(intermediate_dir)

    parquet_path = intermediate_dir / 'final_tableone_df.parquet'
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Strata source not found: {parquet_path}. "
            "Run the tableone pipeline first to generate this file."
        )

    # Read only the columns we need
    cols_needed = ['hospitalization_id'] + list(ENCOUNTER_TYPE_STRATA.values())
    df = pl.scan_parquet(str(parquet_path)).select(cols_needed).collect()

    result = {}
    for stratum_name, flag_col in ENCOUNTER_TYPE_STRATA.items():
        if flag_col not in df.columns:
            continue
        hosp_ids = (
            df.filter(pl.col(flag_col) == 1)
            .get_column('hospitalization_id')
            .unique()
            .to_list()
        )
        result[stratum_name] = set(hosp_ids)

    return result


def filter_icu_windows_by_stratum(
    icu_windows: pl.DataFrame,
    stratum_hosp_ids: Set[str],
) -> pl.DataFrame:
    """Filter ICU time windows to only hospitalization_ids in the stratum."""
    return icu_windows.filter(
        pl.col('hospitalization_id').is_in(list(stratum_hosp_ids))
    )


def filter_lazy_frame_by_stratum(
    lf: pl.LazyFrame,
    stratum_hosp_ids: Set[str],
) -> pl.LazyFrame:
    """Filter a LazyFrame to only hospitalization_ids in the stratum."""
    return lf.filter(
        pl.col('hospitalization_id').is_in(list(stratum_hosp_ids))
    )
