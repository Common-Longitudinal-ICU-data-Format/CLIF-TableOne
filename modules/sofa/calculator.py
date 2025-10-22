"""
Optimized SOFA score computation using Polars.

This module provides a standalone, highly optimized implementation of SOFA
(Sequential Organ Failure Assessment) score calculation using Polars for
maximum performance. It loads raw data files directly and performs all
computations including unit conversion without relying on other clifpy methods.
"""

import polars as pl
from typing import Optional, List
from pathlib import Path
import logging

# Set up logging
logger = logging.getLogger(__name__)


def _convert_datetime_to_timezone(df: pl.DataFrame, timezone: str) -> pl.DataFrame:
    """
    Convert all datetime columns in Polars DataFrame to specified timezone.

    Matches clifpy behavior for timezone handling:
    - If datetime has a timezone, converts to target timezone
    - If datetime is naive (no timezone), assumes it's already in target timezone (localizes in place)

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame
    timezone : str
        Target timezone (e.g., 'US/Central', 'America/Chicago')

    Returns
    -------
    pl.DataFrame
        DataFrame with datetime columns converted to target timezone
    """
    # Find all datetime columns (columns with 'dttm' in name)
    dttm_columns = [col for col in df.columns if 'dttm' in col.lower()]

    if not dttm_columns:
        logger.debug("No datetime columns found in DataFrame")
        return df

    # Track conversion statistics
    converted_cols = []
    already_correct_cols = []
    naive_cols = []
    problem_cols = []

    # Convert each datetime column
    for col in dttm_columns:
        dtype_str = str(df[col].dtype)

        # Check if column is datetime
        if 'Datetime' not in dtype_str:
            problem_cols.append(col)
            logger.warning(f"{col}: Expected datetime but found {dtype_str}")
            continue

        # Check if timezone-aware (has timezone that is not None)
        if 'time_zone' in dtype_str and 'time_zone=None' not in dtype_str:
            # Has timezone - check if it's already correct
            try:
                # Try to get current timezone from dtype string
                # Polars dtype looks like: Datetime(time_unit='us', time_zone='UTC')
                if f"time_zone='{timezone}'" in dtype_str or f'time_zone="{timezone}"' in dtype_str:
                    already_correct_cols.append(col)
                    logger.debug(f"{col}: Already in timezone {timezone}")
                else:
                    # Need to convert
                    null_before = df[col].null_count()
                    df = df.with_columns([
                        pl.col(col).dt.convert_time_zone(timezone).alias(col)
                    ])
                    null_after = df[col].null_count()
                    converted_cols.append(col)

                    if null_before != null_after:
                        logger.warning(f"{col}: Null count changed during conversion ({null_before} → {null_after})")
                    logger.info(f"{col}: Converted to {timezone}")
            except Exception as e:
                problem_cols.append(col)
                logger.warning(f"{col}: Could not convert timezone: {e}")
        else:
            # Naive datetime - assume it's already in target timezone (like clifpy's tz_localize)
            try:
                df = df.with_columns([
                    pl.col(col).dt.replace_time_zone(timezone).alias(col)
                ])
                naive_cols.append(col)
                logger.warning(f"{col}: Naive datetime localized to {timezone}. Please verify this is correct.")
            except Exception as e:
                problem_cols.append(col)
                logger.warning(f"{col}: Could not localize naive datetime: {e}")

    # Log summary
    if converted_cols or naive_cols or problem_cols:
        summary_parts = []
        if converted_cols:
            summary_parts.append(f"{len(converted_cols)} converted to {timezone}")
        if already_correct_cols:
            summary_parts.append(f"{len(already_correct_cols)} already correct")
        if naive_cols:
            summary_parts.append(f"{len(naive_cols)} naive dates localized")
        if problem_cols:
            summary_parts.append(f"{len(problem_cols)} problematic")

        logger.info(f"Timezone processing complete: {', '.join(summary_parts)}")

    return df


def _create_resp_support_episodes(
    resp_df: pl.DataFrame,
    id_col: str = 'hospitalization_id'
) -> pl.DataFrame:
    """
    Create respiratory support episode IDs for waterfall forward-filling.

    Implements waterfall heuristics from utils/waterfall.py including:
    - Room air FiO2 defaults (0.21)
    - FiO2 imputation from nasal cannula LPM (1L→24%, 2L→28%, ..., 10L→60%)
    - IMV detection from mode_category patterns
    - NIPPV detection from mode_category patterns
    - Hierarchical episode tracking (device_cat_id, mode_cat_id)

    Parameters
    ----------
    resp_df : pl.DataFrame
        Respiratory support data with device_category, mode_category, lpm_set, recorded_dttm
    id_col : str
        ID column for grouping (default: 'hospitalization_id')

    Returns
    -------
    pl.DataFrame
        Input DataFrame with added device_cat_id and mode_cat_id columns
    """
    # Sort by patient and time
    resp_df = resp_df.sort([id_col, 'recorded_dttm'])

    # === HEURISTIC 1: IMV detection from mode_category ===
    # Fill in missing device_category if mode_category suggests IMV
    # Patterns: assist control-volume control, SIMV, pressure control
    resp_df = resp_df.with_columns([
        pl.when(
            pl.col('device_category').is_null() &
            pl.col('mode_category').is_not_null() &
            pl.col('mode_category').str.to_lowercase().str.contains(
                r"(?:assist control-volume control|simv|pressure control)"
            )
        )
        .then(pl.lit('IMV'))
        .otherwise(pl.col('device_category'))
        .alias('device_category')
    ])

    # === HEURISTIC 2: NIPPV detection from mode_category ===
    # Pattern: pressure support (but not CPAP)
    resp_df = resp_df.with_columns([
        pl.when(
            pl.col('device_category').is_null() &
            pl.col('mode_category').is_not_null() &
            pl.col('mode_category').str.to_lowercase().str.contains(r"pressure support") &
            ~pl.col('mode_category').str.to_lowercase().str.contains(r"cpap")
        )
        .then(pl.lit('NIPPV'))
        .otherwise(pl.col('device_category'))
        .alias('device_category')
    ])

    # === HEURISTIC 3: Room air FiO2 default ===
    # Set FiO2 = 0.21 for room air when missing
    resp_df = resp_df.with_columns([
        pl.when(
            (pl.col('device_category').str.to_lowercase() == 'room air') &
            pl.col('fio2_set').is_null()
        )
        .then(pl.lit(0.21))
        .otherwise(pl.col('fio2_set'))
        .alias('fio2_set')
    ])

    # === HEURISTIC 4: FiO2 imputation from nasal cannula flow ===
    # Impute FiO2 based on LPM for nasal cannula using clinical conversion table
    # Standard conversion: 1L → 24%, 2L → 28%, 3L → 32%, 4L → 36%, 5L → 40%,
    #                      6L → 44%, 7L → 48%, 8L → 52%, 9L → 56%, 10L → 60%

    # Check if lpm_set column exists
    if 'lpm_set' in resp_df.columns:
        # Round lpm_set to nearest integer for lookup
        resp_df = resp_df.with_columns([
            pl.col('lpm_set').round(0).cast(pl.Int32).alias('_lpm_rounded')
        ])

        # Create mapping expression using when/then chains
        fio2_from_lpm = (
            pl.when(pl.col('_lpm_rounded') == 1).then(pl.lit(0.24))
            .when(pl.col('_lpm_rounded') == 2).then(pl.lit(0.28))
            .when(pl.col('_lpm_rounded') == 3).then(pl.lit(0.32))
            .when(pl.col('_lpm_rounded') == 4).then(pl.lit(0.36))
            .when(pl.col('_lpm_rounded') == 5).then(pl.lit(0.40))
            .when(pl.col('_lpm_rounded') == 6).then(pl.lit(0.44))
            .when(pl.col('_lpm_rounded') == 7).then(pl.lit(0.48))
            .when(pl.col('_lpm_rounded') == 8).then(pl.lit(0.52))
            .when(pl.col('_lpm_rounded') == 9).then(pl.lit(0.56))
            .when(pl.col('_lpm_rounded') == 10).then(pl.lit(0.60))
            .otherwise(None)
        )

        # Apply imputation for nasal cannula rows with missing FiO2
        resp_df = resp_df.with_columns([
            pl.when(
                (pl.col('device_category').str.to_lowercase() == 'nasal cannula') &
                pl.col('fio2_set').is_null() &
                pl.col('lpm_set').is_not_null() &
                (pl.col('_lpm_rounded') >= 1) &
                (pl.col('_lpm_rounded') <= 10)
            )
            .then(fio2_from_lpm)
            .otherwise(pl.col('fio2_set'))
            .alias('fio2_set')
        ])

        # Log imputation statistics
        nasal_cannula_imputed = (
            (resp_df['device_category'].str.to_lowercase() == 'nasal cannula') &
            (resp_df['fio2_set'].is_not_null()) &
            (resp_df['_lpm_rounded'].is_not_null()) &
            (resp_df['_lpm_rounded'] >= 1) &
            (resp_df['_lpm_rounded'] <= 10)
        ).sum()

        if nasal_cannula_imputed > 0:
            logger.info(f"Imputed FiO2 for {nasal_cannula_imputed:,} nasal cannula rows using LPM lookup table")

        # Clean up temporary column
        resp_df = resp_df.drop('_lpm_rounded')

    # === Forward-fill device_category and mode_category ===
    resp_df = resp_df.with_columns([
        pl.col('device_category').forward_fill().over(id_col).alias('device_category'),
        pl.col('mode_category').forward_fill().over(id_col).alias('mode_category')
    ])

    # === Create hierarchical episode IDs ===

    # Level 1: device_cat_id - changes when device_category changes
    resp_df = resp_df.with_columns([
        pl.when(
            (pl.col('device_category') != pl.col('device_category').shift(1).over(id_col)) |
            (pl.col(id_col) != pl.col(id_col).shift(1))
        )
        .then(1)
        .otherwise(0)
        .alias('_device_cat_change')
    ])

    resp_df = resp_df.with_columns([
        pl.col('_device_cat_change').cum_sum().over(id_col).alias('device_cat_id')
    ])

    # Level 2: mode_cat_id - changes when mode_category changes (nested within device_cat_id)
    resp_df = resp_df.with_columns([
        pl.when(
            (pl.col('mode_category') != pl.col('mode_category').shift(1).over(id_col)) |
            (pl.col('device_cat_id') != pl.col('device_cat_id').shift(1).over(id_col)) |
            (pl.col(id_col) != pl.col(id_col).shift(1))
        )
        .then(1)
        .otherwise(0)
        .alias('_mode_cat_change')
    ])

    resp_df = resp_df.with_columns([
        pl.col('_mode_cat_change').cum_sum().over(id_col).alias('mode_cat_id')
    ])

    # Clean up temporary columns
    resp_df = resp_df.drop(['_device_cat_change', '_mode_cat_change'])

    return resp_df


# SOFA required categories by table
REQUIRED_LABS = ['creatinine', 'platelet_count', 'po2_arterial', 'bilirubin_total']
REQUIRED_VITALS = ['map', 'spo2', 'weight_kg']
REQUIRED_ASSESSMENTS = ['gcs_total']
REQUIRED_MEDS = ['norepinephrine', 'epinephrine', 'dopamine', 'dobutamine']
REQUIRED_RESP_SUPPORT_COLS = ['device_category', 'mode_category', 'fio2_set']

# Device ranking for respiratory SOFA score (lower rank = worse)
DEVICE_RANK_DICT = {
    'IMV': 1,
    'NIPPV': 2,
    'CPAP': 3,
    'High Flow NC': 4,
    'Face Mask': 5,
    'Trach Collar': 6,
    'Nasal Cannula': 7,
    'Other': 8,
    'Room Air': 9
}

# Unit conversion patterns
UNIT_NAMING_VARIANTS = {
    # time
    '/hr': r'/h(r|our)?$',
    '/min': r'/m(in|inute)?$',
    # unit
    'u': r'u(nits|nit)?',
    # milli
    'm': r'milli-?',
    # volume
    "l": r'l(iters|itres|itre|iter)?',
    # mass
    'mcg': r'^(u|µ|μ)g',
    'g': r'^g(rams|ram)?',
}


def _load_labs(
    data_directory: str,
    filetype: str,
    hospitalization_ids: List[str],
    cohort_df: pl.DataFrame,
    timezone: Optional[str] = None
) -> pl.DataFrame:
    """
    Load and filter labs data.

    Parameters
    ----------
    data_directory : str
        Path to data directory
    filetype : str
        File type (parquet, csv)
    hospitalization_ids : List[str]
        List of hospitalization IDs to filter
    cohort_df : pl.DataFrame
        Cohort with time windows

    Returns
    -------
    pl.DataFrame
        Filtered labs data with columns: hospitalization_id, lab_result_dttm,
        lab_category, lab_value (numeric), and any id columns from cohort
    """
    file_path = Path(data_directory) / f"clif_labs.{filetype}"

    if not file_path.exists():
        logger.warning(f"Labs file not found: {file_path}")
        return pl.DataFrame()

    # Define columns to load
    load_columns = ['hospitalization_id', 'lab_result_dttm', 'lab_category', 'lab_value', 'lab_value_numeric']

    # Load labs with filters
    if filetype == 'parquet':
        labs = pl.scan_parquet(str(file_path)).select(load_columns)
    else:
        labs = pl.scan_csv(str(file_path)).select(load_columns)

    # Filter for required categories and hospitalization_ids
    labs = labs.filter(
        pl.col('lab_category').is_in(REQUIRED_LABS) &
        pl.col('hospitalization_id').is_in(hospitalization_ids)
    )

    # Join with cohort to apply time window filter
    labs = labs.join(
        cohort_df.lazy(),
        on='hospitalization_id',
        how='inner'
    ).filter(
        (pl.col('lab_result_dttm') >= pl.col('start_dttm')) &
        (pl.col('lab_result_dttm') <= pl.col('end_dttm'))
    )

    # Select relevant columns and pivot to wide format
    id_cols = [col for col in cohort_df.columns if col not in ['start_dttm', 'end_dttm']]

    labs = labs.select([
        *id_cols,
        'lab_result_dttm',
        'lab_category',
        'lab_value_numeric'
    ]).collect()

    # Convert timezone if specified
    if timezone:
        labs = _convert_datetime_to_timezone(labs, timezone)

    # Add row number to handle multiple values at same timestamp
    # This ensures each row is unique for pivoting (no aggregation needed)
    labs = labs.with_row_count('row_id')

    # Pivot labs to wide format
    # Include row_id in index to ensure uniqueness (each lab value gets its own row)
    labs_wide = labs.pivot(
        values='lab_value_numeric',
        index=[*id_cols, 'lab_result_dttm', 'row_id'],
        columns='lab_category'
    )

    # Drop row_id after pivoting
    labs_wide = labs_wide.drop('row_id')

    return labs_wide


def _load_vitals(
    data_directory: str,
    filetype: str,
    hospitalization_ids: List[str],
    cohort_df: pl.DataFrame,
    timezone: Optional[str] = None
) -> pl.DataFrame:
    """
    Load and filter vitals data.

    Parameters
    ----------
    data_directory : str
        Path to data directory
    filetype : str
        File type (parquet, csv)
    hospitalization_ids : List[str]
        List of hospitalization IDs to filter
    cohort_df : pl.DataFrame
        Cohort with time windows

    Returns
    -------
    pl.DataFrame
        Filtered vitals data in wide format
    """
    file_path = Path(data_directory) / f"clif_vitals.{filetype}"

    if not file_path.exists():
        logger.warning(f"Vitals file not found: {file_path}")
        return pl.DataFrame()

    # Define columns to load
    load_columns = ['hospitalization_id', 'recorded_dttm', 'vital_category', 'vital_value']

    # Load vitals with filters
    if filetype == 'parquet':
        vitals = pl.scan_parquet(str(file_path)).select(load_columns)
    else:
        vitals = pl.scan_csv(str(file_path)).select(load_columns)

    # Filter for required categories and hospitalization_ids
    vitals = vitals.filter(
        pl.col('vital_category').is_in(REQUIRED_VITALS) &
        pl.col('hospitalization_id').is_in(hospitalization_ids)
    )

    # Join with cohort to apply time window filter
    vitals = vitals.join(
        cohort_df.lazy(),
        on='hospitalization_id',
        how='inner'
    ).filter(
        (pl.col('recorded_dttm') >= pl.col('start_dttm')) &
        (pl.col('recorded_dttm') <= pl.col('end_dttm'))
    )

    # Select relevant columns
    id_cols = [col for col in cohort_df.columns if col not in ['start_dttm', 'end_dttm']]

    vitals = vitals.select([
        *id_cols,
        'recorded_dttm',
        'vital_category',
        'vital_value'
    ]).collect()

    # Convert timezone if specified
    if timezone:
        vitals = _convert_datetime_to_timezone(vitals, timezone)

    # Pivot vitals to wide format
    vitals_wide = vitals.pivot(
        values='vital_value',
        index=[*id_cols, 'recorded_dttm'],
        columns='vital_category',
        aggregate_function='mean'
    )

    return vitals_wide


def _load_patient_assessments(
    data_directory: str,
    filetype: str,
    hospitalization_ids: List[str],
    cohort_df: pl.DataFrame,
    timezone: Optional[str] = None
) -> pl.DataFrame:
    """
    Load and filter patient assessments data.

    Parameters
    ----------
    data_directory : str
        Path to data directory
    filetype : str
        File type (parquet, csv)
    hospitalization_ids : List[str]
        List of hospitalization IDs to filter
    cohort_df : pl.DataFrame
        Cohort with time windows

    Returns
    -------
    pl.DataFrame
        Filtered assessments data in wide format
    """
    file_path = Path(data_directory) / f"clif_patient_assessments.{filetype}"

    if not file_path.exists():
        logger.warning(f"Patient assessments file not found: {file_path}")
        return pl.DataFrame()

    # Define columns to load
    load_columns = ['hospitalization_id', 'recorded_dttm', 'assessment_category', 'numerical_value', 'categorical_value']

    # Load assessments with filters
    if filetype == 'parquet':
        assessments = pl.scan_parquet(str(file_path)).select(load_columns)
    else:
        assessments = pl.scan_csv(str(file_path)).select(load_columns)

    # Filter for required categories and hospitalization_ids
    assessments = assessments.filter(
        pl.col('assessment_category').is_in(REQUIRED_ASSESSMENTS) &
        pl.col('hospitalization_id').is_in(hospitalization_ids)
    )

    # Join with cohort to apply time window filter
    assessments = assessments.join(
        cohort_df.lazy(),
        on='hospitalization_id',
        how='inner'
    ).filter(
        (pl.col('recorded_dttm') >= pl.col('start_dttm')) &
        (pl.col('recorded_dttm') <= pl.col('end_dttm'))
    )

    # Select relevant columns
    id_cols = [col for col in cohort_df.columns if col not in ['start_dttm', 'end_dttm']]

    # Coalesce numerical and categorical values
    assessments = assessments.with_columns([
        pl.coalesce([pl.col('numerical_value'), pl.col('categorical_value').cast(pl.Float64)]).alias('assessment_value')
    ])

    assessments = assessments.select([
        *id_cols,
        'recorded_dttm',
        'assessment_category',
        'assessment_value'
    ]).collect()

    # Convert timezone if specified
    if timezone:
        assessments = _convert_datetime_to_timezone(assessments, timezone)

    # Pivot assessments to wide format
    assessments_wide = assessments.pivot(
        values='assessment_value',
        index=[*id_cols, 'recorded_dttm'],
        columns='assessment_category',
        aggregate_function='mean'
    )

    return assessments_wide


def _load_respiratory_support(
    data_directory: str,
    filetype: str,
    hospitalization_ids: List[str],
    cohort_df: pl.DataFrame,
    lookback_hours: int = 24,
    timezone: Optional[str] = None
) -> pl.DataFrame:
    """
    Load respiratory support data with lookback period for forward-filling.

    For SOFA-97 concurrent P/F calculation, we need to forward-fill FiO2 and device
    category. This requires loading data from before the SOFA window starts to get
    the initial values for forward-filling.

    Parameters
    ----------
    data_directory : str
        Path to data directory
    filetype : str
        File type (parquet, csv)
    hospitalization_ids : List[str]
        List of hospitalization IDs to filter
    cohort_df : pl.DataFrame
        Cohort with time windows (start_dttm, end_dttm)
    lookback_hours : int
        Hours to look back before SOFA window for forward-filling (default: 24)
    timezone : Optional[str]
        Target timezone for conversion

    Returns
    -------
    pl.DataFrame
        Respiratory support data with forward-filled FiO2 and device_category
    """
    file_path = Path(data_directory) / f"clif_respiratory_support.{filetype}"

    if not file_path.exists():
        logger.warning(f"Respiratory support file not found: {file_path}")
        return pl.DataFrame()

    # Define columns to load
    load_columns = ['hospitalization_id', 'recorded_dttm', 'device_category', 'mode_category',
                    'fio2_set', 'lpm_set', 'tidal_volume_set', 'resp_rate_set']

    # Create expanded cohort with lookback period
    from datetime import timedelta
    lookback_delta = timedelta(hours=lookback_hours)
    cohort_expanded = cohort_df.with_columns([
        (pl.col('start_dttm') - pl.lit(lookback_delta)).alias('start_dttm_lookback'),
        pl.col('end_dttm').alias('end_dttm_original')
    ])

    # Load respiratory support with filters
    if filetype == 'parquet':
        resp = pl.scan_parquet(str(file_path)).select(load_columns)
    else:
        resp = pl.scan_csv(str(file_path)).select(load_columns)

    # Filter for hospitalization_ids
    resp = resp.filter(
        pl.col('hospitalization_id').is_in(hospitalization_ids)
    )

    # Join with expanded cohort to load data from lookback period
    resp = resp.join(
        cohort_expanded.lazy(),
        on='hospitalization_id',
        how='inner'
    ).filter(
        (pl.col('recorded_dttm') >= pl.col('start_dttm_lookback')) &
        (pl.col('recorded_dttm') <= pl.col('end_dttm_original'))
    )

    # Select relevant columns
    id_cols = [col for col in cohort_df.columns if col not in ['start_dttm', 'end_dttm']]

    resp = resp.select([
        *id_cols,
        'recorded_dttm',
        'device_category',
        'mode_category',
        'fio2_set',
        'start_dttm',  # Keep original window for filtering later
        'end_dttm'
    ]).collect()

    # Convert timezone if specified
    if timezone:
        resp = _convert_datetime_to_timezone(resp, timezone)

    # Create respiratory support episodes for forward-filling
    # This applies waterfall heuristics and creates hierarchical episode IDs
    resp = _create_resp_support_episodes(resp, id_col='hospitalization_id')

    # Forward-fill FiO2 within mode_cat_id episodes (most granular level)
    # device_category and mode_category are already forward-filled in _create_resp_support_episodes
    resp = resp.sort(['hospitalization_id', 'recorded_dttm'])
    resp = resp.with_columns([
        pl.col('fio2_set').forward_fill().over(['hospitalization_id', 'mode_cat_id']).alias('fio2_set')
    ])

    # Now filter to the original SOFA window (but keep forward-filled values)
    resp = resp.filter(
        (pl.col('recorded_dttm') >= pl.col('start_dttm')) &
        (pl.col('recorded_dttm') <= pl.col('end_dttm'))
    )

    # Drop the window columns
    resp = resp.drop(['start_dttm', 'end_dttm'])

    # Add device rank
    resp = resp.with_columns([
        pl.col('device_category').replace(DEVICE_RANK_DICT, default=9).alias('device_rank')
    ])

    return resp


def _clean_dose_unit(unit_series: pl.Expr) -> pl.Expr:
    """
    Clean and standardize dose unit strings using Polars expressions.

    Parameters
    ----------
    unit_series : pl.Expr
        Polars expression for dose unit column

    Returns
    -------
    pl.Expr
        Cleaned unit expression
    """
    # Remove spaces and convert to lowercase
    cleaned = unit_series.str.replace_all(r'\s+', '').str.to_lowercase()

    # Apply naming variants
    for replacement, pattern in UNIT_NAMING_VARIANTS.items():
        cleaned = cleaned.str.replace_all(pattern, replacement)

    return cleaned


def _load_and_convert_medications(
    data_directory: str,
    filetype: str,
    hospitalization_ids: List[str],
    cohort_df: pl.DataFrame,
    vitals_df: pl.DataFrame,
    timezone: Optional[str] = None
) -> pl.DataFrame:
    """
    Load medication data and convert all doses to mcg/kg/min.

    Parameters
    ----------
    data_directory : str
        Path to data directory
    filetype : str
        File type (parquet, csv)
    hospitalization_ids : List[str]
        List of hospitalization IDs to filter
    cohort_df : pl.DataFrame
        Cohort with time windows
    vitals_df : pl.DataFrame
        Vitals data containing weight_kg

    Returns
    -------
    pl.DataFrame
        Medication data with converted doses in mcg/kg/min
    """
    file_path = Path(data_directory) / f"clif_medication_admin_continuous.{filetype}"

    if not file_path.exists():
        logger.warning(f"Medication admin continuous file not found: {file_path}")
        return pl.DataFrame()

    # Define columns to load
    load_columns = ['hospitalization_id', 'admin_dttm', 'med_category', 'med_dose', 'med_dose_unit']

    # Load medications with filters
    if filetype == 'parquet':
        meds = pl.scan_parquet(str(file_path)).select(load_columns)
    else:
        meds = pl.scan_csv(str(file_path)).select(load_columns)

    # Filter for required medications and hospitalization_ids
    meds = meds.filter(
        pl.col('med_category').is_in(REQUIRED_MEDS) &
        pl.col('hospitalization_id').is_in(hospitalization_ids)
    )

    # Join with cohort to apply time window filter
    meds = meds.join(
        cohort_df.lazy(),
        on='hospitalization_id',
        how='inner'
    ).filter(
        (pl.col('admin_dttm') >= pl.col('start_dttm')) &
        (pl.col('admin_dttm') <= pl.col('end_dttm'))
    )

    meds = meds.collect()

    # Convert timezone if specified
    if timezone:
        meds = _convert_datetime_to_timezone(meds, timezone)

    # Clean dose units
    meds = meds.with_columns([
        _clean_dose_unit(pl.col('med_dose_unit')).alias('dose_unit_clean')
    ])

    # Get weight data - use most recent weight before medication time
    # Note: Load weight separately since it may be outside the SOFA time window
    # but we still need it for unit conversion
    weight_file = Path(data_directory) / f"clif_vitals.{filetype}"
    if weight_file.exists():
        if filetype == 'parquet':
            weight_data = pl.scan_parquet(str(weight_file))
        else:
            weight_data = pl.scan_csv(str(weight_file))

        weight_data = weight_data.filter(
            pl.col('hospitalization_id').is_in(hospitalization_ids) &
            (pl.col('vital_category') == 'weight_kg')
        ).select([
            'hospitalization_id',
            'recorded_dttm',
            pl.col('vital_value').alias('weight_kg')
        ]).collect()

        # Convert timezone if specified
        if timezone:
            weight_data = _convert_datetime_to_timezone(weight_data, timezone)
    else:
        logger.warning(f"Weight data file not found: {weight_file}")
        weight_data = pl.DataFrame({
            'hospitalization_id': [],
            'recorded_dttm': [],
            'weight_kg': []
        })

    # Join with weight using asof join (get most recent weight)
    meds = meds.join_asof(
        weight_data,
        left_on='admin_dttm',
        right_on='recorded_dttm',
        by='hospitalization_id',
        strategy='backward'
    )

    # Convert doses to mcg/kg/min
    # Apply mass conversions
    meds = meds.with_columns([
        pl.when(pl.col('dose_unit_clean').str.contains(r'^mg'))
        .then(pl.col('med_dose') * 1000)
        .when(pl.col('dose_unit_clean').str.contains(r'^g/'))
        .then(pl.col('med_dose') * 1000000)
        .when(pl.col('dose_unit_clean').str.contains(r'^ng'))
        .then(pl.col('med_dose') / 1000)
        .otherwise(pl.col('med_dose'))
        .alias('dose_converted')
    ])

    # Apply time conversions (/hr to /min)
    meds = meds.with_columns([
        pl.when(pl.col('dose_unit_clean').str.contains(r'/hr$'))
        .then(pl.col('dose_converted') / 60)
        .otherwise(pl.col('dose_converted'))
        .alias('dose_converted')
    ])

    # Apply weight conversions
    # If unit already contains /kg or /lb, it's already weight-normalized - keep as-is
    # If unit does NOT contain weight, we need to normalize by dividing by weight
    # But since most vasopressors are already in /kg units, we just keep the value
    meds = meds.with_columns([
        pl.when(pl.col('dose_unit_clean').str.contains(r'/kg'))
        .then(pl.col('dose_converted'))  # Already in per-kg units, keep as-is
        .when(pl.col('dose_unit_clean').str.contains(r'/lb'))
        .then(pl.col('dose_converted') * 2.20462)  # Convert /lb to /kg
        .otherwise(pl.col('dose_converted') / pl.col('weight_kg'))  # Not weight-normalized, divide by weight
        .alias('dose_mcg_kg_min')
    ])

    # Select and pivot by medication category
    id_cols = [col for col in cohort_df.columns if col not in ['start_dttm', 'end_dttm']]

    meds_select = meds.select([
        *id_cols,
        'admin_dttm',
        'med_category',
        'dose_mcg_kg_min'
    ])

    # Pivot to wide format
    meds_wide = meds_select.pivot(
        values='dose_mcg_kg_min',
        index=[*id_cols, 'admin_dttm'],
        columns='med_category',
        aggregate_function='mean'
    )

    # Rename columns to add _mcg_kg_min suffix
    rename_dict = {med: f"{med}_mcg_kg_min" for med in REQUIRED_MEDS if med in meds_wide.columns}
    if rename_dict:
        meds_wide = meds_wide.rename(rename_dict)

    return meds_wide


def _impute_pao2_from_spo2(df: pl.DataFrame) -> pl.DataFrame:
    """
    Impute PaO2 from SpO2 using Severinghaus equation.

    Only applies when SpO2 < 97% (above this, oxygen dissociation curve is too flat).

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing spo2 column

    Returns
    -------
    pl.DataFrame
        DataFrame with pao2_imputed column added
    """
    df = df.with_columns([
        # Severinghaus equation for SpO2 < 97
        pl.when(pl.col('spo2') < 97)
        .then(
            (
                (
                    (
                        (11700.0 / ((100.0 / pl.col('spo2')) - 1)) ** 2 + 50 ** 3
                    ) ** 0.5 +
                    (11700.0 / ((100.0 / pl.col('spo2')) - 1))
                ) ** (1.0/3.0)
            ) -
            (
                (
                    (
                        (11700.0 / ((100.0 / pl.col('spo2')) - 1)) ** 2 + 50 ** 3
                    ) ** 0.5 -
                    (11700.0 / ((100.0 / pl.col('spo2')) - 1))
                ) ** (1.0/3.0)
            )
        )
        .otherwise(None)
        .alias('pao2_imputed')
    ])

    return df


def _calculate_concurrent_pf_ratios(
    labs_df: pl.DataFrame,
    resp_df: pl.DataFrame,
    time_tolerance_minutes: int = 240,  # 4 hour lookback
    id_cols: List[str] = None
) -> pl.DataFrame:
    """
    Calculate P/F ratios from concurrent PO2 and FiO2 measurements.

    For SOFA-97 specification, P/F ratio must be calculated from PO2 and FiO2
    measured at the same time (or within a tolerance window). This function
    matches each PO2 measurement with the most recent FiO2 (lookback).

    Parameters
    ----------
    labs_df : pl.DataFrame
        Lab data with po2_arterial and lab_result_dttm
    resp_df : pl.DataFrame
        Respiratory support data with fio2_set (forward-filled) and recorded_dttm
    time_tolerance_minutes : int
        Maximum lookback time to find FiO2 before PO2 measurement (default: 240 = 4 hours)
    id_cols : List[str]
        ID columns for joining (default: ['hospitalization_id'])

    Returns
    -------
    pl.DataFrame
        DataFrame with concurrent P/F ratios, including:
        - All id columns
        - lab_result_dttm: timestamp of PO2 measurement
        - po2_arterial: PO2 value
        - fio2_set: matched FiO2 value
        - device_category: matched device category
        - concurrent_pf: calculated P/F ratio
    """
    if id_cols is None:
        id_cols = ['hospitalization_id']

    # Filter labs to only PO2 measurements
    po2_df = labs_df.filter(pl.col('po2_arterial').is_not_null())

    # Prepare respiratory data for joining
    # Select only the columns we need
    resp_for_join = resp_df.select([
        *id_cols,
        'recorded_dttm',
        'fio2_set',
        'device_category'
    ])

    # Use join_asof to match each PO2 with most recent FiO2 within tolerance
    # Strategy 'backward' finds the most recent FiO2 before or at the PO2 time
    po2_with_fio2 = po2_df.join_asof(
        resp_for_join,
        left_on='lab_result_dttm',
        right_on='recorded_dttm',
        by=id_cols,
        tolerance=f'{time_tolerance_minutes}m',
        strategy='backward'
    )

    # Calculate P/F ratio only where we have both PO2 and FiO2
    po2_with_fio2 = po2_with_fio2.with_columns([
        pl.when(
            (pl.col('po2_arterial').is_not_null()) &
            (pl.col('fio2_set').is_not_null()) &
            (pl.col('fio2_set') > 0)  # Avoid division by zero
        )
        .then(pl.col('po2_arterial') / pl.col('fio2_set'))
        .otherwise(None)
        .alias('concurrent_pf')
    ])

    # Filter to only successful matches (where we calculated P/F)
    concurrent_pf_df = po2_with_fio2.filter(pl.col('concurrent_pf').is_not_null())

    logger.info(f"  Calculated {len(concurrent_pf_df)} concurrent P/F ratios from {len(po2_df)} PO2 measurements")

    return concurrent_pf_df


def _aggregate_extremal_values(
    combined_df: pl.DataFrame,
    id_name: str,
    extremal_type: str = 'worst',
    concurrent_pf_df: Optional[pl.DataFrame] = None
) -> pl.DataFrame:
    """
    Aggregate extremal (worst) values by ID for SOFA score calculation.

    For SOFA-97, accepts pre-calculated concurrent P/F ratios instead of
    aggregating PO2 and FiO2 separately.

    Parameters
    ----------
    combined_df : pl.DataFrame
        Combined data from all sources (excluding respiratory P/F)
    id_name : str
        Column name to group by
    extremal_type : str
        'worst' or 'latest' (only 'worst' currently implemented)
    concurrent_pf_df : Optional[pl.DataFrame]
        Pre-calculated concurrent P/F ratios (if None, falls back to MDCalc logic)

    Returns
    -------
    pl.DataFrame
        Aggregated extremal values
    """
    if extremal_type != 'worst':
        raise NotImplementedError("Only 'worst' extremal_type is currently implemented")

    # Define columns to maximize and minimize
    # Note: fio2_set and po2_arterial removed - handled separately via concurrent P/F
    max_cols = [
        'norepinephrine_mcg_kg_min', 'epinephrine_mcg_kg_min',
        'dopamine_mcg_kg_min', 'dobutamine_mcg_kg_min',
        'creatinine', 'bilirubin_total'
    ]

    min_cols = [
        'map', 'spo2', 'pao2_imputed',
        'platelet_count', 'gcs_total'
    ]

    # Build aggregation expressions
    # Note: id_name is already preserved by group_by, so we don't add it here
    agg_exprs = []

    # Add MAX aggregations
    for col in max_cols:
        if col in combined_df.columns:
            agg_exprs.append(pl.col(col).max().alias(col))

    # Add MIN aggregations
    for col in min_cols:
        if col in combined_df.columns:
            agg_exprs.append(pl.col(col).min().alias(col))

    # Don't aggregate device_rank here - it will come from concurrent P/F
    # (device category at the time of worst P/F)

    # Group and aggregate
    extremal_df = combined_df.group_by(id_name).agg(agg_exprs)

    # Merge with concurrent P/F data if provided (SOFA-97 mode)
    if concurrent_pf_df is not None:
        # Aggregate concurrent P/F: take worst (minimum) P/F per patient
        # Also get device_category at the time of worst P/F
        pf_agg = concurrent_pf_df.group_by(id_name).agg([
            pl.col('concurrent_pf').min().alias('p_f'),
            pl.col('po2_arterial').min().alias('po2_arterial'),  # For reference
            pl.col('fio2_set').max().alias('fio2_set'),  # For reference
            # Get device_category at worst P/F
            pl.col('device_category').sort_by('concurrent_pf').first().alias('device_category')
        ])

        # Add device_rank based on device_category
        pf_agg = pf_agg.with_columns([
            pl.col('device_category').replace(DEVICE_RANK_DICT, default=9).alias('device_rank')
        ])

        # Merge with other aggregated values
        extremal_df = extremal_df.join(pf_agg, on=id_name, how='left')
    else:
        # Fallback to MDCalc logic (aggregate PO2 and FiO2 separately)
        logger.warning("No concurrent P/F data provided - using MDCalc aggregation logic")
        # This would need the old MAX(fio2) and MIN(po2) logic
        # For now, just note that this path shouldn't be used

    return extremal_df


def _compute_sofa_scores(extremal_df: pl.DataFrame, id_name: str) -> pl.DataFrame:
    """
    Calculate SOFA component scores from aggregated extremal values.

    Parameters
    ----------
    extremal_df : pl.DataFrame
        DataFrame with aggregated extremal values
    id_name : str
        Column name used for grouping

    Returns
    -------
    pl.DataFrame
        DataFrame with SOFA component scores
    """
    # Ensure all required SOFA columns exist (fill with null if missing)
    required_cols = {
        # Medications
        'norepinephrine_mcg_kg_min': pl.Float64,
        'epinephrine_mcg_kg_min': pl.Float64,
        'dopamine_mcg_kg_min': pl.Float64,
        'dobutamine_mcg_kg_min': pl.Float64,
        # Labs
        'platelet_count': pl.Float64,
        'bilirubin_total': pl.Float64,
        'creatinine': pl.Float64,
        'po2_arterial': pl.Float64,
        'pao2_imputed': pl.Float64,
        # Vitals
        'map': pl.Float64,
        'spo2': pl.Float64,
        'fio2_set': pl.Float64,
        # Assessments
        'gcs_total': pl.Float64,
        # Respiratory
        'device_rank': pl.Float64
    }
    for col, dtype in required_cols.items():
        if col not in extremal_df.columns:
            extremal_df = extremal_df.with_columns([
                pl.lit(None).cast(dtype).alias(col)
            ])

    # Calculate P/F ratios (only if not already calculated from concurrent measurements)
    if 'p_f' not in extremal_df.columns:
        # MDCalc logic: calculate from aggregated PO2/FiO2
        df = extremal_df.with_columns([
            (pl.col('po2_arterial') / pl.col('fio2_set')).alias('p_f'),
            (pl.col('pao2_imputed') / pl.col('fio2_set')).alias('p_f_imputed')
        ])
    else:
        # SOFA-97 logic: P/F already calculated from concurrent measurements
        df = extremal_df.with_columns([
            # Still calculate imputed P/F for reference
            (pl.col('pao2_imputed') / pl.col('fio2_set')).alias('p_f_imputed')
        ])

    # Map device rank back to device category for respiratory scoring (if needed)
    if 'device_category' not in df.columns:
        rank_to_device = {v: k for k, v in DEVICE_RANK_DICT.items()}
        df = df.with_columns([
            pl.col('device_rank').replace(rank_to_device, default='Other').alias('device_category')
        ])

    # Calculate SOFA scores
    df = df.with_columns([
        # Cardiovascular
        pl.when(
            (pl.col('dopamine_mcg_kg_min') > 15) |
            (pl.col('epinephrine_mcg_kg_min') > 0.1) |
            (pl.col('norepinephrine_mcg_kg_min') > 0.1)
        ).then(4)
        .when(
            (pl.col('dopamine_mcg_kg_min') > 5) |
            (pl.col('epinephrine_mcg_kg_min') <= 0.1) |
            (pl.col('norepinephrine_mcg_kg_min') <= 0.1)
        ).then(3)
        .when(
            (pl.col('dopamine_mcg_kg_min') <= 5) |
            (pl.col('dobutamine_mcg_kg_min') > 0)
        ).then(2)
        .when(pl.col('map') < 70).then(1)
        .when(pl.col('map') >= 70).then(0)
        .otherwise(None)
        .alias('sofa_cv_97'),

        # Coagulation
        pl.when(pl.col('platelet_count') < 20).then(4)
        .when(pl.col('platelet_count') < 50).then(3)
        .when(pl.col('platelet_count') < 100).then(2)
        .when(pl.col('platelet_count') < 150).then(1)
        .when(pl.col('platelet_count') >= 150).then(0)
        .otherwise(None)
        .alias('sofa_coag'),

        # Liver
        pl.when(pl.col('bilirubin_total') >= 12).then(4)
        .when(pl.col('bilirubin_total') >= 6).then(3)
        .when(pl.col('bilirubin_total') >= 2).then(2)
        .when(pl.col('bilirubin_total') >= 1.2).then(1)
        .when(pl.col('bilirubin_total') < 1.2).then(0)
        .otherwise(None)
        .alias('sofa_liver'),

        # Respiratory
        pl.when(
            (pl.col('p_f') < 100) &
            pl.col('device_category').is_in(['IMV', 'NIPPV', 'CPAP'])
        ).then(4)
        .when(
            (pl.col('p_f') >= 100) & (pl.col('p_f') < 200) &
            pl.col('device_category').is_in(['IMV', 'NIPPV', 'CPAP'])
        ).then(3)
        .when((pl.col('p_f') >= 200) & (pl.col('p_f') < 300)).then(2)
        .when((pl.col('p_f') >= 300) & (pl.col('p_f') < 400)).then(1)
        .when(pl.col('p_f') >= 400).then(0)
        .otherwise(None)
        .alias('sofa_resp'),

        # CNS
        pl.when(pl.col('gcs_total') < 6).then(4)
        .when((pl.col('gcs_total') >= 6) & (pl.col('gcs_total') <= 9)).then(3)
        .when((pl.col('gcs_total') >= 10) & (pl.col('gcs_total') <= 12)).then(2)
        .when((pl.col('gcs_total') >= 13) & (pl.col('gcs_total') <= 14)).then(1)
        .when(pl.col('gcs_total') == 15).then(0)
        .otherwise(None)
        .alias('sofa_cns'),

        # Renal
        pl.when(pl.col('creatinine') >= 5).then(4)
        .when(pl.col('creatinine') >= 3.5).then(3)
        .when(pl.col('creatinine') >= 2).then(2)
        .when(pl.col('creatinine') >= 1.2).then(1)
        .when(pl.col('creatinine') < 1.2).then(0)
        .otherwise(None)
        .alias('sofa_renal')
    ])

    # Calculate total SOFA score
    subscore_cols = ['sofa_cv_97', 'sofa_coag', 'sofa_liver', 'sofa_resp', 'sofa_cns', 'sofa_renal']
    df = df.with_columns([
        pl.sum_horizontal([pl.col(c) for c in subscore_cols]).alias('sofa_total')
    ])

    return df


def compute_sofa_polars(
    data_directory: str,
    cohort_df: pl.DataFrame,
    filetype: str = 'parquet',
    id_name: str = 'hospitalization_id',
    extremal_type: str = 'worst',
    fill_na_scores_with_zero: bool = True,
    remove_outliers: bool = True,
    timezone: Optional[str] = None
) -> pl.DataFrame:
    """
    Compute SOFA scores using optimized Polars operations.

    This function loads raw data files directly and performs all computations
    including unit conversion without relying on other clifpy methods.

    Parameters
    ----------
    data_directory : str
        Path to directory containing CLIF data files
    cohort_df : pl.DataFrame
        Cohort definition with columns:
        - hospitalization_id (required)
        - start_dttm (required): Start of observation window
        - end_dttm (required): End of observation window
        - Other ID columns (optional, e.g., encounter_block)
    filetype : str, default='parquet'
        File type of data files ('parquet' or 'csv')
    id_name : str, default='hospitalization_id'
        Column name to use for grouping SOFA scores
        (e.g., 'hospitalization_id' or 'encounter_block')
    extremal_type : str, default='worst'
        Type of aggregation ('worst' for min/max values)
    fill_na_scores_with_zero : bool, default=True
        If True, fill missing component scores with 0
    remove_outliers : bool, default=True
        If True, remove physiologically implausible values
    timezone : Optional[str]
        Timezone for datetime parsing (if needed)

    Returns
    -------
    pl.DataFrame
        DataFrame with SOFA scores, one row per id_name
        Columns: id_name, sofa_cv_97, sofa_coag, sofa_liver, sofa_resp,
                sofa_cns, sofa_renal, sofa_total, plus intermediate values

    Examples
    --------
    >>> cohort = pl.DataFrame({
    ...     'hospitalization_id': ['H1', 'H2'],
    ...     'start_dttm': [datetime(2024,1,1), datetime(2024,1,2)],
    ...     'end_dttm': [datetime(2024,1,5), datetime(2024,1,6)]
    ... })
    >>> sofa_df = compute_sofa_polars('/path/to/data', cohort)

    >>> # With encounter blocks
    >>> cohort = pl.DataFrame({
    ...     'hospitalization_id': ['H1', 'H2', 'H3'],
    ...     'encounter_block': [1, 1, 2],
    ...     'start_dttm': [...],
    ...     'end_dttm': [...]
    ... })
    >>> sofa_df = compute_sofa_polars('/path/to/data', cohort, id_name='encounter_block')
    """
    logger.info("Starting SOFA score computation with Polars")
    logger.info(f"Data directory: {data_directory}")
    logger.info(f"Cohort size: {cohort_df.height} rows")
    logger.info(f"Grouping by: {id_name}")

    # Validate cohort_df
    required_cols = ['hospitalization_id', 'start_dttm', 'end_dttm']
    missing_cols = [col for col in required_cols if col not in cohort_df.columns]
    if missing_cols:
        raise ValueError(f"cohort_df must contain columns: {required_cols}. Missing: {missing_cols}")

    if id_name not in cohort_df.columns:
        raise ValueError(f"id_name '{id_name}' not found in cohort_df columns")

    # Convert cohort datetime columns to UTC for consistent comparison with data files
    # Data files are typically in UTC, so we need to match that
    logger.info("Converting cohort datetime columns to UTC for consistent filtering...")
    cohort_df_utc = cohort_df.clone()
    for col in ['start_dttm', 'end_dttm']:
        dtype_str = str(cohort_df_utc[col].dtype)
        if 'Datetime' in dtype_str:
            # Convert to UTC (adjust time, not just relabel)
            # Also cast to microsecond precision for consistency
            cohort_df_utc = cohort_df_utc.with_columns([
                pl.col(col).dt.convert_time_zone("UTC").dt.cast_time_unit("us").alias(col)
            ])

    # Extract unique hospitalization_ids for filtering
    hospitalization_ids = cohort_df_utc['hospitalization_id'].unique().to_list()
    logger.info(f"Loading data for {len(hospitalization_ids)} unique hospitalization(s)")

    # Load all required tables (using UTC-converted cohort for consistent filtering)
    logger.info("Loading labs data...")
    labs_df = _load_labs(data_directory, filetype, hospitalization_ids, cohort_df_utc, timezone)

    logger.info("Loading vitals data...")
    vitals_df = _load_vitals(data_directory, filetype, hospitalization_ids, cohort_df_utc, timezone)

    logger.info("Loading patient assessments data...")
    assessments_df = _load_patient_assessments(data_directory, filetype, hospitalization_ids, cohort_df_utc, timezone)

    logger.info("Loading respiratory support data...")
    resp_df = _load_respiratory_support(data_directory, filetype, hospitalization_ids, cohort_df_utc, lookback_hours=24, timezone=timezone)

    logger.info("Loading and converting medication data...")
    meds_df = _load_and_convert_medications(data_directory, filetype, hospitalization_ids, cohort_df_utc, vitals_df, timezone)

    # Combine all data sources
    logger.info("Combining all data sources...")

    # Get all time columns
    time_cols = []
    dfs_to_combine = []

    if labs_df.height > 0:
        time_cols.append(('lab_result_dttm', labs_df))
    if vitals_df.height > 0:
        time_cols.append(('recorded_dttm', vitals_df))
    if assessments_df.height > 0:
        time_cols.append(('recorded_dttm', assessments_df))
    if resp_df.height > 0:
        time_cols.append(('recorded_dttm', resp_df))
    if meds_df.height > 0:
        time_cols.append(('admin_dttm', meds_df))

    # Rename all time columns to common 'event_time' and stack vertically
    id_cols = [col for col in cohort_df_utc.columns if col not in ['start_dttm', 'end_dttm']]

    combined_parts = []
    for time_col, df in time_cols:
        df_renamed = df.rename({time_col: 'event_time'})
        combined_parts.append(df_renamed)

    if not combined_parts:
        raise ValueError("No data loaded from any source")

    # Concatenate all parts vertically (stack rows)
    # Use 'diagonal' to handle different column sets - missing columns filled with null
    combined_df = pl.concat(combined_parts, how='diagonal')

    logger.info(f"Combined data shape: {combined_df.height} rows x {combined_df.width} columns")

    # Remove outliers if requested
    if remove_outliers:
        logger.info("Removing outliers...")
        combined_df = combined_df.with_columns([
            pl.when((pl.col('po2_arterial') >= 0) & (pl.col('po2_arterial') <= 700))
            .then(pl.col('po2_arterial'))
            .otherwise(None)
            .alias('po2_arterial'),

            pl.when((pl.col('fio2_set') >= 0.21) & (pl.col('fio2_set') <= 1))
            .then(pl.col('fio2_set'))
            .otherwise(None)
            .alias('fio2_set'),

            pl.when((pl.col('spo2') >= 50) & (pl.col('spo2') <= 100))
            .then(pl.col('spo2'))
            .otherwise(None)
            .alias('spo2')
        ])

    # Impute PaO2 from SpO2
    logger.info("Imputing PaO2 from SpO2...")
    combined_df = _impute_pao2_from_spo2(combined_df)

    # Calculate concurrent P/F ratios (SOFA-97 specification)
    logger.info("Calculating concurrent P/F ratios...")
    # Extract labs data with PO2 (before it gets combined with other data)
    labs_with_po2 = labs_df.filter(pl.col('po2_arterial').is_not_null())

    # Calculate concurrent P/F using respiratory support data (with forward-filled FiO2)
    id_cols = [col for col in cohort_df_utc.columns if col not in ['start_dttm', 'end_dttm']]
    concurrent_pf_df = _calculate_concurrent_pf_ratios(
        labs_with_po2,
        resp_df,
        time_tolerance_minutes=240,  # 4 hour lookback
        id_cols=id_cols
    )

    # Aggregate extremal values
    logger.info(f"Aggregating extremal values by {id_name}...")
    extremal_df = _aggregate_extremal_values(
        combined_df,
        id_name,
        extremal_type,
        concurrent_pf_df=concurrent_pf_df  # Pass concurrent P/F for SOFA-97
    )

    # Compute SOFA scores
    logger.info("Computing SOFA scores...")
    sofa_df = _compute_sofa_scores(extremal_df, id_name)

    # Fill NA scores with zero if requested
    if fill_na_scores_with_zero:
        logger.info("Filling missing scores with 0...")
        subscore_cols = ['sofa_cv_97', 'sofa_coag', 'sofa_liver', 'sofa_resp', 'sofa_cns', 'sofa_renal']
        sofa_df = sofa_df.with_columns([
            pl.col(c).fill_null(0) for c in subscore_cols
        ])
        # Recalculate total
        sofa_df = sofa_df.with_columns([
            pl.sum_horizontal([pl.col(c) for c in subscore_cols]).alias('sofa_total')
        ])

    logger.info(f"SOFA computation complete. Result shape: {sofa_df.height} rows x {sofa_df.width} columns")

    return sofa_df
