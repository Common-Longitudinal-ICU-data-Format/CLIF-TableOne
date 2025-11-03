"""Medication dose unit converter using Polars for memory efficiency.

This module provides utilities for converting medication dose units between
different formats and standardizing them to a common base set. It's a Polars-based
reimplementation of CLIFpy's unit converter, optimized for memory efficiency.

Handles:
- Weight-based dosing (/kg, /lb conversions)
- Time unit conversions (/hr → /min)
- Mass conversions (mg, ng, g → mcg)
- Volume conversions (L → mL)
- Unit conversions (milli-units → units)
"""

import polars as pl
from typing import Dict, Tuple, Set, Optional, List
import re
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS - Same as CLIFpy
# ============================================================================

UNIT_NAMING_VARIANTS = {
    # time
    '/hr': r'/h(r|our)?$',
    '/min': r'/m(in|inute)?$',
    # unit -- NOTE: plural always go first to avoid results like "us" or "gs"
    'u': r'u(nits|nit)?',
    # milli
    'm': r'milli-?',
    # volume
    "l": r'l(iters|itres|itre|iter)?',
    # mass
    'mcg': r'^(u|µ|μ)g',
    'g': r'^g(rams|ram)?',
}

AMOUNT_ENDER = r"($|/*)"
MASS_REGEX = rf"^(mcg|mg|ng|g){AMOUNT_ENDER}"
VOLUME_REGEX = rf"^(l|ml){AMOUNT_ENDER}"
UNIT_REGEX = rf"^(u|mu){AMOUNT_ENDER}"

# time
HR_REGEX = r"/hr$"

# mass
MU_REGEX = rf"^(mu){AMOUNT_ENDER}"
MG_REGEX = rf"^(mg){AMOUNT_ENDER}"
NG_REGEX = rf"^(ng){AMOUNT_ENDER}"
G_REGEX = rf"^(g){AMOUNT_ENDER}"

# volume
L_REGEX = rf"^l{AMOUNT_ENDER}"

# weight
LB_REGEX = r"/lb/"
KG_REGEX = r"/kg/"
WEIGHT_REGEX = r"/(lb|kg)/"

# Conversion factors: factor_name → formula (can be numeric or expression)
REGEX_TO_FACTOR_MAPPER = {
    # time -> /min
    HR_REGEX: 1/60,

    # volume -> ml
    L_REGEX: 1000,  # to ml

    # unit -> u
    MU_REGEX: 1/1000,

    # mass -> mcg
    MG_REGEX: 1000,
    NG_REGEX: 1/1000,
    G_REGEX: 1000000,

    # weight -> /kg (these are special - use weight_kg column)
    KG_REGEX: 'weight_kg',
    LB_REGEX: 'weight_kg * 2.20462'
}

ACCEPTABLE_AMOUNT_UNITS = {
    "ml", "l",  # volume
    "mu", "u",  # unit
    "mcg", "mg", "ng", 'g'  # mass
}

def _acceptable_rate_units() -> Set[str]:
    """Generate all acceptable rate unit combinations."""
    acceptable_weight_units = {'/kg', '/lb', ''}
    acceptable_time_units = {'/hr', '/min'}
    return {a + b + c for a in ACCEPTABLE_AMOUNT_UNITS
            for b in acceptable_weight_units
            for c in acceptable_time_units}

ACCEPTABLE_RATE_UNITS = _acceptable_rate_units()
ALL_ACCEPTABLE_UNITS = ACCEPTABLE_RATE_UNITS | ACCEPTABLE_AMOUNT_UNITS

# ============================================================================
# UNIT CLEANING FUNCTIONS
# ============================================================================

def _clean_dose_unit_formats(df: pl.LazyFrame, unit_col: str = 'med_dose_unit') -> pl.LazyFrame:
    """Clean dose unit formatting (spaces, lowercase).

    Parameters
    ----------
    df : pl.LazyFrame
        Input dataframe
    unit_col : str
        Name of unit column to clean

    Returns
    -------
    pl.LazyFrame
        Dataframe with additional '_clean_unit' column
    """
    return df.with_columns([
        pl.col(unit_col)
            .str.replace_all(r'\s+', '')
            .str.to_lowercase()
            .alias('_clean_unit')
    ])

def _clean_dose_unit_names(df: pl.LazyFrame) -> pl.LazyFrame:
    """Clean dose unit name variants (e.g., 'milliliter' → 'ml').

    Parameters
    ----------
    df : pl.LazyFrame
        Input dataframe (should have '_clean_unit' column from _clean_dose_unit_formats)

    Returns
    -------
    pl.LazyFrame
        Dataframe with cleaned '_clean_unit' column
    """
    # Apply naming variants transformations sequentially
    result = df
    for standard, pattern in UNIT_NAMING_VARIANTS.items():
        result = result.with_columns([
            pl.col('_clean_unit')
                .str.replace_all(pattern, standard)
                .alias('_clean_unit')
        ])
    return result

# ============================================================================
# BASE UNIT CONVERSION
# ============================================================================

def _convert_to_base_units(
    df: pl.LazyFrame,
    dose_col: str = 'med_dose',
    unit_col: str = '_clean_unit'
) -> pl.LazyFrame:
    """Convert dose units to base standardized units.

    Converts to: mcg/min (mass rates), ml/min (volume rates), u/min (unit rates)
    or: mcg (mass amounts), ml (volume amounts), u (unit amounts)

    Parameters
    ----------
    df : pl.LazyFrame
        Input dataframe with cleaned units and dose values
    dose_col : str
        Name of dose column
    unit_col : str
        Name of unit column (should be pre-cleaned)

    Returns
    -------
    pl.LazyFrame
        Dataframe with base unit conversion columns
    """
    # Build the conversion expressions
    amount_multiplier = (
        pl.when(pl.col(unit_col).str.contains(L_REGEX)).then(pl.lit(1000))  # L → ml
        .when(pl.col(unit_col).str.contains(MU_REGEX)).then(pl.lit(1/1000))  # mu → u
        .when(pl.col(unit_col).str.contains(MG_REGEX)).then(pl.lit(1000))  # mg → mcg
        .when(pl.col(unit_col).str.contains(NG_REGEX)).then(pl.lit(1/1000))  # ng → mcg
        .when(pl.col(unit_col).str.contains(G_REGEX)).then(pl.lit(1000000))  # g → mcg
        .otherwise(pl.lit(1.0))
    )

    time_multiplier = (
        pl.when(pl.col(unit_col).str.contains(HR_REGEX)).then(pl.lit(1/60))  # /hr → /min
        .otherwise(pl.lit(1.0))
    )

    # Weight multiplier (kg or lb)
    weight_multiplier = (
        pl.when(pl.col(unit_col).str.contains(KG_REGEX))
            .then(pl.col('weight_kg'))
        .when(pl.col(unit_col).str.contains(LB_REGEX))
            .then(pl.col('weight_kg') * 2.20462)
        .otherwise(pl.lit(1.0))
    )

    # Unit class classification
    unit_class = (
        pl.when(pl.col(unit_col).is_in(list(ACCEPTABLE_RATE_UNITS))).then(pl.lit('rate'))
        .when(pl.col(unit_col).is_in(list(ACCEPTABLE_AMOUNT_UNITS))).then(pl.lit('amount'))
        .otherwise(pl.lit('unrecognized'))
    )

    # Is this unit weighted (has /kg or /lb)?
    is_weighted = pl.col(unit_col).str.contains(WEIGHT_REGEX).cast(pl.Int32)

    # Base unit determination
    base_unit = (
        pl.when(is_weighted & pl.col('weight_kg').is_null()).then(pl.col(unit_col))  # Can't convert if missing weight
        .when(unit_class == 'unrecognized').then(pl.col(unit_col))
        .when((unit_class == 'rate') & pl.col(unit_col).str.contains(MASS_REGEX)).then(pl.lit('mcg/min'))
        .when((unit_class == 'rate') & pl.col(unit_col).str.contains(VOLUME_REGEX)).then(pl.lit('ml/min'))
        .when((unit_class == 'rate') & pl.col(unit_col).str.contains(UNIT_REGEX)).then(pl.lit('u/min'))
        .when((unit_class == 'amount') & pl.col(unit_col).str.contains(MASS_REGEX)).then(pl.lit('mcg'))
        .when((unit_class == 'amount') & pl.col(unit_col).str.contains(VOLUME_REGEX)).then(pl.lit('ml'))
        .when((unit_class == 'amount') & pl.col(unit_col).str.contains(UNIT_REGEX)).then(pl.lit('u'))
        .otherwise(pl.col(unit_col))
    )

    # Base dose calculation
    base_dose = (
        pl.when(is_weighted & pl.col('weight_kg').is_null())
            .then(pl.col(dose_col))
        .otherwise(
            pl.col(dose_col) * amount_multiplier * time_multiplier * weight_multiplier
        )
    )

    return df.with_columns([
        unit_class.alias('_unit_class'),
        is_weighted.alias('_weighted'),
        amount_multiplier.alias('_amount_multiplier'),
        time_multiplier.alias('_time_multiplier'),
        weight_multiplier.alias('_weight_multiplier'),
        base_dose.alias('_base_dose'),
        base_unit.alias('_base_unit'),
    ])

# ============================================================================
# PREFERRED UNIT CONVERSION
# ============================================================================

def _convert_to_preferred_units(
    df: pl.LazyFrame,
    override: bool = False
) -> pl.LazyFrame:
    """Convert from base units to medication-specific preferred units.

    Parameters
    ----------
    df : pl.LazyFrame
        Dataframe with base unit conversions (from _convert_to_base_units)
        Must have: _base_dose, _base_unit, _preferred_unit, weight_kg columns
    override : bool
        If False, error on unacceptable preferred units. If True, warn and continue.

    Returns
    -------
    pl.LazyFrame
        Dataframe with preferred unit conversion columns
    """
    # Determine subclass of base unit (mass, volume, unit)
    base_unit_subclass = (
        pl.when(pl.col('_base_unit').str.contains(MASS_REGEX)).then(pl.lit('mass'))
        .when(pl.col('_base_unit').str.contains(VOLUME_REGEX)).then(pl.lit('volume'))
        .when(pl.col('_base_unit').str.contains(UNIT_REGEX)).then(pl.lit('unit'))
        .otherwise(pl.lit('unrecognized'))
    )

    # Determine class and subclass of preferred unit
    preferred_unit_class = (
        pl.when(pl.col('_preferred_unit').is_in(list(ACCEPTABLE_RATE_UNITS))).then(pl.lit('rate'))
        .when(pl.col('_preferred_unit').is_in(list(ACCEPTABLE_AMOUNT_UNITS))).then(pl.lit('amount'))
        .otherwise(pl.lit('unrecognized'))
    )

    preferred_unit_subclass = (
        pl.when(pl.col('_preferred_unit').str.contains(MASS_REGEX)).then(pl.lit('mass'))
        .when(pl.col('_preferred_unit').str.contains(VOLUME_REGEX)).then(pl.lit('volume'))
        .when(pl.col('_preferred_unit').str.contains(UNIT_REGEX)).then(pl.lit('unit'))
        .otherwise(pl.lit('unrecognized'))
    )

    # Is preferred unit weighted?
    is_preferred_weighted = pl.col('_preferred_unit').str.contains(WEIGHT_REGEX).cast(pl.Int32)

    # Conversion multipliers for preferred units
    amount_multiplier_pref = (
        pl.when(pl.col('_preferred_unit').str.contains(L_REGEX)).then(pl.lit(1/1000))  # ml → L
        .when(pl.col('_preferred_unit').str.contains(MU_REGEX)).then(pl.lit(1000))  # u → mu
        .when(pl.col('_preferred_unit').str.contains(MG_REGEX)).then(pl.lit(1/1000))  # mcg → mg
        .when(pl.col('_preferred_unit').str.contains(NG_REGEX)).then(pl.lit(1000))  # mcg → ng
        .when(pl.col('_preferred_unit').str.contains(G_REGEX)).then(pl.lit(1/1000000))  # mcg → g
        .otherwise(pl.lit(1.0))
    )

    time_multiplier_pref = (
        pl.when(pl.col('_preferred_unit').str.contains(HR_REGEX)).then(pl.lit(60))  # /min → /hr
        .otherwise(pl.lit(1.0))
    )

    weight_multiplier_pref = (
        pl.when(pl.col('_preferred_unit').str.contains(KG_REGEX))
            .then(pl.lit(1.0) / pl.col('weight_kg'))
        .when(pl.col('_preferred_unit').str.contains(LB_REGEX))
            .then(pl.lit(1.0) / (pl.col('weight_kg') * 2.20462))
        .otherwise(pl.lit(1.0))
    )

    # Conversion status - determine if conversion is valid
    convert_status = (
        pl.when(is_preferred_weighted & pl.col('weight_kg').is_null())
            .then(pl.lit('cannot convert to a weighted unit if weight_kg is missing'))
        .when(pl.col('_base_unit').is_null())
            .then(pl.lit('original unit is missing'))
        .when(pl.col('_unit_class') == 'unrecognized')
            .then(pl.lit('original unit ' + pl.col('_base_unit') + ' is not recognized'))
        .when(preferred_unit_class == 'unrecognized')
            .then(pl.lit('user-preferred unit ' + pl.col('_preferred_unit') + ' is not recognized'))
        .when(pl.col('_unit_class') != preferred_unit_class)
            .then(pl.lit('cannot convert ' + pl.col('_unit_class') + ' to ' + preferred_unit_class))
        .when(base_unit_subclass != preferred_unit_subclass)
            .then(pl.lit('cannot convert ' + base_unit_subclass + ' to ' + preferred_unit_subclass))
        .otherwise(pl.lit('success'))
    )

    # Final converted dose and unit
    dose_converted = (
        pl.when(convert_status == 'success')
            .then(pl.col('_base_dose') * amount_multiplier_pref * time_multiplier_pref * weight_multiplier_pref)
        .otherwise(pl.col('_base_dose'))
    )

    unit_converted = (
        pl.when(convert_status == 'success')
            .then(pl.col('_preferred_unit'))
        .otherwise(pl.col('_base_unit'))
    )

    return df.with_columns([
        base_unit_subclass.alias('_unit_subclass'),
        preferred_unit_class.alias('_unit_class_preferred'),
        preferred_unit_subclass.alias('_unit_subclass_preferred'),
        is_preferred_weighted.alias('_weighted_preferred'),
        convert_status.alias('_convert_status'),
        amount_multiplier_pref.alias('_amount_multiplier_preferred'),
        time_multiplier_pref.alias('_time_multiplier_preferred'),
        weight_multiplier_pref.alias('_weight_multiplier_preferred'),
        dose_converted.alias('med_dose_converted'),
        unit_converted.alias('med_dose_unit_converted'),
    ])

# ============================================================================
# WEIGHT JOINING (optional weight extraction from vitals)
# ============================================================================

def _join_weights_from_vitals(
    med_df: pl.LazyFrame,
    vitals_df: pl.LazyFrame
) -> pl.LazyFrame:
    """Join most recent weight measurements to medication dataframe.

    Uses join_asof to match each medication to the most recent prior weight measurement.

    Parameters
    ----------
    med_df : pl.LazyFrame
        Medication dataframe (must have hospitalization_id, admin_dttm columns)
    vitals_df : pl.LazyFrame
        Vitals dataframe (must have hospitalization_id, vital_category, vital_value, recorded_dttm)

    Returns
    -------
    pl.LazyFrame
        Medication dataframe with weight_kg column added
    """
    # Filter to weight measurements only
    weights = vitals_df.filter(
        (pl.col('vital_category') == 'weight_kg') &
        (pl.col('vital_value').is_not_null())
    ).select([
        'hospitalization_id',
        pl.col('recorded_dttm').alias('weight_recorded_dttm'),
        pl.col('vital_value').alias('weight_kg')
    ]).sort(['hospitalization_id', 'weight_recorded_dttm'])

    # Join using join_asof to get most recent prior weight
    return med_df.join_asof(
        weights,
        by='hospitalization_id',
        left_on='admin_dttm',
        right_on='weight_recorded_dttm',
        strategy='backward'
    )

# ============================================================================
# CONVERSION SUMMARY/REPORTING
# ============================================================================

def _create_conversion_summary(
    df: pl.LazyFrame,
    group_by: List[str]
) -> pl.LazyFrame:
    """Create summary table of unit conversion patterns and frequencies.

    Parameters
    ----------
    df : pl.LazyFrame
        Dataframe with conversion results
    group_by : List[str]
        Columns to group by for summary

    Returns
    -------
    pl.LazyFrame
        Summary dataframe with counts
    """
    return (df
        .group_by(group_by)
        .agg(pl.count().alias('count'))
        .sort('count', descending=True)
    )

# ============================================================================
# PUBLIC API
# ============================================================================

def convert_dose_units_by_med_category(
    med_df: pl.LazyFrame,
    vitals_df: Optional[pl.LazyFrame] = None,
    preferred_units: Optional[Dict[str, str]] = None,
    show_intermediate: bool = False,
    override: bool = False
) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
    """Convert medication dose units to user-defined preferred units.

    Main public API for medication unit conversion. Performs two-stage conversion:
    1. Standardizes all units to base set (mcg/min, ml/min, u/min for rates)
    2. Converts to medication-specific preferred units if provided

    Parameters
    ----------
    med_df : pl.LazyFrame
        Medication dataframe with required columns:
        - med_dose: Dose values (numeric)
        - med_dose_unit: Dose unit strings (e.g., 'MCG/KG/HR', 'mL/hr')
        - med_category: Medication category identifier
        - weight_kg: Patient weight in kg (optional if vitals_df provided)

    vitals_df : pl.LazyFrame, optional
        Vitals dataframe for extracting weights if not in med_df.
        Required columns:
        - hospitalization_id
        - recorded_dttm
        - vital_category (must include 'weight_kg')
        - vital_value

    preferred_units : dict, optional
        Dictionary mapping med_category → preferred unit strings.
        Example: {'propofol': 'mcg/kg/min', 'insulin': 'u/hr'}
        If None, uses base units as defaults.

    show_intermediate : bool
        If False, drops intermediate calculation columns from output.
        If True, retains all columns including multipliers.

    override : bool
        If True, prints warnings for unacceptable units but continues.
        If False, raises errors.

    Returns
    -------
    Tuple[pl.LazyFrame, pl.LazyFrame]
        (converted_df, summary_df) where:
        - converted_df: Medication data with conversion results
        - summary_df: Summary counts of conversion patterns

    Raises
    ------
    ValueError
        If required columns are missing or conversions fail.

    Examples
    --------
    >>> med_df = pl.LazyFrame({
    ...     'med_category': ['propofol', 'fentanyl'],
    ...     'med_dose': [200, 2],
    ...     'med_dose_unit': ['MCG/KG/MIN', 'mcg/kg/hr'],
    ...     'weight_kg': [70, 80]
    ... })
    >>> preferred = {'propofol': 'mcg/kg/min', 'fentanyl': 'mcg/hr'}
    >>> result_df, summary = convert_dose_units_by_med_category(
    ...     med_df,
    ...     preferred_units=preferred
    ... )
    """
    # Step 1: Add weights if missing
    if vitals_df is not None and 'weight_kg' not in med_df.collect_schema().names():
        logger.info("Adding weight data from vitals...")
        med_df = _join_weights_from_vitals(med_df, vitals_df)

    # Check required columns
    required_cols = {'med_dose', 'med_dose_unit', 'weight_kg'}
    available_cols = set(med_df.collect_schema().names())
    missing_cols = required_cols - available_cols
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Step 2: Clean units
    logger.info("Cleaning dose units...")
    med_df = _clean_dose_unit_formats(med_df)
    med_df = _clean_dose_unit_names(med_df)

    # Step 3: Convert to base units
    logger.info("Converting to base units...")
    med_df = _convert_to_base_units(med_df)

    # Step 4: Add preferred units (or use base units as defaults)
    if preferred_units:
        logger.info(f"Adding preferred units for {len(preferred_units)} medications...")
        preferred_df = pl.DataFrame(
            {'med_category': list(preferred_units.keys()),
             '_preferred_unit': list(preferred_units.values())}
        ).lazy()

        # Left join to preserve all medications
        med_df = med_df.join(
            preferred_df,
            on='med_category',
            how='left'
        )
        # Use base unit if no preferred unit specified
        med_df = med_df.with_columns([
            pl.col('_preferred_unit')
                .fill_null(pl.col('_base_unit'))
                .alias('_preferred_unit')
        ])
    else:
        # Use base units as preferred units
        med_df = med_df.with_columns([
            pl.col('_base_unit').alias('_preferred_unit')
        ])

    # Step 5: Convert to preferred units
    logger.info("Converting to preferred units...")
    med_df = _convert_to_preferred_units(med_df, override=override)

    # Step 6: Create summary
    logger.info("Creating conversion summary...")
    summary_df = _create_conversion_summary(
        med_df,
        group_by=[
            'med_category',
            'med_dose_unit',
            '_clean_unit',
            '_base_unit',
            '_unit_class',
            '_preferred_unit',
            'med_dose_unit_converted',
            '_convert_status'
        ]
    )

    # Step 7: Drop intermediate columns if requested
    if not show_intermediate:
        cols_to_drop = [
            col for col in med_df.collect_schema().names()
            if any(x in col for x in [
                'multiplier', '_rn', '_weight_recorded_dttm',
                '_weighted', '_weighted_preferred', '_base_dose',
                '_unit_class_preferred', '_unit_subclass',
                '_unit_subclass_preferred'
            ])
        ]
        med_df = med_df.drop(cols_to_drop)

    logger.info("Conversion complete!")
    return med_df, summary_df
