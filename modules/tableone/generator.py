#!/usr/bin/env python3
"""
Converted from: code/archives/generate_table_one_2_1.ipynb
"""


# ==============================================================================
# ## CLIF Table One
#
# Author: Kaveri Chhikara
# Date v1: September 8, 2025
#
# This script identifies the cohort of encounters with at least one ICU stay and then summarizes the cohort data into one table. 
# ==============================================================================


# ==============================================================================
# ## Cohort Identification
#
#
# ## Inclusion 
# 1. Adults
# 2. Patients with at least one ICU stay or those who had only emergency department or ward encounters and either died or received life support at any point. Life support is defined as the administration of any vasoactive drugs or respiratory support exceeding low-flow oxygen.
#
# Respiratory support device: 'IMV', 'NIPPV', 'CPAP', 'High Flow NC'  
#
# Vasoactive: 'norepinephrine', 'epinephrine', 'phenylephrine', 'vasopressin',
#     'dopamine', 'angiotensin'
# ==============================================================================

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
import matplotlib.pyplot as plt
import gc
from pathlib import Path
import json
from typing import Union
from tqdm import tqdm
import csv
import sys
import clifpy
import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
import colorsys
import matplotlib.path
import matplotlib.patches
import matplotlib.patheffects
import polars as pl
import re
from modules.sofa.calculator import compute_sofa_polars
from modules.utils.datetime_utils import standardize_datetime_columns

from clifpy.clif_orchestrator import ClifOrchestrator
from clifpy.utils import apply_outlier_handling
from clifpy.utils.ase import compute_ase
from clifpy.utils.comorbidity import calculate_cci
from clifpy.utils.stitching_encounters import stitch_encounters
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
from scipy import stats
from statsmodels.stats.proportion import proportion_confint
from tqdm import tqdm
from typing import Union
from upsetplot import UpSet, from_indicators
from venny4py.venny4py import venny4py
import clifpy
import collections
import colorsys
import csv
import gc
import json
import matplotlib.patches
import matplotlib.path
import matplotlib.patheffects
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import polars as pl
import seaborn as sns
import sys
import warnings


def _waterfall_chunk(args):
    """Process a chunk of encounters through the waterfall (top-level for pickling)."""
    chunk_df, id_col = args
    from clifpy.utils.waterfall import process_resp_support_waterfall
    return process_resp_support_waterfall(chunk_df, id_col=id_col, verbose=False)


def crosstab_demographics(patient_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulate race × ethnicity × sex for NIH enrollment reporting.

    Parameters:
        patient_df: One row per unique patient with columns
                    race_category, ethnicity_category, sex_category.

    Returns:
        DataFrame with race rows, (ethnicity, sex) MultiIndex columns,
        and margin totals.
    """
    # CLIF 2.1 patient schema permissible values
    race_values = [
        'American Indian or Alaska Native',
        'Asian',
        'Native Hawaiian or Other Pacific Islander',
        'Black or African American',
        'White',
        'Other',
        'Unknown',
    ]
    ethnicity_values = ['Non-Hispanic', 'Hispanic', 'Unknown']
    sex_values = ['Female', 'Male', 'Unknown']

    df = patient_df[['race_category', 'ethnicity_category', 'sex_category']].copy()
    df.fillna('Unknown', inplace=True)

    ct = pd.crosstab(
        df['race_category'],
        [df['ethnicity_category'], df['sex_category']],
        margins=True,
        margins_name='Total',
    )

    # Build expected column ordering
    col_tuples = [(eth, sex) for eth in ethnicity_values for sex in sex_values]
    col_tuples.append(('Total', ''))
    desired_columns = pd.MultiIndex.from_tuples(col_tuples)

    row_order = race_values + ['Total']

    ct = ct.reindex(index=row_order, columns=desired_columns, fill_value=0)
    ct.index.name = 'Racial Categories'

    return ct


# ============================================================================
# Extracted summary functions for stratified output generation
# ============================================================================

def _suffixed(basename: str, suffix: str) -> str:
    """Insert *suffix* before the file extension: 'foo.csv' + '_icu' -> 'foo_icu.csv'."""
    stem, ext = os.path.splitext(basename)
    return f'{stem}{suffix}{ext}'


def generate_ventilator_settings_summary(resp_valid, vent_settings, output_dir, suffix=''):
    """Generate ventilator settings summary CSVs."""
    os.makedirs(output_dir, exist_ok=True)
    existing_settings = [col for col in vent_settings if col in resp_valid.columns]
    if not existing_settings or len(resp_valid) == 0:
        return

    medians = resp_valid.groupby(['device_category', 'mode_category'])[existing_settings].median()
    q1 = resp_valid.groupby(['device_category', 'mode_category'])[existing_settings].quantile(0.25)
    q3 = resp_valid.groupby(['device_category', 'mode_category'])[existing_settings].quantile(0.75)

    medians_reset = medians.reset_index()
    q1_reset = q1.reset_index()
    q3_reset = q3.reset_index()

    settings_summary = medians_reset[['device_category', 'mode_category']].copy()
    for setting in existing_settings:
        if setting in medians.columns:
            settings_summary[setting] = (
                medians_reset[setting].round(1).astype(str) + ' (' +
                q1_reset[setting].round(1).astype(str) + '-' +
                q3_reset[setting].round(1).astype(str) + ')'
            )

    rename_dict = {}
    if 'mode_category' in settings_summary.columns:
        rename_dict['mode_category'] = 'ventilator_setting'
    column_mapping = {
        'fio2_set': 'FiO2 Set', 'lpm_set': 'LPM Set',
        'tidal_volume_set': 'Tidal Volume Set', 'resp_rate_set': 'Resp Rate Set',
        'pressure_control_set': 'Pressure Control Set', 'peep_set': 'PEEP Set',
        'pressure_support_set': 'Pressure Support Set', 'flow_rate_set': 'Flow Rate Set'
    }
    for old_name, new_name in column_mapping.items():
        if old_name in settings_summary.columns:
            rename_dict[old_name] = new_name
    settings_summary = settings_summary.rename(columns=rename_dict)
    sort_col = 'ventilator_setting' if 'ventilator_setting' in settings_summary.columns else 'mode_category'
    settings_summary = settings_summary.sort_values(['device_category', sort_col])
    settings_summary.to_csv(os.path.join(output_dir, _suffixed('ventilator_settings_by_device_mode.csv', suffix)), index=False)

    # Counts table
    counts_summary = resp_valid.groupby(['device_category', 'mode_category'])[existing_settings].count().reset_index()
    counts_rename = {}
    if 'mode_category' in counts_summary.columns:
        counts_rename['mode_category'] = 'ventilator_setting'
    for old_name, new_name in column_mapping.items():
        col_n = old_name
        if col_n in counts_summary.columns:
            counts_rename[col_n] = new_name + ' (N)'
    counts_summary = counts_summary.rename(columns=counts_rename)
    sort_col = 'ventilator_setting' if 'ventilator_setting' in counts_summary.columns else 'mode_category'
    counts_summary = counts_summary.sort_values(['device_category', sort_col])
    counts_summary.to_csv(os.path.join(output_dir, _suffixed('ventilator_settings_counts_by_device_mode.csv', suffix)), index=False)

    # Total observations
    total_obs_df = pd.DataFrame({
        'metric': ['total_respiratory_support_observations'],
        'value': [len(resp_valid)]
    })
    total_obs_df.to_csv(os.path.join(output_dir, _suffixed('ventilator_settings_total_observations.csv', suffix)), index=False)


def generate_tidal_volume_stats(resp_imv_post_start, output_dir, suffix=''):
    """Generate tidal volume stats for volume control modes."""
    os.makedirs(output_dir, exist_ok=True)
    volume_control_modes = ['assist control-volume control', 'pressure-regulated volume control']
    volume_mode_data = resp_imv_post_start[
        resp_imv_post_start['mode_category'].isin(volume_control_modes)
    ].copy()
    if len(volume_mode_data) == 0 or 'tidal_volume_set' not in volume_mode_data.columns:
        return

    volume_mode_data['hour_bin'] = volume_mode_data['hours_from_vent_start'].round(0).astype(int)
    volume_mode_data_7d = volume_mode_data[volume_mode_data['hour_bin'] <= 168].copy()

    tv_stats = volume_mode_data_7d.groupby('hour_bin')['tidal_volume_set'].agg([
        ('median', 'median'),
        ('q25', lambda x: x.quantile(0.25)),
        ('q75', lambda x: x.quantile(0.75)),
        ('mean', 'mean'),
        ('std', 'std'),
        ('count', 'count')
    ]).reset_index()
    tv_stats = tv_stats[tv_stats['count'] >= 10]

    tv_stats.to_csv(os.path.join(output_dir, _suffixed('tidal_volume_volume_control_modes.csv', suffix)), index=False)
    tv_stats[['hour_bin', 'mean', 'std', 'count']].to_csv(
        os.path.join(output_dir, _suffixed('tidal_volume_volume_control_modes_mean_sd.csv', suffix)), index=False)


def generate_pressure_control_stats(resp_imv_post_start, output_dir, suffix=''):
    """Generate pressure control stats for pressure control mode."""
    os.makedirs(output_dir, exist_ok=True)
    pressure_mode_data = resp_imv_post_start[
        resp_imv_post_start['mode_category'].isin(['pressure control'])
    ].copy()
    if len(pressure_mode_data) == 0 or 'pressure_control_set' not in pressure_mode_data.columns:
        return

    pressure_mode_data['hour_bin'] = pressure_mode_data['hours_from_vent_start'].round(0).astype(int)
    pressure_mode_data_7d = pressure_mode_data[pressure_mode_data['hour_bin'] <= 168].copy()

    pc_stats = pressure_mode_data_7d.groupby('hour_bin')['pressure_control_set'].agg([
        ('median', 'median'),
        ('q25', lambda x: x.quantile(0.25)),
        ('q75', lambda x: x.quantile(0.75)),
        ('mean', 'mean'),
        ('std', 'std'),
        ('count', 'count')
    ]).reset_index()
    pc_stats = pc_stats[pc_stats['count'] >= 10]

    pc_stats.to_csv(os.path.join(output_dir, _suffixed('pressure_control_pressure_control_mode.csv', suffix)), index=False)
    pc_stats[['hour_bin', 'mean', 'std', 'count']].to_csv(
        os.path.join(output_dir, _suffixed('pressure_control_pressure_control_mode_mean_sd.csv', suffix)), index=False)


def generate_mode_proportions(resp_imv_post_start, output_dir, suffix=''):
    """Generate mode proportions for first 24 hours of IMV."""
    os.makedirs(output_dir, exist_ok=True)
    if len(resp_imv_post_start) == 0:
        return

    imv_first_24h = resp_imv_post_start[
        (resp_imv_post_start['hours_from_vent_start'] >= 0) &
        (resp_imv_post_start['hours_from_vent_start'] <= 24)
    ].copy()
    if len(imv_first_24h) == 0:
        return

    mode_mapping = {
        'assist control-volume control': 'Assist Control-Volume Control',
        'pressure-regulated volume control': 'Pressure-Regulated Volume Control',
        'simv': 'SIMV',
        'pressure support/cpap': 'Pressure Support/CPAP',
        'pressure support': 'Pressure Support/CPAP',
        'cpap': 'Pressure Support/CPAP',
        'pressure control': 'Pressure Control',
    }
    imv_first_24h['mode_group'] = imv_first_24h['mode_category'].str.lower().map(mode_mapping).fillna('Other')

    mode_counts = imv_first_24h['mode_group'].value_counts()
    total_obs = len(imv_first_24h)
    mode_proportions = (mode_counts / total_obs).sort_values(ascending=False)

    plot_data = pd.DataFrame({
        'Mode': mode_proportions.index,
        'Proportion': mode_proportions.values,
        'Count': mode_counts[mode_proportions.index].values
    })
    plot_data.to_csv(os.path.join(output_dir, _suffixed('mode_proportions_first_24h.csv', suffix)), index=False)


def generate_medications_hourly(meds_merged, total_icu_encounters, med_groups, output_dir, suffix=''):
    """Generate hourly medication usage data."""
    os.makedirs(output_dir, exist_ok=True)
    all_meds = [med for meds in med_groups.values() for med in meds]

    if len(meds_merged) == 0 or total_icu_encounters == 0:
        return

    meds_7d = meds_merged[
        (meds_merged['med_lower'].isin(all_meds)) &
        (meds_merged['hour_bin'].notna()) &
        (meds_merged['hour_bin'] >= 0) &
        (meds_merged['hour_bin'] <= 167)
    ]
    if len(meds_7d) == 0:
        return

    pivot = (
        meds_7d
        .groupby(['hour_bin', 'med_lower'])['encounter_block']
        .nunique()
        .unstack(fill_value=0)
        .reindex(index=np.arange(168), columns=all_meds, fill_value=0)
    )
    pct_pivot = (pivot / total_icu_encounters * 100) if total_icu_encounters > 0 else pivot * 0

    hourly_df = pd.DataFrame({'hour': np.arange(168)})
    hourly_df = pd.concat([hourly_df, pivot.add_suffix('_n'), pct_pivot.add_suffix('_pct')], axis=1)
    hourly_df.to_csv(os.path.join(output_dir, _suffixed('medications_hourly_data.csv', suffix)), index=False)


def generate_medications_summary(meds_merged, total_icu_encounters, med_groups, output_dir, suffix=''):
    """Generate medication summary statistics."""
    os.makedirs(output_dir, exist_ok=True)
    all_meds = [med for meds in med_groups.values() for med in meds]

    if len(meds_merged) == 0 or total_icu_encounters == 0:
        return

    meds_7d = meds_merged[
        (meds_merged['med_lower'].isin(all_meds)) &
        (meds_merged['hour_bin'].notna()) &
        (meds_merged['hour_bin'] >= 0) &
        (meds_merged['hour_bin'] <= 167)
    ]
    if len(meds_7d) == 0:
        return

    summary_agg = (
        meds_7d
        .groupby('med_lower')['med_dose']
        .agg(['count', 'median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
        .rename(columns={'count': 'n_admin', '<lambda_0>': 'q1_dose', '<lambda_1>': 'q3_dose'})
    )
    encounter_counts = meds_7d.groupby('med_lower')['encounter_block'].nunique()
    dose_units = meds_7d.groupby('med_lower')['med_dose_unit'].agg(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else '')
    med_to_group = {med: group for group, meds in med_groups.items() for med in meds}

    summary_df = pd.DataFrame({
        'medication': summary_agg.index,
        'n_encounters': encounter_counts.values,
        'pct_encounters': (encounter_counts / total_icu_encounters * 100).values,
        'median_dose': summary_agg['median'].values,
        'q1_dose': summary_agg['q1_dose'].values,
        'q3_dose': summary_agg['q3_dose'].values,
        'dose_unit': dose_units.values
    })
    summary_df['group'] = summary_df['medication'].map(med_to_group)
    summary_df = summary_df[['group', 'medication', 'n_encounters', 'pct_encounters',
                             'median_dose', 'q1_dose', 'q3_dose', 'dose_unit']]
    summary_df.to_csv(os.path.join(output_dir, _suffixed('medications_summary_stats.csv', suffix)), index=False)


def generate_comorbidities(cci_results, output_dir, suffix=''):
    """Generate comorbidity prevalence per 1000 hospitalizations."""
    os.makedirs(output_dir, exist_ok=True)
    if len(cci_results) == 0:
        return

    total_hospitalizations = cci_results['encounter_block'].nunique()
    exclude_columns = {'hospitalization_id', 'encounter_block', 'cci_score'}
    comorbidity_columns = [col for col in cci_results.columns if col not in exclude_columns]
    if not comorbidity_columns:
        return

    comorbidity_counts = cci_results[comorbidity_columns].sum()
    comorbidity_per_1000 = (comorbidity_counts / total_hospitalizations) * 1000
    prevalence_percent = (comorbidity_counts.values / total_hospitalizations * 100).round(2)

    comorbidity_summary = pd.DataFrame({
        'comorbidity': comorbidity_columns,
        'n_patients': comorbidity_counts.values,
        'prevalence_percent': prevalence_percent,
        'per_1000_hospitalizations': comorbidity_per_1000.values.round(1)
    }).sort_values('per_1000_hospitalizations', ascending=False).reset_index(drop=True)

    comorbidity_summary.to_csv(os.path.join(output_dir, _suffixed('comorbidities_per_1000_hospitalizations.csv', suffix)), index=False)

    total_comorbidities = int(comorbidity_counts.sum())
    avg_comorbidities_per_hosp = total_comorbidities / total_hospitalizations if total_hospitalizations > 0 else 0
    most_common_comorbidity = comorbidity_summary.iloc[0]['comorbidity']
    most_common_per_1000 = comorbidity_summary.iloc[0]['per_1000_hospitalizations']

    summary_stats_rows = [
        ['Total hospitalizations', total_hospitalizations],
        ['Total comorbidities across all patients', total_comorbidities],
        ['Average comorbidities per hospitalization', f"{avg_comorbidities_per_hosp:.2f}"],
        ['Most common comorbidity', most_common_comorbidity],
        ['Most common: per 1000 hospitalizations', f"{most_common_per_1000:.1f}"],
    ]
    with open(os.path.join(output_dir, _suffixed('comorbidities_per_1000_hospitalizations_summary.csv', suffix)), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for row in summary_stats_rows:
            writer.writerow(row)


def generate_sofa_mortality(final_tableone_df, output_dir, suffix=''):
    """Generate SOFA-mortality summary."""
    os.makedirs(output_dir, exist_ok=True)
    if 'sofa_total' not in final_tableone_df.columns or 'death_enc' not in final_tableone_df.columns:
        return

    enc_deduped = final_tableone_df.drop_duplicates(subset=['encounter_block'])
    sofa_data = enc_deduped[enc_deduped['sofa_total'].notna()].copy()
    if len(sofa_data) == 0:
        return

    sofa_mortality = sofa_data.groupby('sofa_total').agg(
        count=('death_enc', 'count'),
        mortality_rate=('death_enc', 'mean')
    ).reset_index()
    sofa_mortality.columns = ['sofa_score', 'count', 'mortality_rate']
    sofa_mortality['mortality_rate'] = sofa_mortality['mortality_rate'] * 100
    sofa_mortality['deaths'] = (sofa_mortality['mortality_rate'] / 100) * sofa_mortality['count']

    def wilson_ci(successes, n, confidence=0.95):
        z = stats.norm.ppf((1 + confidence) / 2)
        p_hat = successes / n
        denominator = 1 + z**2 / n
        center = (p_hat + z**2 / (2*n)) / denominator
        margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4*n)) / n) / denominator
        return center * 100, margin * 100

    ci_data = [wilson_ci(deaths, n) if n > 0 else (0, 0)
               for deaths, n in zip(sofa_mortality['deaths'], sofa_mortality['count'])]
    sofa_mortality['ci_center'] = [x[0] for x in ci_data]
    sofa_mortality['ci_margin'] = [x[1] for x in ci_data]
    sofa_mortality['ci_lower'] = (sofa_mortality['mortality_rate'] - sofa_mortality['ci_margin']).clip(lower=0)
    sofa_mortality['ci_upper'] = (sofa_mortality['mortality_rate'] + sofa_mortality['ci_margin']).clip(upper=100)
    sofa_mortality['total_encounters'] = len(sofa_data)

    sofa_export = sofa_mortality[['sofa_score', 'total_encounters', 'count', 'deaths',
                                  'mortality_rate', 'ci_lower', 'ci_upper', 'ci_margin']].copy()
    sofa_export.columns = ['sofa_score', 'total_encounters', 'n_encounters', 'n_deaths',
                           'mortality_rate_percent', 'ci_lower_95', 'ci_upper_95', 'ci_margin_95']
    sofa_export['n_encounters'] = sofa_export['n_encounters'].astype(int)
    sofa_export['total_encounters'] = sofa_export['total_encounters'].astype(int)
    sofa_export['n_deaths'] = sofa_export['n_deaths'].round(0).astype(int)
    for col in ['mortality_rate_percent', 'ci_lower_95', 'ci_upper_95', 'ci_margin_95']:
        sofa_export[col] = sofa_export[col].round(2)

    sofa_export.to_csv(os.path.join(output_dir, _suffixed('sofa_mortality_summary.csv', suffix)), index=False)


def generate_hospice_trends(final_tableone_df, output_dir, suffix=''):
    """Generate hospice vs mortality trends by year."""
    os.makedirs(output_dir, exist_ok=True)
    required_cols = ['encounter_block', 'admission_year', 'hospice_outcome',
                     'expired_outcome', 'hospice_or_expired']
    if not all(c in final_tableone_df.columns for c in required_cols):
        return

    enc_deduped = final_tableone_df.drop_duplicates(subset=['encounter_block'])
    if len(enc_deduped) == 0:
        return

    hospice_trends = enc_deduped.groupby('admission_year').agg({
        'encounter_block': 'count',
        'hospice_outcome': 'sum',
        'expired_outcome': 'sum',
        'hospice_or_expired': 'sum'
    }).reset_index()
    hospice_trends.columns = ['year', 'total_encounters', 'hospice', 'expired', 'hospice_or_expired']

    hospice_trends['hospice_pct'] = hospice_trends['hospice'] / hospice_trends['total_encounters'] * 100
    hospice_trends['mortality_pct'] = hospice_trends['expired'] / hospice_trends['total_encounters'] * 100
    hospice_trends['hospice_among_eol_pct'] = (
        hospice_trends['hospice'] / hospice_trends['hospice_or_expired'] * 100
    )

    def calc_ci(successes, n, confidence=0.95):
        if n == 0:
            return 0, 0
        ci_low, ci_upp = proportion_confint(successes, n, alpha=1-confidence, method='wilson')
        return ci_low * 100, ci_upp * 100

    ci_results = [calc_ci(row['hospice'], row['hospice_or_expired'])
                  for _, row in hospice_trends.iterrows()]
    hospice_trends['hospice_among_eol_ci_lower'] = [x[0] for x in ci_results]
    hospice_trends['hospice_among_eol_ci_upper'] = [x[1] for x in ci_results]

    hospice_trends.to_csv(os.path.join(output_dir, _suffixed('hospice_trends_summary.csv', suffix)), index=False)


def generate_cci_hospice_mortality(final_tableone_df, output_dir, suffix=''):
    """Generate CCI-hospice-mortality comprehensive summary."""
    os.makedirs(output_dir, exist_ok=True)
    required_cols = ['encounter_block', 'cci_score', 'admission_year',
                     'expired_outcome', 'hospice_or_expired', 'hospice_outcome']
    if not all(c in final_tableone_df.columns for c in required_cols):
        return

    tableone_with_cci = final_tableone_df[required_cols].copy()
    tableone_with_cci['cci_category'] = pd.cut(
        tableone_with_cci['cci_score'],
        bins=[-np.inf, 0, 2, 4, np.inf],
        labels=['0 (No comorbidity)', '1-2 (Mild)', '3-4 (Moderate)', '5+ (Severe)']
    )

    cci_summary = tableone_with_cci.groupby(['admission_year', 'cci_category']).agg({
        'encounter_block': 'count',
        'hospice_outcome': 'sum',
        'expired_outcome': 'sum',
        'hospice_or_expired': 'sum'
    }).reset_index()
    cci_summary.columns = ['year', 'cci_category', 'total_encounters',
                           'hospice_count', 'expired_count', 'hospice_or_expired_count']

    cci_summary['mortality_pct'] = cci_summary['expired_count'] / cci_summary['total_encounters'] * 100
    cci_summary['hospice_pct'] = cci_summary['hospice_count'] / cci_summary['total_encounters'] * 100
    cci_summary['combined_eol_pct'] = cci_summary['hospice_or_expired_count'] / cci_summary['total_encounters'] * 100
    cci_summary['hospice_among_eol_pct'] = cci_summary['hospice_count'] / cci_summary['hospice_or_expired_count'] * 100
    cci_summary['hospice_capture_rate'] = cci_summary['hospice_count'] / cci_summary['expired_count'] * 100

    def calc_ci(row):
        if row['hospice_or_expired_count'] == 0:
            return pd.Series({'hospice_eol_ci_lower': np.nan, 'hospice_eol_ci_upper': np.nan})
        ci_low, ci_upp = proportion_confint(
            row['hospice_count'], row['hospice_or_expired_count'], alpha=0.05, method='wilson')
        return pd.Series({'hospice_eol_ci_lower': ci_low * 100, 'hospice_eol_ci_upper': ci_upp * 100})

    cci_summary[['hospice_eol_ci_lower', 'hospice_eol_ci_upper']] = cci_summary.apply(calc_ci, axis=1)

    cci_summary = cci_summary[['year', 'cci_category', 'total_encounters',
                                'expired_count', 'mortality_pct', 'hospice_count', 'hospice_pct',
                                'hospice_or_expired_count', 'combined_eol_pct',
                                'hospice_among_eol_pct', 'hospice_eol_ci_lower', 'hospice_eol_ci_upper',
                                'hospice_capture_rate']]
    cci_summary.to_csv(os.path.join(output_dir, _suffixed('cci_hospice_mortality_comprehensive_summary.csv', suffix)), index=False)

    # Plot data
    categories = ['0 (No comorbidity)', '1-2 (Mild)', '3-4 (Moderate)', '5+ (Severe)']
    plot_data = []
    for category in categories:
        cat_data = cci_summary[cci_summary['cci_category'] == category].sort_values('year')
        plot_data.append(cat_data.assign(cci_category_label=category))
    plot_data_df = pd.concat(plot_data, axis=0)
    plot_data_df.to_csv(os.path.join(output_dir, _suffixed('cci_mortality_hospice_trends_by_year_category_plotdata.csv', suffix)), index=False)


def generate_demographic_crosstab(patient_df, output_dir, suffix=''):
    """Generate NIH enrollment report demographic crosstab."""
    os.makedirs(output_dir, exist_ok=True)
    enrollment_report = crosstab_demographics(patient_df)
    enrollment_report.to_csv(os.path.join(output_dir, _suffixed('demographic_crosstab_race_ethnicity_sex.csv', suffix)))


def main(memory_monitor=None, cohort_mode='critical_illness') -> bool:
    """
    Main execution function for Table One generation.

    Parameters:
        memory_monitor: Optional MemoryMonitor instance for tracking memory usage
        cohort_mode: 'critical_illness' (default) or 'ward'.
            - 'critical_illness': cohort = encounters with ICU stay OR died/hospice.
              Outputs route to output/final/overall/{tableone,figures,...}/.
            - 'ward': cohort = every adult encounter that touched a ward at any point.
              Outputs route to output/final/overall_ward/{tableone,figures,...}/.
              SOFA, ICU LOS, ICU episodes, IMV/ventilator settings, and the
              medication-from-ICU plot are skipped in ward mode (Decisions 1-3
              of the ward Table One design).

    Returns:
        bool: True if generation completed successfully, False otherwise
    """

    # Checkpoint wrapper function
    def checkpoint(label):
        """Record a memory checkpoint if monitor is available."""
        if memory_monitor:
            memory_monitor.checkpoint(label)
        print(f"  [Memory Checkpoint: {label}]")

    # Load configuration
    # Get the project root directory (3 levels up from this file)
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "config" / "config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    checkpoint("Configuration Loaded")

    # Create the output directory for tableone results if it does not already exist
    # Create necessary output directories within output/final

    # Get output paths helper
    def get_output_path(*parts):
        """Helper to get output paths relative to project root"""
        path = project_root / 'output'
        for part in parts:
            path = path / part
        return str(path)

    # Build the full output/final/ tree (overall, strata, validation, meta, ...).
    # In ward mode, also build the parallel overall_ward/ subtree, and bind the
    # local _tableone_dir / _figures_dir / etc. names to the WARD versions of the
    # helpers. All ~80 downstream call sites in this file use these local names,
    # so they automatically resolve to the right cohort tree without further changes.
    from modules.utils.output_paths import (
        ensure_output_tree,
        ensure_ward_output_tree,
        validation_json_reports_dir as _validation_json_reports_dir,
    )
    ensure_output_tree()

    if cohort_mode == 'ward':
        ensure_ward_output_tree()
        from modules.utils.output_paths import (
            ward_tableone_dir as _tableone_dir,
            ward_figures_dir as _figures_dir,
            ward_mcide_dir as _mcide_dir,
            ward_summary_stats_dir as _summary_stats_dir,
        )
    else:
        from modules.utils.output_paths import (
            tableone_dir as _tableone_dir,
            figures_dir as _figures_dir,
            mcide_dir as _mcide_dir,
            summary_stats_dir as _summary_stats_dir,
        )

    # CSV outputs (table_one_overall.csv, mortality_rates.csv, etc.)
    output_dir = str(_tableone_dir())
    # PNG/HTML/PDF outputs (consort, venn, upset, sankey, ...)
    figures_dir = str(_figures_dir())
    mcide_dir = str(_mcide_dir())
    clifpy_dir = str(_validation_json_reports_dir())
    summary_stats_dir = str(_summary_stats_dir())

    print(f"\n=� Configuration:")
    print(f"   Data directory: {config['tables_path']}")
    print(f"   File type: {config['file_type']}")
    print(f"   Timezone: {config['timezone']}")


    # ==============================================================================
    # ## Required columns and categories
    # ==============================================================================

    print("\n" + "=" * 80)
    print("Defining Required Data Elements")
    print("=" * 80)

    # Full patient table 

    # Full hospitalization table 

    # Full ADT table

    # Vitals
    vitals_required_columns = [
        'hospitalization_id',
        'recorded_dttm',
        'vital_category',
        'vital_value'
    ]
    vitals_of_interest = ['heart_rate', 'respiratory_rate', 'sbp', 'dbp', 'map', 'spo2', 'weight_kg', 'height_cm']

    # Respiratory Support 
    rst_required_columns = [
        'hospitalization_id',
        'recorded_dttm',
        'device_name',
        'device_category',
        'mode_name',
        'mode_category',
        'tracheostomy',
        'fio2_set',
        'lpm_set',
        'resp_rate_set',
        'peep_set',
        'resp_rate_obs',
        'tidal_volume_set',
        'pressure_control_set',
        'pressure_support_set',
        'flow_rate_set',           # Added for ventilator settings table
        'peak_inspiratory_pressure_set',
        'peak_inspiratory_pressure_obs',
        'plateau_pressure_obs',
        'minute_vent_obs'
    ]


    # Continuous administered meds
    meds_required_columns = [
        'hospitalization_id',
        'admin_dttm',
        'med_name',
        'med_category',
        'med_dose',
        'med_dose_unit'
    ]
    meds_of_interest = [
        'norepinephrine', 'epinephrine', 'phenylephrine', 'vasopressin',
        'dopamine', 'angiotensin', 'dobutamine', 'milrinone', 'isoproterenol',
        'propofol', 'midazolam', 'lorazepam', 'dexmedetomidine', 
        'vecuronium', 'rocuronium', 'cisatracurium', 'pancuronium'
    ]

    strobe_counts = {}


    # ==============================================================================
    # ## Functions
    # ==============================================================================

    # NOTE: MCIDE collection has been moved to a separate script (generate_mcide_and_stats.py)
    # for better memory efficiency. Run that script independently after validation
    # to collect MCIDE statistics and summary stats without loading large tables into memory.

    # Import MCIDE collector if available (kept for backward compatibility)
    try:
        # Updated import path after refactoring
        from modules.mcide.collector import get_value_counts_polars
        POLARS_AVAILABLE = True
    except ImportError:
        POLARS_AVAILABLE = False
        # print("Note: Polars-based MCIDE collection not available. Using pandas fallback.")

    def get_value_counts_mcide(clif_table, table_name, field_names, output_dir=None, config=None):
        """
        Get N (count) for all unique combinations of the specified fields from CLIF table.
        Uses Polars for efficient scanning if available, otherwise falls back to pandas.

        Parameters
        ----------
        clif_table : CLIF table object or None
            CLIF table object with .df attribute (e.g., clif.patient). Can be None if using Polars.
        table_name : str
            Name of the table (e.g., 'patient', 'labs')
        field_names : list of str
            List of field names to calculate count combinations for.
        output_dir : str, optional
            Directory path to save CSV file. If provided, saves MCIDE CSV.
        config : dict, optional
            Configuration dictionary with tables_path and filetype

        Returns
        -------
        pd.DataFrame or pl.DataFrame
            DataFrame with all unique combinations of field_names and a column 'N' with counts.
        """
        # Try Polars-based approach if available and config provided
        if POLARS_AVAILABLE and config and output_dir:
            try:
                tables_path = config.get('tables_path', '')
                file_type = config.get('filetype', 'parquet')
                table_path = os.path.join(tables_path, f"{table_name}.{file_type}")

                if os.path.exists(table_path):
                    # Use Polars for efficient scanning
                    result = get_value_counts_polars(
                        table_path,
                        table_name,
                        field_names,
                        output_dir,
                        file_type
                    )
                    # Convert to pandas for compatibility if needed
                    if hasattr(result, 'to_pandas'):
                        return result.to_pandas()
                    return result
            except Exception as e:
                print(f"Warning: Polars collection failed for {table_name}, using pandas: {e}")

        # Fallback to pandas approach
        if clif_table is None:
            raise ValueError("CLIF table object required when Polars is not available")

        df = clif_table.df
        # Filter to only valid columns
        valid_fields = [field for field in field_names if field in df.columns]
        if not valid_fields:
            print(f"Warning: None of {field_names} found in {table_name}")
            return pd.DataFrame()

        # Group by all specified fields and count
        combo_counts = (
            df.groupby(valid_fields, dropna=False)
              .size()
              .reset_index(name='N')
        )

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            # Use new naming convention
            columns_str = '_'.join(valid_fields)
            filename = f"{table_name}_{columns_str}_mcide.csv"
            out_path = os.path.join(output_dir, filename)
            combo_counts.to_csv(out_path, index=False)

        return combo_counts

    # Keep original function for backward compatibility
    def get_value_counts(clif_table, field_names, output_dir=None):
        """Legacy function for backward compatibility."""
        # Extract table name from clif_table if possible
        table_name = "unknown"
        if hasattr(clif_table, '__class__'):
            table_name = clif_table.__class__.__name__.lower().replace('table', '')

        return get_value_counts_mcide(clif_table, table_name, field_names, output_dir)

    def create_summary_table(clif_table, numeric_cols, group_by_cols=None, output_dir=None):
        """
        Create summary statistics for numeric columns from CLIF table.
    
        Parameters
        ----------
        clif_table : CLIF table object
            CLIF table object with .df attribute.
        numeric_cols : str or list of str
            Column name(s) to summarize (e.g., 'fio2_set' or ['fio2_set', 'peep_set']).
        group_by_cols : str or list of str, optional
            Column name(s) to group by. If None, provides overall summary.
        output_dir : str, optional
            Directory path to save CSV. If provided, saves summary table.
        
        Returns
        -------
        pd.DataFrame
            Summary with columns: [group_cols (if any), 'variable', 'N', 'missing', 
            'min', 'q25', 'median', 'q75', 'mean', 'max'].
        
        Examples
        --------
        >>> summary = create_summary_table(clif.respiratory_support, 'fio2_set')
        >>> summary = create_summary_table(clif.respiratory_support, 
        ...                                ['fio2_set', 'peep_set'], 
        ...                                group_by_cols='device_category',
        ...                                output_dir=get_output_path('final', 'tableone'))
        """
        df = clif_table.df
        table_name = str(clif_table).split('.')[-1].split()[0]
        if isinstance(numeric_cols, str):
            numeric_cols = [numeric_cols]
        if isinstance(group_by_cols, str):
            group_by_cols = [group_by_cols]
    
        summaries = []
    
        for col in numeric_cols:
            if col not in df.columns:
                continue
        
            agg_dict = {
                col: ['count', lambda x: x.isna().sum(), 'min', 
                      lambda x: x.quantile(0.25), lambda x: x.quantile(0.50),
                      lambda x: x.quantile(0.75), 'mean', 'max']
            }
        
            if group_by_cols:
                summary = df.groupby(group_by_cols).agg(agg_dict)
                summary.columns = summary.columns.droplevel(0)
                summary = summary.reset_index()
            else:
                summary = df.agg(agg_dict).to_frame().T
                summary.columns = summary.columns.droplevel(0)
        
            summary.columns = (list(group_by_cols) if group_by_cols else []) + \
                             ['N', 'missing', 'min', 'q25', 'median', 'q75', 'mean', 'max']
            summary.insert(len(group_by_cols) if group_by_cols else 0, 'variable', col)
            summaries.append(summary)
    
        result = pd.concat(summaries, ignore_index=True)
    
        if output_dir:
            filename = f"{table_name}_{'_'.join(group_by_cols) if group_by_cols else 'overall'}_summary.csv"
            result.to_csv(f"{output_dir}/{filename}", index=False)
    
        return result

    def get_distinct_colors(n):
        """Generate n visually distinct colors."""
        hue_partition = 1 / (n + 1)
        colors = [colorsys.hsv_to_rgb(hue_partition * value, 0.8, 0.5)
                  for value in range(0, n)]
        return reversed(colors[::2] + colors[1::2])


    class Sankey:
        def __init__(self, df,
                     plot_width=8,
                     plot_height=8,
                     gap=0.12,
                     alpha=0.3,
                     fontsize='small',
                     order=None,
                     mapping=None,
                     tag=None,
                     title=None,
                     title_left=None,
                     title_right=None,
                     labels=True,
                     block_width=0.1,
                     block_fontsize=12,
                     flow_color_func=None,
                     colors=None,
                     ax=None
        ):
            self.df = df
            if ax:
                self.plot_width = ax.get_position().width * ax.figure.get_size_inches()[0]
                self.plot_height = ax.get_position().height * ax.figure.get_size_inches()[1]
            else:
                self.plot_width = plot_width
                self.plot_height = plot_height
            self.gap = gap
            self.block_width = block_width
            self.block_fontsize = block_fontsize
            self.alpha = alpha
            self.labels = labels
            self.fontsize = fontsize
            self.order = order
            self.flow_color_func = flow_color_func
            self.mapping_colors = {
                'increase': '#1f721c',
                'decrease': '#ddc90f',
                'mistake': '#dd1616',
                'correct': '#dddddd',
                'novel': '#59a8d6',
            }

            self.init_figure(ax)
            self.init_flows()
            self.init_nodes(order)
            self.init_widths()
        
            # inches per 1 item in x and y
            self.resolution = (plot_height - gap * (len(order) - 1)) / df.shape[0]
        
            if colors is not None:
                self.colors = colors
            else:
                self.colors = {
                    name: colour
                    for name, colour
                    in zip(self.nodes[0].keys(),
                        get_distinct_colors(len(self.nodes[0])))
                }

            self.init_offsets()

        def init_figure(self, ax):
            if ax is None:
                self.fig = plt.figure()
                self.ax = plt.Axes(self.fig, [0, 0, 1, 1])
                self.fig.add_axes(self.ax)
            else:
                self.fig = ax.figure
                self.ax = ax

        def init_flows(self):
            self.flows = []
            n_cols = self.df.columns.size
            for i in range(n_cols - 1):
                x, y = self.df.iloc[:, i], self.df.iloc[:, i + 1]
                self.flows.append(collections.Counter(zip(x, y)))

        def init_nodes(self, order):
            self.nodes = []

            for i in range(self.df.columns.size):
                column = collections.OrderedDict()
                counts = self.df.iloc[:, i].value_counts()
                for item in order:
                    if item in counts:
                        column[item] = counts[item]
                    else:
                        column[item] = 0
                self.nodes.append(column)

        def init_widths(self):
            self.left_stop = self.block_width
            self.right_stop = self.plot_width - self.block_width
            self.stops = []
            n_cols = self.df.columns.size
            self.flow_width = (self.plot_width - self.block_width * (n_cols - 2)) / (n_cols - 1)

            for i in range(1, n_cols):
                stop1 = (self.block_width * i
                         + self.flow_width * (i - 1) + self.flow_width * 7 / 20)
                stop2 = (self.block_width * i
                         + self.flow_width * (i - 1) + self.flow_width * 13 / 20)
                self.stops.append((stop1, stop2))

        def init_offsets(self):
            self.offsets = []

            for col in self.nodes:
                offset = 0
                offsets = collections.OrderedDict()
                for name, size in col.items():
                    offsets[name] = offset
                    offset += size * self.resolution + self.gap
                self.offsets.append(offsets)

        def draw_flow(self, x, left, right, flow, node_offsets_l, node_offsets_r):
            P = matplotlib.path.Path

            left_y = self.offsets[x][left] + node_offsets_l[left]
            right_y = self.offsets[x + 1][right] + node_offsets_r[right]

            flow *= self.resolution

            node_offsets_l[left] += flow
            node_offsets_r[right] += flow
        
            if self.flow_color_func is not None:
                mapping = self.flow_color_func(left, right)
                color = self.mapping_colors[mapping]
            else:
                color = self.colors[left]

            left_x = self.flow_width * x + self.block_width * (x + 1)
            right_x  = left_x + self.flow_width

            path_data = [
                (P.MOVETO, (left_x, -left_y)),
                (P.LINETO, (left_x, -left_y - flow)),
                (P.CURVE4, (self.stops[x][0], -left_y - flow)),
                (P.CURVE4, (self.stops[x][1], -right_y - flow)),
                (P.CURVE4, (right_x, -right_y - flow)),
                (P.LINETO, (right_x, -right_y)),
                (P.CURVE4, (self.stops[x][1], -right_y)),
                (P.CURVE4, (self.stops[x][0], -left_y)),
                (P.CURVE4, (left_x, -left_y)),
                (P.CLOSEPOLY, (left_x, -left_y)),
            ]
            codes, verts = zip(*path_data)
            path = P(verts, codes)
            patch = matplotlib.patches.PathPatch(
                path,
                facecolor=color,
                alpha=0.9 if flow < .02 else self.alpha,
                edgecolor='none',
            )
            self.ax.add_patch(patch)

        def draw_node(self, x, y, size, name):
            if size <= 0:
                return
            y = -list(self.offsets[x].values())[y] - size * self.resolution
            x = self.flow_width * x + self.block_width * x
            color = self.colors[name]
            patch = matplotlib.patches.Rectangle(
                (x, y),
                width=self.block_width,
                height=size * self.resolution,
                facecolor=color,
                edgecolor='none',
            )
            self.ax.add_patch(patch)
            self.ax.text(
                x + self.block_width / 2,
                y + size * self.resolution / 2,
                name,
                color="black",
                va="center",
                ha="center",
                size=self.block_fontsize,
                path_effects=[
                    matplotlib.patheffects.Stroke(linewidth=2, foreground="white"),
                    matplotlib.patheffects.Normal()
                ]
            )

        def draw(self):
            for x, col in enumerate(self.nodes):
                for y, (name, size) in enumerate(col.items()):
                    self.draw_node(x, y, size, name)

            for x, flows in enumerate(self.flows):
                node_offsets_l = collections.Counter()
                node_offsets_r = collections.Counter()

                for (left, right), flow in sorted(
                    flows.items(),
                    key=lambda x: (self.order.index(x[0][0]), self.order.index(x[0][1]))
                ):
                    self.draw_flow(
                        x,
                        left,
                        right,
                        flow,
                        node_offsets_l,
                        node_offsets_r
                    )

            self.ax.set_ylim(
                -self.resolution * self.df.shape[0] - self.gap * (len(self.order) - 1),
                0
            )
            self.ax.set_xlim(
                0,
                self.block_width * self.df.shape[1] + self.flow_width * (self.df.shape[1] - 1)
            )
            self.ax.get_xaxis().set_visible(False)
            self.ax.get_yaxis().set_visible(False)
            for k in self.ax.spines.keys():
                self.ax.spines[k].set_visible(False)


    # ============================================================================
    # Data Preparation Functions
    # ============================================================================

    def simplify_location_category(location):
        """Map location to simplified categories."""
        if pd.isna(location):
            return 'Other'
    
        location_lower = str(location).lower()
    
        if location_lower == 'icu':
            return 'ICU'
        elif location_lower == 'ward':
            return 'Ward'
        elif location_lower == 'ed':
            return 'ED'
        elif location_lower == 'procedural':
            return 'Procedural'
        else:
            return 'Other'


    def create_outcome_df(final_tableone_df):
        """Create outcome dataframe with death indicator."""
        outcome_df = final_tableone_df[['encounter_block', 'death_enc', 'death_dttm', 'discharge_dttm']].copy()
        outcome_df['final_outcome_dttm'] = outcome_df['death_dttm'].fillna(outcome_df['discharge_dttm'])
        outcome_df = outcome_df[['encounter_block', 'death_enc', 'final_outcome_dttm']].drop_duplicates()
        return outcome_df


    def prepare_sankey_wide_format(adt_cohort, encounter_blocks, outcome_df, max_locations=7):
        """
        Transform ADT data into wide format for Sankey diagram.
        Returns DataFrame where each row = encounter, each column = location position.
        """
        # Filter and sort
        adt_filtered = adt_cohort[adt_cohort['encounter_block'].isin(encounter_blocks)].copy()
        adt_filtered = adt_filtered.sort_values(['encounter_block', 'in_dttm'])

        # Simplify locations
        adt_filtered['location_simple'] = adt_filtered['location_category'].apply(simplify_location_category)

        # Drop reverse transitions (ICU/Ward/Procedural to ED) within each encounter
        # These are artifacts from stitching and should be ignored
        initial_count = len(adt_filtered)

        # Create previous location column for each encounter_block
        adt_filtered['prev_location'] = adt_filtered.groupby('encounter_block')['location_simple'].shift(1)

        # Identify reverse transitions to ED
        reverse_to_ed = (
            (adt_filtered['location_simple'] == 'ED') &
            (adt_filtered['prev_location'].isin(['ICU', 'Ward', 'Procedural']))
        )

        # Drop these rows
        rows_to_drop = reverse_to_ed.sum()
        if rows_to_drop > 0:
            print(f"    📋 Dropping {rows_to_drop:,} reverse transitions to ED (from ICU/Ward/Procedural)")
            print(f"       This represents {rows_to_drop/initial_count*100:.2f}% of {initial_count:,} total transitions")

        adt_filtered = adt_filtered[~reverse_to_ed].copy()

        # Drop the helper column
        adt_filtered = adt_filtered.drop(columns=['prev_location'])

        # Add position/segment rank
        adt_filtered['segment_rank'] = adt_filtered.groupby('encounter_block').cumcount() + 1
        adt_filtered = adt_filtered[adt_filtered['segment_rank'] <= max_locations]

        # Pivot to wide format
        sankey_df = adt_filtered.pivot(
            index='encounter_block',
            columns='segment_rank',
            values='location_simple'
        ).reset_index()

        # Rename columns to float format (1.0, 2.0, etc.) to match example
        column_mapping = {col: float(col) for col in sankey_df.columns if col != 'encounter_block'}
        sankey_df = sankey_df.rename(columns=column_mapping)

        # Fill NaN with 'Discharged'
        location_cols = [float(i) for i in range(1, max_locations + 1)]
        for col in location_cols:
            if col not in sankey_df.columns:
                sankey_df[col] = 'Discharged'
            else:
                sankey_df[col] = sankey_df[col].fillna('Discharged')

        # Merge death information
        sankey_df = sankey_df.merge(
            outcome_df[['encounter_block', 'death_enc']],
            on='encounter_block',
            how='left'
        )

        return sankey_df, location_cols



    def propagate_death(df, location_cols):
        """
        Propagate 'Died' status across all subsequent segments.
        Once a patient dies, all future segments show 'Died'.
        """
        def propagate_row(row):
            death_found = False
            for col in location_cols:
                if row[col] == 'Died':
                    death_found = True
                if death_found:
                    row[col] = 'Died'
            return row
    
        df[location_cols] = df[location_cols].apply(propagate_row, axis=1)
        return df


    def mark_final_outcomes(df, location_cols):
        """
        Mark the final outcome for each encounter based on death_enc.
        """
        for idx, row in df.iterrows():
            # Find last actual location (not Discharged/Died)
            last_location_idx = None
            for i, col in enumerate(location_cols):
                if row[col] not in ['Discharged', 'Died']:
                    last_location_idx = i
        
            if last_location_idx is not None:
                # Determine outcome
                if row['death_enc'] == 1:
                    outcome = 'Died'
                else:
                    outcome = 'Discharged'
            
                # Fill remaining positions with outcome
                for i in range(last_location_idx + 1, len(location_cols)):
                    df.at[idx, location_cols[i]] = outcome
    
        return df


    def create_sankey_diagram(adt_cohort, encounter_blocks, outcome_df, 
                             max_locations=7,
                             title="Patient Flow Through Locations",
                             output_file=None,
                             figsize=(16, 8)):
        """
        Create Sankey diagram using custom matplotlib Sankey class.
        """
        # Prepare data in wide format
        sankey_df, location_cols = prepare_sankey_wide_format(
            adt_cohort, encounter_blocks, outcome_df, max_locations
        )
    
        # Mark final outcomes based on death_enc
        sankey_df = mark_final_outcomes(sankey_df, location_cols)
    
        # Propagate death status through remaining positions
        sankey_df = propagate_death(sankey_df, location_cols)
    
        # Define colors (matching your preferences)
        # colors = {
        #     "ED": '#f08080',         # Light coral (red)
        #     "ICU": '#c9a0a0',        # Dusty rose
        #     "Ward": '#87ceeb',       # Sky blue
        #     'Procedural': '#d8bfd8', # Thistle (purple)
        #     "Other": '#d3d3d3',      # Light gray
        #     "Discharged": '#d3d3d3', # Light gray
        #     "Died": '#696969'        # Dim gray (dark)
        # }
        colors = {
                "ED": '#f08080',         # Light coral (keep)
                "ICU": '#87ceeb',        # Sky blue (SWAP with Ward!)
                "Ward": '#d4c5a0',       # Beige/tan (neutral, distinct)
                'Procedural': '#c9a0c9', # Light purple (but be careful)
                "Other": '#e8e8e8',      # Very light gray
                "Discharged": '#b8d4b8', # Light sage green
                "Died": '#696969'        # Dark gray (keep)
            }
    
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    
        # Create Sankey diagram
        diag = Sankey(
            sankey_df[location_cols],
            ax=ax,
            order=["ED", "Ward", "ICU", "Procedural", "Other", "Discharged", "Died"],
            block_width=0.15,
            colors=colors,
            alpha=0.4,
            block_fontsize=10,
            gap=0.05  # Smaller gap between node types
        )
        diag.draw()
    
        # Set title
        ax.set_title(f"{title}\nN={len(encounter_blocks)} encounters", 
                     size=18, pad=20, fontweight='bold')
    
        # Set x-axis ticks for location numbers
        ax.set_xticks([
            diag.block_width / 2 + diag.flow_width * x + diag.block_width * x 
            for x in range(len(location_cols))
        ])
        ax.set_xticklabels([int(col) for col in location_cols])
        ax.set_xlabel("Location number", size=16, fontweight='bold')
        ax.get_xaxis().set_visible(True)
        ax.tick_params(axis="x", pad=10, labelsize=14)
    
        # Save if requested
        if output_file:
            fig.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Saved: {output_file}")
    
        plt.tight_layout()
        return fig, sankey_df


    # ==============================================================================
    # ## Cohort identification
    # ==============================================================================

    print("\n" + "=" * 80)
    print("Loading CLIF Tables")
    print("=" * 80)


    # Initialize ClifOrchestrator
    clif = ClifOrchestrator(
        data_directory=config['tables_path'],
        filetype=config['file_type'],
        timezone=config['timezone'],
        output_directory=clifpy_dir
    )


    # ==============================================================================
    # ## Step0: Load Core Tables
    # ==============================================================================

    # ============================================================================
    # STEP 0: Load Core Tables (Patient, Hospitalization, ADT)
    # ============================================================================
    print("\n" + "=" * 80)
    print("Step 0: Load Core Tables (Patient, Hospitalization, ADT)")
    print("=" * 80)
    core_tables = ['patient', 'hospitalization', 'adt']

    print(f"\nLoading {len(core_tables)} core tables...")
    for table_name in core_tables:
        print(f"   Loading {table_name}...", end=" ")
        try:
            clif.load_table(table_name)
            table = getattr(clif, table_name)
            print(f"✓ ({len(table.df):,} rows)")
        except Exception as e:
            print(f"✗ Error: {e}")
            raise

    print("\nCore tables loaded successfully!")

    hosp_df = clif.hospitalization.df
    adt_df = clif.adt.df

    # Merge to get age information
    all_encounters = pd.merge(
        hosp_df[["patient_id", "hospitalization_id", "admission_dttm", "discharge_dttm", 
                 "age_at_admission", "discharge_category", "admission_type_category"]],
        adt_df[["hospitalization_id", "hospital_id", "in_dttm", "out_dttm", 
                "location_category", "location_type"]],
        on='hospitalization_id',
        how='inner'
    )

    # MCIDE collection moved to separate script: generate_mcide_and_stats.py
    # This runs independently for better memory efficiency
    # get_value_counts_mcide(clif.adt, 'adt', ['location_name', 'location_category', 'location_type'], output_dir=mcide_dir, config=config)
    # get_value_counts_mcide(clif.hospitalization, 'hospitalization', ['discharge_name', 'discharge_category'], output_dir=mcide_dir, config=config)
    # get_value_counts_mcide(clif.hospitalization, 'hospitalization', ['admission_type_name', 'admission_type_category'], output_dir=mcide_dir, config=config)
    # get_value_counts_mcide(clif.patient, 'patient', ['race_name', 'race_category'], output_dir=mcide_dir, config=config)
    # get_value_counts_mcide(clif.patient, 'patient', ['ethnicity_name', 'ethnicity_category'], output_dir=mcide_dir, config=config)

    # Check for duplicates by ['hospitalization_id', 'in_dttm', 'out_dttm']
    dup_counts = all_encounters.duplicated(subset=['hospitalization_id', 'in_dttm', 'out_dttm']).sum()
    if dup_counts > 0:
        print(f"Warning: {dup_counts} duplicate (hospitalization_id, in_dttm, out_dttm) entries found in all_encounters.")
    else:
        print("No duplicate (hospitalization_id, in_dttm, out_dttm) entries found in all_encounters.")


    # ==============================================================================
    # ## Step1: Date & Age filter
    # ==============================================================================

    # ============================================================================
    # STEP 1: Identify Adult Patients (Age >= 18) and Admissions 2018-2024
    # ============================================================================
    print("\n" + "=" * 80)
    print("Step 1: Identifying Adult Patients (Age >= 18) and Admissions 2018-2024")
    print("=" * 80)

    print("Applying initial cohort filters...")

    # Use only the relevant columns from all_encounters
    adult_encounters = all_encounters[
        [
            'patient_id', 'hospitalization_id', 'admission_dttm', 'discharge_dttm',
            'age_at_admission', 'discharge_category', 'admission_type_category' ,'hospital_id',
            'in_dttm', 'out_dttm', 'location_category', 'location_type'
        ]
    ].copy()

    # no filtering by year
    if config['site_name'].lower() == "mimic":
        # MIMIC: only age >= 18, no admit year restriction
        adult_encounters = adult_encounters[
            (adult_encounters['age_at_admission'] >= 18) & (adult_encounters['age_at_admission'].notna())
        ]
    else:
        # Other sites: age >= 18 and admission between 2018-2024 inclusive
        adult_encounters = adult_encounters[
            (adult_encounters['age_at_admission'] >= 18) &
            (adult_encounters['age_at_admission'].notna()) #&
            # (adult_encounters['admission_dttm'].dt.year >= 2018) &
            # (adult_encounters['admission_dttm'].dt.year <= 2024)
        ]

    print(f"\nFiltering Results:")
    print(f"   Total hospitalizations: {len(all_encounters['hospitalization_id'].unique()):,}")
    print(f"   Adult hospitalizations (age >= 18): {len(adult_encounters['hospitalization_id'].unique()):,}")
    print(f"   Excluded (age < 18): {len(all_encounters['hospitalization_id'].unique()) - len(adult_encounters['hospitalization_id'].unique()):,}")


    strobe_counts["0_total_hospitalizations"] = len(all_encounters['hospitalization_id'].unique())
    strobe_counts["1_adult_hospitalizations"] = len(adult_encounters['hospitalization_id'].unique())
    # Get list of adult hospitalization IDs for filtering
    adult_hosp_ids = set(adult_encounters['hospitalization_id'].unique())
    print(f"\n   Unique adult hospitalization IDs: {len(adult_hosp_ids):,}")


    # ==============================================================================
    # ### Stitch hospitalizations 
    #
    # If the `id_col` supplied by user is `hospitalization_id`, then we combine multiple `hospitalization_ids` into a single `encounter_block` for patients who transfer between hospital campuses or return soon after discharge. Hospitalizations that have a gap of **6 hours or less** between the discharge dttm and admission dttm are put in one encounter block.
    #
    # If the `id_col` supplied by user is `hospitalization_joined_id` from the hospitalization table, then we consider the user has already stitched similar encounters, and we will consider that as the primary id column for all table joins moving forward.
    # ==============================================================================


    # stitch hospitalizations
    hosp_filtered = clif.hospitalization.df[clif.hospitalization.df['hospitalization_id'].isin(adult_hosp_ids)]
    adt_filtered = clif.adt.df[clif.adt.df['hospitalization_id'].isin(adult_hosp_ids)]

    hosp_stitched, adt_stitched, encounter_mapping = stitch_encounters(
        hospitalization=hosp_filtered,
        adt=adt_filtered,
        time_interval=6  
    )

    # Direct assignment without additional copies
    clif.hospitalization.df = hosp_stitched
    clif.adt.df = adt_stitched

    # Store the encounter mapping in the orchestrator for later use
    clif.encounter_mapping = encounter_mapping

    # Clean up intermediate variables
    del hosp_filtered, adt_filtered
    gc.collect()
    checkpoint("Core Data Loaded & Stitched")

    # After your stitching code, add these calculations:

    # Calculate stitching statistics
    strobe_counts['1b_before_stitching'] = len(adult_hosp_ids)  # Original adult hospitalizations
    strobe_counts['1b_after_stitching'] = len(hosp_stitched['encounter_block'].unique())  # Unique encounter blocks after stitching
    strobe_counts['1b_stitched_hosp_ids'] = strobe_counts['1b_before_stitching'] - strobe_counts['1b_after_stitching']  # Number of hospitalizations that were linked

    print(f"\nEncounter Stitching Results:")
    print(f"   Number of unique hospitalizations before stitching: {strobe_counts['1b_before_stitching']:,}")
    print(f"   Number of unique encounter blocks after stitching: {strobe_counts['1b_after_stitching']:,}")
    print(f"   Number of linked hospitalization ids: {strobe_counts['1b_stitched_hosp_ids']:,}")

    # Optional: Show the encounter mapping details
    print(f"\nEncounter Mapping Details:")
    print(f"   Total encounter mappings created: {len(encounter_mapping):,}")
    if len(encounter_mapping) > 0:
        # Show some examples of how many original hospitalizations were combined
        mapping_counts = encounter_mapping.groupby('encounter_block').size()
        print(f"   Encounter blocks with multiple hospitalizations: {(mapping_counts > 1).sum():,}")
        print(f"   Maximum hospitalizations combined into one block: {mapping_counts.max()}")


    # ==============================================================================
    # # ADT
    # ==============================================================================

    # Merge all_encounters with encounter_mapping to get encounter_block information
    # Filter to adult hospitalizations before merge to avoid NaN in encounter_block
    all_encounters = all_encounters[all_encounters['hospitalization_id'].isin(adult_hosp_ids)]
    all_encounters = pd.merge(all_encounters, encounter_mapping, on='hospitalization_id', how='left')

    # Convert location_category and discharge_category to lowercase in place (vectorized)
    all_encounters['location_category'] = all_encounters['location_category'].str.lower()
    all_encounters['discharge_category'] = all_encounters['discharge_category'].str.lower()
    all_encounters['admission_type_category'] = all_encounters['admission_type_category'].str.lower()

    # Create vectorized ICU, death, and ward masks
    icu_mask = all_encounters['location_category'].str.contains('icu', na=False)
    death_mask = all_encounters['discharge_category'].isin(['expired', 'hospice'])
    ward_mask = all_encounters['location_category'] == 'ward'

    # Vectorized: For each encounter_block, does any row have ICU, death, or ward? (much faster)
    # Use groupby('encounter_block')[mask].transform('any') to vectorize
    all_encounters['icu_enc'] = icu_mask.groupby(all_encounters['encounter_block']).transform('any').astype(int)
    all_encounters['death_enc'] = death_mask.groupby(all_encounters['encounter_block']).transform('any').astype(int)
    all_encounters['ward_enc'] = ward_mask.groupby(all_encounters['encounter_block']).transform('any').astype(int)

    # Cohort flag deferred until after high_support_enc and vaso_support_enc
    # are computed — see "Define cohort and filter" block after support flags.
    # In ward mode the flag is simple enough to set now.
    if cohort_mode == 'ward':
        all_encounters['cohort_enc'] = all_encounters['ward_enc']

    # Identify encounters where death occurred
    death_encounters = all_encounters[all_encounters['death_enc'] == 1]
    # Identify those that never touched the ICU
    non_icu_deaths = death_encounters[~death_encounters['icu_enc'].astype(bool)]
    # Count the number of unique encounters with deaths outside of ICU
    num_deaths_outside_icu = non_icu_deaths['encounter_block'].nunique()
    # Calculate total deaths (unique encounter blocks with death)
    total_encounters = all_encounters['encounter_block'].nunique()
    # Calculate the percentage
    pct_deaths_outside_icu = (num_deaths_outside_icu / total_encounters * 100) if total_encounters > 0 else 0
    print(f"Number of deaths outside ICU: {num_deaths_outside_icu} ({pct_deaths_outside_icu:.1f}% of all hospitalizations)")

    # Add ICU encounters to strobe counts as 1_icu_encounters
    num_icu_encounters = all_encounters[all_encounters['icu_enc'] == 1]['encounter_block'].nunique()
    # strobe_counts already initialized at beginning of main()
    strobe_counts['1_icu_encounters'] = num_icu_encounters

    encounter_locations = all_encounters.groupby('encounter_block').agg({
        'death_enc': 'max',
        'icu_enc': 'max',  # Did they ever touch ICU?
        'location_category': lambda x: set(x.dropna().str.lower())  # Set of all locations visited
    }).reset_index()

    encounter_locations['has_procedural_or_ld'] = encounter_locations['location_category'].apply(
            lambda locs: any(loc in {'procedural', 'l&d'} for loc in locs)
        )

    # Step 3: Flag as procedural/L&D only if: 
    # - MUST have procedural or L&D
    # - CANNOT have ICU (icu_enc == 0)
    encounter_locations['is_procedural_ld_only'] = (
        (encounter_locations['icu_enc'] == 0) &
        (encounter_locations['has_procedural_or_ld'] == True)
    ).astype(int)

    # Join has_procedural_or_ld (and is_procedural_ld_only if desired) with all_encounters by encounter_block
    all_encounters = all_encounters.merge(
        encounter_locations[['encounter_block', 'is_procedural_ld_only']],
        on='encounter_block',
        how='left'
    )

    # Start final_cohort from ALL adult encounters — cohort_enc will be defined
    # after high_support_enc and vaso_support_enc are computed.
    if cohort_mode == 'ward':
        _ward_hosp_ids = all_encounters.loc[
            all_encounters['cohort_enc'] == 1, 'hospitalization_id'
        ].unique()
        final_cohort = all_encounters[
            all_encounters['hospitalization_id'].isin(_ward_hosp_ids)
        ][['encounter_block', 'icu_enc', 'death_enc', 'cohort_enc', 'is_procedural_ld_only']].drop_duplicates()
    else:
        final_cohort = all_encounters[
            ['encounter_block', 'icu_enc', 'death_enc', 'is_procedural_ld_only']
        ].drop_duplicates()


    # ==============================================================================
    # # Respiratory Support
    # ==============================================================================

    # ============================================================================
    # STEP 2: Load Respiratory Support and Identify Patients on Advanced Respiratory support 
    # ============================================================================
    print("\n" + "=" * 80)
    print(" Loading Respiratory Support and Identifying IMV Patients")
    print("=" * 80)

    # ── Waterfall cache: CI and ward runs produce identical waterfall output
    #    (same adult_hosp_ids filter, same encounter stitching, same device/mode
    #    pre-filter, same deterministic waterfall).  Cache the post-waterfall
    #    df so the second run can skip the expensive recomputation.  Cache is
    #    invalidated if the raw respiratory_support parquet is newer.
    import types as _types
    _waterfall_cache_path = project_root / 'output/intermediate' / 'respiratory_support_waterfall.parquet'
    _raw_resp_path = Path(config['tables_path']) / f"clif_respiratory_support.{config['file_type']}"
    _cache_valid = (
        _waterfall_cache_path.exists()
        and _raw_resp_path.exists()
        and _waterfall_cache_path.stat().st_mtime >= _raw_resp_path.stat().st_mtime
    )

    if _cache_valid:
        print(f"\nLoading cached waterfall from output/intermediate/{_waterfall_cache_path.name}...")
        _cached_df = pd.read_parquet(_waterfall_cache_path)
        # Scope to this run's adult_hosp_ids defensively (cache may contain
        # a broader scope if it was written by a different configuration).
        _cached_df = _cached_df[
            _cached_df['hospitalization_id'].isin(adult_hosp_ids)
        ].copy()
        # Downstream code only touches `.df` on `clif.respiratory_support`
        # (audited 2026-04-20), so a lightweight namespace shell is enough.
        clif.respiratory_support = _types.SimpleNamespace(df=_cached_df)
        print(f"Respiratory support (cached waterfall): {len(_cached_df):,} rows")
        del _cached_df
    else:
        if _waterfall_cache_path.exists():
            print(f"\nRaw respiratory_support newer than cache — recomputing waterfall.")
        print(f"\nLoading respiratory_support table...")
        # Try to load respiratory support with all columns, but handle gracefully if some don't exist
        try:
            clif.load_table('respiratory_support',
                                   columns=rst_required_columns,
                                   filters={'hospitalization_id': list(adult_hosp_ids)})
            print(f"Respiratory support loaded ({len(clif.respiratory_support.df):,} rows)")
        except Exception as e:
            # If specific columns don't exist, try loading without the optional new columns
            print(f"⚠️ Warning: Could not load all requested columns: {e}")
            print("   Attempting to load with core columns only...")

            # Core columns that should always exist
            core_rst_columns = [col for col in rst_required_columns
                               if col not in ['flow_rate_set']]

            clif.load_table('respiratory_support',
                                   columns=core_rst_columns,
                                   filters={'hospitalization_id': list(adult_hosp_ids)})
            print(f"Respiratory support loaded with core columns ({len(clif.respiratory_support.df):,} rows)")

        # MCIDE collection moved to separate script: generate_mcide_and_stats.py
        # get_value_counts_mcide(clif.respiratory_support, 'respiratory_support', ['device_name', 'device_category'], output_dir=mcide_dir, config=config)
        # get_value_counts_mcide(clif.respiratory_support, 'respiratory_support', ['mode_name', 'mode_category'], output_dir=mcide_dir, config=config)

        # Standardize category columns to lowercase
        print(f"\nStandardizing category columns...")
        category_cols = [col for col in clif.respiratory_support.df.columns if col.endswith('_category')]
        for col in category_cols:
            clif.respiratory_support.df[col] = clif.respiratory_support.df[col].str.lower()

        clif.respiratory_support.df = pd.merge(clif.respiratory_support.df, encounter_mapping,
                                                on='hospitalization_id', how='left')

        # Filter to encounters with at least one qualifying device or mode_category
        # that the waterfall could infer as IMV/NIPPV.  Only these can contribute
        # to high_support_enc; waterfalling non-qualifying encounters is wasted work.
        _device_mask = clif.respiratory_support.df['device_category'].isin(
            ['imv', 'nippv', 'cpap', 'high flow nc']
        )
        _mode_mask = clif.respiratory_support.df['mode_category'].str.contains(
            r'(?:assist control-volume control|simv|pressure control)',
            na=False, regex=True,
        ) if 'mode_category' in clif.respiratory_support.df.columns else pd.Series(False, index=clif.respiratory_support.df.index)
        _enc_with_qualifying = clif.respiratory_support.df.loc[
            _device_mask | _mode_mask, 'encounter_block'
        ].unique()
        print(f"Encounters with qualifying resp devices/modes: {len(_enc_with_qualifying):,}")
        clif.respiratory_support.df = clif.respiratory_support.df[
            clif.respiratory_support.df['encounter_block'].isin(_enc_with_qualifying)
        ].copy()
        del _device_mask, _mode_mask, _enc_with_qualifying
        clif.respiratory_support.df = clif.respiratory_support.df.sort_values(
            ['hospitalization_id', 'recorded_dttm']
        )
        apply_outlier_handling(clif.respiratory_support)

        # Run waterfall (parallel by default, sequential fallback)
        print(f"\nApplying waterfall to respiratory support ({len(clif.respiratory_support.df):,} rows)...")
        try:
            from concurrent.futures import ProcessPoolExecutor
            _enc_blocks = clif.respiratory_support.df['encounter_block'].unique()
            _n_workers = min(os.cpu_count() or 4, max(1, len(_enc_blocks) // 10))
            if _n_workers > 1:
                _chunks = np.array_split(_enc_blocks, _n_workers)
                _df_chunks = [
                    (clif.respiratory_support.df[
                        clif.respiratory_support.df['encounter_block'].isin(c)
                    ].copy(), 'encounter_block')
                    for c in _chunks
                ]
                print(f"  Parallel waterfall: {_n_workers} workers, {len(_enc_blocks)} encounters...")
                with ProcessPoolExecutor(max_workers=_n_workers) as executor:
                    _results = list(executor.map(_waterfall_chunk, _df_chunks))
                clif.respiratory_support.df = pd.concat(_results, ignore_index=True)
            else:
                raise ValueError("Too few encounters for parallelization")
        except Exception as e:
            print(f"  Parallel waterfall unavailable ({e}), using sequential...")
            clif.respiratory_support = clif.respiratory_support.waterfall(
                id_col='encounter_block', verbose=True
            )
        print(f"Waterfall complete: {len(clif.respiratory_support.df):,} rows")

        # Persist for the sibling cohort run (CI → ward, or ward → CI).
        _waterfall_cache_path.parent.mkdir(parents=True, exist_ok=True)
        clif.respiratory_support.df.to_parquet(_waterfall_cache_path, index=False)
        print(f"Cached waterfall → output/intermediate/{_waterfall_cache_path.name}")

    # Identify hospitalizations on advanced respiratory support
    # IMV, NIPPV, CPAP always qualify; HFNC qualifies only at >= 30 LPM
    print(f"\nIdentifying hospitalizations with advanced respiratory support devices...")
    always_mask = clif.respiratory_support.df['device_category'].isin(
        ['imv', 'nippv', 'cpap']
    )
    hfnc_mask = (
        (clif.respiratory_support.df['device_category'] == 'high flow nc')
        & (clif.respiratory_support.df['lpm_set'] >= 30)
    )
    advanced_support_hosp_ids = clif.respiratory_support.df.loc[
        always_mask | hfnc_mask, 'encounter_block'
    ].unique()
    print(f"Hospitalizations with advanced resp. support (IMV/NIPPV/CPAP or HFNC>=30 LPM): {len(advanced_support_hosp_ids):,}")

    # Create a DataFrame with advanced_support_hosp_ids and 'high_support_en' == 1
    advanced_support_df = pd.DataFrame({
        'encounter_block': advanced_support_hosp_ids,
        'high_support_enc': 1
    })

    # Perform a left join to add support flags only to existing cohort encounters
    # (prevents adding encounters without is_procedural_ld_only values)
    final_cohort = final_cohort.merge(
        advanced_support_df,
        on='encounter_block',
        how='left'
    )

    # Identify hospitalizations on NIPPV (BiPAP) or HFNC (>= 30 LPM)
    print(f"\nIdentifying hospitalizations with NIPPV/HFNC...")
    nippv_hfnc_hosp_ids = clif.respiratory_support.df.loc[
        (clif.respiratory_support.df['device_category'] == 'nippv')
        | (
            (clif.respiratory_support.df['device_category'] == 'high flow nc')
            & (clif.respiratory_support.df['lpm_set'] >= 30)
        ),
        'encounter_block'
    ].unique()
    print(f"Hospitalizations with NIPPV/HFNC: {len(nippv_hfnc_hosp_ids):,}")

    nippv_hfnc_df = pd.DataFrame({
        'encounter_block': nippv_hfnc_hosp_ids,
        'nippv_hfnc_enc': 1
    })
    final_cohort = final_cohort.merge(nippv_hfnc_df, on='encounter_block', how='left')

    # Detect respiratory failure onset for PF/SF calculation
    print("Detecting respiratory failure onset for PF/SF ratios...")
    from modules.tableone.pf_sf_calculator import detect_respiratory_failure_onset
    resp_failure_onset_df = detect_respiratory_failure_onset(
        clif.respiratory_support.df,
        cohort_ids=final_cohort['encounter_block'].unique(),
        id_col='encounter_block'
    )
    print(f"Respiratory failure onset detected for {len(resp_failure_onset_df):,} encounters")

    # Memory cleanup: Clear respiratory support intermediate data
    print("Clearing respiratory support intermediate data from memory...")
    del advanced_support_df, nippv_hfnc_df
    gc.collect()
    checkpoint("Respiratory Support Processed")


    # ==============================================================================
    # # Vasoactives
    # ==============================================================================

    print(f"\nLoading medication_admin_continuous table...")
    clif.load_table(
        'medication_admin_continuous',
        columns=meds_required_columns,
        filters={
            'hospitalization_id': list(adult_hosp_ids),
            'med_category': meds_of_interest
        }
    )

    # Identify hospitalizations on advanced mechanical support
    print(f"\nIdentifying hospitalizations with advanced respiratory support devices...")
    vasoactive_meds = ['norepinephrine', 'epinephrine', 'phenylephrine', 'vasopressin','dopamine', 'angiotensin']
    clif.medication_admin_continuous.df= pd.merge(clif.medication_admin_continuous.df, encounter_mapping, 
                                            on='hospitalization_id', how='left')
    vasoactive_hosp_ids = clif.medication_admin_continuous.df.loc[
        clif.medication_admin_continuous.df['med_category'].str.lower().isin([d.lower() for d in vasoactive_meds]),
        'encounter_block'
    ].unique()
    print(f"Hospitalizations with any vasoactives. device ({', '.join(vasoactive_meds).upper()}): {len(vasoactive_hosp_ids):,}")
    # strobe_counts["3_vasoactive_hospitalizations"] = len(vasoactive_hosp_ids)

    # Create a DataFrame with advanced_support_hosp_ids and 'high_support_en' == 1
    vasoactives_df = pd.DataFrame({
        'encounter_block': vasoactive_hosp_ids,
        'vaso_support_enc': 1
    })

    # Join vasoactives_df with final cohort using left merge
    # (prevents adding encounters without is_procedural_ld_only values)
    final_cohort = final_cohort.merge(
        vasoactives_df,
        on='encounter_block',
        how='left'
    )

    # Missing flags mean not on that support type
    final_cohort['vaso_support_enc'] = final_cohort['vaso_support_enc'].fillna(0).astype(int)
    final_cohort['high_support_enc'] = final_cohort['high_support_enc'].fillna(0).astype(int)
    final_cohort['nippv_hfnc_enc'] = final_cohort['nippv_hfnc_enc'].fillna(0).astype(int)

    # Set support flags to 0 if is_procedural_ld_only is 1
    # (procedural/L&D only encounters without ICU should not count as having support).
    # In ward mode, ward-touching is the inclusion criterion, so support metrics for
    # ward+procedural encounters should be reported as-is (the procedural visit doesn't
    # invalidate real vasopressor / advanced respiratory use on the ward).
    if cohort_mode != 'ward':
        final_cohort.loc[final_cohort['is_procedural_ld_only'] == 1, 'high_support_enc'] = 0
        final_cohort.loc[final_cohort['is_procedural_ld_only'] == 1, 'vaso_support_enc'] = 0
        final_cohort.loc[final_cohort['is_procedural_ld_only'] == 1, 'nippv_hfnc_enc'] = 0
    # ── Define cohort and filter ────────────────────────────────────────
    # Critical-illness cohort: ICU stay OR died/hospice OR advanced resp
    # support OR vasopressor support, excluding procedural/L&D-only encounters.
    if cohort_mode != 'ward':
        final_cohort['cohort_enc'] = (
            (final_cohort['icu_enc'] | final_cohort['death_enc']
             | final_cohort['high_support_enc'] | final_cohort['vaso_support_enc'])
            & (~final_cohort['is_procedural_ld_only'].astype(bool))
        ).astype(int)
        _pre = len(final_cohort)
        final_cohort = final_cohort[final_cohort['cohort_enc'] == 1].copy()
        print(f"\nCohort filter: {_pre:,} → {len(final_cohort):,} encounter_blocks "
              f"(icu_enc|death_enc|high_support_enc|vaso_support_enc, excl procedural/L&D-only)")

    # Propagate cohort_enc back to all_encounters so downstream merges
    # (e.g. cohort_df at line ~2014) can pick it up.
    _cohort_map = final_cohort[['encounter_block', 'cohort_enc']].drop_duplicates()
    if 'cohort_enc' in all_encounters.columns:
        all_encounters = all_encounters.drop(columns='cohort_enc')
    all_encounters = all_encounters.merge(_cohort_map, on='encounter_block', how='left')
    all_encounters['cohort_enc'] = all_encounters['cohort_enc'].fillna(0).astype(int)

    cohort_enc_hospitalization_ids = all_encounters[
        all_encounters['encounter_block'].isin(final_cohort['encounter_block'])
    ]['hospitalization_id'].unique().tolist()

    strobe_counts["2_advanced_resp_support_hospitalizations"] = (final_cohort['high_support_enc'] == 1).sum()
    strobe_counts["2b_nippv_hfnc_hospitalizations"] = (final_cohort['nippv_hfnc_enc'] == 1).sum()
    strobe_counts["3_vasoactive_hospitalizations"] = (final_cohort['vaso_support_enc'] == 1).sum()

    # ------------------------------------------------------------------
    # ED-vasopressor sub-strata: identify encounters where the first
    # vasopressor was administered in the ED, then classify by whether
    # the patient subsequently went to ICU or Ward.
    # Must run BEFORE medication data is deleted below.
    # ------------------------------------------------------------------
    _vaso_meds_df = clif.medication_admin_continuous.df[
        clif.medication_admin_continuous.df['med_category'].str.lower().isin(
            [d.lower() for d in vasoactive_meds]
        )
    ]
    # Earliest vasopressor admin_dttm per encounter_block
    first_vaso_time = (
        _vaso_meds_df
        .groupby('encounter_block')['admin_dttm']
        .min()
        .reset_index()
        .rename(columns={'admin_dttm': 'first_vaso_dttm'})
    )
    # ED ADT rows (location_category already lowercased at line ~1542)
    ed_adt = all_encounters[
        all_encounters['location_category'] == 'ed'
    ][['encounter_block', 'in_dttm', 'out_dttm']].copy()

    # Temporal join: first pressor falls within an ED ADT window
    vaso_ed_check = first_vaso_time.merge(ed_adt, on='encounter_block', how='inner')
    vaso_ed_check = vaso_ed_check[
        (vaso_ed_check['first_vaso_dttm'] >= vaso_ed_check['in_dttm']) &
        (vaso_ed_check['first_vaso_dttm'] < vaso_ed_check['out_dttm'])
    ]
    # Keep one row per encounter_block (first matching ED window)
    first_vaso_in_ed_df = (
        vaso_ed_check[['encounter_block', 'first_vaso_dttm']]
        .drop_duplicates(subset='encounter_block')
    )
    ed_vaso_blocks = set(first_vaso_in_ed_df['encounter_block'])
    print(f"Encounters with first vasopressor in ED: {len(ed_vaso_blocks):,}")

    # Post-pressor ADT locations for ED-pressor encounters
    _post_pressor_adt = all_encounters[
        all_encounters['encounter_block'].isin(ed_vaso_blocks)
    ][['encounter_block', 'in_dttm', 'location_category']].merge(
        first_vaso_in_ed_df, on='encounter_block', how='inner'
    )
    _post_pressor_adt = _post_pressor_adt[
        _post_pressor_adt['in_dttm'] > _post_pressor_adt['first_vaso_dttm']
    ]

    # Classify: any post-pressor ICU → ed_icu; any post-pressor ward (no ICU) → ed_ward
    _post_locs = _post_pressor_adt.groupby('encounter_block')['location_category'].agg(set)
    ed_vaso_icu_blocks = set(
        _post_locs[_post_locs.apply(lambda locs: any('icu' in loc for loc in locs))].index
    )
    ed_vaso_ward_blocks = set(
        _post_locs[
            _post_locs.apply(lambda locs: 'ward' in locs and not any('icu' in loc for loc in locs))
        ].index
    )
    print(f"  ED→ICU: {len(ed_vaso_icu_blocks):,}, ED→Ward: {len(ed_vaso_ward_blocks):,}")

    del _vaso_meds_df, ed_adt, vaso_ed_check, _post_pressor_adt, _post_locs

    # Memory cleanup: Clear medication initial load data
    print("Clearing medication initial load data from memory...")
    del vasoactives_df, clif.medication_admin_continuous
    gc.collect()
    checkpoint("Medications Processed")
    # Missing icu_enc means not ICU
    final_cohort['icu_enc'] = final_cohort['icu_enc'].fillna(0).astype(int)
    # Sub-strata of vaso and advanced_resp: recipients of each support type
    # split by whether they ever touched ICU. Built after both
    # vaso_support_enc/high_support_enc and icu_enc are finalized (and the
    # is_procedural_ld_only zero-out at L1727 has already run in
    # critical-illness mode), so each pair of flags partitions its parent
    # cohort exactly.
    final_cohort['vaso_icu_enc'] = (
        (final_cohort['vaso_support_enc'] == 1) & (final_cohort['icu_enc'] == 1)
    ).astype(int)
    final_cohort['vaso_no_icu_enc'] = (
        (final_cohort['vaso_support_enc'] == 1) & (final_cohort['icu_enc'] == 0)
    ).astype(int)
    # ED-vasopressor sub-strata: first pressor in ED, split by post-pressor
    # disposition (any subsequent ICU vs ward-only, excluding encounters with
    # neither ICU nor ward after the ED pressor).
    final_cohort['vaso_ed_icu_enc'] = (
        final_cohort['encounter_block'].isin(ed_vaso_icu_blocks)
    ).astype(int)
    final_cohort['vaso_ed_ward_enc'] = (
        final_cohort['encounter_block'].isin(ed_vaso_ward_blocks)
    ).astype(int)
    _n_ed_icu = final_cohort['vaso_ed_icu_enc'].sum()
    _n_ed_ward = final_cohort['vaso_ed_ward_enc'].sum()
    print(f"ED vaso sub-strata (final cohort): ED→ICU: {_n_ed_icu:,}, ED→Ward: {_n_ed_ward:,}")
    final_cohort['high_support_icu_enc'] = (
        (final_cohort['high_support_enc'] == 1) & (final_cohort['icu_enc'] == 1)
    ).astype(int)
    final_cohort['high_support_no_icu_enc'] = (
        (final_cohort['high_support_enc'] == 1) & (final_cohort['icu_enc'] == 0)
    ).astype(int)
    final_cohort['nippv_hfnc_icu_enc'] = (
        (final_cohort['nippv_hfnc_enc'] == 1) & (final_cohort['icu_enc'] == 1)
    ).astype(int)
    final_cohort['nippv_hfnc_no_icu_enc'] = (
        (final_cohort['nippv_hfnc_enc'] == 1) & (final_cohort['icu_enc'] == 0)
    ).astype(int)
    # "Other critically ill" = died in ED/ward without ICU/vaso/resp escalation.
    # With the broadened cohort (icu|death|high_support|vaso), encounters can have
    # icu_enc==0 and death_enc==0 (ward survivors with resp/vaso support).  Those
    # are captured by their respective support flags, not by other_critically_ill.
    final_cohort['other_critically_ill'] = (
        (final_cohort['death_enc'] == 1) &
        (final_cohort['icu_enc'] == 0) &
        (final_cohort['vaso_support_enc'] == 0) &
        (final_cohort['high_support_enc'] == 0)
    ).astype(int)
    # Ward-only catch-all: ward encounters that survived without any critical-care
    # intervention. Only meaningful in ward mode.
    if cohort_mode == 'ward':
        final_cohort['ward_no_critical_care'] = (
            (final_cohort['death_enc'] == 0) &
            (final_cohort['icu_enc'] == 0) &
            (final_cohort['vaso_support_enc'] == 0) &
            (final_cohort['high_support_enc'] == 0)
        ).astype(int)
    # Calculate the count
    strobe_counts['4_other_critically_ill'] = final_cohort.loc[final_cohort['other_critically_ill'] == 1,
                                                                'encounter_block'].nunique()
    strobe_counts['5_all_critically_ill'] = final_cohort['encounter_block'].nunique()
    if cohort_mode == 'ward':
        strobe_counts['6_ward_no_critical_care'] = final_cohort.loc[
            final_cohort['ward_no_critical_care'] == 1, 'encounter_block'
        ].nunique()


    # ==============================================================================
    # # Summary
    # ==============================================================================

    strobe_counts_df = pd.DataFrame(list(strobe_counts.items()), columns=['count_name', 'count_value'])
    strobe_counts_df.to_csv(os.path.join(output_dir, 'strobe_counts.csv'), index=False)
    # Calculate mortality rates
    mortality_rates = {
        'ICU Hospitalizations': final_cohort.loc[final_cohort['icu_enc'] == 1, 'death_enc'].mean() * 100,
        'Advanced Respiratory Support': final_cohort.loc[final_cohort['high_support_enc'] == 1, 'death_enc'].mean() * 100,
        'Vasoactive Hospitalizations': final_cohort.loc[final_cohort['vaso_support_enc'] == 1, 'death_enc'].mean() * 100,
        'Other Critically Ill': final_cohort.loc[final_cohort['other_critically_ill'] == 1, 'death_enc'].mean() * 100,
        'All Critically Ill Adults': final_cohort['death_enc'].mean() * 100,
    }
    if cohort_mode == 'ward':
        # CONSORT diagram in ward mode reports the ward cohort total mortality under
        # a different label so the figure title and box label are consistent.
        mortality_rates['Ward Cohort'] = final_cohort['death_enc'].mean() * 100
    mortality_rates_df = pd.DataFrame(list(mortality_rates.items()), columns=['count_name', 'count_value'])
    mortality_rates_df.to_csv(os.path.join(output_dir, 'mortality_rates.csv'), index=False)

    cohort_df = encounter_mapping.copy()
    cohort_df = cohort_df[cohort_df['encounter_block'].isin(final_cohort['encounter_block'])]

    # Merge cohort_df with all_encounters on hospitalization_id
    cohort_df = cohort_df.merge(
        all_encounters,
        on=['hospitalization_id', 'encounter_block'],
        how='left',
        suffixes=('', '_allenc')
    )


    def create_consort_diagram(strobe_counts, mortality_rates):
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.set_xlim(-1, 13)
        ax.set_ylim(0, 14)
        ax.axis('off')

        box_style = "round,pad=0.1"
        boxes = {}

        def create_box(x, y, width, height, text, box_id=None, fontsize=10, fontweight='normal'):
            box = FancyBboxPatch(
                (x - width/2, y - height/2), width, height,
                boxstyle=box_style, facecolor='white', edgecolor='black', linewidth=1.5
            )
            ax.add_patch(box)
            ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, fontweight=fontweight, wrap=True)
        
            return {
                'x': x, 'y': y, 'width': width, 'height': height,
                'left': x - width/2, 'right': x + width/2,
                'top': y + height/2, 'bottom': y - height/2
            }

        def create_arrow(from_box, to_box):
            x1, y1 = from_box['x'], from_box['bottom'] - 0.1
            x2, y2 = to_box['x'], to_box['top'] + 0.1
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', lw=2, color='black'))

        diagram_title = 'Ward Cohort' if cohort_mode == 'ward' else 'Cohort'
        ax.text(5, 13, diagram_title, ha='center', va='center', fontsize=16, fontweight='bold')

        # Define and arrange the boxes
        box1 = create_box(5, 12, 3, 0.7,
                          f"Total Hospitalizations\nn = {strobe_counts['0_total_hospitalizations']:,}",
                          'total', fontsize=11, fontweight='bold')

        box2 = create_box(5, 10.5, 3, 0.7,
                          f"Stitched Adult Hospitalizations\nn = {strobe_counts['1b_after_stitching']:,}",
                          'adult', fontsize=11, fontweight='bold')
        create_arrow(box1, box2)

        # Define ICU, respiratory support, vasoactive, and other critically ill categories
        box3_icu = create_box(1, 8, 3, 0.9,
                              f"ICU Hospitalizations\nn = {strobe_counts['1_icu_encounters']:,}\nMortality: {mortality_rates['ICU Hospitalizations']:.2f}%",
                              'icu', fontsize=11, fontweight='bold')

        box3_resp = create_box(4.5, 8, 3, 0.9,
                               f"Advanced Respiratory Support\nn = {strobe_counts['2_advanced_resp_support_hospitalizations']:,}\nMortality: {mortality_rates['Advanced Respiratory Support']:.2f}%",
                               'resp_support', fontsize=11, fontweight='bold')

        box3_vaso = create_box(8, 8, 3, 0.9,
                               f"Vasoactive Hospitalizations\nn = {strobe_counts['3_vasoactive_hospitalizations']:,}\nMortality: {mortality_rates['Vasoactive Hospitalizations']:.2f}%",
                               'vasoactive', fontsize=11, fontweight='bold')

        box3_other = create_box(11.3, 8, 3, 0.9,
                                f"Other Critically Ill\nn = {strobe_counts['4_other_critically_ill']:,}\nMortality: {mortality_rates['Other Critically Ill']:.2f}%",
                                'other', fontsize=11, fontweight='bold')

        create_arrow(box2, box3_icu)
        create_arrow(box2, box3_resp)
        create_arrow(box2, box3_vaso)
        create_arrow(box2, box3_other)

        if cohort_mode == 'ward':
            # In ward mode add the survivor catch-all box
            ward_only_n = strobe_counts.get('6_ward_no_critical_care', 0)
            box3_ward_only = create_box(8, 6.3, 4.5, 0.9,
                f"Ward only (survived, no critical care)\nn = {ward_only_n:,}",
                'ward_only', fontsize=10, fontweight='bold')
            create_arrow(box2, box3_ward_only)
            final_label = 'Ward Cohort'
            final_mortality_key = 'Ward Cohort'
        else:
            final_label = 'All Critically Ill Adults'
            final_mortality_key = 'All Critically Ill Adults'

        # Final aggregate box (label depends on cohort_mode)
        box_final = create_box(5.7, 4.5, 5.2, 1.1,
            f"{final_label}\nn = {final_cohort['encounter_block'].nunique():,}\nMortality: {mortality_rates[final_mortality_key]:.2f}%",
            'cohort_total', fontsize=13, fontweight='bold')

        # Do NOT draw arrows from the four groups to the all critically ill adults box

        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'consort_flow_diagram.png'), dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()


    warnings.filterwarnings('ignore', category=FutureWarning, module='upsetplot')

    # Output directories already created by ensure_output_tree() above

    # Prepare final_cohort data for UpSet and venn plots
    summary_df = final_cohort[['encounter_block', 'icu_enc', 'death_enc', 'high_support_enc', 'vaso_support_enc']].drop_duplicates()

    # Rename columns
    summary_df = summary_df.rename(columns={
        'icu_enc': 'ICU Hospitalizations',
        'death_enc': 'Died',
        'high_support_enc': 'Advanced O2 Support',
        'vaso_support_enc': 'Vasoactive Support'
    })

    # Convert to boolean for UpSet and venn
    summary_df['ICU Hospitalizations'] = summary_df['ICU Hospitalizations'].fillna(0).astype(bool)
    summary_df['Died'] = summary_df['Died'].fillna(0).astype(bool)
    summary_df['Advanced O2 Support'] = summary_df['Advanced O2 Support'].fillna(0).astype(bool)
    summary_df['Vasoactive Support'] = summary_df['Vasoactive Support'].fillna(0).astype(bool)
    summary_df.to_csv(os.path.join(output_dir, 'upset_data.csv'), index=False)

    # ========== UpSet Plot ==========
    fig = plt.figure(figsize=(16, 12))
    upset_data = from_indicators(
        ['ICU Hospitalizations', 'Died', 'Advanced O2 Support', 'Vasoactive Support'], 
        data=summary_df.set_index('encounter_block')
    )

    upset = UpSet(upset_data, 
                  subset_size='count',
                  show_counts=True,
                  sort_by='cardinality',
                  element_size=50,
                  with_lines=True)

    upset.plot(fig=fig)

    plt.subplots_adjust(left=0.2, bottom=0.2, right=0.95, top=0.85, hspace=0.3, wspace=0.3)
    plt.suptitle('Clinical Cohort Intersections', fontsize=16, y=0.95)

    # Adjust font sizes for better readability
    for ax in fig.get_axes():
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(12)

    plt.savefig(os.path.join(figures_dir, 'cohort_intersect_upset_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ========== Venn Diagrams ==========

    # ========== 4-way Venn with venny4py ==========
    print("\n" + "="*80)
    print("Generating 4-way Venn Diagram")
    print("="*80)

    # Create dictionary of sets for venny4py
    sets_dict = {
        'ICU Hospitalizations': set(summary_df.loc[summary_df['ICU Hospitalizations'], 'encounter_block']),
        'Died': set(summary_df.loc[summary_df['Died'], 'encounter_block']),
        'Advanced O2 Support': set(summary_df.loc[summary_df['Advanced O2 Support'], 'encounter_block']),
        'Vasoactive Support': set(summary_df.loc[summary_df['Vasoactive Support'], 'encounter_block'])
    }

    # Create 4-way Venn diagram
    fig = plt.figure(figsize=(12, 10))
    venny4py(sets=sets_dict, dpi=300)
    plt.suptitle('4-way Venn Diagram', fontsize=16, y=0.98)
    plt.savefig(os.path.join(figures_dir, 'venn_all_4_groups.png'), dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"✅ Saved: {os.path.join(figures_dir, 'venn_all_4_groups.png')}")

    # Create the diagram
    create_consort_diagram(strobe_counts, mortality_rates)

    # Memory cleanup: Clear CONSORT diagram data
    print("Clearing CONSORT diagram data from memory...")
    plt.close('all')
    del sets_dict, summary_df
    gc.collect()
    checkpoint("CONSORT Diagram Created")


    # ==============================================================================
    # # Demographics
    # ==============================================================================

    # 1. Filter patient table to cohort
    patient_df = clif.patient.df.copy()
    patient_df = patient_df[['patient_id', 'race_category', 'ethnicity_category',
                             'sex_category', 'death_dttm']]

    # Filter patient_df to those in cohort
    patient_df = patient_df[patient_df['patient_id'].isin(cohort_df['patient_id'])]
    # Merge patient_df with cohort_df on patient_id
    cohort_df = patient_df.merge(cohort_df, on='patient_id', how='right')


    # ==============================================================================
    # # Final cohort df
    # ==============================================================================

    final_tableone_df = cohort_df[['patient_id', 'hospitalization_id', 'encounter_block', 'admission_dttm', 
                                    'discharge_dttm', 'age_at_admission', 'discharge_category', 'admission_type_category',
                                    'race_category', 'ethnicity_category', 'sex_category', 'death_dttm', 
                                    'icu_enc', 'death_enc', 'cohort_enc']].drop_duplicates()

    final_cohort_merge_cols = ['encounter_block',
                               'high_support_enc',
                               'high_support_icu_enc', 'high_support_no_icu_enc',
                               'nippv_hfnc_enc',
                               'nippv_hfnc_icu_enc', 'nippv_hfnc_no_icu_enc',
                               'vaso_support_enc',
                               'vaso_icu_enc', 'vaso_no_icu_enc',
                               'vaso_ed_icu_enc', 'vaso_ed_ward_enc',
                               'other_critically_ill']
    if cohort_mode == 'ward':
        final_cohort_merge_cols.append('ward_no_critical_care')
    final_tableone_df = final_tableone_df.merge(
        final_cohort[final_cohort_merge_cols],
        on='encounter_block',
        how="outer"
    )

    # Merge hospital_id onto final_tableone_df by hospitalization_id
    final_tableone_df = final_tableone_df.merge(
        cohort_df[['hospitalization_id', 'hospital_id']].drop_duplicates(),
        on='hospitalization_id',
        how='left'
    )

    # Merge first-pressor-in-ED timestamp for timing metric in stratified tables
    final_tableone_df = final_tableone_df.merge(
        first_vaso_in_ed_df[['encounter_block', 'first_vaso_dttm']],
        on='encounter_block',
        how='left'
    )

    adt_cohort = cohort_df[['patient_id', 'hospitalization_id', 'encounter_block',
                           'hospital_id', 'in_dttm', 'out_dttm', 'location_category',
                           'location_type']].drop_duplicates()

    final_hosp_ids = final_tableone_df['hospitalization_id'].unique().tolist()
    final_patient_ids = final_tableone_df['patient_id'].unique().tolist()
    final_enc_blocks = final_tableone_df['encounter_block'].unique().tolist()


    # ==============================================================================
    # # Cohort Group Sankey
    # ==============================================================================

    outcome_df = create_outcome_df(final_tableone_df)

    print("\n" + "="*80)
    print("CREATING MATPLOTLIB SANKEY DIAGRAMS")
    print("="*80)

    # Example 1: ICU encounters
    # icu_blocks = final_tableone_df[final_tableone_df['icu_enc'] == 1]['encounter_block'].sample(
    #     min(300, final_tableone_df[final_tableone_df['icu_enc'] == 1].shape[0]), 
    #     random_state=42
    # ).tolist()
    # fig1, df1 = create_sankey_diagram(
    #     adt_cohort=adt_cohort,
    #     encounter_blocks=icu_blocks,
    #     outcome_df=outcome_df,
    #     max_locations=7,
    #     title="ICU Patient Flow and Outcomes (Sample 300 encounters)",
    #     output_file=f"{figures_dir}/sankey_matplotlib_icu.png",
    #     figsize=(16, 8)
    # )
    # 
    icu_blocks_all = final_tableone_df[final_tableone_df['icu_enc'] == 1]['encounter_block'].tolist()
    fig1, df1 = create_sankey_diagram(
        adt_cohort=adt_cohort,
        encounter_blocks=icu_blocks_all,
        outcome_df=outcome_df,
        max_locations=7,
        title="ICU Patient Flow and Outcomes",
        output_file=f"{figures_dir}/sankey_matplotlib_icu.png",
        figsize=(16, 8)
    )


    # other critically ill
    other_enc = final_tableone_df[final_tableone_df['other_critically_ill'] == 1]['encounter_block'].tolist()

    print(f"\n Other critically ill  (N={len(other_enc)})")
    fig1, df1 = create_sankey_diagram(
        adt_cohort=adt_cohort,
        encounter_blocks=other_enc,
        outcome_df=outcome_df,
        max_locations=8,
        title="Other critically ill patients Patient Flow and Outcomes",
        output_file=f"{figures_dir}/sankey_matplotlib_others.png",
        figsize=(16, 8)
    )


    # other critically ill
    adv_o2_sup = final_tableone_df[final_tableone_df['high_support_enc'] == 1]['encounter_block'].tolist()
    fig1, df1 = create_sankey_diagram(
        adt_cohort=adt_cohort,
        encounter_blocks=adv_o2_sup,
        outcome_df=outcome_df,
        max_locations=8,
        title="Patients receiving advanced o2 support",
        output_file=f"{figures_dir}/sankey_matplotlib_high_o2_support.png",
        figsize=(16, 8)
    )


    # other critically ill
    vaso_sup = final_tableone_df[final_tableone_df['vaso_support_enc'] == 1]['encounter_block'].tolist()
    fig1, df1 = create_sankey_diagram(
        adt_cohort=adt_cohort,
        encounter_blocks=vaso_sup,
        outcome_df=outcome_df,
        max_locations=8,
        title="Patients receiving Vasoactives",
        output_file=f"{figures_dir}/sankey_matplotlib_vaso_support.png",
        figsize=(16, 8)
    )


    # ==============================================================================
    # Sankey for Interventions Starting in Procedural/Other
    # ==============================================================================

    # Get first location for each encounter
    first_location_per_encounter = adt_cohort.sort_values(['encounter_block', 'in_dttm']).groupby('encounter_block').first().reset_index()

    # High support encounters starting in procedural or other
    adv_o2_sup_proc_other = final_tableone_df[
        (final_tableone_df['high_support_enc'] == 1) &
        (final_tableone_df['encounter_block'].isin(
            first_location_per_encounter[
                first_location_per_encounter['location_category'].isin(['procedural', 'other'])
            ]['encounter_block']
        ))
    ]['encounter_block'].tolist()

    print(f"\nAdvanced O2 Support starting in Procedural/Other (N={len(adv_o2_sup_proc_other)})")

    if len(adv_o2_sup_proc_other) > 0:
        fig1, df1 = create_sankey_diagram(
            adt_cohort=adt_cohort,
            encounter_blocks=adv_o2_sup_proc_other,
            outcome_df=outcome_df,
            max_locations=8,
            title="Advanced O2 Support: Starting in Procedural/Other",
            output_file=f"{figures_dir}/sankey_matplotlib_high_o2_proc_other.png",
            figsize=(16, 8)
        )
    else:
        print("  ⚠️ No encounters found")


    # Vasoactive support encounters starting in procedural or other
    vaso_sup_proc_other = final_tableone_df[
        (final_tableone_df['vaso_support_enc'] == 1) &
        (final_tableone_df['encounter_block'].isin(
            first_location_per_encounter[
                first_location_per_encounter['location_category'].isin(['procedural', 'other'])
            ]['encounter_block']
        ))
    ]['encounter_block'].tolist()

    print(f"\nVasoactive Support starting in Procedural/Other (N={len(vaso_sup_proc_other)})")

    if len(vaso_sup_proc_other) > 0:
        fig1, df1 = create_sankey_diagram(
            adt_cohort=adt_cohort,
            encounter_blocks=vaso_sup_proc_other,
            outcome_df=outcome_df,
            max_locations=8,
            title="Vasoactive Support: Starting in Procedural/Other",
            output_file=f"{figures_dir}/sankey_matplotlib_vaso_proc_other.png",
            figsize=(16, 8)
        )
    else:
        print("  ⚠️ No encounters found")


    # ==============================================================================
    # # Hospital and ICU Admission Summary
    #
    # 1. Get the first ICU dttm for ICU encounters 
    # 2. Calculate ICU LOS and Hospital LOS for each encounter in days. 
    # ==============================================================================

    hosp_admission_summary = (
            adt_cohort
            .groupby('encounter_block')
            .agg(
                min_in_dttm = ('in_dttm', 'min'),
                max_out_dttm = ('out_dttm', 'max'),
                first_admission_location = ('location_category', 'first')
            )
    )
    hosp_admission_summary['hospital_length_of_stay_days'] = (
        (hosp_admission_summary['max_out_dttm'] - hosp_admission_summary['min_in_dttm']) / pd.Timedelta(days=1))

    # lowercase the column, not the entire df
    adt_cohort['location_category'] = (
        adt_cohort['location_category']
        .str.lower()
    )

    # restrict to ICU rows
    icu_df = adt_cohort.query('location_category == "icu"')

    # find first ICU in time per 'encounter_block'
    first_in = (
        icu_df
         .groupby('encounter_block', as_index=False)
         .agg(first_icu_in_dttm=('in_dttm', 'min'))
    )

    # join back to pull the matching out_dttm
    icu_summary = (
        first_in
          # bring in that one row’s out_dttm
          .merge(
              icu_df[['hospitalization_id','in_dttm','out_dttm', 'encounter_block']],
              left_on=['encounter_block', 'first_icu_in_dttm'],
              right_on=['encounter_block', 'in_dttm'],
              how='left'
          )
          .rename(columns={'out_dttm':'first_icu_out_dttm'})
    )

    # compute LOS in days (out - in)
    icu_summary['first_icu_los_days'] = (
        (icu_summary['first_icu_out_dttm'] - icu_summary['first_icu_in_dttm'])
        .dt.total_seconds()
        / (3600 * 24)
    )

    # trim to just the columns you need
    icu_summary = icu_summary[['encounter_block', 'first_icu_in_dttm',
                               'first_icu_out_dttm','first_icu_los_days']]

    # Count ICU episodes per encounter_block using the full ADT timeline.
    # An ICU row starts a NEW episode when a "new-episode" location_category
    # appeared between it and the previous ICU row in the same encounter_block.
    #   same episode (collapse): icu (lateral transfer), procedural, radiology, dialysis
    #   new episode (readmit):   ward, stepdown, l&d, hospice, psych, rehab, other
    #   skipped:                 ed (treated as if the row weren't there)
    NEW_EPISODE_LOCS = {'ward', 'stepdown', 'l&d', 'hospice', 'psych', 'rehab', 'other'}

    adt_sorted = adt_cohort.sort_values(['encounter_block', 'in_dttm', 'out_dttm']).copy()
    adt_sorted['triggers_new_episode'] = adt_sorted['location_category'].isin(NEW_EPISODE_LOCS)
    adt_sorted['trigger_cumsum'] = (
        adt_sorted.groupby('encounter_block')['triggers_new_episode'].cumsum()
    )

    icu_rows = adt_sorted[adt_sorted['location_category'] == 'icu'].copy()
    icu_rows['prev_trigger_cumsum'] = (
        icu_rows.groupby('encounter_block')['trigger_cumsum'].shift(1)
    )
    icu_rows['new_episode'] = (
        icu_rows['prev_trigger_cumsum'].isna()
        | (icu_rows['trigger_cumsum'] > icu_rows['prev_trigger_cumsum'])
    )
    icu_episodes = (
        icu_rows.groupby('encounter_block')['new_episode']
        .sum()
        .astype(int)
        .reset_index(name='icu_episodes')
    )
    print(f"   ICU episodes computed: {icu_episodes['icu_episodes'].sum():,} across {len(icu_episodes):,} encounters")

    # Merge all_ids with icu_summary, icu_episodes, and hosp_admission_summary
    final_tableone_df = (
        final_tableone_df
        .merge(icu_summary, on='encounter_block', how='left')
        .merge(icu_episodes, on='encounter_block', how='left')
        .merge(hosp_admission_summary, on='encounter_block', how='left')
    )
    final_tableone_df['icu_episodes'] = final_tableone_df['icu_episodes'].fillna(0).astype(int)
    final_tableone_df['first_admission_location'] = final_tableone_df['first_admission_location'].fillna('Missing')


    # ==============================================================================
    # # Code Status
    # ==============================================================================

    # ----------------------------------------------------------------------------
    # Load Code Status
    # ----------------------------------------------------------------------------
    print(f"\nLoading code_status table...")
    clif.load_table(
        'code_status'
    )

    # MCIDE collection moved to separate script: generate_mcide_and_stats.py
    # get_value_counts_mcide(clif.code_status, 'code_status', ['code_status_name', 'code_status_category'], output_dir=mcide_dir, config=config)
    print(f"   code_status loaded: {len(clif.code_status.df):,} rows")
    print(f"   Unique code_status categories: {clif.code_status.df['code_status_category'].nunique()}")
    print(f"   Unique code_status patients: {clif.code_status.df['patient_id'].nunique()}")

    # Take the last code_status_category for each patient_id
    code_status_latest = (
        clif.code_status.df.sort_values(['patient_id', 'start_dttm'])
        .groupby('patient_id', as_index=False)
        .last()[['patient_id', 'code_status_category']]
        .rename(columns={'code_status_category': 'last_code_status_category'})
    )

    # Merge with final_tableone_df on patient_id
    final_tableone_df = final_tableone_df.merge(code_status_latest, on='patient_id', how='left')

    # ============================================================================
    # Prepare Aggregated Data for Code Status Visualizations
    # ============================================================================

    encounter_flags = ['icu_enc', 'high_support_enc', 'vaso_support_enc', 'other_critically_ill']
    flag_labels = ['ICU Encounters', 'Advanced Respiratory Support', 'Vasoactive Support', 'Other Critically Ill']

    # Initialize containers
    code_status_counts = {}
    code_status_percentages = {}
    missingness_info = {}

    # Collect aggregated data for each encounter type
    for flag, label in zip(encounter_flags, flag_labels):
        subset = final_tableone_df[final_tableone_df[flag] == 1]
    
        # Total encounters for this type
        total_encounters = len(subset)
    
        # Count missing values
        n_missing = subset['last_code_status_category'].isna().sum()
    
        # Get value counts (including handling of NaN)
        counts = subset['last_code_status_category'].value_counts(dropna=False)
    
        # Store counts
        code_status_counts[label] = counts
    
        # Calculate percentages
        percentages = (counts / total_encounters * 100).round(2)
        code_status_percentages[label] = percentages
    
        # Store missingness information
        missingness_info[label] = {
            'total_encounters': total_encounters,
            'n_missing': n_missing,
            'pct_missing': round(n_missing / total_encounters * 100, 2) if total_encounters > 0 else 0
        }

    # ============================================================================
    # Create DataFrames for Export
    # ============================================================================

    # 1. Counts DataFrame
    df_counts = pd.DataFrame(code_status_counts).fillna(0).astype(int)
    df_counts.index.name = 'code_status_category'

    # Handle NaN index (if exists)
    if df_counts.index.isna().any():
        df_counts.index = df_counts.index.fillna('Missing')

    # 2. Percentages DataFrame
    df_percentages = pd.DataFrame(code_status_percentages).fillna(0)
    df_percentages.index.name = 'code_status_category'

    if df_percentages.index.isna().any():
        df_percentages.index = df_percentages.index.fillna('Missing')

    # 3. Missingness Summary DataFrame
    df_missingness = pd.DataFrame(missingness_info).T
    df_missingness.index.name = 'encounter_type'

    # ============================================================================
    # Add Summary Statistics
    # ============================================================================

    # Add row totals to counts
    df_counts['Total'] = df_counts.sum(axis=1)

    # Add column totals to counts
    df_counts.loc['Total'] = df_counts.sum(axis=0)

    # Add summary to percentages (column sums should be ~100%)
    df_percentages.loc['Total'] = df_percentages.sum(axis=0)

    # ============================================================================
    # Save to CSV Files
    # ============================================================================

    # Save counts
    _counts_path = os.path.join(output_dir, 'code_status_counts_by_encounter_type.csv')
    df_counts.to_csv(_counts_path)
    print(f"✅ Saved: {_counts_path}")

    # Save percentages
    _pct_path = os.path.join(output_dir, 'code_status_percentages_by_encounter_type.csv')
    df_percentages.to_csv(_pct_path)
    print(f"✅ Saved: {_pct_path}")

    # Save missingness summary
    _miss_path = os.path.join(output_dir, 'code_status_missingness_summary.csv')
    df_missingness.to_csv(_miss_path)
    print(f"✅ Saved: {_miss_path}")

    # ============================================================================
    # Create Combined Summary File (Optional)
    # ============================================================================

    # Create a comprehensive summary with counts and percentages
    combined_summary = []

    for col in df_counts.columns[:-1]:  # Exclude 'Total' column
        for idx in df_counts.index[:-1]:  # Exclude 'Total' row
            count = df_counts.loc[idx, col]
            pct = df_percentages.loc[idx, col]
            combined_summary.append({
                'encounter_type': col,
                'code_status': idx,
                'count': count,
                'percentage': pct
            })

    df_combined = pd.DataFrame(combined_summary)
    _combined_path = os.path.join(output_dir, 'code_status_combined_summary.csv')
    df_combined.to_csv(_combined_path, index=False)
    print(f"✅ Saved: {_combined_path}")

    print("\n" + "="*80)
    print("SAVED AGGREGATED DATA (NO PATIENT-LEVEL INFORMATION)")
    print("="*80)
    print(f"1. {_counts_path}")
    print(f"2. {_pct_path}")
    print(f"3. {_miss_path}")
    print(f"4. {_combined_path}")

    # ============================================================================
    # Load the saved aggregated data (can be done in a separate session)
    # ============================================================================

    # Remove 'Total' rows/columns for visualization, but only if 'Total' exists
    def drop_total_and_missing(axis_df):
        # Remove 'Total' and 'Missing' rows/cols if they exist
        for axis in [0, 1]:
            if 'Total' in axis_df.axes[axis]:
                axis_df = axis_df.drop('Total', axis=axis)
            if 'Missing' in axis_df.axes[axis]:
                axis_df = axis_df.drop('Missing', axis=axis)
        return axis_df

    def drop_total_keep_missing(axis_df):
        # Remove only 'Total' rows/cols, but keep 'Missing'
        for axis in [0, 1]:
            if 'Total' in axis_df.axes[axis]:
                axis_df = axis_df.drop('Total', axis=axis)
        return axis_df

    # For counts, we'll keep the "Missing" row if present for plotting, for percentages we'll recompute without it
    df_counts_viz = drop_total_keep_missing(df_counts)

    # Recalculate percentages excluding 'Missing' category from denominator
    def recalc_percentages_exclude_missing(df_counts):
        # Only keep rows that are not 'Total' (already handled), and not 'Missing'
        code_status_rows = [row for row in df_counts.index if row != 'Missing']
        # For each column, divide counts by sum excluding 'Missing'
        df_pct = df_counts.loc[code_status_rows].div(df_counts.loc[code_status_rows].sum(axis=0), axis=1) * 100
        return df_pct

    df_pct_viz = recalc_percentages_exclude_missing(df_counts_viz)

    # ============================================================================
    # VISUALIZATION 1: Stacked Bar Chart with Missingness Indicator
    # ============================================================================

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Define colors (use different color for Missing)
    status_categories = [row for row in df_counts_viz.index if row != 'Missing']
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

    # If 'Missing' exists, assign it a distinct color at the end for plotting counts
    if 'Missing' in df_counts_viz.index:
        status_categories_full = status_categories + ['Missing']
        colors_full = colors + ['#808080']  # Gray for missing
    else:
        status_categories_full = status_categories
        colors_full = colors

    # Plot 1: Absolute counts (stacked bars including Missing if present)
    df_counts_viz.T[status_categories_full].plot(
        kind='bar',
        stacked=True,
        ax=ax1,
        color=colors_full[:len(status_categories_full)],
        edgecolor='black',
        linewidth=0.5
    )
    ax1.set_title('Code Status Distribution by Encounter Type\n(Absolute Counts)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax1.set_xlabel('Encounter Type', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.legend(title='Code Status', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

    # Add missingness annotations
    for i, col in enumerate(df_counts_viz.columns):
        if col in df_missingness.index:
            miss_pct = df_missingness.loc[col, 'pct_missing']
            if miss_pct > 0:
                ax1.text(
                    i, ax1.get_ylim()[1] * 0.95, f'{miss_pct:.1f}% missing',
                    ha='center', va='top', fontsize=9, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
                )

    # Plot 2: Percentages (excluding Missing from numerator/denominator)
    df_pct_viz.T[status_categories].plot(
        kind='bar',
        stacked=True,
        ax=ax2,
        color=colors[:len(status_categories)],
        edgecolor='black',
        linewidth=0.5
    )
    ax2.set_title('Code Status Distribution by Encounter Type\n(Percentages, Excl. Missing)', 
                  fontsize=14, fontweight='bold', pad=20)
    ax2.set_xlabel('Encounter Type', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Percentage', fontsize=12, fontweight='bold')
    ax2.legend(title='Code Status', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'code_status_stacked_bar_with_missingness_excl_missing_cat.png'),
                dpi=300, bbox_inches='tight')
    plt.close()



    # ==============================================================================
    # ADT Location Coverage (overall cohort only)
    # ==============================================================================
    if cohort_mode != 'ward':
        from modules.tableone.adt_coverage import run_adt_location_coverage
        try:
            run_adt_location_coverage(
                clif=clif,
                hospitalization_ids=final_hosp_ids,
                output_csv_dir=output_dir,
                output_fig_dir=figures_dir,
            )
        except Exception as e:
            print(f"[WARN] ADT location coverage analysis failed: {e}")


    # ==============================================================================
    # # IMV encounters
    # ==============================================================================

    # IMV / respiratory characteristics processing computes vent hours / IMV
    # episodes / first-location-at-IMV and runs first-24h ventilator settings
    # analysis (tidal volume curves, pressure control curves, mode proportions).
    # The waterfall already ran earlier (before the high_support_enc flag) so
    # respiratory_support data is already filled. For ward mode the user opted
    # out of these characteristics for memory + time.
    if cohort_mode == 'ward':
        print("\nSkipping IMV / respiratory support characteristics block (ward mode)")

    for _imv_iter in (range(1) if cohort_mode != 'ward' else range(0)):
        # Filter to final cohort hospitalizations (waterfall already applied earlier)
        clif.respiratory_support.df = clif.respiratory_support.df[
            clif.respiratory_support.df['hospitalization_id'].isin(final_hosp_ids)
        ].copy()
        print(f"Respiratory support rows (final cohort): {len(clif.respiratory_support.df):,}")
        clif.respiratory_support.df = clif.respiratory_support.df.sort_values(['hospitalization_id', 'recorded_dttm'])

        resp_stitched = clif.respiratory_support.df

        # Write detailed vent hours debug log to intermediate output
        _vent_log_path = project_root / 'output' / 'intermediate' / 'vent_hours_debug.log'
        os.makedirs(_vent_log_path.parent, exist_ok=True)
        _vent_log = open(_vent_log_path, 'w')
        _vent_log.write("=" * 80 + "\n")
        _vent_log.write("VENTILATOR HOURS COMPUTATION DEBUG LOG\n")
        _vent_log.write("=" * 80 + "\n\n")

        #  Identify IMV rows
        imv_mask = resp_stitched['device_category'].str.contains("imv", case=False, na=False)
        resp_stitched_imv = resp_stitched[imv_mask].copy()

        # Create on_vent column for IMV records
        resp_stitched_imv['on_vent'] = 1

        # Get unique encounter IDs from resp_stitched_imv
        imv_encounters = resp_stitched_imv['encounter_block'].unique()

        # Log sample encounter for debugging
        _log_sample_enc = imv_encounters[0] if len(imv_encounters) > 0 else resp_stitched['encounter_block'].iloc[0]
        sample = resp_stitched[resp_stitched['encounter_block'] == _log_sample_enc][['encounter_block', 'recorded_dttm', 'device_category', 'mode_category']].head(30)
        _vent_log.write(f"--- Post-waterfall sample (encounter: {_log_sample_enc}) ---\n")
        _vent_log.write(sample.to_string(index=False) + "\n")
        _vent_log.write("---\n\n")

        print(f"Number of IMV encounters: {len(imv_encounters):,}")
        strobe_counts["IMV encounters"] = len(imv_encounters)
        # Determine Vent Start/End for Each Encounter
        vent_start_end = resp_stitched_imv.groupby('encounter_block').agg(
            vent_start_time=('recorded_dttm', 'min'),
            vent_end_time=('recorded_dttm', 'max')
        ).reset_index()

        #  Add on_vent flag to final_cohort
        final_tableone_df = final_tableone_df.merge(
            resp_stitched_imv[['encounter_block', 'on_vent']].drop_duplicates(),
            on='encounter_block',
            how='left'
        )
        final_tableone_df['on_vent'] = final_tableone_df['on_vent'].fillna(0).astype(int)

        # ============================================================================
        # Compute accurate IMV hours from waterfall time-series
        # Each row's device is "in effect" until the next observation
        # ============================================================================
        print("\nComputing IMV hours from waterfall data...")
        resp_stitched = resp_stitched.sort_values(['encounter_block', 'recorded_dttm'])

        # Step 1: Get next observation timestamp within each encounter
        resp_stitched['next_recorded_dttm'] = (
            resp_stitched.groupby('encounter_block')['recorded_dttm'].shift(-1)
        )

        # Step 2: Duration this device was "in effect"
        resp_stitched['duration_hours'] = (
            (resp_stitched['next_recorded_dttm'] - resp_stitched['recorded_dttm'])
            .dt.total_seconds() / 3600
        )

        # Log duration calculation for same IMV encounter as waterfall sample
        if len(imv_encounters) > 0:
            sample_dur = resp_stitched[resp_stitched['encounter_block'] == _log_sample_enc][
                ['encounter_block', 'recorded_dttm', 'device_category', 'next_recorded_dttm', 'duration_hours']
            ].head(30)
            _vent_log.write(f"--- Duration calc sample (same encounter: {_log_sample_enc}) ---\n")
            _vent_log.write(sample_dur.to_string(index=False) + "\n")
            _vent_log.write("---\n\n")

        # Step 3: Sum duration only for IMV rows
        imv_hours_per_enc = (
            resp_stitched[resp_stitched['device_category'].str.contains("imv", case=False, na=False)]
            .groupby('encounter_block')['duration_hours']
            .sum()
            .reset_index(name='vent_duration_hours')
        )

        # Log per-encounter IMV hours
        _vent_log.write(f"--- IMV hours per encounter (first 20) ---\n")
        _vent_log.write(imv_hours_per_enc.head(20).to_string(index=False) + "\n")
        _vent_log.write(f"\nTotal encounters with IMV hours: {len(imv_hours_per_enc):,}\n")
        _vent_log.write(f"Total IMV hours across all encounters: {imv_hours_per_enc['vent_duration_hours'].sum():,.0f}\n")
        _vent_log.write("---\n\n")

        # Log summary stats
        _vent_log.write("--- Summary Statistics ---\n")
        _vent_log.write(f"IMV hours distribution per encounter:\n")
        _vent_log.write(imv_hours_per_enc['vent_duration_hours'].describe().to_string() + "\n")
        _vent_log.write("---\n")
        _vent_log.close()
        print(f"   Vent hours debug log written to: {_vent_log_path}")

        # NOTE: IMV episode counting removed pending a proper extubation definition.

        # Merge into final_tableone_df
        final_tableone_df = final_tableone_df.merge(imv_hours_per_enc, on='encounter_block', how='left')
        final_tableone_df['vent_duration_hours'] = final_tableone_df['vent_duration_hours'].fillna(0.0)

        # Cleanup temp columns
        resp_stitched.drop(columns=['next_recorded_dttm', 'duration_hours'], inplace=True)

        # ==============================================================================
        # ## Intubation / extubation event detection
        # Two-lookback / two-lookforward pattern on device_category (clifpy #124)
        # ==============================================================================
        from modules.tableone.extubation_calculator import detect_intubation_extubation

        extub_per_enc, extub_per_episode = detect_intubation_extubation(
            resp_stitched, final_tableone_df, id_col='encounter_block'
        )

        # Drop any existing columns from a prior run before merging (idempotent)
        _extub_cols = [c for c in extub_per_enc.columns if c != 'encounter_block']
        _dup = [c for c in _extub_cols if c in final_tableone_df.columns]
        if _dup:
            final_tableone_df = final_tableone_df.drop(columns=_dup)
        final_tableone_df = final_tableone_df.merge(
            extub_per_enc, on='encounter_block', how='left'
        )

        # Save per-episode table for downstream repeated-measures analyses
        _imv_ep_path = project_root / 'output' / 'intermediate' / 'imv_episodes.csv'
        os.makedirs(_imv_ep_path.parent, exist_ok=True)
        extub_per_episode.to_csv(_imv_ep_path, index=False)
        print(f"  Per-episode IMV data written to: {_imv_ep_path}")

        # ==============================================================================
        # ## Ventilated aggregates (CLIF-TableOne issue #10)
        # KM curve + daily min P/F and S/F post-intubation; PNGs + cross-site CSVs
        # ==============================================================================
        from modules.tableone.extubation_plots import (
            plot_km_time_to_extubation,
            plot_min_pf_sf_per_day_post_intubation,
        )

        from modules.utils.output_paths import ventilated_aggregates_dir
        _vent_agg_dir = ventilated_aggregates_dir()
        _vent_agg_dir.mkdir(parents=True, exist_ok=True)

        try:
            plot_km_time_to_extubation(
                final_tableone_df,
                figures_dir=figures_dir,
                csv_dir=str(_vent_agg_dir),
            )
        except Exception as e:
            print(f"  ⚠️ KM plot failed: {e}")
            import traceback; traceback.print_exc()

        try:
            plot_min_pf_sf_per_day_post_intubation(
                final_tableone_df,
                data_directory=config['tables_path'],
                filetype=config['file_type'],
                timezone=config['timezone'],
                figures_dir=figures_dir,
                csv_dir=str(_vent_agg_dir),
            )
        except Exception as e:
            print(f"  ⚠️ Daily PF/SF plot failed: {e}")
            import traceback; traceback.print_exc()


        # ==============================================================================
        # ## Initial Mode category
        # ==============================================================================

        #  Get Initial Mode Category (first mode after vent start)
        # Subset resp_stitched to only those encounters on IMV
        resp_imv = resp_stitched[resp_stitched['encounter_block'].isin(imv_encounters)].copy()

        # Merge in the vent_start_time
        resp_imv = resp_imv.merge(
            vent_start_end[['encounter_block', 'vent_start_time']],
            on='encounter_block',
            how='left'
        )

        # Filter to only rows at or after vent start
        resp_post_start = resp_imv[
            resp_imv['recorded_dttm'] >= resp_imv['vent_start_time']
        ]

        # Group and take first non-NA mode_category per encounter
        initial_modes = (
            resp_post_start
            .sort_values(['encounter_block', 'recorded_dttm'])
            .groupby('encounter_block', as_index=False)['mode_category']
            .first()
            .rename(columns={'mode_category': 'initial_mode_category'})
        )

        # Fill any entirely-missing groups with "Missing"
        initial_modes['initial_mode_category'] = initial_modes['initial_mode_category'].fillna('Missing')

        # Merge back onto final_cohort
        final_tableone_df = final_tableone_df.merge(
            initial_modes,
            on='encounter_block',
            how='left'
        )

        # If some encounters never went on vent, fill those too
        final_tableone_df['initial_mode_category'] = final_tableone_df['initial_mode_category'].fillna('Missing')

        # Memory cleanup: Clear ventilation mode intermediates (keep vent_start_end for later use)
        print("Clearing ventilation mode intermediate data from memory...")
        del resp_imv, resp_post_start, initial_modes
        gc.collect()
        checkpoint("Ventilation Modes Processed")

        #  Calculate Ventilator Settings Statistics (Median and IQR)
        # Filter resp_stitched to only those encounters on IMV
        # resp_stitched_final = resp_stitched[resp_stitched['encounter_block'].isin(imv_encounters)]

        # # Define numeric columns to aggregate
        # numeric_cols = [
        #     'fio2_set', 'lpm_set', 'resp_rate_set', 'peep_set',
        #     'tidal_volume_set', 'pressure_control_set', 'pressure_support_set'
        # ]

        # # Build named aggregation dict
        # named_aggs = {}
        # for col in numeric_cols:
        #     named_aggs[f'{col}_median'] = (col, 'median')
        #     named_aggs[f'{col}_q1'] = (col, lambda x: x.quantile(0.25))
        #     named_aggs[f'{col}_q3'] = (col, lambda x: x.quantile(0.75))

        # # Aggregate ventilator settings
        # vent_stats = (
        #     resp_stitched_final
        #     .groupby('encounter_block', as_index=False)
        #     .agg(**named_aggs)
        # )

        # # Merge vent stats back to final_cohort
        # final_tableone_df = final_tableone_df.merge(vent_stats, on='encounter_block', how='left')

        #  Find First Location at IMV Start (closest ADT location to vent start)
        # Get minimal ADT cohort with required columns and merge with encounter_block
        print("Find First Location at IMV Start (closest ADT location to vent start)")

        # Merge with vent start times
        adt_vent = pd.merge(
            vent_start_end[['encounter_block', 'vent_start_time']],
            adt_cohort,
            on='encounter_block'
        )

        # Calculate time difference between vent start and ADT in_dttm
        adt_vent['time_diff'] = abs(adt_vent['vent_start_time'] - adt_vent['in_dttm'])

        # Get the closest ADT row for each encounter block
        closest_adt = (
            adt_vent
            .sort_values('time_diff')
            .groupby('encounter_block')
            .first()
            .reset_index()
        )
        closest_adt = closest_adt.rename(columns={'location_category': 'first_location_imv'})

        # Merge back to final_cohort
        final_tableone_df = final_tableone_df.merge(
            closest_adt[['encounter_block', 'first_location_imv']],
            on='encounter_block',
            how='left'
        )

        print("\n=== IMV Encounter Summary Complete ===")
        print(f"Total encounters: {len(final_tableone_df):,}")
        print(f"Encounters on IMV: {final_tableone_df['on_vent'].sum():,}")
        print(f"Initial mode categories:\n{final_tableone_df['initial_mode_category'].value_counts()}")


        # ==============================================================================
        # ## IMV- First 24 hours 
        # ==============================================================================

        # ============================================================================
        # 1. Add vent_start_dttm to final_tableone_df
        # ============================================================================

        # Merge vent_start_time from vent_start_end into final_tableone_df
        final_tableone_df = final_tableone_df.merge(
            vent_start_end[['encounter_block', 'vent_start_time']],
            on='encounter_block',
            how='left'
        )

        # Rename to vent_start_dttm for clarity
        final_tableone_df = final_tableone_df.rename(columns={'vent_start_time': 'vent_start_dttm'})

        print(f"\n✅ Added vent_start_dttm to final_tableone_df")
        print(f"   Encounters with vent_start_dttm: {final_tableone_df['vent_start_dttm'].notna().sum():,}")

        # Memory cleanup: Delete vent_start_end after final use
        del vent_start_end

        # ============================================================================
        # 2. Prepare IMV data with time from vent start
        # ============================================================================

        # Get IMV encounters
        resp_imv = resp_stitched[resp_stitched['encounter_block'].isin(imv_encounters)].copy()

        # Merge vent_start_dttm
        resp_imv = resp_imv.merge(
            final_tableone_df[['encounter_block', 'vent_start_dttm']],
            on='encounter_block',
            how='left'
        )

        # Calculate hours from vent start (time 0)
        resp_imv['hours_from_vent_start'] = (
            (resp_imv['recorded_dttm'] - resp_imv['vent_start_dttm']).dt.total_seconds() / 3600
        )

        # Filter to only records at or after vent start
        resp_imv_post_start = resp_imv[resp_imv['hours_from_vent_start'] >= 0].copy()

        final_tableone_df.columns

        # ============================================================================
        # SUPER OPTIMIZED: Calculate Statistics (10-20x faster!)
        # ============================================================================
        # Define all ventilator settings
        vent_settings = [
            'fio2_set', 'lpm_set', 'tidal_volume_set', 'resp_rate_set',
            'pressure_control_set', 'pressure_support_set', 'peep_set',
            'flow_rate_set'
        ]

        # Check which columns exist
        existing_settings = [col for col in vent_settings if col in resp_imv.columns]

        print(f"\nSettings to calculate: {len(existing_settings)}, vent settings")

        # ✅ SUPER OPTIMIZATION: Calculate each stat separately (pandas can optimize better)
        results = []

        # Median and mean/std (fast built-ins)
        medians = resp_imv.groupby('encounter_block')[existing_settings].median()
        medians.columns = [f'{col}_median' for col in medians.columns]

        means = resp_imv.groupby('encounter_block')[existing_settings].mean()
        means.columns = [f'{col}_mean' for col in means.columns]

        stds = resp_imv.groupby('encounter_block')[existing_settings].std()
        stds.columns = [f'{col}_std' for col in stds.columns]

        # Quantiles (still need to use slower method, but only once each)
        q1 = resp_imv.groupby('encounter_block')[existing_settings].quantile(0.25)
        q1.columns = [f'{col}_q1' for col in q1.columns]

        q3 = resp_imv.groupby('encounter_block')[existing_settings].quantile(0.75)
        q3.columns = [f'{col}_q3' for col in q3.columns]

        # Combine all statistics
        vent_settings_stats = pd.concat([medians, q1, q3, means, stds], axis=1).reset_index()

        print(f"✅ Calculated statistics for {len(existing_settings)} settings")
        print(f"   Total encounters: {len(vent_settings_stats):,}")

        # Merge back
        final_tableone_df = final_tableone_df.merge(
            vent_settings_stats,
            on='encounter_block',
            how='left'
        )

        # ============================================================================
        # 3. Filter for specific mode categories
        # ============================================================================

        # Define mode categories of interest
        volume_control_modes = ['assist control-volume control', 'pressure-regulated volume control']
        pressure_control_mode = ['pressure control']

        # Filter data
        volume_mode_data = resp_imv_post_start[
            resp_imv_post_start['mode_category'].isin(volume_control_modes)
        ].copy()

        pressure_mode_data = resp_imv_post_start[
            resp_imv_post_start['mode_category'].isin(pressure_control_mode)
        ].copy()

        print(f"\n📊 Mode Category Breakdown:")
        print(f"   Volume Control modes: {len(volume_mode_data):,} records")
        for mode in volume_control_modes:
            count = (volume_mode_data['mode_category'] == mode).sum()
            print(f"      - {mode}: {count:,}")
        print(f"   Pressure Control mode: {len(pressure_mode_data):,} records")

        # ============================================================================
        # 4. Plot Median/IQR and Mean/SD Tidal Volume for Volume Control Modes
        # ============================================================================

        # Calculate binned hour from start
        volume_mode_data['hour_bin'] = volume_mode_data['hours_from_vent_start'].round(0).astype(int)
        # Filter to first 168 hours (7 days)
        volume_mode_data_7d = volume_mode_data[volume_mode_data['hour_bin'] <= 168].copy()

        # Group by hour and calculate stats, including mean and std
        tv_stats = volume_mode_data_7d.groupby('hour_bin')['tidal_volume_set'].agg([
            ('median', 'median'),
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75)),
            ('mean', 'mean'),
            ('std', 'std'),
            ('count', 'count')
        ]).reset_index()

        # Filter hours with at least 10 measurements
        tv_stats = tv_stats[tv_stats['count'] >= 10]

        # Save CSV for Tidal Volume data (median/IQR/mean/SD)
        tv_csv_path = os.path.join(output_dir, 'tidal_volume_volume_control_modes.csv')
        tv_stats.to_csv(tv_csv_path, index=False)
        print(f"✅ Saved CSV: {tv_csv_path}")

        # Plot Median/IQR
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(tv_stats['hour_bin'], tv_stats['median'], 'o-', color='#2E86AB', linewidth=2, markersize=4, label='Median')
        ax.fill_between(tv_stats['hour_bin'], tv_stats['q25'], tv_stats['q75'], 
                        alpha=0.3, color='#2E86AB', label='IQR (25th-75th percentile)')
        ax.set_xlabel('Hours from Ventilation Start', fontsize=12, fontweight='bold')
        ax.set_ylabel('Tidal Volume Set (mL)', fontsize=12, fontweight='bold')
        ax.set_title('Tidal Volume Over Time: Volume Control Modes\n(Assist Control-Volume Control & Pressure-Regulated Volume Control)', 
                     fontsize=14, fontweight='bold', pad=15)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 168)
        plt.tight_layout()
        tv_png_path = os.path.join(figures_dir, 'tidal_volume_volume_control_modes.png')
        plt.savefig(tv_png_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n✅ Saved: {tv_png_path}")

        # Plot Mean/SD
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(tv_stats['hour_bin'], tv_stats['mean'], 'o-', color='#e67e22', linewidth=2, markersize=4, label='Mean')
        ax.fill_between(tv_stats['hour_bin'], tv_stats['mean'] - tv_stats['std'], tv_stats['mean'] + tv_stats['std'], 
                        alpha=0.25, color='#e67e22', label='SD (±1 std)')
        ax.set_xlabel('Hours from Ventilation Start', fontsize=12, fontweight='bold')
        ax.set_ylabel('Tidal Volume Set (mL)', fontsize=12, fontweight='bold')
        ax.set_title('Tidal Volume Over Time (Mean ± SD): Volume Control Modes\n(Assist Control-Volume Control & Pressure-Regulated Volume Control)', 
                     fontsize=14, fontweight='bold', pad=15)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 168)
        plt.tight_layout()
        tv_mean_png_path = os.path.join(figures_dir, 'tidal_volume_volume_control_modes_mean_sd.png')
        plt.savefig(tv_mean_png_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {tv_mean_png_path}")

        tv_mean_sd_csv_path = os.path.join(output_dir, 'tidal_volume_volume_control_modes_mean_sd.csv')
        tv_stats[['hour_bin', 'mean', 'std', 'count']].to_csv(tv_mean_sd_csv_path, index=False)
        print(f"✅ Saved CSV: {tv_mean_sd_csv_path}")

        # ============================================================================
        # 5. Plot Median/IQR and Mean/SD Pressure Control for Pressure Control Mode
        # ============================================================================

        # Calculate binned hour from start for pressure control mode
        pressure_mode_data['hour_bin'] = pressure_mode_data['hours_from_vent_start'].round(0).astype(int)
        pressure_mode_data_7d = pressure_mode_data[pressure_mode_data['hour_bin'] <= 168].copy()

        # Group by hour and calculate stats, including mean and std
        pc_stats = pressure_mode_data_7d.groupby('hour_bin')['pressure_control_set'].agg([
            ('median', 'median'),
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75)),
            ('mean', 'mean'),
            ('std', 'std'),
            ('count', 'count')
        ]).reset_index()

        pc_stats = pc_stats[pc_stats['count'] >= 10]

        # Save CSV for Pressure Control data (median/IQR/mean/SD)
        pc_csv_path = os.path.join(output_dir, 'pressure_control_pressure_control_mode.csv')
        pc_stats.to_csv(pc_csv_path, index=False)
        print(f"✅ Saved CSV: {pc_csv_path}")

        # Plot Median/IQR
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(pc_stats['hour_bin'], pc_stats['median'], 'o-', color='#A23B72', linewidth=2, markersize=4, label='Median')
        ax.fill_between(pc_stats['hour_bin'], pc_stats['q25'], pc_stats['q75'], 
                        alpha=0.3, color='#A23B72', label='IQR (25th-75th percentile)')
        ax.set_xlabel('Hours from Ventilation Start', fontsize=12, fontweight='bold')
        ax.set_ylabel('Pressure Control Set (cmH₂O)', fontsize=12, fontweight='bold')
        ax.set_title('Pressure Control Over Time: Pressure Control Mode', 
                     fontsize=14, fontweight='bold', pad=15)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 168)
        plt.tight_layout()
        pc_png_path = os.path.join(figures_dir, 'pressure_control_pressure_control_mode.png')
        plt.savefig(pc_png_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {pc_png_path}")

        # Plot Mean/SD
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(pc_stats['hour_bin'], pc_stats['mean'], 'o-', color='#27ae60', linewidth=2, markersize=4, label='Mean')
        ax.fill_between(pc_stats['hour_bin'], pc_stats['mean'] - pc_stats['std'], pc_stats['mean'] + pc_stats['std'], 
                        alpha=0.25, color='#27ae60', label='SD (±1 std)')
        ax.set_xlabel('Hours from Ventilation Start', fontsize=12, fontweight='bold')
        ax.set_ylabel('Pressure Control Set (cmH₂O)', fontsize=12, fontweight='bold')
        ax.set_title('Pressure Control Over Time (Mean ± SD): Pressure Control Mode', 
                     fontsize=14, fontweight='bold', pad=15)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 168)
        plt.tight_layout()
        pc_mean_png_path = os.path.join(figures_dir, 'pressure_control_pressure_control_mode_mean_sd.png')
        plt.savefig(pc_mean_png_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved: {pc_mean_png_path}")

        pc_mean_sd_csv_path = os.path.join(output_dir, 'pressure_control_pressure_control_mode_mean_sd.csv')
        pc_stats[['hour_bin', 'mean', 'std', 'count']].to_csv(pc_mean_sd_csv_path, index=False)
        print(f"✅ Saved CSV: {pc_mean_sd_csv_path}")

        # ============================================================================
        # Ventilator Settings Table by Device and Mode Category - FIXED
        # ============================================================================


        # Ventilator settings of interest
        vent_settings = [
            'fio2_set',
            'lpm_set',
            'tidal_volume_set',
            'resp_rate_set',
            'pressure_control_set',
            'peep_set',
            'pressure_support_set',
            'flow_rate_set'
        ]

        # ✅ OPTIMIZATION: Use groupby instead of nested loops (10-50x faster!)
        # Use full respiratory support data - ALL device and mode combinations
        resp_valid = resp_stitched.copy()

        # Check which columns actually exist in the data (graceful handling for missing columns)
        existing_settings = [col for col in vent_settings if col in resp_valid.columns]
        missing_settings = [col for col in vent_settings if col not in resp_valid.columns]

        if missing_settings:
            print(f"\n⚠️ Warning: The following ventilator settings are not available in the data: {', '.join(missing_settings)}")
            print(f"   Proceeding with {len(existing_settings)} available settings: {', '.join(existing_settings)}")
        else:
            print(f"\n✅ All {len(existing_settings)} ventilator settings found in the data")

        # Update vent_settings to only include existing columns
        vent_settings = existing_settings

        # Count all device-mode combinations
        group_counts = resp_valid.groupby(['device_category', 'mode_category']).size()

        print(f"Calculating statistics for {len(group_counts)} device-mode combinations from full respiratory support data...")

        # Calculate median, Q1, Q3 (only for existing columns)
        medians = resp_valid.groupby(['device_category', 'mode_category'])[vent_settings].median()
        q1 = resp_valid.groupby(['device_category', 'mode_category'])[vent_settings].quantile(0.25)
        q3 = resp_valid.groupby(['device_category', 'mode_category'])[vent_settings].quantile(0.75)

        # ✅ FIX: Reset index for all before combining
        medians_reset = medians.reset_index()
        q1_reset = q1.reset_index()
        q3_reset = q3.reset_index()

        # Create settings summary starting with device and mode columns
        settings_summary = medians_reset[['device_category', 'mode_category']].copy()

        # ✅ Format as "median (q1-q3)" using .values (no index issues)
        for setting in vent_settings:
            if setting in medians.columns:
                # Use .values to avoid index alignment issues
                settings_summary[setting] = (
                    medians_reset[setting].round(1).astype(str) + ' (' +
                    q1_reset[setting].round(1).astype(str) + '-' +
                    q3_reset[setting].round(1).astype(str) + ')'
                )

        # Build rename dictionary only for columns that exist
        rename_dict = {}
        # Rename mode_category to ventilator_setting for consistency in output
        if 'mode_category' in settings_summary.columns:
            rename_dict['mode_category'] = 'ventilator_setting'

        column_mapping = {
            'fio2_set': 'FiO2 Set',
            'lpm_set': 'LPM Set',
            'tidal_volume_set': 'Tidal Volume Set',
            'resp_rate_set': 'Resp Rate Set',
            'pressure_control_set': 'Pressure Control Set',
            'peep_set': 'PEEP Set',
            'pressure_support_set': 'Pressure Support Set',
            'flow_rate_set': 'Flow Rate Set'
        }

        # Only add columns that exist in settings_summary
        for old_name, new_name in column_mapping.items():
            if old_name in settings_summary.columns:
                rename_dict[old_name] = new_name

        # Rename columns
        settings_summary = settings_summary.rename(columns=rename_dict)

        # Sort by device and mode (use 'ventilator_setting' after rename)
        sort_col = 'ventilator_setting' if 'ventilator_setting' in settings_summary.columns else 'mode_category'
        settings_summary = settings_summary.sort_values(['device_category', sort_col])

        # Display and save
        print("\n" + "="*80)
        print("VENTILATOR SETTINGS BY DEVICE AND MODE CATEGORY")
        print("="*80)
        print(settings_summary.to_string(index=False))

        _vsbdm_path = os.path.join(output_dir, 'ventilator_settings_by_device_mode.csv')
        settings_summary.to_csv(_vsbdm_path, index=False)
        print(f"\n✅ Saved: {_vsbdm_path}")

        # ============================================================================
        # BONUS: Also create counts table (same optimization)
        # ============================================================================

        print("\n" + "="*80)
        print("Creating Observation Counts Table")
        print("="*80)

        # Count non-null observations for each setting (vectorized)
        counts_summary = resp_valid.groupby(['device_category', 'mode_category'])[vent_settings].count().reset_index()

        # Build rename dictionary only for columns that exist
        counts_rename_dict = {}
        # Rename mode_category to ventilator_setting for consistency in output
        if 'mode_category' in counts_summary.columns:
            counts_rename_dict['mode_category'] = 'ventilator_setting'

        counts_column_mapping = {
            'fio2_set': 'FiO2 Set (N)',
            'lpm_set': 'LPM Set (N)',
            'tidal_volume_set': 'Tidal Volume Set (N)',
            'resp_rate_set': 'Resp Rate Set (N)',
            'pressure_control_set': 'Pressure Control Set (N)',
            'peep_set': 'PEEP Set (N)',
            'pressure_support_set': 'Pressure Support Set (N)',
            'flow_rate_set': 'Flow Rate Set (N)'
        }

        # Only add columns that exist in counts_summary
        for old_name, new_name in counts_column_mapping.items():
            if old_name in counts_summary.columns:
                counts_rename_dict[old_name] = new_name

        # Rename columns
        counts_summary = counts_summary.rename(columns=counts_rename_dict)

        # Sort (use 'ventilator_setting' after rename)
        sort_col = 'ventilator_setting' if 'ventilator_setting' in counts_summary.columns else 'mode_category'
        counts_summary = counts_summary.sort_values(['device_category', sort_col])

        print("\nObservation Counts by Device and Mode:")
        print(counts_summary.to_string(index=False))

        _vscbdm_path = os.path.join(output_dir, 'ventilator_settings_counts_by_device_mode.csv')
        counts_summary.to_csv(_vscbdm_path, index=False)
        print(f"\n✅ Saved: {_vscbdm_path}")

        # Save total observations count for table reconstruction
        total_resp_obs = len(resp_valid)  # Total respiratory support observations
        total_obs_df = pd.DataFrame({
            'metric': ['total_respiratory_support_observations'],
            'value': [total_resp_obs]
        })
        _vsto_path = os.path.join(output_dir, 'ventilator_settings_total_observations.csv')
        total_obs_df.to_csv(_vsto_path, index=False)
        print(f"✅ Saved total observations count ({total_resp_obs:,}): {_vsto_path}")

        print("\n" + "="*80)
        print("VENTILATOR MODE PROPORTIONS - FIRST 24 HOURS OF IMV")
        print("="*80)

        # ============================================================================
        # 1. Filter to First 24 Hours of IMV
        # ============================================================================

        # Use the IMV data with hours from vent start that we already created
        # Data is already waterfall-processed from earlier; just filter to first 24h
        imv_first_24h = resp_imv_post_start[
            (resp_imv_post_start['hours_from_vent_start'] >= 0) &
            (resp_imv_post_start['hours_from_vent_start'] <= 24)
        ].copy()

        print(f"\n📊 Data Summary:")
        print(f"   Total IMV records in first 24h: {len(imv_first_24h):,}")
        print(f"   Unique encounters: {imv_first_24h['encounter_block'].nunique():,}")

        # ============================================================================
        # 2. Map Mode Categories to Simplified Groups
        # ============================================================================

        # Define mode category mapping (based on your image)
        mode_mapping = {
            'assist control-volume control': 'Assist Control-Volume Control',
            'pressure-regulated volume control': 'Pressure-Regulated Volume Control',
            'simv': 'SIMV',
            'pressure support/cpap': 'Pressure Support/CPAP',
            'pressure support': 'Pressure Support/CPAP',
            'cpap': 'Pressure Support/CPAP',
            'pressure control': 'Pressure Control',
        }

        # Apply mapping, anything not mapped goes to "Other"
        imv_first_24h['mode_group'] = imv_first_24h['mode_category'].str.lower().map(mode_mapping)
        imv_first_24h['mode_group'] = imv_first_24h['mode_group'].fillna('Other')

        # ============================================================================
        # 3. Calculate Proportions
        # ============================================================================

        # Count observations per mode
        mode_counts = imv_first_24h['mode_group'].value_counts()
        # Use first 24h IMV count for mode proportions analysis
        total_obs = len(imv_first_24h)

        # Calculate proportions
        mode_proportions = (mode_counts / total_obs).sort_values(ascending=False)

        print(f"\n📊 Mode Category Counts:")
        for mode, count in mode_counts.items():
            proportion = count / total_obs
            print(f"   {mode}: {count:,} ({proportion:.1%})")

        # ============================================================================
        # 4. Create DataFrame for Plotting
        # ============================================================================

        plot_data = pd.DataFrame({
            'Mode': mode_proportions.index,
            'Proportion': mode_proportions.values,
            'Count': mode_counts[mode_proportions.index].values
        })

        print("\nPlot Data:")
        print(plot_data.to_string(index=False))

        # Save the data
        _mp24_path = os.path.join(output_dir, 'mode_proportions_first_24h.csv')
        plot_data.to_csv(_mp24_path, index=False)
        print(f"\n✅ Saved: {_mp24_path}")

        # ============================================================================
        # 5. Generate Ventilator Settings Table (Combined Image)
        # ============================================================================
        print("\n" + "="*80)
        print("GENERATING VENTILATOR SETTINGS TABLE IMAGE")
        print("="*80)

        try:
            # Import the ventilator table generation function from the same module
            from .ventilator_table import plot_ventilator_table

            # Generate the table with the full respiratory support dataset count
            save_path = os.path.join(figures_dir, 'ventilator_settings_table.png')
            total_resp_obs = len(resp_stitched)  # Use full respiratory support dataset for the table
            fig = plot_ventilator_table(save_path=save_path, total_observations=total_resp_obs)
            print(f"✅ Ventilator settings table image generated successfully")

        except ImportError as e:
            print(f"⚠️ Could not import ventilator table generation function: {e}")
            print("   Skipping ventilator settings table image generation")
        except Exception as e:
            print(f"⚠️ Error generating ventilator settings table: {e}")
            print("   Skipping ventilator settings table image generation")

        fig, ax = plt.subplots(figsize=(6, 8))
        # Define colors for each mode (matching the image)
        color_map = {
            'Assist Control-Volume Control': '#66c2a5',  # Green
            'Pressure-Regulated Volume Control': '#fc8d62',  # Orange
            'SIMV': '#3288bd',  # Blue
            'Pressure Support/CPAP': '#9e9ac8',  # Purple
            'Pressure Control': '#fee08b',  # Yellow
            'Other': '#e41a8c'  # Pink/Magenta
        }

        # Create vertical stacked bar
        bottom = 0

        for idx, row in plot_data.iterrows():
            mode = row['Mode']
            proportion = row['Proportion']
            count = row['Count']
            color = color_map.get(mode, '#cccccc')
    
            ax.bar(0, proportion, bottom=bottom, width=0.5, 
                  color=color, edgecolor='white', linewidth=2)
    
            # Add text label
            if proportion > 0.03:
                ax.text(0, bottom + proportion/2, f"{proportion:.1%}\n(n={count:,})", 
                       ha='center', va='center', fontsize=9, fontweight='bold',
                       color='white' if proportion > 0.15 else 'black')
    
            bottom += proportion

        # Formatting
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Proportion of Mode Category', fontsize=14, fontweight='bold')
        ax.set_xticks([0])
        ax.set_xticklabels(['Dataset\n(All IMV Encounters)'], fontsize=11)
        ax.set_title('Proportions of Different Ventilator Modes\nUsed in First 24 Hours of IMV', 
                    fontsize=14, fontweight='bold', pad=20)

        # Create legend
        legend_elements = [plt.Rectangle((0,0),1,1, fc=color_map.get(mode, '#cccccc'), 
                                        edgecolor='white', linewidth=2, label=mode)
                          for mode in plot_data['Mode']]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), 
                 fontsize=10, title='Mode Category', title_fontsize=11)

        # Add grid
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        _mp24v_path = os.path.join(figures_dir, 'mode_proportions_first_24h_vertical.png')
        plt.savefig(_mp24v_path, dpi=300, bbox_inches='tight')
        plt.close('all')

        print(f"✅ Saved: {_mp24v_path}")

        # ── 28-day Ventilator-Free Days ────────────────────────────────
        from modules.tableone.vfd_calculator import calculate_ventilator_free_days
        vfd_df = calculate_ventilator_free_days(resp_stitched, final_tableone_df)
        if len(vfd_df) > 0:
            final_tableone_df = final_tableone_df.merge(
                vfd_df, on='encounter_block', how='left'
            )
            print(f"  ✅ VFDs computed for {len(vfd_df):,} IMV encounters")
            print(f"     VFD median: {vfd_df['vfd_28'].median():.0f}, "
                  f"mean: {vfd_df['vfd_28'].mean():.1f}")

        # Memory cleanup: Clear respiratory detailed analysis
        print("Clearing respiratory detailed analysis data from memory...")
        del resp_stitched, resp_stitched_imv, imv_encounters
        plt.close('all')
        gc.collect()
        checkpoint("Respiratory Detailed Analysis Complete")

        final_tableone_df.columns

    # "No IMV" stratum — critically ill encounters that were never intubated
    if 'on_vent' in final_tableone_df.columns:
        final_tableone_df['no_imv_enc'] = (final_tableone_df['on_vent'] == 0).astype(int)
        final_tableone_df['no_imv_icu_enc'] = (
            (final_tableone_df['no_imv_enc'] == 1) & (final_tableone_df['icu_enc'] == 1)
        ).astype(int)
        final_tableone_df['no_imv_no_icu_enc'] = (
            (final_tableone_df['no_imv_enc'] == 1) & (final_tableone_df['icu_enc'] == 0)
        ).astype(int)
        _n_no_imv = final_tableone_df['no_imv_enc'].sum()
        _n_no_imv_icu = final_tableone_df['no_imv_icu_enc'].sum()
        _n_no_imv_no_icu = final_tableone_df['no_imv_no_icu_enc'].sum()
        print(f"\nNo-IMV stratum: {_n_no_imv:,} encounters "
              f"(ICU: {_n_no_imv_icu:,}, no ICU: {_n_no_imv_no_icu:,})")

    # ==============================================================================
    # # Meds
    # ==============================================================================

    print(f"\nLoading medication_admin_continuous table...")
    clif.load_table(
        'medication_admin_continuous',
        columns=meds_required_columns,
        filters={
            'hospitalization_id': final_hosp_ids
        }
    )
    clif.medication_admin_continuous.df= pd.merge(clif.medication_admin_continuous.df, encounter_mapping, 
                                            on='hospitalization_id', how='left')


    print(f"   Medications loaded: {len(clif.medication_admin_continuous.df):,} rows")
    print(f"   Unique medication categories: {clif.medication_admin_continuous.df['med_category'].nunique()}")
    print(f"   Unique medication_admin_continuous hospitalizations: {clif.medication_admin_continuous.df['hospitalization_id'].nunique()}")

    meds_df = clif.medication_admin_continuous.df

    # ============================================================================
    # Medication Flags and Vasopressor Statistics at Encounter Block Level
    # ============================================================================

    print("\n" + "="*80)
    print("CREATING MEDICATION FLAGS AND VASOPRESSOR STATISTICS")
    print("="*80)

    # Medication categories to track
    med_categories = [
        'norepinephrine', 'epinephrine', 'phenylephrine', 'vasopressin', 'dopamine',
        'propofol', 'midazolam', 'lorazepam', 'dexmedetomidine', 'fentanyl',
        'vecuronium', 'rocuronium', 'cisatracurium', 'pancuronium'
    ]

    # Vasopressors that need conversion and dose statistics
    vasopressors = ['norepinephrine', 'epinephrine', 'phenylephrine', 'vasopressin', 'dopamine']

    # ============================================================================
    # 1. Create Binary Flags (0/1) for Each Medication
    # ============================================================================

    print("\n📊 Creating binary flags for medication exposure...")

    # Get unique encounter_block - med_category combinations
    med_encounters = meds_df[meds_df['med_category'].isin(med_categories)].groupby(
        ['encounter_block', 'med_category']
    ).size().reset_index(name='count')

    # Pivot to create binary columns (1 if medication given, 0 if not)
    med_flags = med_encounters.pivot(
        index='encounter_block',
        columns='med_category',
        values='count'
    ).notna().astype(int)

    # Rename columns with suffix
    med_flags.columns = [f'{col}_flag' for col in med_flags.columns]
    med_flags = med_flags.reset_index()

    print(f"✅ Created binary flags for {len(med_flags.columns)-1} medications")
    print(f"   Encounters with medication data: {len(med_flags):,}")

    # Show summary
    for col in med_flags.columns:
        if col != 'encounter_block':
            count = med_flags[col].sum()
            pct = 100 * count / len(med_flags)
            print(f"   {col}: {count:,} encounters ({pct:.1f}%)")

    # ============================================================================
    # 2. Convert Vasopressor Units to mcg/kg/min
    # ============================================================================

    print("\n" + "="*80)
    print("Converting Vasopressor Units")
    print("="*80)

    # Define preferred units for vasopressors
    preferred_units = {
        'norepinephrine': 'mcg/kg/min',
        'epinephrine': 'mcg/kg/min',
        'phenylephrine': 'mcg/kg/min',
        'vasopressin': 'mcg/kg/min',
        'dopamine': 'mcg/kg/min'
    }

    print(f"Converting {len(preferred_units)} vasopressors to mcg/kg/min...")

    # Pre-load only weight_kg vitals using Polars for efficiency
    print("Pre-loading weight data for medication conversion...")
    vitals_path = os.path.join(clif.data_directory, 'clif_vitals.parquet')

    # Use Polars to load only what we need
    weight_df_pl = (
        pl.scan_parquet(vitals_path)
        .filter(
            (pl.col('hospitalization_id').is_in(final_hosp_ids)) &
            (pl.col('vital_category') == 'weight_kg')
        )
        .collect()
    )

    # Detect medication timestamp format to match it
    # Access CLIFpy's loaded medication data to get timezone and time precision
    med_df_sample = pl.from_pandas(clif.medication_admin_continuous.df.head(1))
    admin_dttm_dtype = str(med_df_sample['admin_dttm'].dtype)

    # Extract timezone from dtype string (e.g., "Datetime(time_unit='us', time_zone='America/Chicago')")
    timezone_match = re.search(r"time_zone=['\"]([^'\"]+)['\"]", admin_dttm_dtype)
    med_timezone = timezone_match.group(1) if timezone_match else config['timezone']

    # Extract time unit from dtype string (e.g., 'us', 'ns', 'ms')
    time_unit_match = re.search(r"time_unit=['\"]([^'\"]+)['\"]", admin_dttm_dtype)
    med_time_unit = time_unit_match.group(1) if time_unit_match else 'us'

    print(f"Detected medication timestamp format: timezone={med_timezone}, time_unit={med_time_unit}")

    # Apply unified timezone and time unit standardization
    weight_df_pl = standardize_datetime_columns(
        weight_df_pl,
        target_timezone=med_timezone,
        target_time_unit=med_time_unit,
        datetime_columns=['recorded_dttm']
    )

    # Convert to Pandas for CLIF compatibility
    weight_vitals_df = weight_df_pl.to_pandas()

    # Add timezone information to match medication data (prevents DuckDB type mismatch)
    # Handle DST transitions using clifpy's standard approach
    if 'recorded_dttm' in weight_vitals_df.columns and weight_vitals_df['recorded_dttm'].dt.tz is None:
        weight_vitals_df['recorded_dttm'] = weight_vitals_df['recorded_dttm'].dt.tz_localize(
            config['timezone'],
            ambiguous=True,              # Assume DST for ambiguous times (fall back)
            nonexistent='shift_forward'  # Shift forward nonexistent times (spring forward)
        )

    print(f"Loaded {len(weight_vitals_df):,} weight measurements for {weight_df_pl['hospitalization_id'].n_unique()} hospitalizations")

    # Convert units (uses clifpy orchestrator)
    clif.convert_dose_units_for_continuous_meds(
        preferred_units=preferred_units,
        vitals_df=weight_vitals_df,  # Pass pre-loaded weight data
        override=True,
        save_to_table=True,
        hospitalization_ids=final_hosp_ids
    )

    # Get converted data
    meds_converted = clif.medication_admin_continuous.df_converted.copy()

    # Check conversion results
    conversion_counts = clif.medication_admin_continuous.conversion_counts

    print("\n=== Conversion Summary ===")
    success_count = conversion_counts[conversion_counts['_convert_status'] == 'success']['count'].sum()
    total_count = conversion_counts['count'].sum()
    print(f"Successful conversions: {success_count:,} / {total_count:,} ({100*success_count/total_count:.1f}%)")

    # Show any failed conversions
    failed_conversions = conversion_counts[conversion_counts['_convert_status'] != 'success']
    if len(failed_conversions) > 0:
        print(f"\n⚠️ Found {len(failed_conversions)} conversion issues:")
        print(failed_conversions[['med_category', '_clean_unit', '_convert_status', 'count']].to_string(index=False))

    # Clean up weight data to free memory
    del weight_vitals_df, weight_df_pl
    gc.collect()
    print("✓ Cleaned up weight data from memory")

    # ============================================================================
    # 3. Calculate Median and IQR for Vasopressors (Optimized)
    # ============================================================================

    print("\n" + "="*80)
    print("Calculating Vasopressor Dose Statistics")
    print("="*80)

    # Filter to vasopressors only and successfully converted doses
    vaso_df = meds_converted[
        (meds_converted['med_category'].isin(vasopressors)) &
        (meds_converted['_convert_status'] == 'success')
    ].copy()

    print(f"Vasopressor records for analysis: {len(vaso_df):,}")

    # Check which vasopressors exist in data
    existing_vasos = [v for v in vasopressors if v in vaso_df['med_category'].unique()]
    print(f"Vasopressors found: {existing_vasos}")

    # Calculate statistics for each vasopressor separately (more efficient)
    vaso_stats_list = []

    for vaso in existing_vasos:
        vaso_subset = vaso_df[vaso_df['med_category'] == vaso]

        # Calculate median, Q1, Q3 (vectorized)
        dose_stats = vaso_subset.groupby('encounter_block')['med_dose'].agg([
            ('median', 'median'),
            ('q1', lambda x: x.quantile(0.25)),
            ('q3', lambda x: x.quantile(0.75))
        ])

        # Rename columns with medication prefix
        dose_stats.columns = [f'{vaso}_{col}' for col in dose_stats.columns]
        dose_stats = dose_stats.reset_index()

        vaso_stats_list.append(dose_stats)

        print(f"   {vaso}: {len(dose_stats):,} encounters with dose data")

    # Merge all vasopressor statistics
    if vaso_stats_list:
        vaso_stats = vaso_stats_list[0]
        for dose_stats in vaso_stats_list[1:]:
            vaso_stats = vaso_stats.merge(dose_stats, on='encounter_block', how='outer')
    
        print(f"\n✅ Calculated dose statistics for {len(existing_vasos)} vasopressors")
        print(f"   Total encounters with vasopressor data: {len(vaso_stats):,}")
    else:
        vaso_stats = pd.DataFrame({'encounter_block': []})
        print("\n⚠️ No vasopressor data found for statistics")

    # ============================================================================
    # 3b. Norepinephrine Equivalent (NEE) — peak vasopressor intensity
    # ============================================================================
    # NEE weights from Khanna et al. / CLIF-epidemiology-of-CRRT.
    # Catecholamines in mcg/kg/min; vasopressin originally u/min (× 2.5).
    # All doses here are post-clifpy conversion (mcg/kg/min).
    _NEE_WEIGHTS = {
        'norepinephrine': 1.0,
        'epinephrine': 1.0,
        'phenylephrine': 0.1,
        'dopamine': 0.01,
        'vasopressin': 2.5,
        'angiotensin': 10.0,
    }

    if len(vaso_df) > 0:
        print("\nComputing Norepinephrine Equivalent (NEE)...")
        _nee_df = vaso_df[['encounter_block', 'admin_dttm', 'med_category', 'med_dose']].copy()
        _nee_df['weighted_dose'] = _nee_df['med_category'].map(_NEE_WEIGHTS).fillna(0) * _nee_df['med_dose']

        # Round to nearest hour to align concurrent medications.
        # Strip timezone first — floor('h') fails on DST fall-back hours
        # (AmbiguousTimeError). Timezone is irrelevant for hourly bucketing.
        _nee_df['hour'] = _nee_df['admin_dttm'].dt.tz_localize(None).dt.floor('h')

        # Sum weighted doses per encounter per hour → instantaneous NEE
        _hourly_nee = (
            _nee_df.groupby(['encounter_block', 'hour'])['weighted_dose']
            .sum()
            .reset_index(name='nee')
        )

        # Per-encounter peak and median NEE
        nee_per_enc = (
            _hourly_nee.groupby('encounter_block')['nee']
            .agg(nee_peak='max', nee_median='median')
            .reset_index()
        )
        print(f"  ✅ NEE computed for {len(nee_per_enc):,} encounters")
        print(f"     Peak NEE median: {nee_per_enc['nee_peak'].median():.3f}, "
              f"Median NEE median: {nee_per_enc['nee_median'].median():.3f} mcg/kg/min")

        del _nee_df, _hourly_nee
    else:
        nee_per_enc = pd.DataFrame(columns=['encounter_block', 'nee_peak', 'nee_median'])

    # ============================================================================
    # 4. Merge Everything to final_tableone_df
    # ============================================================================

    print("\n" + "="*80)
    print("Merging to final_tableone_df")
    print("="*80)

    initial_cols = len(final_tableone_df.columns)

    # Merge medication flags
    final_tableone_df = final_tableone_df.merge(
        med_flags,
        on='encounter_block',
        how='left'
    )

    # Fill NaN with 0 for medication flags (encounters without that medication)
    flag_cols = [col for col in med_flags.columns if col.endswith('_flag') and col != 'encounter_block']
    for col in flag_cols:
        final_tableone_df[col] = final_tableone_df[col].fillna(0).astype(int)

    print(f"✅ Added {len(flag_cols)} medication flag columns")

    # Merge vasopressor statistics
    if len(vaso_stats) > 0:
        final_tableone_df = final_tableone_df.merge(
            vaso_stats,
            on='encounter_block',
            how='left'
        )

        vaso_stat_cols = [col for col in vaso_stats.columns if col != 'encounter_block']

    # Merge NEE
    if len(nee_per_enc) > 0:
        final_tableone_df = final_tableone_df.merge(
            nee_per_enc, on='encounter_block', how='left'
        )
        print(f"✅ Added nee_peak column")
        print(f"✅ Added {len(vaso_stat_cols)} vasopressor dose statistic columns")

    new_cols = len(final_tableone_df.columns) - initial_cols

    print(f"\n✅ Total new columns added: {new_cols}")
    print(f"   Final tableone columns: {len(final_tableone_df.columns)}")

    # ============================================================================
    # 5. Summary Report
    # ============================================================================

    print("\n" + "="*80)
    print("MEDICATION SUMMARY")
    print("="*80)

    print("\n📋 Medication Flag Columns Added:")
    for col in sorted(flag_cols):
        med_name = col.replace('_flag', '')
        count = final_tableone_df[col].sum()
        pct = 100 * count / len(final_tableone_df)
        print(f"   {med_name:30s}: {count:6,} encounters ({pct:5.1f}%)")

    if len(vaso_stats) > 0:
        print("\n📋 Vasopressor Dose Statistics Columns Added:")
        for vaso in existing_vasos:
            median_col = f'{vaso}_median'
            if median_col in final_tableone_df.columns:
                count = final_tableone_df[median_col].notna().sum()
                print(f"   {vaso}:")
                print(f"      - {median_col}")
                print(f"      - {vaso}_q1")
                print(f"      - {vaso}_q3")
                print(f"      ({count:,} encounters with dose data)")

    print("\n" + "="*80)
    print("✅ MEDICATION PROCESSING COMPLETE")
    print("="*80)

    check = clif.medication_admin_continuous.df_converted.copy()


    # # Merge on encounter_block
    # meds_merged = meds_df.merge(
    #     final_tableone_df[['encounter_block','first_icu_in_dttm']], 
    #     on='encounter_block', 
    #     how='inner'
    # )
    # del meds_df

    # # Calculate hours from ICU admission
    # meds_merged['hours_from_icu'] = (
    #     pd.to_datetime(meds_merged['admin_dttm']) - pd.to_datetime(meds_merged['first_icu_in_dttm'])
    # ).dt.total_seconds() / 3600


    # ==============================================================================
    # ## First 24 hrs of ICU
    # ==============================================================================


    # Medication-from-ICU hourly plot is anchored to first_icu_in_dttm and
    # meaningful only for the critical-illness cohort. Decision 3: skip the
    # entire plot block (medications_hourly_data.csv, plotly area curves,
    # median dose plots, medications_summary_stats.csv) in ward mode. The
    # block is wrapped in a 0-or-1 iteration for-loop so the existing code
    # keeps its indentation but the body iterates 0 times in ward mode.
    if cohort_mode == 'ward':
        print("\nSkipping medication-from-ICU hourly plot block (ward mode)")

    for _icu_med_plot_iter in (range(1) if cohort_mode != 'ward' else range(0)):
        # Define medication groups
        med_groups = {
            'vasoactive': ['norepinephrine', 'epinephrine', 'phenylephrine', 'vasopressin', 'dopamine'],
            'sedative': ['propofol', 'midazolam', 'lorazepam', 'dexmedetomidine', 'fentanyl'],
            'paralytic': ['vecuronium', 'rocuronium', 'cisatracurium', 'pancuronium']
        }

        all_meds = [med for meds in med_groups.values() for med in meds]

        # Merge and calculate hours from ICU (vectorized)
        meds_merged = meds_df.merge(
            final_tableone_df[['encounter_block', 'first_icu_in_dttm']],
            on='encounter_block',
            how='inner'
        )

        meds_merged['hours_from_icu'] = (
            pd.to_datetime(meds_merged['admin_dttm']) - pd.to_datetime(meds_merged['first_icu_in_dttm'])
        ).dt.total_seconds() / 3600

        # Filter and bin (vectorized), handle non-finite for hour_bin, avoid IntCastingNaNError
        meds_merged['med_lower'] = meds_merged['med_category'].str.lower()
        finite_mask = np.isfinite(meds_merged['hours_from_icu'])
        meds_merged['hour_bin'] = np.nan
        meds_merged.loc[finite_mask, 'hour_bin'] = np.floor(meds_merged.loc[finite_mask, 'hours_from_icu'])
        meds_merged['hour_bin'] = meds_merged['hour_bin'].astype('Int64')

        meds_7d = meds_merged[
            (meds_merged['med_lower'].isin(all_meds)) &
            (meds_merged['hour_bin'].notna()) &
            (meds_merged['hour_bin'] >= 0) &
            (meds_merged['hour_bin'] <= 167)
        ]

        total_icu_encounters = final_tableone_df[final_tableone_df['icu_enc'] == 1]['encounter_block'].nunique()

        # ============================================================================
        #  hourly counts and percentages
        # ============================================================================

        pivot = (
            meds_7d
            .groupby(['hour_bin', 'med_lower'])['encounter_block']
            .nunique()
            .unstack(fill_value=0)
            .reindex(index=np.arange(168), columns=all_meds, fill_value=0)
        )

        pct_pivot = (pivot / total_icu_encounters * 100) if total_icu_encounters > 0 else pivot * 0

        hourly_df = pd.DataFrame({'hour': np.arange(168)})
        hourly_df = pd.concat([
            hourly_df,
            pivot.add_suffix('_n'),
            pct_pivot.add_suffix('_pct')
        ], axis=1)

        _mhd_path = os.path.join(output_dir, 'medications_hourly_data.csv')
        hourly_df.to_csv(_mhd_path, index=False)
        print(f"✅ Saved: {_mhd_path}")

        # ============================================================================
        # Plotly plotting functions (interactive area plots)
        # ============================================================================

        colors = {
            'norepinephrine': '#1f77b4', 'epinephrine': '#ff7f0e', 'phenylephrine': '#2ca02c',
            'vasopressin': '#d62728', 'dopamine': '#9467bd',
            'propofol': '#e377c2', 'midazolam': '#7f7f7f', 'lorazepam': '#bcbd22', 'dexmedetomidine': '#17becf',
            'vecuronium': '#8c564b', 'rocuronium': '#f7b6d2', 'cisatracurium': '#c49c94', 'pancuronium': '#dbdb8d'
        }

        def hex_to_rgba(hex_color, alpha=0.2):
            """Convert hex RGB color like '#1f77b4' to 'rgba(R,G,B,A)' string."""
            hex_color = hex_color.lstrip('#')
            if len(hex_color) == 6:
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                return f'rgba({r},{g},{b},{alpha})'
            # Fallback to gray if something is wrong
            return f'rgba(180,180,180,{alpha})'

        def plotly_medication_group(group_name, meds, hourly_df, output_path_html):
            fig = go.Figure()
            hours = hourly_df['hour'].values

            for med in meds:
                pct_col = f"{med}_pct"
                if pct_col in hourly_df.columns:
                    color = colors.get(med, '#333')
                    fillcolor = (
                        hex_to_rgba(color, 0.2)
                        if color.startswith("#") and len(color) == 7
                        else "rgba(180,180,180,0.15)"
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=hours,
                            y=hourly_df[pct_col],
                            mode='lines',
                            name=med.capitalize(),
                            line=dict(color=color, width=3),
                            fill='tozeroy',
                            fillcolor=fillcolor,
                            opacity=0.8,
                            hovertemplate=f"{med.capitalize()}<br>Hour: %{{x}}<br>% ICU: %{{y:.2f}}<extra></extra>"
                        )
                    )

            fig.update_layout(
                title=f"{group_name.capitalize()} Medication Use in First 7 Days of ICU",
                xaxis_title="Hours from ICU Admission",
                yaxis_title="% of ICU Encounters",
                xaxis=dict(range=[0, 168]),
                yaxis=dict(range=[0, None]),
                legend=dict(title="Medication", font=dict(size=12)),
                template="simple_white",
                font=dict(size=14),
                margin=dict(l=50, r=20, t=70, b=50)
            )

            # Save interactive plot as HTML
            pio.write_html(fig, output_path_html)
            # fig.show()  # REMOVED: Don't auto-open browser

        # Generate all 3 interactive plots (save as HTML in figures/)
        plotly_medication_group(
            'vasoactive', med_groups['vasoactive'], hourly_df,
            os.path.join(figures_dir, 'vasoactive_area_curve_7d.html')
        )
        plotly_medication_group(
            'sedative', med_groups['sedative'], hourly_df,
            os.path.join(figures_dir, 'sedative_area_curve_7d.html')
        )
        plotly_medication_group(
            'paralytic', med_groups['paralytic'], hourly_df,
            os.path.join(figures_dir, 'paralytic_area_curve_7d.html')
        )

        print("\n✅ All medication plots (plotly) created and saved as HTML!")


        # Define medication groups
        med_groups = {
            'vasoactive': ['norepinephrine', 'epinephrine', 'phenylephrine', 'vasopressin', 'dopamine'],
            'sedative': ['propofol', 'midazolam', 'lorazepam', 'dexmedetomidine', 'fentanyl'],
            'paralytic': ['vecuronium', 'rocuronium', 'cisatracurium', 'pancuronium']
        }

        # Lower-cased mapping for safety
        med_to_group = {med: group for group, meds in med_groups.items() for med in meds}
        all_meds = [med for meds in med_groups.values() for med in meds]

        # Merge and preprocess (same as before)
        meds_merged = meds_df.merge(
            final_tableone_df[['encounter_block', 'first_icu_in_dttm']],
            on='encounter_block',
            how='inner'
        )

        meds_merged['hours_from_icu'] = (
            pd.to_datetime(meds_merged['admin_dttm']) - pd.to_datetime(meds_merged['first_icu_in_dttm'])
        ).dt.total_seconds() / 3600

        meds_merged['med_lower'] = meds_merged['med_category'].str.lower()
        finite_mask = np.isfinite(meds_merged['hours_from_icu'])
        meds_merged['hour_bin'] = np.nan
        meds_merged.loc[finite_mask, 'hour_bin'] = np.floor(meds_merged.loc[finite_mask, 'hours_from_icu'])
        meds_merged['hour_bin'] = meds_merged['hour_bin'].astype('Int64')

        meds_7d = meds_merged[
            (meds_merged['med_lower'].isin(all_meds)) &
            (meds_merged['hour_bin'].notna()) &
            (meds_merged['hour_bin'] >= 0) &
            (meds_merged['hour_bin'] <= 167)
        ].copy()

        # =======================
        # Line plot: median dose by hour since ICU admission (per med group)
        # =======================

        colors = {
            'norepinephrine': '#1f77b4', 'epinephrine': '#ff7f0e', 'phenylephrine': '#2ca02c',
            'vasopressin': '#d62728', 'dopamine': '#9467bd',
            'propofol': '#e377c2', 'midazolam': '#7f7f7f', 'lorazepam': '#bcbd22', 'dexmedetomidine': '#17becf',
            'vecuronium': '#8c564b', 'rocuronium': '#f7b6d2', 'cisatracurium': '#c49c94', 'pancuronium': '#dbdb8d'
        }

        def plot_median_dose_line_by_hour(group_name, meds, meds_7d, output_path_html):
            fig = go.Figure()
            for med in meds:
                med_data = meds_7d[meds_7d['med_lower'] == med]
                # For each hour, compute the median dose (across all encounters)
                hourly_median = (
                    med_data.groupby('hour_bin')['med_dose']
                    .median()
                    .reset_index()
                    .sort_values('hour_bin')
                )
                color = colors.get(med, '#333')
                if not hourly_median.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=hourly_median['hour_bin'],
                            y=hourly_median['med_dose'],
                            mode='lines+markers',
                            name=med.capitalize(),
                            line=dict(color=color, width=3),
                            marker=dict(size=6),
                            hovertemplate=f"{med.capitalize()}<br>Hour: %{{x}}<br>Median Dose: %{{y:.2f}}<extra></extra>"
                        )
                    )
            fig.update_layout(
                title=f"Median {group_name.capitalize()} Dose by Hour Since ICU Admission",
                xaxis_title="Hours from ICU Admission",
                yaxis_title="Median Dose",
                legend=dict(title="Medication", font=dict(size=12)),
                template="simple_white",
                font=dict(size=14),
                margin=dict(l=50, r=20, t=70, b=50),
                xaxis=dict(range=[0, 168])
            )
            pio.write_html(fig, output_path_html)
            # fig.show()  # REMOVED: Don't auto-open browser

        # Generate and save plots for each medication group (lines: median dose over time)
        plot_median_dose_line_by_hour(
            'vasoactive', med_groups['vasoactive'], meds_7d,
            os.path.join(figures_dir, 'vasoactive_median_dose_by_hour.html')
        )
        plot_median_dose_line_by_hour(
            'sedative', med_groups['sedative'], meds_7d,
            os.path.join(figures_dir, 'sedative_median_dose_by_hour.html')
        )
        plot_median_dose_line_by_hour(
            'paralytic', med_groups['paralytic'], meds_7d,
            os.path.join(figures_dir, 'paralytic_median_dose_by_hour.html')
        )

        print("\n✅ All median dose line plots by hour (plotly) created and saved as HTML!")

        # ============================================================================
        #  summary statistics
        # ============================================================================

        # Group by medication once, calculate all stats on med_dose
        summary_agg = (
            meds_7d
            .groupby('med_lower')['med_dose']
            .agg(['count', 'median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
            .rename(columns={'count': 'n_admin', '<lambda_0>': 'q1_dose', '<lambda_1>': 'q3_dose'})
        )

        # Count unique encounters per medication
        encounter_counts = meds_7d.groupby('med_lower')['encounter_block'].nunique()

        # Get most common dose unit per medication
        dose_units = meds_7d.groupby('med_lower')['med_dose_unit'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else '')

        # Combine and add group labels
        summary_df = pd.DataFrame({
            'medication': summary_agg.index,
            'n_encounters': encounter_counts.values,
            'pct_encounters': (encounter_counts / total_icu_encounters * 100).values,
            'median_dose': summary_agg['median'].values,
            'q1_dose': summary_agg['q1_dose'].values,
            'q3_dose': summary_agg['q3_dose'].values,
            'dose_unit': dose_units.values
        })

        # Add group labels (vectorized with map)
        med_to_group = {med: group for group, meds in med_groups.items() for med in meds}
        summary_df['group'] = summary_df['medication'].map(med_to_group)
        summary_df = summary_df[['group', 'medication', 'n_encounters', 'pct_encounters', 
                                 'median_dose', 'q1_dose', 'q3_dose', 'dose_unit']]

        _mss_path = os.path.join(output_dir, 'medications_summary_stats.csv')
        summary_df.to_csv(_mss_path, index=False)
        print(f"✅ Saved: {_mss_path}")


        # Cleanup of in-loop intermediates (meds_7d and summary_df only
        # exist when the medication-from-ICU plot block ran)
        del meds_7d, summary_df
    # Memory cleanup: Clear medication processing data
    print("Clearing medication processing data from memory...")
    del meds_df, meds_converted, vaso_df, vaso_stats, med_flags
    plt.close('all')
    gc.collect()
    checkpoint("Medication Analysis Complete")


    # ==============================================================================
    # # Labs
    # ==============================================================================

    # ----------------------------------------------------------------------------
    # Load Labs
    # ----------------------------------------------------------------------------
    # print(f"\nLoading labs table...")
    # clif.load_table(
    #     'labs',
    #     columns=['hospitalization_id', 'lab_result_dttm', 'lab_category', 'lab_value_numeric', 'reference_unit'],
    #     filters={
    #         'hospitalization_id': final_hosp_ids
    #     }
    # )
    # clif.labs.df= pd.merge(clif.labs.df, encounter_mapping, 
    #                                         on='hospitalization_id', how='left')


    print("Applying outlier handling to Labs data...")
    print("=" * 50)
    # MCIDE collection moved to separate script: generate_mcide_and_stats.py
    # This avoids loading large tables into memory
    # MCIDE and summary statistics moved to separate script (generate_mcide_and_stats.py)
    # get_value_counts_mcide(clif.labs, 'labs', ['lab_name', 'lab_category', 'lab_loinc_code'], output_dir=mcide_dir, config=config)
    # apply_outlier_handling(clif.labs)
    # create_summary_table(clif.labs, 'lab_value_numeric',group_by_cols='lab_category',
    #                     output_dir=summary_stats_dir)
    # Summary statistics moved to MCIDE collection (generate_mcide_and_stats.py)
    # create_summary_table(clif.labs, 'lab_value_numeric',group_by_cols=['lab_category', 'reference_unit'],
    #                     output_dir=summary_stats_dir)

    # ==============================================================================
    # # Comorbidity Index
    # ==============================================================================

    print(f"\nLoading vitals table...")
    clif.load_table(
        'hospital_diagnosis',
        filters={
            'hospitalization_id': final_hosp_ids
        }
    )

    cci_results = calculate_cci( clif.hospital_diagnosis, hierarchy=True)

    cci_results = (
        cci_results.merge(encounter_mapping, on="hospitalization_id")
        .drop(columns=["hospitalization_id"])
        .groupby("encounter_block")
        .max()
        .reset_index()
    )
    # Join with final_tableone_df on encounter_block
    final_tableone_df = final_tableone_df.merge(cci_results, on="encounter_block", how="left")
    final_tableone_df.columns

    # Calculate comorbidities per 1000 hospitalizations, and save results WITH statistical summaries in CSV
    # Step 1: Get the total number of unique hospitalizations
    total_hospitalizations = cci_results['encounter_block'].nunique()
    print(f"Total hospitalizations: {total_hospitalizations:,}")

    # Step 2: Define the comorbidity columns (exclude IDs and total score)
    exclude_columns = {'hospitalization_id', 'encounter_block', 'cci_score'}
    comorbidity_columns = [col for col in cci_results.columns if col not in exclude_columns]

    # Step 3: Calculate the count of each comorbidity (assume binary indicators)
    comorbidity_counts = cci_results[comorbidity_columns].sum()

    # Step 4: Compute prevalence rates
    comorbidity_per_1000 = (comorbidity_counts / total_hospitalizations) * 1000
    prevalence_percent = (comorbidity_counts.values / total_hospitalizations * 100).round(2)

    # Step 5: Create a summary dataframe
    comorbidity_summary = pd.DataFrame({
        'comorbidity': comorbidity_columns,
        'n_patients': comorbidity_counts.values,
        'prevalence_percent': prevalence_percent,
        'per_1000_hospitalizations': comorbidity_per_1000.values.round(1)
    })

    # Sort by per 1000 prevalence
    comorbidity_summary = comorbidity_summary.sort_values('per_1000_hospitalizations', ascending=False).reset_index(drop=True)

    # Step 6: Prepare summary statistics for output
    total_comorbidities = int(comorbidity_counts.sum())
    avg_comorbidities_per_hosp = total_comorbidities / total_hospitalizations if total_hospitalizations > 0 else 0
    most_common_comorbidity = comorbidity_summary.iloc[0]['comorbidity']
    most_common_per_1000 = comorbidity_summary.iloc[0]['per_1000_hospitalizations']

    # Step 7: Save both table and summary statistics to CSV

    # First, write the comorbidity table to CSV
    out_csv = os.path.join(output_dir, 'comorbidities_per_1000_hospitalizations.csv')
    comorbidity_summary.to_csv(out_csv, index=False)

    # Write summary statistics to a second csv, and then append to same file as lines at the end

    summary_stats = [
        ['Total hospitalizations', total_hospitalizations],
        ['Total comorbidities across all patients', total_comorbidities],
        ['Average comorbidities per hospitalization', f"{avg_comorbidities_per_hosp:.2f}"],
        ['Most common comorbidity', most_common_comorbidity],
        ['Most common: per 1000 hospitalizations', f"{most_common_per_1000:.1f}"],
    ]

    # Save the summary stats to a separate CSV for clarity (and also appending to the main comorbidity file for convenience)
    summary_csv = os.path.join(output_dir, 'comorbidities_per_1000_hospitalizations_summary.csv')
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        for row in summary_stats:
            writer.writerow(row)

    print(f"\nComorbidity table saved to: {out_csv}")
    print(f"Summary statistics saved to: {summary_csv}")

    # Step 8: Bar plot

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(
        comorbidity_summary['comorbidity'], 
        comorbidity_summary['per_1000_hospitalizations'],
        color='#7FA8B8',
        edgecolor='black',
        linewidth=0.5
    )

    ax.set_xlabel('Per 1000 Hospitalizations', fontsize=12, fontweight='bold')
    ax.set_ylabel('Comorbidity', fontsize=12, fontweight='bold')
    ax.set_title('Comorbidity Prevalence per 1000 Hospitalizations', fontsize=14, fontweight='bold', pad=20)

    # Add value labels
    for i, (idx, row) in enumerate(comorbidity_summary.iterrows()):
        ax.text(row['per_1000_hospitalizations'] + 5, i, f"{row['per_1000_hospitalizations']:.1f}", va='center', fontsize=9)

    ax.grid(axis='x', linestyle='-', alpha=0.3)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'comorbidities_per_1000_barplot.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    checkpoint("CCI Complete")

    # ==============================================================================
    # # ECMO
    # ==============================================================================

    print(f"\nLoading ECMO table...")
    try:
        clif.load_table(
            'ecmo_mcs',
            filters={
                'hospitalization_id': final_hosp_ids
            }
        )
        clif.ecmo_mcs.df = pd.merge(
            clif.ecmo_mcs.df,
            encounter_mapping,
            on='hospitalization_id',
            how='left'
        )
        # Add on_ecmo = 1 for all encounter blocks in this df
        clif.ecmo_mcs.df['on_ecmo'] = 1
        # Keep only encounter_block and on_ecmo
        ecmo_df = clif.ecmo_mcs.df[['encounter_block', 'on_ecmo']].drop_duplicates()
        # Join with final_tableone_df on encounter_block (left join)
        final_tableone_df = final_tableone_df.merge(ecmo_df, on='encounter_block', how='left')
        # Optionally fill NaN with 0 if you want on_ecmo=0 for non-ECMO
        final_tableone_df['on_ecmo'] = final_tableone_df['on_ecmo'].fillna(0).astype(int)
        # MCIDE and summary statistics moved to separate script: generate_mcide_and_stats.py
        # get_value_counts_mcide(clif.ecmo_mcs, 'ecmo_mcs', ['device_name', 'device_category'], output_dir=mcide_dir, config=config)
        # create_summary_table(clif.ecmo_mcs,
        #                         ['device_rate', 'sweep', 'fdO2','flow'],
        #                         group_by_cols='device_category',
        #                         output_dir=summary_stats_dir)
    except FileNotFoundError as e:
        print(f"Warning: Failed to load the ECMO table: {e}. Proceeding without ECMO data.")

    checkpoint("ECMO Complete")

    # ==============================================================================
    # # CRRT Therapy
    # ==============================================================================

    print(f"\nLoading crrt_therapy table...")
    try:
        clif.load_table(
            'crrt_therapy',
            filters={
                'hospitalization_id': final_hosp_ids
            }
        )
        clif.crrt_therapy.df = pd.merge(
            clif.crrt_therapy.df,
            encounter_mapping,
            on='hospitalization_id',
            how='left'
        )
        clif.crrt_therapy.df['on_crrt'] = 1
        # Keep only encounter_block and on_crrt
        on_crrt_df = clif.crrt_therapy.df[['encounter_block', 'on_crrt']].drop_duplicates()
        # Join with final_tableone_df on encounter_block (left join)
        final_tableone_df = final_tableone_df.merge(on_crrt_df, on='encounter_block', how='left')
        # Before proceeding, ensure 'on_crrt' column exists after the merge; if not, create it
        if 'on_crrt' not in final_tableone_df.columns:
            final_tableone_df['on_crrt'] = 0
        else:
            final_tableone_df['on_crrt'] = final_tableone_df['on_crrt'].fillna(0).astype(int)
        # MCIDE and summary statistics moved to separate script: generate_mcide_and_stats.py
        # get_value_counts_mcide(clif.crrt_therapy, 'crrt_therapy', ['crrt_mode_name', 'crrt_mode_category'], output_dir=mcide_dir, config=config)
        # create_summary_table(clif.crrt_therapy,
        #                         ['blood_flow_rate', 'pre_filter_replacement_fluid_rate', 'post_filter_replacement_fluid_rate',
        #                             'dialysate_flow_rate', 'ultrafiltration_out'],
        #                         group_by_cols='crrt_mode_category',
        #                         output_dir=summary_stats_dir)
    except FileNotFoundError as e:
        print(f"Warning: Failed to load the ECMO table: {e}. Proceeding without CRRT data.")

    checkpoint("CRRT Complete")

    # ==============================================================================
    # # Sepsis (CDC Adult Sepsis Event)
    # ==============================================================================

    if cohort_mode == 'ward':
        print("\nSkipping ASE computation (ward mode)")
        final_tableone_df['sepsis_events_by_sepsis_col'] = 0
        final_tableone_df['sepsis_events_by_episode_id'] = 0
    else:
        print(f"\nComputing Adult Sepsis Events (ASE)...")
        ASE_BATCH_SIZE = 50_000
        try:
            if len(final_hosp_ids) > ASE_BATCH_SIZE:
                # Batch to keep DuckDB memory bounded
                batches = [
                    final_hosp_ids[i:i + ASE_BATCH_SIZE]
                    for i in range(0, len(final_hosp_ids), ASE_BATCH_SIZE)
                ]
                print(f"   Processing {len(final_hosp_ids):,} hospitalizations in {len(batches)} batches of ≤{ASE_BATCH_SIZE:,}")
                ase_parts = []
                for idx, batch_ids in enumerate(batches, 1):
                    print(f"   Batch {idx}/{len(batches)} ({len(batch_ids):,} IDs)...")
                    part = compute_ase(
                        hospitalization_ids=batch_ids,
                        config_path=config_path,
                        data_directory=config['tables_path'],
                        filetype=config['file_type'],
                        verbose=False
                    )
                    ase_parts.append(part)
                    gc.collect()
                ase_df = pd.concat(ase_parts, ignore_index=True)
                del ase_parts
                gc.collect()
                print(f"   All batches complete — {len(ase_df):,} total rows")
            else:
                ase_df = compute_ase(
                    hospitalization_ids=final_hosp_ids,
                    config_path=config_path,
                    data_directory=config['tables_path'],
                    filetype=config['file_type'],
                    verbose=True
                )

            # Print sample of ASE results
            print("\n--- ASE Results Sample (first 20 rows) ---")
            print(ase_df.head(20).to_string())
            print(f"--- End ASE Sample ({len(ase_df)} total rows) ---\n")

            # Map hospitalization_id → encounter_block
            ase_enc = ase_df.merge(
                encounter_mapping[['hospitalization_id', 'encounter_block']],
                on='hospitalization_id',
                how='inner'
            )

            # Count sepsis events per encounter_block using both methods
            sepsis_rows = ase_enc[ase_enc['sepsis'] == 1]
            counts_by_sepsis = sepsis_rows.groupby('encounter_block').size().reset_index(name='sepsis_events_by_sepsis_col')
            counts_by_episode = sepsis_rows.groupby('encounter_block')['episode_id'].count().reset_index(name='sepsis_events_by_episode_id')

            sepsis_counts = counts_by_sepsis.merge(counts_by_episode, on='encounter_block', how='outer')

            final_tableone_df = final_tableone_df.merge(sepsis_counts, on='encounter_block', how='left')
            final_tableone_df['sepsis_events_by_sepsis_col'] = final_tableone_df['sepsis_events_by_sepsis_col'].fillna(0).astype(int)
            final_tableone_df['sepsis_events_by_episode_id'] = final_tableone_df['sepsis_events_by_episode_id'].fillna(0).astype(int)

            total_by_sepsis = final_tableone_df['sepsis_events_by_sepsis_col'].sum()
            total_by_episode = final_tableone_df['sepsis_events_by_episode_id'].sum()
            enc_with_sepsis = (final_tableone_df['sepsis_events_by_sepsis_col'] > 0).sum()
            print(f"   Total sepsis events (via sepsis col): {total_by_sepsis:,}")
            print(f"   Total sepsis events (via episode_id): {total_by_episode:,}")
            print(f"   Encounters with >=1 sepsis event: {enc_with_sepsis:,} / {len(final_tableone_df):,}")
        except Exception as e:
            print(f"Warning: Failed to compute ASE: {e}. Proceeding without sepsis data.")
            final_tableone_df['sepsis_events_by_sepsis_col'] = 0
            final_tableone_df['sepsis_events_by_episode_id'] = 0

    checkpoint("Sepsis Complete")

    # ==============================================================================
    # # Patient Assessments
    # ==============================================================================

    # print(f"\nLoading patient_assessments table...")
    # try:
    #     clif.load_table(
    #         'patient_assessments',
    #         filters={
    #             'hospitalization_id': final_hosp_ids
    #         }
    #     )
    #     clif.patient_assessments.df = pd.merge(
    #         clif.patient_assessments.df,
    #         encounter_mapping,
    #         on='hospitalization_id',
    #         how='left'
    #     )
    #     # MCIDE collection moved to separate script: generate_mcide_and_stats.py
    #     # get_value_counts_mcide(clif.patient_assessments, 'patient_assessments', ['assessment_name', 'assessment_category', 'assessment_group'], output_dir=mcide_dir, config=config)
    # except FileNotFoundError as e:
    #     print(f"Warning: Failed to load the ECMO table: {e}. Proceeding without Patient Assessments data.")


    # ==============================================================================
    # # Number of clinical events
    # ==============================================================================
    # COMMENTED OUT: This section fails when table.df is None after memory cleanup
    # TODO: Fix this section to handle None dataframes or reload tables as needed

    # clif.get_loaded_tables()

    # def count_clinical_events(clif_table, composite_key_fields, verbose=True):
    #     """
    #     Count unique clinical events in a CLIF table based on composite key.
    #     
    #     A clinical event is defined as a unique combination of the composite key fields,
    #     with all fields non-null.
    #     
    #     Parameters:
    #     -----------
    #     clif_table : CLIFTable object
    #         The CLIF table object (e.g., clif.vitals, clif.labs, clif.medication_admin_continuous)
    #     composite_key_fields : list of str
    #         List of column names that form the composite key.
    #         Example: ['hospitalization_id', 'recorded_dttm', 'vital_category']
    #     verbose : bool, default=True
    #         If True, print summary statistics
    #         
    #     Returns:
    #     --------
    #     dict with keys:
    #         - 'n_events': Number of unique non-null clinical events
    #         - 'n_total_rows': Total rows in table
    #         - 'n_rows_with_nulls': Rows with null values in composite key
    #         - 'n_valid_rows': Rows with all composite key fields non-null
    #         - 'pct_valid': Percentage of rows that are valid
    #         
    #     Example:
    #     --------
    #     >>> # Count unique vital sign measurements
    #     >>> result = count_clinical_events(
    #     ...     clif.vitals, 
    #     ...     ['hospitalization_id', 'recorded_dttm', 'vital_category']
    #     ... )
    #     >>> print(f"Unique vital measurements: {result['n_events']:,}")
    #     """
    #     # Get dataframe
    #     df = clif_table.df
    #     table_name = clif_table.__class__.__name__
    #     
    #     # Check that all composite key fields exist
    #     missing_fields = [f for f in composite_key_fields if f not in df.columns]
    #     if missing_fields:
    #         raise ValueError(f"Fields not found in {table_name}: {missing_fields}")
    #     
    #     # Total rows
    #     n_total = len(df)
    #     
    #     # Drop rows where ANY composite key field is null
    #     df_valid = df.dropna(subset=composite_key_fields)
    #     n_valid = len(df_valid)
    #     n_with_nulls = n_total - n_valid
    #     
    #     # Count unique combinations of composite key fields (clinical events)
    #     n_events = df_valid[composite_key_fields].drop_duplicates().shape[0]
    #     
    #     # Calculate percentage
    #     pct_valid = 100 * n_valid / n_total if n_total > 0 else 0
    #     
    #     # Prepare results
    #     results = {
    #         'n_events': n_events,
    #         'n_total_rows': n_total,
    #         'n_rows_with_nulls': n_with_nulls,
    #         'n_valid_rows': n_valid,
    #         'pct_valid': pct_valid
    #     }
    #     
    #     # Print summary if verbose
    #     if verbose:
    #         print(f"\n{'='*80}")
    #         print(f"Clinical Events Count: {table_name}")
    #         print(f"{'='*80}")
    #         print(f"Composite key: {composite_key_fields}")
    #         print(f"\nTotal rows in table:        {n_total:>12,}")
    #         print(f"Rows with null in key:      {n_with_nulls:>12,} ({100*n_with_nulls/n_total:.1f}%)")
    #         print(f"Valid rows (non-null key):  {n_valid:>12,} ({pct_valid:.1f}%)")
    #         print(f"Unique clinical events:     {n_events:>12,}")
    #         print(f"{'='*80}")
    #     
    #     return results
    #     
    # def count_all_clinical_events(clif_obj):
    #     """
    #     Count clinical events for all common CLIF tables.
    #     
    #     Parameters:
    #     -----------
    #     clif_obj : CLIF object
    #         The main CLIF object containing all tables
    #         
    #     Returns:
    #     --------
    #     pd.DataFrame : Summary of clinical events for all tables
    #     """
    #     tables_config = {
    #         'vitals': ['hospitalization_id', 'recorded_dttm', 'vital_category'],
    #         'labs': ['hospitalization_id', 'lab_result_dttm', 'lab_category'],
    #         'medication_admin_continuous': ['hospitalization_id', 'admin_dttm', 'med_category'],
    #         'respiratory_support': ['hospitalization_id', 'recorded_dttm'],
    #         'adt': ['hospitalization_id', 'in_dttm'],
    #         'patient_assessments': ['hospitalization_id', 'recorded_dttm', 'assessment_category'],
    #         'ecmo_mcs': ['hospitalization_id', 'recorded_dttm', 'device_category'],
    #         'crrt_therapy' : ['hospitalization_id', 'recorded_dttm', 'crrt_mode_category'],
    #     }
    #     
    #     results = []
    #     
    #     for table_name, composite_key in tables_config.items():
    #         if hasattr(clif_obj, table_name):
    #             table = getattr(clif_obj, table_name)
    #             if hasattr(table, 'df') and len(table.df) > 0:
    #                 result = count_clinical_events(table, composite_key, verbose=False)
    #                 results.append({
    #                     'table': table_name,
    #                     'composite_key': str(composite_key),
    #                     'n_events': result['n_events'],
    #                     'n_total_rows': result['n_total_rows'],
    #                     'pct_valid': result['pct_valid']
    #                 })
    #     
    #     summary_df = pd.DataFrame(results)
    #     
    #     print("\n" + "="*80)
    #     print("CLINICAL EVENTS SUMMARY - ALL TABLES")
    #     print("="*80)
    #     print(summary_df.to_string(index=False))
    #     print("="*80)
    #     
    #     return summary_df
    # 
    # # Run batch summary
    # events_summary = count_all_clinical_events(clif)

    os.makedirs(project_root / 'output/intermediate', exist_ok=True)
    # Debug parquet write — cohort-aware filename so the ward run doesn't clobber
    # the critical-illness debug parquet (or vice versa).
    _test_parquet_name = (
        'final_tableone_ward_df_test.parquet'
        if cohort_mode == 'ward'
        else 'final_tableone_df_test.parquet'
    )
    final_tableone_df.to_parquet(project_root / 'output/intermediate' / _test_parquet_name)
    # ==============================================================================
    # # SOFA calculation
    # ==============================================================================

    # SOFA computation is anchored to first_icu_in_dttm and only computed for
    # icu_enc==1 encounters. Decision 2: skip the entire block in ward mode.
    # The block is wrapped in a 0-or-1 iteration for-loop so the existing code
    # keeps its indentation but the body iterates 0 times in ward mode.
    if cohort_mode == 'ward':
        print("\nSkipping SOFA computation block (ward mode)")

    for _sofa_iter in (range(1) if cohort_mode != 'ward' else range(0)):
        checkpoint("Starting SOFA Computation")
        print("Preparing SOFA cohort for Polars computation...")

        # Filter to icu_enc == 1
        sofa_cohort_df = final_tableone_df[final_tableone_df['icu_enc'] == 1][['hospitalization_id', 'encounter_block', 'first_icu_in_dttm']].copy()
        sofa_cohort_df['start_dttm'] = sofa_cohort_df['first_icu_in_dttm']
        sofa_cohort_df['end_dttm'] = sofa_cohort_df['start_dttm'] + pd.Timedelta(hours=24)
        sofa_cohort_df = sofa_cohort_df[['hospitalization_id', 'encounter_block', 'start_dttm', 'end_dttm']]
        sofa_cohort_ids = cohort_df['hospitalization_id'].astype(str).unique().tolist()

        # Convert to Polars
        sofa_cohort_pl = pl.from_pandas(sofa_cohort_df)

        print(f"Cohort shape: {sofa_cohort_pl.shape}")
        print(f"Cohort columns: {sofa_cohort_pl.columns}")

        # Compute SOFA scores using Polars
        print("\n" + "="*60)
        print("Computing SOFA scores with Polars...")
        print("="*60 + "\n")

        sofa_scores_pl = compute_sofa_polars(
            data_directory=config['tables_path'],
            cohort_df=sofa_cohort_pl,
            filetype=config['file_type'],
            id_name='encounter_block',
            extremal_type='worst',
            fill_na_scores_with_zero=True,
            remove_outliers=True,
            timezone=config['timezone']
        )

        # Convert back to Pandas for compatibility with rest of code
        sofa_scores = sofa_scores_pl.to_pandas()

        print(f"\n✅ SOFA computation complete!")
        print(f"Result shape: {sofa_scores.shape}")
        print(f"\nFirst few rows:")
        print(sofa_scores.head())

        # Note: death_enc will come from final_tableone_df when we merge later
        # For now, get death_enc temporarily for mortality calculations
        sofa_scores_with_death = sofa_scores.merge(
            final_tableone_df[['encounter_block', 'death_enc']],
            how='left',
            on='encounter_block'
        )

        #  Prepare the data
        # Group by SOFA score and calculate mortality rate and counts
        sofa_mortality = sofa_scores_with_death.groupby('sofa_total').agg({
            'death_enc': ['mean', 'count']
        }).reset_index()

        sofa_mortality.columns = ['sofa_score', 'mortality_rate', 'count']
        sofa_mortality['mortality_rate'] = sofa_mortality['mortality_rate'] * 100  # Convert to percentage

        # Step 2: Calculate confidence intervals (optional, for error bars)
        # Using Wilson score interval for binomial proportions
        def wilson_ci(successes, n, confidence=0.95):
            z = stats.norm.ppf((1 + confidence) / 2)
            p_hat = successes / n
            denominator = 1 + z**2 / n
            center = (p_hat + z**2 / (2*n)) / denominator
            margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4*n)) / n) / denominator
            return center * 100, margin * 100

        # Calculate number of deaths per score
        sofa_mortality['deaths'] = (sofa_mortality['mortality_rate'] / 100) * sofa_mortality['count']

        # Calculate confidence intervals
        ci_data = [wilson_ci(deaths, n) if n > 0 else (0, 0) 
                   for deaths, n in zip(sofa_mortality['deaths'], sofa_mortality['count'])]
        sofa_mortality['ci_center'] = [x[0] for x in ci_data]
        sofa_mortality['ci_margin'] = [x[1] for x in ci_data]

        # Step 3: Create the plot
        fig, ax = plt.subplots(figsize=(14, 6))

        # Create bar chart
        bars = ax.bar(sofa_mortality['sofa_score'], 
                      sofa_mortality['mortality_rate'],
                      color='#7FA8B8',  # Steel blue color similar to the image
                      edgecolor='black',
                      linewidth=0.5,
                      alpha=0.9)

        # Add error bars
        ax.errorbar(sofa_mortality['sofa_score'], 
                    sofa_mortality['mortality_rate'],
                    yerr=sofa_mortality['ci_margin'],
                    fmt='none',
                    ecolor='black',
                    capsize=3,
                    capthick=1,
                    alpha=0.7)

        # Customize the plot
        ax.set_xlabel('SOFA Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mortality, %', fontsize=12, fontweight='bold')
        ax.set_title('Mortality by SOFA Score (First 24hr of ICU admission)', fontsize=14, fontweight='bold', pad=20)

        # Set y-axis limits
        ax.set_ylim(0, 100)

        # Add grid for readability
        ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='gray')
        ax.set_axisbelow(True)

        # Set x-axis ticks to show all SOFA scores
        ax.set_xticks(range(int(sofa_mortality['sofa_score'].min()), 
                            int(sofa_mortality['sofa_score'].max()) + 1))

        # Add count labels below x-axis
        counts_text = '\n'.join([
            'No. of patients per score',
            '  '.join([f'{int(count)}' for count in sofa_mortality['count']])
        ])

        # Create a second table-like annotation below the plot
        fig.text(0.1, -0.05, 'No. of patients per score', 
                 ha='left', fontsize=10, weight='bold')

        # Add individual counts
        x_positions = np.linspace(0.15, 0.9, len(sofa_mortality))
        for i, (score, count) in enumerate(zip(sofa_mortality['sofa_score'], sofa_mortality['count'])):
            if i < len(x_positions):
                fig.text(x_positions[i], -0.08, f'{int(count)}', 
                        ha='center', fontsize=8)

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for patient counts

        # Save the figure
        plt.savefig(os.path.join(figures_dir, 'sofa_mortality_histogram.png'),
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()


        # Step 5: Prepare data for CSV export
        # Calculate lower and upper confidence interval bounds
        sofa_mortality['ci_lower'] = sofa_mortality['mortality_rate'] - sofa_mortality['ci_margin']
        sofa_mortality['ci_upper'] = sofa_mortality['mortality_rate'] + sofa_mortality['ci_margin']

        # Ensure CI bounds are within valid range [0, 100]
        sofa_mortality['ci_lower'] = sofa_mortality['ci_lower'].clip(lower=0)
        sofa_mortality['ci_upper'] = sofa_mortality['ci_upper'].clip(upper=100)
        sofa_mortality['total_encounters'] = sofa_scores['encounter_block'].nunique()
        # Create export dataframe with all relevant columns
        sofa_export = sofa_mortality[[
            'sofa_score', 
            'total_encounters',
            'count',
            'deaths',
            'mortality_rate', 
            'ci_lower',
            'ci_upper',
            'ci_margin'
        ]].copy()

        # Rename columns for clarity
        sofa_export.columns = [
            'sofa_score',
            'total_encounters',
            'n_encounters',
            'n_deaths',
            'mortality_rate_percent',
            'ci_lower_95',
            'ci_upper_95',
            'ci_margin_95'
        ]

        # Round numeric columns for readability
        sofa_export['n_encounters'] = sofa_export['n_encounters'].astype(int)
        sofa_export['total_encounters'] = sofa_export['total_encounters'].astype(int)
        sofa_export['n_deaths'] = sofa_export['n_deaths'].round(0).astype(int)
        sofa_export['mortality_rate_percent'] = sofa_export['mortality_rate_percent'].round(2)
        sofa_export['ci_lower_95'] = sofa_export['ci_lower_95'].round(2)
        sofa_export['ci_upper_95'] = sofa_export['ci_upper_95'].round(2)
        sofa_export['ci_margin_95'] = sofa_export['ci_margin_95'].round(2)

        # Save to CSV
        output_path = os.path.join(output_dir, 'sofa_mortality_summary.csv')
        sofa_export.to_csv(output_path, index=False)

        print(f"\n=== SOFA Mortality Summary Saved ===")
        print(f"File saved to: {output_path}")

        # Join sofa_scores with final_tableone_df on 'encounter_block'
        # Note: sofa_scores doesn't have death_enc anymore, so no conflict
        final_tableone_df = final_tableone_df.merge(sofa_scores, on='encounter_block', how='left')

        # AGGRESSIVE MEMORY CLEANUP: Clear all SOFA-related data
        print("\n" + "="*80)
        print("AGGRESSIVE MEMORY CLEANUP AFTER SOFA COMPUTATION")
        print("="*80)
        print("Clearing SOFA computation data from memory...")

        # Delete SOFA computation variables
        del sofa_scores, sofa_mortality, sofa_export, sofa_cohort_df, sofa_cohort_ids, sofa_cohort_pl, sofa_scores_pl
        del ci_data

        # Force garbage collection
        gc.collect()
        print("✅ SOFA data cleared from memory")
        print("="*80 + "\n")

        checkpoint("SOFA Computation Complete")

    # ==============================================================================
    # # Outside of Table1- on whole dataset
    # ==============================================================================


    # ==============================================================================
    # ## Hospice Use vs Mortality Trends
    # ==============================================================================
    # CCI/hospice analysis only for overall critical-illness cohort.

    if cohort_mode == 'ward':
        print("\nSkipping CCI/hospice analysis (ward mode)")
    else:

        # ============================================================================
        # Prepare Comprehensive Hospice Trend Data
        # ============================================================================

        # Extract year from admission_dttm
        final_tableone_df['admission_year'] = final_tableone_df['admission_dttm'].dt.year

        # Create outcome variables (encounter-level, matching death_enc pattern)
        _hospice_mask = (final_tableone_df['discharge_category'].str.lower() == 'hospice')
        final_tableone_df['hospice_outcome'] = _hospice_mask.groupby(
            final_tableone_df['encounter_block']
        ).transform('any').astype(int)

        _expired_mask = (final_tableone_df['discharge_category'].str.lower() == 'expired')
        final_tableone_df['expired_outcome'] = _expired_mask.groupby(
            final_tableone_df['encounter_block']
        ).transform('any').astype(int)

        final_tableone_df['hospice_or_expired'] = (
            final_tableone_df['hospice_outcome'] | final_tableone_df['expired_outcome']
        ).astype(int)

        # ============================================================================
        # Aggregate All Metrics by Year (Single DataFrame)
        # ============================================================================

        # Deduplicate to one row per encounter for aggregations (outcome flags are
        # now encounter-level, so summing multi-row encounters would over-count)
        _enc_deduped = final_tableone_df.drop_duplicates(subset=['encounter_block'])

        hospice_trends = _enc_deduped.groupby('admission_year').agg({
            'encounter_block': 'count',  # Total encounters
            'hospice_outcome': 'sum',    # Hospice discharges
            'expired_outcome': 'sum',    # Deaths
            'hospice_or_expired': 'sum'  # Combined end-of-life
        }).reset_index()

        hospice_trends.columns = ['year', 'total_encounters', 'hospice', 'expired', 'hospice_or_expired']

        # Calculate percentages (of all encounters)
        hospice_trends['hospice_pct'] = (hospice_trends['hospice'] / hospice_trends['total_encounters'] * 100)
        hospice_trends['mortality_pct'] = (hospice_trends['expired'] / hospice_trends['total_encounters'] * 100)

        # Calculate hospice proportion among end-of-life patients
        hospice_trends['hospice_among_eol_pct'] = (
            hospice_trends['hospice'] / hospice_trends['hospice_or_expired'] * 100
        )

        # Calculate Wilson score confidence intervals
        def calculate_ci(successes, n, confidence=0.95):
            """Calculate Wilson score confidence interval for proportions"""
            if n == 0:
                return 0, 0
            ci_low, ci_upp = proportion_confint(successes, n, alpha=1-confidence, method='wilson')
            return ci_low * 100, ci_upp * 100

        # CIs for hospice among end-of-life
        ci_results = [
            calculate_ci(row['hospice'], row['hospice_or_expired'])
            for _, row in hospice_trends.iterrows()
        ]
        hospice_trends['hospice_among_eol_ci_lower'] = [x[0] for x in ci_results]
        hospice_trends['hospice_among_eol_ci_upper'] = [x[1] for x in ci_results]

        # ============================================================================
        # Save Results DataFrame
        # ============================================================================

        _hts_path = os.path.join(output_dir, 'hospice_trends_summary.csv')
        hospice_trends.to_csv(_hts_path, index=False)
        print(f"✅ Saved: {_hts_path}")

        # ============================================================================
        # Create Combined Figure
        # ============================================================================

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        x_labels = hospice_trends['year'].astype(str).tolist()
        x_pos = np.arange(len(x_labels))

        # ----------------------------------------------------------------------------
        # TOP PANEL: Mortality vs Hospice Trends (% of all encounters)
        # ----------------------------------------------------------------------------

        mortality_pct = hospice_trends['mortality_pct'].values
        hospice_pct = hospice_trends['hospice_pct'].values

        # Plot MORTALITY line
        ax1.plot(x_pos, mortality_pct, 
                 marker='o', markersize=14, 
                 color='#666666', 
                 linewidth=3,
                 markerfacecolor='#666666',
                 markeredgecolor='black',
                 markeredgewidth=1,
                 label='MORTALITY',
                 zorder=3)

        # Plot HOSPICE line
        ax1.plot(x_pos, hospice_pct,
                 marker='s', markersize=12,
                 color='#000000',
                 linewidth=3,
                 markerfacecolor='#000000',
                 markeredgecolor='black',
                 markeredgewidth=1,
                 label='HOSPICE',
                 zorder=3)

        # Add percentage labels
        for i, (mort, hosp) in enumerate(zip(mortality_pct, hospice_pct)):
            ax1.text(i, mort + 0.2, f'{mort:.2f}%', 
                     ha='center', va='bottom', 
                     fontsize=11, fontweight='bold',
                     color='#333333')
    
            ax1.text(i, hosp - 0.2, f'{hosp:.2f}%',
                     ha='center', va='top',
                     fontsize=11, fontweight='bold',
                     color='#000000')

        # Add text labels
        mid_y_mortality = np.mean(mortality_pct)
        mid_y_hospice = np.mean(hospice_pct)

        ax1.text(len(x_pos)/2, mid_y_mortality + 0.1, 'MORTALITY', 
                 fontsize=16, fontweight='normal',
                 color='#666666', ha='center',
                 style='italic')

        ax1.text(len(x_pos)/2, mid_y_hospice - 0.1, 'HOSPICE',
                 fontsize=16, fontweight='normal',
                 color='#000000', ha='center',
                 style='italic')

        # Customize axes
        ax1.set_ylabel('PERCENT OF ALL ENCOUNTERS', fontsize=14, fontweight='bold', labelpad=10)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(x_labels, fontsize=12, fontweight='bold')

        # Calculate ylim with defensive checks for NaN/Inf values
        hosp_min = np.nanmin(hospice_pct) if len(hospice_pct) > 0 and not np.all(np.isnan(hospice_pct)) else 0
        mort_max = np.nanmax(mortality_pct) if len(mortality_pct) > 0 and not np.all(np.isnan(mortality_pct)) else 10

        # Provide defaults if still invalid
        if np.isnan(hosp_min) or np.isinf(hosp_min):
            hosp_min = 0
        if np.isnan(mort_max) or np.isinf(mort_max):
            mort_max = 10

        ax1.set_ylim(hosp_min - 0.5, mort_max + 1)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_linewidth(2)
        ax1.spines['bottom'].set_linewidth(2)
        ax1.yaxis.grid(True, linestyle=':', alpha=0.3, linewidth=1, color='gray')
        ax1.set_axisbelow(True)
        ax1.set_title('End-of-Life Outcomes Over Time', fontsize=16, fontweight='bold', pad=15)

        # ----------------------------------------------------------------------------
        # BOTTOM PANEL: Hospice Proportion Among End-of-Life Patients
        # ----------------------------------------------------------------------------

        hospice_among_eol_pct = hospice_trends['hospice_among_eol_pct'].values
        ci_lower = hospice_trends['hospice_among_eol_ci_lower'].values
        ci_upper = hospice_trends['hospice_among_eol_ci_upper'].values

        # Plot line with confidence interval
        ax2.plot(x_pos, hospice_among_eol_pct,
                 marker='D', markersize=12,
                 color='#2E86AB',
                 linewidth=3,
                 markerfacecolor='#2E86AB',
                 markeredgecolor='black',
                 markeredgewidth=1,
                 zorder=3)

        # Add confidence interval shading
        ax2.fill_between(x_pos, ci_lower, ci_upper,
                         alpha=0.3, color='#2E86AB')

        # Add percentage labels
        for i, pct in enumerate(hospice_among_eol_pct):
            ax2.text(i, pct + 1.5, f'{pct:.1f}%',
                     ha='center', va='bottom',
                     fontsize=11, fontweight='bold',
                     color='#2E86AB')

        # Customize axes
        ax2.set_xlabel('YEAR', fontsize=14, fontweight='bold', labelpad=10)
        ax2.set_ylabel('PERCENT', fontsize=14, fontweight='bold', labelpad=10)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(x_labels, fontsize=12, fontweight='bold')

        # Calculate ylim with defensive checks for NaN/Inf values
        ci_upper_max = np.nanmax(ci_upper) if len(ci_upper) > 0 and not np.all(np.isnan(ci_upper)) else 100
        if np.isnan(ci_upper_max) or np.isinf(ci_upper_max):
            ci_upper_max = 100  # Default to 100% for percentage plot

        ax2.set_ylim(0, ci_upper_max + 5)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_linewidth(2)
        ax2.spines['bottom'].set_linewidth(2)
        ax2.yaxis.grid(True, linestyle=':', alpha=0.3, linewidth=1, color='gray')
        ax2.set_axisbelow(True)
        ax2.set_title('Hospice Discharge Among End-of-Life Patients\n(Hospice / [Hospice + Expired])',
                     fontsize=16, fontweight='bold', pad=15)

        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'hospice_mortality_combined_trends.png'),
                    dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()


        print("\n✅ Hospice trends analysis complete!")


        # ==============================================================================
        # ## CCI Hospice Trends
        # ==============================================================================


        # ============================================================================
        # Merge CCI Results with Final TableOne (if not already done)
        # ============================================================================
        # final_tableone_df['admission_year'] = final_tableone_df['admission_dttm'].dt.year
        # tableone_with_cci = final_tableone_df.merge(
        #     cci_results[['hospitalization_id', 'cci_score']], 
        #     on='hospitalization_id', 
        #     how='left'
        # )
        tableone_with_cci = final_tableone_df[['encounter_block','cci_score','admission_year',
                                                'expired_outcome', 'hospice_or_expired', 'hospice_outcome']]
        # Create CCI Categories
        tableone_with_cci['cci_category'] = pd.cut(
            tableone_with_cci['cci_score'],
            bins=[-np.inf, 0, 2, 4, np.inf],
            labels=['0 (No comorbidity)', '1-2 (Mild)', '3-4 (Moderate)', '5+ (Severe)']
        )

        # ============================================================================
        # Create Comprehensive Summary by CCI Category and Year
        # ============================================================================

        cci_summary = tableone_with_cci.groupby(['admission_year', 'cci_category']).agg({
            'encounter_block': 'count',
            'hospice_outcome': 'sum',
            'expired_outcome': 'sum',
            'hospice_or_expired': 'sum'
        }).reset_index()

        cci_summary.columns = ['year', 'cci_category', 'total_encounters', 
                               'hospice_count', 'expired_count', 'hospice_or_expired_count']

        # Calculate all key metrics
        cci_summary['mortality_pct'] = (
            cci_summary['expired_count'] / cci_summary['total_encounters'] * 100
        )
        cci_summary['hospice_pct'] = (
            cci_summary['hospice_count'] / cci_summary['total_encounters'] * 100
        )
        cci_summary['combined_eol_pct'] = (
            cci_summary['hospice_or_expired_count'] / cci_summary['total_encounters'] * 100
        )
        cci_summary['hospice_among_eol_pct'] = (
            cci_summary['hospice_count'] / cci_summary['hospice_or_expired_count'] * 100
        )
        cci_summary['hospice_capture_rate'] = (
            cci_summary['hospice_count'] / cci_summary['expired_count'] * 100
        )

        # Calculate confidence intervals for hospice among EOL
        def calculate_ci(row):
            if row['hospice_or_expired_count'] == 0:
                return pd.Series({'hospice_eol_ci_lower': np.nan, 'hospice_eol_ci_upper': np.nan})
            ci_low, ci_upp = proportion_confint(
                row['hospice_count'], row['hospice_or_expired_count'], 
                alpha=0.05, method='wilson'
            )
            return pd.Series({
                'hospice_eol_ci_lower': ci_low * 100, 
                'hospice_eol_ci_upper': ci_upp * 100
            })

        cci_summary[['hospice_eol_ci_lower', 'hospice_eol_ci_upper']] = cci_summary.apply(calculate_ci, axis=1)

        # Reorder columns for clarity
        cci_summary = cci_summary[[
            'year', 'cci_category', 'total_encounters',
            'expired_count', 'mortality_pct',
            'hospice_count', 'hospice_pct',
            'hospice_or_expired_count', 'combined_eol_pct',
            'hospice_among_eol_pct', 'hospice_eol_ci_lower', 'hospice_eol_ci_upper',
            'hospice_capture_rate'
        ]]

        # Save comprehensive summary
        _cci_summary_path = os.path.join(output_dir, 'cci_hospice_mortality_comprehensive_summary.csv')
        cci_summary.to_csv(_cci_summary_path, index=False)
        print(f"✅ Saved: {_cci_summary_path}")

        # Save the plotting data ("data behind the figure") to a separate CSV file
        # This replicates the data used for each panel in the grid:
        plot_data = []
        categories = ['0 (No comorbidity)', '1-2 (Mild)', '3-4 (Moderate)', '5+ (Severe)']

        for category in categories:
            cat_data = cci_summary[cci_summary['cci_category'] == category].sort_values('year')
            plot_data.append(
                cat_data.assign(cci_category_label=category)
            )

        plot_data_df = pd.concat(plot_data, axis=0)

        _cci_plot_path = os.path.join(output_dir, 'cci_mortality_hospice_trends_by_year_category_plotdata.csv')
        plot_data_df.to_csv(_cci_plot_path, index=False)
        print(f"✅ Saved plotting data for figure: {_cci_plot_path}")

        # Display summary statistics
        print("\n" + "="*80)
        print("COMPREHENSIVE CCI HOSPICE-MORTALITY SUMMARY")
        print("="*80)
        print(f"\nTotal records: {len(cci_summary)}")
        print(f"Years covered: {cci_summary['year'].min():.0f} - {cci_summary['year'].max():.0f}")
        print(f"CCI categories: {cci_summary['cci_category'].nunique()}")
        print(f"\nTotal encounters across all years: {cci_summary['total_encounters'].sum():,}")
        print(f"Total deaths: {cci_summary['expired_count'].sum():,}")
        print(f"Total hospice discharges: {cci_summary['hospice_count'].sum():,}")

        # ============================================================================
        # Create Unified Visualization: 2x2 Grid with Mortality vs Hospice
        # ============================================================================

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        axes = axes.flatten()

        colors_mortality = '#666666'
        colors_hospice = '#000000'

        for i, (category, ax) in enumerate(zip(categories, axes)):
            data = cci_summary[cci_summary['cci_category'] == category].sort_values('year')

            # Check if data is empty or insufficient
            if data.empty or len(data) == 0:
                ax.text(0.5, 0.5, f'No data available\nfor CCI: {category}',
                        ha='center', va='center', fontsize=12,
                        fontweight='bold', color='#666666',
                        transform=ax.transAxes)
                ax.set_title(f'CCI: {category}', fontsize=14, fontweight='bold', pad=10)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                continue

            x_labels = data['year'].astype(str).tolist()
            x_pos = np.arange(len(x_labels))

            mortality_pct = data['mortality_pct'].values
            hospice_pct = data['hospice_pct'].values

            # Plot MORTALITY line
            ax.plot(x_pos, mortality_pct,
                    marker='o', markersize=12,
                    color=colors_mortality,
                    linewidth=3,
                    markerfacecolor=colors_mortality,
                    markeredgecolor='black',
                    markeredgewidth=1,
                    label='MORTALITY',
                    zorder=3)

            # Plot HOSPICE line
            ax.plot(x_pos, hospice_pct,
                    marker='s', markersize=10,
                    color=colors_hospice,
                    linewidth=3,
                    markerfacecolor=colors_hospice,
                    markeredgecolor='black',
                    markeredgewidth=1,
                    label='HOSPICE',
                    zorder=3)

            # Add percentage labels
            for j, (mort, hosp) in enumerate(zip(mortality_pct, hospice_pct)):
                ax.text(j, mort + 0.5, f'{mort:.1f}%',
                        ha='center', va='bottom',
                        fontsize=9, fontweight='bold',
                        color='#333333')

                ax.text(j, hosp - 0.5, f'{hosp:.1f}%',
                        ha='center', va='top',
                        fontsize=9, fontweight='bold',
                        color='#000000')

            # Add text labels in plot area
            mid_y_mortality = np.mean(mortality_pct)
            mid_y_hospice = np.mean(hospice_pct)

            if mid_y_mortality - mid_y_hospice > 3:
                ax.text(len(x_pos)/2, mid_y_mortality + 1, 'MORTALITY',
                        fontsize=13, fontweight='normal',
                        color='#666666', ha='center',
                        style='italic', alpha=0.7)

                ax.text(len(x_pos)/2, mid_y_hospice - 1, 'HOSPICE',
                        fontsize=13, fontweight='normal',
                        color='#000000', ha='center',
                        style='italic', alpha=0.7)

            # Customize axes
            ax.set_ylabel('PERCENT OF ALL ENCOUNTERS', fontsize=11, fontweight='bold')
            ax.set_xlabel('YEAR', fontsize=11, fontweight='bold')
            ax.set_title(f'CCI: {category}', fontsize=14, fontweight='bold', pad=10)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, fontsize=10, fontweight='bold')

            # Calculate y_max with defensive checks for NaN/Inf values
            mort_max = np.nanmax(mortality_pct) if len(mortality_pct) > 0 and not np.all(np.isnan(mortality_pct)) else 0
            hosp_max = np.nanmax(hospice_pct) if len(hospice_pct) > 0 and not np.all(np.isnan(hospice_pct)) else 0
            y_max = max(mort_max, hosp_max)

            # Provide default if still invalid
            if np.isnan(y_max) or np.isinf(y_max) or y_max == 0:
                y_max = 10  # Default axis range

            ax.set_ylim(0, y_max + 3)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)

            ax.yaxis.grid(True, linestyle=':', alpha=0.3, linewidth=1, color='gray')
            ax.set_axisbelow(True)

            if i == 0:
                ax.legend(loc='upper left', fontsize=10, frameon=True,
                         fancybox=True, shadow=True)

        plt.suptitle('Mortality vs Hospice Trends by Comorbidity Burden',
                     fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'cci_mortality_hospice_comprehensive.png'),
                    dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()

        checkpoint("Mortality Trends Complete")

    strobe_counts


    # ==============================================================================
    # # TableOne
    # ==============================================================================

    cols = final_tableone_df.columns
    for x in cols:
        print(x)

    # ============================================================================
    # OPTIMIZED: Prepare Data and Create Table One
    # ============================================================================


    # ============================================================================
    # Step 1: Prepare final_tableone_df for Table One (Optimized)
    # ============================================================================

    print("\n" + "="*80)
    print("PREPARING DATA FOR TABLE ONE")
    print("="*80)

    print(f"\nOriginal final_tableone_df shape: {final_tableone_df.shape}")
    print(f"Unique encounter_blocks: {final_tableone_df['encounter_block'].nunique()}")

    # ✅ OPTIMIZATION: Check and deduplicate in one step
    duplicates = final_tableone_df['encounter_block'].duplicated().sum()
    print(f"Duplicate encounter_blocks: {duplicates}")

    # Debug: encounters where death_enc=1 but a row has neither expired nor hospice outcome
    # (expired_outcome/hospice_outcome only exist in critical-illness mode, not ward)
    if cohort_mode != 'ward':
        _mort_cols = ['patient_id', 'hospitalization_id', 'encounter_block',
                      'admission_dttm', 'discharge_dttm', 'discharge_category',
                      'death_enc', 'expired_outcome', 'hospice_outcome']
        _death_rows = final_tableone_df[final_tableone_df['death_enc'] == 1][_mort_cols]
        _problem_blocks = _death_rows.groupby('encounter_block').filter(
            lambda g: ((g['expired_outcome'] == 0) & (g['hospice_outcome'] == 0)).any()
        ).sort_values(['encounter_block', 'admission_dttm'])

        os.makedirs(project_root / 'output/intermediate', exist_ok=True)
        _problem_blocks.to_csv(
            project_root / 'output/intermediate/mortality_dedup_debug.csv', index=False
        )
        print(f"\n🔍 Mortality dedup debug: {_problem_blocks['encounter_block'].nunique()} encounter_blocks "
              f"with death_enc=1 but missing outcome breakdown")
        print(f"   Saved: output/intermediate/mortality_dedup_debug.csv ({len(_problem_blocks)} rows)")

    if duplicates > 0:
        print("\n⚠️  Multiple rows per encounter_block found. Keeping last value...")
        tableone_df = final_tableone_df.drop_duplicates(subset=['encounter_block'], keep='last')
    else:
        tableone_df = final_tableone_df

    if cohort_mode != 'ward' and duplicates > 0 and len(_problem_blocks) > 0:
        _post = tableone_df[tableone_df['encounter_block'].isin(
            _problem_blocks['encounter_block'].unique()
        )][_mort_cols]
        _post.to_csv(
            project_root / 'output/intermediate/mortality_dedup_debug_post.csv', index=False
        )
        print(f"   Post-dedup: {len(_post)} rows saved to mortality_dedup_debug_post.csv")
        del _post
    if cohort_mode != 'ward':
        del _death_rows, _problem_blocks

    print(f"\nFinal tableone_df shape: {tableone_df.shape}")

    # ✅ OPTIMIZATION: Create patient demographics once, reuse throughout
    patient_df = tableone_df[['patient_id', 'race_category', 'ethnicity_category', 'sex_category', 'age_at_admission']].drop_duplicates('patient_id')
    print(f"Unique patients: {len(patient_df):,}")

    # ============================================================================
    # Step 2: OPTIMIZED Table One Generation Function
    # ============================================================================

    def make_table_one_optimized(df, patient_demographics, id_col='encounter_block'):
        """
        Optimized Table One generation - pre-computes values, minimizes iterations.
        """
        rows = []
    
        # ✅ PRE-COMPUTE: Common values used throughout
        N_enc = len(df)
        N_pat = len(patient_demographics)
        # Count unique hospitals
        N_hospitals = df['hospital_id'].nunique() if 'hospital_id' in df.columns else None

        # ✅ OPTIMIZATION: Compute all summations once
        flag_sums = {}
        for col in df.columns:
            if col.endswith('_flag') or col in ['icu_enc', 'death_enc', 'high_support_enc',
                                                 'vaso_support_enc', 'other_critically_ill',
                                                 'ward_no_critical_care',
                                                 'on_vent', 'on_crrt', 'hospice_outcome',
                                                 'expired_outcome',
                                                 'sepsis_events_by_sepsis_col',
                                                 'icu_episodes',
                                                 'imv_episodes']:
                if col in df.columns:
                    flag_sums[col] = df[col].sum()
    
        # ✅ OPTIMIZATION: Compute all medians/quantiles once for continuous vars
        continuous_stats = {}
        for col in ['age_at_admission', 'first_icu_los_days', 'hospital_length_of_stay_days', 'cci_score',
                    'p_f', 'p_f_imputed', 'sofa_cv_97', 'sofa_coag', 'sofa_liver', 
                    'sofa_resp', 'sofa_cns', 'sofa_renal', 'sofa_total']:
            if col in df.columns:
                data = df[col].dropna() if col != 'age_at_admission' else patient_demographics['age_at_admission'].dropna()
                if len(data) > 0:
                    continuous_stats[col] = {
                        'median': data.median(),
                        'q1': data.quantile(0.25),
                        'q3': data.quantile(0.75)
                    }
    
        # -------------------------------------------------------------------------
        # 1. Sample Size
        # -------------------------------------------------------------------------
        rows.append(("N: Encounter blocks", f"{N_enc:,}"))
        rows.append(("N: Unique patients", f"{N_pat:,}"))

        # Add hospital count if available
        if N_hospitals is not None:
            rows.append(("N: Hospitals", f"{N_hospitals:,}"))
    
        # -------------------------------------------------------------------------
        # 2. Demographics
        # -------------------------------------------------------------------------
        # Age
        if 'age_at_admission' in continuous_stats:
            s = continuous_stats['age_at_admission']
            rows.append(("Age at admission, median [Q1, Q3]",
                         f"{s['median']:.0f} [{s['q1']:.0f}, {s['q3']:.0f}]"))
    
        # ✅ OPTIMIZATION: Vectorized categorical function with pre-computed denominator
        def cat_n_pct_fast(data, col, title, denominator):
            vc = data[col].value_counts(dropna=False)
            for lvl, cnt in vc.items():
                pct = 100 * cnt / denominator
                lvl_str = str(lvl) if pd.notna(lvl) else 'Missing'
                rows.append((f"  {title}: {lvl_str}", f"{cnt:,} ({pct:.1f}%)"))
    
        cat_n_pct_fast(patient_demographics, 'race_category', 'Race', N_pat)
        cat_n_pct_fast(patient_demographics, 'ethnicity_category', 'Ethnicity', N_pat)
        cat_n_pct_fast(patient_demographics, 'sex_category', 'Sex', N_pat)
    
        # -------------------------------------------------------------------------
        # 3. Encounter Types (use pre-computed sums)
        # -------------------------------------------------------------------------
        rows.append(("Encounter Types", ""))

        # Base 4 rows: identical labels in both modes. "Other critically ill" keeps
        # its original meaning (died in ED/ward without ICU/vaso/resp escalation) —
        # the flag definition was tightened upstream to add an explicit death_enc==1
        # check, which is a no-op in critical-illness mode.
        encounter_type_rows = [
            ('icu_enc', 'ICU encounters'),
            ('high_support_enc', 'Advanced respiratory support'),
            ('vaso_support_enc', 'Vasoactive support'),
            ('other_critically_ill', 'Other critically ill'),
        ]
        # In ward mode add a 5th catch-all row for ward survivors with no critical
        # care intervention. Mutually exclusive with the other four.
        if cohort_mode == 'ward':
            encounter_type_rows.append(
                ('ward_no_critical_care', 'Ward only (survived, no critical care)')
            )

        for flag, label in encounter_type_rows:
            if flag in flag_sums:
                n = flag_sums[flag]
                rows.append((f"  {label}, n (%)", f"{n:,} ({100*n/N_enc:.1f}%)"))

        # Drop ICU episodes rows in ward mode (ICU-severity metric, not strata overlap)
        if cohort_mode != 'ward' and 'icu_episodes' in flag_sums:
            total_icu_eps = flag_sums['icu_episodes']
            enc_with_icu = (df['icu_episodes'] > 0).sum()
            rows.append(("ICU episodes, total n", f"{total_icu_eps:,}"))
            rows.append(("Encounters with >=1 ICU episode, n (%)",
                         f"{enc_with_icu:,} ({100*enc_with_icu/N_enc:.1f}%)"))
    
        # -------------------------------------------------------------------------
        # 4. Mortality (use pre-computed sums)
        # -------------------------------------------------------------------------
        if 'death_enc' in flag_sums:
            mort_n = flag_sums['death_enc']
            rows.append(("Hospital mortality, n (%)", f"{mort_n:,} ({100*mort_n/N_enc:.1f}%)"))
        
            for flag, label in [('hospice_outcome', 'Discharged to hospice'),
                                ('expired_outcome', 'Expired')]:
                if flag in flag_sums:
                    n = flag_sums[flag]
                    rows.append((f"  {label}, n (%)", f"{n:,} ({100*n/N_enc:.1f}%)"))
    
        # -------------------------------------------------------------------------
        # 5. Admission and Location
        # -------------------------------------------------------------------------
        cat_n_pct_fast(df, 'first_admission_location', 'First admission location', N_enc)
        if 'admission_type_category' in df.columns:
            cat_n_pct_fast(df, 'admission_type_category', 'Admission type', N_enc)
    
        # -------------------------------------------------------------------------
        # 6. Length of Stay (use pre-computed stats)
        # -------------------------------------------------------------------------
        # Drop "ICU length of stay" in ward mode (ICU-severity metric, not strata overlap)
        los_pairs = [('hospital_length_of_stay_days', 'Hospital length of stay (days)')]
        if cohort_mode != 'ward':
            los_pairs.insert(0, ('first_icu_los_days', 'ICU length of stay (days)'))
        for col, label in los_pairs:
            if col in continuous_stats:
                s = continuous_stats[col]
                rows.append((f"{label}, median [Q1, Q3]",
                            f"{s['median']:.1f} [{s['q1']:.1f}, {s['q3']:.1f}]"))
    
        # -------------------------------------------------------------------------
        # 7. Comorbidities
        # -------------------------------------------------------------------------
        if 'cci_score' in continuous_stats:
            s = continuous_stats['cci_score']
            rows.append(("Charlson Comorbidity Index, median [Q1, Q3]",
                         f"{s['median']:.0f} [{s['q1']:.0f}, {s['q3']:.0f}]"))
    
        # ✅ OPTIMIZATION: Get all comorbidity columns at once
        comorbidities = [
            'myocardial_infarction', 'congestive_heart_failure', 'peripheral_vascular_disease',
            'cerebrovascular_disease', 'dementia', 'chronic_pulmonary_disease',
            'connective_tissue_disease', 'peptic_ulcer_disease', 'mild_liver_disease',
            'diabetes_uncomplicated', 'diabetes_with_complications', 'hemiplegia',
            'renal_disease', 'cancer', 'moderate_severe_liver_disease',
            'metastatic_solid_tumor', 'aids'
        ]
    
        comorb_present = [c for c in comorbidities if c in df.columns]
        if comorb_present:
            # ✅ OPTIMIZATION: Sum all at once with one operation
            comorb_sums = df[comorb_present].sum()
            comorb_with_counts = comorb_sums[comorb_sums > 0].sort_values(ascending=False)
        
            if len(comorb_with_counts) > 0:
                rows.append(("Comorbidities, n (%)", ""))
                for comorb, n in comorb_with_counts.items():
                    name = comorb.replace('_', ' ').title()
                    rows.append((f"  {name}", f"{int(n):,} ({100*n/N_enc:.1f}%)"))
        # -------------------------------------------------------------------------
        # 8. SOFA Scores (skipped in ward mode — SOFA isn't computed for ward cohort)
        # -------------------------------------------------------------------------
        if cohort_mode != 'ward':
            rows.append(("SOFA Scores", ""))

        # Total SOFA
        if 'sofa_total' in continuous_stats:
            s = continuous_stats['sofa_total']
            rows.append(("  Total SOFA score, median [Q1, Q3]",
                         f"{s['median']:.1f} [{s['q1']:.1f}, {s['q3']:.1f}]"))
    
        # Individual SOFA components
        sofa_components = [
            ('sofa_resp', 'Respiratory'),
            ('sofa_coag', 'Coagulation'),
            ('sofa_liver', 'Liver'),
            ('sofa_cv_97', 'Cardiovascular'),
            ('sofa_cns', 'CNS'),
            ('sofa_renal', 'Renal')
        ]
    
        for col, label in sofa_components:
            if col in continuous_stats:
                s = continuous_stats[col]
                rows.append((f"    {label}, median [Q1, Q3]",
                            f"{s['median']:.1f} [{s['q1']:.1f}, {s['q3']:.1f}]"))
    
        # P/F ratio (optional, if you want to include it)
        if 'p_f' in continuous_stats:
            s = continuous_stats['p_f']
            rows.append(("  P/F ratio, median [Q1, Q3]",
                         f"{s['median']:.0f} [{s['q1']:.0f}, {s['q3']:.0f}]"))
    
        if 'p_f_imputed' in continuous_stats:
            s = continuous_stats['p_f_imputed']
            rows.append(("  P/F ratio (imputed), median [Q1, Q3]",
                         f"{s['median']:.0f} [{s['q1']:.0f}, {s['q3']:.0f}]"))
    
        # -------------------------------------------------------------------------
        # 8. CRRT
        # -------------------------------------------------------------------------
        if 'on_crrt' in flag_sums:
            crrt_n = flag_sums['on_crrt']
            rows.append(("CRRT, n (%)", f"{crrt_n:,} ({100*crrt_n/N_enc:.1f}%)"))

        # -------------------------------------------------------------------------
        # 8b. Sepsis (CDC Adult Sepsis Event)
        # -------------------------------------------------------------------------
        if 'sepsis_events_by_sepsis_col' in flag_sums:
            total_by_sepsis = flag_sums['sepsis_events_by_sepsis_col']
            rows.append(("Sepsis events (CDC ASE), n", f"{total_by_sepsis:,}"))
        if 'sepsis_events_by_sepsis_col' in df.columns:
            enc_with_sepsis = (df['sepsis_events_by_sepsis_col'] > 0).sum()
            rows.append(("Encounters with >=1 sepsis event, n (%)",
                         f"{enc_with_sepsis:,} ({100*enc_with_sepsis/N_enc:.1f}%)"))

        # -------------------------------------------------------------------------
        # 9. Mechanical Ventilation (skipped entirely in ward mode — IMV processing
        # block is skipped upstream so on_vent / vent_duration_hours / imv_episodes
        # / first_location_imv / initial_mode_category are not in the dataframe).
        # -------------------------------------------------------------------------
        if cohort_mode == 'ward':
            imv_n = 0
        else:
            imv_n = flag_sums.get('on_vent', 0)
            rows.append(("Invasive mechanical ventilation, n (%)", f"{imv_n:,} ({100*imv_n/N_enc:.1f}%)"))

        # Ventilator hours (total across all encounters in this subset)
        if 'vent_duration_hours' in df.columns:
            total_vent_hours = df['vent_duration_hours'].sum()
            vent_hours_millions = total_vent_hours / 1_000_000
            if vent_hours_millions >= 0.01:
                rows.append(("Ventilator hours (millions)", f"{vent_hours_millions:.2f}"))
            else:
                rows.append(("Ventilator hours", f"{total_vent_hours:,.0f}"))

        if imv_n > 0:
            # ✅ OPTIMIZATION: Filter once, reuse
            imv_subset = df[df['on_vent'] == 1]
        
            cat_n_pct_fast(imv_subset, 'first_location_imv', 'First location at IMV start', imv_n)
            cat_n_pct_fast(imv_subset, 'initial_mode_category', 'Initial ventilator mode', imv_n)
        
            # ✅ OPTIMIZATION: Compute all vent settings at once
            vent_settings = {
                'fio2_set': 'FiO2 (%)',
                'peep_set': 'PEEP (cmH2O)',
                'resp_rate_set': 'Respiratory rate (breaths/min)',
                'tidal_volume_set': 'Tidal volume (mL)'
            }
        
            for setting, label in vent_settings.items():
                med_col = f'{setting}_median'
                if med_col in imv_subset.columns:
                    # ✅ OPTIMIZATION: Calculate all quantiles in one go
                    setting_stats = imv_subset[[med_col, f'{setting}_q1', f'{setting}_q3']].median()
                    rows.append((f"  {label}, median [Q1, Q3]",
                                f"{setting_stats.iloc[0]:.1f} [{setting_stats.iloc[1]:.1f}, {setting_stats.iloc[2]:.1f}]"))

            # ── Extubation metrics (two-lookback/two-lookforward detection) ──
            if 'extubation_status' in imv_subset.columns:
                _extub_only = imv_subset[imv_subset['extubation_status'] == 'extubated']
                if len(_extub_only) > 0 and 'time_to_extubation_hours' in _extub_only.columns:
                    _t = _extub_only['time_to_extubation_hours'].dropna()
                    if len(_t) > 0:
                        rows.append(("Time to extubation (hrs), median [Q1, Q3]",
                                     f"{_t.median():.1f} [{_t.quantile(.25):.1f}, {_t.quantile(.75):.1f}]"))
                cat_n_pct_fast(imv_subset, 'extubation_status', 'Extubation outcome', imv_n)

            if 'pre_admission_imv' in df.columns:
                _n = int(df['pre_admission_imv'].fillna(0).sum())
                if _n > 0:
                    rows.append(("Pre-admit IMV (excluded from time-to-extubation), n (%)",
                                 f"{_n:,} ({100*_n/N_enc:.1f}%)"))

            if 'intubated_within_24hr_admit' in df.columns:
                _n = int(df['intubated_within_24hr_admit'].fillna(0).sum())
                rows.append(("Intubated ≤24hr of admission, n (%)",
                             f"{_n:,} ({100*_n/N_enc:.1f}%)"))

            if 'imv_episodes_n' in df.columns:
                _n_reint = int((df['imv_episodes_n'].fillna(0) > 1).sum())
                rows.append(("Reintubation (≥2 IMV episodes), n (%)",
                             f"{_n_reint:,} ({100*_n_reint/N_enc:.1f}%)"))

                if 'time_to_reintubation_hours' in imv_subset.columns:
                    _r = imv_subset['time_to_reintubation_hours'].dropna()
                    if len(_r) > 0:
                        rows.append(("  Time to reintubation (hrs), median [Q1, Q3]",
                                     f"{_r.median():.1f} [{_r.quantile(.25):.1f}, {_r.quantile(.75):.1f}]"))

                if 'extubation_failure_48hr' in imv_subset.columns and 'extubation_status' in imv_subset.columns:
                    _extubated = imv_subset[imv_subset['extubation_status'] == 'extubated']
                    if len(_extubated) > 0:
                        _n_fail = int(_extubated['extubation_failure_48hr'].fillna(0).sum())
                        rows.append(("  Extubation failure ≤48hr, n (% of extubated)",
                                     f"{_n_fail:,} ({100*_n_fail/len(_extubated):.1f}%)"))

        # -------------------------------------------------------------------------
        # 10. Vasopressors
        # -------------------------------------------------------------------------
        vaso_flags = {
            'norepinephrine': 'Norepinephrine',
            'epinephrine': 'Epinephrine',
            'phenylephrine': 'Phenylephrine',
            'vasopressin': 'Vasopressin',
            'dopamine': 'Dopamine'
        }
    
        # ✅ OPTIMIZATION: Check if any vasopressor flags exist first
        vaso_flag_cols = [f'{v}_flag' for v in vaso_flags.keys()]
        vaso_flags_present = [c for c in vaso_flag_cols if c in df.columns]
    
        if vaso_flags_present:
            # ✅ OPTIMIZATION: Compute any_vasopressor using max (vectorized)
            vaso_enc_n = df[vaso_flags_present].max(axis=1).sum()
            rows.append(("Vasopressor encounters, n (%)", f"{vaso_enc_n:,} ({100*vaso_enc_n/N_enc:.1f}%)"))
        
            for vaso, name in vaso_flags.items():
                flag_col = f'{vaso}_flag'
                if flag_col in flag_sums:
                    n = flag_sums[flag_col]
                    rows.append((f"  {name}, n (%)", f"{n:,} ({100*n/N_enc:.1f}%)"))
                
                    # Dose statistics
                    if f'{vaso}_median' in df.columns and n > 0:
                        vaso_subset = df[df[flag_col] == 1]
                        vaso_dose_stats = vaso_subset[[f'{vaso}_median', f'{vaso}_q1', f'{vaso}_q3']].median()
                        rows.append((f"    {name} dose (mcg/kg/min), median [Q1, Q3]",
                                    f"{vaso_dose_stats.iloc[0]:.2f} [{vaso_dose_stats.iloc[1]:.2f}, {vaso_dose_stats.iloc[2]:.2f}]"))
    
        # -------------------------------------------------------------------------
        # 11. Sedatives and Analgesics (use pre-computed sums)
        # -------------------------------------------------------------------------
        sedation_meds = {
            'propofol_flag': 'Propofol',
            'midazolam_flag': 'Midazolam',
            'lorazepam_flag': 'Lorazepam',
            'dexmedetomidine_flag': 'Dexmedetomidine',
            'fentanyl_flag': 'Fentanyl'
        }
    
        sed_present = {k: v for k, v in sedation_meds.items() if k in flag_sums}
        if sed_present:
            rows.append(("Sedatives and analgesics, n (%)", ""))
            for flag_col, name in sed_present.items():
                n = flag_sums[flag_col]
                rows.append((f"  {name}", f"{n:,} ({100*n/N_enc:.1f}%)"))
    
        # -------------------------------------------------------------------------
        # 12. Neuromuscular Blocking Agents (use pre-computed sums)
        # -------------------------------------------------------------------------
        nmba_meds = {
            'cisatracurium_flag': 'Cisatracurium',
            'rocuronium_flag': 'Rocuronium'
        }
    
        nmba_present = {k: v for k, v in nmba_meds.items() if k in flag_sums}
        if nmba_present:
            rows.append(("Neuromuscular blocking agents, n (%)", ""))
            for flag_col, name in nmba_present.items():
                n = flag_sums[flag_col]
                rows.append((f"  {name}", f"{n:,} ({100*n/N_enc:.1f}%)"))

        # -------------------------------------------------------------------------
        # 13. Medications during IMV
        # -------------------------------------------------------------------------
        if 'on_vent' in df.columns:
            imv_df = df[df['on_vent'] == 1]
            N_imv = len(imv_df)

            if N_imv > 0:
                rows.append(("Medications during IMV (N={:,})".format(N_imv), ""))

                # Vasopressors during IMV
                imv_vaso_flags_present = [f'{v}_flag' for v in vaso_flags.keys() if f'{v}_flag' in imv_df.columns]

                if imv_vaso_flags_present:
                    imv_vaso_n = imv_df[imv_vaso_flags_present].max(axis=1).sum()
                    rows.append(("  Vasopressors, n (%)", f"{imv_vaso_n:,} ({100*imv_vaso_n/N_imv:.1f}%)"))

                    for vaso, name in vaso_flags.items():
                        flag_col = f'{vaso}_flag'
                        if flag_col in imv_df.columns:
                            n = imv_df[flag_col].sum()
                            if n > 0:
                                rows.append((f"    {name}, n (%)", f"{n:,} ({100*n/N_imv:.1f}%)"))
                                if f'{vaso}_median' in imv_df.columns:
                                    vaso_imv_subset = imv_df[imv_df[flag_col] == 1]
                                    vaso_dose_stats = vaso_imv_subset[[f'{vaso}_median', f'{vaso}_q1', f'{vaso}_q3']].median()
                                    rows.append((f"      {name} dose (mcg/kg/min), median [Q1, Q3]",
                                                f"{vaso_dose_stats.iloc[0]:.2f} [{vaso_dose_stats.iloc[1]:.2f}, {vaso_dose_stats.iloc[2]:.2f}]"))

                # Sedatives and analgesics during IMV
                imv_sed_present = {k: v for k, v in sedation_meds.items() if k in imv_df.columns}
                if imv_sed_present:
                    rows.append(("  Sedatives and analgesics, n (%)", ""))
                    for flag_col, name in imv_sed_present.items():
                        n = imv_df[flag_col].sum()
                        if n > 0:
                            rows.append((f"    {name}", f"{n:,} ({100*n/N_imv:.1f}%)"))

                # NMBAs during IMV
                imv_nmba_present = {k: v for k, v in nmba_meds.items() if k in imv_df.columns}
                if imv_nmba_present:
                    rows.append(("  Neuromuscular blocking agents, n (%)", ""))
                    for flag_col, name in imv_nmba_present.items():
                        n = imv_df[flag_col].sum()
                        if n > 0:
                            rows.append((f"    {name}", f"{n:,} ({100*n/N_imv:.1f}%)"))

        # Assemble DataFrame
        return pd.DataFrame(rows, columns=["Variable", "Overall"])


    # ============================================================================
    # Step 3: Generate Table One (Optimized)
    # ============================================================================

    print("\n" + "="*80)
    print("GENERATING TABLE ONE (OPTIMIZED)")
    print("="*80)

    # Generate overall table
    tbl_overall = make_table_one_optimized(tableone_df, patient_df)

    print("\n" + "="*80)
    print("TABLE ONE - OVERALL")
    print("="*80)
    print(tbl_overall.to_string(index=False))

    # Save
    _t1o_path = os.path.join(output_dir, 'table_one_overall.csv')
    tbl_overall.to_csv(_t1o_path, index=False)
    print(f"\n✅ Saved: {_t1o_path}")

    checkpoint("Overall Table One Computed")

    # ============================================================================
    # Step 4: Generate by Year (Optimized)
    # ============================================================================

    if 'admission_year' in tableone_df.columns:
        print("\n" + "="*80)
        print("GENERATING TABLE ONE BY YEAR (OPTIMIZED)")
        print("="*80)
    
        # ✅ OPTIMIZATION: Get unique years once, sort once
        years = sorted(tableone_df['admission_year'].dropna().unique())
        print(f"Years found: {years}")
    
        var_order = tbl_overall["Variable"].tolist()
        results = {"Overall": tbl_overall.set_index("Variable")["Overall"]}
    
        # ✅ OPTIMIZATION: Group once, iterate over groups
        for yr in years:
            # Filter year data
            df_year = tableone_df[tableone_df["admission_year"] == yr]
            pat_year = patient_df[patient_df['patient_id'].isin(df_year['patient_id'])]
        
            # Generate table for this year
            tbl_year = make_table_one_optimized(df_year, pat_year)
            results[str(int(yr))] = tbl_year.set_index("Variable")["Overall"]
    
        # Create wide DataFrame
        table_by_year = (
            pd.DataFrame(results)
            .reindex(var_order)
            .reset_index()
            .rename(columns={"index": "Variable"})
        )
    
        print("\n" + "="*80)
        print("TABLE ONE - BY YEAR")
        print("="*80)
        print(table_by_year.to_string(index=False))
    
        # Save
        _t1by_path = os.path.join(output_dir, 'table_one_by_year.csv')
        table_by_year.to_csv(_t1by_path, index=False)
        print(f"\n✅ Saved: {_t1by_path}")

    # ============================================================================
    # Step 4b: Compute PF/SF ratios for advanced_resp strata
    # ============================================================================
    pf_sf_per_encounter = None
    if cohort_mode != 'ward' and resp_failure_onset_df is not None and len(resp_failure_onset_df) > 0:
        print("\n" + "="*80)
        print("COMPUTING PF/SF RATIOS (first 24h of respiratory failure)")
        print("="*80)
        from modules.tableone.pf_sf_calculator import calculate_pf_sf_ratios
        try:
            pf_sf_per_encounter = calculate_pf_sf_ratios(
                onset_df=resp_failure_onset_df,
                data_directory=config['tables_path'],
                filetype=config['file_type'],
                timezone=config['timezone'],
                id_col='encounter_block'
            )
            print(f"PF/SF ratios computed for {len(pf_sf_per_encounter):,} encounters")
        except Exception as e:
            print(f"⚠️ PF/SF calculation failed: {e}")
            import traceback
            traceback.print_exc()
        checkpoint("PF/SF Ratios Computed")

    # ============================================================================
    # Step 5: Generate Table Ones by Encounter Type (with year columns)
    # ============================================================================

    # Stratified by-year Table Ones (ICU / advanced_resp / vaso / deaths) are
    # not wanted in ward mode (Phase 2 Decision P2-1). The loop is wrapped in a
    # 0-or-1 iteration for-loop so the existing code keeps its indentation but
    # the body iterates 0 times in ward mode.
    if cohort_mode == 'ward':
        print("\nSkipping stratified by-year Table Ones (ward mode)")

    for _strat_by_year_iter in (range(1) if cohort_mode != 'ward' else range(0)):
        print("\n" + "="*80)
        print("GENERATING TABLE ONES BY ENCOUNTER TYPE")
        print("="*80)

        from modules.strata import ENCOUNTER_TYPE_STRATA as encounter_type_strata
        from modules.utils.output_paths import parse_stratum
        _pf_sf_strata_data = {}  # Collect PF/SF data for advanced_resp comparison figure
        _no_imv_pf_sf_data = {}  # Collect PF/SF data for no_imv comparison figure

        for stratum_name, col in encounter_type_strata.items():
            if col not in tableone_df.columns:
                print(f"  ⚠️ Skipping {stratum_name}: column '{col}' not found")
                continue

            df_strat = tableone_df[tableone_df[col] == 1]
            if len(df_strat) == 0:
                print(f"  ⚠️ Skipping {stratum_name}: no encounters")
                continue

            _, _strat_suffix = parse_stratum(stratum_name)

            pat_strat = patient_df[patient_df['patient_id'].isin(df_strat['patient_id'])]
            print(f"\n  {stratum_name}: {len(df_strat):,} encounters, {len(pat_strat):,} patients")

            # Overall for this stratum
            tbl_strat = make_table_one_optimized(df_strat, pat_strat)
            strat_var_order = tbl_strat['Variable'].tolist()
            strat_results = {'Overall': tbl_strat.set_index('Variable')['Overall']}

            # By year within stratum — only for top-level strata.  Sub-strata
            # (names containing '/', e.g. advanced_resp/icu, vaso/ed_icu) are
            # kept overall-only to cut runtime.
            _is_sub_stratum = '/' in stratum_name
            if not _is_sub_stratum and 'admission_year' in df_strat.columns:
                strat_years = sorted(df_strat['admission_year'].dropna().unique())
                for yr in strat_years:
                    df_yr = df_strat[df_strat['admission_year'] == yr]
                    pat_yr = pat_strat[pat_strat['patient_id'].isin(df_yr['patient_id'])]
                    if len(df_yr) > 0:
                        tbl_yr = make_table_one_optimized(df_yr, pat_yr)
                        strat_results[str(int(yr))] = tbl_yr.set_index('Variable')['Overall']

            strat_table = (
                pd.DataFrame(strat_results)
                .reindex(strat_var_order)
                .reset_index()
                .rename(columns={'index': 'Variable'})
            )

            # ── Pre/post respiratory device LOS rows ──────────────────────
            _DEVICE_LOS_STRATA = {
                'advanced_resp', 'advanced_resp/icu', 'advanced_resp/no_icu',
                'nippv_hfnc', 'nippv_hfnc/icu', 'nippv_hfnc/no_icu',
                'no_imv', 'no_imv/icu', 'no_imv/no_icu',
            }
            if (stratum_name in _DEVICE_LOS_STRATA
                    and resp_failure_onset_df is not None
                    and len(resp_failure_onset_df) > 0):
                _onset_cols = resp_failure_onset_df[['encounter_block', 'onset_dttm']]
                _strat_onset = df_strat.merge(_onset_cols, on='encounter_block', how='inner')

                if len(_strat_onset) > 0:
                    _strat_onset['pre_device_los_days'] = (
                        _strat_onset['onset_dttm'] - _strat_onset['admission_dttm']
                    ).dt.total_seconds() / 86400
                    _strat_onset['post_device_los_days'] = (
                        _strat_onset['discharge_dttm'] - _strat_onset['onset_dttm']
                    ).dt.total_seconds() / 86400

                    def _device_los_stat(series):
                        d = series.dropna()
                        if len(d) == 0:
                            return ''
                        return f"{d.median():.1f} [{d.quantile(0.25):.1f}, {d.quantile(0.75):.1f}]"

                    _los_new = {'Variable': [
                        'Resp. device onset, n (%)',
                        '  Pre-device LOS (days), median [Q1, Q3]',
                        '  Post-device LOS (days), median [Q1, Q3]',
                    ]}
                    for _col in strat_table.columns:
                        if _col == 'Variable':
                            continue
                        if _col == 'Overall':
                            _n = len(_strat_onset)
                            _N = len(df_strat)
                            _los_new[_col] = [
                                f"{_n:,} ({100*_n/_N:.1f}%)",
                                _device_los_stat(_strat_onset['pre_device_los_days']),
                                _device_los_stat(_strat_onset['post_device_los_days']),
                            ]
                        else:  # year columns
                            _yr_onset = _strat_onset[_strat_onset['admission_year'] == int(_col)]
                            _yr_strat = df_strat[df_strat['admission_year'] == int(_col)]
                            _n_yr = len(_yr_onset)
                            _N_yr = len(_yr_strat)
                            _pct = f"{100*_n_yr/_N_yr:.1f}%" if _N_yr > 0 else "0.0%"
                            _los_new[_col] = [
                                f"{_n_yr:,} ({_pct})",
                                _device_los_stat(_yr_onset['pre_device_los_days']),
                                _device_los_stat(_yr_onset['post_device_los_days']),
                            ]

                    _los_rows = pd.DataFrame(_los_new)
                    _los_idx = strat_table[
                        strat_table['Variable'].str.contains('length of stay', case=False, na=False)
                    ].index
                    _insert_pos = _los_idx[-1] + 1 if len(_los_idx) > 0 else len(strat_table)
                    strat_table = pd.concat([
                        strat_table.iloc[:_insert_pos],
                        _los_rows,
                        strat_table.iloc[_insert_pos:],
                    ], ignore_index=True)
                    print(f"  ✅ Added pre/post device LOS rows ({len(_strat_onset):,} with onset)")

            # ── 28-day VFD rows (any stratum with IMV encounters) ────────
            if 'vfd_28' in df_strat.columns:
                _vfd_data = df_strat['vfd_28'].dropna()
                if len(_vfd_data) > 0:
                    def _vfd_stat(series):
                        d = series.dropna()
                        if len(d) == 0:
                            return ''
                        return f"{d.median():.0f} [{d.quantile(0.25):.0f}, {d.quantile(0.75):.0f}]"

                    _vfd_new = {'Variable': [
                        '28-day VFD (IMV encounters), n (%)',
                        '  VFD, median [Q1, Q3]',
                    ]}
                    for _col in strat_table.columns:
                        if _col == 'Variable':
                            continue
                        if _col == 'Overall':
                            _n_imv = len(_vfd_data)
                            _N = len(df_strat)
                            _vfd_new[_col] = [
                                f"{_n_imv:,} ({100*_n_imv/_N:.1f}%)",
                                _vfd_stat(_vfd_data),
                            ]
                        else:  # year columns
                            _yr_vfd = df_strat.loc[
                                df_strat['admission_year'] == int(_col), 'vfd_28'
                            ].dropna()
                            _yr_N = len(df_strat[df_strat['admission_year'] == int(_col)])
                            _pct = f"{100*len(_yr_vfd)/_yr_N:.1f}%" if _yr_N > 0 else "0.0%"
                            _vfd_new[_col] = [
                                f"{len(_yr_vfd):,} ({_pct})",
                                _vfd_stat(_yr_vfd),
                            ]

                    _vfd_rows = pd.DataFrame(_vfd_new)
                    # Insert after pre/post device LOS rows, or after LOS section
                    _vfd_idx = strat_table[
                        strat_table['Variable'].str.contains(
                            'Post-device LOS|length of stay', case=False, na=False
                        )
                    ].index
                    _vfd_insert = _vfd_idx[-1] + 1 if len(_vfd_idx) > 0 else len(strat_table)
                    strat_table = pd.concat([
                        strat_table.iloc[:_vfd_insert],
                        _vfd_rows,
                        strat_table.iloc[_vfd_insert:],
                    ], ignore_index=True)
                    print(f"  ✅ Added VFD rows ({len(_vfd_data):,} IMV encounters, "
                          f"median VFD={_vfd_data.median():.0f})")

            # ── NEE rows (vaso strata) ─────────────────────────────────
            _NEE_STRATA = {
                'vaso', 'vaso/icu', 'vaso/no_icu',
                'vaso/ed_icu', 'vaso/ed_ward',
            }
            if stratum_name in _NEE_STRATA and 'nee_peak' in df_strat.columns:
                _nee_peak = df_strat['nee_peak'].dropna()
                _nee_med = df_strat['nee_median'].dropna() if 'nee_median' in df_strat.columns else pd.Series(dtype=float)
                if len(_nee_peak) > 0:
                    def _nee_stat(series):
                        d = series.dropna()
                        if len(d) == 0:
                            return ''
                        return f"{d.median():.3f} [{d.quantile(0.25):.3f}, {d.quantile(0.75):.3f}]"

                    _nee_new = {'Variable': [
                        'Peak NEE (mcg/kg/min), median [Q1, Q3]',
                        'Median NEE (mcg/kg/min), median [Q1, Q3]',
                    ]}
                    for _col in strat_table.columns:
                        if _col == 'Variable':
                            continue
                        if _col == 'Overall':
                            _nee_new[_col] = [
                                _nee_stat(_nee_peak),
                                _nee_stat(_nee_med),
                            ]
                        else:  # year columns
                            _yr_mask = df_strat['admission_year'] == int(_col)
                            _yr_peak = df_strat.loc[_yr_mask, 'nee_peak'].dropna()
                            _yr_med = df_strat.loc[_yr_mask, 'nee_median'].dropna() if 'nee_median' in df_strat.columns else pd.Series(dtype=float)
                            _nee_new[_col] = [
                                _nee_stat(_yr_peak),
                                _nee_stat(_yr_med),
                            ]

                    _nee_rows = pd.DataFrame(_nee_new)
                    # Insert after VFD rows, or after LOS section
                    _nee_idx = strat_table[
                        strat_table['Variable'].str.contains(
                            'VFD|Post-device LOS|length of stay', case=False, na=False
                        )
                    ].index
                    _nee_insert = _nee_idx[-1] + 1 if len(_nee_idx) > 0 else len(strat_table)
                    strat_table = pd.concat([
                        strat_table.iloc[:_nee_insert],
                        _nee_rows,
                        strat_table.iloc[_nee_insert:],
                    ], ignore_index=True)
                    print(f"  ✅ Added NEE rows ({len(_nee_peak):,} encounters, "
                          f"peak NEE median={_nee_peak.median():.3f})")

            # ── Time to ICU/Ward after first ED pressor ───────────────
            _ED_VASO_TIMING_STRATA = {'vaso/ed_icu', 'vaso/ed_ward'}
            if stratum_name in _ED_VASO_TIMING_STRATA and 'first_vaso_dttm' in df_strat.columns:
                _target_loc = 'icu' if stratum_name == 'vaso/ed_icu' else 'ward'
                _target_label = 'ICU' if _target_loc == 'icu' else 'Ward'

                # First post-pressor target-location in_dttm per encounter
                _timing_adt = adt_cohort[
                    adt_cohort['location_category'].str.contains(_target_loc, na=False)
                    if _target_loc == 'icu'
                    else adt_cohort['location_category'] == _target_loc
                ][['encounter_block', 'in_dttm']].copy()
                _timing_adt = _timing_adt.rename(columns={'in_dttm': 'target_in_dttm'})

                _timing = df_strat[['encounter_block', 'first_vaso_dttm']].dropna(
                    subset='first_vaso_dttm'
                ).merge(_timing_adt, on='encounter_block', how='inner')
                _timing = _timing[_timing['target_in_dttm'] > _timing['first_vaso_dttm']]

                if len(_timing) > 0:
                    # Keep earliest post-pressor target location per encounter
                    _timing = _timing.sort_values('target_in_dttm').drop_duplicates(
                        subset='encounter_block', keep='first'
                    )
                    _timing['hours_to_target'] = (
                        _timing['target_in_dttm'] - _timing['first_vaso_dttm']
                    ).dt.total_seconds() / 3600

                    def _timing_stat(series):
                        d = series.dropna()
                        if len(d) == 0:
                            return ''
                        return f"{d.median():.1f} [{d.quantile(0.25):.1f}, {d.quantile(0.75):.1f}]"

                    _timing_new = {'Variable': [
                        f'Time to {_target_label} after first pressor (hours), median [Q1, Q3]',
                    ]}
                    # Merge timing back to df_strat for year splits
                    _timing_by_enc = _timing[['encounter_block', 'hours_to_target']]
                    _df_with_timing = df_strat.merge(_timing_by_enc, on='encounter_block', how='left')

                    for _col in strat_table.columns:
                        if _col == 'Variable':
                            continue
                        if _col == 'Overall':
                            _timing_new[_col] = [_timing_stat(_df_with_timing['hours_to_target'])]
                        else:  # year columns
                            _yr_timing = _df_with_timing.loc[
                                _df_with_timing['admission_year'] == int(_col), 'hours_to_target'
                            ]
                            _timing_new[_col] = [_timing_stat(_yr_timing)]

                    _timing_rows = pd.DataFrame(_timing_new)
                    # Insert after NEE rows, or after LOS section
                    _timing_idx = strat_table[
                        strat_table['Variable'].str.contains(
                            'NEE|VFD|Post-device LOS|length of stay', case=False, na=False
                        )
                    ].index
                    _timing_insert = _timing_idx[-1] + 1 if len(_timing_idx) > 0 else len(strat_table)
                    strat_table = pd.concat([
                        strat_table.iloc[:_timing_insert],
                        _timing_rows,
                        strat_table.iloc[_timing_insert:],
                    ], ignore_index=True)
                    print(f"  ✅ Added time-to-{_target_label} row "
                          f"(median={_timing['hours_to_target'].median():.1f}h, "
                          f"n={len(_timing):,})")

            # Slugify stratum_name for the filename — sub-strata resolve to the
            # parent directory so no nested icu/no_icu subdirs are created.
            _strat_slug = stratum_name.replace('/', '_')
            out_path = os.path.join(str(_tableone_dir(stratum=stratum_name)),
                                      f'table_one_{_strat_slug}_by_year.csv')
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            strat_table.to_csv(out_path, index=False)
            print(f"  ✅ Saved: {out_path}")

            # PF/SF CSV output for advanced_resp strata
            _PFVSF_STRATA = {
                        'advanced_resp', 'advanced_resp/icu', 'advanced_resp/no_icu',
                        'no_imv', 'no_imv/icu', 'no_imv/no_icu',
                    }
            if stratum_name in _PFVSF_STRATA and pf_sf_per_encounter is not None and len(pf_sf_per_encounter) > 0:
                from modules.tableone.pf_sf_calculator import generate_aggregate_stats
                strat_enc_ids = df_strat['encounter_block'].unique()
                strat_pf_sf = pf_sf_per_encounter[pf_sf_per_encounter['encounter_block'].isin(strat_enc_ids)]
                if len(strat_pf_sf) > 0:
                    strat_tableone_dir = str(_tableone_dir(stratum=stratum_name))
                    os.makedirs(strat_tableone_dir, exist_ok=True)
                    # Per-encounter CSV
                    pf_sf_path = os.path.join(strat_tableone_dir, _suffixed('pf_sf_summary_24h.csv', _strat_suffix))
                    strat_pf_sf.to_csv(pf_sf_path, index=False)
                    print(f"  ✅ PF/SF per-encounter: {pf_sf_path}")
                    # Aggregate stats CSV
                    agg_stats = generate_aggregate_stats(strat_pf_sf)
                    agg_path = os.path.join(strat_tableone_dir, _suffixed('pf_sf_aggregate_stats.csv', _strat_suffix))
                    agg_stats.to_csv(agg_path, index=False)
                    print(f"  ✅ PF/SF aggregate stats: {agg_path}")
                    # Collect for comparison figures
                    _adv_fig_label = {'advanced_resp': 'Overall', 'advanced_resp/icu': 'ICU', 'advanced_resp/no_icu': 'No ICU'}
                    if stratum_name in _adv_fig_label:
                        _pf_sf_strata_data[_adv_fig_label[stratum_name]] = strat_pf_sf
                    _nimv_fig_label = {'no_imv': 'Overall', 'no_imv/icu': 'ICU', 'no_imv/no_icu': 'No ICU'}
                    if stratum_name in _nimv_fig_label:
                        _no_imv_pf_sf_data[_nimv_fig_label[stratum_name]] = strat_pf_sf

        # Generate PF/SF comparison figure across strata
        if len(_pf_sf_strata_data) > 0 and pf_sf_per_encounter is not None:
            from modules.tableone.pf_sf_calculator import generate_pf_sf_comparison_figure
            _fig_dir = os.path.join(str(project_root), 'output', 'final', 'strata', 'advanced_resp', 'figures')
            _fig_path = os.path.join(_fig_dir, 'pf_sf_comparison_overall_icu_noicu.png')
            try:
                generate_pf_sf_comparison_figure(_pf_sf_strata_data, _fig_path)
                print(f"  ✅ PF/SF comparison figure: {_fig_path}")
            except Exception as e:
                print(f"  ⚠️ PF/SF comparison figure failed: {e}")

        # Generate PF/SF comparison figure for no_imv strata
        if len(_no_imv_pf_sf_data) > 0 and pf_sf_per_encounter is not None:
            from modules.tableone.pf_sf_calculator import generate_pf_sf_comparison_figure
            _nimv_fig_dir = os.path.join(str(project_root), 'output', 'final', 'strata', 'no_imv', 'figures')
            _nimv_fig_path = os.path.join(_nimv_fig_dir, 'pf_sf_comparison_overall_icu_noicu.png')
            try:
                generate_pf_sf_comparison_figure(_no_imv_pf_sf_data, _nimv_fig_path)
                print(f"  ✅ PF/SF comparison figure (no_imv): {_nimv_fig_path}")
            except Exception as e:
                print(f"  ⚠️ PF/SF comparison figure (no_imv) failed: {e}")

        checkpoint("Stratified Table Ones Complete")

    # ============================================================================
    # NIH Enrollment Report — race × ethnicity × sex cross-tabulation
    # ============================================================================
    print("\n" + "="*80)
    print("GENERATING NIH ENROLLMENT REPORT")
    print("="*80)

    enrollment_report = crosstab_demographics(patient_df)
    print(enrollment_report.to_string())

    _dx_path = os.path.join(output_dir, 'demographic_crosstab_race_ethnicity_sex.csv')
    enrollment_report.to_csv(_dx_path)
    print(f"\n✅ Saved: {_dx_path}")

    print("\n" + "="*80)
    print("✅ TABLE ONE GENERATION COMPLETE")
    print("="*80)

    os.makedirs(project_root / 'output/intermediate', exist_ok=True)
    # Cohort-aware parquet filename: ward run writes to a parallel file so downstream
    # pipelines (ECDF, collection stats, MCIDE) that read final_tableone_df.parquet
    # via modules/strata.py continue to see the critical-illness cohort untouched.
    _final_parquet_name = (
        'final_tableone_ward_df.parquet'
        if cohort_mode == 'ward'
        else 'final_tableone_df.parquet'
    )
    final_tableone_df.to_parquet(project_root / 'output/intermediate' / _final_parquet_name)

    # ============================================================================
    # Step 6: Generate Stratified Summary CSVs by Encounter Type
    # ============================================================================

    # Stratified per-encounter-type subdirectory generation is broken in ward
    # mode (Phase 2 Decision P2-2): the loop body references resp_valid,
    # resp_imv_post_start, meds_merged, med_groups, and vent_settings, all of
    # which are defined inside the IMV / med-from-ICU plot for-loop wrappers
    # added in Phase 1 and therefore undefined when cohort_mode == "ward".
    # The try/except blocks would catch the NameErrors but produce 44 noisy
    # ❌ log lines and empty per-stratum subdirs. Skip the entire loop in ward
    # mode.
    if cohort_mode == 'ward':
        print("\nSkipping stratified per-encounter-type subdirectory generation (ward mode)")

    for _strat_subdir_iter in (range(1) if cohort_mode != 'ward' else range(0)):
        print("\n" + "="*80)
        print("GENERATING STRATIFIED SUMMARY CSVs BY ENCOUNTER TYPE")
        print("="*80)

        from modules.strata import ENCOUNTER_TYPE_STRATA
        from modules.utils.output_paths import parse_stratum as _parse_stratum

        for stratum_name, col in ENCOUNTER_TYPE_STRATA.items():
            if col not in final_tableone_df.columns:
                print(f"  ⚠️ Skipping {stratum_name}: column '{col}' not found")
                continue

            strat_df = final_tableone_df[final_tableone_df[col] == 1].copy()
            if len(strat_df) == 0:
                print(f"  ⚠️ Skipping {stratum_name}: no encounters")
                continue

            _, strat_suffix = _parse_stratum(stratum_name)

            strat_hosp_ids = set(strat_df['hospitalization_id'].unique())
            strat_patient_ids = set(strat_df['patient_id'].unique())
            strat_enc_blocks = set(strat_df['encounter_block'].unique())
            strat_output_dir = str(_tableone_dir(stratum=stratum_name))
            os.makedirs(strat_output_dir, exist_ok=True)

            n_enc = len(strat_df.drop_duplicates(subset=['encounter_block']))
            print(f"\n  {stratum_name}: {n_enc:,} encounters → {strat_output_dir}")

            # Filter intermediate DataFrames to this stratum using encounter_block
            try:
                strat_resp_valid = resp_valid[resp_valid['encounter_block'].isin(strat_enc_blocks)]
            except Exception:
                strat_resp_valid = pd.DataFrame()
            try:
                strat_resp_imv = resp_imv_post_start[resp_imv_post_start['encounter_block'].isin(strat_enc_blocks)]
            except Exception:
                strat_resp_imv = pd.DataFrame()
            try:
                strat_meds = meds_merged[meds_merged['encounter_block'].isin(strat_enc_blocks)]
            except Exception:
                strat_meds = pd.DataFrame()
            try:
                strat_cci = cci_results[cci_results['encounter_block'].isin(strat_enc_blocks)]
            except Exception:
                strat_cci = pd.DataFrame()
            strat_patient = patient_df[patient_df['patient_id'].isin(strat_patient_ids)]
            strat_icu_encounters = strat_df[strat_df['icu_enc'] == 1]['encounter_block'].nunique() if 'icu_enc' in strat_df.columns else 0

            try:
                generate_ventilator_settings_summary(strat_resp_valid, vent_settings, strat_output_dir, suffix=strat_suffix)
                print(f"    ✅ ventilator_settings")
            except Exception as e:
                print(f"    ❌ ventilator_settings: {e}")

            try:
                generate_tidal_volume_stats(strat_resp_imv, strat_output_dir, suffix=strat_suffix)
                print(f"    ✅ tidal_volume_stats")
            except Exception as e:
                print(f"    ❌ tidal_volume_stats: {e}")

            try:
                generate_pressure_control_stats(strat_resp_imv, strat_output_dir, suffix=strat_suffix)
                print(f"    ✅ pressure_control_stats")
            except Exception as e:
                print(f"    ❌ pressure_control_stats: {e}")

            try:
                generate_mode_proportions(strat_resp_imv, strat_output_dir, suffix=strat_suffix)
                print(f"    ✅ mode_proportions")
            except Exception as e:
                print(f"    ❌ mode_proportions: {e}")

            try:
                generate_medications_hourly(strat_meds, strat_icu_encounters, med_groups, strat_output_dir, suffix=strat_suffix)
                print(f"    ✅ medications_hourly")
            except Exception as e:
                print(f"    ❌ medications_hourly: {e}")

            try:
                generate_medications_summary(strat_meds, strat_icu_encounters, med_groups, strat_output_dir, suffix=strat_suffix)
                print(f"    ✅ medications_summary")
            except Exception as e:
                print(f"    ❌ medications_summary: {e}")

            try:
                generate_comorbidities(strat_cci, strat_output_dir, suffix=strat_suffix)
                print(f"    ✅ comorbidities")
            except Exception as e:
                print(f"    ❌ comorbidities: {e}")

            try:
                generate_sofa_mortality(strat_df, strat_output_dir, suffix=strat_suffix)
                print(f"    ✅ sofa_mortality")
            except Exception as e:
                print(f"    ❌ sofa_mortality: {e}")

            try:
                generate_demographic_crosstab(strat_patient, strat_output_dir, suffix=strat_suffix)
                print(f"    ✅ demographic_crosstab")
            except Exception as e:
                print(f"    ❌ demographic_crosstab: {e}")

        print("\n" + "="*80)
        print("✅ STRATIFIED SUMMARY GENERATION COMPLETE")
        print("="*80)

    # Final memory cleanup
    print("\nFinal memory cleanup...")
    plt.close('all')
    gc.collect()
    checkpoint("Table One Generation Complete")
    print("✅ Memory cleanup complete")

    # Return success
    checkpoint("Final Cleanup - Ready to Exit")
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)


