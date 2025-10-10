"""
Sampling utilities for CLIF Table One Analysis.

Provides functions to generate and manage stratified samples of ICU hospitalizations
for use with large CLIF tables.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Set, Optional
from datetime import datetime


def get_icu_hospitalizations_from_adt(adt_df: pd.DataFrame) -> Set[str]:
    """
    Extract hospitalization_ids that touched ICU at least once from ADT table.

    Parameters:
    -----------
    adt_df : pd.DataFrame
        ADT table dataframe with hospitalization_id and location_category columns

    Returns:
    --------
    set
        Set of unique hospitalization_ids that have location_category = 'icu'
    """
    if adt_df is None or adt_df.empty:
        return set()

    # Check required columns
    if 'hospitalization_id' not in adt_df.columns or 'location_category' not in adt_df.columns:
        raise ValueError("ADT dataframe must contain 'hospitalization_id' and 'location_category' columns")

    # Filter for ICU locations and get unique hospitalization_ids
    icu_mask = adt_df['location_category'] == 'icu'
    icu_hosp_ids = set(adt_df[icu_mask]['hospitalization_id'].dropna().unique())

    print(f"Found {len(icu_hosp_ids):,} unique ICU hospitalizations")

    return icu_hosp_ids


def generate_stratified_sample(
    hospitalization_df: pd.DataFrame,
    icu_hosp_ids: Set[str],
    sample_size: int = 1000
) -> List[str]:
    """
    Generate stratified sample of ICU hospitalizations proportional by year.

    Parameters:
    -----------
    hospitalization_df : pd.DataFrame
        Hospitalization table dataframe with hospitalization_id and admission_dttm
    icu_hosp_ids : set
        Set of hospitalization_ids that touched ICU
    sample_size : int
        Total number of hospitalizations to sample (default: 1000)

    Returns:
    --------
    list
        List of sampled hospitalization_ids
    """
    if hospitalization_df is None or hospitalization_df.empty:
        return []

    # Check required columns
    if 'hospitalization_id' not in hospitalization_df.columns:
        raise ValueError("Hospitalization dataframe must contain 'hospitalization_id' column")

    if 'admission_dttm' not in hospitalization_df.columns:
        raise ValueError("Hospitalization dataframe must contain 'admission_dttm' column")

    # Filter to only ICU hospitalizations
    icu_hosps = hospitalization_df[
        hospitalization_df['hospitalization_id'].isin(icu_hosp_ids)
    ].copy()

    if icu_hosps.empty:
        print("Warning: No ICU hospitalizations found in hospitalization table")
        return []

    # Extract year from admission_dttm
    icu_hosps['admission_year'] = pd.to_datetime(
        icu_hosps['admission_dttm'], errors='coerce'
    ).dt.year

    # Remove rows with invalid years
    icu_hosps = icu_hosps.dropna(subset=['admission_year'])

    if icu_hosps.empty:
        print("Warning: No valid admission dates found")
        return []

    # Calculate total ICU hospitalizations
    total_icu = len(icu_hosps)

    # If total ICU hospitalizations is less than sample size, return all
    if total_icu <= sample_size:
        print(f"Total ICU hospitalizations ({total_icu:,}) ≤ sample size ({sample_size:,}). Using all ICU hospitalizations.")
        return icu_hosps['hospitalization_id'].tolist()

    # Group by year and calculate proportional samples
    year_counts = icu_hosps.groupby('admission_year').size()

    # Calculate proportional sample per year
    year_samples = (year_counts / total_icu * sample_size).round().astype(int)

    # Adjust to ensure we get exactly sample_size
    # If rounding causes discrepancy, adjust the largest year
    sample_diff = sample_size - year_samples.sum()
    if sample_diff != 0:
        largest_year = year_counts.idxmax()
        year_samples[largest_year] += sample_diff

    # Sample from each year
    sampled_ids = []
    np.random.seed(42)  # For reproducibility

    for year, n_samples in year_samples.items():
        year_hosps = icu_hosps[icu_hosps['admission_year'] == year]['hospitalization_id'].values

        # Sample with replacement if needed (shouldn't be needed with proportional sampling)
        if n_samples > len(year_hosps):
            sampled = np.random.choice(year_hosps, size=n_samples, replace=True)
        else:
            sampled = np.random.choice(year_hosps, size=n_samples, replace=False)

        sampled_ids.extend(sampled.tolist())
        print(f"  Year {int(year)}: sampled {n_samples:,} / {len(year_hosps):,} ({n_samples/len(year_hosps)*100:.1f}%)")

    print(f"Total sampled: {len(sampled_ids):,} hospitalizations")

    return sampled_ids


def save_sample_list(hosp_ids: List[str], output_dir: str) -> None:
    """
    Save sample hospitalization_ids to intermediate folder.

    Parameters:
    -----------
    hosp_ids : list
        List of hospitalization_ids to save
    output_dir : str
        Base output directory (will create intermediate subfolder)
    """
    # Create intermediate directory
    intermediate_dir = Path(output_dir) / 'intermediate'
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    # Create sample file path
    sample_file = intermediate_dir / 'icu_sample_1k_hospitalizations.csv'

    # Create dataframe with metadata
    df = pd.DataFrame({'hospitalization_id': hosp_ids})

    # Add metadata as a header comment (will be saved as first row, then read with comment='#')
    metadata = [
        f"# ICU Sample - 1k Hospitalizations (Stratified Proportional by Year)",
        f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Total Count: {len(hosp_ids):,}",
        f"# Sampling Method: Stratified (proportional by admission year)",
        f"#"
    ]

    # Save with metadata as comments
    with open(sample_file, 'w') as f:
        for line in metadata:
            f.write(line + '\n')
        df.to_csv(f, index=False)

    print(f"✅ Saved sample to: {sample_file}")


def load_sample_list(output_dir: str) -> Optional[List[str]]:
    """
    Load existing sample hospitalization_ids from intermediate folder.

    Parameters:
    -----------
    output_dir : str
        Base output directory

    Returns:
    --------
    list or None
        List of hospitalization_ids if file exists, None otherwise
    """
    intermediate_dir = Path(output_dir) / 'intermediate'
    sample_file = intermediate_dir / 'icu_sample_1k_hospitalizations.csv'

    if not sample_file.exists():
        return None

    try:
        # Read CSV, skipping comment lines
        df = pd.read_csv(sample_file, comment='#')

        if 'hospitalization_id' not in df.columns:
            print(f"Warning: Sample file {sample_file} missing 'hospitalization_id' column")
            return None

        hosp_ids = df['hospitalization_id'].dropna().astype(str).tolist()
        print(f"📊 Loaded sample: {len(hosp_ids):,} hospitalizations")

        return hosp_ids

    except Exception as e:
        print(f"Error loading sample file: {e}")
        return None


def sample_exists(output_dir: str) -> bool:
    """
    Check if sample file exists in intermediate folder.

    Parameters:
    -----------
    output_dir : str
        Base output directory

    Returns:
    --------
    bool
        True if sample file exists, False otherwise
    """
    intermediate_dir = Path(output_dir) / 'intermediate'
    sample_file = intermediate_dir / 'icu_sample_1k_hospitalizations.csv'

    return sample_file.exists()
