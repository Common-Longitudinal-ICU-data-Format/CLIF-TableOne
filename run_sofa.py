#!/usr/bin/env python3
"""
Standalone SOFA Score Computation

This script computes SOFA scores independently from the main TableOne pipeline.
It reads the final_tableone_df from the intermediate folder, prepares the cohort,
computes SOFA scores, and saves the results back to the intermediate folder.

Usage:
    uv run run_sofa.py

Prerequisites:
    - TableOne must have been run at least once to generate final_tableone_df.parquet
    - Config file must be present with data paths
"""

import sys
import json
import pandas as pd
import polars as pl
from pathlib import Path
from datetime import datetime

# Import SOFA calculator
from modules.sofa.calculator import compute_sofa_polars


def load_config():
    """Load configuration from config.json."""
    config_path = Path("config/config.json")
    if not config_path.exists():
        raise FileNotFoundError(
            "config.json not found. Please ensure it exists in the project root."
        )

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config


def load_final_tableone():
    """Load final_tableone_df from intermediate folder."""
    intermediate_path = Path("output/intermediate/final_tableone_df_test.parquet")

    if not intermediate_path.exists():
        raise FileNotFoundError(
            f"Could not find {intermediate_path}.\n"
            "Please run TableOne first to generate this file:\n"
            "  uv run run_tableone.py"
        )

    print(f"Loading final_tableone_df from: {intermediate_path}")
    df = pd.read_parquet(intermediate_path)
    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")

    return df


def prepare_sofa_cohort(final_tableone_df):
    """
    Prepare SOFA cohort from final_tableone_df.

    Parameters
    ----------
    final_tableone_df : pd.DataFrame
        The final TableOne dataframe

    Returns
    -------
    pl.DataFrame
        Polars DataFrame with cohort ready for SOFA computation
    """
    print("\nPreparing SOFA cohort...")

    # Filter to ICU encounters only
    icu_df = final_tableone_df[final_tableone_df['icu_enc'] == 1].copy()
    print(f"  ICU encounters: {len(icu_df):,}")

    if len(icu_df) == 0:
        raise ValueError("No ICU encounters found in final_tableone_df")

    # Check required columns
    required_cols = ['hospitalization_id', 'encounter_block', 'first_icu_in_dttm']
    missing_cols = [col for col in required_cols if col not in icu_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Prepare cohort with time windows
    sofa_cohort_df = icu_df[required_cols].copy()
    sofa_cohort_df['start_dttm'] = sofa_cohort_df['first_icu_in_dttm']
    sofa_cohort_df['end_dttm'] = sofa_cohort_df['start_dttm'] + pd.Timedelta(hours=24)

    # Select final columns
    sofa_cohort_df = sofa_cohort_df[['hospitalization_id', 'encounter_block', 'start_dttm', 'end_dttm']]

    # Convert to Polars
    sofa_cohort_pl = pl.from_pandas(sofa_cohort_df)

    # Normalize hospitalization_id to Utf8 for consistency with data files
    # (prevents Utf8 vs LargeUtf8 type mismatch issues during joins)
    sofa_cohort_pl = sofa_cohort_pl.with_columns([
        pl.col('hospitalization_id').cast(pl.Utf8).alias('hospitalization_id')
    ])

    print(f"  Cohort shape: {sofa_cohort_pl.shape}")
    print(f"  Cohort columns: {sofa_cohort_pl.columns}")
    print(f"  Unique encounter blocks: {sofa_cohort_pl['encounter_block'].n_unique()}")

    return sofa_cohort_pl


def compute_sofa(config, cohort_df):
    """
    Compute SOFA scores.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    cohort_df : pl.DataFrame
        Cohort for SOFA computation

    Returns
    -------
    pl.DataFrame
        SOFA scores
    """
    print("\n" + "="*60)
    print("Computing SOFA scores with Polars...")
    print("="*60 + "\n")

    sofa_scores_pl = compute_sofa_polars(
        data_directory=config['tables_path'],
        cohort_df=cohort_df,
        filetype=config['file_type'],
        id_name='encounter_block',
        extremal_type='worst',
        fill_na_scores_with_zero=True,
        remove_outliers=True,
        timezone=config['timezone']
    )

    return sofa_scores_pl


def save_sofa_scores(sofa_scores_pl):
    """
    Save SOFA scores to intermediate folder.

    Parameters
    ----------
    sofa_scores_pl : pl.DataFrame
        SOFA scores to save
    """
    # Ensure intermediate directory exists
    intermediate_dir = Path("output/intermediate")
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    # Save as both parquet and csv
    parquet_path = intermediate_dir / "sofa_scores.parquet"

    print(f"\nSaving SOFA scores...")
    print(f"  Parquet: {parquet_path}")
    sofa_scores_pl.write_parquet(parquet_path)

    print(f"\n✅ SOFA scores saved successfully!")


def display_results_summary(sofa_scores_pl):
    """Display summary of SOFA computation results."""
    print("\n" + "="*60)
    print("SOFA Computation Complete!")
    print("="*60)
    print(f"\nResult shape: {sofa_scores_pl.shape}")
    print(f"Columns: {sofa_scores_pl.columns}")

    # Convert to pandas for easier display
    sofa_scores_pd = sofa_scores_pl.to_pandas()

    print(f"\nSample results (first 5 rows):")
    print(sofa_scores_pd.head().to_string())

    # Display SOFA component summary statistics
    score_cols = [col for col in sofa_scores_pd.columns if col.startswith('sofa_')]
    if score_cols:
        print(f"\nSOFA component summary:")
        print(sofa_scores_pd[score_cols].describe().round(2).to_string())


def main():
    """Main entry point for standalone SOFA computation."""
    start_time = datetime.now()

    try:
        print("="*60)
        print("Standalone SOFA Score Computation")
        print("="*60)
        print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Step 1: Load configuration
        print("Step 1: Loading configuration...")
        config = load_config()
        print(f"  Data directory: {config['tables_path']}")
        print(f"  File type: {config['file_type']}")
        print(f"  Timezone: {config['timezone']}")

        # Step 2: Load final_tableone_df
        print("\nStep 2: Loading final_tableone_df...")
        final_tableone_df = load_final_tableone()

        # Step 3: Prepare SOFA cohort
        print("\nStep 3: Preparing SOFA cohort...")
        sofa_cohort_pl = prepare_sofa_cohort(final_tableone_df)

        # Step 4: Compute SOFA scores
        print("\nStep 4: Computing SOFA scores...")
        sofa_scores_pl = compute_sofa(config, sofa_cohort_pl)

        # Step 5: Save results
        print("\nStep 5: Saving results...")
        save_sofa_scores(sofa_scores_pl)

        # Step 6: Display summary
        display_results_summary(sofa_scores_pl)

        # Completion message
        end_time = datetime.now()
        duration = end_time - start_time
        print("\n" + "="*60)
        print(f"Completed in {duration}")
        print("="*60)

        sys.exit(0)

    except FileNotFoundError as e:
        print(f"\n❌ File not found error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ Validation error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
