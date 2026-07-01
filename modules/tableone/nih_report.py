"""NIH-style demographic enrollment report (race × ethnicity × sex crosstab).

Produces the table that gets saved as
`output/final/.../tableone/demographic_crosstab_race_ethnicity_sex.csv` and
fed to the dashboard's Demographics tab.

Extracted from `generator.py` as a pure refactor.
"""

from __future__ import annotations

import os

import pandas as pd

from ._helpers import _suffixed


__all__ = ["crosstab_demographics", "generate_demographic_crosstab"]


def crosstab_demographics(patient_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulate race × ethnicity × sex for NIH enrollment reporting.

    Parameters:
        patient_df: One row per unique patient with columns
                    race_category, ethnicity_category, sex_category.

    Returns:
        DataFrame with race rows, (ethnicity, sex) MultiIndex columns,
        and margin totals. Cells with 1–9 are suppressed to '<10' and the
        affected row/col/grand totals are deducted by the same amount so
        visible cells still sum to displayed totals.
    """
    # CLIF 2.1 patient schema permissible values
    # CLIF 3.0 patient schema permissible values (lowercase snake_case)
    race_values = [
        "american_indian_or_alaska_native",
        "asian",
        "native_hawaiian_or_other_pacific_islander",
        "middle_eastern_or_north_african",
        "black_or_african_american",
        "white",
        "other",
        "unknown",
    ]
    ethnicity_values = ["non_hispanic", "hispanic", "unknown"]
    sex_values = ["female", "male", "unknown"]

    df = patient_df[["race_category", "ethnicity_category", "sex_category"]].copy()
    df.fillna("unknown", inplace=True)

    ct = pd.crosstab(
        df["race_category"],
        [df["ethnicity_category"], df["sex_category"]],
        margins=True,
        margins_name="Total",
    )

    # Build expected column ordering
    col_tuples = [(eth, sex) for eth in ethnicity_values for sex in sex_values]
    col_tuples.append(("Total", ""))
    desired_columns = pd.MultiIndex.from_tuples(col_tuples)

    row_order = race_values + ["Total"]

    ct = ct.reindex(index=row_order, columns=desired_columns, fill_value=0)
    ct.index.name = "Racial Categories"

    # Small-cell suppression: replace body cells with values 1–9 with "<10"
    # and deduct from the row total, column total, and grand total so visible
    # cells still sum to the displayed totals. Zeros are left as 0.
    ct = ct.astype(object)
    n_rows, n_cols = ct.shape
    for i in range(n_rows - 1):
        for j in range(n_cols - 1):
            v = ct.iat[i, j]
            if 1 <= v < 10:
                ct.iat[i, j] = "<10"
                ct.iat[i, -1] = ct.iat[i, -1] - v
                ct.iat[-1, j] = ct.iat[-1, j] - v
                ct.iat[-1, -1] = ct.iat[-1, -1] - v

    return ct


def generate_demographic_crosstab(patient_df: pd.DataFrame, output_dir: str, suffix: str = "") -> None:
    """Generate NIH enrollment report demographic crosstab and save to CSV."""
    os.makedirs(output_dir, exist_ok=True)
    enrollment_report = crosstab_demographics(patient_df)
    enrollment_report.to_csv(
        os.path.join(output_dir, _suffixed("demographic_crosstab_race_ethnicity_sex.csv", suffix))
    )
