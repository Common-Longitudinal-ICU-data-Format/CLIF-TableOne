"""Small internal helpers used across the tableone submodules.

Extracted from `generator.py` as a pure refactor — same behavior, just
reduces the size of generator.py and lets the stat-generation modules
share these utilities without re-duplicating them.
"""

from __future__ import annotations

import os

import pandas as pd


__all__ = ["_suffixed", "_combine_sub_stratum_halves"]


def _suffixed(basename: str, suffix: str) -> str:
    """Insert *suffix* before the file extension: 'foo.csv' + '_icu' -> 'foo_icu.csv'."""
    stem, ext = os.path.splitext(basename)
    return f"{stem}{suffix}{ext}"


def _combine_sub_stratum_halves(left_df, right_df, left_col, right_col):
    """Side-by-side merge of two sub-stratum Table Ones.

    Preserves left_df's row order, then appends any Variables present
    only in right_df. Missing cells become empty strings.
    """
    left_vars = left_df["Variable"].tolist()
    right_vars = right_df["Variable"].tolist()
    ordered = left_vars + [v for v in right_vars if v not in set(left_vars)]
    merged = (
        pd.DataFrame({"Variable": ordered})
        .merge(left_df, on="Variable", how="left")
        .merge(right_df, on="Variable", how="left")
        .fillna("")
    )
    return merged[["Variable", left_col, right_col]]
