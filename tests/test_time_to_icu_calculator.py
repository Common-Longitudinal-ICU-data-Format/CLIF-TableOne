"""Unit tests for modules.tableone.time_to_icu_calculator."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from modules.tableone.time_to_icu_calculator import (
    calculate_time_to_icu_after_pressor,
    summarize_time_to_icu,
)


UTC = timezone.utc


def _utc(year, month, day, hour=0, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


# ---------------------------------------------------------------------------
# calculate_time_to_icu_after_pressor
# ---------------------------------------------------------------------------

class TestCalculateTimeToIcu:
    def test_basic_positive_delta(self):
        df = pd.DataFrame([{
            "encounter_block":              "E1",
            "vaso_ed_icu_enc":              1,
            "first_pressor_admin_dttm":     _utc(2024, 1, 1, 10, 0),
            "first_icu_in_dttm":            _utc(2024, 1, 1, 12, 30),
        }])
        out = calculate_time_to_icu_after_pressor(df)
        assert len(out) == 1
        assert out.iloc[0]["time_to_icu_hours"] == pytest.approx(2.5)

    def test_excludes_non_cohort_encounters(self):
        df = pd.DataFrame([
            {"encounter_block": "E1", "vaso_ed_icu_enc": 1,
             "first_pressor_admin_dttm": _utc(2024, 1, 1, 10),
             "first_icu_in_dttm":        _utc(2024, 1, 1, 14)},
            {"encounter_block": "E2", "vaso_ed_icu_enc": 0,  # not in cohort
             "first_pressor_admin_dttm": _utc(2024, 1, 1, 10),
             "first_icu_in_dttm":        _utc(2024, 1, 1, 12)},
        ])
        out = calculate_time_to_icu_after_pressor(df)
        assert set(out["encounter_block"]) == {"E1"}

    def test_drops_rows_missing_either_dttm(self):
        df = pd.DataFrame([
            {"encounter_block": "E1", "vaso_ed_icu_enc": 1,
             "first_pressor_admin_dttm": _utc(2024, 1, 1, 10),
             "first_icu_in_dttm":        _utc(2024, 1, 1, 14)},
            {"encounter_block": "E2", "vaso_ed_icu_enc": 1,
             "first_pressor_admin_dttm": pd.NaT,            # missing
             "first_icu_in_dttm":        _utc(2024, 1, 1, 14)},
            {"encounter_block": "E3", "vaso_ed_icu_enc": 1,
             "first_pressor_admin_dttm": _utc(2024, 1, 1, 10),
             "first_icu_in_dttm":        pd.NaT},           # missing
        ])
        out = calculate_time_to_icu_after_pressor(df)
        assert set(out["encounter_block"]) == {"E1"}

    def test_negative_delta_kept(self):
        """Negative delta = data integrity signal; pass through."""
        df = pd.DataFrame([{
            "encounter_block":              "E1",
            "vaso_ed_icu_enc":              1,
            "first_pressor_admin_dttm":     _utc(2024, 1, 1, 14),  # AFTER ICU
            "first_icu_in_dttm":            _utc(2024, 1, 1, 10),  # BEFORE pressor
        }])
        out = calculate_time_to_icu_after_pressor(df)
        assert len(out) == 1
        assert out.iloc[0]["time_to_icu_hours"] == pytest.approx(-4.0)

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=[
            "encounter_block", "vaso_ed_icu_enc",
            "first_pressor_admin_dttm", "first_icu_in_dttm",
        ])
        out = calculate_time_to_icu_after_pressor(df)
        assert len(out) == 0

    def test_no_cohort_encounters(self):
        df = pd.DataFrame([
            {"encounter_block": "E1", "vaso_ed_icu_enc": 0,
             "first_pressor_admin_dttm": _utc(2024, 1, 1, 10),
             "first_icu_in_dttm":        _utc(2024, 1, 1, 14)},
        ])
        out = calculate_time_to_icu_after_pressor(df)
        assert len(out) == 0

    def test_missing_required_columns_returns_empty(self):
        df = pd.DataFrame([{"encounter_block": "E1"}])  # no other cols
        out = calculate_time_to_icu_after_pressor(df)
        assert len(out) == 0

    def test_custom_column_names(self):
        df = pd.DataFrame([{
            "enc":            "E1",
            "ed_icu_flag":    1,
            "press_dttm":     _utc(2024, 1, 1, 10),
            "icu_dttm":       _utc(2024, 1, 1, 13),
        }])
        out = calculate_time_to_icu_after_pressor(
            df,
            id_col="enc",
            pressor_dttm_col="press_dttm",
            icu_dttm_col="icu_dttm",
            flag_col="ed_icu_flag",
        )
        assert len(out) == 1
        assert out.iloc[0]["time_to_icu_hours"] == pytest.approx(3.0)

    def test_naive_datetimes_are_coerced(self):
        """If columns aren't datetime-typed, function coerces them via UTC."""
        df = pd.DataFrame([{
            "encounter_block":              "E1",
            "vaso_ed_icu_enc":              1,
            "first_pressor_admin_dttm":     "2024-01-01T10:00:00",
            "first_icu_in_dttm":            "2024-01-01T13:00:00",
        }])
        out = calculate_time_to_icu_after_pressor(df)
        assert len(out) == 1
        assert out.iloc[0]["time_to_icu_hours"] == pytest.approx(3.0)

    def test_multiple_encounters(self):
        df = pd.DataFrame([
            {"encounter_block": "E1", "vaso_ed_icu_enc": 1,
             "first_pressor_admin_dttm": _utc(2024, 1, 1, 10),
             "first_icu_in_dttm":        _utc(2024, 1, 1, 11)},
            {"encounter_block": "E2", "vaso_ed_icu_enc": 1,
             "first_pressor_admin_dttm": _utc(2024, 1, 1, 10),
             "first_icu_in_dttm":        _utc(2024, 1, 1, 14)},
            {"encounter_block": "E3", "vaso_ed_icu_enc": 1,
             "first_pressor_admin_dttm": _utc(2024, 1, 1, 10),
             "first_icu_in_dttm":        _utc(2024, 1, 2, 10)},
        ])
        out = calculate_time_to_icu_after_pressor(df).set_index("encounter_block")
        assert out.loc["E1", "time_to_icu_hours"] == pytest.approx(1.0)
        assert out.loc["E2", "time_to_icu_hours"] == pytest.approx(4.0)
        assert out.loc["E3", "time_to_icu_hours"] == pytest.approx(24.0)


# ---------------------------------------------------------------------------
# summarize_time_to_icu
# ---------------------------------------------------------------------------

class TestSummarizeTimeToIcu:
    def test_basic_summary(self):
        df = pd.DataFrame({"time_to_icu_hours": [1.0, 2.0, 3.0, 4.0, 5.0]})
        s = summarize_time_to_icu(df)
        assert s["n"] == 5
        assert s["median"] == pytest.approx(3.0)
        assert s["q1"] == pytest.approx(2.0)
        assert s["q3"] == pytest.approx(4.0)
        assert s["mean"] == pytest.approx(3.0)
        assert s["n_negative"] == 0

    def test_empty(self):
        df = pd.DataFrame(columns=["time_to_icu_hours"])
        s = summarize_time_to_icu(df)
        assert s["n"] == 0
        assert s["n_negative"] == 0
        # All other stats should be NaN
        import math
        assert math.isnan(s["median"])

    def test_single_value(self):
        df = pd.DataFrame({"time_to_icu_hours": [3.5]})
        s = summarize_time_to_icu(df)
        assert s["n"] == 1
        assert s["median"] == pytest.approx(3.5)
        # std is NaN with only one value (ddof=1)
        import math
        assert math.isnan(s["std"])

    def test_counts_negatives(self):
        df = pd.DataFrame({"time_to_icu_hours": [-2.0, -1.0, 0.5, 1.0, 2.0]})
        s = summarize_time_to_icu(df)
        assert s["n"] == 5
        assert s["n_negative"] == 2

    def test_median_with_negatives(self):
        """Negatives are kept in the summary so analysts see the data-quality signal."""
        df = pd.DataFrame({"time_to_icu_hours": [-10.0, 1.0, 2.0, 3.0, 4.0]})
        s = summarize_time_to_icu(df)
        assert s["median"] == pytest.approx(2.0)  # median includes the negative


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
