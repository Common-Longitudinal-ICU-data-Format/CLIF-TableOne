"""Unit tests for modules.tableone.nidfd_calculator.

Synthetic-data tests covering all 4 NIDFD rules:
  1. Death within window → 0
  2. No NI observations in window → 28
  3. Still on NI at day 28 → 0
  4. Otherwise: window_end_day − last_ni_day

Plus edge cases: HFNC threshold, multiple episodes, mixed devices.

Run with:
    .venv/bin/python -m pytest tests/test_nidfd_calculator.py -v
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from modules.tableone.nidfd_calculator import (
    DEFAULT_HFNC_LPM_THRESHOLD,
    calculate_non_invasive_device_free_days,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

UTC = timezone.utc


def _make_encounter(
    eid: str,
    ni_start: datetime,
    discharge: datetime,
    death: datetime | None = None,
    death_enc: int = 0,
    nippv_hfnc: int = 1,
):
    return {
        "encounter_block":      eid,
        "nippv_hfnc_enc":       nippv_hfnc,
        "ni_device_start_dttm": ni_start,
        "discharge_dttm":       discharge,
        "death_dttm":           death,
        "death_enc":            death_enc,
    }


def _make_resp_row(eid, recorded, device, lpm=np.nan):
    return {
        "encounter_block":  eid,
        "recorded_dttm":    recorded,
        "device_category":  device,
        "lpm_set":          lpm,
    }


def _utc(year, month, day, hour=0, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Rule-by-rule tests
# ---------------------------------------------------------------------------

class TestRule1_DeathInWindow:
    """Rule 1: death within 28-day window → NIDFD = 0."""

    def test_death_during_ni_use(self):
        encounter = pd.DataFrame([_make_encounter(
            "E1",
            ni_start=_utc(2024, 1, 1, 10),
            discharge=_utc(2024, 1, 5),
            death=_utc(2024, 1, 5),  # death day 5, within window
            death_enc=1,
        )])
        resp = pd.DataFrame([
            _make_resp_row("E1", _utc(2024, 1, 1, 10), "nippv"),
            _make_resp_row("E1", _utc(2024, 1, 3), "nippv"),
        ])
        out = calculate_non_invasive_device_free_days(resp, encounter)
        assert out.set_index("encounter_block").loc["E1", "nidfd_28"] == 0

    def test_death_dttm_missing_falls_back_to_discharge_dttm(self):
        """death_enc==1 but death_dttm is NaT → use discharge_dttm as death."""
        encounter = pd.DataFrame([_make_encounter(
            "E1",
            ni_start=_utc(2024, 1, 1),
            discharge=_utc(2024, 1, 5),  # discharged-as-expired day 5
            death=None,                   # missing dttm
            death_enc=1,                  # but flag says died
        )])
        resp = pd.DataFrame([_make_resp_row("E1", _utc(2024, 1, 1), "nippv")])
        out = calculate_non_invasive_device_free_days(resp, encounter)
        assert out.set_index("encounter_block").loc["E1", "nidfd_28"] == 0

    def test_death_outside_window_does_not_zero(self):
        """Death after day 28 → death rule doesn't apply.

        Patient transitions off NI on day 3, then dies on day 35 (past
        window). The death-rule should NOT zero NIDFD; should compute
        free days from the off-NI transition.
        """
        encounter = pd.DataFrame([_make_encounter(
            "E1",
            ni_start=_utc(2024, 1, 1),
            discharge=_utc(2024, 2, 5),
            death=_utc(2024, 2, 5),
            death_enc=1,
        )])
        resp = pd.DataFrame([
            _make_resp_row("E1", _utc(2024, 1, 1), "nippv"),
            _make_resp_row("E1", _utc(2024, 1, 3), "nasal cannula"),  # off NI
        ])
        out = calculate_non_invasive_device_free_days(resp, encounter)
        nidfd = out.set_index("encounter_block").loc["E1", "nidfd_28"]
        # Last NI ≈ Jan 3. NIDFD ≈ 28 - 3 = 25 (give or take normalize() rounding).
        assert 24 <= nidfd <= 26


class TestRule2_NoNiInWindow:
    """Rule 2: encounter flagged as nippv_hfnc_enc but no NI rows in window
    → NIDFD = 28 (rare edge case, treat as fully free)."""

    def test_qualifying_rows_outside_window(self):
        encounter = pd.DataFrame([_make_encounter(
            "E1",
            ni_start=_utc(2024, 1, 1),
            discharge=_utc(2024, 3, 1),  # well past 28 days
        )])
        # All resp observations are AFTER the 28-day window — qualifies the
        # encounter (nippv_hfnc_enc==1 was set elsewhere), but no episodes
        # fall within the 28-day window starting from ni_device_start_dttm.
        resp = pd.DataFrame([
            _make_resp_row("E1", _utc(2024, 2, 15), "nippv"),  # day ~45
            _make_resp_row("E1", _utc(2024, 2, 16), "nippv"),
        ])
        out = calculate_non_invasive_device_free_days(resp, encounter)
        assert out.set_index("encounter_block").loc["E1", "nidfd_28"] == 28


class TestRule3_StillOnAtDay28:
    """Rule 3: NI continues past day 28 → NIDFD = 0.

    Triggered when there's a transition-off observation (non-NI device)
    within the window AT OR PAST window_end. The transition-off itself
    is the signal — see test_continuous_ni_with_observed_transition.
    """

    def test_continuous_ni_with_observed_transition_at_day_28(self):
        """The patient is on NI through the window AND we observe a
        non-NI transition right at window_end. Algorithm correctly
        identifies this as 'still on at day 28' (NIDFD=0)."""
        encounter = pd.DataFrame([_make_encounter(
            "E1",
            ni_start=_utc(2024, 1, 1),
            discharge=_utc(2024, 2, 15),
        )])
        resp = pd.DataFrame([
            _make_resp_row("E1", _utc(2024, 1, 1), "nippv"),
            _make_resp_row("E1", _utc(2024, 1, 15), "nippv"),
            _make_resp_row("E1", _utc(2024, 1, 28), "nippv"),                 # at window_end
            _make_resp_row("E1", _utc(2024, 1, 28, 12), "nasal cannula"),     # transition just past
        ])
        out = calculate_non_invasive_device_free_days(resp, encounter)
        # Last NI ≈ Jan 28 12:00 (clamped to window_end). Rule 3 fires.
        assert out.set_index("encounter_block").loc["E1", "nidfd_28"] == 0

    def test_continuous_ni_no_observed_transition_known_limitation(self):
        """KNOWN LIMITATION (shared with vfd_calculator): if a patient is
        continuously on NI through window_end with NO transition-off
        observed in the window, the algorithm classifies them as
        'no NI in window' (NIDFD=28) instead of 'still on' (NIDFD=0).

        This test documents the current behavior. A proper fix requires
        interval-based episode tracking and should land alongside the
        same fix in vfd_calculator.py.
        """
        encounter = pd.DataFrame([_make_encounter(
            "E1",
            ni_start=_utc(2024, 1, 1),
            discharge=_utc(2024, 2, 15),
        )])
        resp = pd.DataFrame([
            _make_resp_row("E1", _utc(2024, 1, 1), "nippv"),
            _make_resp_row("E1", _utc(2024, 1, 15), "nippv"),
            _make_resp_row("E1", _utc(2024, 1, 28, 12), "nippv"),  # within window
            _make_resp_row("E1", _utc(2024, 2, 5), "nippv"),       # past day 28, no transition
        ])
        out = calculate_non_invasive_device_free_days(resp, encounter)
        # Bug: returns 28 even though patient was continuously on NI through day 28.
        # When fixed, this assertion should be updated to expect 0.
        assert out.set_index("encounter_block").loc["E1", "nidfd_28"] == 28


class TestRule4_NormalCase:
    """Rule 4: free days = window_end_day − last_ni_day."""

    def test_short_ni_episode(self):
        """NI from day 1 to day 3, then off → free days ≈ 25."""
        encounter = pd.DataFrame([_make_encounter(
            "E1",
            ni_start=_utc(2024, 1, 1),
            discharge=_utc(2024, 1, 20),
        )])
        resp = pd.DataFrame([
            _make_resp_row("E1", _utc(2024, 1, 1), "nippv"),
            _make_resp_row("E1", _utc(2024, 1, 3, 12), "nippv"),
            _make_resp_row("E1", _utc(2024, 1, 4), "nasal cannula"),  # off NI
            _make_resp_row("E1", _utc(2024, 1, 5), "nasal cannula"),
        ])
        out = calculate_non_invasive_device_free_days(resp, encounter)
        nidfd = out.set_index("encounter_block").loc["E1", "nidfd_28"]
        # Window: Jan 1 → Jan 28. Last NI ≈ Jan 4 (transition off).
        # NIDFD = 28 - 4 = 24 (give or take a day from datetime.normalize() rounding).
        assert 23 <= nidfd <= 25

    def test_reapplication_does_not_count_intermediate_free_days(self):
        """Patient on NI day 1, off day 2-5, back on day 6 → only days 6-28 of free
        time count after final episode end. Re-application zeros prior progress."""
        encounter = pd.DataFrame([_make_encounter(
            "E1",
            ni_start=_utc(2024, 1, 1),
            discharge=_utc(2024, 2, 15),
        )])
        resp = pd.DataFrame([
            _make_resp_row("E1", _utc(2024, 1, 1), "nippv"),         # episode 1 start
            _make_resp_row("E1", _utc(2024, 1, 2), "nasal cannula"),  # off
            _make_resp_row("E1", _utc(2024, 1, 3), "nasal cannula"),
            _make_resp_row("E1", _utc(2024, 1, 6), "nippv"),          # episode 2 start
            _make_resp_row("E1", _utc(2024, 1, 8), "nasal cannula"),  # off after re-app
        ])
        out = calculate_non_invasive_device_free_days(resp, encounter)
        nidfd = out.set_index("encounter_block").loc["E1", "nidfd_28"]
        # Last NI episode end ≈ Jan 8. NIDFD ≈ 28 - 8 = 20. Intermediate free
        # days (Jan 2-5) do NOT count. Anything ≥ 25 means we incorrectly
        # gave credit for intermediate free time.
        assert nidfd <= 22  # bounded above (would be ~20)
        assert nidfd >= 19  # bounded below


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_no_qualifying_encounters(self):
        """nippv_hfnc_enc == 0 for all encounters → empty result."""
        encounter = pd.DataFrame([_make_encounter(
            "E1",
            ni_start=_utc(2024, 1, 1),
            discharge=_utc(2024, 1, 5),
            nippv_hfnc=0,  # didn't qualify
        )])
        resp = pd.DataFrame([_make_resp_row("E1", _utc(2024, 1, 1), "nippv")])
        out = calculate_non_invasive_device_free_days(resp, encounter)
        assert len(out) == 0

    def test_missing_ni_device_start_dttm(self):
        """nippv_hfnc_enc == 1 but ni_device_start_dttm is NaT → skipped."""
        encounter = pd.DataFrame([_make_encounter(
            "E1",
            ni_start=None,
            discharge=_utc(2024, 1, 5),
        )])
        resp = pd.DataFrame([_make_resp_row("E1", _utc(2024, 1, 1), "nippv")])
        out = calculate_non_invasive_device_free_days(resp, encounter)
        assert len(out) == 0

    def test_hfnc_below_threshold_does_not_count(self):
        """HFNC at <30 LPM is not advanced → episode never starts."""
        encounter = pd.DataFrame([_make_encounter(
            "E1",
            ni_start=_utc(2024, 1, 1),
            discharge=_utc(2024, 1, 20),
        )])
        # HFNC at 20 LPM (below default threshold of 30)
        resp = pd.DataFrame([
            _make_resp_row("E1", _utc(2024, 1, 1), "high flow nc", lpm=20),
            _make_resp_row("E1", _utc(2024, 1, 5), "high flow nc", lpm=20),
        ])
        out = calculate_non_invasive_device_free_days(resp, encounter)
        # No NI episodes in window → NIDFD = 28 (Rule 2)
        assert out.set_index("encounter_block").loc["E1", "nidfd_28"] == 28

    def test_hfnc_at_threshold_qualifies(self):
        """HFNC ≥30 LPM is advanced — should count as NI."""
        encounter = pd.DataFrame([_make_encounter(
            "E1",
            ni_start=_utc(2024, 1, 1),
            discharge=_utc(2024, 1, 20),
        )])
        resp = pd.DataFrame([
            _make_resp_row("E1", _utc(2024, 1, 1), "high flow nc", lpm=30),
            _make_resp_row("E1", _utc(2024, 1, 3), "nasal cannula"),  # off
        ])
        out = calculate_non_invasive_device_free_days(resp, encounter)
        nidfd = out.set_index("encounter_block").loc["E1", "nidfd_28"]
        assert 24 <= nidfd <= 26

    def test_custom_hfnc_threshold(self):
        encounter = pd.DataFrame([_make_encounter(
            "E1",
            ni_start=_utc(2024, 1, 1),
            discharge=_utc(2024, 1, 20),
        )])
        resp = pd.DataFrame([
            _make_resp_row("E1", _utc(2024, 1, 1), "high flow nc", lpm=22),
            _make_resp_row("E1", _utc(2024, 1, 3), "nasal cannula"),
        ])
        # Default threshold 30 → no qualifying episode → NIDFD=28
        out_default = calculate_non_invasive_device_free_days(resp, encounter)
        assert out_default.set_index("encounter_block").loc["E1", "nidfd_28"] == 28

        # Custom threshold 20 → qualifies → real NIDFD computed
        out_low = calculate_non_invasive_device_free_days(
            resp, encounter, hfnc_lpm_threshold=20,
        )
        assert out_low.set_index("encounter_block").loc["E1", "nidfd_28"] < 28

    def test_returns_empty_when_required_columns_missing(self):
        encounter = pd.DataFrame([{
            "encounter_block": "E1",
            # missing nippv_hfnc_enc, ni_device_start_dttm, etc.
        }])
        resp = pd.DataFrame([_make_resp_row("E1", _utc(2024, 1, 1), "nippv")])
        out = calculate_non_invasive_device_free_days(resp, encounter)
        assert len(out) == 0

    def test_cpap_counts_as_ni(self):
        encounter = pd.DataFrame([_make_encounter(
            "E1",
            ni_start=_utc(2024, 1, 1),
            discharge=_utc(2024, 1, 20),
        )])
        resp = pd.DataFrame([
            _make_resp_row("E1", _utc(2024, 1, 1), "cpap"),
            _make_resp_row("E1", _utc(2024, 1, 3), "nasal cannula"),
        ])
        out = calculate_non_invasive_device_free_days(resp, encounter)
        nidfd = out.set_index("encounter_block").loc["E1", "nidfd_28"]
        assert 24 <= nidfd <= 26

    def test_imv_alone_does_not_qualify_for_ni(self):
        """IMV is not non-invasive — VFD covers it, NIDFD should not."""
        encounter = pd.DataFrame([_make_encounter(
            "E1",
            ni_start=_utc(2024, 1, 1),
            discharge=_utc(2024, 1, 20),
        )])
        resp = pd.DataFrame([
            _make_resp_row("E1", _utc(2024, 1, 1), "imv"),
            _make_resp_row("E1", _utc(2024, 1, 3), "imv"),
        ])
        out = calculate_non_invasive_device_free_days(resp, encounter)
        # No NI episodes in window → NIDFD = 28 (encounter qualifies because
        # we trust the caller's nippv_hfnc_enc flag, but no NI obs to anchor
        # the last_ni_dttm computation, so falls into Rule 2)
        assert out.set_index("encounter_block").loc["E1", "nidfd_28"] == 28


# ---------------------------------------------------------------------------
# Multi-encounter sanity
# ---------------------------------------------------------------------------

class TestMultiEncounterShape:
    def test_returns_one_row_per_qualifying_encounter(self):
        encounter = pd.DataFrame([
            _make_encounter("E1", _utc(2024, 1, 1), _utc(2024, 1, 20)),
            _make_encounter("E2", _utc(2024, 2, 1), _utc(2024, 2, 20)),
            _make_encounter("E3", _utc(2024, 3, 1), _utc(2024, 3, 20),
                            nippv_hfnc=0),  # not qualifying
        ])
        resp = pd.DataFrame([
            _make_resp_row("E1", _utc(2024, 1, 1), "nippv"),
            _make_resp_row("E2", _utc(2024, 2, 1), "nippv"),
            _make_resp_row("E3", _utc(2024, 3, 1), "nasal cannula"),
        ])
        out = calculate_non_invasive_device_free_days(resp, encounter)
        assert set(out["encounter_block"]) == {"E1", "E2"}
        assert "nidfd_28" in out.columns
        assert out["nidfd_28"].dtype == int


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
