"""Unit tests for modules.tableone.cohort_builder.

Synthetic-data tests covering each cohort flag in isolation plus the
top-level orchestrator. These do NOT require clifpy or the real CLIF
data — every fixture is a tiny pd.DataFrame mimicking a CLIF table's
schema.

Run with:
    .venv/bin/python -m pytest tests/test_cohort_builder.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from modules.tableone.cohort_builder import (
    DEFAULT_HFNC_LPM_THRESHOLD,
    DEFAULT_VASO_MEDS,
    CohortResult,
    build_critical_illness_cohort,
    compute_high_support_encounters,
    compute_icu_death_ward_flags,
    compute_is_procedural_ld_only,
    compute_nippv_hfnc_encounters,
    compute_vaso_support_encounters,
)


# ---------------------------------------------------------------------------
# Fixtures: small synthetic tables
# ---------------------------------------------------------------------------

def _adt(rows):
    """rows: list of (encounter_block, location_category, discharge_category)."""
    return pd.DataFrame(rows, columns=["encounter_block", "location_category", "discharge_category"])


def _resp(rows):
    """rows: list of (encounter_block, device_category, lpm_set)."""
    return pd.DataFrame(rows, columns=["encounter_block", "device_category", "lpm_set"])


def _meds(rows):
    """rows: list of (encounter_block, med_category)."""
    return pd.DataFrame(rows, columns=["encounter_block", "med_category"])


# ---------------------------------------------------------------------------
# compute_icu_death_ward_flags
# ---------------------------------------------------------------------------

class TestIcuDeathWardFlags:
    def test_basic_icu(self):
        adt = _adt([
            ("E1", "ICU",  "home"),
            ("E2", "ward", "home"),
        ])
        out = compute_icu_death_ward_flags(adt)
        out = out.set_index("encounter_block")
        assert out.loc["E1", "icu_enc"] == 1
        assert out.loc["E2", "icu_enc"] == 0
        assert out.loc["E2", "ward_enc"] == 1
        assert out.loc["E1", "ward_enc"] == 0

    def test_icu_substring_match(self):
        # location_category 'icu' substring should match — e.g. 'micu', 'sicu'
        adt = _adt([("E1", "MICU", "home")])
        out = compute_icu_death_ward_flags(adt)
        assert out.set_index("encounter_block").loc["E1", "icu_enc"] == 1

    def test_death_flag_set_for_expired_or_hospice(self):
        adt = _adt([
            ("E1", "ward", "expired"),
            ("E2", "ward", "hospice"),
            ("E3", "ward", "home"),
        ])
        out = compute_icu_death_ward_flags(adt).set_index("encounter_block")
        assert out.loc["E1", "death_enc"] == 1
        assert out.loc["E2", "death_enc"] == 1
        assert out.loc["E3", "death_enc"] == 0

    def test_lowercases_inputs(self):
        adt = _adt([
            ("E1", "WaRd", "EXPIRED"),
        ])
        out = compute_icu_death_ward_flags(adt).set_index("encounter_block")
        assert out.loc["E1", "ward_enc"] == 1
        assert out.loc["E1", "death_enc"] == 1

    def test_aggregates_across_multiple_rows(self):
        # An encounter with multiple ADT rows should get icu_enc=1 if ANY row is ICU
        adt = _adt([
            ("E1", "ed",  "home"),
            ("E1", "ICU", "home"),
            ("E1", "ward", "home"),
        ])
        out = compute_icu_death_ward_flags(adt).set_index("encounter_block")
        assert out.loc["E1", "icu_enc"] == 1

    def test_raises_on_missing_columns(self):
        with pytest.raises(KeyError):
            compute_icu_death_ward_flags(pd.DataFrame({"encounter_block": ["E1"]}))


# ---------------------------------------------------------------------------
# compute_is_procedural_ld_only
# ---------------------------------------------------------------------------

class TestProceduralLdOnly:
    def test_marks_procedural_only_encounter(self):
        adt = _adt([("E1", "procedural", "home")])
        flags = compute_icu_death_ward_flags(adt)
        out = compute_is_procedural_ld_only(adt, flags).set_index("encounter_block")
        assert out.loc["E1", "is_procedural_ld_only"] == 1

    def test_marks_ld_only_encounter(self):
        adt = _adt([("E1", "L&D", "home")])
        flags = compute_icu_death_ward_flags(adt)
        out = compute_is_procedural_ld_only(adt, flags).set_index("encounter_block")
        assert out.loc["E1", "is_procedural_ld_only"] == 1

    def test_does_not_mark_when_icu_was_touched(self):
        adt = _adt([
            ("E1", "procedural", "home"),
            ("E1", "ICU",        "home"),  # also touched ICU → not "only" procedural
        ])
        flags = compute_icu_death_ward_flags(adt)
        out = compute_is_procedural_ld_only(adt, flags).set_index("encounter_block")
        assert out.loc["E1", "is_procedural_ld_only"] == 0

    def test_does_not_mark_pure_ward(self):
        adt = _adt([("E1", "ward", "home")])
        flags = compute_icu_death_ward_flags(adt)
        out = compute_is_procedural_ld_only(adt, flags).set_index("encounter_block")
        assert out.loc["E1", "is_procedural_ld_only"] == 0


# ---------------------------------------------------------------------------
# compute_high_support_encounters
# ---------------------------------------------------------------------------

class TestHighSupport:
    def test_imv_always_qualifies(self):
        rs = _resp([("E1", "imv", None)])
        assert compute_high_support_encounters(rs) == {"E1"}

    def test_nippv_always_qualifies(self):
        rs = _resp([("E1", "nippv", None)])
        assert compute_high_support_encounters(rs) == {"E1"}

    def test_cpap_always_qualifies(self):
        rs = _resp([("E1", "cpap", None)])
        assert compute_high_support_encounters(rs) == {"E1"}

    def test_hfnc_qualifies_at_or_above_threshold(self):
        rs = _resp([
            ("E1", "high flow nc", 30),
            ("E2", "high flow nc", 50),
        ])
        assert compute_high_support_encounters(rs) == {"E1", "E2"}

    def test_hfnc_below_threshold_does_not_qualify(self):
        rs = _resp([
            ("E1", "high flow nc", 29),
            ("E2", "high flow nc", 10),
        ])
        assert compute_high_support_encounters(rs) == set()

    def test_custom_threshold(self):
        rs = _resp([
            ("E1", "high flow nc", 35),
            ("E2", "high flow nc", 50),
        ])
        assert compute_high_support_encounters(rs, hfnc_lpm_threshold=40) == {"E2"}

    def test_unknown_device_does_not_qualify(self):
        rs = _resp([
            ("E1", "room air",  None),
            ("E2", "nasal cannula", None),
        ])
        assert compute_high_support_encounters(rs) == set()

    def test_handles_uppercase_device_names(self):
        rs = _resp([("E1", "IMV", None)])
        assert compute_high_support_encounters(rs) == {"E1"}

    def test_works_without_lpm_set_column(self):
        rs = pd.DataFrame({
            "encounter_block":   ["E1", "E2"],
            "device_category":   ["imv", "high flow nc"],  # E2 has no lpm_set info
        })
        # E1 always qualifies; E2 doesn't because we have no lpm info to compare
        assert compute_high_support_encounters(rs) == {"E1"}

    def test_empty_input(self):
        rs = _resp([])
        assert compute_high_support_encounters(rs) == set()


# ---------------------------------------------------------------------------
# compute_nippv_hfnc_encounters
# ---------------------------------------------------------------------------

class TestNippvHfnc:
    def test_includes_nippv_and_qualifying_hfnc(self):
        rs = _resp([
            ("E1", "nippv",        None),
            ("E2", "high flow nc", 30),
            ("E3", "high flow nc", 20),  # below threshold
        ])
        assert compute_nippv_hfnc_encounters(rs) == {"E1", "E2"}

    def test_excludes_imv_and_cpap(self):
        # Important: this is a NIPPV/HFNC-specific subset, not the broader high_support
        rs = _resp([
            ("E1", "imv",   None),
            ("E2", "cpap",  None),
            ("E3", "nippv", None),
        ])
        assert compute_nippv_hfnc_encounters(rs) == {"E3"}


# ---------------------------------------------------------------------------
# compute_vaso_support_encounters
# ---------------------------------------------------------------------------

class TestVasoSupport:
    def test_default_meds(self):
        meds = _meds([
            ("E1", "norepinephrine"),
            ("E2", "epinephrine"),
            ("E3", "vasopressin"),
            ("E4", "phenylephrine"),
            ("E5", "dopamine"),
            ("E6", "angiotensin"),
            ("E7", "ibuprofen"),  # not vasoactive
        ])
        assert compute_vaso_support_encounters(meds) == {"E1", "E2", "E3", "E4", "E5", "E6"}

    def test_lowercases_input(self):
        meds = _meds([("E1", "Norepinephrine"), ("E2", "EPINEPHRINE")])
        assert compute_vaso_support_encounters(meds) == {"E1", "E2"}

    def test_custom_med_list(self):
        meds = _meds([("E1", "fentanyl")])
        assert compute_vaso_support_encounters(meds, vaso_meds=("fentanyl",)) == {"E1"}

    def test_empty_input(self):
        assert compute_vaso_support_encounters(_meds([])) == set()


# ---------------------------------------------------------------------------
# build_critical_illness_cohort (integration)
# ---------------------------------------------------------------------------

class TestBuildCriticalIllnessCohort:
    """End-to-end: test the orchestrator against a tiny synthetic site."""

    def _site(self):
        """Build a tiny synthetic 7-encounter site that exercises every flag.

        Encounters:
          E_ICU      — ICU stay only          → cohort
          E_ADV      — ward + IMV (no ICU)    → cohort (high_support)
          E_VASO     — ward + norepi (no ICU) → cohort (vaso)
          E_DEATH    — ward, expired (no ICU) → cohort (death)
          E_PROC_VASO — procedural-only + vaso → NOT cohort (procedural-only zeroes vaso)
          E_PROC_PURE — procedural-only, no support → NOT cohort
          E_WARD     — ward only, no support → NOT cohort
        """
        adt = _adt([
            ("E_ICU",       "ICU",        "home"),
            ("E_ADV",       "ward",       "home"),
            ("E_VASO",      "ward",       "home"),
            ("E_DEATH",     "ward",       "expired"),
            ("E_PROC_VASO", "procedural", "home"),
            ("E_PROC_PURE", "procedural", "home"),
            ("E_WARD",      "ward",       "home"),
        ])
        resp = _resp([
            ("E_ADV",       "imv",          None),
            ("E_PROC_PURE", "high flow nc", 5),  # below threshold; doesn't matter anyway
        ])
        meds = _meds([
            ("E_VASO",      "norepinephrine"),
            ("E_PROC_VASO", "norepinephrine"),  # given during procedural-only stay
        ])
        return adt, resp, meds

    def test_correct_cohort_membership(self):
        adt, resp, meds = self._site()
        result = build_critical_illness_cohort(adt, resp, meds)
        # ICU, advanced resp, vaso, death — all 4 in
        # procedural-only (with or without vaso), and pure ward — all out
        assert result.encounter_blocks == {"E_ICU", "E_ADV", "E_VASO", "E_DEATH"}

    def test_returns_correct_stats(self):
        adt, resp, meds = self._site()
        result = build_critical_illness_cohort(adt, resp, meds)
        assert result.stats["n_total_encounters"] == 7
        assert result.stats["n_icu"] == 1
        assert result.stats["n_death"] == 1
        assert result.stats["n_high_support"] == 1   # E_ADV (E_PROC_PURE's hfnc<30)
        assert result.stats["n_vaso"] == 1           # E_VASO (E_PROC_VASO zeroed because procedural-only)
        assert result.stats["n_procedural_ld_only"] == 2  # E_PROC_VASO + E_PROC_PURE
        assert result.stats["n_critical_illness"] == 4

    def test_procedural_zeroing_visible_in_flags_df(self):
        """E_PROC_VASO got vasopressors but is procedural-only → vaso_support_enc must be 0."""
        adt, resp, meds = self._site()
        result = build_critical_illness_cohort(adt, resp, meds)
        proc_vaso_row = result.flags_df.set_index("encounter_block").loc["E_PROC_VASO"]
        assert proc_vaso_row["is_procedural_ld_only"] == 1
        assert proc_vaso_row["vaso_support_enc"] == 0  # zeroed
        assert proc_vaso_row["cohort_enc"] == 0

    def test_hospitalization_id_resolution(self):
        adt, resp, meds = self._site()
        # Provide a 1:1 hospitalization_id ↔ encounter_block mapping
        mapping = pd.DataFrame({
            "hospitalization_id": ["H1", "H2", "H3", "H4", "H5", "H6", "H7"],
            "encounter_block":   ["E_ICU", "E_ADV", "E_VASO", "E_DEATH",
                                  "E_PROC_VASO", "E_PROC_PURE", "E_WARD"],
        })
        result = build_critical_illness_cohort(
            adt, resp, meds, encounter_mapping_df=mapping
        )
        assert result.hospitalization_ids == {"H1", "H2", "H3", "H4"}

    def test_hospitalization_ids_empty_without_mapping(self):
        adt, resp, meds = self._site()
        result = build_critical_illness_cohort(adt, resp, meds)
        assert result.hospitalization_ids == set()

    def test_many_to_one_hospitalization_mapping(self):
        """A stitched encounter_block may have multiple hospitalization_ids."""
        adt, resp, meds = self._site()
        mapping = pd.DataFrame({
            "hospitalization_id": ["H1a", "H1b", "H2", "H3", "H4", "H5", "H6", "H7"],
            "encounter_block":   ["E_ICU", "E_ICU",  "E_ADV", "E_VASO",
                                  "E_DEATH", "E_PROC_VASO", "E_PROC_PURE", "E_WARD"],
        })
        result = build_critical_illness_cohort(
            adt, resp, meds, encounter_mapping_df=mapping
        )
        # Both H1a and H1b should appear since they both map to E_ICU which is in cohort
        assert {"H1a", "H1b"}.issubset(result.hospitalization_ids)
        assert "H5" not in result.hospitalization_ids  # E_PROC_VASO not in cohort

    def test_hfnc_threshold_change_propagates(self):
        adt = _adt([
            ("E1", "ward", "home"),  # not in cohort by default
        ])
        resp = _resp([
            ("E1", "high flow nc", 25),  # below default threshold of 30
        ])
        meds = _meds([])

        # default threshold (30): E1 not advanced, not in cohort
        r1 = build_critical_illness_cohort(adt, resp, meds)
        assert r1.encounter_blocks == set()

        # custom threshold (20): E1 IS advanced, IS in cohort
        r2 = build_critical_illness_cohort(adt, resp, meds, hfnc_lpm_threshold=20)
        assert r2.encounter_blocks == {"E1"}

    def test_custom_vaso_meds(self):
        adt = _adt([("E1", "ward", "home")])
        resp = _resp([])
        meds = _meds([("E1", "fentanyl")])

        # default vaso meds: fentanyl is not in the list, so E1 NOT in cohort
        r1 = build_critical_illness_cohort(adt, resp, meds)
        assert r1.encounter_blocks == set()

        # custom vaso meds: E1 IS in cohort
        r2 = build_critical_illness_cohort(
            adt, resp, meds, vaso_meds=("fentanyl",)
        )
        assert r2.encounter_blocks == {"E1"}


# ---------------------------------------------------------------------------
# Integration with cohort_filter.compute_or_use_cached_filtered_tables
# ---------------------------------------------------------------------------

class TestCohortBuilderFeedsFilter:
    """The whole point: make sure the output of cohort_builder feeds cleanly
    into cohort_filter.compute_or_use_cached_filtered_tables() via a closure.
    """

    def test_can_be_passed_as_callable_to_filter_orchestrator(self, tmp_path):
        # Set up a tiny CLIF source dir
        import pyarrow as pa
        import pyarrow.parquet as pq
        source_dir = tmp_path / "src"
        source_dir.mkdir()
        ids = [f"H{i:03d}" for i in range(20)]
        # Synthetic labs file
        pq.write_table(
            pa.table({
                "hospitalization_id": pa.array(ids * 5),
                "lab_value":          pa.array(list(range(100))),
            }),
            str(source_dir / "clif_labs.parquet"),
        )

        # Synthetic ADT, resp, meds for 5 critically ill encounters
        # Map E0..E4 to H0..H4 (in cohort), E5..E19 not in cohort
        mapping = pd.DataFrame({
            "hospitalization_id": ids,
            "encounter_block":    [f"E{i:03d}" for i in range(20)],
        })
        adt = _adt([
            ("E000", "ICU", "home"),    # in cohort
            ("E001", "ICU", "home"),    # in cohort
            ("E002", "ward", "expired"),  # in cohort (death)
            ("E003", "ward", "home"),   # in cohort via vaso
            ("E004", "ward", "home"),   # in cohort via high_support
        ] + [(f"E{i:03d}", "ward", "home") for i in range(5, 20)])
        resp = _resp([("E004", "imv", None)])
        meds = _meds([("E003", "norepinephrine")])

        # Build cohort
        result = build_critical_illness_cohort(
            adt, resp, meds, encounter_mapping_df=mapping
        )
        assert len(result.hospitalization_ids) == 5

        # Now feed into the filter orchestrator
        from modules.tableone.cohort_filter import (
            compute_or_use_cached_filtered_tables, hash_cohort_definition,
            TableFilterSpec,
        )
        manifest, cached = compute_or_use_cached_filtered_tables(
            source_dir=source_dir,
            dest_dir=tmp_path / "dst",
            cohort_hash=hash_cohort_definition("test.v1"),
            spec=(TableFilterSpec("clif_labs.parquet", "labs_cohort.parquet"),),
            cohort_ids_callable=lambda: result.hospitalization_ids,
        )
        assert cached is False
        assert manifest.n_cohort_ids == 5

        # Filtered parquet has only the 5 cohort hospitalization_ids
        filtered = pq.read_table(str(tmp_path / "dst" / "labs_cohort.parquet")).to_pandas()
        assert set(filtered["hospitalization_id"]) == result.hospitalization_ids
        assert len(filtered) == 25  # 5 hosp × 5 lab rows each


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
