"""Cohort definition for the critical-illness Table One.

Mirrors the cohort logic from `generator.py:main()` (lines ~1100–1440), but
extracted as pure functions so it can:
  - run BEFORE the heavy CLIF tables (labs/vitals/meds_intermittent/etc.)
    are loaded — feeding `cohort_filter.compute_or_use_cached_filtered_tables`,
  - be unit-tested in isolation against small synthetic DataFrames,
  - eventually replace the inline logic in `main()` with a single call.

The critical-illness cohort:

    cohort_enc = (icu_enc | death_enc | high_support_enc | vaso_support_enc)
                 & ~is_procedural_ld_only

where:

  - ``icu_enc``       = encounter ever had a row with location_category
                        containing 'icu'
  - ``death_enc``     = encounter ever had a row with discharge_category
                        ∈ {'expired', 'hospice'}
  - ``high_support_enc`` = encounter ever received imv/nippv/cpap
                           (unconditionally) or 'hfnc' with
                           lpm_set ≥ ``hfnc_lpm_threshold`` (default 30)
  - ``nippv_hfnc_enc``   = encounter ever received 'nippv' or
                           'hfnc' with lpm_set ≥ threshold
  - ``vaso_support_enc`` = encounter ever received any med in
                           ``vaso_meds`` (default: norepinephrine,
                           epinephrine, phenylephrine, vasopressin,
                           dopamine, angiotensin)
  - ``is_procedural_ld_only`` = encounter never touched ICU AND visited
                                 a 'procedural' or 'l_and_d' location at
                                 least once

Procedural/L&D-only encounters get their support flags zeroed (so a
patient who got vasopressors during a procedural-only stay isn't
counted as critically ill — see Phase 2 fix landed earlier).

These functions all key on ``encounter_block`` (post-stitching). The
caller is expected to have run encounter stitching first
(``clifpy.utils.stitching_encounters.stitch_encounters``) and to provide
DataFrames with ``encounter_block`` already populated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

import pandas as pd


__all__ = [
    "DEFAULT_VASO_MEDS",
    "DEFAULT_HFNC_LPM_THRESHOLD",
    "CohortResult",
    "compute_icu_death_ward_flags",
    "compute_is_procedural_ld_only",
    "compute_high_support_encounters",
    "compute_nippv_hfnc_encounters",
    "compute_vaso_support_encounters",
    "build_critical_illness_cohort",
]


DEFAULT_VASO_MEDS: tuple[str, ...] = (
    "norepinephrine",
    "epinephrine",
    "phenylephrine",
    "vasopressin",
    "dopamine",
    "angiotensin",
)

DEFAULT_HFNC_LPM_THRESHOLD: float = 30.0


@dataclass
class CohortResult:
    """The output of :func:`build_critical_illness_cohort`.

    Attributes:
        encounter_blocks: set of encounter_block IDs in the critically-ill cohort.
        hospitalization_ids: set of hospitalization_id values in the cohort.
            (Provided when an `encounter_mapping_df` with the hosp→block link
            was passed in; otherwise empty.)
        flags_df: per-encounter_block flag table with columns
            ``encounter_block, icu_enc, death_enc, ward_enc, high_support_enc,
            nippv_hfnc_enc, vaso_support_enc, is_procedural_ld_only, cohort_enc``.
            One row per encounter_block.
        stats: small dict of counts for logging / strobe construction.
    """

    encounter_blocks: set
    hospitalization_ids: set = field(default_factory=set)
    flags_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    stats: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-flag computations
# ---------------------------------------------------------------------------

def compute_icu_death_ward_flags(adt_df: pd.DataFrame) -> pd.DataFrame:
    """Compute icu_enc / death_enc / ward_enc per encounter_block.

    Args:
        adt_df: DataFrame with at least
            ``encounter_block, location_category, discharge_category``.
            ``location_category`` and ``discharge_category`` will be
            lower-cased before matching (the `main()` upstream does this
            in place; we do it on a copy here).

    Returns:
        DataFrame keyed by encounter_block with int 0/1 columns
        ``icu_enc, death_enc, ward_enc``.
    """
    if not {"encounter_block", "location_category", "discharge_category"}.issubset(adt_df.columns):
        raise KeyError(
            "adt_df must contain encounter_block, location_category, discharge_category"
        )

    df = adt_df[["encounter_block", "location_category", "discharge_category"]].copy()
    df["location_category"] = df["location_category"].astype(str).str.lower()
    df["discharge_category"] = df["discharge_category"].astype(str).str.lower()

    df["_icu"] = df["location_category"].str.contains("icu", na=False)
    df["_death"] = df["discharge_category"].isin({"expired", "hospice"})
    df["_ward"] = df["location_category"] == "ward"

    out = (
        df.groupby("encounter_block", as_index=False)
        .agg(
            icu_enc=("_icu", "any"),
            death_enc=("_death", "any"),
            ward_enc=("_ward", "any"),
        )
    )
    for c in ("icu_enc", "death_enc", "ward_enc"):
        out[c] = out[c].astype(int)
    return out


def compute_is_procedural_ld_only(
    adt_df: pd.DataFrame,
    icu_flags: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the ``is_procedural_ld_only`` flag per encounter_block.

    Defined as: never touched ICU AND visited a 'procedural' or 'l_and_d'
    location at least once.

    Args:
        adt_df: same df used in :func:`compute_icu_death_ward_flags`.
        icu_flags: DataFrame keyed by encounter_block with at least
            ``encounter_block, icu_enc`` (output of
            :func:`compute_icu_death_ward_flags`).

    Returns:
        DataFrame with ``encounter_block, is_procedural_ld_only`` (int 0/1).
    """
    df = adt_df[["encounter_block", "location_category"]].copy()
    df["location_category"] = df["location_category"].astype(str).str.lower()
    df["_is_proc_or_ld"] = df["location_category"].isin({"procedural", "l_and_d"})

    has_proc_or_ld = (
        df.groupby("encounter_block", as_index=False)
        .agg(has_procedural_or_ld=("_is_proc_or_ld", "any"))
    )

    merged = has_proc_or_ld.merge(icu_flags[["encounter_block", "icu_enc"]], on="encounter_block", how="left")
    merged["is_procedural_ld_only"] = (
        (merged["icu_enc"] == 0) & (merged["has_procedural_or_ld"])
    ).astype(int)
    return merged[["encounter_block", "is_procedural_ld_only"]]


def compute_high_support_encounters(
    resp_support_df: pd.DataFrame,
    *,
    hfnc_lpm_threshold: float = DEFAULT_HFNC_LPM_THRESHOLD,
) -> set:
    """Set of encounter_blocks with imv/nippv/cpap (always) or hfnc≥threshold.

    Args:
        resp_support_df: must contain ``encounter_block, device_category``.
            ``lpm_set`` only required if any rows have device_category=='hfnc'.
        hfnc_lpm_threshold: minimum LPM for HFNC to qualify as advanced
            respiratory support (default 30).
    """
    if "encounter_block" not in resp_support_df.columns or "device_category" not in resp_support_df.columns:
        raise KeyError("resp_support_df must contain encounter_block, device_category")

    dev = resp_support_df["device_category"].astype(str).str.lower()
    always_mask = dev.isin({"imv", "nippv", "cpap"})

    if "lpm_set" in resp_support_df.columns:
        hfnc_mask = (dev == "hfnc") & (
            pd.to_numeric(resp_support_df["lpm_set"], errors="coerce") >= hfnc_lpm_threshold
        )
    else:
        hfnc_mask = pd.Series(False, index=resp_support_df.index)

    return set(resp_support_df.loc[always_mask | hfnc_mask, "encounter_block"].unique())


def compute_nippv_hfnc_encounters(
    resp_support_df: pd.DataFrame,
    *,
    hfnc_lpm_threshold: float = DEFAULT_HFNC_LPM_THRESHOLD,
) -> set:
    """Set of encounter_blocks on NIPPV or HFNC≥threshold (no IMV/CPAP)."""
    if "encounter_block" not in resp_support_df.columns or "device_category" not in resp_support_df.columns:
        raise KeyError("resp_support_df must contain encounter_block, device_category")

    dev = resp_support_df["device_category"].astype(str).str.lower()
    nippv_mask = dev == "nippv"

    if "lpm_set" in resp_support_df.columns:
        hfnc_mask = (dev == "hfnc") & (
            pd.to_numeric(resp_support_df["lpm_set"], errors="coerce") >= hfnc_lpm_threshold
        )
    else:
        hfnc_mask = pd.Series(False, index=resp_support_df.index)

    return set(resp_support_df.loc[nippv_mask | hfnc_mask, "encounter_block"].unique())


def compute_vaso_support_encounters(
    meds_continuous_df: pd.DataFrame,
    *,
    vaso_meds: Iterable[str] = DEFAULT_VASO_MEDS,
) -> set:
    """Set of encounter_blocks that received any med in ``vaso_meds``.

    Args:
        meds_continuous_df: must contain ``encounter_block, med_category``.
        vaso_meds: iterable of med_category values to count as vasoactive.
    """
    if (
        "encounter_block" not in meds_continuous_df.columns
        or "med_category" not in meds_continuous_df.columns
    ):
        raise KeyError("meds_continuous_df must contain encounter_block, med_category")

    vaso_set = {str(m).lower() for m in vaso_meds}
    is_vaso = meds_continuous_df["med_category"].astype(str).str.lower().isin(vaso_set)
    return set(meds_continuous_df.loc[is_vaso, "encounter_block"].unique())


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def build_critical_illness_cohort(
    adt_df: pd.DataFrame,
    resp_support_df: pd.DataFrame,
    meds_continuous_df: pd.DataFrame,
    *,
    encounter_mapping_df: Optional[pd.DataFrame] = None,
    vaso_meds: Iterable[str] = DEFAULT_VASO_MEDS,
    hfnc_lpm_threshold: float = DEFAULT_HFNC_LPM_THRESHOLD,
) -> CohortResult:
    """Compute the critical-illness cohort end-to-end from the four cheap tables.

    Args:
        adt_df: encounter_block + location_category + discharge_category
            (post-stitching). One row per ADT event; the function
            aggregates per-encounter.
        resp_support_df: encounter_block + device_category (+ lpm_set
            when device_category includes 'hfnc').
        meds_continuous_df: encounter_block + med_category.
        encounter_mapping_df: optional 2-column DataFrame with
            ``hospitalization_id, encounter_block`` (the post-stitching
            mapping). When provided, the result includes a populated
            `hospitalization_ids` set for use by the filter step.
        vaso_meds: med_category values that count as vasoactive.
        hfnc_lpm_threshold: HFNC LPM minimum to qualify as advanced resp
            support (default 30).

    Returns:
        CohortResult with the encounter_block set, optional hospitalization_id
        set, the per-encounter flags DataFrame, and a small stats dict.
    """
    # 1. Per-encounter ICU/death/ward flags from ADT
    icu_death_ward = compute_icu_death_ward_flags(adt_df)

    # 2. Procedural/L&D-only flag (depends on icu_enc)
    proc_ld = compute_is_procedural_ld_only(adt_df, icu_death_ward)

    # 3. High-support and NIPPV/HFNC sets from respiratory_support
    high_support_set = compute_high_support_encounters(
        resp_support_df, hfnc_lpm_threshold=hfnc_lpm_threshold
    )
    nippv_hfnc_set = compute_nippv_hfnc_encounters(
        resp_support_df, hfnc_lpm_threshold=hfnc_lpm_threshold
    )

    # 4. Vasoactive set from medication_admin_continuous
    vaso_set = compute_vaso_support_encounters(meds_continuous_df, vaso_meds=vaso_meds)

    # 5. Assemble the per-encounter flag table
    flags = icu_death_ward.merge(proc_ld, on="encounter_block", how="left")
    flags["is_procedural_ld_only"] = flags["is_procedural_ld_only"].fillna(0).astype(int)
    flags["high_support_enc"] = flags["encounter_block"].isin(high_support_set).astype(int)
    flags["nippv_hfnc_enc"] = flags["encounter_block"].isin(nippv_hfnc_set).astype(int)
    flags["vaso_support_enc"] = flags["encounter_block"].isin(vaso_set).astype(int)

    # 6. Apply the procedural/L&D-only zeroing — same rule landed in the
    #    earlier Phase 2 fix to keep ward and CI cohorts comparable.
    proc_only_mask = flags["is_procedural_ld_only"] == 1
    flags.loc[proc_only_mask, "high_support_enc"] = 0
    flags.loc[proc_only_mask, "nippv_hfnc_enc"] = 0
    flags.loc[proc_only_mask, "vaso_support_enc"] = 0

    # 7. cohort_enc = ICU OR death OR adv-resp OR vaso, excl procedural/L&D-only
    flags["cohort_enc"] = (
        (flags["icu_enc"].astype(bool)
         | flags["death_enc"].astype(bool)
         | flags["high_support_enc"].astype(bool)
         | flags["vaso_support_enc"].astype(bool))
        & (~flags["is_procedural_ld_only"].astype(bool))
    ).astype(int)

    cohort_blocks = set(flags.loc[flags["cohort_enc"] == 1, "encounter_block"])

    hosp_ids: set = set()
    if encounter_mapping_df is not None:
        if (
            "hospitalization_id" not in encounter_mapping_df.columns
            or "encounter_block" not in encounter_mapping_df.columns
        ):
            raise KeyError(
                "encounter_mapping_df must contain hospitalization_id and encounter_block"
            )
        hosp_ids = set(
            encounter_mapping_df.loc[
                encounter_mapping_df["encounter_block"].isin(cohort_blocks),
                "hospitalization_id",
            ].unique()
        )

    stats = {
        "n_total_encounters":          int(flags["encounter_block"].nunique()),
        "n_icu":                       int(flags["icu_enc"].sum()),
        "n_death":                     int(flags["death_enc"].sum()),
        "n_high_support":              int(flags["high_support_enc"].sum()),
        "n_nippv_hfnc":                int(flags["nippv_hfnc_enc"].sum()),
        "n_vaso":                      int(flags["vaso_support_enc"].sum()),
        "n_procedural_ld_only":        int(flags["is_procedural_ld_only"].sum()),
        "n_critical_illness":          int(flags["cohort_enc"].sum()),
    }

    return CohortResult(
        encounter_blocks=cohort_blocks,
        hospitalization_ids=hosp_ids,
        flags_df=flags,
        stats=stats,
    )
