"""Table One result routes - serve images, CSVs, and tab data.

Routes are cohort-aware. The canonical paths take a {cohort} segment:
  /api/tableone/{cohort}/available
  /api/tableone/{cohort}/data/{tab}
  /api/tableone/{cohort}/images/{filename}
  /api/tableone/{cohort}/strobe

where cohort ∈ {ci, ward}. The legacy paths (/api/tableone/available,
/api/tableone/data/{tab}, /api/tableone/images/{filename}) are kept as
aliases that resolve to cohort='ci' so the existing UI keeps working
during the dashboard rewrite.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from server import session
import pandas as pd
from pathlib import Path

from modules.utils.output_paths import (
    tableone_dir,
    figures_dir,
    ward_tableone_dir,
    ward_figures_dir,
)

# Top-level strata (the parent directories under output/final/strata/).
# Sub-strata like 'advanced_resp/icu' live as suffixed CSVs inside the
# parent dir per parse_stratum() in output_paths.py — we surface them
# implicitly through the *_icu_vs_no_icu.csv comparison tables.
_STRATA = ("icu", "advanced_resp", "nippv_hfnc", "vaso", "no_imv", "deaths")

router = APIRouter(prefix="/api/tableone", tags=["tableone"])


# ── cohort path resolution ───────────────────────────────────────────

_VALID_COHORTS = ("ci", "ward")


def _resolve_dirs(cohort: str) -> tuple[Path, Path]:
    """Return (csv_dir, figures_dir) for the requested cohort."""
    if cohort == "ci":
        return tableone_dir(), figures_dir()
    if cohort == "ward":
        return ward_tableone_dir(), ward_figures_dir()
    raise HTTPException(404, f"Unknown cohort: {cohort}. Expected one of {_VALID_COHORTS}.")


# ── canonical (cohort-aware) routes ──────────────────────────────────

@router.get("/{cohort}/available")
async def check_available(cohort: str):
    csv_d, fig_d = _resolve_dirs(cohort)
    available = (
        csv_d.exists()
        and fig_d.exists()
        and (csv_d / "table_one_overall.csv").exists()
    )
    return {"available": available, "cohort": cohort}


@router.get("/{cohort}/images/{filename:path}")
async def get_image(cohort: str, filename: str):
    csv_d, fig_d = _resolve_dirs(cohort)
    filepath = fig_d / filename
    if not filepath.exists():
        filepath = csv_d / filename
    if not filepath.exists():
        raise HTTPException(404, f"Image not found: {filename}")
    if filename.endswith(".html"):
        media = "text/html"
    elif filename.endswith(".png"):
        media = "image/png"
    elif filename.endswith(".jpg") or filename.endswith(".jpeg"):
        media = "image/jpeg"
    else:
        media = "application/octet-stream"
    return FileResponse(str(filepath), media_type=media)


@router.get("/{cohort}/strobe")
async def get_strobe(cohort: str):
    """Return the small JSON payload that drives the Overview tab and the
    SVG cohort flow: strobe counts, mortality rates, and a curated set of
    headline rows from table_one_overall.csv. Total response < 4 KB.
    """
    csv_d, _ = _resolve_dirs(cohort)

    out: dict = {"cohort": cohort, "strobe": {}, "mortality": {}, "kpi": {}}

    strobe_path = csv_d / "strobe_counts.csv"
    if strobe_path.exists():
        df = pd.read_csv(strobe_path)
        for _, row in df.iterrows():
            try:
                out["strobe"][row["count_name"]] = int(row["count_value"])
            except (ValueError, TypeError):
                out["strobe"][row["count_name"]] = row["count_value"]

    mortality_path = csv_d / "mortality_rates.csv"
    if mortality_path.exists():
        df = pd.read_csv(mortality_path)
        for _, row in df.iterrows():
            try:
                out["mortality"][row["count_name"]] = float(row["count_value"])
            except (ValueError, TypeError):
                pass

    # Headline rows pulled verbatim from table_one_overall.csv. Keep this
    # list small and stable — the UI maps these to KPI cards by exact key.
    overall_path = csv_d / "table_one_overall.csv"
    if overall_path.exists():
        df = pd.read_csv(overall_path)
        if "Variable" in df.columns and "Overall" in df.columns:
            wanted = {
                "N: Encounter blocks": "n_encounters",
                "N: Unique patients": "n_patients",
                "N: Hospitals": "n_hospitals",
                "Age at admission, median [Q1, Q3]": "age_median_iqr",
                "Hospital mortality, n (%)": "hospital_mortality",
                "  Sex: Male": "sex_male",
                "  Sex: Female": "sex_female",
                "ICU episodes, total n": "icu_episodes_total",
                "Sepsis events (CDC ASE), n": "sepsis_events_total",
                "Encounters with >=1 sepsis event, n (%)": "sepsis_encounters",
            }
            lookup = dict(zip(df["Variable"].str.strip(), df["Overall"]))
            for key_in_csv, alias in wanted.items():
                v = lookup.get(key_in_csv.strip())
                if v is not None and pd.notna(v):
                    out["kpi"][alias] = v

    return out


# ── strata (CI cohort only — ward doesn't generate strata outputs) ───

@router.get("/{cohort}/strata")
async def list_strata(cohort: str):
    """List the strata that have outputs on disk for the requested cohort.

    Returns: { cohort, strata: [ {key, label, has_by_year, has_icu_vs_no_icu,
                                  has_ed_icu_vs_ed_ward}, ... ] }
    Empty `strata` list if the cohort doesn't have a strata tree (e.g. ward).
    """
    if cohort != "ci":
        # Ward currently doesn't generate strata outputs. Return empty so the
        # UI can still call this safely and render "no strata for this cohort".
        return {"cohort": cohort, "strata": []}

    csv_d, _ = _resolve_dirs(cohort)
    base = csv_d.parent.parent / "strata"  # output/final/strata/
    if not base.exists():
        return {"cohort": cohort, "strata": []}

    out_strata = []
    labels = {
        "icu": "ICU",
        "advanced_resp": "Advanced respiratory",
        "nippv_hfnc": "NIPPV / HFNC",
        "vaso": "Vasoactive",
        "no_imv": "No IMV",
        "deaths": "Deaths",
    }
    for s in _STRATA:
        s_dir = base / s / "tableone"
        if not s_dir.exists():
            continue
        out_strata.append({
            "key": s,
            "label": labels.get(s, s),
            "has_by_year":           (s_dir / f"table_one_{s}_by_year.csv").exists(),
            "has_icu_vs_no_icu":     (s_dir / f"table_one_{s}_icu_vs_no_icu.csv").exists(),
            "has_ed_icu_vs_ed_ward": (s_dir / f"table_one_{s}_ed_icu_vs_ed_ward.csv").exists(),
        })
    return {"cohort": cohort, "strata": out_strata}


@router.get("/{cohort}/strata/{stratum}")
async def get_stratum_data(cohort: str, stratum: str):
    """Return Table One CSVs for the picked stratum as JSON-safe tables.

    Always lazy: only reads the requested stratum's files. Per-tab payload
    capped at the file sizes on disk (typically <50 KB total).
    """
    if cohort != "ci":
        raise HTTPException(404, f"Strata not available for cohort={cohort}")
    if stratum not in _STRATA:
        raise HTTPException(404, f"Unknown stratum: {stratum}")

    csv_d, _ = _resolve_dirs(cohort)
    s_dir = csv_d.parent.parent / "strata" / stratum / "tableone"
    if not s_dir.exists():
        raise HTTPException(404, f"Stratum directory missing: {stratum}")

    out: dict = {"cohort": cohort, "stratum": stratum, "tables": {}}

    # Each entry: (file_basename, table_id_in_response, human_label)
    candidates = [
        (f"table_one_{stratum}_by_year.csv",            "by_year",            "By year"),
        (f"table_one_{stratum}_icu_vs_no_icu.csv",      "icu_vs_no_icu",      "ICU vs Non-ICU"),
        (f"table_one_{stratum}_ed_icu_vs_ed_ward.csv",  "ed_icu_vs_ed_ward",  "ED→ICU vs ED→Ward"),
    ]
    for fname, tid, label in candidates:
        p = s_dir / fname
        if p.exists():
            df = pd.read_csv(p)
            tbl = _df_to_table(df)
            tbl["label"] = label
            tbl["filename"] = fname
            out["tables"][tid] = tbl

    return out


@router.get("/{cohort}/data/{tab}")
async def get_tab_data(cohort: str, tab: str):
    """Return data for a specific tab in the requested cohort."""
    csv_d, fig_d = _resolve_dirs(cohort)
    if not csv_d.exists():
        raise HTTPException(404, f"Table One results not found for cohort={cohort}")

    handlers = {
        "cohort":        _cohort_data,
        "demographics":  _demographics_data,
        "medications":   _medications_data,
        "imv":           _imv_data,
        "sofa_cci":      _sofa_cci_data,
        "comorbidities": _comorbidities_data,
        "outcomes":      _outcomes_data,
    }

    handler = handlers.get(tab)
    if handler is None:
        raise HTTPException(404, f"Unknown tab: {tab}")

    return handler(csv_d, fig_d)


# ── legacy aliases (no cohort segment → defaults to 'ci') ───────────
# These keep the existing UI working without changes. New UI code should
# use the canonical /{cohort}/ paths above.

@router.get("/available")
async def check_available_legacy():
    return await check_available("ci")


@router.get("/images/{filename:path}")
async def get_image_legacy(filename: str):
    return await get_image("ci", filename)


@router.get("/data/{tab}")
async def get_tab_data_legacy(tab: str):
    return await get_tab_data("ci", tab)


# ── helpers ──────────────────────────────────────────────────────────

def _file_exists_or_none(path: Path):
    return str(path.name) if path.exists() else None


def _df_to_table(df: pd.DataFrame) -> dict:
    """Convert DataFrame to JSON-safe table dict, replacing NaN with None."""
    return {
        "columns": df.columns.tolist(),
        "data": df.where(df.notnull(), None).to_dict(orient="records"),
    }


def _cohort_data(csv_d: Path, fig_d: Path):
    sankeys = []
    if fig_d.exists():
        for fname, label in [
            ("sankey_matplotlib_icu.png", "ICU Patients"),
            ("sankey_matplotlib_others.png", "Other Patients"),
            ("sankey_matplotlib_high_o2_support.png", "High O2 Support"),
            ("sankey_matplotlib_vaso_support.png", "Vasopressor Support"),
        ]:
            if (fig_d / fname).exists():
                sankeys.append({"filename": fname, "label": label})

    return {
        "consort": _file_exists_or_none(fig_d / "consort_flow_diagram.png"),
        "upset": _file_exists_or_none(fig_d / "cohort_intersect_upset_plot.png"),
        "venn": _file_exists_or_none(fig_d / "venn_all_4_groups.png"),
        "code_status": _file_exists_or_none(
            fig_d / "code_status_stacked_bar_with_missingness_excl_missing_cat.png"
        ),
        "sankeys": sankeys,
    }


def _demographics_slice(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the demographics portion of a full table-one DataFrame.

    Cuts before 'SOFA Scores' row, which is where clinical data begins.
    """
    if "Variable" not in df.columns:
        return df
    mask = df["Variable"].str.strip() == "SOFA Scores"
    idx = df.index[mask]
    if len(idx) > 0:
        return df.iloc[:idx[0]].reset_index(drop=True)
    return df


def _demographics_data(csv_d: Path, fig_d: Path):
    """Return the demographics-portion Table One.

    Prefer table_one_by_year.csv when it exists — that file already has an
    'Overall' column alongside per-year columns, so showing the standalone
    table_one_overall.csv on top of it is redundant. Fall back to overall
    only when by_year isn't generated (e.g. ward mode skips the by-year
    breakdown).
    """
    result: dict = {"tables": {}, "images": []}

    by_year = csv_d / "table_one_by_year.csv"
    overall = csv_d / "table_one_overall.csv"

    if by_year.exists():
        df = _demographics_slice(pd.read_csv(by_year))
        result["tables"]["by_year"] = _df_to_table(df)
    elif overall.exists():
        df = _demographics_slice(pd.read_csv(overall))
        result["tables"]["overall"] = _df_to_table(df)

    return result


def _medications_data(csv_d: Path, fig_d: Path):
    result: dict = {"html_plots": [], "csv_files": []}

    for label, fname in [
        ("Vasoactive Area Curve (7d)", "vasoactive_area_curve_7d.html"),
        ("Vasoactive Median Dose by Hour", "vasoactive_median_dose_by_hour.html"),
        ("Sedative Area Curve (7d)", "sedative_area_curve_7d.html"),
        ("Sedative Median Dose by Hour", "sedative_median_dose_by_hour.html"),
        ("Paralytic Area Curve (7d)", "paralytic_area_curve_7d.html"),
        ("Paralytic Median Dose by Hour", "paralytic_median_dose_by_hour.html"),
    ]:
        if (fig_d / fname).exists():
            result["html_plots"].append({"label": label, "filename": fname})

    meds_csv = csv_d / "medications_summary_stats.csv"
    if meds_csv.exists():
        df = pd.read_csv(meds_csv)
        tbl = _df_to_table(df)
        tbl["label"] = "Medication Summary Statistics"
        result["csv_files"].append(tbl)

    return result


def _imv_data(csv_d: Path, fig_d: Path):
    """Backs the 'Ventilation & PF/SF' tab (kept under tab id 'imv' for
    URL stability). Surfaces the IMV-related figures + ventilator settings
    tables that already existed, plus three new pieces from the 2026 work:
      - km_time_to_extubation.png + min_pf_sf_per_day_post_intubation.png
        (CI cohort only — generated under overall/figures/)
      - pf_sf_aggregate_stats from the advanced_resp + no_imv strata,
        each as overall / icu / no_icu sets (aggregate-only, no PHI).

    Note: pf_sf_summary_24h.csv is INTENTIONALLY NOT surfaced — it's
    per-encounter data with encounter_block IDs (patient-level), similar
    to upset_data which we already moved to /intermediate. Treat that
    file as not-shareable and route consortium consumers to the
    aggregate stats instead.
    """
    result: dict = {"images": [], "csv_files": []}

    for label, fname in [
        ("Tidal Volume - Volume Control Modes",       "tidal_volume_volume_control_modes.png"),
        ("Pressure Control Mode",                     "pressure_control_pressure_control_mode.png"),
        ("Mode Proportions (First 24h)",              "mode_proportions_first_24h_vertical.png"),
        ("Ventilator Settings Table",                 "ventilator_settings_table.png"),
        ("Time to Extubation (Kaplan–Meier)",         "km_time_to_extubation.png"),
        ("Min PF/SF per Day Post-Intubation",         "min_pf_sf_per_day_post_intubation.png"),
    ]:
        if (fig_d / fname).exists():
            result["images"].append({"label": label, "filename": fname})

    for label, fname in [
        ("Ventilator Settings by Device Mode", "ventilator_settings_by_device_mode.csv"),
        ("Ventilator Settings Counts",         "ventilator_settings_counts_by_device_mode.csv"),
    ]:
        if (csv_d / fname).exists():
            df = pd.read_csv(csv_d / fname)
            tbl = _df_to_table(df)
            tbl["label"] = label
            tbl["filename"] = fname
            result["csv_files"].append(tbl)

    # PF/SF aggregate stats from the strata. Only meaningful for the CI
    # cohort (where stratified outputs exist). Each stratum contributes
    # three tables: overall / icu / no_icu — keep them paired in label
    # order so reviewers see them grouped.
    strata_root = csv_d.parent.parent / "strata"  # output/final/strata/
    if strata_root.exists():
        for parent_label, parent_key in (
            ("Advanced respiratory", "advanced_resp"),
            ("No IMV",                "no_imv"),
        ):
            s_dir = strata_root / parent_key / "tableone"
            if not s_dir.exists():
                continue
            for sub_label, sub_suffix in (("overall", ""), ("ICU", "_icu"), ("Non-ICU", "_no_icu")):
                f = s_dir / f"pf_sf_aggregate_stats{sub_suffix}.csv"
                if f.exists():
                    df = pd.read_csv(f)
                    tbl = _df_to_table(df)
                    tbl["label"]    = f"PF/SF aggregate — {parent_label} ({sub_label})"
                    tbl["filename"] = f.name
                    result["csv_files"].append(tbl)

    return result


def _sofa_cci_data(csv_d: Path, fig_d: Path):
    """SOFA + CCI mortality only. Comorbidity prevalence + per-1000 table
    moved out into the dedicated Comorbidities tab via _comorbidities_data().
    """
    result: dict = {"images": [], "csv_files": []}

    for label, fname in [
        ("SOFA Score & Mortality",  "sofa_mortality_histogram.png"),
        ("CCI Mortality & Hospice", "cci_mortality_hospice_comprehensive.png"),
    ]:
        if (fig_d / fname).exists():
            result["images"].append({"label": label, "filename": fname})

    sofa_summary = csv_d / "sofa_mortality_summary.csv"
    if sofa_summary.exists():
        df = pd.read_csv(sofa_summary)
        tbl = _df_to_table(df)
        tbl["label"]    = "SOFA mortality summary"
        tbl["filename"] = "sofa_mortality_summary.csv"
        result["csv_files"].append(tbl)

    cci_summary = csv_d / "cci_hospice_mortality_comprehensive_summary.csv"
    if cci_summary.exists():
        df = pd.read_csv(cci_summary)
        tbl = _df_to_table(df)
        tbl["label"]    = "CCI hospice/mortality comprehensive"
        tbl["filename"] = "cci_hospice_mortality_comprehensive_summary.csv"
        result["csv_files"].append(tbl)

    return result


def _comorbidities_data(csv_d: Path, fig_d: Path):
    """Dedicated comorbidities tab: prevalence bar + per-1000 + summary."""
    result: dict = {"images": [], "csv_files": []}

    fname = "comorbidities_per_1000_barplot.png"
    if (fig_d / fname).exists():
        result["images"].append({"label": "Comorbidity Prevalence (per 1000)", "filename": fname})

    for label, fname in [
        ("Comorbidities per 1000 hospitalizations",         "comorbidities_per_1000_hospitalizations.csv"),
        ("Comorbidities per 1000 — summary",                "comorbidities_per_1000_hospitalizations_summary.csv"),
    ]:
        p = csv_d / fname
        if p.exists():
            df = pd.read_csv(p)
            tbl = _df_to_table(df)
            tbl["label"]    = label
            tbl["filename"] = fname
            result["csv_files"].append(tbl)

    return result


def _outcomes_data(csv_d: Path, fig_d: Path):
    """Outcomes tab: mortality + hospice trends + code-status outcomes.

    Brings in the previously-orphaned summaries:
      - mortality_rates.csv (also shown on Overview, repeated here as a
        sortable table for reviewers digging deeper)
      - hospice_trends_summary.csv
      - cci_mortality_hospice_trends_by_year_category_plotdata.csv
      - code_status_combined_summary.csv
      - code_status_counts_by_encounter_type.csv
    """
    result: dict = {"images": [], "csv_files": []}

    for label, fname in [
        ("Hospice & Mortality Trends", "hospice_mortality_combined_trends.png"),
        ("Code Status Distribution",   "code_status_stacked_bar_with_missingness_excl_missing_cat.png"),
    ]:
        if (fig_d / fname).exists():
            result["images"].append({"label": label, "filename": fname})

    for label, fname in [
        ("Mortality rates by encounter type", "mortality_rates.csv"),
        ("Hospice trends — summary",          "hospice_trends_summary.csv"),
        ("CCI mortality/hospice — trend by year × category",
                                              "cci_mortality_hospice_trends_by_year_category_plotdata.csv"),
        ("Code status — combined summary",    "code_status_combined_summary.csv"),
        ("Code status — counts by encounter type",
                                              "code_status_counts_by_encounter_type.csv"),
        ("Code status — % by encounter type", "code_status_percentages_by_encounter_type.csv"),
        ("Code status — missingness summary", "code_status_missingness_summary.csv"),
    ]:
        p = csv_d / fname
        if p.exists():
            df = pd.read_csv(p)
            tbl = _df_to_table(df)
            tbl["label"]    = label
            tbl["filename"] = fname
            result["csv_files"].append(tbl)

    return result
