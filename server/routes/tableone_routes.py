"""Table One result routes - serve images, CSVs, and tab data."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from server import session
import pandas as pd
from pathlib import Path

router = APIRouter(prefix="/api/tableone", tags=["tableone"])


def _tableone_dir() -> Path:
    config = session.get("config") or {}
    output_dir = config.get("output_dir", "output")
    return Path(output_dir) / "final" / "tableone"


@router.get("/available")
async def check_available():
    d = _tableone_dir()
    available = (
        d.exists()
        and (d / "consort_flow_diagram.png").exists()
        and (d / "table_one_overall.csv").exists()
    )
    return {"available": available}


@router.get("/images/{filename:path}")
async def get_image(filename: str):
    d = _tableone_dir()
    filepath = d / filename
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


@router.get("/data/{tab}")
async def get_tab_data(tab: str):
    """Return data for a specific tab."""
    d = _tableone_dir()
    if not d.exists():
        raise HTTPException(404, "Table One results not found")

    handlers = {
        "cohort": _cohort_data,
        "demographics": _demographics_data,
        "medications": _medications_data,
        "imv": _imv_data,
        "sofa_cci": _sofa_cci_data,
        "outcomes": _outcomes_data,
    }

    handler = handlers.get(tab)
    if handler is None:
        raise HTTPException(404, f"Unknown tab: {tab}")

    return handler(d)


# ── helpers ──────────────────────────────────────────────────────────

def _file_exists_or_none(path: Path):
    return str(path.name) if path.exists() else None


def _df_to_table(df: pd.DataFrame) -> dict:
    """Convert DataFrame to JSON-safe table dict, replacing NaN with None."""
    return {
        "columns": df.columns.tolist(),
        "data": df.where(df.notnull(), None).to_dict(orient="records"),
    }


def _cohort_data(d: Path):
    figures_dir = d / "figures"
    sankeys = []
    if figures_dir.exists():
        for fname, label in [
            ("sankey_matplotlib_icu.png", "ICU Patients"),
            ("sankey_matplotlib_others.png", "Other Patients"),
            ("sankey_matplotlib_high_o2_support.png", "High O2 Support"),
            ("sankey_matplotlib_vaso_support.png", "Vasopressor Support"),
        ]:
            if (figures_dir / fname).exists():
                sankeys.append({"filename": f"figures/{fname}", "label": label})

    return {
        "consort": _file_exists_or_none(d / "consort_flow_diagram.png"),
        "upset": _file_exists_or_none(d / "cohort_intersect_upset_plot.png"),
        "venn": _file_exists_or_none(d / "venn_all_4_groups.png"),
        "code_status": _file_exists_or_none(
            d / "code_status_stacked_bar_with_missingness_excl_missing_cat.png"
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


def _demographics_data(d: Path):
    result: dict = {"tables": {}, "images": []}

    by_year = d / "table_one_by_year.csv"
    if by_year.exists():
        df = _demographics_slice(pd.read_csv(by_year))
        result["tables"]["by_year"] = _df_to_table(df)

    overall = d / "table_one_overall.csv"
    if overall.exists():
        df = _demographics_slice(pd.read_csv(overall))
        result["tables"]["overall"] = _df_to_table(df)

    return result


def _medications_data(d: Path):
    result: dict = {"html_plots": [], "csv_files": []}

    for label, fname in [
        ("Vasoactive Area Curve (7d)", "vasoactive_area_curve_7d.html"),
        ("Vasoactive Median Dose by Hour", "vasoactive_median_dose_by_hour.html"),
        ("Sedative Area Curve (7d)", "sedative_area_curve_7d.html"),
        ("Sedative Median Dose by Hour", "sedative_median_dose_by_hour.html"),
        ("Paralytic Area Curve (7d)", "paralytic_area_curve_7d.html"),
        ("Paralytic Median Dose by Hour", "paralytic_median_dose_by_hour.html"),
    ]:
        if (d / fname).exists():
            result["html_plots"].append({"label": label, "filename": fname})

    meds_csv = d / "medications_summary_stats.csv"
    if meds_csv.exists():
        df = pd.read_csv(meds_csv)
        tbl = _df_to_table(df)
        tbl["label"] = "Medication Summary Statistics"
        result["csv_files"].append(tbl)

    return result


def _imv_data(d: Path):
    result: dict = {"images": [], "csv_files": []}

    for label, fname in [
        ("Tidal Volume - Volume Control Modes", "tidal_volume_volume_control_modes.png"),
        ("Pressure Control Mode", "pressure_control_pressure_control_mode.png"),
        ("Mode Proportions (First 24h)", "mode_proportions_first_24h_vertical.png"),
        ("Ventilator Settings Table", "ventilator_settings_table.png"),
    ]:
        if (d / fname).exists():
            result["images"].append({"label": label, "filename": fname})

    for label, fname in [
        ("Ventilator Settings by Device Mode", "ventilator_settings_by_device_mode.csv"),
        ("Ventilator Settings Counts", "ventilator_settings_counts_by_device_mode.csv"),
    ]:
        if (d / fname).exists():
            df = pd.read_csv(d / fname)
            tbl = _df_to_table(df)
            tbl["label"] = label
            tbl["filename"] = fname
            result["csv_files"].append(tbl)

    return result


def _sofa_cci_data(d: Path):
    result: dict = {"images": [], "csv_files": []}

    for label, fname in [
        ("SOFA Score & Mortality", "sofa_mortality_histogram.png"),
        ("CCI Mortality & Hospice", "cci_mortality_hospice_comprehensive.png"),
        ("Comorbidity Prevalence", "comorbidities_per_1000_barplot.png"),
    ]:
        if (d / fname).exists():
            result["images"].append({"label": label, "filename": fname})

    comorbid_csv = d / "comorbidities_per_1000_hospitalizations.csv"
    if comorbid_csv.exists():
        df = pd.read_csv(comorbid_csv)
        tbl = _df_to_table(df)
        tbl["label"] = "Comorbidities per 1000 Hospitalizations"
        result["csv_files"].append(tbl)

    return result


def _outcomes_data(d: Path):
    result: dict = {"images": []}

    for label, fname in [
        ("Hospice & Mortality Trends", "hospice_mortality_combined_trends.png"),
    ]:
        if (d / fname).exists():
            result["images"].append({"label": label, "filename": fname})

    return result
