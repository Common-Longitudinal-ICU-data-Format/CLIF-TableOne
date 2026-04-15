# CLIF Table One

A validation and Table-One generation tool for the CLIF 2.1 (Common Longitudinal ICU Format) data standard. It runs `clifpy` data-quality checks on all 18 CLIF tables, builds a critical-illness Table One (and an optional ward Table One), pre-computes ECDF / quantile-bin distributions for downstream visualizations, and serves everything through a FastAPI web app.

---

## 1. Supported tables

All 18 CLIF 2.1 tables:

- **Core:** patient, hospitalization, ADT
- **Clinical:** code_status, labs, vitals, patient_assessments, patient_procedures, hospital_diagnosis
- **Respiratory:** respiratory_support, position
- **Medications:** medication_admin_continuous, medication_admin_intermittent
- **Microbiology:** culture, non-culture, susceptibility
- **Devices:** crrt_therapy, ecmo_mcs

---

## 2. Prerequisites

- **Python 3.11+**
- **[`uv`](https://docs.astral.sh/uv/)** package manager:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **CLIF 2.1 data** in `parquet` (or `csv` / `fst`) format

---

## 3. Install

```bash
cd CLIF-TableOne
uv sync
```

> Always invoke commands through `uv run` so the project's virtualenv (not your system Python) is used.

---

## 4. Configure

Copy the template and fill in your site details:

```bash
cp config/config_template.json config/config.json
```

Edit `config/config.json`:

```json
{
    "site_name": "Your_Site_Name",
    "tables_path": "/path/to/your/clif/data",
    "file_type": "parquet",
    "timezone": "US/Central"
}
```

| Field | Description |
|---|---|
| `site_name` | Your institution name (e.g. `UCMC`, `MIMIC`) |
| `tables_path` | Absolute path to the directory holding your CLIF files (no trailing `/`) |
| `file_type` | `parquet`, `csv`, or `fst` |
| `timezone` | Site timezone, e.g. `US/Central`, `US/Eastern`, `America/Chicago` |

---

## 5. Run the analysis pipeline

```bash
uv run python run_project.py --no-summary --get-ecdf
```

This validates all 18 CLIF tables, builds the critical-illness Table One, and computes ECDF / bins distributions.

| Flag | Purpose |
|---|---|
| `--no-summary` | Skip per-table summary statistics generation |
| `--get-ecdf` | Compute ECDF + quantile bins for labs / vitals / respiratory_support |
| `--ward` | **Also** generate the parallel ward Table One (see §5a) |
| `--ward-only` | Only generate the ward Table One (skip validation + critical-illness Table One) |
| `--validate-only` | Run validation only |
| `--tableone-only` | Run Table One only |
| `--get-ecdf-only` | Run ECDF only |

For the full flag list: `uv run python run_project.py --help`. Advanced workflows and granular per-table commands live in [`advanced_usage.md`](advanced_usage.md).

### 5a. Ward Table One (optional)

In addition to the **critical-illness** cohort (encounters with an ICU stay OR died/hospice), you can generate a parallel **ward** Table One whose cohort is **every adult hospitalization that touched a ward at any point**:

```bash
uv run python run_project.py --no-summary --get-ecdf --ward
```

The ward Table One runs **after** the critical-illness Table One in an **isolated subprocess** so peak memory equals the larger of the two cohorts (not the sum) — important on 16 GB systems. Outputs go to `output/final/overall_ward/{tableone,figures,...}/`. Downstream pipelines (ECDF, MCIDE) keep reading the critical-illness parquet — they are unaffected.

| Command | What it runs |
|---|---|
| `uv run python run_project.py --ward` | Validation + critical-illness Table One + ward Table One |
| `uv run python run_project.py --get-ecdf --ward` | Full workflow + ward Table One + ECDF |
| `uv run python run_project.py --tableone-only --ward` | Both Table Ones (no validation, no ECDF) |
| `uv run python run_project.py --ward-only` | Just the ward Table One |
| `uv run run_tableone_ward.py` | Direct ward entry point (bypasses the project runner) |
| `uv run run_tableone_all.py` | Both Table Ones in subprocess isolation (bypasses the project runner) |

**What's in the ward Table One:**

- Cohort: every adult hospitalization that touched a ward — includes ward→ICU, ward→death, ward-only, etc.
- Encounter Types section has **5 rows** showing how many ward encounters fall into each critical-illness stratum:
  1. ICU encounters (ward→ICU subset)
  2. Advanced respiratory support
  3. Vasoactive support
  4. Other critically ill (died on ED/ward without escalation)
  5. **Ward only (survived, no critical care)** — the survivor catch-all
- **Stripped** from the ward Table One (memory + time optimization, ICU-centric metrics): SOFA scores, ICU LOS, ICU episodes, IMV/ventilator settings, medication-from-ICU plot
- All other sections (demographics, mortality, hospital LOS, comorbidities, CRRT, sepsis, code status) are present and computed against the ward cohort.

---

## 6. Launch the web app

```bash
uv run uvicorn server.main:app --reload
```

Open **http://127.0.0.1:8000** in a browser.

| Flag | Purpose |
|---|---|
| `--reload` | Auto-restart on code changes (development mode) |
| `--host 0.0.0.0` | Allow access from other machines on the network |
| `--port 8080` | Use a different port (default: 8000) |

The web app is FastAPI + a vanilla-JS SPA under `static/`. The legacy Streamlit app is kept as `app_streamlit.py` but is no longer the primary interface.

Tabs:

- **Validation** — Per-table DQA results
- **mCIDE** — Value counts and summary statistics
- **Table One Results** — Cohort analysis: demographics, medications, IMV, SOFA/CCI, outcomes
- **Feedback** — Classify validation errors as Accepted / Rejected / Pending; saves to `output/final/validation/feedback/<table>_validation_response.json` and updates the table status

---

## 7. Reviewing validation errors

The feedback workflow lets reviewers classify validation errors:

1. Open the **Validation** tab
2. Enable **"Review Status-Affecting Errors"** and scroll to the error list
3. For each error, choose:
   - **Accepted** — legitimate issue requiring attention
   - **Rejected** — site-specific variation (provide a justification)
   - **Pending** — not yet reviewed
4. Save feedback. Table status is recomputed:
   - Status becomes **complete** only when **all** errors are rejected
   - Any accepted or pending error keeps the table at its original status

After classifying errors, click **Regenerate reports** on the Home page to recompile the combined PDF + consolidated CSV.

---

## 8. Cohort definitions

This is the part most likely to surprise readers, so it gets its own section.

### 8.1 Adult filter

```python
# modules/tableone/generator.py:1441
adult_encounters = adult_encounters[
    (adult_encounters['age_at_admission'] >= 18) & (adult_encounters['age_at_admission'].notna())
]
```

All cohorts require `age_at_admission >= 18`. There is no admission-year restriction in the current code (the year filter at lines 1448-1449 is commented out for all sites).

### 8.2 Encounter stitching

Linked admissions are joined into a single `encounter_block` *before* cohort selection, using `clifpy.utils.stitching_encounters.stitch_encounters` (`modules/tableone/generator.py:61`). Every flag below is computed at the **encounter-block level**, not the `hospitalization_id` level — so a patient who bounces ED → ward → ICU within a single stitched encounter counts as one ICU encounter, not three.

### 8.3 Per-encounter-block flags

Every flag is a 0/1 indicator on the stitched encounter block. The strata in §9 are just filters on these flags.

| Flag | Definition | Set at |
|---|---|---|
| `icu_enc` | Encounter ever had a row with `location_category` containing `'icu'` | `generator.py:1533, 1539` |
| `death_enc` | Encounter ever had a row with `discharge_category in ('expired', 'hospice')` | `generator.py:1534, 1540` |
| `ward_enc` | Encounter ever had a row with `location_category == 'ward'` | `generator.py:1535, 1541` |
| `high_support_enc` | Encounter ever received `imv` / `nippv` / `cpap` / `high flow nc` (`respiratory_support.device_category`) | `generator.py:1646, 1659` |
| `nippv_hfnc_enc` | Encounter ever received `nippv` or `high flow nc` with `lpm_set >= 30` (`respiratory_support.device_category`) | `generator.py:1725-1740` |
| `vaso_support_enc` | Encounter ever received `norepinephrine`, `epinephrine`, `phenylephrine`, `vasopressin`, `dopamine`, or `angiotensin` (`medication_admin_continuous.med_category`) | `generator.py:1693, 1706` |
| `vaso_ed_icu_enc` | First vasopressor was administered in the ED AND any subsequent ADT location includes ICU | `generator.py:1904-1906` |
| `vaso_ed_ward_enc` | First vasopressor was administered in the ED AND any subsequent ADT location includes ward but NOT ICU | `generator.py:1907-1909` |
| `no_imv_enc` | Encounter is in the critical-illness cohort AND never received invasive mechanical ventilation (`on_vent == 0`) | `generator.py:3449` |
| `no_imv_icu_enc` | `no_imv_enc AND icu_enc` | `generator.py:3450-3452` |
| `no_imv_no_icu_enc` | `no_imv_enc AND NOT icu_enc` | `generator.py:3453-3455` |
| `is_procedural_ld_only` | No ICU AND only `procedural` / `l&d` locations — in critical-illness mode this zeros out the support flags so a procedural-only encounter cannot be flagged as advanced resp / vaso | `generator.py:1584-1587, 1727-1729` |
| `other_critically_ill` | `death_enc==1 AND icu_enc==0 AND vaso_support_enc==0 AND high_support_enc==0` (died in ED/ward without escalation) | `generator.py:1764-1769` |

### 8.4 Critical-illness cohort (default)

This is what `run_project.py` produces in `output/final/overall/`:

```python
# modules/tableone/generator.py:1549
all_encounters['cohort_enc'] = (all_encounters['icu_enc'] | all_encounters['death_enc']).astype(int)
```

> Adults whose encounter block touched an ICU **OR** discharged as expired/hospice.

**Vasoactive and advanced respiratory support are not inclusion criteria.** They are flags computed *after* the cohort is fixed and are used only for stratification (§9). Concretely: an adult who got norepinephrine on the ward, never touched ICU, and survived → **not** in the critical-illness cohort. The code comment at `generator.py:1761` even points this out — *"in critical-illness mode (any encounter in the cohort with icu_enc==0 already had death_enc==1)"*.

### 8.5 Ward cohort

This is what `run_project.py --ward` adds to `output/final/overall_ward/`:

```python
# modules/tableone/generator.py:1547
all_encounters['cohort_enc'] = all_encounters['ward_enc']
```

> Adults whose encounter block touched a ward at any point — includes ward→ICU, ward→death, and ward-only encounters.

### 8.6 Stale docstring note

The legacy comment at `modules/tableone/generator.py:22-23` describes inclusion as *"at least one ICU stay or those who had only emergency department or ward encounters and either died or received life support"*. The "received life support" branch is **not** implemented by the code at line 1549 — vaso/advanced-resp encounters that never reached ICU and survived are excluded. The docstring is on a separate cleanup list.

---

## 9. Strata

The strata directories under `output/final/strata/` are subsets of the **critical-illness cohort**, filtered by a single flag. The mapping is the single source of truth in `modules/strata.py:24-41`:

| Directory | Flag | Definition | ECDF temporal window |
|---|---|---|---|
| `strata/icu/` | `icu_enc` | Encounter touched a location with `location_category == 'icu'` | ICU stay (`in_dttm` → `out_dttm` from ADT) |
| `strata/advanced_resp/` | `high_support_enc` | Received `imv` / `nippv` / `cpap` / `high flow nc` at any point | First qualifying device `recorded_dttm` → `discharge_dttm` |
| `strata/advanced_resp/icu/` | `high_support_icu_enc` | Advanced resp **AND** ICU | Same as `advanced_resp/` |
| `strata/advanced_resp/no_icu/` | `high_support_no_icu_enc` | Advanced resp **AND NOT** ICU (by construction these are deaths) | Same as `advanced_resp/` |
| `strata/nippv_hfnc/` | `nippv_hfnc_enc` | Received `nippv` or `high flow nc` with `lpm_set >= 30` | First qualifying device `recorded_dttm` → `discharge_dttm` |
| `strata/nippv_hfnc/icu/` | `nippv_hfnc_icu_enc` | NIPPV/HFNC **AND** ICU | Same as `nippv_hfnc/` |
| `strata/nippv_hfnc/no_icu/` | `nippv_hfnc_no_icu_enc` | NIPPV/HFNC **AND NOT** ICU (by construction deaths) | Same as `nippv_hfnc/` |
| `strata/vaso/` | `vaso_support_enc` | Received `norepinephrine`, `epinephrine`, `phenylephrine`, `vasopressin`, `dopamine`, or `angiotensin` | First vasopressor `admin_dttm` → `discharge_dttm` |
| `strata/vaso/icu/` | `vaso_icu_enc` | Vaso **AND** ICU | Same as `vaso/` |
| `strata/vaso/no_icu/` | `vaso_no_icu_enc` | Vaso **AND NOT** ICU (by construction deaths) | Same as `vaso/` |
| `strata/vaso/ed_icu/` | `vaso_ed_icu_enc` | First vasopressor in ED **AND** any subsequent location includes ICU | Same as `vaso/` |
| `strata/vaso/ed_ward/` | `vaso_ed_ward_enc` | First vasopressor in ED **AND** any subsequent location includes ward but **NOT** ICU | Same as `vaso/` |
| `strata/no_imv/` | `no_imv_enc` | Critically ill but **never** received invasive mechanical ventilation (`on_vent == 0`) | ICU stay |
| `strata/no_imv/icu/` | `no_imv_icu_enc` | No IMV **AND** ICU | Same as `no_imv/` |
| `strata/no_imv/no_icu/` | `no_imv_no_icu_enc` | No IMV **AND NOT** ICU (by construction deaths) | Same as `no_imv/` |
| `strata/deaths/` | `death_enc` | `discharge_category in ('expired', 'hospice')` | ICU stay (`in_dttm` → `out_dttm` from ADT) |

Strata are not independent cohorts — every stratum is a subset of `cohort_enc == 1`. So `strata/vaso/no_icu/` is "patients in the critical-illness cohort who got a vasopressor but never touched an ICU" — and because the cohort selector at line 1549 already requires ICU OR death, every patient in that stratum is necessarily a death.

> **ECDF temporal windows:** The `overall/`, `icu`, `deaths`, and `no_imv` strata use ICU stay windows from the ADT table. The `vaso`, `advanced_resp`, and `nippv_hfnc` strata (including their `/icu` and `/no_icu` sub-strata) use **event-onset windows** — starting from the first qualifying medication or device placement through `discharge_dttm`. This ensures that `/no_icu` sub-strata (which have no ICU windows) still produce ECDF/bins output, and that the distributions reflect physiology from the onset of the clinical escalation. See `modules/ecdf/generator.py:STRATUM_WINDOW_TYPE` for the mapping.

---

## 10. Output layout

```
output/final/
├── overall/                    Critical-illness cohort
│   ├── tableone/               Demographics, mortality, comorbidities, ventilator, medications, ...
│   ├── figures/                CONSORT, sankey, venn, upset, area curves
│   ├── ecdf/                   {labs,vitals,respiratory_support}/ ECDF parquets
│   ├── bins/                   {labs,vitals,respiratory_support}/ quantile-bin parquets
│   ├── summary_stats/          Per-category mean/median/IQR
│   └── mcide/                  Minimum CDE value counts
├── overall_ward/               Ward cohort (only present when --ward is used)
│   └── {tableone,figures,summary_stats}/
├── strata/                     Stratified critical-illness subsets
│   ├── icu/                    {tableone,figures,ecdf,bins,summary_stats}/
│   ├── advanced_resp/          {... + icu/ + no_icu/}
│   ├── nippv_hfnc/             {... + icu/ + no_icu/}
│   ├── vaso/                   {... + icu/ + no_icu/ + ed_icu/ + ed_ward/}
│   ├── no_imv/                 {... + icu/ + no_icu/}
│   └── deaths/                 {tableone,figures,ecdf,bins,summary_stats}/
├── validation/                 Data quality assessment
│   ├── json_reports/           <table>_dqa.json + supporting CSVs
│   ├── consolidated/           consolidated_validation.csv + summary JSONs
│   ├── feedback/               <table>_validation_response.json (user-classified errors)
│   ├── monthly_trends/         Monthly trend CSVs
│   └── pdf_reports/            Per-table PDFs + combined PDF
├── stats/                      collection_statistics.csv (+ per-stratum variants)
├── meta/                       Execution reports, workflow logs, run metadata
│   └── workflow_logs/          Timestamped pipeline stdout/stderr
└── configs/                    Snapshot of config files used for the run
```

For **what each file contains**, **the cohort filter applied**, and **which module generates it**, see [`OUTPUT_REFERENCE.md`](OUTPUT_REFERENCE.md). It also has a worked example for `albumin_g_dL.parquet`.

A short note on the ECDF time window, since it's frequently misremembered: ECDF and bin parquets cover the **entire ICU stay window** for each encounter (`in_dttm` ≤ value timestamp ≤ `out_dttm`, summed across every `location_category == 'icu'` row in `clif_adt`). There is **no 168-hour cap**. The 24h/48h/72h numbers in `stats/collection_statistics*.csv` are observation-count coverage statistics, not the ECDF input filter (`modules/ecdf/statistics.py:196-200` vs `modules/ecdf/generator.py:507-511`).

---

## 11. Windows

Use the provided wrapper scripts that handle UTF-8 encoding:

**Batch:**
```batch
run_project_windows.bat --no-summary --get-ecdf
app_windows.bat
```

**PowerShell:**
```powershell
.\run_project_windows.ps1 --no-summary --get-ecdf
.\app_windows.ps1
```

If you still see encoding errors with emojis or Unicode characters, set `PYTHONIOENCODING=utf-8`:

```batch
:: Command Prompt
set PYTHONIOENCODING=utf-8
python run_project.py
```

```powershell
# PowerShell
$env:PYTHONIOENCODING="utf-8"
python run_project.py
```

Or enable system-wide UTF-8: Settings → Time & Language → Language → Administrative language settings → Change system locale → check **"Beta: Use Unicode UTF-8 for worldwide language support"** → restart.

Or use Python's UTF-8 mode for a one-off run:

```batch
python -X utf8 run_project.py
```

---

## 12. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'fastapi'` | You ran `uvicorn` with system Python. Use `uv run uvicorn ...` instead. |
| `Could not load config` on startup | Ensure `config/config.json` exists and is valid JSON. |
| `uv sync` fails on `clifpy` | `clifpy` is now a PyPI dependency (no editable install needed). Ensure your `uv` is up to date and rerun `uv sync`. |
| Port already in use | Use `--port 8080` or kill the existing process. |
| Out of memory on the full dataset | Use `--ward-only` separately, or run validation and Table One in sequence rather than together. |

---

## 13. Further reading

- [`OUTPUT_REFERENCE.md`](OUTPUT_REFERENCE.md) — Per-file detail for everything under `output/final/`
- [`advanced_usage.md`](advanced_usage.md) — Granular per-table validation, sampling workflows, ECDF configuration
- [`TABLEONE_VIEWER_GUIDE.md`](TABLEONE_VIEWER_GUIDE.md) — Web app viewer documentation
- [`CHANGE.md`](CHANGE.md) — Release notes
- [CLIF documentation](https://clif-icu.com)
- [clifpy](https://github.com/Common-Longitudinal-ICU-data-Format/clifpy)

---

## Support

For issues or questions, please open an issue in the project repository.
