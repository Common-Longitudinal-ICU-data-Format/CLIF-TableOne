# CLIF Table One

A validation and Table-One generation tool for all 16 beta-ready CLIF 2.1 tables [see data dictionary](https://clif-icu.com/data-dictionary/data-dictionary-2.1.0).

---

## 1. Prerequisites

- **Python 3.11+**
- **[`uv`](https://docs.astral.sh/uv/)** package manager:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **CLIF 2.1 data** in `parquet` (or `csv` / `fst`) format


## 2. Install

```bash
cd CLIF-TableOne
uv sync
```

> Always invoke commands through `uv run` so the project's virtualenv (not your system Python) is used.


## 3. Configure

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

## 4. Run the validation and tableone pipeline

```bash
uv run python run_project.py --no-summary --get-ecdf --ward
```

This validates all 16 beta-ready CLIF tables, builds the critical-illness and ward Table One, and computes ECDF / bins distributions.

| Flag | Purpose |
|---|---|
| `--no-summary` | Skip per-table summary statistics generation |
| `--get-ecdf` | Compute ECDF + quantile bins for labs / vitals / respiratory_support |
| `--ward` | **Also** generate the parallel ward Table One (see §5a) |
| `--ward-only` | Only generate the ward Table One (skip validation + critical-illness Table One) |
| `--validate-only` | Run validation only |
| `--tableone-only` | Run Table One only |
| `--get-ecdf-only` | Run ECDF only |

For the full flag list: `uv run python run_project.py --help`. Advanced workflows and granular per-table commands live in [`advanced_usage.md`](advanced_usage.md). ### DEMA - TO REVIEW

**Windows users:** use `run_project_windows.bat` (Command Prompt) or `run_project_windows.ps1` (PowerShell) instead. Both set `PYTHONIOENCODING=utf-8` / `PYTHONUTF8=1` (so emojis and Unicode log output render correctly) and forward all flags to `run_project.py`:

```bat
REM Command Prompt
run_project_windows.bat --no-summary --get-ecdf --ward
```

```powershell
# PowerShell
.\run_project_windows.ps1 --no-summary --get-ecdf --ward
```

Companion `run_analysis_windows.bat` / `.ps1` wrap `run_analysis.py` the same way.

### 5. Ward Table One 

In addition to the **critical-illness** cohort (see §8.4 for the exact definition), it generates a parallel **ward** Table One whose cohort is **every adult hospitalization that touched a ward at any point**. If you want to run just the ward tableone:

```bash
uv run python run_project.py --ward-only
```

**What's in the ward Table One:**

- Cohort: every adult hospitalization that touched a ward — includes ward→ICU, ward→death, ward-only, etc.
- Encounter Types section has **5 rows** showing how many ward encounters fall into each critical-illness stratum:
  1. ICU encounters (ward→ICU subset)
  2. Advanced respiratory support
  3. Vasoactive support
  4. Other critically ill (died on ED/ward without escalation)
  5. **Ward only (survived, no critical care)** — the survivor catch-all


## 6. To relaunch the web app

```bash
uv run uvicorn server.main:app --reload
```

Open **http://127.0.0.1:8000** in a browser.

| Flag | Purpose |
|---|---|
| `--reload` | Auto-restart on code changes (development mode) |
| `--host 0.0.0.0` | Allow access from other machines on the network |
| `--port 8080` | Use a different port (default: 8000) |

**Windows users:** use `app_fastapi_windows.bat` (Command Prompt) or `app_fastapi_windows.ps1` (PowerShell). Both set `PYTHONIOENCODING=utf-8` / `PYTHONUTF8=1` and launch `uvicorn server.main:app --reload`, forwarding any extra flags:

```bat
REM Command Prompt
app_fastapi_windows.bat --port 8080
```

```powershell
# PowerShell
.\app_fastapi_windows.ps1 --port 8080
```

Tabs:

- **Table One Results** — Cohort analysis: demographics, medications, IMV, SOFA/CCI, outcomes
- **Validation** — Overall DQA results
  - **Feedback** — Classify validation errors as Accepted / Rejected / Pending; saves to `output/final/validation/feedback/<table>_validation_response.json` and updates the table status

## 7. Reviewing validation errors

The feedback workflow lets reviewers classify validation errors:

1. Open the **Validation** tab
2. For each error, choose:
   - **Accepted** — legitimate issue requiring attention
   - **Rejected** — site-specific variation (provide a justification)
   - **Pending** — not yet reviewed
3. Save feedback. Table status is recomputed, any accepted or pending error keeps the table at its original status

---

# Brief pipeline description

### 1. Adult filter

```python
adult_encounters = adult_encounters[
    (adult_encounters['age_at_admission'] >= 18) & (adult_encounters['age_at_admission'].notna())
]
```

All cohorts require `age_at_admission >= 18`.

### 2. Encounter stitching

Linked admissions are joined into a single `encounter_block` *before* cohort selection, using `clifpy.utils.stitching_encounters.stitch_encounters` (`modules/tableone/generator.py:62`) with a 6-hour window — any two `hospitalization_id`s where discharge-to-next-admission ≤ 6 h get merged. Every flag below is computed at the **encounter-block level**, not the `hospitalization_id` level — so a patient discharged and readmitted within 6 hours counts as **one** encounter, not two.

### 3. Per-encounter-block flags

Every flag is a 0/1 indicator on the stitched encounter block. The strata in §9 are just filters on these flags.

| Flag | Definition |
|---|---|
| `icu_enc` | Encounter ever had a row with `location_category` containing `'icu'` |
| `death_enc` | Encounter ever had a row with `discharge_category in ('expired', 'hospice')` |
| `ward_enc` | Encounter ever had a row with `location_category == 'ward'` |
| `high_support_enc` | Encounter ever received `imv` / `nippv` / `cpap` (unconditionally) or `high flow nc` with `lpm_set >= 30` (`respiratory_support.device_category`) |
| `nippv_hfnc_enc` | Encounter ever received `nippv` or `high flow nc` with `lpm_set >= 30` (`respiratory_support.device_category`) |
| `vaso_support_enc` | Encounter ever received `norepinephrine`, `epinephrine`, `phenylephrine`, `vasopressin`, `dopamine`, or `angiotensin` (`medication_admin_continuous.med_category`) |
| `vaso_ed_icu_enc` | First vasopressor was administered in the ED AND any subsequent ADT location includes ICU |
| `vaso_ed_ward_enc` | First vasopressor was administered in the ED AND any subsequent ADT location includes ward but NOT ICU |
| `no_imv_enc` | Encounter is in the critical-illness cohort AND never received invasive mechanical ventilation (`on_vent == 0`; `on_vent` is set to 1 when the encounter has any IMV observation in the respiratory support data) |
| `no_imv_icu_enc` | `no_imv_enc AND icu_enc` |
| `no_imv_no_icu_enc` | `no_imv_enc AND NOT icu_enc` |
| `is_procedural_ld_only` | `icu_enc == 0 AND any(loc in {'procedural', 'l&d'})` |
| `other_critically_ill` | `death_enc==1 AND icu_enc==0 AND vaso_support_enc==0 AND high_support_enc==0` — i.e. the encounter expired/hospice-discharged but never touched ICU, never received vasoactive medications, and never received advanced respiratory support (IMV/NIPPV/CPAP/HFNC≥30); net effect is death in ED or on the ward without escalation |

### 4. Critical-illness cohort 

```python
# modules/tableone/generator.py:1837-1841
final_cohort['cohort_enc'] = (
    (final_cohort['icu_enc'] | final_cohort['death_enc']
     | final_cohort['high_support_enc'] | final_cohort['vaso_support_enc'])
    & (~final_cohort['is_procedural_ld_only'].astype(bool))
).astype(int)
```

> Adults whose encounter block touched an ICU **OR** discharged as expired/hospice in ED/Ward **OR** received advanced respiratory support **OR** received vasoactive medications — excluding encounters flagged `is_procedural_ld_only`.

This means the cohort includes ward survivors who received vasoactive medications or advanced respiratory support but never touched an ICU. Concretely: an adult who got norepinephrine on the ward, never touched ICU, and survived → **in** the critical-illness cohort. The `no_icu` sub-strata (§9) therefore contain both deaths and ward survivors with support, not exclusively deaths.


### 5. Strata

The strata directories under `output/final/strata/` are subsets of the **critical-illness cohort**, filtered by a single flag. The mapping is the single source of truth in `modules/strata.py:24-41`:

| Directory | Flag | Definition | ECDF temporal window |
|---|---|---|---|
| `strata/icu/` | `icu_enc` | Encounter touched a location with `location_category == 'icu'` | ICU stay (`in_dttm` → `out_dttm` from ADT) |
| `strata/advanced_resp/` | `high_support_enc` | Received `imv` / `nippv` / `cpap` (unconditionally) or `high flow nc` with `lpm_set >= 30` at any point | First qualifying device `recorded_dttm` → `discharge_dttm` |
| `strata/advanced_resp/icu/` | `high_support_icu_enc` | Advanced resp **AND** ICU | Same as `advanced_resp/` |
| `strata/advanced_resp/no_icu/` | `high_support_no_icu_enc` | Advanced resp **AND NOT** ICU (includes deaths and ward survivors with support) | Same as `advanced_resp/` |
| `strata/nippv_hfnc/` | `nippv_hfnc_enc` | Received `nippv` or `high flow nc` with `lpm_set >= 30` | First qualifying device `recorded_dttm` → `discharge_dttm` |
| `strata/nippv_hfnc/icu/` | `nippv_hfnc_icu_enc` | NIPPV/HFNC **AND** ICU | Same as `nippv_hfnc/` |
| `strata/nippv_hfnc/no_icu/` | `nippv_hfnc_no_icu_enc` | NIPPV/HFNC **AND NOT** ICU (includes deaths and ward survivors with support) | Same as `nippv_hfnc/` |
| `strata/vaso/` | `vaso_support_enc` | Received `norepinephrine`, `epinephrine`, `phenylephrine`, `vasopressin`, `dopamine`, or `angiotensin` | First vasopressor `admin_dttm` → `discharge_dttm` |
| `strata/vaso/icu/` | `vaso_icu_enc` | Vaso **AND** ICU | Same as `vaso/` |
| `strata/vaso/no_icu/` | `vaso_no_icu_enc` | Vaso **AND NOT** ICU (includes deaths and ward survivors on vasopressors) | Same as `vaso/` |
| `strata/vaso/ed_icu/` | `vaso_ed_icu_enc` | First vasopressor in ED **AND** any subsequent location includes ICU | Same as `vaso/` |
| `strata/vaso/ed_ward/` | `vaso_ed_ward_enc` | First vasopressor in ED **AND** any subsequent location includes ward but **NOT** ICU | Same as `vaso/` |
| `strata/no_imv/` | `no_imv_enc` | Critically ill but **never** received invasive mechanical ventilation (`on_vent == 0`) | ICU stay |
| `strata/no_imv/icu/` | `no_imv_icu_enc` | No IMV **AND** ICU | Same as `no_imv/` |
| `strata/no_imv/no_icu/` | `no_imv_no_icu_enc` | No IMV **AND NOT** ICU (includes deaths and ward survivors without IMV) | Same as `no_imv/` |
| `strata/deaths/` | `death_enc` | `ED/ward discharge_category in ('expired', 'hospice')` | ICU stay (`in_dttm` → `out_dttm` from ADT) |


## Output layout

```
output/
├── final/
│   ├── overall/                    Critical-illness cohort
│   │   ├── tableone/               Demographics, mortality, comorbidities, ventilator, medications, ...
│   │   ├── figures/                CONSORT, sankey, venn, upset, area curves, KM time-to-extubation, daily P/F-S/F
│   │   ├── ecdf/                   {labs,vitals,respiratory_support}/ ECDF parquets
│   │   ├── bins/                   {labs,vitals,respiratory_support}/ quantile-bin parquets
│   │   ├── summary_stats/          Per-category mean/median/IQR
│   │   ├── mcide/                  Minimum CDE value counts
│   │   └── ventilated_aggregates/  Cross-site KM + daily P/F|S/F aggregates (no PHI)
│   │       ├── km_time_to_extubation.csv                Kaplan-Meier numbers per (stratum, timepoint)
│   │       └── min_pf_sf_per_day_post_intubation.csv    Daily min P/F & S/F median/IQR (long format)
│   ├── overall_ward/               Ward cohort (only present when --ward is used)
│   │   └── {tableone,figures,summary_stats}/
│   ├── strata/                     Stratified critical-illness subsets
│   │   ├── icu/                    {tableone,figures,ecdf,bins,summary_stats}/
│   │   ├── advanced_resp/          {... + icu/ + no_icu/}
│   │   ├── nippv_hfnc/             {... + icu/ + no_icu/}
│   │   ├── vaso/                   {... + icu/ + no_icu/ + ed_icu/ + ed_ward/}
│   │   ├── no_imv/                 {... + icu/ + no_icu/}
│   │   └── deaths/                 {tableone,figures,ecdf,bins,summary_stats}/
│   ├── validation/                 Data quality assessment
│   │   ├── json_reports/           <table>_dqa.json + supporting CSVs
│   │   ├── consolidated/           consolidated_validation.csv + summary JSONs
│   │   ├── feedback/               <table>_validation_response.json (user-classified errors)
│   │   ├── monthly_trends/         Monthly trend CSVs
│   │   └── pdf_reports/            Per-table PDFs + combined PDF
│   ├── stats/                      collection_statistics.csv (+ per-stratum variants)
│   └── meta/                       Execution reports, workflow logs, run metadata
│       ├── configs/                Snapshot of config files used for the run
│       └── workflow_logs/          Timestamped pipeline stdout/stderr
└── intermediate/                   Debug logs + scratch (critical-illness only)
    ├── vent_hours_debug.log        Per-encounter waterfall + IMV-hours debug trace
    └── imv_episodes.csv            Per-IMV-episode timestamps (repeated-measures analyses)
```

For **what each file contains**, **the cohort filter applied**, and **which module generates it**, see [`OUTPUT_REFERENCE.md`](OUTPUT_REFERENCE.md).


## Further reading

- [`OUTPUT_REFERENCE.md`](OUTPUT_REFERENCE.md) — Per-file detail for everything under `output/final/`
- [`advanced_usage.md`](advanced_usage.md) — Granular per-table validation, sampling workflows, ECDF configuration
- [`TABLEONE_VIEWER_GUIDE.md`](TABLEONE_VIEWER_GUIDE.md) — Web app viewer documentation

---

## Support

For issues or questions, please open an issue in the project repository.
