# How to Run CLIF-TableOne

## Prerequisites

- **Python 3.11+**
- **[UV package manager](https://docs.astral.sh/uv/)** — install with:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **CLIF 2.1 data** in parquet (or CSV/FST) format

## 1. Install Dependencies

```bash
cd CLIF-TableOne
uv sync
```

> **Note:** Always use `uv run` to execute commands — this ensures the project's virtual environment is used (not your system Python).

## 2. Configure

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
| `site_name` | Your institution name (e.g., `UCMC`, `MIMIC`) |
| `tables_path` | Absolute path to the directory containing your CLIF data files (no trailing `/`) |
| `file_type` | Data format: `parquet`, `csv`, or `fst` |
| `timezone` | Your site's timezone (e.g., `US/Central`, `US/Eastern`, `America/Chicago`) |

## 3. Run the Analysis Pipeline

```bash
uv run python run_project.py --no-summary --get-ecdf
```

This validates all 18 CLIF tables, collects MCIDE data, generates the (critical-illness) Table One, and computes ECDF bins.

| Flag | Purpose |
|---|---|
| `--no-summary` | Skip summary generation |
| `--get-ecdf` | Compute ECDF distributions for visualizations |
| `--ward` | **Also** generate the parallel Ward Table One (see section 3a below) |
| `--ward-only` | Only generate the Ward Table One (no validation, no critical-illness Table One) |

For all available options: `uv run python run_project.py --help`

### 3a. Ward Table One (optional)

In addition to the **critical-illness** cohort (encounters with an ICU stay OR died/hospice), you can generate a parallel **ward** Table One whose cohort is **every adult hospitalization that touched a ward at any point** (`location_category == 'ward'`):

```bash
uv run python run_project.py --no-summary --get-ecdf --ward
```

The ward Table One runs **after** the critical-illness Table One in an **isolated subprocess** so peak memory equals the larger of the two cohorts (not the sum) — important for 16GB systems. Outputs go to `output/final/overall_ward/{tableone,figures,...}/` and the intermediate parquet to `output/intermediate/final_tableone_ward_df.parquet`. Downstream pipelines (ECDF, MCIDE) are unaffected — they continue to read the critical-illness parquet.

**Common ward-related commands:**

| Command | What it runs |
|---|---|
| `uv run python run_project.py --ward` | Validation + critical-illness Table One + **ward Table One** |
| `uv run python run_project.py --get-ecdf --ward` | Full workflow + ward Table One + ECDF |
| `uv run python run_project.py --tableone-only --ward` | Both Table Ones (no validation, no ECDF) |
| `uv run python run_project.py --ward-only` | Just the ward Table One |
| `uv run run_tableone_ward.py` | Direct entry point (ward only, bypasses project runner) |
| `uv run run_tableone_all.py` | Both Table Ones in subprocess isolation (bypasses project runner) |

**What's in the ward Table One:**

- Cohort: every adult hospitalization that touched a ward — includes ward→ICU, ward→death, ward-only, etc.
- Encounter Types section has **5 rows** showing how many ward encounters fall into each critical-illness stratum:
  1. ICU encounters (ward → ICU subset)
  2. Advanced respiratory support
  3. Vasoactive support
  4. Other critically ill (died on ED/ward without escalation)
  5. **Ward only (survived, no critical care)** — survivor catch-all
- **Stripped from the ward Table One** (memory/time optimization, ICU-centric metrics): SOFA scores, ICU length of stay, ICU episodes, IMV/ventilator settings (waterfall, first-24h vent settings), medication-from-ICU plot
- All other sections (demographics, mortality, hospital LOS, comorbidities, CRRT, sepsis, code status) are present and computed against the ward cohort

## 4. Launch the Web App

```bash
uv run uvicorn server.main:app --reload
```

Open **http://127.0.0.1:8000** in your browser.

| Flag | Purpose |
|---|---|
| `--reload` | Auto-restart on code changes (development mode) |
| `--host 0.0.0.0` | Allow access from other machines on the network |
| `--port 8080` | Use a different port (default: 8000) |

## 5. Using the App

- **Validation Tab** — Review validation results for each CLIF table
- **mCIDE Tab** — View mCIDE and summary statistics
- **Table One Results** — Cohort analysis, demographics, medications, IMV, SOFA/CCI, outcomes
- **Feedback** — Classify validation errors as Accepted/Rejected/Pending; save to update table status

## Windows

Use the provided scripts that handle UTF-8 encoding:

```batch
run_project_windows.bat --no-summary --get-ecdf
```

Or PowerShell:
```powershell
.\run_project_windows.ps1 --no-summary --get-ecdf
```

If you encounter Unicode issues, set `PYTHONIOENCODING=utf-8` or enable system-wide UTF-8 (see [README.md](README.md#windows-unicode-troubleshooting)).

## Output

Results are written to `output/final/`. The directory tree is organized by **cohort first, artifact type second**:

```
output/final/
├── overall/                    # Critical-illness cohort (ICU stay OR died/hospice)
│   ├── tableone/              # table_one_overall.csv, mortality_rates, etc.
│   ├── figures/               # CONSORT, sankey, venn, upset, ventilator settings, ...
│   ├── ecdf/                  # ECDF parquets ({labs,vitals,respiratory_support}/)
│   ├── bins/                  # Binned distributions
│   ├── summary_stats/         # Summary statistics JSONs/CSVs
│   └── mcide/                 # MCIDE value counts
├── overall_ward/               # Ward cohort (only if --ward was used)
│   ├── tableone/              # ward Table One CSVs (no SOFA, no ICU LOS, no IMV settings)
│   ├── figures/               # ward CONSORT, sankey, code status, etc.
│   ├── summary_stats/         # ward summary stats
│   └── strata/                # stratified ward outputs (subsets within the ward cohort)
│       ├── icu/{tableone,figures,summary_stats}/
│       ├── advanced_resp/{...}
│       ├── vaso/{...}
│       └── deaths/{...}
├── strata/                     # Stratified critical-illness outputs
│   ├── icu/{tableone,figures,ecdf,bins,summary_stats}/
│   ├── advanced_resp/{...}
│   ├── vaso/{...}
│   └── deaths/{...}
├── validation/                 # Data quality assessment (DQA)
│   ├── json_reports/          # <table>_dqa.json + missing_data_stats / errors CSVs
│   ├── consolidated/          # consolidated_validation.csv + summary JSONs
│   ├── feedback/              # *_validation_response.json (user-classified errors)
│   ├── monthly_trends/        # monthly trend CSVs
│   └── pdf_reports/           # validation report PDFs (per-table + combined)
├── configs/                    # config snapshots used for the run
├── meta/                       # run metadata, execution reports, log files
│   └── workflow_logs/         # timestamped pipeline execution logs
└── stats/                      # collection_statistics.csv (ECDF coverage)
```

Intermediate parquets (in `output/intermediate/`):

| File | Cohort | Consumed by |
|---|---|---|
| `final_tableone_df.parquet` | critical-illness | ECDF, MCIDE, collection stats (via `modules/strata.py`) |
| `final_tableone_ward_df.parquet` | ward (only if `--ward` was used) | not consumed by downstream pipelines (parallel file by design) |

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'fastapi'` | You ran `uvicorn` with system Python. Use `uv run uvicorn ...` instead. |
| `Could not load config` on startup | Ensure `config/config.json` exists and is valid JSON. |
| `uv sync` fails on clifpy | clifpy must be at `/Users/dema/WD/clifpy` (or update `pyproject.toml` path). |
| Port already in use | Use `--port 8080` or kill the existing process. |
