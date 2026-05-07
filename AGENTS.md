# CLIF TableOne — AI Assistant Context

You are helping a site analyst work with the CLIF TableOne pipeline: validating CLIF 2.1 clinical data, generating Table One statistics, and reviewing results.

## Quick Start

```bash
uv sync                                          # Install dependencies
cp config/config_template.json config/config.json # Create config
# Edit config.json with site details
uv run python run_project.py --no-summary         # Run everything
```

## Project Structure

- `config/config.json` — Site configuration (tables_path, timezone, file_type)
- `modules/tableone/generator.py` — Main Table One generation pipeline
- `modules/ecdf/generator.py` — ECDF/bins computation
- `run_project.py` — Workflow orchestrator (validation + tableone + ward + ECDF)
- `server/` — FastAPI dashboard for interactive review

## Key Commands

```bash
uv run python run_project.py --no-summary         # Full run (validation + CI + ward + ECDF)
uv run python run_project.py --tableone-only       # CI + ward tableone only (skip validation/ECDF)
uv run python run_project.py --validate-only       # Validation only
uv run python run_project.py --no-ward --no-ecdf   # CI tableone only (fastest)
uv run uvicorn server.main:app --port 8000         # Launch dashboard
```

## Reviewing Validation Results

When asked to review validation results, read these files:

1. `output/final/validation/json_reports/*_dqa.json` — Per-table DQA results (conformance, completeness, relational, plausibility checks)
2. `output/final/validation/consolidated/consolidated_validation.csv` — Summary across all tables
3. `output/final/meta/workflow_logs/workflow_execution_latest.log` — Pipeline log with errors/warnings
4. `output/final/meta/unit_mismatches.log` — Lab units that didn't match CLIF schema (ECDF only)

### DQA JSON Structure

Each `*_dqa.json` has these sections:
- `conformance` — Table exists, required columns present, correct data types, datetime formats
- `completeness` — Null rates per column, missing value patterns
- `relational` — Foreign keys match across tables (e.g., hospitalization_id in adt exists in hospitalization)
- `plausibility` — Values within expected clinical ranges

### How to Prioritize Validation Errors

1. **Conformance errors (fix first)** — Missing columns or wrong types will crash the pipeline
2. **Relational errors (fix next)** — Orphan IDs cause silent data loss in joins
3. **Completeness warnings (investigate if >5%)** — Small amounts of nulls (<1%) are normal
4. **Plausibility warnings (informational)** — Outliers are handled by the pipeline; review if flagged rates are high

### Common Fixes by Error Type

| Error | Cause | Fix |
|---|---|---|
| `column X not found` | Site ETL missing this column | Add column to ETL mapping |
| `N% null in column Y` | Missing data | Acceptable if <1%; investigate source if >5% |
| `UNIT MISMATCH: found 'mm[hg]' expected 'mmhg'` | Unit spelling differs from CLIF spec | Normalize in ETL or update clifpy schema |
| `NOT IN CLIF SPEC: category (unit)` | Site has extra labs not in CLIF 2.1 | Safe to ignore — these are excluded from ECDF |
| `N values outside range [X, Y]` | Outliers in clinical data | Pipeline handles outliers; review if rate >10% |
| `hospitalization_id not found in hospitalization table` | Orphan foreign key | Fix ETL join; data from these encounters is silently dropped |

### Reviewing Table One Results

Key output files:
- `output/final/overall/tableone/table_one_overall.csv` — CI cohort Table One
- `output/final/overall_ward/tableone/table_one_overall.csv` — Ward cohort Table One
- `output/final/overall/tableone/strobe_counts.csv` — Cohort flow counts
- `output/final/overall/tableone/mortality_rates.csv` — Per-group mortality

Cross-check these for consistency:
- `strobe_counts.csv` N should match Table One N for the same cohort
- Sepsis encounters in strobe should match "Encounters with >=1 sepsis event" in Table One
- Mortality rate in strobe should match "Hospital mortality" percentage in Table One

### Strata

Stratified results live under `output/final/strata/{icu,advanced_resp,nippv_hfnc,vaso,no_imv,deaths}/tableone/`. Each stratum is a subset of the critical-illness cohort filtered by a single encounter flag. See README.md section 5 for the full mapping.

## Troubleshooting

### Pipeline crashes
1. Check `output/final/meta/workflow_logs/workflow_execution_latest.log` for the traceback
2. Common causes: missing columns (conformance), datetime timezone mismatches, null values in location_category

### Windows-specific
- Use `run_project_windows.bat` or set `PYTHONIOENCODING=utf-8` before running
- If running from a mapped drive (Z:\), paths may resolve differently — this is handled in the code

### Memory issues
- Add `--no-ecdf` to skip ECDF generation (saves ~8 GB)
- The pipeline year-shards Table One processing to cap per-year memory
- Check the execution report at `output/final/meta/tableone_execution_report.txt` for peak memory

## Do NOT

- Do not modify files under `output/` manually — they are regenerated on each run
- Do not commit `config/config.json` (it's gitignored; contains site-specific paths)
- Do not skip validation errors without understanding the downstream impact
