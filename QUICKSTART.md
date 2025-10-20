# CLIF TableOne - Quick Start Guide

Get started with CLIF validation and table one generation in 3 easy steps!

## Prerequisites

1. **Python 3.8+** installed
2. **UV package manager** installed ([install instructions](https://docs.astral.sh/uv/))
3. **CLIF 2.1 data** in parquet or CSV format

## Step 1: Setup (2 minutes)

```bash
# Clone or navigate to project
cd CLIF-TableOne

# Install dependencies
uv sync

# Configure your site
cp config/config.json.example config/config.json  
```

**Edit `config/config.json`:**
```json
{
    "site_name": "Your Hospital Name",
    "tables_path": "/path/to/your/clif/data",
    "filetype": "parquet",
    "timezone": "America/Chicago"
}
```

## Step 2: Run Complete Workflow 

### Option A: Full Analysis with Sample (Recommended for testing)

```bash
python run_project.py --sample
```

This runs:
- ✅ Validation of all 18 CLIF tables (1k ICU sample)
- ✅ Table One generation with visualizations
- ⏱️ Time: ~10-15 minutes

### Option B: Full Analysis with Complete Dataset

```bash
python run_project.py
```

This runs:
- ✅ Validation of all 18 CLIF tables (complete data)
- ✅ Table One generation with visualizations
- ⏱️ Time: ~45-90 minutes

### Option C: Validate Only (Quick Check)

```bash
python run_project.py --validate-only --sample
```

This runs:
- ✅ Validation only (skip table one)
- ⏱️ Time: ~5-10 minutes

## Step 3: Review Results

### Validation Reports
```
output/final/reports/
├── patient_validation_report.pdf
├── adt_validation_report.pdf
├── combined_validation_report.pdf
└── ...
```

### Table One Outputs
```
output/final/tableone/
├── table_one_overall.csv          # Main results table
├── table_one_by_year.csv          # Stratified by year
├── consort_flow_diagram.png       # Cohort flow
├── execution_report.txt           # Memory/timing stats
└── ... (many visualizations)
```

## Common Workflows

### 1. Development/Testing
```bash
# Quick validation check with sample
python run_project.py --validate-only --sample

# If validation looks good, run table one
python run_project.py --tableone-only
```

### 2. Production Run
```bash
# Complete analysis with full dataset
python run_project.py
```

### 3. Specific Tables Only
```bash
# Just patient, hospitalization, and ADT
python run_project.py --tables patient hospitalization adt
```

### 4. Continue Despite Warnings
```bash
# Validate with sample, continue to table one even if warnings
python run_project.py --sample --continue-on-error
```

### 5. Launch App After Completion
```bash
# Run workflow and automatically launch Streamlit app
python run_project.py --sample --launch-app
```

## Web Application Alternative

For interactive exploration:

```bash
uv run streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## Troubleshooting

### "Configuration file not found"
- Ensure `config/config.json` exists
- Check path is `config/config.json` relative to project root

### "Data directory not found"
- Verify `tables_path` in config points to correct location
- Check that CLIF parquet/csv files exist in that directory

### "Memory Error"
- Use `--sample` flag for faster analysis
- Close other applications
- Consider running on machine with more RAM

### "Table not found"
- Ensure file names match expected format:
  - `patient.parquet` or `clif_patient.parquet`
  - `hospitalization.parquet` or `clif_hospitalization.parquet`
  - etc.

## Next Steps

1. **Review validation reports** in `output/final/reports/`
2. **Check table one results** in `output/final/tableone/`
3. **Explore web app** for interactive analysis
4. **Set up automation** (see README.md for cron examples)

## Performance Guide

| Command | Dataset | Time | Use Case |
|---------|---------|------|----------|
| `--validate-only --sample` | Sample | ~5 min | Quick check |
| `--sample` | Sample | ~10-15 min | Development |
| `--validate-only` | Full | ~20-30 min | Pre-flight check |
| (no flags) | Full | ~45-90 min | Production |

## Help & Documentation

- **Full documentation**: See [README.md](README.md)
- **Table one details**: See [code/README_TABLE_ONE.md](code/README_TABLE_ONE.md)
- **Command help**: `python run_project.py --help`
- **Issues**: Report at [GitHub Issues](https://github.com/your-org/CLIF-TableOne/issues)

## Example Session

```bash
# 1. Setup
cd CLIF-TableOne
uv sync
# Edit config/config.json

# 2. Quick validation test
python run_project.py --validate-only --sample
# Review: output/final/reports/

# 3. If validation looks good, run full analysis
python run_project.py --sample
# Review: output/final/tableone/

# 4. Explore interactively
uv run streamlit run app.py
```

That's it! You're now ready to validate and analyze your CLIF data. 🚀
