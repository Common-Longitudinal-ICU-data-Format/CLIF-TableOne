# CLIF 2.1 Validation & Summarization Tool

A comprehensive tool for validating and analyzing CLIF 2.1 data tables using clifpy. 

## Features

- ✅ **CLIF 2.1 Validation** - Full schema and data quality validation using clifpy
- 📊 **Summary Statistics** - Data distributions, missingness analysis, and quality metrics
- 💾 **User Feedback System** - Accept/reject validation errors with site-specific justifications
- 🔄 **Persistent Caching** - Results persist across sessions for efficient workflow
- 📈 **Interactive Visualizations** - Year distributions, missingness charts, and quality metrics
- 📄 **PDF Reports** - Automated generation of validation and summary reports
- 🎯 **Multiple Interfaces** - Web app for exploration, CLI for automation
- 📊 **Table One Viewer** - Interactive display of cohort analysis with CONSORT diagrams, demographics, medications, ventilation, and outcomes

## Supported Tables

All 18 CLIF 2.1 tables are supported:
- **Core**: Patient, Hospitalization, ADT
- **Clinical**: Code Status, Labs, Vitals, Patient Assessments, Patient Procedures, Hospital Diagnosis
- **Respiratory**: Respiratory Support, Position
- **Medications**: Medication Admin (Continuous & Intermittent)
- **Microbiology**: Culture, Non-culture, Susceptibility
- **Devices**: CRRT Therapy, ECMO_MCS

## Installation

**Prerequisites:**
- Python 3.8+
- UV package manager ([install instructions](https://docs.astral.sh/uv/))
- CLIF 2.1 data in parquet 

**Setup:**
```bash
cd CLIF-TableOne
uv sync
```

## Configuration

Create or update `config/config.json`:

```json
{
    "site_name": "Your Hospital Name",
    "tables_path": "/path/to/clif/data",
    "filetype": "parquet",
    "timezone": "Your timezone as America/Chicago etc.."
}
```

## Quick Start - Run Complete Workflow

**Recommended command to run the complete analysis pipeline:**

```bash
uv run python run_project.py --sample --get-ecdf
```

This single command runs:
1. ✅ **Validation** - Validates all 18 CLIF tables (with 1k ICU sample)
2. 📊 **Table One** - Generates cohort analysis with CONSORT diagrams
3. 📈 **ECDF Bins** - Computes distributions for labs/vitals/respiratory data
4. 🚀 **App Launch** - Automatically opens the Streamlit web interface

**Time:** ~15-20 minutes
**Output:** All validation reports, Table One CSVs/visualizations, and ECDF data for EDA apps

---

## Advanced Usage

### Other Workflow Options

```bash
# Full dataset (no sampling, takes 45-90 min)
uv run python run_project.py --get-ecdf

# Skip automatic app launch
uv run python run_project.py --sample --get-ecdf --no-launch-app

# Validation only (quick data quality check)
uv run python run_project.py --validate-only --sample

# Table One only (skip validation)
uv run python run_project.py --tableone-only

# ECDF only (for EDA app setup)
uv run python run_project.py --get-ecdf-only
uv run python run_project.py --get-ecdf-only --visualize

# Specific tables
uv run python run_project.py --tables patient adt hospitalization
```

### What is ECDF?

The get-ecdf feature generates ECDF (Empirical Cumulative Distribution Function) and quantile bins for labs/vitals/respiratory data during ICU stays. This data is used by exploratory data analysis (EDA) applications for visualization and statistical analysis.

**When to use:**
- Setting up an EDA app that needs pre-computed distributions
- Generating statistical baselines for data quality monitoring
- Creating reference distributions for visualization tools

**Requirements:**
- `config/config.json` - Main CLIF configuration
- `get-ecdfd_data/ecdf_config/outlier_config.yaml` - Outlier filtering configuration
- `get-ecdfd_data/ecdf_config/lab_vital_config.yaml` - Binning configuration for labs/vitals
- `get-ecdfd_data/utils.py` - Binning utility functions

**Output:**
```
output/final/
├── configs/                # Configuration files
├── ecdf/                   # ECDF parquet files
│   ├── labs/              # One file per (category, unit)
│   ├── vitals/            # One file per category
│   └── respiratory_support/
├── bins/                   # Bin parquet files
│   ├── labs/
│   ├── vitals/
│   └── respiratory_support/
├── plots/                  # HTML visualizations (with --visualize)
└── unit_mismatches.log     # Processing log
```

## Web Application

If you just want to launch the app:

```bash
uv run streamlit run app.py
```

The app will open at `http://localhost:8501`


### Workflow Steps

1. **CLIF Validation** (Optional)
   - Validates all 18 CLIF tables using clifpy
   - Generates PDF validation reports
   - Creates summary statistics
   - Optional: Uses 1k ICU sample for faster processing

2. **Table One Generation** (Optional)
   - Generates comprehensive cohort analysis
   - Creates CONSORT diagrams and visualizations
   - Memory-optimized for large datasets
   - Produces final CSV tables and reports

3. **Get ECDF ECDF Bins** (Optional)
   - Computes distributions for labs/vitals/respiratory data during ICU stays
   - Generates quantile bins for EDA visualization
   - Optional: Creates HTML plots with --visualize flag

4. **Automatic App Launch**
   - Launches Streamlit app after successful completion
   - 3-second countdown with skip option

### Output Structure

```
output/final/
├── reports/              # Validation PDFs
│   ├── patient_validation_report.pdf
│   ├── combined_validation_report.pdf
│   └── ...
├── results/              # Validation CSVs
│   ├── patient_summary.csv
│   └── ...
├── tableone/            # Table One outputs
│   ├── table_one_overall.csv
│   ├── table_one_by_year.csv
│   ├── consort_flow_diagram.png
│   ├── execution_report.txt
│   └── ...
├── ecdf/                # ECDF distributions (if --get-ecdf)
│   ├── labs/
│   ├── vitals/
│   └── respiratory_support/
├── bins/                # Quantile bins (if --get-ecdf)
│   ├── labs/
│   ├── vitals/
│   └── respiratory_support/
├── plots/               # HTML visualizations (if --visualize)
├── configs/             # Get ECDF configs (if --get-ecdf)
└── unit_mismatches.log  # Get ECDF log (if --get-ecdf)
```

### All Command Options

```bash
Workflow Control:
  --validate-only          Only run validation step
  --tableone-only          Only run table one generation step
  --get-ecdf-only        Only run get-ecdf ECDF bins step
  --get-ecdf             Include get-ecdf in workflow
  --visualize              Generate HTML visualizations (for get-ecdf)
  --continue-on-error      Continue even if previous step fails
  --no-launch-app          Skip automatic Streamlit app launch

Validation Options:
  --tables TABLE [TABLE ...]
                          Specific tables to validate
  --sample                Use 1k ICU sample for faster analysis
  --verbose, -v           Enable verbose output

Configuration:
  --config CONFIG         Path to configuration file
```

### Workflow

1. **Initial Setup**
   - Configure `config/config.json` with site information
   - Ensure CLIF data files are in the specified `tables_path`

2. **Running Analysis**
   - Select table from sidebar
   - Choose analysis options (validation, summary)
   - Click "🚀 Run Analysis"

3. **Reviewing Validation Errors**
   - Enable "📋 Review Status-Affecting Errors" in Validation tab
   - For each error:
     - **Accepted** - Valid issue requiring attention
     - **Rejected** - Site-specific variation (provide justification)
     - **Pending** - Not yet reviewed
   - Click "💾 Save Feedback"
   - Status automatically adjusts based on feedback

4. **Cached Results**
   - Results persist in session until re-run
   - Sidebar shows status and timestamp for each table
   - "Clear All Cache" button resets all tables

5. **Table One Results Viewer**
   - Click "📊 Table One Results" in sidebar (appears after generation)
   - Explore tabs: Cohort, Demographics, Medications, IMV, SOFA & CCI, Hospice & Outcomes
   - See [TABLEONE_VIEWER_GUIDE.md](TABLEONE_VIEWER_GUIDE.md) for details

6. **Output Files**
   - `{table}_validation_report.pdf` - PDF validation report
   - `{table}_validation_response.json` - User feedback decisions
   - `{table}_summary_validation.json` - Raw validation results
   - `{table}_summary.csv` - Summary statistics table

## Command Line Interface

### Basic Usage

```bash
# Single table
uv run python run_analysis.py --patient --validate --summary

# Multiple tables
uv run python run_analysis.py --patient --hospitalization --validate --summary

# All tables
uv run python run_analysis.py --all --validate --summary

# With sample (recommended for large datasets)
uv run python run_analysis.py --all --validate --sample

# Custom configuration
uv run python run_analysis.py --config custom/config.json --patient --validate
```

### Options

**Table Selection:**
- `--patient`, `--hospitalization`, `--adt`, `--labs`, `--vitals`, etc.
- `--all` - Analyze all 18 tables

**Operations:**
- `--validate` - Run clifpy validation
- `--summary` - Generate summary statistics

**Configuration:**
- `--config PATH` - Config file path (default: `config/config.json`)
- `--output-dir PATH` - Override output directory

**Performance:**
- `--sample` - Use 1k ICU sample (recommended for `--all`)
  - Reduces runtime from 30-60 min to 5-10 min
  - Automatically uses existing sample or creates from ADT table
  - Core tables (patient, hospitalization, ADT) always use full data

**Output Control:**
- `--verbose` / `-v` - Detailed progress output
- `--quiet` / `-q` - Minimal output (errors only)
- `--no-pdf` - Skip PDF generation (JSON only)

### Exit Codes

- `0` - Success
- `1` - All tables failed
- `2` - Partial success
- `130` - Interrupted (Ctrl+C)


## Feedback System

The user feedback system allows sites to classify validation errors based on their specific context:

### How It Works

1. **Error Classification**: Each validation error can be:
   - **Accepted**: Legitimate issue requiring attention
   - **Rejected**: Site-specific variation (with justification)
   - **Pending**: Not yet reviewed

2. **Status Recalculation**:
   - Status becomes "complete" only when ALL errors are rejected
   - Accepted or pending errors keep original status
   - Provides audit trail of all decisions

3. **Persistence**: Feedback saved to `{table}_validation_response.json`
