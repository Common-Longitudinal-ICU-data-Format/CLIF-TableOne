# CLIF 2.1 Validation & Summarization Tool

A comprehensive tool for validating and analyzing CLIF 2.1 data tables using clifpy. Available as both an interactive web application and command-line interface.

**🚀 New User?** Check out the [Quick Start Guide](QUICKSTART.md) to get running in 3 steps!

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
- **Devices**: CRRT Therapy, ECMO/MCS

## Installation

**Prerequisites:**
- Python 3.8+
- UV package manager ([install instructions](https://docs.astral.sh/uv/))
- CLIF 2.1 data in parquet or CSV format

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
    "timezone": "America/Chicago"
}
```

## Complete Workflow Runner (Recommended)

The automated workflow runner orchestrates the complete analysis pipeline:

### Basic Usage

```bash
# Complete workflow with 1k sample (recommended for testing)
python run_project.py --sample

# Full dataset analysis
python run_project.py

# Validation only
python run_project.py --validate-only --sample

# Table One only (skip validation)
python run_project.py --tableone-only

# Specific tables
python run_project.py --tables patient adt hospitalization

# Continue despite validation warnings
python run_project.py --sample --continue-on-error

# Skip automatic app launch
python run_project.py --sample --no-launch-app
```

### Workflow Steps

1. **CLIF Validation**
   - Validates all 18 CLIF tables using clifpy
   - Generates PDF validation reports
   - Creates summary statistics
   - Optional: Uses 1k ICU sample for faster processing

2. **Table One Generation**
   - Generates comprehensive cohort analysis
   - Creates CONSORT diagrams and visualizations
   - Memory-optimized for large datasets
   - Produces final CSV tables and reports

3. **Automatic App Launch**
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
└── tableone/            # Table One outputs
    ├── table_one_overall.csv
    ├── table_one_by_year.csv
    ├── consort_flow_diagram.png
    ├── execution_report.txt
    └── ...
```

### Performance Guide

| Command | Dataset | Time | Use Case |
|---------|---------|------|----------|
| `--validate-only --sample` | Sample | ~5 min | Quick check |
| `--sample` | Sample | ~10-15 min | Development |
| `--validate-only` | Full | ~20-30 min | Pre-flight check |
| (no flags) | Full | ~45-90 min | Production |

### All Options

```bash
Workflow Control:
  --validate-only          Only run validation step
  --tableone-only          Only run table one generation step
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

## Web Application

### Launch

```bash
uv run streamlit run app.py
```

The app will open at `http://localhost:8501`

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

### Automation

**Cron Job (Daily Validation):**
```bash
# Run at 2 AM daily with 1k sample
0 2 * * * cd /path/to/CLIF-TableOne && uv run python run_analysis.py --all --validate --sample --quiet >> logs/analysis.log 2>&1
```

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

### Use Cases

- Site-specific categories (IRB-approved custom categories)
- Known data patterns (documented site-specific characteristics)
- Progressive review (mark errors as pending until clinical review)
- Audit trail (track all validation decisions with timestamps)

See [FEEDBACK_SYSTEM.md](FEEDBACK_SYSTEM.md) for technical documentation.

## Requirements

**System:**
- Python 3.8+
- UV (package manager)

**Dependencies** (automatically managed by UV):
- streamlit - Web interface
- pandas - Data manipulation
- clifpy - CLIF data validation
- plotly - Interactive visualizations
- reportlab - PDF generation
- duckdb - SQL queries
- pytz - Timezone handling

**Development Dependencies:**
- jupyter - Notebooks
- matplotlib - Static plots
- seaborn - Statistical visualizations

All dependencies are automatically installed via `uv sync`.

## Troubleshooting

### Configuration Issues

**"Configuration file not found"**
- Ensure `config/config.json` exists
- Check path is relative to project root

**"Data directory not found"**
- Verify `tables_path` in config.json points to correct location
- Check that CLIF parquet/csv files exist in that directory

**"Table not found"**
- Ensure file names match expected format:
  - `patient.parquet` or `clif_patient.parquet`
  - `hospitalization.parquet` or `clif_hospitalization.parquet`

### Performance Issues

**"Memory Error"**
- Use `--sample` flag for faster analysis with smaller dataset
- Close other applications to free up RAM
- Use `--validate-only` to skip memory-intensive table one generation

## Documentation

- **Quick Start**: [QUICKSTART.md](QUICKSTART.md)
- **Table One Generation**: [code/README_TABLE_ONE.md](code/README_TABLE_ONE.md)
- **Table One Viewer**: [TABLEONE_VIEWER_GUIDE.md](TABLEONE_VIEWER_GUIDE.md)
- **Feedback System**: [FEEDBACK_SYSTEM.md](FEEDBACK_SYSTEM.md)
- **Command Help**: `python run_project.py --help`
- **Issues**: [GitHub Issues](https://github.com/Common-Longitudinal-ICU-data-Format/CLIF-TableOne/issues)

## License

See CLIF consortium documentation for licensing information.
