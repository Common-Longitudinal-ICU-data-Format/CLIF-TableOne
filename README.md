# CLIF 2.1 Validation & Summarization Tool

A comprehensive tool for validating and analyzing CLIF 2.1 data tables using clifpy. Available as both an interactive web application and command-line interface.

## Features

- ✅ **CLIF 2.1 Validation** - Full schema and data quality validation using clifpy
- 📊 **Summary Statistics** - Data distributions, missingness analysis, and quality metrics
- 💾 **User Feedback System** - Accept/reject validation errors with site-specific justifications
- 🔄 **Persistent Caching** - Results persist across sessions for efficient workflow
- 📈 **Interactive Visualizations** - Year distributions, missingness charts, and quality metrics
- 📄 **PDF Reports** - Automated generation of validation and summary reports
- 🎯 **Multiple Interfaces** - Web app for exploration, CLI for automation

## Supported Tables

Currently implemented analyzers:
- **Patient** - Demographics, mortality, language categories
- **Hospitalization** - Admission patterns, discharge data, age distributions
- **ADT** - Location tracking, ICU identification, transfer analysis

## Quick Start

### Installation

This project uses [UV](https://docs.astral.sh/uv/) for fast, reliable dependency management.

**1. Install UV** (if not already installed):
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

**2. Set up the project**:
```bash
cd CLIF-TableOne

# Initialize UV and install dependencies
uv sync
```

That's it! UV automatically creates a virtual environment and installs all dependencies.

### Web Application

```bash
# Run the Streamlit app
uv run streamlit run app.py
```

The app will open at `http://localhost:8501`

**Optional: Custom port**
```bash
uv run streamlit run app.py --server.port 8501
```

### Command Line Interface

```bash
# Validate and summarize patient table
uv run python run_analysis.py --patient --validate --summary

# Process multiple tables
uv run python run_analysis.py --patient --hospitalization --adt --validate --summary

# Custom config file
uv run python run_analysis.py --config path/to/config.json --patient --validate
```

### Jupyter Notebooks

```bash
# Launch Jupyter Lab
uv run jupyter lab
```

## Configuration

Create or update `config/config.json`:

```json
{
    "site_name": "Your Hospital Name",
    "tables_path": "/path/to/clif/data",
    "filetype": "parquet",
    "timezone": "America/Chicago",
}
```

## Web Application Guide

### 1. Initial Setup
1. Install UV and run `uv sync` to set up dependencies
2. Configure `config/config.json` with your site information
3. Ensure CLIF data files are in the specified `tables_path`
4. Launch app with `uv run streamlit run app.py`

### 2. Running Analysis
1. Select table from sidebar
2. Choose analysis options (validation, summary)
3. Click "🚀 Run Analysis"
4. View results in Validation and Summary tabs

### 3. Reviewing Validation Errors
The feedback system allows you to classify errors based on site-specific context:

**In the Validation tab:**
1. Enable "📋 Review Status-Affecting Errors"
2. For each error, select:
   - **Pending** - Not yet reviewed (default)
   - **Accepted** - Valid issue that needs attention
   - **Rejected** - Site-specific, not an issue (provide reason)
3. Click "💾 Save Feedback"
4. Status automatically adjusts:
   - All errors rejected → Status becomes "complete"
   - Any errors accepted/pending → Status remains original

**Status-Affecting vs Informational Errors:**
- Status-affecting errors require your review and affect validation status
- Informational errors are for awareness only (e.g., extra columns, minor warnings)

### 4. Cached Results
- Analysis results persist in session until re-run
- Sidebar shows status and timestamp for each table
- Click "Re-analyze table" checkbox to force fresh analysis
- "Clear All Cache" button resets all tables

### 5. Output Files
All results saved to `output/final/`:
- `{table}_validation_report.pdf` - PDF validation report
- `{table}_validation_response.json` - User feedback decisions
- `{table}_summary_validation.json` - Raw validation results
- `{table}_summary.csv` - Summary statistics table

## Command Line Interface Guide

### Basic Commands

```bash
# Single table with validation and summary
uv run python run_analysis.py --patient --validate --summary

# Multiple specific tables
uv run python run_analysis.py --patient --hospitalization --validate --summary

# All implemented tables
uv run python run_analysis.py --all --validate --summary
```

### Advanced Options

```bash
# Custom configuration
uv run python run_analysis.py --config custom/config.json --patient --validate

# Verbose output
uv run python run_analysis.py --patient --validate --summary --verbose

# Quiet mode (errors only)
uv run python run_analysis.py --all --validate --summary --quiet

# Override output directory
uv run python run_analysis.py --patient --validate --output-dir custom/output
```

### CLI Options Reference

**Table Selection:**
- `--patient` - Analyze patient table
- `--hospitalization` - Analyze hospitalization table
- `--adt` - Analyze ADT table
- `--all` - Analyze all implemented tables

**Operations:**
- `--validate` - Run clifpy validation
- `--summary` - Generate summary statistics

**Configuration:**
- `--config PATH` - Config file path (default: `config/config.json`)
- `--output-dir PATH` - Override output directory

**Output Control:**
- `--verbose` / `-v` - Detailed progress output
- `--quiet` / `-q` - Minimal output (errors only)
- `--no-pdf` - Skip PDF generation (JSON only)

### Exit Codes
- `0` - Success
- `1` - All tables failed
- `2` - Partial success
- `130` - Interrupted (Ctrl+C)

### Automation Examples

**Cron Job (Daily Validation):**
```bash
# Run at 2 AM daily
0 2 * * * cd /path/to/CLIF-TableOne && uv run python run_analysis.py --all --validate --quiet >> logs/analysis.log 2>&1
```

**CI/CD Pipeline:**
```bash
#!/bin/bash
uv run python run_analysis.py --all --validate --summary
if [ $? -ne 0 ]; then
    echo "Validation failed"
    exit 1
fi
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

- **Site-Specific Categories**: Reject errors for IRB-approved custom categories
- **Known Data Patterns**: Document legitimate site-specific data characteristics
- **Progressive Review**: Mark errors as pending until clinical review
- **Audit Trail**: Track all validation decisions with timestamps and reasons

See [FEEDBACK_SYSTEM.md](FEEDBACK_SYSTEM.md) for detailed technical documentation.

## Requirements

**System:**
- Python 3.8+
- UV (package manager) - See installation section above

**Dependencies** (automatically managed by UV):
- streamlit - Web interface
- pandas - Data manipulation
- numpy - Numerical operations
- clifpy - CLIF data validation
- plotly - Interactive visualizations
- reportlab - PDF generation
- duckdb - SQL queries
- pytz - Timezone handling

**Development Dependencies:**
- jupyter - Notebooks
- matplotlib - Static plots
- seaborn - Statistical visualizations
- tableone - Table generation
- tqdm - Progress bars

All dependencies are automatically installed via `uv sync`. No manual installation needed!

## Why UV?

This project uses UV instead of traditional pip/virtualenv for several benefits:

- ⚡ **10-100x faster** than pip at installing packages
- 🔒 **Reproducible builds** with `uv.lock` lockfile
- 🎯 **Automatic venv management** - no need to activate/deactivate
- 📦 **Clean dependencies** - only direct deps in `pyproject.toml`
- 🚀 **Better caching** - faster subsequent installs

### Common UV Commands

```bash
# Install/sync dependencies
uv sync

# Add a new dependency
uv add package-name

# Add a dev dependency
uv add --group dev package-name

# Update dependencies
uv lock --upgrade

# Run any command with project dependencies
uv run <command>

# See installed packages
uv pip list
```

## Troubleshooting

### Data Not Found
- Verify `tables_path` in config.json points to correct directory
- Ensure files match configured `filetype` (parquet/csv)
- Check file naming: `patient.parquet` or `clif_patient.parquet` (both work)

## License

See CLIF consortium documentation for licensing information.

---

