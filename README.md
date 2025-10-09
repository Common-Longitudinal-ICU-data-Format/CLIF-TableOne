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

### Web Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Command Line Interface

```bash
# Validate and summarize patient table
python run_analysis.py --patient --validate --summary

# Process multiple tables
python run_analysis.py --patient --hospitalization --adt --validate --summary

# Custom config file
python run_analysis.py --config path/to/config.json --patient --validate
```

## Configuration

Create or update `config/config.json`:

```json
{
    "site_name": "Your Hospital Name",
    "site_id": "YOUR_ID",
    "tables_path": "/path/to/clif/data",
    "filetype": "parquet",
    "timezone": "America/Chicago",
    "output_dir": "output"
}
```

## Web Application Guide

### 1. Initial Setup
1. Configure `config/config.json` with your site information
2. Ensure CLIF data files are in the specified `tables_path`
3. Launch app with `streamlit run app.py`

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
python run_analysis.py --patient --validate --summary

# Multiple specific tables
python run_analysis.py --patient --hospitalization --validate --summary

# All implemented tables
python run_analysis.py --all --validate --summary
```

### Advanced Options

```bash
# Custom configuration
python run_analysis.py --config custom/config.json --patient --validate

# Verbose output
python run_analysis.py --patient --validate --summary --verbose

# Quiet mode (errors only)
python run_analysis.py --all --validate --summary --quiet

# Override output directory
python run_analysis.py --patient --validate --output-dir custom/output
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
0 2 * * * cd /path/to/CLIF-TableOne && python run_analysis.py --all --validate --quiet >> logs/analysis.log 2>&1
```

**CI/CD Pipeline:**
```bash
#!/bin/bash
python run_analysis.py --all --validate --summary
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

## Project Structure

```
CLIF-TableOne/
├── app.py                          # Streamlit web application
├── run_analysis.py                 # Command-line interface
├── modules/
│   ├── tables/                     # Table-specific analyzers
│   │   ├── base_table_analyzer.py  # Base analyzer class
│   │   ├── patient_analysis.py     # Patient table
│   │   ├── hospitalization_analysis.py
│   │   └── adt_analysis.py         # ADT table
│   ├── utils/                      # Shared utilities
│   │   ├── validation.py           # Error classification
│   │   ├── missingness.py          # Missingness analysis
│   │   ├── feedback.py             # User feedback system
│   │   └── cache_manager.py        # State persistence
│   └── cli/                        # CLI-specific modules
│       └── pdf_generator.py        # PDF report generation
├── config/
│   └── config.json                 # Site configuration
└── output/
    └── final/                      # Analysis outputs
```

## Requirements

**Core:**
- Python 3.8+
- streamlit
- pandas
- numpy
- clifpy
- plotly

**Optional:**
- reportlab (for PDF generation)
- duckdb (for efficient year distributions)

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Troubleshooting

### Data Not Found
- Verify `tables_path` in config.json points to correct directory
- Ensure files match configured `filetype` (parquet/csv)
- Check file naming: `patient.parquet` or `clif_patient.parquet` (both work)

### clifpy Errors
```bash
pip install --upgrade clifpy
```

### Missing Dependencies
```bash
pip install -r requirements.txt
```

### PDF Generation Fails
- Install reportlab: `pip install reportlab`
- CLI will fall back to text reports (.txt) if reportlab unavailable

## Comparison: Web App vs CLI

| Feature | Web App | CLI |
|---------|---------|-----|
| **Interface** | Interactive GUI | Terminal |
| **Best For** | Exploration, review | Automation, batch |
| **Visualization** | Charts & tables | Text output |
| **Feedback System** | Interactive review | JSON output |
| **Automation** | Manual only | Scripts & cron |
| **Remote Access** | Port forwarding | SSH-friendly |

## Contributing

To add support for a new CLIF table:

1. Create analyzer in `modules/tables/{table}_analysis.py`
2. Inherit from `BaseTableAnalyzer`
3. Implement required methods:
   - `load_table()` - Load using clifpy
   - `get_data_info()` - Table-specific metrics
   - `analyze_distributions()` - Distribution analysis
4. Add to `TABLE_ANALYZERS` dict in `app.py` and `run_analysis.py`

See existing analyzers for examples.

## License

See CLIF consortium documentation for licensing information.

---

## Legacy: Table One Generation

The original Table One generation feature for ICU cohort summarization is available in the `code/` directory. This legacy feature generates yearly and overall summaries for ICU encounters.

### Legacy Requirements
See legacy documentation in `code/` for the original Table One generation requirements and usage.

**Note**: The main focus of this repository is now the CLIF 2.1 Validation & Summarization Tool described above.
