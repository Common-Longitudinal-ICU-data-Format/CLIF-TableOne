# CLIF 2.1 Validation & Summarization Tool

A comprehensive tool for validating and analyzing CLIF 2.1 data tables using clifpy with Table One generation.

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
- CLIF 2.1 data in parquet format

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

## Quick Start - Complete Workflow

Run the complete analysis pipeline with sampling (recommended for first run):

```bash
# Works on Linux, MacOS, and Windows
uv run python run_project.py --sample --no-summary --get-ecdf
```

This command:
1. Validates all 18 CLIF tables using a 1k ICU sample
2. Collects MCIDE data
3. Generates Table One analysis
4. Computes ECDF bins for visualizations
5. Automatically launches the Streamlit app

**Time:** ~10-15 minutes with sampling, 45-90 minutes without

### Windows Users: Optional Batch File

For convenience, Windows users can also use the provided batch file:

```batch
run_project_windows.bat --sample --no-summary --get-ecdf
```

The batch file is optional - it just pre-sets UTF-8 encoding environment variables, but the Python script now handles UTF-8 automatically.

## Web Application

To launch the Streamlit app independently:

```bash
uv run streamlit run app.py
```

The app will open at `http://localhost:8501`

## Output Structure

```
output/final/
├── reports/              # Validation PDF reports
│   ├── patient_validation_report.pdf
│   ├── combined_validation_report.pdf
│   └── ...
├── results/              # Validation CSV summaries
│   ├── patient_summary.csv
│   └── ...
├── tableone/            # Table One outputs
│   ├── table_one_overall.csv
│   ├── table_one_by_year.csv
│   ├── consort_flow_diagram.png
│   ├── execution_report.txt
│   ├── mcide/          # MCIDE value counts
│   ├── summary_stats/  # MCIDE numerical summaries
│   └── plots/          # Medication analysis plots
├── ecdf/                # ECDF distributions
│   ├── labs/
│   ├── vitals/
│   └── respiratory_support/
├── bins/                # Quantile bins for visualization
│   ├── labs/
│   ├── vitals/
│   └── respiratory_support/
└── configs/             # ECDF configuration files
```

## Workflow

### 1. Setup
- Install prerequisites and dependencies using `uv sync`
- Configure `config/config.json` with your site information
- Ensure CLIF data files are in the specified `tables_path`

### 2. Configuration
- Verify paths and settings in config file
- Choose whether to use sampling for faster initial runs

### 3. Main Command
- Run `run_project.py` with appropriate flags: `uv run python run_project.py --sample --no-summary --get-ecdf`
- Monitor progress in the terminal
- Wait for automatic app launch or launch manually

### 4. Using the App
- **Validation Tab**: Review validation results for each table
- **mCIDE Tab**: View mCIDE and summary stats if generated. 
- **Table One Results**: Access comprehensive cohort analysis (appears after generation)

### 5. Reviewing Validation Errors

The feedback system allows classification of validation errors:

- **Review Errors**: Enable "📋 Review Status-Affecting Errors" in the Validation tab. Scroll to the bottom of the page. 
- **Classify Each Error**:
  - **Accepted**: Legitimate issue requiring attention
  - **Rejected**: Site-specific variation (provide justification)
  - **Pending**: Not yet reviewed
- **Save Feedback**: Click "💾 Save Feedback" to persist decisions
- **Status Updates**: Table status automatically adjusts based on feedback
  - Status becomes "complete" only when ALL errors are rejected
  - Accepted or pending errors maintain original status

Feedback is saved to `output/final/results/{table}_validation_response.json`.
To recompile the combined report after providing feedback, click the Regenerate reports button on the Home page. 

### 6. Table One Results

After Table One generation completes:

1. Click "📊 Table One Results" in the sidebar
2. Explore analysis across tabs:
   - **Cohort**: CONSORT diagram and cohort flow
   - **Demographics**: Patient characteristics
   - **Medications**: Medication usage analysis with visualizations
   - **IMV**: Invasive mechanical ventilation metrics
   - **SOFA & CCI**: Severity and comorbidity scores
   - **Hospice & Outcomes**: End-of-life care and outcomes

See [TABLEONE_VIEWER_GUIDE.md](TABLEONE_VIEWER_GUIDE.md) for detailed viewer documentation.

## Windows Notes

### UTF-8 Support
UTF-8 encoding is now built into the script. Emojis and Unicode characters should work automatically.

If you still encounter encoding issues with very old Windows configurations:
- Use the provided batch file: `run_project_windows.bat`
- Or set environment variable: `set PYTHONIOENCODING=utf-8`

## Advanced Usage

For detailed command-line options, additional workflows, and advanced configurations, see [advanced_usage.md](advanced_usage.md).

## Support

For issues or questions, please create an issue in the project repository.