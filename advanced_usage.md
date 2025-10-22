# Advanced Usage Guide

This guide covers advanced command-line options, workflows, and configurations for the CLIF Table One tool.

## Command Line Interface

### run_project.py - Complete Workflow Orchestration

The `run_project.py` script orchestrates the complete analysis pipeline.

#### Basic Commands

```bash
# Full analysis with sampling (recommended)
uv run python run_project.py --sample --no-summary --get-ecdf

# Full dataset analysis (45-90 minutes)
uv run python run_project.py --get-ecdf

# Skip automatic app launch
uv run python run_project.py --sample --no-summary --get-ecdf --no-launch-app

# Validation only
uv run python run_project.py --validate-only --sample --no-summary

# Table One only (skip validation)
uv run python run_project.py --tableone-only

# ECDF bins computation only
uv run python run_project.py --get-ecdf-only
uv run python run_project.py --get-ecdf-only --visualize

# Specific tables validation
uv run python run_project.py --tables patient adt hospitalization
```

#### All Command Options

```
Workflow Control:
  --validate-only          Only run validation step
  --tableone-only          Only run table one generation step
  --get-ecdf-only          Only run ECDF bins computation step
  --get-ecdf               Include ECDF in workflow
  --visualize              Generate HTML visualizations (for ECDF)
  --continue-on-error      Continue even if previous step fails
  --no-launch-app          Skip automatic Streamlit app launch

Validation Options:
  --tables TABLE [TABLE ...]
                          Specific tables to validate
  --sample                Use 1k ICU sample for faster analysis
  --no-summary            Skip summary statistics generation
  --verbose, -v           Enable verbose output

Configuration:
  --config CONFIG         Path to configuration file
```

### run_analysis.py - Validation and Summary Analysis

The `run_analysis.py` script provides granular control over validation and summary generation.

#### Basic Commands

```bash
# Single table with both validation and summary
uv run python run_analysis.py --patient --validate --summary

# Multiple tables with validation only
uv run python run_analysis.py --patient --hospitalization --validate

# All implemented tables
uv run python run_analysis.py --all --validate --summary

# Use 1k ICU sample for faster analysis
uv run python run_analysis.py --labs --validate --summary --sample

# Specify custom config file
uv run python run_analysis.py --config path/to/config.json --patient --validate

# Verbose output for debugging
uv run python run_analysis.py --patient --validate --summary --verbose

# Quiet mode (minimal output)
uv run python run_analysis.py --all --validate --summary --quiet
```

#### Table Selection Options

```
--patient                    Patient table
--hospitalization           Hospitalization table
--adt                       ADT table
--code_status               Code status table
--crrt_therapy              CRRT therapy table
--ecmo_mcs                  ECMO/MCS table
--hospital_diagnosis        Hospital diagnosis table
--labs                      Labs table
--medication_admin_continuous    Continuous medications
--medication_admin_intermittent  Intermittent medications
--microbiology_culture      Microbiology culture table
--microbiology_nonculture   Microbiology non-culture table
--microbiology_susceptibility    Susceptibility table
--patient_assessments       Patient assessments table
--patient_procedures        Patient procedures table
--position                  Position table
--respiratory_support       Respiratory support table
--vitals                    Vitals table
--all                       All implemented tables
```

#### Operations

```
--validate                  Run validation using clifpy
--summary                   Generate summary statistics
```

#### Output Control

```
--verbose, -v               Enable verbose output
--quiet, -q                 Minimize output (only errors and final summary)
--no-pdf                    Disable PDF report generation (JSON only)
--sample                    Use 1k ICU sample for faster analysis
```

#### Exit Codes

- `0` - Success
- `1` - All tables failed
- `2` - Partial success (some tables failed)
- `130` - Interrupted by user (Ctrl+C)

## Advanced Workflows

### Performance Optimization with Sampling

The `--sample` flag creates or uses a 1k patient ICU sample for faster processing:

```bash
# Create sample and run validation
uv run python run_project.py --sample --validate-only

# Sample behavior:
# 1. First run creates sample from ADT table
# 2. Sample saved to output/final/sample_1k_icu_hospitalizations.csv
# 3. Subsequent runs reuse existing sample
# 4. Core tables (patient, hospitalization, ADT) always use full data
# 5. Other tables filter to sample hospitalization IDs
```

Benefits:
- Reduces runtime from 30-60 min to 5-10 min for all tables
- Maintains validation accuracy for data quality checks
- Ideal for iterative development and testing

### ECDF Bins Configuration

ECDF (Empirical Cumulative Distribution Function) bins are used for visualizations in the EDA app.

#### Required Configuration Files

1. **Outlier Configuration** (`get-ecdf_data/ecdf_config/outlier_config.yaml`):
```yaml
labs:
  albumin_g_dl:
    lower: 0.5
    upper: 10
  bicarbonate_meq_l:
    lower: 5
    upper: 50
  # ... more lab configurations

vitals:
  heart_rate_bpm:
    lower: 20
    upper: 250
  # ... more vital configurations
```

2. **Binning Configuration** (`get-ecdf_data/ecdf_config/lab_vital_config.yaml`):
```yaml
labs:
  albumin_g_dl:
    bins: [0, 2.0, 2.5, 3.0, 3.5, 4.0, 100]
    labels: ["<2.0", "2.0-2.5", "2.5-3.0", "3.0-3.5", "3.5-4.0", "≥4.0"]
  # ... more lab configurations

vitals:
  heart_rate_bpm:
    bins: [0, 60, 100, 120, 150, 300]
    labels: ["<60", "60-100", "100-120", "120-150", "≥150"]
  # ... more vital configurations
```

## Configuration Details

### Main Configuration (config/config.json)

```json
{
    "site_name": "Your Hospital Name",
    "site_id": "YOUR_ID",
    "tables_path": "/path/to/clif/data",
    "filetype": "parquet",
    "timezone": "America/Chicago",
    "output_dir": "output"  // Optional, defaults to "output"
}
```

### Table One Configuration

The Table One generation uses several internal configurations:

- **Cohort Definition**: ICU stays ≥24 hours
- **MCIDE Collection**: Automated collection of clinically important data elements
- **Medication Analysis**: Vasoactives, sedatives, paralytics with dose conversions
- **SOFA Scoring**: Automated calculation with missing data handling
- **CCI Calculation**: Charlson Comorbidity Index from ICD codes

### Memory Optimization

The Table One generation includes memory optimization features:

1. **Chunked Processing**: Large tables processed in chunks
2. **Selective Loading**: Only required columns loaded for analysis
3. **Weight Data Pre-loading**: Optimized medication dose conversion
4. **Garbage Collection**: Aggressive memory cleanup between steps

## Troubleshooting

### Common Issues

#### Out of Memory Errors
```bash
# Use sampling for initial runs
uv run python run_project.py --sample --no-summary

# Increase system swap if needed
# Monitor with: watch -n 1 free -h
```

#### Validation Failures
```bash
# Check specific table details
uv run python run_analysis.py --patient --validate --verbose

# Review validation report
open output/final/reports/patient_validation_report.pdf
```

#### ECDF Generation Errors
```bash
# Check configuration files
cat get-ecdf_data/ecdf_config/outlier_config.yaml
cat get-ecdf_data/ecdf_config/lab_vital_config.yaml

# Review unit mismatches
cat output/final/unit_mismatches.log
```

### Log Files

- **Validation Logs**: `output/final/results/{table}_summary_validation.json`
- **Table One Execution**: `output/final/tableone/execution_report.txt`
- **ECDF Processing**: `output/final/unit_mismatches.log`


## Additional Resources

- [TABLEONE_VIEWER_GUIDE.md](TABLEONE_VIEWER_GUIDE.md) - Detailed guide for Table One results viewer
- [CLIF Documentation](https://clif-icu.com) - CLIF consortium documentation
- [clifpy Documentation](https://github.com/Common-Longitudinal-ICU-data-Format/clifpy) - CLIF validation library