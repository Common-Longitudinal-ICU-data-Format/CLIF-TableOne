# Advanced Usage Guide

This guide covers advanced command-line options, workflows, and configurations for the CLIF Table One tool.

## Command Line Interface

### run_project.py - Complete Workflow Orchestration

The `run_project.py` script orchestrates the complete analysis pipeline.

#### Basic Commands

```bash
# Full workflow: validation + CI + ward + ECDF (recommended)
uv run python run_project.py --no-summary

# Skip automatic web app launch
uv run python run_project.py --no-summary --no-launch-app

# CI table one only (fastest, no validation/ward/ECDF)
uv run python run_project.py --tableone-only --no-ward --no-ecdf

# Validation only
uv run python run_project.py --validate-only --no-summary

# Skip ward (CI + ECDF only)
uv run python run_project.py --no-summary --no-ward

# Skip ECDF (CI + ward only, saves memory)
uv run python run_project.py --no-summary --no-ecdf

# Ward table one only (runs in an isolated subprocess)
uv run python run_project.py --ward-only

# ECDF bins computation only
uv run python run_project.py --get-ecdf-only
uv run python run_project.py --get-ecdf-only --visualize

# Specific tables validation
uv run python run_project.py --tables patient adt hospitalization

# Force rebuild filtered CLIF table cache
uv run python run_project.py --no-summary --force-refresh
```

#### All Command Options

```
Workflow Control:
  --validate-only          Only run validation step
  --tableone-only          Only run critical-illness + ward table one (no validation/ECDF)
  --ward-only              Only run ward table one generation step (isolated subprocess)
  --no-ward                Skip ward table one generation (default: ON)
  --get-ecdf-only          Only run ECDF bins computation step
  --no-ecdf                Skip ECDF generation (default: ON)
  --visualize              Generate interactive HTML distribution viewers (for ECDF)
  --force-refresh          Bypass filtered-CLIF-table cache and rebuild from raw source
  --continue-on-error      Continue even if previous step fails
  --no-launch-app          Skip automatic web app launch

Validation Options:
  --tables TABLE [TABLE ...]
                          Specific tables to validate
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

# Specify custom config file
uv run python run_analysis.py --config path/to/config.json --patient --validate

# Override output directory
uv run python run_analysis.py --all --validate --output-dir custom/output

# Verbose output for debugging
uv run python run_analysis.py --patient --validate --summary --verbose

# Quiet mode (minimal output)
uv run python run_analysis.py --all --validate --summary --quiet
```

#### Table Selection Options

```
--patient                        Patient table
--hospitalization               Hospitalization table
--adt                           ADT table
--code_status                   Code status table
--crrt_therapy                  CRRT therapy table
--hospital_diagnosis            Hospital diagnosis table
--labs                          Labs table
--medication_admin_continuous   Continuous medications
--medication_admin_intermittent Intermittent medications
--microbiology_culture          Microbiology culture table
--microbiology_susceptibility   Susceptibility table
--patient_assessments           Patient assessments table
--patient_procedures            Patient procedures table
--position                      Position table
--respiratory_support           Respiratory support table
--vitals                        Vitals table
--all                           All implemented tables
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
--output-dir DIR            Override output directory from config
```

#### Exit Codes

- `0` - Success
- `1` - All tables failed
- `2` - Partial success (some tables failed)
- `130` - Interrupted by user (Ctrl+C)

### Specialized Runners

For targeted reruns of a single pipeline stage:

| Script | Purpose |
|---|---|
| `run_tableone.py` | Critical-illness Table One only |
| `run_tableone_ward.py` | Ward Table One only (isolated subprocess) |
| `run_tableone_all.py` | Both critical-illness and ward Table Ones |
| `run_ecdf.py` | ECDF bins computation |
| `run_sofa.py` | SOFA scoring |

Each writes to the same `output/intermediate/` and `output/final/` tree as `run_project.py`. Use them when iterating on one stage after a full pipeline run.

## Advanced Workflows

### ECDF Bins Configuration

ECDF (Empirical Cumulative Distribution Function) bins are used for visualizations in the web app.

#### Required Configuration Files

1. **Outlier Configuration** (`modules/ecdf/config/outlier_config.yaml`):
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

2. **Binning Configuration** (`modules/ecdf/config/lab_vital_config.yaml`):
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
    "output_dir": "output"
}
```

### Cohort Definitions

The critical-illness and ward cohorts, plus every stratum (`icu`, `advanced_resp`, `nippv_hfnc`, `vaso`, `deaths`, `no_imv`, etc.), are defined in [README.md §8 — Cohort Definitions](README.md#8-cohort-definitions). The single source of truth for the inclusion flags is `modules/strata.py:24-41`.

### Memory Optimization

The Table One generation includes memory optimization features:

1. **Chunked Processing**: Large tables processed in chunks
2. **Selective Loading**: Only required columns loaded for analysis
3. **Weight Data Pre-loading**: Optimized medication dose conversion
4. **Garbage Collection**: Aggressive memory cleanup between steps
5. **Isolated ward subprocess**: The ward cohort runs in a separate process (default ON, skip with `--no-ward`) so peak RAM equals the larger of the two cohorts, not the sum

## Troubleshooting

### Common Issues

#### Out of Memory Errors
```bash
# Run stages separately instead of a single full pipeline
uv run python run_project.py --validate-only --no-summary
uv run python run_project.py --tableone-only
uv run python run_project.py --get-ecdf-only

# Ward cohort is already isolated via subprocess under --ward / --ward-only
# Monitor with: watch -n 1 free -h   (Linux)  |  vm_stat 1   (macOS)
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
cat modules/ecdf/config/outlier_config.yaml
cat modules/ecdf/config/lab_vital_config.yaml

# Review schema-vs-data unit mismatches (site-side data issues)
cat output/final/meta/unit_mismatches.log

# Review ECDF coverage gaps (schema-valid cats the bin config doesn't know)
cat output/final/meta/ecdf_coverage_gaps.log
```

### Log Files

- **Validation Logs**: `output/final/results/{table}_summary_validation.json`
- **Table One Execution**: `output/final/tableone/execution_report.txt`
- **ECDF Schema Mismatches**: `output/final/meta/unit_mismatches.log`
- **ECDF Coverage Gaps**: `output/final/meta/ecdf_coverage_gaps.log`


## Additional Resources

- [README.md](README.md) - Project overview, quickstart, and cohort definitions
- [OUTPUT_REFERENCE.md](OUTPUT_REFERENCE.md) - Per-file detail for every output artifact
- [CLIF Documentation](https://clif-icu.com) - CLIF consortium documentation
- [clifpy Documentation](https://github.com/Common-Longitudinal-ICU-data-Format/clifpy) - CLIF validation library
