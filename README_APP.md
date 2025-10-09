# CLIF 2.1 Table One Analysis System

## Overview
This is a modular Streamlit application for validating and analyzing CLIF 2.1 data tables using the clifpy package. The system provides comprehensive validation and summary statistics for each CLIF table.

## Features
- ✅ Config-based setup - users update their config file with site info and data paths
- 📊 Dual-tab interface: Validation Results and Summary Statistics
- 🏥 CLIF logo display in the header
- 📋 Left sidebar for table selection and options
- 🔍 Comprehensive validation using clifpy
- 📈 Missingness analysis and distribution statistics
- 💾 Export results to JSON files

## Setup

### 1. Install Dependencies
```bash
pip install streamlit pandas plotly clifpy scipy
```

### 2. Configure Your Site
Update the `config/config.json` file with your site information:
```json
{
    "site_name": "Your Hospital Name",
    "site_id": "YOUR_ID",
    "tables_path": "/path/to/your/clif/tables",
    "filetype": "parquet",
    "timezone": "UTC",
    "output_dir": "output",
    "analysis_settings": {
        "calculate_sofa": true,
        "outlier_detection": false,
        "missingness_threshold": 0.95
    }
}
```

### 3. Add CLIF Logo (Optional)
Place your CLIF logo image at `assets/clif_logo.png`

### 4. Run the Application
```bash
streamlit run app.py
```

## Project Structure
```
CLIF-TableOne/
├── app.py                          # Main Streamlit application
├── assets/
│   └── clif_logo.png              # CLIF logo (optional)
├── modules/
│   ├── tables/
│   │   ├── base_table_analyzer.py # Base class for analyzers
│   │   └── patient_analysis.py    # Patient table analyzer
│   └── utils/
│       ├── validation.py          # Validation utilities
│       ├── missingness.py          # Missingness analysis
│       └── distributions.py        # Distribution analysis
├── config/
│   └── config.json                # Site configuration file
└── output/
    ├── intermediate/               # Patient-level data
    └── final/                      # Aggregated summaries
```

## Usage

1. **Launch the app**: `streamlit run app.py`

2. **Configure settings**:
   - Enter path to your config file (default: `config/config.json`)
   - App will load your site information automatically

3. **Select table**:
   - Choose a table from the left sidebar
   - Currently, Patient table is fully implemented
   - Other tables coming soon

4. **Run analysis**:
   - Check "Run Validation" for clifpy validation
   - Click "Run Analysis" button

5. **View results**:
   - **Validation Results Tab**: Shows clifpy validation status and errors
   - **Summary Statistics Tab**: Shows missingness, distributions, and demographics

6. **Save results**:
   - Click "Save Validation Results" or "Save Summary Statistics"
   - Files saved to `output/final/` directory

## Current Status

### Implemented
- ✅ Patient table analyzer with full validation and statistics
- ✅ Config-based setup for site information
- ✅ Dual-tab interface (Validation & Summary)
- ✅ CLIF logo support
- ✅ Comprehensive missingness analysis
- ✅ Distribution analysis for categorical and numeric data
- ✅ Export functionality

### Coming Soon
- 🚧 Remaining 17 CLIF tables (hospitalization, ADT, vitals, labs, etc.)
- 🚧 SOFA score calculation using clifpy
- 🚧 Cohort identification (CLIF 2.1 criteria)
- 🚧 Cross-table analysis
- 🚧 Table One generation
- 🚧 ECDF visualizations
- 🚧 PDF export

## Validation Status Meanings
- **Complete** ✅: Table passes all validation checks
- **Partial** ⚠️: Table has minor issues (e.g., invalid categories)
- **Incomplete** ❌: Table has critical issues (e.g., missing required columns)

## Troubleshooting

### clifpy not installed
```bash
pip install clifpy
```

### Data files not found
- Check that `tables_path` in config.json points to correct directory
- Ensure files match the configured `filetype` (parquet/csv)
- File names should match CLIF table names (e.g., `patient.parquet`)

### Config file errors
- Ensure config.json is valid JSON format
- All required fields must be present
- Use absolute paths for `tables_path`

## Next Steps
To add more tables, create new analyzer classes in `modules/tables/` following the pattern in `patient_analysis.py`. Each analyzer should:
1. Inherit from `BaseTableAnalyzer`
2. Implement `load_table()` using clifpy
3. Implement `get_data_info()` for table-specific info
4. Implement `analyze_distributions()` for table-specific distributions

## Contact
For issues or questions about the CLIF schema, refer to the CLIF consortium documentation.