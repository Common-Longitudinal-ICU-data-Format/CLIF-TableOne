# Table One Results Viewer - User Guide

The Streamlit app now includes a comprehensive Table One Results Viewer that displays all generated cohort analysis results in an organized, interactive interface.

## Accessing the Viewer

### Availability
The **📊 Table One Results** button appears in the left navigation bar (below the Home button) **only when** Table One results are available in `output/final/tableone/`.

### Requirements
To generate Table One results, run:
```bash
# Generate Table One (after validation)
python run_project.py --sample

# OR run table one only
python run_project.py --tableone-only
```

## Navigation Structure

The Table One viewer is organized into **6 tabs**:

### 1. 🏥 Cohort Tab
Displays cohort identification and flow diagrams:

**Visualizations:**
- **CONSORT Flow Diagram** - Patient inclusion/exclusion flow
- **UpSet Plot** - Cohort intersection analysis showing overlaps between cohort definitions
- **Venn Diagram** - 4-way Venn diagram of cohort overlaps
- **Code Status Distribution** - Stacked bar chart with missingness analysis
- **Sankey Diagrams** - Patient flow visualizations for:
  - ICU patients
  - Other patients
  - High O2 support patients
  - Vasopressor support patients

### 2. 👥 Demographics Tab
Displays demographic data and Table One results:

**Key Metrics Summary (10 metrics in 2 rows):**

*Row 1 - Core Demographics:*
- 👥 **Hospitalizations** - Total number of encounters in cohort
- 🎂 **Median Age** - Median age of patients
- 👩 **Female** - Percentage of female patients
- 💔 **Mortality** - Overall mortality rate
- ⏱️ **ICU LOS** - ICU length of stay (days)

*Row 2 - Clinical Interventions & Outcomes:*
- 🫁 **Mech Ventilation** - Mechanical ventilation rate
- 💉 **Vasopressors** - Vasopressor usage rate
- 🩺 **CRRT** - Continuous renal replacement therapy rate
- 🏥 **Hospital LOS** - Hospital length of stay (days)
- 🕊️ **Hospice Discharge** - Discharge to hospice rate

**Content:**
- **Table One by Year** - Cohort characteristics stratified by admission year
  - Interactive dataframe display
  - Downloadable CSV
- **Table One Overall** - Overall cohort summary statistics
  - Interactive dataframe display
  - Downloadable CSV
- **Data Summary** - Additional metrics:
  - Total cohort combinations
  - Total hospitalizations
  - Number of variables collected

### 3. 💊 Medications Tab
Displays comprehensive medication analysis for vasoactives, sedatives, and paralytics:

**Vasoactive Medications (💉):**
- Area Under Curve (7 days) - Interactive HTML plot
- Median Dose by Hour - Interactive HTML plot

**Sedative Medications (😴):**
- Area Under Curve (7 days) - Interactive HTML plot
- Median Dose by Hour - Interactive HTML plot

**Paralytic Medications (🦴):**
- Area Under Curve (7 days) - Interactive HTML plot
- Median Dose by Hour - Interactive HTML plot

**Summary Statistics:**
- Medication usage statistics table (downloadable CSV)

All visualizations are stacked vertically for easy comparison and are fully interactive Plotly charts.

### 4. 🫁 IMV Tab
Displays invasive mechanical ventilation analysis:

**Visualizations:**
- **Tidal Volume** - Distribution in volume control modes
- **Pressure Control** - Settings in pressure control mode
- **Mode Proportions** - Ventilation mode usage in first 24 hours

**Ventilator Settings Data Tables:**
- **Ventilator Settings Summary** - Settings grouped by device mode (downloadable CSV)
- **Ventilator Settings Counts** - Count statistics by device mode (downloadable CSV)

Both tables displayed side-by-side for easy comparison.

### 5. 📊 SOFA & CCI Tab
Displays SOFA scores and Charlson Comorbidity Index analysis:

**Visualizations:**
- **SOFA Score & Mortality** - Histogram of SOFA scores with mortality overlay
- **Charlson Comorbidity Index** - Mortality & hospice analysis by CCI
- **Comorbidity Prevalence** - Bar plot of comorbidities per 1,000 hospitalizations
  - Expandable data table (downloadable CSV)

### 6. 🏥 Hospice & Outcomes Tab
Displays hospice and mortality outcome analysis:

**Visualizations:**
- **Hospice & Mortality Trends** - Combined temporal trends over time

## Features

### Interactive Elements
- **Expandable Sections** - Click to expand/collapse detailed data
- **Download Buttons** - Export CSV files for further analysis
- **Interactive HTML Plots** - Zoom, pan, and hover for details on Plotly charts
- **Scrollable Views** - Large tables and plots are scrollable within their containers

### Execution Summary
At the top of the viewer, click "ℹ️ Generation Summary" to see:
- Memory usage statistics
- Peak memory consumption
- Execution time
- Checkpoint information

## File Locations

All displayed results are from:
```
output/final/tableone/
├── consort_flow_diagram.png
├── cohort_intersect_upset_plot.png
├── venn_all_4_groups.png
├── code_status_stacked_bar_with_missingness_excl_missing_cat.png
├── table_one_by_year.csv
├── table_one_overall.csv
├── tidal_volume_volume_control_modes.png
├── pressure_control_pressure_control_mode.png
├── mode_proportions_first_24h_vertical.png
├── hospice_mortality_combined_trends.png
├── cci_mortality_hospice_comprehensive.png
├── sofa_mortality_histogram.png
├── comorbidities_per_1000_barplot.png
├── vasoactive_area_curve_7d.html
├── vasoactive_median_dose_by_hour.html
├── sedative_area_curve_7d.html
├── sedative_median_dose_by_hour.html
├── paralytic_area_curve_7d.html
├── paralytic_median_dose_by_hour.html
├── medications_summary_stats.csv
├── comorbidities_per_1000_hospitalizations.csv
├── ventilator_settings_by_device_mode.csv
├── ventilator_settings_counts_by_device_mode.csv
├── execution_report.txt
└── figures/
    ├── sankey_matplotlib_icu.png
    ├── sankey_matplotlib_others.png
    ├── sankey_matplotlib_high_o2_support.png
    └── sankey_matplotlib_vaso_support.png
```

## Missing Files

If certain visualizations are not available, the viewer will display:
- ⚠️ Warning - For critical files (CONSORT diagram, Table One CSVs)
- ℹ️ Info - For optional files (SOFA histogram, specific Sankey diagrams)

The viewer gracefully handles missing files and continues to display all available results.

## Workflow Integration

### Complete Workflow
```bash
# 1. Run validation and table one generation
python run_project.py --sample

# 2. Launch app
uv run streamlit run app.py

# 3. Click "📊 Table One Results" in left navigation
```

### Or with Auto-Launch
```bash
# Automatically launch app after generation
python run_project.py --sample --launch-app

# Then click "📊 Table One Results" when app opens
```

## Tips

1. **Large HTML Files** - Vasoactive/medication plots are interactive but may take a moment to load
2. **Download Data** - Use download buttons to export tables for presentations/papers
3. **Zoom on Images** - Click images to view in full screen
4. **Execution Report** - Check this first if results look incomplete
5. **Missing Files** - If key visualizations are missing, check `output/final/tableone/execution_report.txt` for errors during generation

## Troubleshooting

### Button Not Visible
**Problem:** "📊 Table One Results" button doesn't appear in navigation

**Solution:**
- Ensure Table One has been generated
- Check that `output/final/tableone/consort_flow_diagram.png` exists
- Check that `output/final/tableone/table_one_overall.csv` exists
- Refresh the app page

### Empty or Missing Content
**Problem:** Tab shows warnings about missing files

**Solution:**
- Check `output/final/tableone/execution_report.txt` for generation errors
- Re-run Table One generation: `python run_project.py --tableone-only`
- Ensure sufficient memory was available during generation

### HTML Plots Not Loading
**Problem:** Vasoactive/medication plots show blank

**Solution:**
- Wait a few seconds for large HTML files to load
- Check browser console for errors
- Verify `.html` files exist in `output/final/tableone/`
- Try refreshing the page

## Technical Details

### Module Location
`modules/tableone_viewer.py` - Contains all viewer functions

### Functions
- `check_tableone_results_available()` - Checks if results exist
- `show_tableone_results()` - Main viewer function
- `display_cohort_tab()` - Cohort visualizations
- `display_demographics_tab()` - Demographics and Table One with key metrics
- `display_medications_tab()` - Combined vasoactives, sedatives, and paralytics
- `display_imv_tab()` - Invasive mechanical ventilation
- `display_comorbidities_tab()` - SOFA scores and CCI analysis
- `display_hospice_tab()` - Hospice and mortality outcomes

### Integration
The viewer is integrated into `app.py`:
- Added to imports (line 50)
- Navigation button (lines 524-529)
- Page routing (lines 689-691)
- Sidebar logic updated to handle Table One page

## Future Enhancements

Potential additions for future versions:
- Export all visualizations as a ZIP file
- PDF report generation from viewer
- Comparison view for multiple runs
- Custom date range filtering
- Export to presentation format (PowerPoint)

---

For questions or issues, please refer to the main [README.md](README.md) or report issues on GitHub.
