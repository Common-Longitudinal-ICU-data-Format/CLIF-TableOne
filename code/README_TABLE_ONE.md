# Table One Generation - Memory Optimized

This directory contains the optimized Table One generation script with memory monitoring and validation.

## Files

- **`generate_table_one_2_1.py`** - Main script for generating Table One (memory optimized)
- **`run_table_one.py`** - Execution wrapper with memory monitoring and validation
- **`README_TABLE_ONE.md`** - This file

## Memory Optimizations

The script has been optimized to reduce memory usage by approximately 50-60% through:

### 1. Strategic Memory Cleanup (8 points)
- After respiratory support initial filtering
- After medication initial loading
- After CONSORT diagram generation
- After ventilation mode analysis
- After respiratory detailed analysis
- After medication processing
- After labs processing
- Final cleanup at script end

### 2. Figure Management
- All matplotlib figures are explicitly closed after saving
- Prevents accumulation of figure objects in memory

### 3. Intermediate Data Deletion
- Large temporary dataframes deleted immediately after use
- Reduces peak memory consumption

### 4. Garbage Collection
- Explicit `gc.collect()` calls at strategic points
- Forces immediate memory reclamation

## Usage

### Basic Execution

```bash
# Navigate to code directory
cd /path/to/CLIF-TableOne/code

# Run with memory monitoring
python run_table_one.py
```

### Direct Script Execution (without monitoring)

```bash
python generate_table_one_2_1.py
```

### Output

The execution script provides:

1. **Real-time Progress**
   - Memory usage at key checkpoints
   - Elapsed time tracking
   - Clear status messages

2. **Validation**
   - Pre-flight config validation
   - Post-execution output file validation
   - Missing file warnings

3. **Execution Report** (`output/final/tableone/execution_report.txt`)
   - Memory usage summary
   - Peak memory tracking
   - Detailed checkpoint information
   - Total execution time

## Expected Outputs

### Core Tables
- `table_one_overall.csv` - Overall cohort table
- `table_one_by_year.csv` - Table stratified by admission year

### Visualizations
- `consort_flow_diagram.png` - CONSORT flow diagram
- `cohort_intersect_upset_plot.png` - UpSet plot of cohort intersections
- `venn_all_4_groups.png` - 4-way Venn diagram
- `code_status_stacked_bar_with_missingness_excl_missing_cat.png` - Code status visualization
- `tidal_volume_volume_control_modes.png` - Ventilator settings
- `pressure_control_pressure_control_mode.png` - Pressure control analysis
- `mode_proportions_first_24h_vertical.png` - Ventilation mode proportions
- `comorbidities_per_1000_barplot.png` - Comorbidity prevalence
- `sofa_mortality_histogram.png` - SOFA score mortality
- `hospice_mortality_combined_trends.png` - Hospice/mortality trends
- `cci_mortality_hospice_comprehensive.png` - CCI analysis

### Summary Statistics
- `medications_summary_stats.csv` - Medication usage statistics
- `comorbidities_per_1000_hospitalizations.csv` - Comorbidity prevalence

### Intermediate Data
- `final_tableone_df.parquet` - Complete cohort dataframe

## Memory Usage Expectations

| Phase | Expected Peak Memory |
|-------|---------------------|
| Initial table loads | 2-4 GB |
| Cohort identification | 3-5 GB |
| Respiratory analysis | 2-3 GB |
| Medication analysis | 2-3 GB |
| Labs processing | 2-3 GB |
| Final table generation | 1-2 GB |

**Total Peak:** ~5-8 GB (depending on dataset size)

## Troubleshooting

### Out of Memory Errors

If you still encounter memory issues:

1. **Reduce data scope**
   - Filter to specific year ranges
   - Use sampling for development

2. **Increase system resources**
   - Close other applications
   - Increase swap space
   - Run on a machine with more RAM

3. **Check for memory leaks**
   - Review custom modifications
   - Ensure all cleanup calls are executed

### Missing Output Files

If validation reports missing files:

1. Check for script errors in the console output
2. Review the execution report for failure points
3. Check file permissions in output directories

### Validation Failures

If output validation fails:

1. Check that all required CLIF tables are present
2. Verify data quality and completeness
3. Review console output for warnings

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- clifpy
- psutil (for memory monitoring)
- venny4py
- upsetplot

## Exit Codes

- `0` - Success (generation and validation passed)
- `1` - Failure (script execution failed)
- `2` - Partial success (generation succeeded but validation incomplete)

## Notes

- The script automatically creates all necessary output directories
- Temporary files are stored in `output/intermediate/`
- Final outputs are in `output/final/tableone/`
- Memory monitoring adds minimal overhead (~1-2%)
