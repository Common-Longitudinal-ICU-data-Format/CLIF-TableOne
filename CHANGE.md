# Changelog

## Quick Reference

| Date | Changes |
|------|---------|
| 2026-01-09 | [Remove 1k Sampling Function](#2026-01-09-remove-1k-sampling-function) |
| 2025-01-09 | [Venn Diagram Output Paths](#2025-01-09-venn-diagram-output-paths) &#124; [Remove Archive Folder](#2025-01-09-remove-archive-folder) &#124; [Execution Report Enhancements](#2025-01-09-execution-report-enhancements) |

---

## 2026-01-09: Remove 1k Sampling Function

### Summary
Removed `modules/utils/sampling.py` which contained stratified sampling utilities for development/testing with smaller datasets.

### Deleted Files
- `modules/utils/sampling.py` - Contained `generate_stratified_sample()` and `get_icu_hospitalizations_from_adt()` functions

---

## 2025-01-09: Venn Diagram Output Paths

### Summary
Updated venny4py output paths to save generated files to proper output directories instead of the repository root.

### Changes Made

#### File Modified: `modules/tableone/generator.py`

**1. New Import**
- Added `import shutil` for file operations

**2. Updated `venny4py` call (line 1411)**
- Added `out` parameter to save `Venn_4.png` to `output/final/tableone/figures/`
- Previously saved to repository root

**3. New: Move Intersections File (lines 1417-1422)**
- After venny4py generates files, moves `Intersections_4.txt` from figures folder to `output/intermediate/`
- Uses `shutil.move` so file only exists in final destination

### Output Locations
| File | Old Location | New Location |
|------|--------------|--------------|
| `Venn_4.png` | Repository root | `output/final/tableone/figures/` |
| `Intersections_4.txt` | Repository root | `output/intermediate/` |

---

## 2025-01-09: Remove Archive Folder

### Summary
Removed the `code/archives/` folder containing legacy/deprecated scripts. Historical code is now preserved via git release tags instead.

### Deleted Files
- `code/archives/README.md`
- `code/archives/clif_report_card.py`
- `code/archives/convert_notebook.py`
- `code/archives/generate_clif_report_card.py`
- `code/archives/generate_table_one.bat`
- `code/archives/generate_table_one.ipynb`
- `code/archives/generate_table_one.sh`
- `code/archives/generate_table_one_2_1.ipynb`
- `code/archives/generate_table_one_2_1.py.backup`
- `code/archives/generate_table_one_2_1_og.py`
- `code/archives/pyCLIF.py`
- `code/archives/sofa_score.py`

### Rationale
Instead of maintaining archived code in the repository, releases are now tagged in git to preserve historical versions.

---

## 2025-01-09: Execution Report Enhancements

### Summary
Added comprehensive system resource and input data size reporting to the Table One execution report (`output/final/tableone/execution_report.txt`). This provides visibility into the hardware environment and input data scale before execution begins.

### Changes Made

#### File Modified: `modules/tableone/runner.py`

**1. New Imports and Constants (lines 13, 19-27)**
- Added `platform` module for cross-platform detection
- Added `CLIF_TABLES` constant listing all 18 CLIF input tables:
  - patient, hospitalization, adt, labs, vitals
  - medication_admin_continuous, medication_admin_intermittent
  - patient_assessments, respiratory_support, position
  - hospital_diagnosis, microbiology_culture, crrt_therapy
  - patient_procedures, microbiology_susceptibility, ecmo_mcs
  - microbiology_nonculture, code_status

**2. New Method: `get_system_resources()` (lines 123-170)**
- Captures total and available RAM using `psutil`
- Cross-platform GPU detection with graceful fallbacks:
  - NVIDIA GPUs via `pynvml` (reports VRAM total/available)
  - AMD GPUs via `pyamdgpuinfo` (reports VRAM total)
  - Apple Silicon detection (notes unified memory architecture)
  - Falls back to "None detected" if no GPU found

**3. New Method: `scan_input_tables()` (lines 172-211)**
- Scans all 18 CLIF tables from `tables_path` before execution
- Uses DuckDB for efficient row/column counting without loading full data into memory
- Collects per-table: file size (MB), row count, column count
- Handles missing tables gracefully (marks as "Not found")
- Supports both parquet and CSV file formats

**4. Updated Method: `run()` (lines 371-375)**
- Calls `get_system_resources()` and `scan_input_tables()` after config validation
- Stores results in `self.system_resources` and `self.input_table_sizes`
- Adds new memory checkpoint: "Input Tables Scanned"

**5. Updated Method: `generate_report()` (lines 326-374)**
- Added "SYSTEM RESOURCES (at start)" section:
  - RAM total/available with percentage
  - GPU type and name
  - VRAM total/available (for NVIDIA/AMD)
- Added "INPUT DATA SIZE SUMMARY" section:
  - Table listing with rows, columns, file size for each of 18 tables
  - Totals row with aggregate statistics
  - Missing tables marked as "Not found"

### New Report Sections

The execution report now includes two new sections after "MEMORY CHECKPOINTS":

```
================================================================================
SYSTEM RESOURCES (at start)
================================================================================

RAM:    Total: 64.0 GB | Available: 45.2 GB (70.6%)

GPU:    NVIDIA GeForce RTX 3090
VRAM:   Total: 24.0 GB | Available: 22.1 GB (92.1%)

================================================================================
INPUT DATA SIZE SUMMARY
================================================================================

Table                                        Rows     Cols    File Size
--------------------------------------------------------------------------------
clif_patient                               12,345       15       2.30 MB
clif_hospitalization                       98,765       28       8.10 MB
clif_adt                                  156,789       12       5.40 MB
...
--------------------------------------------------------------------------------
Total (18 tables)                         500,000      ---      45.20 MB
```

### Dependencies

**Required (already installed):**
- `duckdb` - For efficient table scanning
- `psutil` - For RAM monitoring (already in use)
- `platform` - Built-in Python module

**Optional (for GPU detection):**
- `pynvml` - NVIDIA GPU support (`uv add pynvml`)
- `pyamdgpuinfo` - AMD GPU support (`uv add pyamdgpuinfo`)

GPU packages are optional - if not installed, the code gracefully falls back to reporting "None detected" or "Apple Silicon" on macOS.

### Testing

Run the Table One generation and verify the new sections appear in the report:

```bash
python -m modules.tableone.runner
cat output/final/tableone/execution_report.txt
```
