# How to Run CLIF-TableOne

## Prerequisites

- **Python 3.11+**
- **[UV package manager](https://docs.astral.sh/uv/)** — install with:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **CLIF 2.1 data** in parquet (or CSV/FST) format

## 1. Install Dependencies

```bash
cd CLIF-TableOne
uv sync
```

> **Note:** Always use `uv run` to execute commands — this ensures the project's virtual environment is used (not your system Python).

## 2. Configure

Copy the template and fill in your site details:

```bash
cp config/config_template.json config/config.json
```

Edit `config/config.json`:

```json
{
    "site_name": "Your_Site_Name",
    "tables_path": "/path/to/your/clif/data",
    "file_type": "parquet",
    "timezone": "US/Central"
}
```

| Field | Description |
|---|---|
| `site_name` | Your institution name (e.g., `UCMC`, `MIMIC`) |
| `tables_path` | Absolute path to the directory containing your CLIF data files (no trailing `/`) |
| `file_type` | Data format: `parquet`, `csv`, or `fst` |
| `timezone` | Your site's timezone (e.g., `US/Central`, `US/Eastern`, `America/Chicago`) |

## 3. Run the Analysis Pipeline

```bash
uv run python run_project.py --no-summary --get-ecdf
```

This validates all 18 CLIF tables, collects MCIDE data, generates Table One, and computes ECDF bins.

| Flag | Purpose |
|---|---|
| `--sample` | Use a 1k ICU sample for faster runs (~10-15 min vs 45-90 min) |
| `--no-summary` | Skip summary generation |
| `--get-ecdf` | Compute ECDF distributions for visualizations |

For all available options: `uv run python run_project.py --help`

## 4. Launch the Web App

```bash
uv run uvicorn server.main:app --reload
```

Open **http://127.0.0.1:8000** in your browser.

| Flag | Purpose |
|---|---|
| `--reload` | Auto-restart on code changes (development mode) |
| `--host 0.0.0.0` | Allow access from other machines on the network |
| `--port 8080` | Use a different port (default: 8000) |

## 5. Using the App

- **Validation Tab** — Review validation results for each CLIF table
- **mCIDE Tab** — View mCIDE and summary statistics
- **Table One Results** — Cohort analysis, demographics, medications, IMV, SOFA/CCI, outcomes
- **Feedback** — Classify validation errors as Accepted/Rejected/Pending; save to update table status

## Windows

Use the provided scripts that handle UTF-8 encoding:

```batch
run_project_windows.bat --no-summary --get-ecdf
```

Or PowerShell:
```powershell
.\run_project_windows.ps1 --no-summary --get-ecdf
```

If you encounter Unicode issues, set `PYTHONIOENCODING=utf-8` or enable system-wide UTF-8 (see [README.md](README.md#windows-unicode-troubleshooting)).

## Output

Results are written to `output/final/`:

```
output/final/
├── reports/       # Validation PDF reports
├── results/       # Validation CSV summaries
├── tableone/      # Table One outputs, MCIDE, plots
├── ecdf/          # ECDF distributions
├── bins/          # Quantile bins for visualization
└── configs/       # ECDF configuration files
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'fastapi'` | You ran `uvicorn` with system Python. Use `uv run uvicorn ...` instead. |
| `Could not load config` on startup | Ensure `config/config.json` exists and is valid JSON. |
| `uv sync` fails on clifpy | clifpy must be at `/Users/dema/WD/clifpy` (or update `pyproject.toml` path). |
| Port already in use | Use `--port 8080` or kill the existing process. |
