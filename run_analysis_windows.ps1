# PowerShell script to run run_analysis.py with UTF-8 encoding support
# This ensures emojis and Unicode characters display correctly on Windows

Write-Host "Setting UTF-8 encoding for Python..." -ForegroundColor Green
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

Write-Host ""
Write-Host "Running CLIF Analysis with UTF-8 support..." -ForegroundColor Cyan
Write-Host ""

# Run the Python script with all command line arguments
uv run python run_analysis.py $args

# Check exit code
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Error occurred. Exit code: $LASTEXITCODE" -ForegroundColor Red
    Read-Host "Press Enter to continue"
}