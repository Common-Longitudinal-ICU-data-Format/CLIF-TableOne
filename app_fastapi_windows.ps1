# PowerShell script to run the FastAPI app with UTF-8 encoding support
# This ensures emojis and Unicode characters display correctly on Windows

Write-Host "Setting UTF-8 encoding for Python..." -ForegroundColor Green
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

Write-Host ""
Write-Host "Starting CLIF Table One FastAPI App with UTF-8 support..." -ForegroundColor Cyan
Write-Host "Open http://127.0.0.1:8000 in a browser."
Write-Host ""

uv run uvicorn server.main:app --reload $args

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Error occurred. Exit code: $LASTEXITCODE" -ForegroundColor Red
    Read-Host "Press Enter to continue"
}
