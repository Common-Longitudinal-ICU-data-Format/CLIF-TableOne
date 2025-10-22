@echo off
REM Windows batch file to run run_project.py with UTF-8 encoding support
REM This ensures emojis and Unicode characters display correctly on Windows

echo Setting UTF-8 encoding for Python...
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

echo.
echo Running CLIF Table One Project with UTF-8 support...
echo.

uv run python run_project.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error occurred. Exit code: %ERRORLEVEL%
    pause
)