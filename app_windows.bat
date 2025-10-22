@echo off
REM Windows batch file to run Streamlit app with UTF-8 encoding support
REM This ensures emojis and Unicode characters display correctly on Windows

echo Setting UTF-8 encoding for Python...
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

echo.
echo Starting CLIF Table One Streamlit App with UTF-8 support...
echo.

uv run streamlit run app.py %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error occurred. Exit code: %ERRORLEVEL%
    pause
)