@echo off
REM Windows batch file to run the FastAPI app with UTF-8 encoding support
REM This ensures emojis and Unicode characters display correctly on Windows

echo Setting UTF-8 encoding for Python...
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

echo.
echo Starting CLIF Table One FastAPI App with UTF-8 support...
echo Open http://127.0.0.1:8000 in a browser.
echo.

uv run uvicorn server.main:app --reload %*

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Error occurred. Exit code: %ERRORLEVEL%
    pause
)
