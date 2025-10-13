@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

REM ── Step 0: Go to script directory ──
cd /d %~dp0

REM ── Step 1: Create virtual environment if missing ──
if not exist ".clif_table_one\" (
    echo Creating virtual environment...
    python -m venv .clif_table_one
) else (
    echo Virtual environment already exists.
)

REM ── Step 2: Activate virtual environment ──
call .clif_table_one\Scripts\activate.bat

REM ── Step 3: Install required packages ──
echo Installing dependencies...
pip install --quiet -r requirements.txt
pip install --quiet jupyter ipykernel papermill

REM ── Step 4: Register kernel ──
python -m ipykernel install --user --name=.clif_table_one --display-name="Python (clif_table_one)"

REM ── Step 5: Set environment variables ──
set PYTHONWARNINGS=ignore
set PYTHONPATH=%cd%\code;%PYTHONPATH%

REM ── Step 6: Change to code directory ──
cd code

REM ── Step 7: Create logs folder ──
if not exist logs (
    mkdir logs
)

REM ── Step 8: Run analysis notebooks using papermill ──
echo.
echo Running generate_table_one.ipynb ...
papermill generate_table_one.ipynb generate_table_one.ipynb > logs\generate_table_one.log
echo Finished generate_table_one.ipynb

REM ── Step 10: Done ──
echo.
echo ✅ All steps completed successfully!
pause
exit /b
