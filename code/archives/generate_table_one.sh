#!/usr/bin/env bash

#  Combined setup and notebook execution script for CLIF project (Mac/Linux)

set -e
set -o pipefail

# ── ANSI colours for pretty output ─────────────────────────────────────────────
PURPLE="\033[35m"
CYAN="\033[36m"
GREEN="\033[32m"
RESET="\033[0m"

separator() {
  echo -e "${PURPLE}==================================================${RESET}"
}

# ── 1. Create virtual environment ──────────────────────────────────────────────
separator
if [ ! -d ".clif_table_one" ]; then
  echo -e "${CYAN}Creating virtual environment (.clif_table_one)...${RESET}"
  python3 -m venv .clif_table_one
else
  echo -e "${CYAN}Virtual environment already exists.${RESET}"
fi

# ── 2. Activate virtual environment ────────────────────────────────────────────
separator
echo -e "${CYAN}Activating virtual environment...${RESET}"
# shellcheck source=/dev/null
source .clif_table_one/bin/activate

# ── 3. Upgrade pip ─────────────────────────────────────────────────────────────
separator
echo -e "${CYAN}Upgrading pip...${RESET}"
python -m pip install --upgrade pip

# ── 4. Install required packages ───────────────────────────────────────────────
separator
echo -e "${CYAN}Installing dependencies...${RESET}"
pip install --quiet -r requirements.txt
pip install --quiet jupyter ipykernel

# ── 5. Register Jupyter kernel ─────────────────────────────────────────────────
separator
echo -e "${CYAN}Registering Jupyter kernel...${RESET}"
python -m ipykernel install --user --name=..clif_table_one --display-name="Python (clif_table_one)"

# ── 6. Change to code directory ────────────────────────────────────────────────
separator
echo -e "${CYAN}Changing to code directory...${RESET}"
cd code || { echo "❌  'code' directory not found."; exit 1; }

# ── 7. Convert and execute notebooks, streaming + logging ──────────────────────
mkdir -p logs

NOTEBOOKS=(
  "generate_table_one.ipynb"
)

for nb in "${NOTEBOOKS[@]}"; do
  base_name="${nb%.ipynb}"
  log_file="logs/${base_name}.log"
  separator
  echo -e "${CYAN}Executing ${nb} and logging output to ${log_file}...${RESET}"
  export MPLBACKEND=Agg
  jupyter nbconvert --to script --stdout "$nb" | python 2>&1 | tee "$log_file"
done

# ── 9. Done ────────────────────────────────────────────────────────────────────
separator
echo -e "${GREEN}✅ All setup and analysis scripts completed successfully!${RESET}"

read -rp "Press [Enter] to exit..."
