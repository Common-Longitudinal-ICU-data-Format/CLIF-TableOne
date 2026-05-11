"""MCIDE data routes."""

import os
import json
import glob

import pandas as pd
from fastapi import APIRouter, HTTPException
from server import session
from modules.mcide.viewer import get_table_mcide_files

router = APIRouter(prefix="/api", tags=["mcide"])


@router.get("/mcide/{name}")
async def get_mcide(name: str):
    """Return MCIDE CSV data and summary stats as JSON for a table."""
    from modules.utils.output_paths import mcide_dir as _mcide_dir, summary_stats_dir as _summary_stats_dir
    config = session.get("config") or {}
    output_dir = config.get("output_dir", "output")
    mcide_dir = str(_mcide_dir())
    stats_dir = str(_summary_stats_dir())

    if not os.path.exists(mcide_dir):
        raise HTTPException(404, "No MCIDE data available")

    table_files = get_table_mcide_files(mcide_dir, name)

    result = {"mcide_files": [], "stats_files": []}

    for filepath in table_files:
        try:
            df = pd.read_csv(filepath).fillna("")
            basename = os.path.basename(filepath).replace('_mcide.csv', '').replace('clif_', '')
            result["mcide_files"].append({
                "name": basename,
                "columns": df.columns.tolist(),
                "data": df.to_dict(orient="records"),
                "row_count": len(df),
                "total_n": int(df["N"].sum()) if "N" in df.columns else None,
            })
        except Exception as e:
            result["mcide_files"].append({
                "name": os.path.basename(filepath),
                "error": str(e),
            })

    # Load summary stats
    if os.path.exists(stats_dir):
        patterns = [f"{name}_*.json"]
        if name == 'crrt_therapy':
            patterns = ["crrt_*.json"]
        for pattern in patterns:
            for stats_file in sorted(glob.glob(os.path.join(stats_dir, pattern))):
                try:
                    with open(stats_file, 'r', encoding='utf-8') as f:
                        stats_data = json.load(f)
                    result["stats_files"].append({
                        "name": os.path.basename(stats_file).replace('.json', ''),
                        "data": stats_data,
                    })
                except Exception as e:
                    result["stats_files"].append({
                        "name": os.path.basename(stats_file),
                        "error": str(e),
                    })

    return result
