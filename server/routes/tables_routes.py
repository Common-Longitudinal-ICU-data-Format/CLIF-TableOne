"""Table listing routes."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter
from server import session
from server.services import cache_service
from modules.utils.feedback import load_feedback

router = APIRouter(prefix="/api", tags=["tables"])

TABLE_DISPLAY_NAMES = {
    'patient': 'Patient',
    'hospitalization': 'Hospitalization',
    'adt': 'ADT',
    'code_status': 'Code Status',
    'crrt_therapy': 'CRRT Therapy',
    'ecmo_mcs': 'ECMO/MCS',
    'hospital_diagnosis': 'Hospital Diagnosis',
    'labs': 'Labs',
    'medication_admin_continuous': 'Medication Admin (Continuous)',
    'medication_admin_intermittent': 'Medication Admin (Intermittent)',
    'microbiology_culture': 'Microbiology Culture',
    'microbiology_nonculture': 'Microbiology Non-Culture',
    'microbiology_susceptibility': 'Microbiology Susceptibility',
    'patient_assessments': 'Patient Assessments',
    'patient_procedures': 'Patient Procedures',
    'position': 'Position',
    'respiratory_support': 'Respiratory Support',
    'vitals': 'Vitals',
}

ALL_TABLES = list(TABLE_DISPLAY_NAMES.keys())


def _file_meta(name: str, config: dict) -> dict:
    """Get file metadata via os.stat() — no file I/O, just inode lookup."""
    tables_path = config.get("tables_path", "")
    filetype = config.get("file_type", "parquet")
    for prefix in ("", "clif_"):
        path = os.path.join(tables_path, f"{prefix}{name}.{filetype}")
        try:
            st = os.stat(path)
            return {
                "file_exists": True,
                "file_size_bytes": st.st_size,
                "file_modified": st.st_mtime,
            }
        except OSError:
            continue
    return {"file_exists": False, "file_size_bytes": None, "file_modified": None}


def _save_file_metadata(tables: dict, config: dict) -> None:
    """Save file metadata snapshot to output/final/file_metadata.json."""
    output_dir = config.get("output_dir", "output")
    final_dir = Path(output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tables": {
            name: {
                "display_name": info["display_name"],
                "file_exists": info["file_exists"],
                "file_size_bytes": info["file_size_bytes"],
                "file_modified": info["file_modified"],
            }
            for name, info in tables.items()
        },
    }
    (final_dir / "file_metadata.json").write_text(json.dumps(meta, indent=2))


@router.get("/tables")
async def get_tables():
    """Return all 18 tables with status/timestamp info."""
    cache_service.init()
    config = session.get("config") or {}
    output_dir = config.get("output_dir", "output")
    tables = {}
    for name in ALL_TABLES:
        cached = cache_service.get(name)
        info = {
            "display_name": TABLE_DISPLAY_NAMES[name],
            "status": cache_service.status(name) if cached else "not_analyzed",
            "timestamp": cached["timestamp"] if cached else None,
            "validation_complete": cached.get("validation_complete", False) if cached else False,
            "summary_complete": cached.get("summary_complete", False) if cached else False,
        }
        fb = load_feedback(output_dir, name)
        if fb and fb.get("timestamp"):
            info["feedback_timestamp"] = fb["timestamp"]
            r = fb.get("rejected_count", 0)
            a = fb.get("accepted_count", 0)
            p = fb.get("pending_count", 0)
            info["feedback_summary"] = f"{r}R/{a}A/{p}P"
        info.update(_file_meta(name, config))
        tables[name] = info

    # Persist file metadata snapshot to output/final/
    _save_file_metadata(tables, config)

    return {"tables": tables}
