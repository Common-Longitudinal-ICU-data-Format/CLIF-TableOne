"""Table listing routes."""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from fastapi import APIRouter
from server import session
from server.services import cache_service
from modules.utils.feedback import load_feedback

import yaml

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["tables"])

# Cache schema metadata keyed by (path, mtime) to avoid re-reading unchanged files
_schema_cache: dict[tuple[str, float], dict] = {}

# Cache CLIF schema versions (loaded once from clifpy YAML files)
_clif_version_cache: dict[str, str | None] = {}


def _get_clif_version(table_name: str) -> str | None:
    """Read the CLIF spec version from the clifpy schema YAML for a table."""
    if table_name in _clif_version_cache:
        return _clif_version_cache[table_name]
    try:
        import clifpy
        schema_dir = Path(clifpy.__file__).parent / "schemas"
        schema_file = schema_dir / f"{table_name}_schema.yaml"
        if schema_file.exists():
            with open(schema_file) as f:
                schema = yaml.safe_load(f)
            version = schema.get("version")
            _clif_version_cache[table_name] = version
            return version
    except Exception:
        pass
    _clif_version_cache[table_name] = None
    return None

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
                "file_name": os.path.basename(path),
                "file_size_bytes": st.st_size,
                "file_modified": st.st_mtime,
                "_resolved_path": path,
                "_mtime": st.st_mtime,
            }
        except OSError:
            continue
    return {"file_exists": False, "file_name": None, "file_size_bytes": None, "file_modified": None}


def _table_schema_meta(path: str, mtime: float, filetype: str) -> dict:
    """Read column names, types, and row count from a data file.

    For parquet: reads only the file footer (very fast).
    For CSV: uses polars lazy scanning.
    Results are cached by (path, mtime).
    """
    cache_key = (path, mtime)
    if cache_key in _schema_cache:
        return _schema_cache[cache_key]

    try:
        if filetype == "parquet":
            schema = pl.read_parquet_schema(path)
            columns = list(schema.keys())
            data_types = {col: str(dtype) for col, dtype in schema.items()}
            # Row count from pyarrow metadata (footer only, no data read)
            try:
                import pyarrow.parquet as pq
                pf_meta = pq.read_metadata(path)
                row_count = pf_meta.num_rows
            except Exception:
                row_count = pl.scan_parquet(path).select(pl.len()).collect().item()
        else:
            # CSV
            lf = pl.scan_csv(path)
            schema = lf.collect_schema()
            columns = list(schema.keys())
            data_types = {col: str(dtype) for col, dtype in schema.items()}
            row_count = lf.select(pl.len()).collect().item()

        result = {
            "row_count": row_count,
            "num_columns": len(columns),
            "columns": columns,
            "data_types": data_types,
        }
    except Exception as e:
        logger.warning("Failed to read schema for %s: %s", path, e)
        result = {"row_count": None, "num_columns": None, "columns": None, "data_types": None}

    _schema_cache[cache_key] = result
    return result


def _save_file_metadata(tables: dict, config: dict) -> None:
    """Save file metadata snapshot to output/final/meta/file_metadata.json."""
    output_dir = config.get("output_dir", "output")
    filetype = config.get("file_type", "parquet")
    final_dir = Path(output_dir) / "final" / "meta"
    final_dir.mkdir(parents=True, exist_ok=True)

    enriched_tables = {}
    for name, info in tables.items():
        entry = {
            "display_name": info["display_name"],
            "validated_against": f"CLIF v{v}" if (v := _get_clif_version(name)) else None,
            "file_exists": info["file_exists"],
            "file_name": info.get("file_name"),
            "file_size_bytes": info["file_size_bytes"],
            "file_size_gb": round(info["file_size_bytes"] / (1024 ** 3), 6) if info["file_size_bytes"] else None,
            "file_modified": info["file_modified"],
        }
        # Enrich with schema metadata for files that exist
        resolved_path = info.get("_resolved_path")
        mtime = info.get("_mtime")
        if resolved_path and mtime:
            schema_meta = _table_schema_meta(resolved_path, mtime, filetype)
            entry.update(schema_meta)
        else:
            entry.update({"row_count": None, "num_columns": None, "columns": None, "data_types": None})
        enriched_tables[name] = entry

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tables": enriched_tables,
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
            "validated_against": f"CLIF v{v}" if (v := _get_clif_version(name)) else None,
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

    # Strip internal keys before returning API response
    for info in tables.values():
        info.pop("_resolved_path", None)
        info.pop("_mtime", None)

    return {"tables": tables}
