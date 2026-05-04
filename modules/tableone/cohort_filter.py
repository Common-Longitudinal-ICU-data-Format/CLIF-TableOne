"""Cohort-first filtering with manifest-based cache.

The core idea (from the Phase 2 refactor plan in dev/schema_audit.md):
    1. Define the critical-illness cohort using only the cheap CLIF tables
       (adt + hospitalization + patient + respiratory_support + meds_continuous).
    2. Persist the cohort `hospitalization_id` set to disk.
    3. Filter every other heavy CLIF table (labs, vitals, meds_intermittent,
       patient_assessments, position, microbiology_culture, crrt_therapy) to
       cohort-only rows and write slim parquets to
       `output/intermediate/clif_filtered/<table>_cohort.parquet`.
    4. The rest of the pipeline reads from the slim parquets, never the raw
       full-site tables.

For UCMC (~30% of all hospitalizations are critically ill), this is a 3× memory
reduction on the heavy tables. For UMN at 4M total hospitalizations, it's the
difference between OOM and finishing.

A manifest (`_manifest.json`) sits alongside the filtered parquets describing
the source files' (mtime, size) and a hash of the cohort definition. On
subsequent runs the manifest is read first; if everything matches, the cached
filtered parquets are reused. A `--force-refresh` flag bypasses the cache.

This module is designed to be standalone — it does NOT define the cohort
logic itself (that lives in `generator.py:main()` and uses many in-flight
DataFrames). It provides:
    - manifest read/write/validate
    - cohort-definition hashing
    - per-table filter that streams parquet row groups (memory-bounded,
      independent of source size)
    - orchestration helpers

Integration notes for wiring this into `main()`:
    See the docstring of `compute_or_use_cached_filtered_tables()` below.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable, Optional

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


__all__ = [
    "MANIFEST_FILENAME",
    "DEFAULT_FILTER_SPEC",
    "TableFilterSpec",
    "Manifest",
    "hash_cohort_definition",
    "write_manifest",
    "read_manifest",
    "is_cache_valid",
    "filter_table_to_cohort",
    "filter_all_clif_tables_to_cohort",
    "compute_or_use_cached_filtered_tables",
]


MANIFEST_FILENAME = "_manifest.json"


# ---------------------------------------------------------------------------
# What to filter
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TableFilterSpec:
    """Spec for filtering one CLIF table to a cohort.

    Attributes:
        source_filename: parquet file under `source_dir` (e.g. 'clif_labs.parquet')
        dest_filename:   parquet file under `dest_dir`  (e.g. 'labs_cohort.parquet')
        id_column:       column in the table that joins to hospitalization_id.
                         Almost always 'hospitalization_id'; some tables (patient,
                         hospital_diagnosis) use 'patient_id' or stay unfiltered.
        columns_to_keep: optional projection. None = keep all columns.
                         Recommend providing this from dev/schema_audit.md
                         allow-list; saves additional memory + IO.
    """

    source_filename: str
    dest_filename: str
    id_column: str = "hospitalization_id"
    columns_to_keep: Optional[tuple[str, ...]] = None


# Default filter spec for the heavy CLIF 2.1 tables. Add/remove as needed.
# Column projections come from `dev/schema_audit.md` Section 6 (the Phase-4
# allow-list); leave None for now to preserve all columns. We can tighten
# projections in Phase 4 once Phase 2 is integrated.
DEFAULT_FILTER_SPEC: tuple[TableFilterSpec, ...] = (
    # Only tables that main() actually reads from the filtered cache.
    # respiratory_support and medication_admin_continuous are inputs to the
    # cohort definition itself, loaded fully BEFORE this filter runs.
    TableFilterSpec("clif_vitals.parquet",      "vitals_cohort.parquet",      "hospitalization_id"),
    TableFilterSpec("clif_crrt_therapy.parquet", "crrt_therapy_cohort.parquet", "hospitalization_id"),
)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

@dataclass
class Manifest:
    """Describes a cached filtered-cohort directory.

    The cache hits on the next run iff:
      - source_dir is unchanged
      - every source file in `source_files` still has the same (mtime, size)
      - cohort_hash is unchanged
    """

    source_dir: str
    cohort_hash: str
    source_files: dict[str, dict] = field(default_factory=dict)  # name → {mtime, size_bytes}
    produced_at: Optional[str] = None
    n_cohort_ids: Optional[int] = None
    note: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True, default=str)

    @classmethod
    def from_json(cls, text: str) -> "Manifest":
        data = json.loads(text)
        return cls(**data)


def hash_cohort_definition(version: str, params: Optional[dict] = None) -> str:
    """Stable hash for the cohort logic identifier + parameters.

    `version` should change whenever the cohort flag computations in
    `generator.py:main()` change in a way that would invalidate caches.
    `params` should include any tunable knobs (age threshold, year range,
    HFNC LPM threshold, etc.) — anything whose value affects the cohort.

    Returns a 16-char hex prefix (collision-resistant enough for cache
    invalidation; full hex is overkill for a manifest file).
    """
    payload = json.dumps(
        {"version": version, "params": params or {}},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _file_signature(path: Path) -> dict:
    """Filesystem signature of a parquet file: (mtime, size). Cheap to read."""
    st = path.stat()
    return {"mtime": st.st_mtime, "size_bytes": st.st_size}


def write_manifest(manifest: Manifest, dest_dir: Path) -> Path:
    """Serialize a Manifest to `<dest_dir>/_manifest.json`. Creates dest_dir if needed."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / MANIFEST_FILENAME
    out.write_text(manifest.to_json(), encoding="utf-8")
    return out


def read_manifest(dest_dir: Path) -> Optional[Manifest]:
    """Load the manifest if it exists; otherwise return None."""
    p = dest_dir / MANIFEST_FILENAME
    if not p.exists():
        return None
    try:
        return Manifest.from_json(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def is_cache_valid(
    manifest: Optional[Manifest],
    source_dir: Path,
    cohort_hash: str,
    expected_filenames: Iterable[str],
) -> bool:
    """Check whether the existing manifest matches current source state.

    Returns True only if:
      - manifest exists
      - cohort_hash matches
      - source_dir matches
      - every expected_filename exists in source_dir AND has the same
        (mtime, size_bytes) as recorded in the manifest

    Any divergence → False (cache miss → recompute).
    """
    if manifest is None:
        return False
    if str(source_dir) != manifest.source_dir:
        return False
    if cohort_hash != manifest.cohort_hash:
        return False

    for fname in expected_filenames:
        src = source_dir / fname
        if not src.exists():
            return False
        recorded = manifest.source_files.get(fname)
        if recorded is None:
            return False
        sig = _file_signature(src)
        # mtime can be a tiny float drift from JSON round-trip; allow 1us slack
        if abs(float(recorded["mtime"]) - sig["mtime"]) > 1e-6:
            return False
        if int(recorded["size_bytes"]) != sig["size_bytes"]:
            return False

    return True


# ---------------------------------------------------------------------------
# Per-table filter (streaming via pyarrow row groups)
# ---------------------------------------------------------------------------

def filter_table_to_cohort(
    source_path: Path,
    dest_path: Path,
    cohort_ids: set,
    *,
    id_column: str = "hospitalization_id",
    columns_to_keep: Optional[Iterable[str]] = None,
    compression: str = "snappy",
) -> dict:
    """Stream a parquet file row-group at a time, keep only rows whose
    `id_column` is in `cohort_ids`, and write a slim parquet.

    Memory is bounded by row-group size, NOT by total source size — this is
    why the same code works on UCMC (3 GB labs) and UMN (60+ GB labs).

    Returns a small stats dict: {n_rows_in, n_rows_out, n_columns_in,
    n_columns_out, source_path, dest_path, dest_size_bytes}.
    """
    if not source_path.exists():
        raise FileNotFoundError(f"source parquet not found: {source_path}")

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(str(source_path))
    schema = pf.schema_arrow

    # Resolve column projection: if columns_to_keep was supplied, intersect
    # with what's actually in the schema. Always include id_column.
    if columns_to_keep is None:
        cols_proj = None
    else:
        wanted = set(columns_to_keep) | {id_column}
        present = [c for c in schema.names if c in wanted]
        cols_proj = present if len(present) > 0 else None

    if id_column not in schema.names:
        raise KeyError(
            f"id_column {id_column!r} not in {source_path.name} schema. "
            f"Available: {schema.names[:10]}{'...' if len(schema.names) > 10 else ''}"
        )

    # Build cohort_arr typed to match the source column. Without an explicit
    # type, an empty Python list collapses to `null` dtype and `is_in` errors
    # with "Array type doesn't match type of values set: string vs null".
    id_dtype = schema.field(id_column).type
    cohort_arr = pa.array(list(cohort_ids), type=id_dtype)

    n_rows_in = 0
    n_rows_out = 0
    writer: Optional[pq.ParquetWriter] = None

    try:
        for rg_idx in range(pf.num_row_groups):
            rg = pf.read_row_group(rg_idx, columns=cols_proj)
            n_rows_in += rg.num_rows
            mask = pc.is_in(rg[id_column], value_set=cohort_arr)
            kept = rg.filter(mask)
            if kept.num_rows == 0:
                continue
            n_rows_out += kept.num_rows
            if writer is None:
                writer = pq.ParquetWriter(str(dest_path), kept.schema, compression=compression)
            writer.write_table(kept)
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        # No rows matched; create an empty parquet with the schema so the
        # downstream pipeline can still scan_parquet without erroring.
        empty = (
            pa.table([pa.array([], type=schema.field(c).type) for c in (cols_proj or schema.names)],
                     names=(cols_proj or schema.names))
        )
        pq.write_table(empty, str(dest_path), compression=compression)

    return {
        "n_rows_in":      n_rows_in,
        "n_rows_out":     n_rows_out,
        "n_columns_in":   len(schema.names),
        "n_columns_out":  len(cols_proj) if cols_proj is not None else len(schema.names),
        "source_path":    str(source_path),
        "dest_path":      str(dest_path),
        "dest_size_bytes": dest_path.stat().st_size if dest_path.exists() else 0,
    }


# ---------------------------------------------------------------------------
# Orchestrator (filter all spec'd tables + write manifest)
# ---------------------------------------------------------------------------

def filter_all_clif_tables_to_cohort(
    source_dir: Path,
    dest_dir: Path,
    cohort_ids: set,
    cohort_hash: str,
    *,
    spec: Iterable[TableFilterSpec] = DEFAULT_FILTER_SPEC,
    skip_missing: bool = True,
    note: str = "",
) -> Manifest:
    """Filter every spec'd CLIF table to cohort-only and write the manifest.

    Args:
        source_dir: where the raw CLIF parquets live (the `tables_path` from config).
        dest_dir:   where to write filtered parquets + manifest. Typically
                    `output/intermediate/clif_filtered/`.
        cohort_ids: hospitalization_ids in the cohort.
        cohort_hash: stable hash from `hash_cohort_definition()` so the
                    manifest can detect cohort-logic changes.
        spec:       which tables to filter. Defaults to DEFAULT_FILTER_SPEC.
        skip_missing: if True, sources not present on disk are silently
                    skipped (e.g. UCMC has no microbiology_susceptibility).
        note:       free-form note saved in the manifest.

    Returns: the freshly-written Manifest.
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    source_files_recorded: dict[str, dict] = {}

    for s in spec:
        src = source_dir / s.source_filename
        if not src.exists():
            if skip_missing:
                continue
            raise FileNotFoundError(f"required source missing: {src}")

        dst = dest_dir / s.dest_filename
        filter_table_to_cohort(
            source_path=src,
            dest_path=dst,
            cohort_ids=cohort_ids,
            id_column=s.id_column,
            columns_to_keep=s.columns_to_keep,
        )
        source_files_recorded[s.source_filename] = _file_signature(src)

    from datetime import datetime, timezone
    manifest = Manifest(
        source_dir=str(source_dir),
        cohort_hash=cohort_hash,
        source_files=source_files_recorded,
        produced_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n_cohort_ids=len(cohort_ids),
        note=note,
    )
    write_manifest(manifest, dest_dir)
    return manifest


def compute_or_use_cached_filtered_tables(
    source_dir: Path,
    dest_dir: Path,
    cohort_hash: str,
    *,
    spec: Iterable[TableFilterSpec] = DEFAULT_FILTER_SPEC,
    cohort_ids_callable=None,  # () -> set, called only on cache miss
    force_refresh: bool = False,
) -> tuple[Manifest, bool]:
    """High-level entry: hit cache or recompute filtered tables.

    Args:
        source_dir, dest_dir, cohort_hash, spec: see filter_all_clif_tables_to_cohort.
        cohort_ids_callable: zero-arg function that returns the cohort id set.
                             Called ONLY when cache misses. This lets you
                             pass in a closure that does the (potentially
                             expensive) cohort definition without paying the
                             cost on cache hits.
        force_refresh: if True, ignore cache and rebuild.

    Returns: (manifest, cache_was_hit). `manifest` is the active manifest
             (either newly written or the cached one). `cache_was_hit` is
             True iff we used the cache without recomputing.

    INTEGRATION INTO main():

        from modules.tableone.cohort_filter import (
            compute_or_use_cached_filtered_tables, hash_cohort_definition,
        )

        # Compute cohort definition first (cheap-table joins)
        # ... cohort_hosp_ids = compute_cohort_from_cheap_tables(...)

        cohort_hash = hash_cohort_definition(
            version="2026.05.cohort.v1",
            params={"age_threshold": 18, "year_range": (2018, 2024),
                    "hfnc_lpm_threshold": 30, ...},
        )

        manifest, cached = compute_or_use_cached_filtered_tables(
            source_dir=Path(config["tables_path"]),
            dest_dir=Path("output/intermediate/clif_filtered"),
            cohort_hash=cohort_hash,
            cohort_ids_callable=lambda: set(cohort_hosp_ids),
            force_refresh=args.force_refresh,
        )

        if cached:
            print(f"✅ Filtered CLIF cache hit ({manifest.produced_at}) "
                  f"— reusing {len(manifest.source_files)} tables")
        else:
            print(f"✅ Filtered CLIF tables built — {len(manifest.source_files)} tables, "
                  f"cohort N={manifest.n_cohort_ids:,}")

        # From here, the rest of main() reads the slim parquets:
        #   labs = pd.read_parquet("output/intermediate/clif_filtered/labs_cohort.parquet")
        # ... etc.
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)

    expected = [s.source_filename for s in spec]

    if not force_refresh:
        manifest = read_manifest(dest_dir)
        if is_cache_valid(manifest, source_dir, cohort_hash, expected):
            return manifest, True  # cache hit — nothing to do

    if cohort_ids_callable is None:
        raise ValueError(
            "cohort_ids_callable must be provided when cache misses"
        )

    cohort_ids = set(cohort_ids_callable())
    new_manifest = filter_all_clif_tables_to_cohort(
        source_dir=source_dir,
        dest_dir=dest_dir,
        cohort_ids=cohort_ids,
        cohort_hash=cohort_hash,
        spec=spec,
    )
    return new_manifest, False
