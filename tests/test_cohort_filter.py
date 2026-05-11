"""Unit tests for modules.tableone.cohort_filter.

Covers the cache-manifest layer (hash, write/read, is_cache_valid) plus the
streaming row-group filter (filter_table_to_cohort) and the orchestrator.

Run with:
    .venv/bin/python -m pytest tests/test_cohort_filter.py -v
Or standalone:
    .venv/bin/python tests/test_cohort_filter.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# Allow `python tests/test_cohort_filter.py` without PYTHONPATH gymnastics.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from modules.tableone.cohort_filter import (
    MANIFEST_FILENAME,
    Manifest,
    TableFilterSpec,
    compute_or_use_cached_filtered_tables,
    filter_all_clif_tables_to_cohort,
    filter_table_to_cohort,
    hash_cohort_definition,
    is_cache_valid,
    read_manifest,
    write_manifest,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_parquet(path: Path, hospitalization_ids, **extra_columns):
    """Build a tiny parquet with `hospitalization_id` + arbitrary extras."""
    cols = {"hospitalization_id": pa.array(hospitalization_ids, type=pa.string())}
    for k, v in extra_columns.items():
        cols[k] = pa.array(v)
    table = pa.table(cols)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))


def _make_clif_source(source_dir: Path, n_per_table: int = 50) -> set:
    """Create a tiny synthetic CLIF source directory with 3 tables.
    Returns the set of all hospitalization_ids in the source.
    """
    ids = [f"H{i:05d}" for i in range(n_per_table)]
    _write_parquet(source_dir / "clif_labs.parquet",
                   ids * 3,  # 3 lab rows per encounter
                   lab_value=[float(i) for i in range(n_per_table * 3)])
    _write_parquet(source_dir / "clif_vitals.parquet",
                   ids * 5,  # 5 vitals rows per encounter
                   vital_value=[float(i) for i in range(n_per_table * 5)])
    _write_parquet(source_dir / "clif_crrt_therapy.parquet",
                   ids[:5],  # only 5 encounters had crrt
                   crrt_mode_category=["cvvh"] * 5)
    return set(ids)


# ---------------------------------------------------------------------------
# hash_cohort_definition
# ---------------------------------------------------------------------------

class TestHashCohortDefinition:
    def test_stable_across_calls(self):
        h1 = hash_cohort_definition("v1", {"age": 18})
        h2 = hash_cohort_definition("v1", {"age": 18})
        assert h1 == h2

    def test_changes_with_version(self):
        h1 = hash_cohort_definition("v1", {"age": 18})
        h2 = hash_cohort_definition("v2", {"age": 18})
        assert h1 != h2

    def test_changes_with_params(self):
        h1 = hash_cohort_definition("v1", {"age": 18})
        h2 = hash_cohort_definition("v1", {"age": 21})
        assert h1 != h2

    def test_param_order_doesnt_matter(self):
        h1 = hash_cohort_definition("v1", {"a": 1, "b": 2})
        h2 = hash_cohort_definition("v1", {"b": 2, "a": 1})
        assert h1 == h2

    def test_returns_short_hex(self):
        h = hash_cohort_definition("v1")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_handles_no_params(self):
        h1 = hash_cohort_definition("v1")
        h2 = hash_cohort_definition("v1", None)
        h3 = hash_cohort_definition("v1", {})
        assert h1 == h2 == h3


# ---------------------------------------------------------------------------
# Manifest read/write roundtrip
# ---------------------------------------------------------------------------

class TestManifestRoundtrip:
    def test_write_then_read_returns_same(self, tmp_path):
        m = Manifest(
            source_dir="/some/path",
            cohort_hash="abc123",
            source_files={
                "clif_labs.parquet": {"mtime": 1730412345.0, "size_bytes": 4823491200},
            },
            produced_at="2026-04-30T15:42:01+00:00",
            n_cohort_ids=49596,
            note="test",
        )
        write_manifest(m, tmp_path)
        assert (tmp_path / MANIFEST_FILENAME).exists()

        m2 = read_manifest(tmp_path)
        assert m2 is not None
        assert m2.source_dir == m.source_dir
        assert m2.cohort_hash == m.cohort_hash
        assert m2.n_cohort_ids == m.n_cohort_ids
        assert m2.source_files == m.source_files

    def test_read_returns_none_when_missing(self, tmp_path):
        assert read_manifest(tmp_path) is None

    def test_read_returns_none_on_invalid_json(self, tmp_path):
        (tmp_path / MANIFEST_FILENAME).write_text("{ invalid json !!!", encoding="utf-8")
        assert read_manifest(tmp_path) is None


# ---------------------------------------------------------------------------
# is_cache_valid
# ---------------------------------------------------------------------------

class TestIsCacheValid:
    def test_returns_false_when_manifest_none(self, tmp_path):
        assert not is_cache_valid(None, tmp_path, "h", ["x"])

    def test_returns_false_when_cohort_hash_changed(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.parquet").write_bytes(b"x" * 100)

        m = Manifest(
            source_dir=str(src),
            cohort_hash="OLD_HASH",
            source_files={"a.parquet": {"mtime": (src / "a.parquet").stat().st_mtime,
                                         "size_bytes": 100}},
        )
        assert not is_cache_valid(m, src, "NEW_HASH", ["a.parquet"])

    def test_returns_false_when_source_dir_changed(self, tmp_path):
        src1 = tmp_path / "src1"; src1.mkdir()
        src2 = tmp_path / "src2"; src2.mkdir()
        (src2 / "a.parquet").write_bytes(b"x" * 100)

        m = Manifest(
            source_dir=str(src1),
            cohort_hash="h",
            source_files={"a.parquet": {"mtime": (src2 / "a.parquet").stat().st_mtime,
                                         "size_bytes": 100}},
        )
        assert not is_cache_valid(m, src2, "h", ["a.parquet"])

    def test_returns_false_when_file_size_changed(self, tmp_path):
        src = tmp_path / "src"; src.mkdir()
        f = src / "a.parquet"
        f.write_bytes(b"x" * 100)

        m = Manifest(
            source_dir=str(src),
            cohort_hash="h",
            source_files={"a.parquet": {"mtime": f.stat().st_mtime,
                                         "size_bytes": 50}},  # WRONG size
        )
        assert not is_cache_valid(m, src, "h", ["a.parquet"])

    def test_returns_false_when_file_mtime_changed(self, tmp_path):
        src = tmp_path / "src"; src.mkdir()
        f = src / "a.parquet"
        f.write_bytes(b"x" * 100)

        m = Manifest(
            source_dir=str(src),
            cohort_hash="h",
            source_files={"a.parquet": {"mtime": f.stat().st_mtime - 100,  # OLD mtime
                                         "size_bytes": 100}},
        )
        assert not is_cache_valid(m, src, "h", ["a.parquet"])

    def test_returns_false_when_expected_file_missing(self, tmp_path):
        src = tmp_path / "src"; src.mkdir()
        m = Manifest(
            source_dir=str(src),
            cohort_hash="h",
            source_files={"a.parquet": {"mtime": 1.0, "size_bytes": 100}},
        )
        # File was in manifest but doesn't exist on disk now.
        assert not is_cache_valid(m, src, "h", ["a.parquet"])

    def test_returns_true_when_everything_matches(self, tmp_path):
        src = tmp_path / "src"; src.mkdir()
        f = src / "a.parquet"
        f.write_bytes(b"x" * 100)

        m = Manifest(
            source_dir=str(src),
            cohort_hash="h",
            source_files={"a.parquet": {"mtime": f.stat().st_mtime,
                                         "size_bytes": 100}},
        )
        assert is_cache_valid(m, src, "h", ["a.parquet"])

    def test_handles_multiple_files(self, tmp_path):
        src = tmp_path / "src"; src.mkdir()
        for name in ("a.parquet", "b.parquet", "c.parquet"):
            (src / name).write_bytes(b"x" * 100)

        m = Manifest(
            source_dir=str(src),
            cohort_hash="h",
            source_files={
                name: {"mtime": (src / name).stat().st_mtime, "size_bytes": 100}
                for name in ("a.parquet", "b.parquet", "c.parquet")
            },
        )
        assert is_cache_valid(m, src, "h", ["a.parquet", "b.parquet", "c.parquet"])


# ---------------------------------------------------------------------------
# filter_table_to_cohort
# ---------------------------------------------------------------------------

class TestFilterTableToCohort:
    def test_keeps_only_cohort_ids(self, tmp_path):
        src = tmp_path / "src.parquet"
        ids = [f"H{i}" for i in range(10)]
        _write_parquet(src, ids, value=list(range(10)))

        cohort = {"H1", "H3", "H5", "H7"}
        dst = tmp_path / "out.parquet"
        stats = filter_table_to_cohort(src, dst, cohort)

        out = pq.read_table(str(dst)).to_pandas()
        assert set(out["hospitalization_id"]) == cohort
        assert stats["n_rows_in"] == 10
        assert stats["n_rows_out"] == 4

    def test_dest_parquet_is_smaller_than_source(self, tmp_path):
        src = tmp_path / "src.parquet"
        ids = [f"H{i}" for i in range(1000)]
        _write_parquet(src, ids, value=list(range(1000)))

        cohort = {f"H{i}" for i in range(50)}  # 5%
        dst = tmp_path / "out.parquet"
        filter_table_to_cohort(src, dst, cohort)

        # Filtered file should be substantially smaller (allow some compression overhead)
        assert dst.stat().st_size < src.stat().st_size

    def test_empty_cohort_writes_empty_parquet(self, tmp_path):
        src = tmp_path / "src.parquet"
        ids = [f"H{i}" for i in range(10)]
        _write_parquet(src, ids, value=list(range(10)))

        dst = tmp_path / "out.parquet"
        stats = filter_table_to_cohort(src, dst, set())

        assert dst.exists()
        out = pq.read_table(str(dst)).to_pandas()
        assert len(out) == 0
        assert stats["n_rows_out"] == 0
        # Schema should be preserved so downstream scan_parquet works
        assert "hospitalization_id" in out.columns

    def test_column_projection(self, tmp_path):
        src = tmp_path / "src.parquet"
        ids = [f"H{i}" for i in range(10)]
        _write_parquet(src, ids,
                       lab_category=["albumin"] * 10,
                       lab_value=list(range(10)),
                       reference_unit=["g/dl"] * 10)

        cohort = {"H1", "H2", "H3"}
        dst = tmp_path / "out.parquet"
        filter_table_to_cohort(
            src, dst, cohort,
            columns_to_keep=["lab_category", "lab_value"],  # drop reference_unit
        )

        out_schema = pq.read_schema(str(dst))
        assert "hospitalization_id" in out_schema.names
        assert "lab_category" in out_schema.names
        assert "lab_value" in out_schema.names
        assert "reference_unit" not in out_schema.names  # projected away

    def test_id_column_always_kept_even_if_not_in_projection(self, tmp_path):
        src = tmp_path / "src.parquet"
        ids = [f"H{i}" for i in range(10)]
        _write_parquet(src, ids, lab_value=list(range(10)))

        dst = tmp_path / "out.parquet"
        # User asks for lab_value only — id_column must still be retained
        # so the filter can run AND so downstream code that joins on the id
        # still works.
        filter_table_to_cohort(
            src, dst, {"H1"},
            columns_to_keep=["lab_value"],
        )
        out_schema = pq.read_schema(str(dst))
        assert "hospitalization_id" in out_schema.names

    def test_raises_when_id_column_missing(self, tmp_path):
        src = tmp_path / "src.parquet"
        # No hospitalization_id column at all
        table = pa.table({"some_other_id": pa.array(["a", "b"]), "v": pa.array([1, 2])})
        pq.write_table(table, str(src))

        dst = tmp_path / "out.parquet"
        with pytest.raises(KeyError, match="hospitalization_id"):
            filter_table_to_cohort(src, dst, {"a"})

    def test_raises_when_source_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            filter_table_to_cohort(
                tmp_path / "no_such.parquet",
                tmp_path / "out.parquet",
                {"a"},
            )


# ---------------------------------------------------------------------------
# filter_all_clif_tables_to_cohort
# ---------------------------------------------------------------------------

class TestFilterAllClifTablesToCohort:
    def test_filters_all_spec_tables_and_writes_manifest(self, tmp_path):
        src = tmp_path / "src"
        all_ids = _make_clif_source(src)
        cohort = set(list(all_ids)[:10])  # first 10 encounters

        dst = tmp_path / "dst"
        spec = (
            TableFilterSpec("clif_labs.parquet",         "labs_cohort.parquet"),
            TableFilterSpec("clif_vitals.parquet",       "vitals_cohort.parquet"),
            TableFilterSpec("clif_crrt_therapy.parquet", "crrt_therapy_cohort.parquet"),
        )
        manifest = filter_all_clif_tables_to_cohort(
            source_dir=src, dest_dir=dst, cohort_ids=cohort,
            cohort_hash="testhash", spec=spec,
        )

        # All three filtered files exist
        assert (dst / "labs_cohort.parquet").exists()
        assert (dst / "vitals_cohort.parquet").exists()
        assert (dst / "crrt_therapy_cohort.parquet").exists()

        # Manifest written and parseable
        m2 = read_manifest(dst)
        assert m2 is not None
        assert m2.cohort_hash == "testhash"
        assert m2.n_cohort_ids == 10
        assert set(m2.source_files.keys()) == {
            "clif_labs.parquet", "clif_vitals.parquet", "clif_crrt_therapy.parquet"
        }

        # Filtered tables actually contain only cohort encounters
        labs = pq.read_table(str(dst / "labs_cohort.parquet")).to_pandas()
        assert set(labs["hospitalization_id"]) <= cohort

    def test_skips_missing_source_when_skip_missing_true(self, tmp_path):
        src = tmp_path / "src"
        _make_clif_source(src)

        dst = tmp_path / "dst"
        spec = (
            TableFilterSpec("clif_labs.parquet",                  "labs_cohort.parquet"),
            TableFilterSpec("clif_does_not_exist.parquet",        "ghost_cohort.parquet"),
        )
        manifest = filter_all_clif_tables_to_cohort(
            source_dir=src, dest_dir=dst,
            cohort_ids={"H00001"}, cohort_hash="h",
            spec=spec, skip_missing=True,
        )
        # Only the labs table got filtered + recorded
        assert "clif_labs.parquet" in manifest.source_files
        assert "clif_does_not_exist.parquet" not in manifest.source_files
        assert (dst / "labs_cohort.parquet").exists()
        assert not (dst / "ghost_cohort.parquet").exists()

    def test_raises_on_missing_source_when_skip_missing_false(self, tmp_path):
        src = tmp_path / "src"
        _make_clif_source(src)

        dst = tmp_path / "dst"
        spec = (TableFilterSpec("clif_does_not_exist.parquet", "ghost.parquet"),)
        with pytest.raises(FileNotFoundError):
            filter_all_clif_tables_to_cohort(
                source_dir=src, dest_dir=dst,
                cohort_ids={"H00001"}, cohort_hash="h",
                spec=spec, skip_missing=False,
            )


# ---------------------------------------------------------------------------
# compute_or_use_cached_filtered_tables (cache-hit / cache-miss)
# ---------------------------------------------------------------------------

class TestComputeOrUseCached:
    def _spec(self):
        return (
            TableFilterSpec("clif_labs.parquet",   "labs_cohort.parquet"),
            TableFilterSpec("clif_vitals.parquet", "vitals_cohort.parquet"),
        )

    def test_first_call_misses_cache_and_writes_files(self, tmp_path):
        src = tmp_path / "src"
        _make_clif_source(src)
        dst = tmp_path / "dst"

        manifest, cached = compute_or_use_cached_filtered_tables(
            source_dir=src, dest_dir=dst,
            cohort_hash="h",
            spec=self._spec(),
            cohort_ids_callable=lambda: {"H00001", "H00002"},
        )
        assert cached is False
        assert (dst / "labs_cohort.parquet").exists()
        assert (dst / "vitals_cohort.parquet").exists()
        assert manifest.cohort_hash == "h"

    def test_second_call_hits_cache_without_recomputing(self, tmp_path):
        src = tmp_path / "src"
        _make_clif_source(src)
        dst = tmp_path / "dst"

        # First call: computes
        compute_or_use_cached_filtered_tables(
            source_dir=src, dest_dir=dst, cohort_hash="h",
            spec=self._spec(),
            cohort_ids_callable=lambda: {"H00001", "H00002"},
        )

        # Second call: should NOT call the callable
        callable_invocations = []
        def boom():
            callable_invocations.append(1)
            return {"H00001"}  # different cohort — but we shouldn't get here

        m2, cached = compute_or_use_cached_filtered_tables(
            source_dir=src, dest_dir=dst, cohort_hash="h",
            spec=self._spec(),
            cohort_ids_callable=boom,
        )
        assert cached is True
        assert callable_invocations == []  # callable was NOT invoked

    def test_cohort_hash_change_invalidates_cache(self, tmp_path):
        src = tmp_path / "src"
        _make_clif_source(src)
        dst = tmp_path / "dst"

        compute_or_use_cached_filtered_tables(
            source_dir=src, dest_dir=dst, cohort_hash="OLD",
            spec=self._spec(),
            cohort_ids_callable=lambda: {"H00001"},
        )

        m2, cached = compute_or_use_cached_filtered_tables(
            source_dir=src, dest_dir=dst, cohort_hash="NEW",  # changed!
            spec=self._spec(),
            cohort_ids_callable=lambda: {"H00001", "H00002"},
        )
        assert cached is False
        assert m2.cohort_hash == "NEW"

    def test_force_refresh_bypasses_cache(self, tmp_path):
        src = tmp_path / "src"
        _make_clif_source(src)
        dst = tmp_path / "dst"

        compute_or_use_cached_filtered_tables(
            source_dir=src, dest_dir=dst, cohort_hash="h",
            spec=self._spec(),
            cohort_ids_callable=lambda: {"H00001"},
        )

        m2, cached = compute_or_use_cached_filtered_tables(
            source_dir=src, dest_dir=dst, cohort_hash="h",
            spec=self._spec(),
            cohort_ids_callable=lambda: {"H00001", "H00002"},
            force_refresh=True,  # ignore cache
        )
        assert cached is False

    def test_source_file_change_invalidates_cache(self, tmp_path):
        src = tmp_path / "src"
        _make_clif_source(src)
        dst = tmp_path / "dst"

        compute_or_use_cached_filtered_tables(
            source_dir=src, dest_dir=dst, cohort_hash="h",
            spec=self._spec(),
            cohort_ids_callable=lambda: {"H00001"},
        )

        # Touch a source file to change its mtime + content
        time.sleep(0.05)  # ensure mtime tick
        ids = [f"H{i:05d}" for i in range(60)]  # different size now
        _write_parquet(src / "clif_labs.parquet", ids,
                       lab_value=[float(i) for i in range(60)])

        called_again = []
        m2, cached = compute_or_use_cached_filtered_tables(
            source_dir=src, dest_dir=dst, cohort_hash="h",
            spec=self._spec(),
            cohort_ids_callable=lambda: (called_again.append(1) or {"H00001"}),
        )
        assert cached is False
        assert called_again == [1]  # callable WAS invoked

    def test_missing_callable_on_cache_miss_raises(self, tmp_path):
        src = tmp_path / "src"
        _make_clif_source(src)
        dst = tmp_path / "dst"

        with pytest.raises(ValueError, match="cohort_ids_callable"):
            compute_or_use_cached_filtered_tables(
                source_dir=src, dest_dir=dst, cohort_hash="h",
                spec=self._spec(),
                cohort_ids_callable=None,  # no callable, but cache misses
            )


# ---------------------------------------------------------------------------
# Sanity: real-world-ish row group streaming
# ---------------------------------------------------------------------------

class TestStreaming:
    def test_multiple_row_groups_handled_correctly(self, tmp_path):
        """Confirm filtering across explicit multi-row-group parquet works."""
        src = tmp_path / "src.parquet"

        # Build a multi-row-group parquet by writing in chunks
        ids_total = []
        with pq.ParquetWriter(
            str(src),
            schema=pa.schema([("hospitalization_id", pa.string()), ("v", pa.int64())]),
        ) as writer:
            for chunk in range(5):
                ids_chunk = [f"H{chunk}_{i:03d}" for i in range(100)]
                ids_total.extend(ids_chunk)
                t = pa.table({
                    "hospitalization_id": pa.array(ids_chunk),
                    "v": pa.array(list(range(100))),
                })
                writer.write_table(t)

        # Cohort: pick across multiple row groups
        cohort = {"H0_010", "H2_050", "H4_099"}

        dst = tmp_path / "out.parquet"
        stats = filter_table_to_cohort(src, dst, cohort)
        assert stats["n_rows_out"] == 3

        out = pq.read_table(str(dst)).to_pandas()
        assert set(out["hospitalization_id"]) == cohort


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
