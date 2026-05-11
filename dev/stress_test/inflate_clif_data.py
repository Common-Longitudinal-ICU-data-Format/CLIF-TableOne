"""
Row-multiplies a CLIF parquet dataset by a user-specified factor, with consistent
patient_id / hospitalization_id suffixing across every table so referential
integrity is preserved (FK joins still resolve, but every copy is a distinct cohort).

Usage:
    uv run python dev/stress_test/inflate_clif_data.py --src /path/to/2.1.0 --dst /path/to/2.1.0_x3 --factor 3

Design notes:
    - Iterates row groups to bound peak memory independently of source table size.
    - Copy 0 emits rows unchanged; copies 1..N-1 append suffix "__c{i}" to every
      table's patient_id and hospitalization_id columns (whichever are present).
    - Parquet column compression will crush near-identical rows, so the on-disk
      size will be <<N*original even at large factors. In-RAM size scales linearly
      with row count, which is what drives pipeline memory pressure.
"""

import argparse
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

ID_COLUMNS = ("patient_id", "hospitalization_id")
INT_OFFSET_PER_COPY = 10**12  # keeps copy ID ranges disjoint from real ones


def rewrite_ids(table: pa.Table, copy_idx: int) -> pa.Table:
    """Offset every ID column present on the table so copy_idx rows are distinct.

    Strings: append '__c{copy_idx}'. Ints: add copy_idx * 10^12.
    Unexpected dtypes are left alone (with a stderr warning per copy).
    """
    suffix = f"__c{copy_idx}"
    int_offset = copy_idx * INT_OFFSET_PER_COPY
    for col_name in ID_COLUMNS:
        if col_name not in table.column_names:
            continue
        idx = table.schema.get_field_index(col_name)
        col = table.column(col_name)
        col_type = col.type
        if pa.types.is_string(col_type) or pa.types.is_large_string(col_type):
            suffix_arr = pa.array([suffix] * len(col), type=col_type)
            empty_sep = pa.scalar("", type=col_type)
            new_col = pc.binary_join_element_wise(col, suffix_arr, empty_sep)
            if new_col.type != col_type:
                new_col = new_col.cast(col_type)
        elif pa.types.is_integer(col_type):
            new_col = pc.add(col, pa.scalar(int_offset, type=col_type))
        else:
            print(f"  warn: {col_name} has unhandled dtype {col_type}; leaving duplicate IDs", file=sys.stderr)
            continue
        table = table.set_column(idx, col_name, new_col)
    return table


def inflate_one_table(src_path: Path, dst_path: Path, factor: int) -> dict:
    """Copy src_path to dst_path, duplicating rows `factor` times with ID suffixing."""
    t0 = time.time()
    reader = pq.ParquetFile(src_path)
    schema = reader.schema_arrow
    has_ids = any(c in schema.names for c in ID_COLUMNS)

    rows_in = reader.metadata.num_rows
    rows_out = 0

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(dst_path, schema, compression="snappy")
    try:
        for rg_idx in range(reader.num_row_groups):
            batch = reader.read_row_group(rg_idx)
            for copy_idx in range(factor):
                if copy_idx == 0 or not has_ids:
                    out = batch
                else:
                    out = rewrite_ids(batch, copy_idx)
                writer.write_table(out)
                rows_out += len(out)
    finally:
        writer.close()

    bytes_in = src_path.stat().st_size
    bytes_out = dst_path.stat().st_size
    elapsed = time.time() - t0
    return {
        "table": src_path.name,
        "rows_in": rows_in,
        "rows_out": rows_out,
        "bytes_in_mb": bytes_in / 1_048_576,
        "bytes_out_mb": bytes_out / 1_048_576,
        "elapsed_sec": elapsed,
        "has_ids": has_ids,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True, type=Path, help="Source CLIF parquet directory")
    ap.add_argument("--dst", required=True, type=Path, help="Destination directory for inflated parquets")
    ap.add_argument("--factor", required=True, type=int, help="Row multiplication factor (>=1)")
    args = ap.parse_args()

    if args.factor < 1:
        print("error: --factor must be >= 1", file=sys.stderr)
        return 2
    if not args.src.is_dir():
        print(f"error: --src not a directory: {args.src}", file=sys.stderr)
        return 2
    if args.dst.exists() and any(args.dst.iterdir()):
        print(f"error: --dst exists and is non-empty: {args.dst}", file=sys.stderr)
        return 2
    args.dst.mkdir(parents=True, exist_ok=True)

    parquets = sorted(args.src.glob("*.parquet"))
    if not parquets:
        print(f"error: no *.parquet files in {args.src}", file=sys.stderr)
        return 2

    print(f"Inflating {len(parquets)} tables from {args.src} -> {args.dst} at factor {args.factor}")
    print(f"{'table':<45} {'rows_in':>12} {'rows_out':>12} {'MB_in':>8} {'MB_out':>8} {'sec':>6}  ids")
    print("-" * 110)

    results = []
    total_t0 = time.time()
    for src_path in parquets:
        dst_path = args.dst / src_path.name
        try:
            r = inflate_one_table(src_path, dst_path, args.factor)
        except Exception as e:
            print(f"{src_path.name:<45} ERROR: {e}")
            continue
        results.append(r)
        print(
            f"{r['table']:<45} {r['rows_in']:>12,} {r['rows_out']:>12,} "
            f"{r['bytes_in_mb']:>8.1f} {r['bytes_out_mb']:>8.1f} {r['elapsed_sec']:>6.1f}  "
            f"{'y' if r['has_ids'] else '-'}"
        )

    total_in = sum(r["bytes_in_mb"] for r in results)
    total_out = sum(r["bytes_out_mb"] for r in results)
    total_sec = time.time() - total_t0
    print("-" * 110)
    print(f"{'TOTAL':<45} {'':>12} {'':>12} {total_in:>8.1f} {total_out:>8.1f} {total_sec:>6.1f}")
    print(f"\nDone in {total_sec:.1f}s. Point config/config.json tables_path at: {args.dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
