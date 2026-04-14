# Data Engine Guide: Why DuckDB, Not Polars

A plain-language explanation of how this pipeline loads data, why it matters on
different hardware, and where we're headed.

---

## Table of Contents

1. [The 60-Second Version](#the-60-second-version)
2. [Hardware Basics You Need](#hardware-basics-you-need)
3. [How Data Engines Work](#how-data-engines-work)
4. [DuckDB vs Polars — What's Actually Different](#duckdb-vs-polars--whats-actually-different)
5. [How This Pipeline Loads Data Today](#how-this-pipeline-loads-data-today)
6. [The Five Stress-Test Scenarios](#the-five-stress-test-scenarios)
7. [The Fix: Unified DuckDB Loading](#the-fix-unified-duckdb-loading)
8. [Glossary](#glossary)

---

## The 60-Second Version

Your computer has two kinds of storage that matter here:

- **RAM** (fast, small, expensive) — where data lives while being processed
- **Disk** (slow, large, cheap) — where parquet files sit

When you run the pipeline, it reads parquet files from disk into RAM so it can
do math on them. If the data is bigger than available RAM, the program crashes
("OOM" — out of memory).

**DuckDB** is a data engine that's smart about this — it only loads pieces of
data at a time, and when it needs more room, it temporarily puts data back on
disk. Think of it like a small desk: you pull out one folder, work on it, put
it back, pull out the next.

**Polars** is faster when everything fits in RAM, but it tries to keep
everything on the desk at once. When the desk is too small, things fall off
(crash).

**Our pipeline currently uses both** — in different places, loading the same
files multiple times, with no fallback when polars isn't available. That's
the problem we're fixing.

---

## Hardware Basics You Need

### RAM (Random Access Memory)

Think of RAM as your desk — it's where you spread out papers you're actively
working on.

| Fact | Detail |
|------|--------|
| What it is | Short-term memory your computer uses while running programs |
| Typical size | 8 GB, 16 GB, 32 GB, 64 GB |
| Speed | Very fast — ~50 GB/sec read speed |
| What happens when it fills up | Programs crash (OOM) or the OS starts using disk as fake RAM (swap), which is 100x slower |
| Cost | ~$3-5 per GB |

**Key insight:** When you open a 5 GB parquet file, the data has to fit in RAM
to be processed. But your OS, Python itself, and other programs are already
using 3-6 GB. So on a 16 GB machine, you really only have ~10-12 GB available
for data.

### Disk (SSD / Hard Drive)

Think of disk as a filing cabinet — lots of space, but you have to get up and
walk to it every time you need something.

| Fact | Detail |
|------|--------|
| What it is | Permanent storage where your files live |
| Typical size | 256 GB to 2 TB |
| Speed | SSD: ~3 GB/sec, HDD: ~0.1 GB/sec (15-50x slower than RAM) |
| What happens when it fills up | You can't save more files |

### The RAM Problem Visualized

```
16 GB RAM on your machine:
┌──────────────────────────────────────────────────────┐
│ OS + background apps           ░░░░░ (~4 GB)         │
│ Python interpreter + libraries ░░ (~1-2 GB)          │
│ ─── available for your data ─────────────────────    │
│ YOUR DATA GOES HERE            ░░░░░░░░ (~10-12 GB)  │
└──────────────────────────────────────────────────────┘

5 GB parquet file:
  → Loads into ~5-8 GB of RAM (parquet is compressed on disk)
  → Joins/filters may create temporary copies → peak ~10-15 GB
  → Fits on 16 GB, but barely. One bad allocation = crash.

20 GB parquet file:
  → Loads into ~20-30 GB of RAM
  → Does NOT fit on 16 GB. Period.
  → Unless the engine is smart enough to process it in pieces.
```

### Why Parquet Files Expand in RAM

A parquet file on disk is **compressed** (like a zip file). A 5 GB parquet file
might contain 15-20 GB of actual data. When you load it, it decompresses:

```
Disk:  clif_labs.parquet     5 GB  (compressed)
        ↓ decompress
RAM:   DataFrame in memory   12 GB (uncompressed columns, indexes, metadata)
        ↓ join/filter
RAM:   temporary copy         8 GB (intermediate result during join)
        ↓
Peak RAM usage:              20 GB  ← this is what kills 16 GB machines
```

Smart engines (DuckDB) avoid this by decompressing and processing one chunk at
a time, never holding the full dataset in RAM.

### Windows vs Mac — Why It Matters

| | Mac (Apple Silicon) | Windows |
|--|---------------------|---------|
| **Memory management** | Unified memory (RAM + GPU share), aggressive compression | Traditional split memory, less compression |
| **Swap behavior** | Fast swap on NVMe, OS is aggressive about paging | Slower pagefile, OS is less aggressive |
| **Python memory** | ARM-optimized allocators, generally tighter | x86 allocators tend to fragment more |
| **Polars behavior** | Usually works fine on 16 GB with 5 GB data | More likely to OOM on same data |
| **Net effect** | More forgiving — Mac 16 GB "acts like" 18-20 GB | Less forgiving — Windows 16 GB is a hard 16 GB |

**Why Windows is harder:** Windows' memory allocator is less aggressive about
reclaiming freed memory. When Python frees a large DataFrame, Mac gives that
memory back to the OS quickly. Windows often holds onto it "just in case,"
which means the next allocation sees less available RAM than it should.

---

## How Data Engines Work

A "data engine" is the library that reads your parquet files and lets you
filter, join, and aggregate them. The two engines in this project are **Polars**
and **DuckDB**.

### The Library Analogy

Imagine you need to find every book about "sodium" in a library with
1 million books.

**Polars approach (in-memory):**
1. Bring ALL 1 million books to your desk
2. Flip through each one, keep the sodium books
3. Put the rest back

If your desk is big enough, this is very fast. If your desk is small, books
fall on the floor (crash).

**DuckDB approach (streaming/spilling):**
1. Go to the shelf, pull out 1,000 books at a time
2. Check each batch, keep sodium books in a results pile
3. If the results pile gets too big, put some back on the shelf temporarily
4. Repeat until done

Slower per-book, but you never need a desk bigger than 1,000 books.

### What "Lazy Evaluation" Means

Both engines support "lazy" mode — you describe what you want, and they figure
out the most efficient way to get it:

```python
# This does NOT load data yet — it just builds a plan
plan = pl.scan_parquet("clif_labs.parquet").filter(
    pl.col("lab_category") == "sodium"
)

# THIS is when data actually loads
result = plan.collect()          # polars: load it all
result = plan.collect(streaming=True)  # polars: try to stream chunks
```

DuckDB is lazy by default — every SQL query builds a plan first:

```python
# DuckDB: builds plan AND executes it efficiently in one step
result = con.execute("""
    SELECT * FROM read_parquet('clif_labs.parquet')
    WHERE lab_category = 'sodium'
""")
```

### What "Streaming" Means

Streaming = processing data in chunks rather than all at once.

```
Without streaming (whole file in RAM):
┌──────────────────────────────────────┐
│ ALL 50 million lab rows loaded       │  ← 12 GB in RAM
│ Filter: keep sodium rows             │
│ Result: 2 million sodium rows        │  ← 0.5 GB in RAM
│ Total peak: 12.5 GB                  │
└──────────────────────────────────────┘

With streaming (chunks):
┌──────────────┐
│ Chunk 1: 1M rows  → filter → keep 40K sodium rows │  ← 0.3 GB peak
│ Chunk 2: 1M rows  → filter → keep 38K sodium rows │  ← 0.3 GB peak
│ ...repeat 50 times...                              │
│ Result: 2 million sodium rows                      │  ← 0.5 GB in RAM
│ Total peak: 0.8 GB                                 │
└──────────────┘
```

**DuckDB always streams.** Polars streams only when you ask for it
(`streaming=True`), and even then, some operations (like joins) may fall
back to loading everything.

---

## DuckDB vs Polars — What's Actually Different

### Architecture

```
Polars:
  Python code → Polars (Rust engine, in-process)
                  → reads parquet
                  → processes in RAM
                  → returns Polars DataFrame (lives in RAM)

DuckDB:
  Python code → DuckDB (C++ engine, in-process)
                  → reads parquet
                  → processes with disk spilling
                  → returns Arrow table (lives in RAM, zero-copy to pandas)
```

### Head-to-Head for Our Workload

Our ECDF pipeline does: **scan parquet → filter by category → join with time
windows → collect values → compute ECDF**.

| Operation | Polars | DuckDB | Winner |
|-----------|--------|--------|--------|
| **Scan + filter parquet** | Fast. Predicate pushdown skips irrelevant row groups | Fast. Same predicate pushdown | Tie |
| **Join (data × time windows)** | Streaming join is limited — can materialize both sides in RAM | Hash join with automatic disk spill | **DuckDB** — critical for 20 GB data |
| **Memory release after query** | Python/Rust allocator holds freed pages | Arena allocator returns pages to OS | **DuckDB** — critical for sequential steps |
| **API ergonomics** | Beautiful chained expressions | SQL strings (less Pythonic) | **Polars** |
| **Windows compatibility** | Some users report OOM | Consistent across platforms | **DuckDB** |
| **No-install fallback** | Must be installed (`pip install polars`) | Must be installed (`pip install duckdb`) | Tie |
| **Speed on small data (<2 GB)** | Very fast (Rust, vectorized) | Fast (C++, vectorized) | **Polars** (marginal) |
| **Speed on large data (>RAM)** | Crashes or degrades | Graceful disk spill | **DuckDB** |

### The Join Problem — Why This Matters Most

The most expensive operation in the ECDF generator is the join at
`generator.py:652`:

```python
data_icu = data_category.join(
    icu_windows.lazy(),
    on='hospitalization_id',
    how='inner'
)
```

This joins every lab/vital measurement with ICU time windows. For a large
dataset:

```
clif_labs.parquet (filtered to "sodium"): 2 million rows
icu_windows: 20,000 rows

Join result: 2 million rows (each lab row matched to its ICU window)
```

**Polars:** Builds a hash table from `icu_windows` in RAM, then probes it with
each lab row. If the lab data is large, the probe side must be in RAM too.
With `streaming=True`, polars *tries* to chunk this, but the hash table for the
join key still lives entirely in RAM.

**DuckDB:** Same hash join algorithm, but if the hash table exceeds a memory
budget, DuckDB **partitions it to disk** and processes each partition
separately. This is why DuckDB handles 20 GB data on 16 GB RAM — it
automatically spills to disk when needed.

### Memory Release — The Silent Killer

After processing "sodium", the pipeline moves on to "potassium". The sodium
data should be freed. But:

**Polars/Python:** Python's memory allocator (`pymalloc`) requests large blocks
from the OS and subdivides them. When you free a DataFrame, Python marks the
blocks as available *internally*, but often doesn't return them to the OS.
The OS still thinks Python is using that memory.

```
After processing sodium with polars:
┌─────────────────────────────────────────┐
│ OS sees Python using:    8 GB           │
│ Python internally free:  5 GB           │  ← "available" but OS doesn't know
│ Python actually using:   3 GB           │
│ OS available for Python: 8 GB           │  ← OS thinks only 8 GB left
│ Next allocation of 9 GB: OOM CRASH      │  ← even though 5 GB is "free"
└─────────────────────────────────────────┘
```

**DuckDB:** Uses its own arena allocator. When a query finishes, DuckDB calls
`munmap()` / `VirtualFree()`, which immediately returns pages to the OS.

```
After processing sodium with DuckDB:
┌─────────────────────────────────────────┐
│ OS sees Python using:    3 GB           │  ← DuckDB released its 5 GB
│ OS available for Python: 13 GB          │
│ Next allocation of 9 GB: works fine     │
└─────────────────────────────────────────┘
```

**This compounds across 50+ lab categories processed in sequence.** Each
category leaks a little memory with polars. By category #40, you've
accumulated enough unreturned memory to OOM — even though you only need RAM
for one category at a time.

---

## How This Pipeline Loads Data Today

### The Problem: Three Loaders, No Sharing

```
run_project.py --get-ecdf --ward --visualize

Step 1: Validation
  └─ clif_loader.py → DuckDB → Arrow → pandas
     Loads: labs, adt, vitals, resp_support, hospitalization, ...

     ↓ (Python GC frees — hopefully)

Step 2: Table One (critical-illness)
  └─ tableone/generator.py → clifpy ClifOrchestrator → polars internally
     Loads: labs, adt, vitals, resp_support, hospitalization, ...  ← SAME FILES

     ↓ (Python GC frees — hopefully)

Step 2b: Ward Table One (subprocess)
  └─ Separate Python process
     Loads: labs, adt, vitals, resp_support, hospitalization, ...  ← SAME FILES

     ↓ (subprocess exits, OS reclaims)

Step 3: ECDF Generation
  └─ ecdf/generator.py → raw polars pl.scan_parquet()
     Loads: labs, adt, vitals, resp_support, hospitalization, meds  ← SAME FILES
```

**The same 5 GB of parquet files are read from disk 3-4 times**, through three
different code paths, using two different engines, with different fallback
behavior.

### Why The ECDF Generator Doesn't Use clif_loader.py

Simply: it was built independently. The ECDF module was written to use polars
directly, while `clif_loader.py` was built later to solve Windows OOM issues
in the validation path. Nobody went back to retrofit the ECDF generator.

This means:
- `clif_loader.py` has a DuckDB backend — but only validation uses it
- The ECDF generator hard-imports polars at line 50 — no fallback
- The tableone generator uses clifpy's loader — a third path
- If polars isn't installed, the ECDF generator crashes immediately

### Within the ECDF Generator: Same File Scanned Repeatedly

Even within `generator.py` alone, the same parquet file is opened multiple
times:

```
clif_labs.parquet is opened:
  1. discover_lab_category_units()  → line 196  (scan to find unique categories)
  2. process_category("sodium")     → line 588  (scan + filter to sodium)
  3. process_category("potassium")  → line 588  (scan + filter to potassium)
  4. process_category("glucose")    → line 588  (scan + filter to glucose)
  ... repeated for all ~50 lab categories
  
  Then repeated for EACH stratum (ICU, advanced_resp, nippv_hfnc, vaso, deaths):
  5. process_category("sodium")     → line 588
  6. process_category("potassium")  → line 588
  ... × 5 strata
```

This is ~200+ scans of the same parquet file. Each scan is lazy (polars only
reads the matching rows), so it's not 200x the memory — but it IS 200x the
disk I/O for reading parquet metadata and scanning row groups.

DuckDB handles this better because it caches parquet metadata across queries
within the same connection.

---

## The Five Stress-Test Scenarios

### Scenario 1: Windows 16 GB / 5 GB Data

```
Available RAM: ~10-12 GB (after OS + Python)
Data in RAM:   ~8-15 GB (5 GB parquet decompresses to 2-3x)
```

**Today:** Probably works, but tight. The polars join at `generator.py:652`
is the bottleneck — if a single lab category has millions of rows, the join
intermediate can spike past available RAM. Memory fragmentation across 50+
categories compounds the risk.

**With DuckDB:** Comfortable. Disk spill handles any spike, arena allocator
keeps memory clean between categories.

### Scenario 2: Mac 16 GB / 5 GB Data

**Today:** Works. Mac's unified memory and aggressive swap make this
forgiving. This is your current dev environment and baseline.

**With DuckDB:** Also works, slightly less peak RAM.

### Scenario 3: Windows 16 GB / 20 GB Data

```
Available RAM: ~10-12 GB
Data in RAM:   ~30-50 GB (20 GB parquet decompressed)
```

**Today:** Will crash. 20 GB of data cannot fit in 12 GB of RAM. Polars
streaming helps for simple filter-and-collect, but the join operation needs
to build a hash table — and with 20 GB of data, that hash table exceeds
available memory. No disk spill = OOM.

**With DuckDB:** Works. DuckDB's query engine automatically spills join hash
tables and sort buffers to disk. It will be slower than in-memory (disk I/O),
but it will complete rather than crash.

### Scenario 4: No Polars / 5 GB Data

**Today:** Crashes at `generator.py:50` with:
```
ModuleNotFoundError: No module named 'polars'
```

The ECDF generator hard-imports polars. There is no fallback. Game over.

The validation step works fine (uses DuckDB via `clif_loader.py`). The tableone
step may or may not work depending on clifpy's polars dependency. But the ECDF
step is dead.

**With DuckDB:** Works perfectly — that's the whole point of unifying on DuckDB.

### Scenario 5: No Polars / 20 GB Data

**Today:** Same crash as Scenario 4. Doesn't even get to try.

**With DuckDB:** Works — DuckDB handles 20 GB on 16 GB via disk spill, and
doesn't need polars at all.

### Summary Table

| Scenario | Today (polars) | After fix (DuckDB) |
|----------|---------------|---------------------|
| 1. Win 16GB / 5GB | Risky (might OOM) | Comfortable |
| 2. Mac 16GB / 5GB | Works | Works |
| 3. Win 16GB / 20GB | **Crashes** (OOM) | Works (disk spill) |
| 4. No polars / 5GB | **Crashes** (import error) | Works |
| 5. No polars / 20GB | **Crashes** (import error) | Works (disk spill) |

---

## The Fix: Unified DuckDB Loading

### What Changes

Replace the three independent loading paths with one shared data layer based
on DuckDB:

```
BEFORE (today):
  Validation   → clif_loader.py (DuckDB)
  Table One    → clifpy loader (polars)
  ECDF         → raw polars pl.scan_parquet()

AFTER (unified):
  Validation   → clif_loader.py (DuckDB)
  Table One    → clif_loader.py (DuckDB)     ← shares the same loader
  ECDF         → clif_loader.py (DuckDB)     ← shares the same loader
```

### What This Gives Us

1. **One place to configure the backend** — not three
2. **DuckDB by default** — works on all 5 scenarios
3. **Optional polars fast-path** — set `CLIF_BACKEND=polars` on beefy machines
4. **Same parquet files read once per step** — DuckDB connection can cache
   metadata across queries within a step
5. **Simpler dependency list** — polars becomes optional, not required

### What the ECDF Code Looks Like After

```python
# Before: hard polars dependency
import polars as pl
data_lazy = pl.scan_parquet(file_path)
values_df = data_lazy.filter(...).join(...).collect(streaming=True)

# After: DuckDB with one connection per run
result = con.execute("""
    SELECT d.vital_value
    FROM read_parquet(?) d
    JOIN icu_windows w ON d.hospitalization_id = w.hospitalization_id
    WHERE d.vital_category = ?
      AND d.recorded_dttm BETWEEN w.in_dttm AND w.out_dttm
""", [file_path, category]).fetchnumpy()
# fetchnumpy() returns a dict of numpy arrays — exactly what ECDF needs
```

---

## Glossary

| Term | Meaning |
|------|---------|
| **RAM** | Random Access Memory — fast, temporary storage where running programs keep data |
| **OOM** | Out Of Memory — crash that happens when a program tries to use more RAM than is available |
| **Parquet** | A compressed file format for tabular data. Smaller on disk than CSV, but must decompress when loaded into RAM |
| **DataFrame** | A table of data in memory (rows and columns), like a spreadsheet |
| **Lazy evaluation** | Building a plan for what to compute, without actually computing it yet |
| **Streaming** | Processing data in small chunks instead of loading it all at once |
| **Disk spill** | When an engine temporarily writes data back to disk because RAM is full, then reads it back as needed |
| **GC (Garbage Collection)** | Python's automatic process for freeing memory that's no longer used |
| **Memory fragmentation** | When freed memory is scattered in small pieces that can't be reused for large allocations |
| **Arena allocator** | A memory management strategy (used by DuckDB) that allocates in large blocks and frees them all at once — avoids fragmentation |
| **Predicate pushdown** | An optimization where filters are applied during file reading, so unmatched rows never enter RAM |
| **Row group** | Parquet files are divided into row groups (~100K rows each). Smart engines skip entire row groups that don't match your filter |
| **Hash join** | A join algorithm that builds a lookup table (hash table) from one side, then probes it with each row from the other side |
| **Zero-copy** | Passing data between libraries without making a duplicate — e.g., DuckDB → Arrow → pandas without copying the bytes |
| **Swap / Pagefile** | When the OS uses disk space as fake RAM. Much slower, but prevents immediate crashes |
| **Arrow** | Apache Arrow — a standard in-memory format for columnar data. DuckDB and pandas can share data through Arrow without copying |
