# CLIF-TableOne-2026 — Cross-Site Aggregation

`aggregate_sites.py` combines the per-site CLIF-TableOne-2026 result files into
single files that show, for every value, **one column per site plus an Overall
("ALL") column** computed across all sites.

---

## 1. What it reads

The source `--root` contains one subfolder per site, each holding a `final/`
results tree:

```
<root>/
  UCMC/final/...
  MIMIC/final/...
  <SITE>/final/...
```

- Sites are auto-discovered: any immediate subdirectory that contains a
  `final/` folder.
- Only files under `final/` are processed.
- The **`meta/` and `validation/`** folders inside each `final/` are **ignored**
  (per project decision).
- A relative file path is aggregated if it is present in **at least one** site.

The default `--root` is the **Box folder**:

```
/Users/dema/Library/CloudStorage/Box-Box/CLIF/Projects/CLIF-TableOne-2026
```

## 2. What it writes

Outputs mirror the source layout under `--out` (default `<root>/_aggregated`).
For keyed/tabular files the column layout is:

```
<key columns> | <metric>__<SITE1> | <metric>__<SITE2> | ... | <metric>__ALL
```

`__ALL` is the cross-site Overall. CSV stays CSV; parquet histograms stay
parquet.

## 3. How to run

```bash
# Against the Box default path:
python aggregate_sites.py

# Against the local sample, custom output:
python aggregate_sites.py --root /Users/dema/WD/CLIF-2026 --out /tmp/agg

# Restrict to specific sites:
python aggregate_sites.py --sites UCMC,MIMIC

# Skip parquet histograms:
python aggregate_sites.py --no-parquet
```

Requires `pandas` and `pyarrow` (for parquet).

### macOS note — Box / Full Disk Access
The Box CloudStorage path is protected by macOS privacy (TCC). If you run the
script from a host that lacks file access (e.g. Claude.app), reads fail with
"Operation not permitted." Either:
- grant **Full Disk Access** to the app under
  *System Settings → Privacy & Security → Full Disk Access* and restart it, or
- run the script from a regular Terminal that already has access to Box.

---

## 4. Per-column aggregation rules (the Overall column)

The Overall value for each numeric column is chosen by the column's **meaning**
(name) and then refined by its **data** (see §5). Rules:

| Column meaning | Examples | Overall (`__ALL`) rule |
|---|---|---|
| Count / total | `n`, `count`, `total_obs`, `*_count`, `n_patients`, `n_deaths` | **SUM** across sites |
| Mean | `mean`, `mean_*` | **sample-size-weighted mean** |
| Percentage / rate / proportion | `pct`, `percent`, `proportion`, `rate`, `prevalence`, `per_1000` | **weighted mean** (or exact denominator recovery, see §6) |
| Minimum | `min`, `p0` | **min of site mins** |
| Maximum | `max`, `p100` | **max of site maxes** |
| Lower IQR bound | `q1`, `q25`, `p25`, lower quartile | **min of site lower bounds** ⟵ *user rule* |
| Upper IQR bound | `q3`, `q75`, `p75`, upper quartile | **max of site upper bounds** ⟵ *user rule* |
| Median | `median`, `p50` | **weighted mean of site medians** |
| IQR width (single column) | `iqr` | **max of site widths** |
| SD / std | `sd`, `std` | weighted mean (approximation) |
| Anything else | — | weighted mean (DEFAULT) |

### The IQR "envelope" rule (explicit user decision)
> *"Take the min of the mins and max of the maxes observed in the IQR."*

So for a median/IQR triple the Overall is:
`Overall Q1 = min(site Q1s)`, `Overall Q3 = max(site Q3s)`. This deliberately
produces the **widest** plausible interval rather than a pooled quantile
estimate. The same envelope logic applies to display strings like
`"62 [48, 72]"`.

## 5. Data-aware refinement

After name-based classification, columns left as DEFAULT/weighted-mean are
inspected: if **every observed value across sites is integer-like**, the column
is reclassified to **SUM** (it is really a count with a human-readable name,
e.g. `"ICU Encounters"`, `"Resp Rate Set (N)"`). This prevents counts from being
silently averaged. Weight-column lookup is **case-insensitive**
(`Count` == `count`).

## 6. Exact percentage pooling (denominator recovery)

A weighted mean of percentages is only approximate. Where the underlying
denominator can be recovered, the script pools exactly:

- **`*_pct` ↔ `*_n` pairs**: `Overall pct = 100 · Σ n / Σ (n / (pct/100))`.
  If the numerator is genuinely 0 across sites (drug never given), Overall = 0.0.
- **Display TableOne `n (%)` cells**: each site's own denominator is recovered
  per cell as `n_i / (pct_i/100)`, then `Overall % = 100 · Σ n / Σ denom`.
  (Demographic percentages are of unique patients, not encounter blocks — this
  recovery uses each cell's real base.)
- **Comorbidities**: per-site total-hospitalization denominator is recovered as
  the median of `n_patients / (prevalence_percent/100)`; `n_patients` is summed
  and prevalence / per-1000 recomputed from the pooled totals.

## 7. Special-cased file types

Files that don't fit the generic keyed-table pattern have dedicated handlers
(dispatch order in `main()`):

1. **Demographic crosstab** — two header rows (`header=[0,1]`); numeric cells
   summed; `<10` masked cells treated as missing.
2. **Comorbidities** — denominator recovery as in §6.
3. **2-column name/value tables** (`Metric`/`Value`, `count_name`/`count_value`)
   — only rows that are clearly totals/counts (`^total|number|count` or
   `_count`, plus a short force-sum allowlist) are summed; rate rows are left
   blank in Overall because there is no in-file denominator.
4. **Display TableOne string tables** — parse `"22,034 (58.2%)"` and
   `"62 [48, 72]"`; pool n(%) by denominator recovery and median rows by the
   IQR envelope.
5. **bins / ecdf parquet histograms** — merged on bin keys; `count` summed;
   `percentage__ALL` recomputed from pooled counts.

## 8. Cells intentionally left blank in Overall

Some Overall values cannot be pooled from the available columns and are left
blank by design (not a bug):

- **Rates with no in-file denominator** — `mortality_rates`,
  `code_status_percentages` (e.g. "per 1000 hospitalizations" rows). Summing or
  averaging them would be wrong; the denominator isn't present in the file.
- **Kaplan–Meier survival curves** — `km_time_to_extubation` (survival
  probabilities are not additive across sites).
- **Event-rate columns** — `adt_event_capture` events-per-hour.

These are the only blank Overall cells in a clean run; everything else pools.

## 9. Verification status

A full run over the sample produced **1,154 output files**
(`162 CSV + 15 crosstab + 976 parquet`, **0 skipped, 0 errors**). Blank Overall
cells were reduced from 158 → 16 through the fixes above, and all 16 remaining
were confirmed to be legitimately non-poolable (§8). Spot results were
hand-verified for mCIDE sums, summary_stats min/max/Q1/Q3/weighted-mean,
SOFA rates, collection_statistics IQR, display-table denominators,
comorbidities, histogram bins, mode proportions, and medication `*_pct`
recovery.
