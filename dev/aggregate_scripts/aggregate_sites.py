#!/usr/bin/env python3
"""
Aggregate CLIF-TableOne-2026 per-site outputs into combined, cross-site files.

Each site lives in <root>/<SITE>/final/... . For every result file under final/
(excluding meta/ and validation/), this script collects the file from every site
and writes a combined version that shows, for each value:

    <key columns> | <metric>__<SITE1> | <metric>__<SITE2> | ... | <metric>__ALL

The __ALL column is the cross-site "Overall", computed per a rule chosen by the
column's meaning (see classify_metric / aggregation_notes.md):

    counts (n, total_obs, *_count, ...)  -> SUM across sites
    means / percentages / rates          -> sample-size WEIGHTED mean
    min                                  -> min of site mins
    max                                  -> max of site maxes
    q1 / q25 / lower IQR bound           -> min of site lower bounds   (user rule)
    q3 / q75 / upper IQR bound           -> max of site upper bounds   (user rule)
    median                               -> weighted mean of site medians
    iqr (single width column)            -> max of site widths
    sd / std                             -> weighted mean (approximation)

Display TableOne string tables (e.g. "22,034 (58.2%)", "62 [48, 72]"), the
demographic crosstab, mCIDE counts and the bins/ecdf parquet histograms have
dedicated handlers.

Outputs mirror the source folder layout under <out>/.

Usage:
    python aggregate_sites.py --root <dir-of-site-folders> --out <dir>
    python aggregate_sites.py            # uses the Box default path below
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import math
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Default source = the Box folder. Override with --root.
DEFAULT_ROOT = ("/Users/dema/Library/CloudStorage/Box-Box/CLIF/Projects/"
                "CLIF-TableOne-2026")

EXCLUDE_TOP_DIRS = {"meta", "validation"}
ALL_LABEL = "ALL"

# ---------------------------------------------------------------------------
# Column-name classification
# ---------------------------------------------------------------------------
COUNT_PAT = re.compile(
    r"^(n|count|total_obs|total_observations|total_distinct_observations|"
    r"total_number_of_stays|total_encounters|total_days|total_hours|"
    r"n_patients|n_encounters|n_deaths|n_stays|n_events|n_missing|"
    r"distinct_encounters|missing|at_risk|observed_events|"
    r"expired_count|hospice_count|hospice_or_expired_count|"
    r"hospice_or_expired|hospice|expired|count_value)$"
)
COUNT_SUFFIX_PAT = re.compile(r"(_n|_count|_obs|_events|_stays|_encounters)$")
RATE_PAT = re.compile(
    r"(mean|pct|percent|percentage|proportion|rate|prevalence|per_1000|"
    r"survival_prob|prob|events_per_hour|ci_lower|ci_upper|ci_margin|"
    r"capture_rate|combined_eol|hospice_among_eol|mortality)"
)

WEIGHT_PRIORITY = [
    "n", "n_encounters", "total_obs", "total_number_of_stays",
    "total_encounters", "count", "n_patients", "n_stays",
    "distinct_encounters", "total_observations", "N",
]


def classify_metric(name: str) -> str:
    """Return aggregation rule key for a numeric column name."""
    n = name.strip().lower()
    if n == "min" or n.endswith("_min") or n.startswith("min_"):
        return "MIN"
    if n == "max" or n.endswith("_max") or n.startswith("max_"):
        return "MAX"
    if re.search(r"(^|_)(q1|q25|q_?1)$", n) or n in ("q1_dose", "q1_stay_hours"):
        return "QLOW"
    if re.search(r"(^|_)(q3|q75|q_?3)$", n) or n in ("q3_dose", "q3_stay_hours"):
        return "QHIGH"
    if "median" in n:
        return "MEDIAN"
    if n in ("sd", "std"):
        return "SDW"
    if n == "iqr" or n.endswith("_iqr"):
        return "IQRW"
    if COUNT_PAT.match(n) or COUNT_SUFFIX_PAT.search(n):
        return "SUM"
    if RATE_PAT.search(n):
        return "MEANW"
    # Unknown column name: resolved later using the data (integer-like -> SUM).
    return "DEFAULT"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def discover_sites(root: Path, sites_arg: str | None) -> list[str]:
    if sites_arg:
        return [s.strip() for s in sites_arg.split(",") if s.strip()]
    out = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "final").is_dir():
            out.append(child.name)
    return out


def collect_rel_files(root: Path, sites: list[str], suffix: str) -> set[str]:
    """Relative-to-final paths (excluding meta/ & validation/) present in any site."""
    rels: set[str] = set()
    for s in sites:
        base = root / s / "final"
        if not base.is_dir():
            continue
        for dp, _dirs, fns in os.walk(base):
            rel_dir = os.path.relpath(dp, base)
            top = rel_dir.split(os.sep)[0]
            if top in EXCLUDE_TOP_DIRS:
                continue
            for fn in fns:
                if fn.endswith(suffix):
                    rels.add(os.path.normpath(os.path.join(rel_dir, fn)))
    return rels


def to_num(series: pd.Series) -> pd.Series:
    """Coerce a possibly-formatted string column to numeric (strip commas)."""
    if series.dtype.kind in "if":
        return series
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def is_value_col(frames: list[pd.DataFrame], col: str) -> bool:
    """A column is a numeric 'value' col if >=50% of its non-null cells parse as numbers."""
    total = 0
    good = 0
    for df in frames:
        if col not in df.columns:
            continue
        s = df[col]
        nonnull = s.notna().sum()
        total += nonnull
        good += to_num(s).notna().sum()
    return total > 0 and good >= 0.5 * total


def weighted_mean(values: list[float], weights: list[float]) -> float | None:
    pairs = [(v, w) for v, w in zip(values, weights)
             if v is not None and not _nan(v) and w is not None
             and not _nan(w) and w > 0]
    if not pairs:
        vv = [v for v in values if v is not None and not _nan(v)]
        return float(np.mean(vv)) if vv else None
    num = sum(v * w for v, w in pairs)
    den = sum(w for _v, w in pairs)
    return num / den if den else None


def _nan(x) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True


def combine_value(rule: str, values: list[float], weights: list[float]):
    vv = [v for v in values if v is not None and not _nan(v)]
    if not vv:
        return None
    if rule == "SUM":
        return sum(vv)
    if rule in ("MIN", "QLOW"):
        return min(vv)
    if rule in ("MAX", "QHIGH", "IQRW"):
        return max(vv)
    if rule in ("MEANW", "MEDIAN", "SDW"):
        return weighted_mean(values, weights)
    return sum(vv)


def round_smart(x):
    if x is None or _nan(x):
        return None
    f = float(x)
    if f == int(f):
        return int(f)
    return round(f, 6)


# ---------------------------------------------------------------------------
# Generic keyed-table aggregator
# ---------------------------------------------------------------------------
def agg_keyed_table(rel: str, site_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = list(site_frames.values())
    cols = list(frames[0].columns)
    for df in frames[1:]:
        for c in df.columns:
            if c not in cols:
                cols.append(c)

    value_cols = [c for c in cols if is_value_col(frames, c)]
    key_cols = [c for c in cols if c not in value_cols]
    if not key_cols:  # degenerate: treat first col as key
        key_cols, value_cols = [cols[0]], cols[1:]

    rules = {c: classify_metric(c) for c in value_cols}
    # Case-insensitive weight lookup (e.g. a column literally named "Count").
    lower_map = {c.lower(): c for c in value_cols}
    weight_col = next((lower_map[w.lower()] for w in WEIGHT_PRIORITY
                       if w.lower() in lower_map), None)

    # Resolve from the data. Any all-integer column is a count -> SUM (this also
    # rescues count columns whose name accidentally hits a stat keyword, e.g.
    # "Resp Rate Set (N)" matching 'rate'). Genuine means/rates are fractional and
    # stay weighted; min/max/median/quartiles were already name-classified above.
    for c in value_cols:
        if rules[c] not in ("DEFAULT", "MEANW"):
            continue
        vals = pd.concat([to_num(df[c]) for df in frames if c in df.columns])
        vals = vals.dropna()
        integer_like = len(vals) > 0 and bool((vals == vals.round()).all())
        if integer_like:
            rules[c] = "SUM"
        elif rules[c] == "DEFAULT":
            rules[c] = "MEANW"

    # Percentage columns paired with a same-prefix count column (e.g.
    # norepinephrine_pct <-> norepinephrine_n): recover each site's denominator
    # from count/(pct/100) so the Overall percentage is exact.
    pct_pair = {}
    for c in value_cols:
        m = re.match(r"^(.+)_pct$", c.lower())
        if m and f"{m.group(1)}_n" in lower_map:
            pct_pair[c] = lower_map[f"{m.group(1)}_n"]

    # Normalise each site frame: numeric value cols, string keys, index by keys.
    norm = {}
    for site, df in site_frames.items():
        d = df.copy()
        for c in value_cols:
            if c in d.columns:
                d[c] = to_num(d[c])
        for c in key_cols:
            if c in d.columns:
                d[c] = d[c].astype(str).str.strip()
            else:
                d[c] = np.nan
        d = d[[c for c in key_cols + value_cols if c in d.columns]]
        # collapse duplicate keys defensively
        d = d.groupby(key_cols, dropna=False, sort=False).first().reset_index()
        norm[site] = d.set_index(key_cols)

    all_keys = None
    for d in norm.values():
        idx = d.index
        all_keys = idx if all_keys is None else all_keys.union(idx, sort=False)

    sites = list(norm.keys())
    out_rows = []
    for key in all_keys:
        row = {}
        if len(key_cols) == 1:
            row[key_cols[0]] = key
        else:
            for kc, kv in zip(key_cols, key):
                row[kc] = kv
        for c in value_cols:
            per_site_vals, per_site_w = [], []
            for s in sites:
                d = norm[s]
                v = d[c].get(key) if c in d.columns else None
                w = d[weight_col].get(key) if (weight_col and weight_col in d.columns) else None
                v = None if (v is None or _nan(v)) else float(v)
                w = None if (w is None or _nan(w)) else float(w)
                row[f"{c}__{s}"] = round_smart(v)
                per_site_vals.append(v)
                per_site_w.append(w)
            rule = rules[c]
            if c in pct_pair:
                # Recover denominators from the paired count column, pool exactly.
                ncol = pct_pair[c]
                num_all, den_all, saw = 0.0, 0.0, False
                for s, pv in zip(sites, per_site_vals):
                    nv = norm[s][ncol].get(key) if ncol in norm[s].columns else None
                    nv = None if (nv is None or _nan(nv)) else float(nv)
                    if nv is not None:
                        saw = True
                        num_all += nv
                        if pv is not None and pv > 0:
                            den_all += nv / (pv / 100.0)
                if den_all > 0:
                    all_val = 100 * num_all / den_all
                elif saw and num_all == 0:
                    all_val = 0.0  # numerator is zero -> percentage is zero
                else:
                    all_val = None
            elif rule in ("MEANW", "MEDIAN", "SDW") and weight_col is None:
                # No sample-size column to weight by -> don't fabricate an Overall.
                all_val = None
            else:
                all_val = combine_value(rule, per_site_vals, per_site_w)
            row[f"{c}__{ALL_LABEL}"] = round_smart(all_val)
        out_rows.append(row)

    ordered = list(key_cols)
    for c in value_cols:
        ordered += [f"{c}__{s}" for s in sites] + [f"{c}__{ALL_LABEL}"]
    return pd.DataFrame(out_rows, columns=ordered)


# ---------------------------------------------------------------------------
# Display TableOne string tables  (Variable + per-stratum string columns)
# ---------------------------------------------------------------------------
NUM = r"[-+]?[\d,]*\.?\d+"
RE_N_PCT = re.compile(rf"^\s*({NUM})\s*\(\s*({NUM})\s*%\s*\)\s*$")
RE_MED_IQR = re.compile(rf"^\s*({NUM})\s*\[\s*({NUM})\s*,\s*({NUM})\s*\]\s*$")
RE_PLAIN = re.compile(rf"^\s*({NUM})\s*$")


def _f(s: str) -> float:
    return float(str(s).replace(",", ""))


def parse_cell(val):
    """Return ('kind', payload) for a TableOne display cell."""
    if val is None or (isinstance(val, float) and _nan(val)):
        return ("blank", None)
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return ("blank", None)
    m = RE_N_PCT.match(s)
    if m:
        return ("n_pct", (_f(m.group(1)), _f(m.group(2))))
    m = RE_MED_IQR.match(s)
    if m:
        return ("med_iqr", (_f(m.group(1)), _f(m.group(2)), _f(m.group(3))))
    m = RE_PLAIN.match(s)
    if m:
        return ("plain", _f(m.group(1)))
    return ("text", s)


def fmt_int(x) -> str:
    return f"{int(round(x)):,}"


def fmt_pct(x) -> str:
    return f"{x:.1f}"


# Patient-level variables denominate by "N: Unique patients"; everything else by
# "N: Encounter blocks" (hybrid label map, per the table_one denominator rule).
PATIENT_LEVEL_RE = re.compile(
    r"^\s*(race|ethnicity|sex|gender|primary\s+language|language|"
    r"insurance|payer)\b", re.I)
# How close a cell's back-calculated denominator must be to the labeled N row to
# trust the exact integer instead of the recovered value (guards nested
# sub-population denominators like "Medications during IMV (N=...)").
DENOM_SNAP_TOL = 0.05


def _ref_denom(indexed_df, order, vc, want_patients):
    """Exact denominator for a site/column from the 'N: Unique patients' /
    'N: Encounter blocks' rows. Returns float or None."""
    needle = "unique patients" if want_patients else "encounter blocks"
    for v in order:
        if v not in indexed_df.index:
            continue
        if needle in str(v).lower():
            kind, payload = parse_cell(_get(indexed_df, v, vc))
            if kind == "plain":
                return float(payload)
            if kind == "n_pct":
                return float(payload[0])
    return None


def agg_display_tableone(rel: str, site_frames: dict[str, pd.DataFrame],
                         var_col: str, value_cols: list[str]) -> pd.DataFrame:
    """One ALL column per original value column. Counts sum; n(%) recompute using
    the per-stratum denominator (the 'N: Encounter blocks'/first N row); medians use
    weighted point + [min q1, max q3] envelope."""
    sites = list(site_frames.keys())
    # Row matching is case-insensitive but indentation-sensitive: we key rows on
    # the lowercased label *including* its leading whitespace. Lowercasing folds
    # case differences across sites ("Race: White" == "Race: white"); keeping the
    # leading spaces preserves nesting depth so rows that share a label at
    # different indent levels (e.g. "  Norepinephrine" vs "    Norepinephrine")
    # are NOT merged. Trailing whitespace is ignored.
    def _row_key(x):
        return str(x).rstrip().lower()

    # union of variable rows (by key) preserving first-seen order; remember the
    # first original label seen for each key to use as the display label.
    order, seen, display = [], set(), {}
    for df in site_frames.values():
        for orig in df[var_col].astype(str):
            k = _row_key(orig)
            if k not in seen:
                seen.add(k); order.append(k); display[k] = orig

    indexed = {s: df.set_index(df[var_col].astype(str).map(_row_key))
               for s, df in site_frames.items()}

    # Per-site "patient/encounter N" used to weight median rows: the largest
    # plain-integer value in any row whose label starts with 'N' or 'N:'.
    def site_n(site, vc):
        d = indexed[site]
        best = None
        for v in order:
            if v not in d.index:
                continue
            if re.match(r"\s*N\b|.*\bunique patients\b|.*encounter blocks", v, re.I):
                kind, payload = parse_cell(_get(d, v, vc))
                if kind == "plain":
                    best = payload if best is None else max(best, payload)
        return best

    # Per (site, value column) exact denominators from the N rows, computed once.
    ref_pat = {(s, vc): _ref_denom(indexed[s], order, vc, True)
               for s in sites for vc in value_cols}
    ref_enc = {(s, vc): _ref_denom(indexed[s], order, vc, False)
               for s in sites for vc in value_cols}

    rows = []
    for var in order:
        row = {var_col: display[var]}
        patient_level = bool(PATIENT_LEVEL_RE.match(str(var)))
        ref = ref_pat if patient_level else ref_enc
        for vc in value_cols:
            parsed = []   # aligned per-site (kind, payload)
            for s in sites:
                cell = _get(indexed[s], var, vc)
                row[f"{vc}__{s}"] = "" if _blank(cell) else cell
                parsed.append(parse_cell(cell))
            row[f"{vc}__{ALL_LABEL}"] = _all_display(
                parsed, [site_n(s, vc) for s in sites],
                [ref[(s, vc)] for s in sites])
        rows.append(row)

    ordered = [var_col]
    for vc in value_cols:
        ordered += [f"{vc}__{s}" for s in sites] + [f"{vc}__{ALL_LABEL}"]
    return pd.DataFrame(rows, columns=ordered)


def _blank(v):
    return v is None or (isinstance(v, float) and _nan(v)) or str(v).strip() == ""


def _get(indexed_df, var, col):
    if var not in indexed_df.index or col not in indexed_df.columns:
        return None
    val = indexed_df[col].get(var)
    if isinstance(val, pd.Series):
        val = val.iloc[0]
    return val


def _all_display(parsed, site_ns, ref_denoms=None):
    """parsed: list of (kind, payload) per site; site_ns: per-site N for median
    weighting; ref_denoms: per-site exact denominator from the labeled N row
    (unique patients / encounter blocks), or None when unavailable."""
    present = [k for k, _ in parsed if k != "blank"]
    if not present:
        return ""
    if ref_denoms is None:
        ref_denoms = [None] * len(parsed)
    dom = max(set(present), key=present.count)
    if dom == "n_pct":
        # Hybrid denominator: prefer the exact labeled N row, but only when the
        # cell's own back-calculated denominator (n / (pct/100)) agrees with it.
        # Disagreement means a nested sub-population denominator (e.g.
        # "Medications during IMV (N=...)") -> trust the recovered value instead.
        total_n = 0.0
        denom_sum = 0.0
        for (kind, payload), exact in zip(parsed, ref_denoms):
            if kind != "n_pct":
                continue
            n_i, pct_i = payload
            total_n += n_i
            recovered = n_i / (pct_i / 100.0) if pct_i > 0 else None
            if (exact and exact > 0 and
                    (recovered is None
                     or abs(recovered - exact) <= DENOM_SNAP_TOL * exact)):
                denom_sum += exact            # exact integer (also fixes pct=0.0%)
            elif recovered is not None:
                denom_sum += recovered        # nested sub-population denominator
            elif exact and exact > 0:
                denom_sum += exact
        if denom_sum > 0:
            return f"{fmt_int(total_n)} ({fmt_pct(100*total_n/denom_sum)}%)"
        return fmt_int(total_n)
    if dom == "med_iqr":
        meds = [p[0] for k, p in parsed if k == "med_iqr"]
        q1s = [p[1] for k, p in parsed if k == "med_iqr"]
        q3s = [p[2] for k, p in parsed if k == "med_iqr"]
        w = [n if (n and not _nan(n)) else 1.0 for n in site_ns[:len(meds)]]
        med = weighted_mean(meds, w) if meds else None
        if med is None:
            return ""
        return f"{_fmt_num(med)} [{_fmt_num(min(q1s))}, {_fmt_num(max(q3s))}]"
    if dom == "plain":
        plains = [p for k, p in parsed if k == "plain"]
        return fmt_int(sum(plains)) if plains else ""
    # text / mixed -> not poolable
    return ""


def _fmt_num(x):
    f = float(x)
    return str(int(round(f))) if abs(f - round(f)) < 1e-9 else f"{f:.1f}"


# ---------------------------------------------------------------------------
# Comorbidities per-1000 (recover total-hospitalizations denominator per site)
# ---------------------------------------------------------------------------
def agg_comorbidities(rel: str, site_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    sites = list(site_frames.keys())
    norm = {}
    site_denom = {}
    for s, df in site_frames.items():
        d = df.copy()
        d["comorbidity"] = d["comorbidity"].astype(str).str.strip()
        for c in ("n_patients", "prevalence_percent", "per_1000_hospitalizations"):
            d[c] = to_num(d[c])
        d = d.set_index("comorbidity")
        norm[s] = d
        # denom = n_patients / (prevalence/100); take the modal/first valid value
        recov = (d["n_patients"] / (d["prevalence_percent"] / 100.0)).replace(
            [np.inf, -np.inf], np.nan).dropna()
        site_denom[s] = float(round(recov.median())) if len(recov) else None

    order, seen = [], set()
    for d in norm.values():
        for r in d.index:
            if r not in seen:
                seen.add(r); order.append(r)

    rows = []
    for com in order:
        row = {"comorbidity": com}
        n_all = 0.0
        for s in sites:
            v = norm[s]["n_patients"].get(com)
            v = None if (v is None or _nan(v)) else float(v)
            row[f"n_patients__{s}"] = round_smart(v)
            row[f"prevalence_percent__{s}"] = round_smart(norm[s]["prevalence_percent"].get(com))
            row[f"per_1000_hospitalizations__{s}"] = round_smart(
                norm[s]["per_1000_hospitalizations"].get(com))
            if v is not None:
                n_all += v
        denom_all = sum(d for d in site_denom.values() if d)
        row["n_patients__ALL"] = round_smart(n_all)
        if denom_all:
            row["prevalence_percent__ALL"] = round_smart(100 * n_all / denom_all)
            row["per_1000_hospitalizations__ALL"] = round_smart(1000 * n_all / denom_all)
        else:
            row["prevalence_percent__ALL"] = None
            row["per_1000_hospitalizations__ALL"] = None
        rows.append(row)

    ordered = ["comorbidity"]
    for c in ("n_patients", "prevalence_percent", "per_1000_hospitalizations"):
        ordered += [f"{c}__{s}" for s in sites] + [f"{c}__ALL"]
    return pd.DataFrame(rows, columns=ordered)


# ---------------------------------------------------------------------------
# name/value tables (strobe_counts, mortality_rates, *_summary, total_observations)
# ---------------------------------------------------------------------------
def agg_name_value(rel: str, site_frames: dict[str, pd.DataFrame],
                   label_col: str, value_col: str) -> pd.DataFrame:
    """Counts (label starts with total/n/number/count) -> sum; everything else
    (rates/averages/strings) -> per-site only, ALL left blank to avoid bad math."""
    base = os.path.basename(rel)
    force_sum = base in ("strobe_counts.csv",
                         "ventilator_settings_total_observations.csv")
    sites = list(site_frames.keys())
    norm = {s: df.set_index(df[label_col].astype(str).str.strip())[value_col]
            for s, df in site_frames.items()}
    order, seen = [], set()
    for s in sites:
        for r in norm[s].index:
            if r not in seen:
                seen.add(r); order.append(r)

    sum_re = re.compile(r"^\s*(total\b|number\b|count\b)|_count\b", re.I)
    rows = []
    for lbl in order:
        row = {label_col: lbl}
        vals = []
        for s in sites:
            raw = norm[s].get(lbl)
            if isinstance(raw, pd.Series):
                raw = raw.iloc[0]
            row[f"{value_col}__{s}"] = "" if _blank(raw) else raw
            num = to_num(pd.Series([raw])).iloc[0]
            vals.append(None if _nan(num) else float(num))
        numeric = [v for v in vals if v is not None]
        do_sum = force_sum or bool(sum_re.match(str(lbl)))
        if do_sum and numeric:
            row[f"{value_col}__ALL"] = round_smart(sum(numeric))
        else:
            row[f"{value_col}__ALL"] = ""
        rows.append(row)
    ordered = ([label_col] + [f"{value_col}__{s}" for s in sites]
               + [f"{value_col}__ALL"])
    return pd.DataFrame(rows, columns=ordered)


# ---------------------------------------------------------------------------
# Demographic crosstab (two header rows + <10 masking)
# ---------------------------------------------------------------------------
def agg_crosstab(rel: str, root: Path, sites: list[str]) -> pd.DataFrame | None:
    raw = {}
    for s in sites:
        p = root / s / "final" / rel
        if p.exists():
            raw[s] = pd.read_csv(p, header=[0, 1], index_col=0)
    if not raw:
        return None
    sites_present = list(raw.keys())
    # union of (row labels) and columns
    base = raw[sites_present[0]]
    cols = list(base.columns)
    row_order, seen = [], set()
    for df in raw.values():
        for r in df.index.astype(str):
            if r not in seen:
                seen.add(r); row_order.append(r)

    def cell_num(df, r, c):
        try:
            v = df.loc[r, c]
        except KeyError:
            return None
        if isinstance(v, pd.Series):
            v = v.iloc[0]
        sv = str(v).strip()
        if sv in ("", "nan", "<10"):
            return None
        try:
            return float(str(sv).replace(",", ""))
        except ValueError:
            return None

    out = pd.DataFrame(index=row_order,
                       columns=pd.MultiIndex.from_tuples(cols))
    for r in row_order:
        for c in cols:
            vals = [cell_num(raw[s], r, c) for s in sites_present]
            vals = [v for v in vals if v is not None]
            out.loc[r, c] = int(sum(vals)) if vals else ""
    out.index.name = ""
    return out


# ---------------------------------------------------------------------------
# bins / ecdf parquet histograms -> sum bin counts, recompute percentage
# ---------------------------------------------------------------------------
def agg_bins_parquet(rel: str, root: Path, sites: list[str]) -> pd.DataFrame | None:
    cand = ("segment", "bin_num", "bin_min", "bin_max",
            "interval", "x", "value", "quantile")
    # Keep only sites whose parquet really is a bins/ecdf table: it must have a
    # "count" column AND at least one bin-key column. Sites with an empty or
    # differently-shaped placeholder file for this metric are skipped (not fatal).
    usable = {}
    for s in sites:
        p = root / s / "final" / rel
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p)
        except Exception as e:  # noqa: BLE001
            print(f"  ! parquet read {s}/{rel}: {e}", file=sys.stderr)
            continue
        if df is None or not len(df.columns):
            continue
        if "count" in df.columns and any(c in df.columns for c in cand):
            usable[s] = df
    if not usable:
        return None
    # Merge keys = bin-key columns present in EVERY usable site (intersection),
    # so schema differences between sites can't crash the merge.
    common = set.intersection(*[set(df.columns) for df in usable.values()])
    keys = [c for c in cand if c in common]
    if not keys:
        return None
    merged = None
    for s, df in usable.items():
        sub = df[keys + ["count"]].copy()
        sub = sub.rename(columns={"count": f"count__{s}"})
        merged = sub if merged is None else merged.merge(sub, on=keys, how="outer")
    cnt_cols = [c for c in merged.columns if c.startswith("count__")]
    merged["count__ALL"] = merged[cnt_cols].sum(axis=1, skipna=True)
    if any("percentage" in df.columns for df in usable.values()):
        tot = merged["count__ALL"].sum()
        merged["percentage__ALL"] = (100 * merged["count__ALL"] / tot) if tot else np.nan
    return merged.sort_values(keys).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def read_csv_safe(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""])
    except Exception as e:  # noqa: BLE001
        print(f"  ! read error {path}: {e}", file=sys.stderr)
        return None


def is_display_tableone(rel: str, frames: dict[str, pd.DataFrame]) -> tuple[bool, str, list]:
    df = next(iter(frames.values()))
    cols = list(df.columns)
    name = os.path.basename(rel)
    if cols and cols[0] == "Variable":
        return True, "Variable", cols[1:]
    if name.endswith("_icu_vs_no_icu.csv") or name.endswith("_ed_icu_vs_ed_ward.csv"):
        return True, cols[0], cols[1:]
    return False, "", []


def write_csv(df: pd.DataFrame, out_path: Path, index: bool = False):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=index)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT,
                    help="Folder containing per-site subfolders (default: Box path)")
    ap.add_argument("--out", default=None,
                    help="Output folder (default: <root>/_aggregated)")
    ap.add_argument("--sites", default=None,
                    help="Comma-separated site names (default: auto-discover)")
    ap.add_argument("--no-parquet", action="store_true",
                    help="Skip overall/bins and overall/ecdf parquet aggregation")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    if not root.is_dir():
        sys.exit(f"ERROR: root not found or not accessible: {root}")
    out = Path(args.out).expanduser() if args.out else root / "_aggregated"
    sites = discover_sites(root, args.sites)
    if len(sites) < 1:
        sys.exit(f"ERROR: no site subfolders with final/ found under {root}")
    print(f"Root : {root}")
    print(f"Out  : {out}")
    print(f"Sites: {sites}\n")

    summary = {"sites": sites, "csv_written": 0, "parquet_written": 0,
               "skipped": [], "crosstab": 0}

    # ----- CSV files -----
    csv_rels = sorted(collect_rel_files(root, sites, ".csv"))
    n_csv = len(csv_rels)
    print(f"Processing {n_csv} CSV files...", flush=True)
    for i, rel in enumerate(csv_rels, 1):
        print(f"  [{i}/{n_csv}] {rel}", flush=True)
        frames = {}
        for s in sites:
            p = root / s / "final" / rel
            if p.exists():
                df = read_csv_safe(p)
                if df is not None and len(df.columns):
                    frames[s] = df
        if not frames:
            continue
        base = os.path.basename(rel)
        cols0 = list(next(iter(frames.values())).columns)
        try:
            if base.startswith("demographic_crosstab"):
                res = agg_crosstab(rel, root, sites)
                if res is not None:
                    write_csv(res, out / rel, index=True)
                    summary["crosstab"] += 1
                    continue
            if (base.startswith("comorbidities_per_1000_hospitalizations")
                    and not base.endswith("_summary.csv")
                    and "comorbidity" in cols0):
                res = agg_comorbidities(rel, frames)
                write_csv(res, out / rel)
                summary["csv_written"] += 1
                continue
            if len(cols0) == 2 and cols0[0] in (
                    "Metric", "count_name", "metric") and cols0[1] in (
                    "Value", "count_value", "value"):
                res = agg_name_value(rel, frames, cols0[0], cols0[1])
                write_csv(res, out / rel)
                summary["csv_written"] += 1
                continue
            disp, var_col, val_cols = is_display_tableone(rel, frames)
            if disp:
                res = agg_display_tableone(rel, frames, var_col, val_cols)
            else:
                res = agg_keyed_table(rel, frames)
            write_csv(res, out / rel)
            summary["csv_written"] += 1
        except Exception as e:  # noqa: BLE001
            print(f"  ! agg error {rel}: {e}", file=sys.stderr)
            summary["skipped"].append(f"{rel}: {e}")

    # ----- parquet bins/ecdf -----
    if not args.no_parquet:
        pq_rels = sorted(collect_rel_files(root, sites, ".parquet"))
        pq_todo = [r for r in pq_rels
                   if "bins" in r.split(os.sep) or "ecdf" in r.split(os.sep)]
        n_pq = len(pq_todo)
        print(f"Processing {n_pq} parquet histograms...", flush=True)
        for j, rel in enumerate(pq_todo, 1):
            print(f"  [{j}/{n_pq}] {rel}", flush=True)
            try:
                res = agg_bins_parquet(rel, root, sites)
                if res is not None:
                    op = out / rel
                    op.parent.mkdir(parents=True, exist_ok=True)
                    res.to_parquet(op, index=False)
                    summary["parquet_written"] += 1
            except Exception as e:  # noqa: BLE001
                print(f"  ! parquet error {rel}: {e}", file=sys.stderr)
                summary["skipped"].append(f"{rel}: {e}")

    (out).mkdir(parents=True, exist_ok=True)
    with open(out / "_aggregation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. CSV: {summary['csv_written']}  "
          f"crosstab: {summary['crosstab']}  "
          f"parquet: {summary['parquet_written']}  "
          f"skipped: {len(summary['skipped'])}")
    if summary["skipped"]:
        print("Skipped:")
        for s in summary["skipped"]:
            print(f"  - {s}")


if __name__ == "__main__":
    main()
