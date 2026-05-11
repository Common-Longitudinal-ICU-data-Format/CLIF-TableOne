"""Small-cell suppression for Table One outputs.

Reads raw (unsuppressed) Table One CSVs from ``output/intermediate/tableone/``,
applies the merge + cell-suppression rules declared in
``config/tableone_merge_rules.yaml``, and writes a consortium-safe
(shareable) copy under ``output/final/.../tableone/``.

Input CSVs follow the Table One convention established by the generator:

    Variable                                         | Overall
    -------------------------------------------------|----------------
    N: Encounter blocks                              | 314,828
    Age at admission, median [Q1, Q3]                | 60 [41, 72]
    Race: White                                      | 78,313 (39.4%)
    Race: Black or African American                  | 93,050 (46.8%)
    Race: Other                                      | 6,595 (3.3%)
    ...

The first column is always ``Variable``; subsequent columns are one value
per stratum (``Overall``, ``ICU``, ``Advanced Respiratory Support``, …).
Rows in the ``Prefix: value`` format are treated as members of the
``Prefix`` group and eligible for merging.

Outputs:
 - ``scan_small_cells(intermediate_root, rules) -> list[SmallCell]`` —
   walks the intermediate tree and returns a flat list of unresolved
   small cells (those still N < threshold after rules apply).
 - ``apply_suppression_to_tree(intermediate_root, final_root, rules)`` —
   writes the suppressed CSVs to ``final_root``.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class SuppressionConfig:
    threshold: int = 10
    token: str = '<10'
    recompute_percentages: bool = True
    apply_complementary: bool = False


@dataclass
class MergeRules:
    """Parsed merge rules keyed by variable prefix.

    ``variables[variable_label][merged_group_label]`` is a list of source
    row values to collapse into the merged group.
    """
    version: int = 1
    variables: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    suppression: SuppressionConfig = field(default_factory=SuppressionConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> 'MergeRules':
        path = Path(path)
        if not path.exists():
            return cls()
        with path.open() as f:
            raw = yaml.safe_load(f) or {}
        rules = cls(version=int(raw.get('version', 1)))
        supp = raw.get('suppression') or {}
        rules.suppression = SuppressionConfig(
            threshold=int(supp.get('threshold', 10)),
            token=str(supp.get('token', '<10')),
            recompute_percentages=bool(supp.get('recompute_percentages', True)),
            apply_complementary=bool(supp.get('apply_complementary', False)),
        )
        for key, value in raw.items():
            if key in ('version', 'suppression'):
                continue
            if isinstance(value, dict):
                rules.variables[key] = {
                    str(merged_label): [str(v) for v in (sources or [])]
                    for merged_label, sources in value.items()
                }
        return rules


@dataclass
class SmallCell:
    """One unresolved small cell in the intermediate tree."""
    cohort: str        # 'ci' or 'ward'
    stratum: str       # 'overall' or 'icu' / 'vaso' / ...
    csv_path: str      # relative path of the source CSV
    column: str        # data column (stratum label, e.g. 'Overall')
    variable: str      # parent variable (e.g. 'Race') or '' if not groupable
    row: str           # row label (e.g. 'Other' or 'age at admission')
    raw_n: int
    merged_n: Optional[int]         # None if no merge applies, else the merged N
    merged_label: Optional[str]     # None if no merge applies
    status: str        # 'resolved_by_merge' | 'still_small' | 'complementary' | 'group_suppressed'
    complementary_target: Optional[str] = None  # sibling row that got complementary-suppressed


# ---------------------------------------------------------------------------
# Cell parsing
# ---------------------------------------------------------------------------

_COUNT_WITH_PCT = re.compile(r'^\s*(\d[\d,]*)\s*(?:\(([\d.]+)\s*%\))?\s*$')


@dataclass
class Cell:
    raw: str                    # original string as stored in the CSV
    n: Optional[int]            # parsed count, or None if not a count cell
    pct: Optional[float]        # parsed percentage, or None
    suppressed: bool = False    # set true once we decide to suppress this cell

    @property
    def is_count(self) -> bool:
        return self.n is not None


def parse_cell(value: Any) -> Cell:
    """Parse a Table One cell into its count/percentage parts.

    Non-count cells (blank, text, continuous like "60 [41, 72]") come back
    with ``n = None`` and are passed through untouched by the suppression
    pass.
    """
    if value is None:
        return Cell(raw='', n=None, pct=None)
    s = str(value).strip()
    if not s:
        return Cell(raw=s, n=None, pct=None)
    m = _COUNT_WITH_PCT.match(s)
    if not m:
        return Cell(raw=s, n=None, pct=None)
    n = int(m.group(1).replace(',', ''))
    pct = float(m.group(2)) if m.group(2) else None
    return Cell(raw=s, n=n, pct=pct)


def format_cell(n: int, pct: Optional[float] = None) -> str:
    """Render a count (+ optional percentage) in Table One style."""
    if pct is None:
        return f'{n:,}'
    return f'{n:,} ({pct:.1f}%)'


# ---------------------------------------------------------------------------
# Variable-prefix parsing
# ---------------------------------------------------------------------------

def split_variable(row_label: str) -> tuple[str, str]:
    """Split a Table One row label into ``(variable, value)``.

    Only "Prefix: value" labels are treated as groupable; everything else
    returns ``('', label)`` (ungrouped — no merging possible, but
    cell-level suppression still applies).

    Examples:
        "Race: White"              -> ("Race", "White")
        "Admission type: osh"      -> ("Admission type", "osh")
        "Age at admission, median" -> ("", "Age at admission, median")
        "ICU encounters, n (%)"    -> ("", "ICU encounters, n (%)")
    """
    if ': ' not in row_label:
        return '', row_label
    variable, value = row_label.split(': ', 1)
    return variable.strip(), value.strip()


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def apply_merges(df: pd.DataFrame, rules: MergeRules) -> pd.DataFrame:
    """Collapse source rows into merged rows per the rules.

    Operates on a Table One DataFrame whose first column is ``Variable``
    and remaining columns are count strata. Rows without a ``Variable:
    value`` label pass through untouched.

    Percentages on merged cells are recomputed against the same column
    total as the source rows (inferred from any sibling row whose pct is
    known, falling back to no-percentage on the merged cell).
    """
    if df.empty:
        return df
    var_col = df.columns[0]
    data_cols = list(df.columns[1:])
    out_rows: list[dict] = []
    skip = set()

    for i, row in df.iterrows():
        if i in skip:
            continue
        label = str(row[var_col])
        variable, value = split_variable(label)
        group_rules = rules.variables.get(variable) if variable else None
        if not group_rules:
            out_rows.append(row.to_dict())
            continue
        # Find which merged group (if any) this row belongs to
        merged_label: Optional[str] = None
        for m_label, sources in group_rules.items():
            if value in sources:
                merged_label = m_label
                break
        if merged_label is None:
            out_rows.append(row.to_dict())
            continue
        # Gather all rows in this variable that belong to this merged group
        merged_values = set(group_rules[merged_label])
        merged_row = {var_col: f'{variable}: {merged_label}'}
        for col in data_cols:
            total_n = 0
            any_count = False
            any_pct_source = None
            for j, other in df.iterrows():
                if j < i:
                    continue  # Already emitted unmerged; skip
                ov = str(other[var_col])
                ovar, oval = split_variable(ov)
                if ovar == variable and oval in merged_values:
                    c = parse_cell(other[col])
                    if c.is_count:
                        total_n += c.n
                        any_count = True
                        if c.pct is not None and any_pct_source is None:
                            any_pct_source = (c.n, c.pct)
                    skip.add(j)
            if not any_count:
                merged_row[col] = ''
                continue
            # Recompute percentage if we can infer the column total from any
            # contributing source row with a known pct
            if rules.suppression.recompute_percentages and any_pct_source:
                src_n, src_pct = any_pct_source
                group_total = (src_n / src_pct * 100.0) if src_pct > 0 else None
                new_pct = (total_n / group_total * 100.0) if group_total else None
            else:
                new_pct = None
            merged_row[col] = format_cell(total_n, new_pct)
        out_rows.append(merged_row)

    return pd.DataFrame(out_rows, columns=df.columns).reset_index(drop=True)


def _group_row_indices(df: pd.DataFrame) -> dict[str, list[int]]:
    """Return a mapping ``variable -> list of row indices`` for group-aware
    operations. Rows whose label isn't in ``Prefix: value`` form are
    grouped under the empty string ``''`` (treated as ungrouped)."""
    var_col = df.columns[0]
    groups: dict[str, list[int]] = {}
    for i, label in enumerate(df[var_col].astype(str)):
        variable, _ = split_variable(label)
        groups.setdefault(variable, []).append(i)
    return groups


def apply_cell_suppression(
    df: pd.DataFrame, config: SuppressionConfig,
) -> tuple[pd.DataFrame, list[tuple[int, str, Optional[int]]]]:
    """Apply cell-level and complementary suppression to count cells.

    Returns the (possibly modified) DataFrame and a list of
    ``(row_index, column, complementary_target_row_index)`` tuples for
    logging — one entry per suppressed cell, with the sibling row index
    if complementary suppression fired.
    """
    if df.empty:
        return df, []
    out = df.copy()
    var_col = out.columns[0]
    data_cols = list(out.columns[1:])
    groups = _group_row_indices(out)
    log: list[tuple[int, str, Optional[int]]] = []

    for col in data_cols:
        for _variable, indices in groups.items():
            if _variable in PASS_THROUGH_VARIABLES:
                continue  # operational metadata rows (e.g. N: …) pass through
            # Parse all cells in this group+column
            cells = {i: parse_cell(out.at[i, col]) for i in indices}
            count_items = {i: c for i, c in cells.items() if c.is_count}
            if not count_items:
                continue
            small = [i for i, c in count_items.items() if 0 < c.n < config.threshold]
            if not small:
                continue
            # If the group total itself is < threshold, suppress the whole group
            group_total = sum(c.n for c in count_items.values())
            if group_total > 0 and group_total < config.threshold:
                for i in count_items:
                    out.at[i, col] = config.token
                    log.append((i, col, None))
                continue
            if len(small) == 1 and config.apply_complementary:
                # Suppress the small cell + the smallest non-suppressed sibling
                victim = small[0]
                remaining = {i: c.n for i, c in count_items.items() if i != victim}
                if remaining:
                    complementary = min(remaining, key=lambda k: remaining[k])
                    out.at[victim, col] = config.token
                    out.at[complementary, col] = config.token
                    log.append((victim, col, complementary))
                    log.append((complementary, col, victim))
                else:
                    out.at[victim, col] = config.token
                    log.append((victim, col, None))
            else:
                # Two or more are already small — suppress them all; no
                # complementary needed
                for i in small:
                    out.at[i, col] = config.token
                    log.append((i, col, None))
    return out, log


def suppress_dataframe(
    df: pd.DataFrame, rules: MergeRules,
) -> tuple[pd.DataFrame, list[tuple[int, str, Optional[int]]]]:
    """Apply merges, then cell suppression. Returns (df, suppression_log)."""
    merged = apply_merges(df, rules)
    return apply_cell_suppression(merged, rules.suppression)


# ---------------------------------------------------------------------------
# Tree walk (intermediate → final)
# ---------------------------------------------------------------------------

#: Glob pattern for Table One result CSVs. The suppression + calibration
#: flow intentionally targets only files matching this pattern — other
#: analytical CSVs the generator writes alongside Table One (mortality
#: rates, strobe counts, comorbidities, etc.) aren't in scope and stay in
#: ``output/final/`` unsuppressed.
TABLEONE_CSV_GLOB = 'table_one_*.csv'

#: Variable-prefix labels that are **operational metadata**, not patient-
#: level counts: e.g. ``N: Hospitals``, ``N: Encounter blocks``,
#: ``N: Unique patients``. These rows pass through suppression unchanged
#: — they aren't categorical siblings that sum to a total (Hospitals,
#: encounters, and patients are independent metrics) and ``<10`` isn't a
#: meaningful suppression for a 1-site cohort.
PASS_THROUGH_VARIABLES = frozenset({'N'})


def _csv_files(root: Path, pattern: str = TABLEONE_CSV_GLOB) -> list[Path]:
    if not root.exists():
        return []
    return [p for p in root.rglob(pattern) if p.is_file()]


def apply_suppression_to_tree(
    intermediate_root: Path,
    final_root: Path,
    rules: MergeRules,
) -> list[Path]:
    """Read every CSV under ``intermediate_root`` and write its suppressed
    counterpart under ``final_root``, preserving the directory layout.

    Non-CSV files are not copied. Returns the list of written paths.
    """
    written: list[Path] = []
    intermediate_root = intermediate_root.resolve()
    for src in _csv_files(intermediate_root):
        rel = src.resolve().relative_to(intermediate_root)
        dst = final_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            df = pd.read_csv(src, dtype=str, keep_default_na=False)
        except Exception:
            # Not parseable as a DataFrame (empty / corrupt) — copy through
            shutil.copy2(src, dst)
            written.append(dst)
            continue
        if df.empty or len(df.columns) < 2:
            # Nothing to suppress; pass through
            df.to_csv(dst, index=False)
            written.append(dst)
            continue
        suppressed, _log = suppress_dataframe(df, rules)
        suppressed.to_csv(dst, index=False)
        written.append(dst)
    return written


def scan_small_cells(
    intermediate_root: Path,
    rules: MergeRules,
    cohort: str = '',
) -> list[SmallCell]:
    """Walk ``intermediate_root`` and report every small cell after rules apply.

    ``cohort`` is stamped on every returned record for the UI; the caller
    typically walks both CI and Ward intermediate trees and tags each.
    """
    out: list[SmallCell] = []
    intermediate_root = intermediate_root.resolve()
    for src in _csv_files(intermediate_root):
        rel = src.resolve().relative_to(intermediate_root)
        stratum = _infer_stratum_from_path(rel)
        try:
            df = pd.read_csv(src, dtype=str, keep_default_na=False)
        except Exception:
            continue
        if df.empty or len(df.columns) < 2:
            continue
        # Snapshot of pre-merge counts by (variable, value) so we can tag
        # a cell with its original raw_n and merged_n side-by-side.
        var_col = df.columns[0]
        data_cols = list(df.columns[1:])
        raw_index: dict[tuple[str, str, str], int] = {}
        for i, label in enumerate(df[var_col].astype(str)):
            var, val = split_variable(label)
            for col in data_cols:
                c = parse_cell(df.at[i, col])
                if c.is_count:
                    raw_index[(var, val, col)] = c.n
        merged = apply_merges(df, rules)
        config = rules.suppression
        groups = _group_row_indices(merged)
        for col in data_cols:
            for variable, indices in groups.items():
                if variable in PASS_THROUGH_VARIABLES:
                    continue  # operational metadata rows — no suppression
                # Gather cells in the merged view for this group+col
                cells = {i: parse_cell(merged.at[i, col]) for i in indices}
                count_items = {i: c for i, c in cells.items() if c.is_count}
                if not count_items:
                    continue
                group_total = sum(c.n for c in count_items.values())
                small_cells = [i for i, c in count_items.items() if 0 < c.n < config.threshold]
                if not small_cells:
                    continue
                group_suppressed = group_total > 0 and group_total < config.threshold
                complementary = None
                if (not group_suppressed and len(small_cells) == 1
                        and config.apply_complementary):
                    remaining = {i: c.n for i, c in count_items.items() if i not in small_cells}
                    if remaining:
                        complementary = min(remaining, key=lambda k: remaining[k])
                for i in small_cells:
                    merged_label_raw = str(merged.at[i, var_col])
                    merged_var, merged_val = split_variable(merged_label_raw)
                    # Look up raw_n from the pre-merge index if the row
                    # wasn't itself the result of a merge
                    raw_n = raw_index.get((merged_var, merged_val, col))
                    merged_n = count_items[i].n
                    status = ('group_suppressed' if group_suppressed else 'still_small')
                    out.append(SmallCell(
                        cohort=cohort,
                        stratum=stratum,
                        csv_path=str(rel),
                        column=col,
                        variable=merged_var,
                        row=merged_val,
                        raw_n=raw_n if raw_n is not None else merged_n,
                        merged_n=None if raw_n is None else merged_n,
                        merged_label=merged_label_raw if raw_n is None else None,
                        status=status,
                        complementary_target=(
                            str(merged.at[complementary, var_col])
                            if complementary is not None else None
                        ),
                    ))
    return out


def _infer_stratum_from_path(rel: Path) -> str:
    """Derive a friendly stratum label from a Table One CSV's relative path.

    Under ``output/intermediate/tableone/``:
      ``overall/foo.csv``             → ``'overall'``
      ``strata/icu/foo.csv``          → ``'icu'``
      ``overall_ward/foo.csv``        → ``'overall_ward'``
      ``overall_ward/strata/icu/foo`` → ``'ward_icu'``
    """
    parts = rel.parts
    if not parts:
        return ''
    if parts[0] == 'overall':
        return 'overall'
    if parts[0] == 'overall_ward':
        if len(parts) >= 3 and parts[1] == 'strata':
            return f'ward_{parts[2]}'
        return 'overall_ward'
    if parts[0] == 'strata' and len(parts) >= 2:
        return parts[1]
    return parts[0]
