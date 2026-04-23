"""Cross-site small-cell calibration.

Dev-only tool. See README.md for the workflow. Run from the repo root:

    .venv/bin/python dev/calibrate_suppression/calibrate.py

Reads every site subdirectory under ``dev/calibrate_suppression/sites/``,
scans each site's Table One CSVs, and emits:

 * ``out/cross_site_review.csv`` — one row per
   ``(csv_file, variable, row, data_column)`` with a column per site
   showing the count (or ``<10`` when below threshold, or blank when the
   site doesn't report that row).
 * ``out/candidate_merges.md`` — grouped by variable, listing rows that
   are small in ≥ 1 site (sorted by # sites small desc). These are the
   strongest candidates for canonical merging in
   ``config/tableone_merge_rules.yaml``.
 * ``out/current_rules_coverage.md`` — how well the currently-committed
   YAML handles the cross-site small-cell set (what it resolves by merge
   today vs. what still leaks).
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional

# Allow running directly (no PYTHONPATH gymnastics)
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402

from modules.tableone.suppression import (  # noqa: E402
    MergeRules, TABLEONE_CSV_GLOB, apply_merges, parse_cell, split_variable,
)

_HERE = Path(__file__).resolve().parent
SITES_DIR = _HERE / 'sites'
OUT_DIR = _HERE / 'out'
RULES_PATH = _REPO_ROOT / 'config' / 'tableone_merge_rules.yaml'


def discover_sites() -> list[tuple[str, Path]]:
    """Return (site_name, tableone_root) pairs, sorted by site name."""
    if not SITES_DIR.exists():
        return []
    out = []
    for child in sorted(SITES_DIR.iterdir()):
        if not child.is_dir() or child.name.startswith('.'):
            continue
        out.append((child.name, child))
    return out


def walk_csvs(site_root: Path) -> list[Path]:
    """Recursively gather every Table One result CSV under a site's tree.

    Only files matching the ``TABLEONE_CSV_GLOB`` pattern (``table_one_*.csv``)
    are considered — the other analytical CSVs the generator writes
    alongside Table One are not in the suppression scope.
    """
    return sorted(p for p in site_root.rglob(TABLEONE_CSV_GLOB) if p.is_file())


def site_csv_index(site_root: Path) -> dict[str, pd.DataFrame]:
    """Read every CSV under a site's Table One tree into a dict keyed by
    the path relative to the site root (e.g. ``'overall/table_one_overall.csv'``).
    """
    index = {}
    for p in walk_csvs(site_root):
        rel = str(p.relative_to(site_root))
        try:
            index[rel] = pd.read_csv(p, dtype=str, keep_default_na=False)
        except Exception as e:  # noqa: BLE001
            print(f'  [warn] could not parse {rel}: {e}', file=sys.stderr)
    return index


def build_matrix(
    site_dfs: dict[str, dict[str, pd.DataFrame]],
    threshold: int,
    token: str,
) -> pd.DataFrame:
    """Build a long-form matrix across all sites.

    Index = (csv_file, variable, row, data_column).
    Columns = one per site, values = N or ``token`` (when < threshold),
    plus ``n_sites_small`` (count of sites where this cell is small) and
    ``n_sites_reporting`` (count of sites where the cell has any count).
    """
    site_names = list(site_dfs.keys())
    # { (csv, var, row, col) -> { site_name -> int_or_None } }
    cells: dict[tuple[str, str, str, str], dict[str, Optional[int]]] = defaultdict(dict)
    for site, files in site_dfs.items():
        for rel, df in files.items():
            if df.empty or len(df.columns) < 2:
                continue
            var_col = df.columns[0]
            data_cols = list(df.columns[1:])
            for _, row in df.iterrows():
                label = str(row[var_col])
                variable, value = split_variable(label)
                for dc in data_cols:
                    c = parse_cell(row[dc])
                    if not c.is_count:
                        continue
                    cells[(rel, variable, value, dc)][site] = c.n
    records = []
    for (rel, variable, value, dc), per_site in cells.items():
        rec = {
            'csv_file': rel,
            'variable': variable,
            'row': value,
            'data_column': dc,
        }
        n_small = 0
        n_reporting = 0
        for site in site_names:
            n = per_site.get(site)
            if n is None:
                rec[site] = ''
                continue
            n_reporting += 1
            if 0 < n < threshold:
                rec[site] = token
                n_small += 1
            else:
                rec[site] = n
        rec['n_sites_reporting'] = n_reporting
        rec['n_sites_small'] = n_small
        records.append(rec)
    cols = ['csv_file', 'variable', 'row', 'data_column',
            *site_names, 'n_sites_reporting', 'n_sites_small']
    return pd.DataFrame(records, columns=cols)


def write_candidate_merges(matrix: pd.DataFrame, dest: Path,
                           current_rules: MergeRules) -> None:
    """Group candidate rows by variable (rows small in ≥1 site), flag any
    that are already covered by the canonical rules, and write a markdown
    report for the maintainer to review."""
    candidates = matrix[matrix['n_sites_small'] >= 1].copy()
    by_variable: dict[str, pd.DataFrame] = {}
    for variable, grp in candidates.groupby('variable'):
        if not variable:
            continue  # un-grouped rows (e.g. 'Age at admission, median') — can't be merged
        by_variable[variable] = grp

    site_cols = [c for c in matrix.columns
                 if c not in ('csv_file', 'variable', 'row', 'data_column',
                              'n_sites_reporting', 'n_sites_small')]

    lines = [
        '# Candidate merge rules for `config/tableone_merge_rules.yaml`',
        '',
        'Rows below are small in ≥ 1 site across the calibration cohort.',
        'Rows already covered by the current YAML are tagged **[in rules]**.',
        'Rows not yet covered are candidates to add — use clinical judgement.',
        '',
    ]
    for variable, grp in sorted(by_variable.items()):
        lines.append(f'## {variable}')
        lines.append('')
        rule_groups = current_rules.variables.get(variable, {})
        rule_lookup = {
            source: merged
            for merged, sources in rule_groups.items()
            for source in sources
        }
        # Aggregate per (row) across all csv_file/data_column occurrences
        agg = (grp.groupby('row')
               .agg(n_sites_small=('n_sites_small', 'max'),
                    n_sites_reporting=('n_sites_reporting', 'max'))
               .reset_index()
               .sort_values(['n_sites_small', 'row'],
                            ascending=[False, True]))
        for _, r in agg.iterrows():
            flag = ''
            if r['row'] in rule_lookup:
                flag = f'  **[in rules → {rule_lookup[r["row"]]}]**'
            lines.append(f'- `{r["row"]}` — small in '
                         f'{int(r["n_sites_small"])}/{int(r["n_sites_reporting"])} '
                         f'reporting sites{flag}')
        # Also list any current merges that no candidate hit (these are rules
        # that apparently aren't needed based on this calibration — worth a
        # look, but don't remove without human check)
        covered_labels = set(agg['row'])
        current_covered = [s for s in rule_lookup if s not in covered_labels]
        if current_covered:
            lines.append('')
            lines.append(
                f'  _Current rule also merges (but not seen small across calibration set):_ '
                f'{", ".join(f"`{x}`" for x in current_covered)}'
            )
        lines.append('')
    dest.write_text('\n'.join(lines), encoding='utf-8')


def write_rules_coverage(matrix: pd.DataFrame, dest: Path,
                         current_rules: MergeRules) -> None:
    """Report: of the small cells seen across the calibration set, how many
    does the currently-committed YAML resolve via its merge rules?"""
    site_cols = [c for c in matrix.columns
                 if c not in ('csv_file', 'variable', 'row', 'data_column',
                              'n_sites_reporting', 'n_sites_small')]
    small = matrix[matrix['n_sites_small'] >= 1]
    total_small = len(small)
    resolved_by_merge = 0
    residual_rows = []
    for _, r in small.iterrows():
        var = r['variable']
        val = r['row']
        rule_groups = current_rules.variables.get(var, {})
        is_covered = any(val in src for src in rule_groups.values())
        if is_covered:
            resolved_by_merge += 1
        else:
            residual_rows.append(r)
    lines = [
        '# Coverage of current canonical rules',
        '',
        f'Calibration set: {len(site_cols)} site(s).',
        f'Total (csv, variable, row, column) cells small in ≥ 1 site: '
        f'**{total_small}**',
        f'Resolved by current YAML merges: **{resolved_by_merge}**',
        f'Residual (still small after merges, would need cell-suppression '
        f'or a new merge): **{total_small - resolved_by_merge}**',
        '',
    ]
    if residual_rows:
        lines += ['## Residual cells',
                  '',
                  '| csv | variable | row | column | sites small |',
                  '|---|---|---|---|---|']
        for r in residual_rows[:200]:
            lines.append(
                f"| `{r['csv_file']}` | `{r['variable'] or '-'}` | "
                f"`{r['row']}` | `{r['data_column']}` | "
                f"{int(r['n_sites_small'])}/{int(r['n_sites_reporting'])} |"
            )
        if len(residual_rows) > 200:
            lines.append(f'\n… {len(residual_rows) - 200} more residual rows')
    dest.write_text('\n'.join(lines), encoding='utf-8')


def main() -> int:
    sites = discover_sites()
    if len(sites) < 2:
        print(
            f'Need at least 2 sites under {SITES_DIR.relative_to(_REPO_ROOT)}/, '
            f'got {len(sites)}. See README.md.',
            file=sys.stderr,
        )
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rules = MergeRules.from_yaml(RULES_PATH)
    threshold = rules.suppression.threshold
    token = rules.suppression.token

    print(f'Scanning {len(sites)} sites:', ', '.join(name for name, _ in sites))
    site_dfs = {name: site_csv_index(path) for name, path in sites}
    matrix = build_matrix(site_dfs, threshold, token)

    matrix_path = OUT_DIR / 'cross_site_review.csv'
    matrix.to_csv(matrix_path, index=False)
    print(f'  wrote {matrix_path.relative_to(_REPO_ROOT)} '
          f'({len(matrix)} rows)')

    candidate_path = OUT_DIR / 'candidate_merges.md'
    write_candidate_merges(matrix, candidate_path, rules)
    print(f'  wrote {candidate_path.relative_to(_REPO_ROOT)}')

    coverage_path = OUT_DIR / 'current_rules_coverage.md'
    write_rules_coverage(matrix, coverage_path, rules)
    print(f'  wrote {coverage_path.relative_to(_REPO_ROOT)}')

    n_small = int(matrix['n_sites_small'].ge(1).sum())
    print(f'\nSummary: {n_small} cells small in ≥ 1 site '
          f'(threshold = {threshold}, token = "{token}")')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
