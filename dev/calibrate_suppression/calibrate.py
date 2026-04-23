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
    MergeRules, PASS_THROUGH_VARIABLES, TABLEONE_CSV_GLOB,
    apply_merges, parse_cell, split_variable,
)
import json  # noqa: E402

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
        # Skip operational-metadata rows (N: Hospitals, N: Encounter blocks,
        # N: Unique patients) — they aren't patient-level counts and don't
        # benefit from merging/suppression analysis.
        if variable in PASS_THROUGH_VARIABLES:
            continue
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


def _cohort_from_path(rel: str) -> str:
    """Derive a cohort label from the CSV's path within a site tree.

    Expects paths like ``tableone/overall/table_one_overall.csv`` or
    ``tableone/strata/icu/table_one_icu_by_year.csv``. The label is used
    to let the HTML reviewer filter 'overall only' vs 'strata'.
    """
    parts = rel.split('/')
    if 'strata' in parts:
        i = parts.index('strata')
        # overall_ward/strata/icu/... vs strata/icu/...
        cohort = 'ward_' if 'overall_ward' in parts[:i] else ''
        if i + 1 < len(parts):
            return f"{cohort}strata:{parts[i + 1]}"
        return f"{cohort}strata"
    if 'overall_ward' in parts:
        return 'overall_ward'
    if 'overall' in parts:
        return 'overall'
    return 'other'


def write_html_review(
    matrix: pd.DataFrame,
    site_names: list[str],
    dest: Path,
    rules: MergeRules,
) -> None:
    """Emit a single-file, self-contained HTML review page.

    Records are **aggregated by (cohort, variable, row)**. For each cell,
    we report per-site how many data columns were small and how many
    reported any count, so year-breakdown tables with 20+ year columns
    collapse to one row like "mimic: 18/21 cols <10" instead of 20
    separate rows. Only (cohort, variable, row) entries with at least
    one small data column anywhere are included.

    The full per-column grain is still available in
    ``cross_site_review.csv`` for anyone who wants to drill in.
    """
    rule_lookup: dict[tuple[str, str], str] = {}
    for variable, groups in rules.variables.items():
        for merged_label, sources in groups.items():
            for src in sources:
                rule_lookup[(variable, src)] = merged_label

    # Aggregate by (cohort, variable, row). For each key, collect per-site
    # totals of (small_cols, total_cols) across every csv_file and column
    # the row appears in for that cohort.
    from collections import defaultdict
    # key -> {site: [small, total]}
    agg: dict[tuple[str, str, str], dict[str, list[int]]] = defaultdict(
        lambda: {s: [0, 0] for s in site_names}
    )
    for _, r in matrix.iterrows():
        cohort = _cohort_from_path(str(r['csv_file']))
        var = str(r['variable'])
        row = str(r['row'])
        key = (cohort, var, row)
        for s in site_names:
            v = r[s]
            if v == '' or v is None:
                continue  # site didn't report this cell
            agg[key][s][1] += 1  # total_cols
            if v == rules.suppression.token or (
                isinstance(v, (int, float)) and 0 < v < rules.suppression.threshold
            ):
                agg[key][s][0] += 1  # small_cols

    records = []
    for (cohort, var, row), per_site in agg.items():
        n_sites_small = sum(1 for st in per_site.values() if st[0] > 0)
        n_sites_reporting = sum(1 for st in per_site.values() if st[1] > 0)
        if n_sites_small == 0:
            continue  # aggregation row with no small cells anywhere — skip
        merged_to = rule_lookup.get((var, row))
        rec = {
            'cohort': cohort,
            'variable': var,
            'row': row,
            'sites': {
                s: ({'small': st[0], 'total': st[1]} if st[1] > 0 else None)
                for s, st in per_site.items()
            },
            'n_sites_small': n_sites_small,
            'n_sites_reporting': n_sites_reporting,
            'status': 'in_rules' if merged_to else 'residual',
            'merged_to': merged_to,
        }
        # Flat per-site small-count keys so the site columns in the HTML
        # table can be sorted directly (sorter reads r[sortKey]).
        for s, st in per_site.items():
            rec[f'site_{s}_small'] = st[0]
        records.append(rec)

    cohorts = sorted({rec['cohort'] for rec in records})
    threshold = rules.suppression.threshold
    token = rules.suppression.token
    total_rows = len(records)
    resolved = sum(1 for rec in records if rec['status'] == 'in_rules')
    residual = total_rows - resolved

    # Capture the current YAML rules as a JSON-shape object + the
    # leading-comment block so the downloaded edited YAML preserves the
    # maintainer-facing docs.
    rules_json = {
        'version': rules.version,
        'variables': {
            var: {g: list(srcs) for g, srcs in groups.items()}
            for var, groups in rules.variables.items()
        },
        'suppression': {
            'threshold': rules.suppression.threshold,
            'token': rules.suppression.token,
            'recompute_percentages': rules.suppression.recompute_percentages,
            'apply_complementary': rules.suppression.apply_complementary,
        },
    }
    header_comments = _extract_yaml_header(RULES_PATH)

    payload = {
        'sites': site_names,
        'cohorts': cohorts,
        'threshold': threshold,
        'token': token,
        'total_small': total_rows,   # now "aggregated rows with ≥1 small cell"
        'resolved': resolved,
        'residual': residual,
        'records': records,
        'rules': rules_json,
        'yaml_header': header_comments,
    }

    html = _render_html(payload)
    dest.write_text(html, encoding='utf-8')


def _extract_yaml_header(path: Path) -> str:
    """Return the leading comment block (consecutive ``#`` lines + a
    blank line) at the top of the YAML file so the download can
    reproduce it. Returns empty string if the file doesn't exist yet or
    has no leading comments."""
    if not path.exists():
        return ''
    lines = path.read_text(encoding='utf-8').splitlines()
    out = []
    for line in lines:
        if line.startswith('#') or line.strip() == '':
            out.append(line)
            if len(out) > 1 and out[-1].strip() == '' and out[-2].startswith('#'):
                # First blank after a block of comments — include and stop
                break
        else:
            break
    return '\n'.join(out)


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Cross-site small-cell review</title>
<style>
  :root {
    --bg: #0f172a; --surface: #1e293b; --border: #334155;
    --text: #e2e8f0; --muted: #94a3b8;
    --accent: #60a5fa; --success: #34d399; --warning: #fbbf24; --danger: #f87171;
  }
  @media (prefers-color-scheme: light) {
    :root {
      --bg: #f8fafc; --surface: #ffffff; --border: #e2e8f0;
      --text: #0f172a; --muted: #64748b;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; padding: 24px; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: var(--bg); color: var(--text); font-size: 14px; line-height: 1.5; }
  h1 { margin: 0 0 8px; font-size: 22px; }
  .subtitle { color: var(--muted); margin-bottom: 20px; }
  .summary { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px; }
  .card { background: var(--surface); border: 1px solid var(--border);
          border-radius: 8px; padding: 12px 16px; min-width: 140px; }
  .card .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
  .card .value { font-size: 22px; font-weight: 600; margin-top: 4px; }
  .card.total .value { color: var(--accent); }
  .card.resolved .value { color: var(--success); }
  .card.residual .value { color: var(--danger); }
  .filters { background: var(--surface); border: 1px solid var(--border);
             border-radius: 8px; padding: 16px; margin-bottom: 16px;
             display: flex; gap: 20px; flex-wrap: wrap; align-items: center; }
  .filter-group { display: flex; flex-direction: column; gap: 4px; }
  .filter-group label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.5px; }
  .filter-group select, .filter-group input[type="search"] {
    background: var(--bg); color: var(--text); border: 1px solid var(--border);
    border-radius: 4px; padding: 6px 8px; font-size: 13px; min-width: 140px;
  }
  .filter-group.checkboxes { flex-direction: row; gap: 12px; align-items: center; }
  .filter-group.checkboxes label { text-transform: none; font-size: 13px; color: var(--text);
                                   display: flex; gap: 6px; align-items: center; letter-spacing: 0; }
  .result-count { margin-left: auto; color: var(--muted); font-size: 13px; }
  table { width: 100%; border-collapse: collapse; background: var(--surface);
          border: 1px solid var(--border); border-radius: 8px; overflow: hidden; font-size: 13px; }
  thead { background: var(--bg); position: sticky; top: 0; z-index: 1; }
  th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border); }
  th { font-weight: 600; font-size: 12px; cursor: pointer; user-select: none; white-space: nowrap; }
  th:hover { background: var(--surface); }
  th .sort-arrow { opacity: 0.4; margin-left: 4px; }
  th.sorted .sort-arrow { opacity: 1; color: var(--accent); }
  td.num { text-align: right; font-variant-numeric: tabular-nums; }
  td.small-cell { color: var(--danger); font-weight: 600; }
  td.not-reporting { color: var(--muted); font-style: italic; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 500; white-space: nowrap; }
  .badge.in-rules { background: rgba(52, 211, 153, 0.15); color: var(--success); }
  .badge.residual { background: rgba(248, 113, 113, 0.15); color: var(--danger); }
  .badge.pending { background: rgba(251, 191, 36, 0.15); color: var(--warning); margin-left: 4px; }
  .cohort-badge { font-size: 11px; color: var(--muted); white-space: nowrap; }
  code { background: var(--bg); padding: 1px 4px; border-radius: 3px; font-size: 12px; }
  .empty { text-align: center; padding: 40px; color: var(--muted); }
  .action-btn { background: var(--surface); color: var(--text); border: 1px solid var(--border);
                border-radius: 4px; padding: 2px 8px; font-size: 11px; cursor: pointer; }
  .action-btn:hover { border-color: var(--accent); color: var(--accent); }
  .action-btn.add { color: var(--success); border-color: rgba(52, 211, 153, 0.3); }
  .action-btn.remove { color: var(--danger); border-color: rgba(248, 113, 113, 0.3); }
  .popover { background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
             padding: 10px; margin-top: 6px; display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
  .popover select, .popover input { background: var(--surface); color: var(--text);
                                     border: 1px solid var(--border); border-radius: 4px; padding: 4px 8px;
                                     font-size: 12px; }
  .footer { position: fixed; bottom: 0; left: 0; right: 0; background: var(--surface);
            border-top: 1px solid var(--border); padding: 12px 24px; display: none;
            justify-content: space-between; align-items: center; gap: 16px; z-index: 5;
            box-shadow: 0 -4px 16px rgba(0,0,0,0.25); }
  .footer.active { display: flex; }
  .footer .count { font-weight: 600; color: var(--warning); }
  .btn-primary { background: var(--accent); color: #0f172a; border: 0; border-radius: 6px;
                 padding: 8px 16px; font-weight: 600; font-size: 13px; cursor: pointer; }
  .btn-primary:hover { opacity: 0.9; }
  .btn-outline { background: transparent; color: var(--text); border: 1px solid var(--border);
                 border-radius: 6px; padding: 8px 16px; font-size: 13px; cursor: pointer; }
  .btn-outline:hover { border-color: var(--danger); color: var(--danger); }
  .tabs { display: flex; gap: 4px; margin-bottom: 16px; border-bottom: 1px solid var(--border); }
  .tab { background: transparent; color: var(--muted); border: 0; border-bottom: 2px solid transparent;
         padding: 10px 16px; font-size: 14px; cursor: pointer; font-weight: 500; }
  .tab:hover { color: var(--text); }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }
</style>
</head>
<body>
  <h1>Cross-site small-cell review</h1>
  <div class="subtitle">Threshold: <code>{THRESHOLD}</code> · Token: <code>{TOKEN}</code> · Sites: <code>{SITES_CSV}</code>
    <br><span id="view-description">Rows are aggregated by <code>(cohort, variable, row)</code>. Each site cell shows
    <strong>small / total</strong> — number of data columns where N&lt;{THRESHOLD} over the number of columns reporting any count (across all Table One files + year columns in this cohort).</span></div>

  <div class="tabs" role="tablist">
    <button class="tab active" id="tab-cohort" data-view="cohort">By cohort</button>
    <button class="tab" id="tab-unique" data-view="unique">Unique rows (for merge-rule decisions)</button>
  </div>

  <div class="summary">
    <div class="card total"><div class="label">Rows with ≥1 small cell</div><div class="value" id="total">–</div></div>
    <div class="card resolved"><div class="label">Covered by YAML</div><div class="value" id="resolved">–</div></div>
    <div class="card residual"><div class="label">Residual (needs rule or suppression)</div><div class="value" id="residual">–</div></div>
  </div>

  <div class="filters">
    <div class="filter-group">
      <label for="f-cohort">Cohort</label>
      <select id="f-cohort"><option value="">(all)</option></select>
    </div>
    <div class="filter-group">
      <label for="f-variable">Variable</label>
      <select id="f-variable"><option value="">(all)</option></select>
    </div>
    <div class="filter-group">
      <label for="f-status">Status</label>
      <select id="f-status">
        <option value="">(all)</option>
        <option value="residual">Residual only</option>
        <option value="in_rules">Covered only</option>
      </select>
    </div>
    <div class="filter-group">
      <label for="f-min-sites">Min sites small</label>
      <select id="f-min-sites">
        <option value="1">≥ 1</option>
        <option value="2">≥ 2</option>
      </select>
    </div>
    <div class="filter-group">
      <label for="f-search">Search row/variable</label>
      <input type="search" id="f-search" placeholder="e.g. stepdown">
    </div>
    <div class="filter-group checkboxes">
      <label><input type="checkbox" id="f-overall-only"> Overall tables only (hide strata)</label>
      <label><input type="checkbox" id="f-consistent"> Small across all reporting sites</label>
    </div>
    <div class="result-count" id="result-count"></div>
  </div>

  <table id="tbl">
    <thead>
      <tr>
        <th data-key="cohort">Cohort<span class="sort-arrow">↕</span></th>
        <th data-key="variable">Variable<span class="sort-arrow">↕</span></th>
        <th data-key="row">Row<span class="sort-arrow">↕</span></th>
        {SITE_HEADERS}
        <th data-key="n_sites_small" class="sorted">Small sites<span class="sort-arrow">↓</span></th>
        <th data-key="status">Status<span class="sort-arrow">↕</span></th>
        <th>Action</th>
      </tr>
    </thead>
    <tbody id="tbody"></tbody>
  </table>
  <div id="empty" class="empty" style="display:none;">No rows match your filters.</div>

  <div class="footer" id="footer">
    <div><span class="count" id="pending-count">0</span> pending change(s) — original rules diff awaiting export.</div>
    <div style="display:flex; gap:8px;">
      <button class="btn-outline" id="btn-reset">Reset</button>
      <button class="btn-primary" id="btn-download">Download tableone_merge_rules.yaml</button>
    </div>
  </div>

<script id="data" type="application/json">{DATA_JSON}</script>
<script>
const payload = JSON.parse(document.getElementById('data').textContent);
const SITES = payload.sites;
const TOKEN = payload.token;
const rows = payload.records;
const YAML_HEADER = payload.yaml_header || '';
const STORAGE_KEY = 'tableone-calibrate';

// Deep-clone the original rules so we can diff against them to compute
// "pending changes". `workingRules` is the mutable copy the user edits.
const originalRules = JSON.parse(JSON.stringify(payload.rules));
let workingRules = loadWorkingRules();

function loadWorkingRules() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return JSON.parse(raw);
  } catch (e) { /* localStorage disabled or corrupt — fall back */ }
  return JSON.parse(JSON.stringify(payload.rules));
}
function saveWorkingRules() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(workingRules)); }
  catch (e) { /* best-effort */ }
}

// Look up whether (variable, row) is covered by a merge rule in the
// given rules object. Returns the merged-group label, or null.
function lookupMerge(rules, variable, row) {
  const groups = rules.variables && rules.variables[variable];
  if (!groups) return null;
  for (const [groupLabel, sources] of Object.entries(groups)) {
    if (Array.isArray(sources) && sources.includes(row)) return groupLabel;
  }
  return null;
}

document.getElementById('total').textContent = payload.total_small.toLocaleString();
document.getElementById('resolved').textContent = payload.resolved.toLocaleString();
document.getElementById('residual').textContent = payload.residual.toLocaleString();

// Populate filter dropdowns
const cohortSel = document.getElementById('f-cohort');
for (const c of payload.cohorts) {
  const opt = document.createElement('option'); opt.value = c; opt.textContent = c; cohortSel.appendChild(opt);
}
const varSel = document.getElementById('f-variable');
const vars = [...new Set(rows.map(r => r.variable).filter(Boolean))].sort();
for (const v of vars) {
  const opt = document.createElement('option'); opt.value = v; opt.textContent = v; varSel.appendChild(opt);
}

// Sort state
let sortKey = 'n_sites_small';
let sortDir = 'desc';
let view = 'cohort';   // 'cohort' or 'unique'

// Aggregate the by-cohort rows to one record per (variable, row).
// Used by the "Unique rows" tab — canonical view for merge-rule
// decisions since merge rules don't vary by cohort.
function aggregateUnique(rows) {
  const map = new Map();
  for (const r of rows) {
    const key = (r.variable || '') + '\x1f' + r.row;
    if (!map.has(key)) {
      map.set(key, {
        variable: r.variable,
        row: r.row,
        sites: Object.fromEntries(SITES.map(s => [s, {small: 0, total: 0}])),
        cohorts: new Set(),
        cohorts_small: new Set(),
      });
    }
    const agg = map.get(key);
    agg.cohorts.add(r.cohort);
    if (r.n_sites_small > 0) agg.cohorts_small.add(r.cohort);
    for (const s of SITES) {
      const v = r.sites[s];
      if (v) {
        agg.sites[s].small += v.small;
        agg.sites[s].total += v.total;
      }
    }
  }
  const out = [];
  for (const a of map.values()) {
    const n_sites_small = SITES.filter(s => a.sites[s].small > 0).length;
    const n_sites_reporting = SITES.filter(s => a.sites[s].total > 0).length;
    if (n_sites_small === 0) continue;
    const rec = {
      variable: a.variable,
      row: a.row,
      sites: a.sites,
      n_sites_small,
      n_sites_reporting,
      n_cohorts: a.cohorts.size,
      n_cohorts_small: a.cohorts_small.size,
    };
    for (const s of SITES) rec[`site_${s}_small`] = a.sites[s].small;
    out.push(rec);
  }
  return out;
}

const uniqueRows = aggregateUnique(rows);

function sortRows(arr) {
  const mult = sortDir === 'asc' ? 1 : -1;
  return arr.slice().sort((a, b) => {
    const va = a[sortKey] ?? '';
    const vb = b[sortKey] ?? '';
    if (typeof va === 'number' && typeof vb === 'number') return (va - vb) * mult;
    return String(va).localeCompare(String(vb)) * mult;
  });
}

function applyFilters() {
  const cohort = document.getElementById('f-cohort').value;
  const variable = document.getElementById('f-variable').value;
  const status = document.getElementById('f-status').value;
  const minSites = parseInt(document.getElementById('f-min-sites').value, 10);
  const search = document.getElementById('f-search').value.toLowerCase().trim();
  const overallOnly = document.getElementById('f-overall-only').checked;
  const consistent = document.getElementById('f-consistent').checked;

  const sourceRows = view === 'unique' ? uniqueRows : rows;

  let filtered = sourceRows.filter(r => {
    if (view === 'cohort') {
      if (cohort && r.cohort !== cohort) return false;
      if (overallOnly && !(r.cohort || '').startsWith('overall')) return false;
    }
    // Status is recomputed against the live workingRules so pending
    // edits filter correctly on both tabs.
    const workingGroup = r.variable ? lookupMerge(workingRules, r.variable, r.row) : null;
    const computedStatus = workingGroup ? 'in_rules' : 'residual';
    if (variable && r.variable !== variable) return false;
    if (status && computedStatus !== status) return false;
    if (r.n_sites_small < minSites) return false;
    if (consistent && r.n_sites_small !== r.n_sites_reporting) return false;
    if (search) {
      const hay = (r.variable + ' ' + r.row + ' ' + (r.cohort || '')).toLowerCase();
      if (!hay.includes(search)) return false;
    }
    return true;
  });
  filtered = sortRows(filtered);

  document.getElementById('result-count').textContent =
    `${filtered.length.toLocaleString()} of ${sourceRows.length.toLocaleString()} rows`;

  const tbody = document.getElementById('tbody');
  const empty = document.getElementById('empty');
  if (filtered.length === 0) {
    tbody.innerHTML = '';
    empty.style.display = 'block';
    return;
  }
  empty.style.display = 'none';

  const html = filtered.map(r => {
    const siteCells = SITES.map(s => {
      const v = r.sites[s];
      if (!v || v.total === 0) return '<td class="num not-reporting">–</td>';
      if (v.small === 0) return `<td class="num">0 / ${v.total}</td>`;
      const pct = v.total ? Math.round(v.small / v.total * 100) : 0;
      return `<td class="num small-cell" title="${v.small} of ${v.total} data columns below threshold (${pct}%)"><strong>${v.small}</strong> / ${v.total}</td>`;
    }).join('');
    const workingGroup = r.variable ? lookupMerge(workingRules, r.variable, r.row) : null;
    const originalGroup = r.variable ? lookupMerge(originalRules, r.variable, r.row) : null;
    const isPending = workingGroup !== originalGroup;
    const statusBadge = workingGroup
      ? `<span class="badge in-rules">→ ${escapeHtml(workingGroup)}</span>`
      : `<span class="badge residual">residual</span>`;
    const pendingTag = isPending ? '<span class="badge pending">pending</span>' : '';
    let action = '';
    if (r.variable) {
      if (workingGroup) {
        action = `<button class="action-btn remove" data-v="${escapeAttr(r.variable)}" data-r="${escapeAttr(r.row)}" data-g="${escapeAttr(workingGroup)}" onclick="removeFromMerge(this)">✕ remove</button>`;
      } else {
        action = `<button class="action-btn add" data-v="${escapeAttr(r.variable)}" data-r="${escapeAttr(r.row)}" onclick="openMergePopover(this)">＋ merge</button>`;
      }
    }
    if (view === 'unique') {
      return `<tr>
        <td>${escapeHtml(r.variable) || '<em>(ungrouped)</em>'}</td>
        <td><strong>${escapeHtml(r.row)}</strong></td>
        ${siteCells}
        <td class="num"><strong>${r.n_cohorts_small}</strong>/${r.n_cohorts}</td>
        <td class="num"><strong>${r.n_sites_small}</strong>/${r.n_sites_reporting}</td>
        <td>${statusBadge}${pendingTag}</td>
        <td>${action}</td>
      </tr>`;
    }
    return `<tr>
      <td><span class="cohort-badge">${escapeHtml(r.cohort)}</span></td>
      <td>${escapeHtml(r.variable) || '<em>(ungrouped)</em>'}</td>
      <td><strong>${escapeHtml(r.row)}</strong></td>
      ${siteCells}
      <td class="num"><strong>${r.n_sites_small}</strong>/${r.n_sites_reporting}</td>
      <td>${statusBadge}${pendingTag}</td>
      <td>${action}</td>
    </tr>`;
  }).join('');
  tbody.innerHTML = html;
}

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[c]);
}
function escapeAttr(s) {
  return String(s ?? '').replace(/"/g, '&quot;').replace(/</g, '&lt;');
}

// ─── Interactive merge editing ─────────────────────────────────────
function pendingCount() {
  // Count (variable, row, group) triples that differ between original
  // and working rules. Treat absence as a distinct state.
  let count = 0;
  const visited = new Set();
  const walk = (rules, tag) => {
    for (const [v, groups] of Object.entries(rules.variables || {})) {
      for (const [g, sources] of Object.entries(groups)) {
        for (const row of sources) {
          const key = `${v}  ${row}`;
          const other = (tag === 'work') ? originalRules : workingRules;
          const otherGroup = lookupMerge(other, v, row);
          if (otherGroup !== g && !visited.has(key)) {
            visited.add(key);
            count++;
          }
        }
      }
    }
  };
  walk(workingRules, 'work');
  walk(originalRules, 'orig');
  return count;
}

function updateFooter() {
  const n = pendingCount();
  document.getElementById('pending-count').textContent = n;
  document.getElementById('footer').classList.toggle('active', n > 0);
}

window.openMergePopover = function(btn) {
  const variable = btn.dataset.v;
  const row = btn.dataset.r;
  const existing = (workingRules.variables && workingRules.variables[variable]) || {};
  const groups = Object.keys(existing);
  const cell = btn.parentElement;
  const pop = document.createElement('div');
  pop.className = 'popover';
  pop.innerHTML = `
    <select class="group-select">
      ${groups.map(g => `<option value="${escapeAttr(g)}">${escapeHtml(g)}</option>`).join('')}
      <option value="__new__">＋ new group…</option>
    </select>
    <input type="text" class="new-group" placeholder="New group label" style="display:none;min-width:160px;">
    <button class="btn-primary confirm-btn" style="padding:4px 10px;font-size:12px;">Add</button>
    <button class="btn-outline cancel-btn" style="padding:4px 10px;font-size:12px;">Cancel</button>
  `;
  btn.style.display = 'none';
  cell.appendChild(pop);
  const sel = pop.querySelector('.group-select');
  const newInput = pop.querySelector('.new-group');
  sel.addEventListener('change', () => {
    newInput.style.display = sel.value === '__new__' ? '' : 'none';
    if (sel.value === '__new__') newInput.focus();
  });
  if (groups.length === 0) { sel.value = '__new__'; newInput.style.display = ''; }
  pop.querySelector('.confirm-btn').addEventListener('click', () => {
    let group = sel.value;
    if (group === '__new__') {
      group = newInput.value.trim();
      if (!group) { newInput.focus(); return; }
    }
    addToMerge(variable, row, group);
    applyFilters();   // re-render the row with new state
    updateFooter();
  });
  pop.querySelector('.cancel-btn').addEventListener('click', () => {
    cell.removeChild(pop);
    btn.style.display = '';
  });
};

window.removeFromMerge = function(btn) {
  const variable = btn.dataset.v;
  const row = btn.dataset.r;
  const group = btn.dataset.g;
  const groups = workingRules.variables[variable];
  if (!groups || !groups[group]) return;
  groups[group] = groups[group].filter(s => s !== row);
  if (groups[group].length === 0) delete groups[group];
  if (Object.keys(groups).length === 0) delete workingRules.variables[variable];
  saveWorkingRules();
  applyFilters();
  updateFooter();
};

function addToMerge(variable, row, group) {
  workingRules.variables = workingRules.variables || {};
  workingRules.variables[variable] = workingRules.variables[variable] || {};
  workingRules.variables[variable][group] = workingRules.variables[variable][group] || [];
  if (!workingRules.variables[variable][group].includes(row)) {
    workingRules.variables[variable][group].push(row);
  }
  saveWorkingRules();
}

// ─── YAML download ──────────────────────────────────────────────────
function toYAML(rules) {
  const lines = [];
  if (YAML_HEADER) {
    lines.push(YAML_HEADER.trimEnd(), '');
  }
  lines.push(`version: ${rules.version || 1}`, '');
  const needsQuote = s => /[:#{}[\],&*!|>'"%@`]/.test(s) || /^\s|\s$/.test(s)
    || /^[-?&*\d]/.test(s) || s.trim() === '' || /^(true|false|null|yes|no|on|off)$/i.test(s);
  const q = s => needsQuote(s) ? JSON.stringify(s) : s;
  for (const [v, groups] of Object.entries(rules.variables || {})) {
    lines.push(`${q(v)}:`);
    for (const [g, sources] of Object.entries(groups)) {
      lines.push(`  ${q(g)}:`);
      for (const src of sources) {
        lines.push(`    - ${q(src)}`);
      }
    }
    lines.push('');
  }
  const sup = rules.suppression || {};
  lines.push('suppression:');
  lines.push(`  threshold: ${sup.threshold ?? 10}`);
  lines.push(`  token: ${JSON.stringify(sup.token ?? '<10')}`);
  lines.push(`  recompute_percentages: ${sup.recompute_percentages !== false}`);
  lines.push(`  apply_complementary: ${sup.apply_complementary !== false}`);
  return lines.join('\n') + '\n';
}

document.getElementById('btn-download').addEventListener('click', () => {
  const yaml = toYAML(workingRules);
  const blob = new Blob([yaml], { type: 'text/yaml' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = 'tableone_merge_rules.yaml';
  a.click();
  URL.revokeObjectURL(url);
});

document.getElementById('btn-reset').addEventListener('click', () => {
  if (!confirm('Discard all pending changes and reset to the committed YAML?')) return;
  workingRules = JSON.parse(JSON.stringify(originalRules));
  saveWorkingRules();
  applyFilters();
  updateFooter();
});

// Sort handlers
document.querySelectorAll('th[data-key]').forEach(th => {
  th.addEventListener('click', () => {
    const key = th.dataset.key;
    if (sortKey === key) {
      sortDir = sortDir === 'asc' ? 'desc' : 'asc';
    } else {
      sortKey = key;
      sortDir = (key === 'n_sites_small') ? 'desc' : 'asc';
    }
    document.querySelectorAll('th[data-key]').forEach(h => {
      h.classList.toggle('sorted', h.dataset.key === sortKey);
      const arrow = h.querySelector('.sort-arrow');
      if (arrow) arrow.textContent = h.dataset.key === sortKey ? (sortDir === 'asc' ? '↑' : '↓') : '↕';
    });
    applyFilters();
  });
});

// Filter handlers
document.querySelectorAll('select, input[type="search"], input[type="checkbox"]').forEach(el =>
  el.addEventListener(el.type === 'search' ? 'input' : 'change', applyFilters));

function setView(next) {
  view = next;
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.view === view));
  const thead = document.querySelector('#tbl thead tr');
  const siteTh = SITES.map(s => `<th data-key="site_${s}_small">${s}<span class="sort-arrow">↕</span></th>`).join('');
  if (view === 'unique') {
    thead.innerHTML = `
      <th data-key="variable">Variable<span class="sort-arrow">↕</span></th>
      <th data-key="row">Row<span class="sort-arrow">↕</span></th>
      ${siteTh}
      <th data-key="n_cohorts_small">Cohorts small<span class="sort-arrow">↕</span></th>
      <th data-key="n_sites_small" class="sorted">Small sites<span class="sort-arrow">↓</span></th>
      <th data-key="status">Status<span class="sort-arrow">↕</span></th>
      <th>Action</th>`;
    document.getElementById('view-description').innerHTML =
      'One row per <code>(variable, row)</code>, aggregated across all cohorts. Site cells sum small/total across every Table One file. Use this tab to pick canonical merge rules — row actions here change rules globally.';
    // Hide cohort filter + "overall only" checkbox on this tab
    document.getElementById('f-cohort').parentElement.style.display = 'none';
    document.getElementById('f-overall-only').parentElement.parentElement.style.display = 'none';
  } else {
    thead.innerHTML = `
      <th data-key="cohort">Cohort<span class="sort-arrow">↕</span></th>
      <th data-key="variable">Variable<span class="sort-arrow">↕</span></th>
      <th data-key="row">Row<span class="sort-arrow">↕</span></th>
      ${siteTh}
      <th data-key="n_sites_small" class="sorted">Small sites<span class="sort-arrow">↓</span></th>
      <th data-key="status">Status<span class="sort-arrow">↕</span></th>
      <th>Action</th>`;
    document.getElementById('view-description').innerHTML =
      `Rows are aggregated by <code>(cohort, variable, row)</code>. Each site cell shows <strong>small / total</strong> — number of data columns where N&lt;${payload.threshold} over the number of columns reporting any count (across all Table One files + year columns in this cohort).`;
    document.getElementById('f-cohort').parentElement.style.display = '';
    document.getElementById('f-overall-only').parentElement.parentElement.style.display = '';
  }
  // Re-bind sort handlers on the new headers
  document.querySelectorAll('th[data-key]').forEach(th => {
    th.addEventListener('click', () => {
      const key = th.dataset.key;
      if (sortKey === key) sortDir = sortDir === 'asc' ? 'desc' : 'asc';
      else { sortKey = key; sortDir = (key === 'n_sites_small' || key === 'n_cohorts_small') ? 'desc' : 'asc'; }
      document.querySelectorAll('th[data-key]').forEach(h => {
        h.classList.toggle('sorted', h.dataset.key === sortKey);
        const arrow = h.querySelector('.sort-arrow');
        if (arrow) arrow.textContent = h.dataset.key === sortKey ? (sortDir === 'asc' ? '↑' : '↓') : '↕';
      });
      applyFilters();
    });
  });
  applyFilters();
}

document.querySelectorAll('.tab').forEach(t =>
  t.addEventListener('click', () => setView(t.dataset.view)));

applyFilters();
updateFooter();
</script>
</body>
</html>
"""


def _render_html(payload: dict) -> str:
    site_headers = ''.join(
        f'<th data-key="site_{s}_small">{s}<span class="sort-arrow">↕</span></th>'
        for s in payload['sites']
    )
    # Pretty-print JSON so the embedded script is readable if anyone peeks
    data_json = json.dumps(payload, ensure_ascii=False, default=str)
    html = _HTML_TEMPLATE
    html = html.replace('{THRESHOLD}', str(payload['threshold']))
    html = html.replace('{TOKEN}', payload['token'])
    html = html.replace('{SITES_CSV}', ', '.join(payload['sites']))
    html = html.replace('{SITE_HEADERS}', site_headers)
    html = html.replace('{DATA_JSON}', data_json)
    return html


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

    html_path = OUT_DIR / 'cross_site_review.html'
    write_html_review(matrix, [name for name, _ in sites], html_path, rules)
    print(f'  wrote {html_path.relative_to(_REPO_ROOT)}')

    n_small = int(matrix['n_sites_small'].ge(1).sum())
    print(f'\nSummary: {n_small} cells small in ≥ 1 site '
          f'(threshold = {threshold}, token = "{token}")')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
