"""
Interactive ECDF distribution viewer.

Generates self-contained HTML files with embedded plotly.js that let
users browse bins-histogram + ECDF overlays via a dropdown selector.

For strata with sub-strata (advanced_resp, vaso), the HTML shows
side-by-side panels: Overall | ICU | No ICU.
"""

import json
import os
from html import escape
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml


# ============================================================================
# Segment color palette (matches clinical reference ranges)
# ============================================================================

SEGMENT_COLORS = {
    'below': '#3182bd',
    'normal': '#31a354',
    'above': '#de2d26',
}
FLAT_COLOR = '#4292c6'
ECDF_COLOR = '#000000'


# ============================================================================
# Data loading
# ============================================================================

def load_lab_vital_config(config_path: str = 'meta/configs/lab_vital_config.yaml') -> dict:
    """Load lab/vital configuration for normal ranges."""
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def discover_files(
    base_dir: str,
    table_type: Optional[str] = None,
) -> List[Tuple[str, str, Optional[str]]]:
    """Discover available bins/ECDF parquet pairs.

    Returns list of (table_type, category, unit) tuples.
    Only discovers *base* metrics (no _icu/_no_icu suffixed files).
    """
    files = []
    tables = [table_type] if table_type else ['labs', 'vitals', 'respiratory_support']

    for table in tables:
        bins_dir = os.path.join(base_dir, 'bins', table)
        if not os.path.exists(bins_dir):
            continue

        for filename in sorted(os.listdir(bins_dir)):
            if not filename.endswith('.parquet'):
                continue
            basename = filename.replace('.parquet', '')

            # Skip sub-strata suffixed files — we discover those separately
            if basename.endswith('_icu') or basename.endswith('_no_icu'):
                continue

            if table == 'labs':
                parts = basename.rsplit('_', 1)
                if len(parts) == 2:
                    category, unit_safe = parts
                    unit = unit_safe.replace('_', '/')
                    files.append((table, category, unit))
            else:
                files.append((table, basename, None))

    return files


def _read_metric_data(
    base_dir: str,
    table_type: str,
    filename: str,
) -> Optional[Dict]:
    """Read bins + ecdf parquets for one metric, return as JSON-serializable dict."""
    bins_path = os.path.join(base_dir, 'bins', table_type, filename)
    ecdf_path = os.path.join(base_dir, 'ecdf', table_type, filename)

    if not os.path.exists(bins_path) or not os.path.exists(ecdf_path):
        return None

    bins_df = pd.read_parquet(bins_path)
    ecdf_df = pd.read_parquet(ecdf_path).sort_values('value')

    bins_list = []
    for _, row in bins_df.iterrows():
        bins_list.append({
            'segment': row.get('segment', 'flat'),
            'bin_min': row['bin_min'],
            'bin_max': row['bin_max'],
            'count': int(row['count']),
            'percentage': round(float(row['percentage']), 1),
            'interval': row.get('interval', ''),
        })

    return {
        'bins': bins_list,
        'ecdf_values': ecdf_df['value'].tolist(),
        'ecdf_probs': ecdf_df['probability'].tolist(),
    }


def _collect_all_metrics(
    base_dir: str,
    files: List[Tuple[str, str, Optional[str]]],
    sub_strata_suffixes: Optional[List[str]] = None,
) -> Dict:
    """Collect all metric data for a stratum.

    Returns dict keyed by metric_key with panel data for each sub-stratum.
    """
    all_data = {}

    for table_type, category, unit in files:
        if table_type == 'labs' and unit:
            unit_safe = unit.replace('/', '_')
            filename = f'{category}_{unit_safe}.parquet'
            display = f'{category.replace("_", " ").title()} ({unit})'
            metric_key = f'{table_type}/{category}_{unit_safe}'
        else:
            filename = f'{category}.parquet'
            display = category.replace('_', ' ').title()
            metric_key = f'{table_type}/{category}'

        panels = {}

        # Base panel (overall for this stratum)
        base_data = _read_metric_data(base_dir, table_type, filename)
        if base_data is None:
            continue
        panels['overall'] = base_data

        # Sub-strata panels
        if sub_strata_suffixes:
            for suffix in sub_strata_suffixes:
                suffixed_filename = filename.replace('.parquet', f'{suffix}.parquet')
                sub_data = _read_metric_data(base_dir, table_type, suffixed_filename)
                if sub_data is not None:
                    panels[suffix.lstrip('_')] = sub_data

        all_data[metric_key] = {
            'display': display,
            'table_type': table_type,
            'panels': panels,
        }

    return all_data


# ============================================================================
# HTML generation
# ============================================================================

_PLOTLY_CDN = 'https://cdn.plot.ly/plotly-2.35.2.min.js'


def _build_html(
    title: str,
    panel_labels: List[str],
    metrics_data: Dict,
    config: dict,
) -> str:
    """Build a self-contained HTML string with embedded data and plotly.js."""

    # Group metrics by table_type for the dropdown
    grouped: Dict[str, List[Tuple[str, str]]] = {}
    for key, info in metrics_data.items():
        tt = info['table_type']
        grouped.setdefault(tt, []).append((key, info['display']))

    # Build <optgroup> options
    optgroups_html = []
    first_key = None
    for tt in ['labs', 'vitals', 'respiratory_support']:
        if tt not in grouped:
            continue
        label = tt.replace('_', ' ').title()
        opts = []
        for key, display in grouped[tt]:
            if first_key is None:
                first_key = key
            opts.append(f'        <option value="{escape(key)}">{escape(display)}</option>')
        optgroups_html.append(f'      <optgroup label="{label}">\n' + '\n'.join(opts) + '\n      </optgroup>')

    select_html = '\n'.join(optgroups_html)
    n_panels = len(panel_labels)

    # Build normal range lookup from config
    normal_ranges = {}
    for tt in ['labs', 'vitals']:
        if tt in config:
            for cat, cat_cfg in config[tt].items():
                nr = cat_cfg.get('normal_range')
                if nr:
                    for key in metrics_data:
                        if key.startswith(f'{tt}/{cat}'):
                            normal_ranges[key] = nr

    # Panel divs
    panel_divs = []
    for i, label in enumerate(panel_labels):
        panel_divs.append(f'''
      <div class="panel">
        <h3>{escape(label)}</h3>
        <div id="plot-{i}" style="width:100%;height:500px;"></div>
      </div>''')

    # Segment colors JS object
    seg_colors_js = json.dumps(SEGMENT_COLORS)

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{escape(title)}</title>
  <script src="{_PLOTLY_CDN}"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           margin: 20px; background: #fafafa; }}
    h1 {{ color: #1F4E79; margin-bottom: 4px; }}
    .controls {{ margin: 16px 0; }}
    .controls select {{ font-size: 15px; padding: 6px 12px; min-width: 300px; }}
    .panels {{ display: flex; gap: 12px; flex-wrap: wrap; }}
    .panel {{ flex: 1; min-width: 350px; background: white; border: 1px solid #ddd;
              border-radius: 6px; padding: 8px; }}
    .panel h3 {{ text-align: center; margin: 4px 0 0 0; color: #333; font-size: 14px; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <div class="controls">
    <label for="metric-select"><b>Metric:</b></label>
    <select id="metric-select" onchange="updatePlots()">
{select_html}
    </select>
  </div>
  <div class="panels">
{"".join(panel_divs)}
  </div>

  <script>
    const DATA = {json.dumps(metrics_data, separators=(',', ':'))};
    const NORMAL_RANGES = {json.dumps(normal_ranges, separators=(',', ':'))};
    const PANEL_LABELS = {json.dumps(panel_labels)};
    const PANEL_KEYS = {json.dumps([l.lower().replace(' ', '_').replace('-', '_') for l in panel_labels])};
    const SEG_COLORS = {seg_colors_js};
    const FLAT_COLOR = "{FLAT_COLOR}";

    function buildTraces(binData, ecdfData) {{
      const traces = [];
      // Bins as bar chart
      for (const b of binData) {{
        const center = (b.bin_min + b.bin_max) / 2;
        const width = b.bin_max - b.bin_min;
        const color = b.segment === 'flat' ? FLAT_COLOR : (SEG_COLORS[b.segment] || '#999');
        traces.push({{
          type: 'bar',
          x: [center], y: [b.count], width: [width],
          marker: {{ color: color, line: {{ color: 'rgba(0,0,0,0.3)', width: 1 }} }},
          hovertemplate: '<b>' + b.interval + '</b><br>Count: ' + b.count.toLocaleString()
            + '<br>Pct: ' + b.percentage + '%<extra></extra>',
          showlegend: false,
        }});
      }}
      // ECDF overlay
      if (ecdfData.ecdf_values.length > 0) {{
        const maxCount = Math.max(...binData.map(b => b.count));
        const scaledProbs = ecdfData.ecdf_probs.map(p => p * maxCount);
        traces.push({{
          type: 'scatter', mode: 'lines',
          x: ecdfData.ecdf_values, y: scaledProbs,
          line: {{ color: '{ECDF_COLOR}', width: 2.5 }},
          name: 'ECDF',
          hovertemplate: '<b>ECDF</b><br>Value: %{{x:.2f}}<br>CDF: %{{customdata:.3f}}<extra></extra>',
          customdata: ecdfData.ecdf_probs,
          showlegend: false,
        }});
      }}
      return traces;
    }}

    function updatePlots() {{
      const key = document.getElementById('metric-select').value;
      const metric = DATA[key];
      if (!metric) return;
      const nr = NORMAL_RANGES[key] || null;

      for (let i = 0; i < PANEL_KEYS.length; i++) {{
        const panelKey = PANEL_KEYS[i];
        const panelData = metric.panels[panelKey];
        const divId = 'plot-' + i;

        if (!panelData) {{
          Plotly.purge(divId);
          document.getElementById(divId).innerHTML =
            '<p style="text-align:center;color:#999;margin-top:40px;">No data</p>';
          continue;
        }}

        const traces = buildTraces(panelData.bins, panelData);
        const shapes = [];
        if (nr) {{
          shapes.push(
            {{ type:'line', x0:nr.lower, x1:nr.lower, y0:0, y1:1, yref:'paper',
               line:{{ color:'green', width:2, dash:'dash' }} }},
            {{ type:'line', x0:nr.upper, x1:nr.upper, y0:0, y1:1, yref:'paper',
               line:{{ color:'green', width:2, dash:'dash' }} }}
          );
        }}
        const layout = {{
          xaxis: {{ title: metric.display, gridcolor: 'rgba(200,200,200,0.3)' }},
          yaxis: {{ title: 'Count', gridcolor: 'rgba(200,200,200,0.3)' }},
          plot_bgcolor: 'white', paper_bgcolor: 'white',
          margin: {{ t: 10, b: 50, l: 60, r: 20 }},
          shapes: shapes,
          barmode: 'stack',
        }};
        Plotly.react(divId, traces, layout, {{ responsive: true }});
      }}
    }}

    // Initial render
    updatePlots();
  </script>
</body>
</html>'''


def generate_interactive_html(
    base_dir: str,
    output_path: str,
    title: str,
    sub_strata_suffixes: Optional[List[str]] = None,
    panel_labels: Optional[List[str]] = None,
) -> bool:
    """Generate an interactive HTML distribution viewer for one stratum.

    Parameters
    ----------
    base_dir : str
        Cohort root directory containing bins/ and ecdf/ subdirs.
    output_path : str
        Path to write the HTML file.
    title : str
        Page title.
    sub_strata_suffixes : list[str], optional
        E.g. ['_icu', '_no_icu'] for complex strata.
    panel_labels : list[str], optional
        Display labels for panels. First is always 'Overall'.
        E.g. ['Overall', 'ICU', 'No ICU'].

    Returns
    -------
    bool
        True if successful.
    """
    files = discover_files(base_dir)
    if not files:
        return False

    if panel_labels is None:
        panel_labels = ['Overall']

    # Load normal-range config
    from modules.utils.output_paths import configs_dir
    config_path = str(configs_dir() / 'lab_vital_config.yaml')
    config = load_lab_vital_config(config_path)

    metrics_data = _collect_all_metrics(base_dir, files, sub_strata_suffixes)
    if not metrics_data:
        return False

    html = _build_html(title, panel_labels, metrics_data, config)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return True
