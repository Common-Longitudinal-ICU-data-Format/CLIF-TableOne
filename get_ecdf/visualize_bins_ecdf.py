#!/usr/bin/env python3
"""
Visualize ECDF Bins and ECDF

This script creates combined visualizations of bins and ECDF from ECDF data.

Usage:
    # Generate all categories (labs, vitals, respiratory_support)
    python get_ecdf/visualize_bins_ecdf.py

    # Generate single category
    python get_ecdf/visualize_bins_ecdf.py --category sodium --unit "mmol/L" --table labs

    # Generate all labs
    python get_ecdf/visualize_bins_ecdf.py --table labs

    # Generate all vitals
    python get_ecdf/visualize_bins_ecdf.py --table vitals

    # Generate all respiratory support
    python get_ecdf/visualize_bins_ecdf.py --table respiratory_support

Output:
    output/final/plots/{table_type}/{category}_{unit}.html
"""

import os
import argparse
from pathlib import Path
from typing import Optional, List, Tuple
import polars as pl
import plotly.graph_objects as go
import yaml


# ============================================================================
# Configuration
# ============================================================================

# Medical color palette
COLORS = {
    'below': ['#08519c', '#3182bd', '#6baed6', '#9ecae1', '#c6dbef'],  # Blue (low/cold)
    'normal': ['#006d2c', '#31a354', '#74c476', '#a1d99b', '#c7e9c0'],  # Green (healthy)
    'above': ['#a50f15', '#de2d26', '#fb6a4a', '#fc9272', '#fcbba1']   # Red (warning/high)
}


# ============================================================================
# Helper Functions
# ============================================================================

def get_bin_color(bin_info: dict, all_bins: List[dict]) -> str:
    """
    Get color for a bin based on its segment and position.

    For flat bins (respiratory_support), uses a single color gradient.

    Args:
        bin_info: Dictionary with bin information
        all_bins: List of all bins

    Returns:
        Hex color string
    """
    # Check if this is a flat bin (no segment)
    if 'segment' not in bin_info:
        # Use a single gradient for flat bins (blue gradient)
        flat_colors = ['#08519c', '#2171b5', '#4292c6', '#6baed6', '#9ecae1', '#c6dbef', '#deebf7', '#f7fbff']
        bin_idx = all_bins.index(bin_info)
        color_idx = int(bin_idx / len(all_bins) * len(flat_colors))
        color_idx = min(color_idx, len(flat_colors) - 1)
        return flat_colors[color_idx]

    # Segmented bins (labs/vitals)
    segment = bin_info['segment']
    segment_bins = [b for b in all_bins if b['segment'] == segment]
    bin_idx = segment_bins.index(bin_info)

    # Map bin index to color index
    color_idx = int(bin_idx / len(segment_bins) * len(COLORS[segment]))
    color_idx = min(color_idx, len(COLORS[segment]) - 1)

    return COLORS[segment][color_idx]


def load_lab_vital_config(config_path: str = 'configs/lab_vital_config.yaml') -> dict:
    """Load lab/vital configuration."""
    if not os.path.exists(config_path):
        return {}

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ============================================================================
# Visualization Functions
# ============================================================================

def create_combined_plot(
    bins_df: pl.DataFrame,
    ecdf_df: pl.DataFrame,
    category: str,
    unit: Optional[str] = None,
    normal_range: Optional[dict] = None
) -> go.Figure:
    """
    Create combined plot with bins (histogram) and ECDF.

    Args:
        bins_df: Polars DataFrame with bin data
        ecdf_df: Polars DataFrame with ECDF data
        category: Category name
        unit: Unit string (for labs)
        normal_range: Dict with 'lower' and 'upper' keys

    Returns:
        Plotly Figure object
    """
    fig = go.Figure()

    # Convert to list of dicts for easier processing
    bins = bins_df.to_dicts()

    # ========================================================================
    # Plot Histogram Bars
    # ========================================================================

    for bin_info in bins:
        center = (bin_info['bin_min'] + bin_info['bin_max']) / 2
        width = bin_info['bin_max'] - bin_info['bin_min']
        color = get_bin_color(bin_info, bins)

        # Get or compute interval notation
        if 'interval' in bin_info:
            interval = bin_info['interval']
        else:
            # Compute on-the-fly for backward compatibility
            if bin_info['bin_num'] == 1:
                interval = f"[{bin_info['bin_min']:.2f}, {bin_info['bin_max']:.2f}]"
            else:
                interval = f"({bin_info['bin_min']:.2f}, {bin_info['bin_max']:.2f}]"

        # Create hover text
        if 'segment' in bin_info:
            hover_text = (
                f"<b>{interval}</b><br>"
                f"Count: {bin_info['count']:,}<br>"
                f"Percentage: {bin_info['percentage']:.1f}%<br>"
                f"Segment: {bin_info['segment']}"
            )
        else:
            # Flat bins (respiratory_support)
            hover_text = (
                f"<b>{interval}</b><br>"
                f"Count: {bin_info['count']:,}<br>"
                f"Percentage: {bin_info['percentage']:.1f}%"
            )

        fig.add_trace(go.Bar(
            x=[center],
            y=[bin_info['count']],
            width=[width],
            marker=dict(
                color=color,
                line=dict(color='rgba(0,0,0,0.3)', width=1)
            ),
            name=interval,
            hovertemplate=hover_text + '<extra></extra>',
            showlegend=True
        ))

    # ========================================================================
    # Plot ECDF
    # ========================================================================

    if len(ecdf_df) > 0:
        # Sort by value to ensure proper line rendering (prevents filled polygon effect)
        ecdf_sorted = ecdf_df.sort('value')
        ecdf_values = ecdf_sorted['value'].to_numpy()
        ecdf_probs = ecdf_sorted['probability'].to_numpy()

        # Scale ECDF to match histogram height
        max_count = max(b['count'] for b in bins)
        ecdf_scaled = ecdf_probs * max_count

        fig.add_trace(go.Scatter(
            x=ecdf_values,
            y=ecdf_scaled,
            mode='lines',
            line=dict(color='black', width=3),
            name='ECDF',
            hovertemplate='<b>ECDF</b><br>Value: %{x:.2f}<br>CDF: %{customdata:.3f}<extra></extra>',
            customdata=ecdf_probs,
            showlegend=True
        ))

    # ========================================================================
    # Add Normal Range Lines
    # ========================================================================

    if normal_range:
        fig.add_vline(
            x=normal_range['lower'],
            line=dict(color='green', width=2, dash='dash'),
            annotation_text="Normal Lower",
            annotation_position="top"
        )

        fig.add_vline(
            x=normal_range['upper'],
            line=dict(color='green', width=2, dash='dash'),
            annotation_text="Normal Upper",
            annotation_position="top"
        )

    # ========================================================================
    # Update Layout
    # ========================================================================

    # Create title
    title_parts = [category.replace('_', ' ').title()]
    if unit:
        title_parts.append(f"({unit})")

    # Add bin configuration info
    # Check if bins have segments (labs/vitals) or are flat (respiratory_support)
    has_segments = any('segment' in b for b in bins)

    if has_segments:
        segments = {}
        for b in bins:
            seg = b['segment']
            if seg not in segments:
                segments[seg] = 0
            segments[seg] += 1

        bin_config = f"Bins: {segments.get('below', 0)}-{segments.get('normal', 0)}-{segments.get('above', 0)}"

        if normal_range:
            subtitle = f"<sub>{bin_config} | Normal Range: {normal_range['lower']}-{normal_range['upper']}</sub>"
        else:
            subtitle = f"<sub>{bin_config}</sub>"
    else:
        # Flat bins (respiratory_support)
        bin_config = f"Bins: {len(bins)} (flat quantiles)"
        subtitle = f"<sub>{bin_config}</sub>"

    title = f"{' '.join(title_parts)}<br>{subtitle}"

    fig.update_layout(
        title=title,
        xaxis_title=f"{category.replace('_', ' ').title()} {f'({unit})' if unit else ''}".strip(),
        yaxis_title="Count",
        plot_bgcolor='white',
        height=600,
        showlegend=True,
        legend=dict(
            title="Bins & ECDF",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=1.15
        )
    )

    fig.update_xaxes(gridcolor='rgba(200,200,200,0.3)')
    fig.update_yaxes(gridcolor='rgba(200,200,200,0.3)')

    return fig


# ============================================================================
# File Discovery and Processing
# ============================================================================

def discover_files(
    base_dir: str = 'output/final',
    table_type: Optional[str] = None
) -> List[Tuple[str, str, Optional[str]]]:
    """
    Discover all available bin/ECDF file pairs.

    Args:
        base_dir: Base directory with ECDF data
        table_type: Optional filter for 'labs', 'vitals', or 'respiratory_support'

    Returns:
        List of tuples: (table_type, category, unit)
    """
    files = []

    # Determine which tables to process
    if table_type:
        tables = [table_type]
    else:
        tables = ['labs', 'vitals', 'respiratory_support']

    for table in tables:
        bins_dir = os.path.join(base_dir, 'bins', table)

        if not os.path.exists(bins_dir):
            continue

        # List all parquet files in bins directory
        for filename in os.listdir(bins_dir):
            if not filename.endswith('.parquet'):
                continue

            # Parse filename
            basename = filename.replace('.parquet', '')

            if table == 'labs':
                # Labs: category_unit format
                parts = basename.rsplit('_', 1)
                if len(parts) == 2:
                    category, unit_safe = parts
                    # Reverse sanitization (approximate)
                    unit = unit_safe.replace('_', '/')
                    files.append((table, category, unit))
            else:
                # Vitals and respiratory_support: just category/column name
                category = basename
                files.append((table, category, None))

    return files


def process_category(
    table_type: str,
    category: str,
    unit: Optional[str],
    base_dir: str = 'output/final',
    output_dir: str = 'output/final/plots'
) -> bool:
    """
    Process a single category: load data and create visualization.

    Args:
        table_type: 'labs', 'vitals', or 'respiratory_support'
        category: Category/column name
        unit: Unit string (for labs only)
        base_dir: Base directory with ECDF data
        output_dir: Output directory for plots

    Returns:
        True if successful, False otherwise
    """
    # Construct filenames
    if table_type == 'labs' and unit:
        # Sanitize unit for filename
        unit_safe = unit.replace('/', '_')
        filename = f'{category}_{unit_safe}.parquet'
    else:
        filename = f'{category}.parquet'

    bins_path = os.path.join(base_dir, 'bins', table_type, filename)
    ecdf_path = os.path.join(base_dir, 'ecdf', table_type, filename)

    # Check if files exist
    if not os.path.exists(bins_path):
        print(f"  ⚠️  Bins file not found: {bins_path}")
        return False

    if not os.path.exists(ecdf_path):
        print(f"  ⚠️  ECDF file not found: {ecdf_path}")
        return False

    # Load data
    bins_df = pl.read_parquet(bins_path)
    ecdf_df = pl.read_parquet(ecdf_path)

    # Load config for normal range
    config = load_lab_vital_config(os.path.join(base_dir, 'configs/lab_vital_config.yaml'))

    normal_range = None
    if table_type in config and category in config[table_type]:
        normal_range = config[table_type][category].get('normal_range')

    # Create plot
    fig = create_combined_plot(
        bins_df=bins_df,
        ecdf_df=ecdf_df,
        category=category,
        unit=unit,
        normal_range=normal_range
    )

    # Save plot
    output_subdir = os.path.join(output_dir, table_type)
    os.makedirs(output_subdir, exist_ok=True)

    output_path = os.path.join(output_subdir, filename.replace('.parquet', '.html'))
    fig.write_html(output_path)

    print(f"  ✓ {category} {f'({unit})' if unit else ''}: {output_path}")

    return True


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Visualize ECDF bins and ECDF data'
    )
    parser.add_argument(
        '--category',
        type=str,
        help='Specific category to process (e.g., sodium, heart_rate)'
    )
    parser.add_argument(
        '--unit',
        type=str,
        help='Unit for lab category (e.g., "mmol/L")'
    )
    parser.add_argument(
        '--table',
        type=str,
        choices=['labs', 'vitals', 'respiratory_support'],
        help='Table type to process (labs, vitals, or respiratory_support)'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='output/final',
        help='Base directory with ECDF data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/final/plots',
        help='Output directory for plots'
    )

    args = parser.parse_args()

    print("\n" + "="*80)
    print("Visualize ECDF Bins and ECDF")
    print("="*80 + "\n")

    # ========================================================================
    # Single Category Mode
    # ========================================================================

    if args.category:
        if not args.table:
            print("❌ Error: --table required when --category is specified")
            return

        print(f"Processing single category: {args.category}")
        if args.unit:
            print(f"Unit: {args.unit}")
        print()

        success = process_category(
            table_type=args.table,
            category=args.category,
            unit=args.unit,
            base_dir=args.base_dir,
            output_dir=args.output_dir
        )

        if success:
            print("\n✅ Done!")
        else:
            print("\n❌ Failed!")

        return

    # ========================================================================
    # Batch Mode
    # ========================================================================

    print("Discovering files...")
    files = discover_files(base_dir=args.base_dir, table_type=args.table)

    if not files:
        print("❌ No files found!")
        return

    print(f"✓ Found {len(files)} categories\n")

    # Process labs
    if args.table is None or args.table == 'labs':
        labs_files = [(t, c, u) for t, c, u in files if t == 'labs']

        if labs_files:
            print("="*80)
            print(f"Processing Labs ({len(labs_files)} categories)")
            print("="*80)

            for table_type, category, unit in labs_files:
                process_category(
                    table_type=table_type,
                    category=category,
                    unit=unit,
                    base_dir=args.base_dir,
                    output_dir=args.output_dir
                )

            print()

    # Process vitals
    if args.table is None or args.table == 'vitals':
        vitals_files = [(t, c, u) for t, c, u in files if t == 'vitals']

        if vitals_files:
            print("="*80)
            print(f"Processing Vitals ({len(vitals_files)} categories)")
            print("="*80)

            for table_type, category, unit in vitals_files:
                process_category(
                    table_type=table_type,
                    category=category,
                    unit=unit,
                    base_dir=args.base_dir,
                    output_dir=args.output_dir
                )

            print()

    # Process respiratory support
    if args.table is None or args.table == 'respiratory_support':
        resp_files = [(t, c, u) for t, c, u in files if t == 'respiratory_support']

        if resp_files:
            print("="*80)
            print(f"Processing Respiratory Support ({len(resp_files)} columns)")
            print("="*80)

            for table_type, category, unit in resp_files:
                process_category(
                    table_type=table_type,
                    category=category,
                    unit=unit,
                    base_dir=args.base_dir,
                    output_dir=args.output_dir
                )

            print()

    print("="*80)
    print(f"✅ Done! Plots saved to: {args.output_dir}/")
    print("="*80)


if __name__ == '__main__':
    main()
