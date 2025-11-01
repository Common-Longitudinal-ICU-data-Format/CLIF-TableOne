#!/usr/bin/env python3
"""
Generate a formatted table of ventilator settings by device mode.
Combines median/IQR values with observation percentages.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import os
from pathlib import Path

def get_output_path(*parts):
    """Helper to get output paths relative to project root"""
    project_root = Path(__file__).parent.parent.parent
    path = project_root / 'output'
    for part in parts:
        path = path / part
    return str(path)

def load_ventilator_data():
    """Load the ventilator settings and counts data."""
    # Use the get_output_path helper for consistent path handling
    settings_df = pd.read_csv(get_output_path('final', 'tableone', 'ventilator_settings_by_device_mode.csv'))
    counts_df = pd.read_csv(get_output_path('final', 'tableone', 'ventilator_settings_counts_by_device_mode.csv'))
    # Try to load total observations if available
    total_obs_file = get_output_path('final', 'tableone', 'ventilator_settings_total_observations.csv')

    # Try to load saved total observations
    total_obs = None
    if os.path.exists(total_obs_file):
        try:
            total_obs_df = pd.read_csv(total_obs_file)
            total_obs = int(total_obs_df['value'].iloc[0])
            print(f"Loaded total observations from file: {total_obs:,}")
        except:
            pass

    return settings_df, counts_df, total_obs

def calculate_percentages(counts_df, total_observations=None):
    """Calculate percentages for each setting.

    Parameters:
    -----------
    counts_df : DataFrame
        DataFrame with observation counts
    total_observations : int, optional
        Total number of respiratory support observations. If not provided, will sum FiO2 column.
    """
    # Calculate total observations across all device modes for each setting
    settings_cols = [col for col in counts_df.columns if col not in ['device_category', 'ventilator_setting']]

    # Get the total respiratory support observations
    if total_observations is not None:
        total_obs = total_observations
    else:
        # Use the sum of all FiO2 observations as a proxy for total observations
        # since FiO2 is recorded for most ventilator modes
        total_obs = counts_df['FiO2 Set (N)'].sum()

    # Create percentage dataframe with percentages of total observations
    pct_df = counts_df.copy()
    for col in settings_cols:
        pct_col = col.replace(' (N)', ' (%)')
        # Calculate percentage of total observations, not column total
        pct_df[pct_col] = (counts_df[col] / total_obs * 100).round(2)

    return pct_df, total_obs

def format_cell_content(median_iqr, percentage, count):
    """Format cell content with median (IQR) and percentage."""
    if pd.isna(count) or count == 0 or 'nan' in str(median_iqr).lower():
        return "-"

    # Parse median and IQR from string format "median (q1-q3)"
    try:
        parts = median_iqr.strip().split('(')
        if len(parts) == 2:
            median = parts[0].strip()
            iqr = '(' + parts[1]
            # Format percentage with proper display
            if percentage >= 10:
                pct_str = f"{percentage:.1f}%"
            elif percentage >= 1:
                pct_str = f"{percentage:.1f}%"
            else:
                pct_str = f"{percentage:.2f}%"
            return f"{median}\n{iqr}\n{pct_str}"
        else:
            return f"{median_iqr}\n{percentage:.1f}%"
    except:
        return f"{median_iqr}\n{percentage:.1f}%"

def create_ventilator_table(total_observations=None):
    """Create the formatted ventilator settings table.

    Parameters:
    -----------
    total_observations : int, optional
        Total number of respiratory support observations. If not provided, will be calculated from data.
    """
    # Load data
    settings_df, counts_df, saved_total_obs = load_ventilator_data()

    # Use saved total if not provided
    if total_observations is None:
        total_observations = saved_total_obs

    # Calculate percentages
    pct_df, total_obs = calculate_percentages(counts_df, total_observations)

    # Determine the mode column name (could be 'mode_category' or 'ventilator_setting')
    mode_col = 'mode_category' if 'mode_category' in settings_df.columns else 'ventilator_setting'

    # All possible settings to display (will filter to available columns)
    all_settings_display = {
        'FiO2 Set': 'FiO₂',
        'LPM Set': 'LPM\n(L/min)',
        'Tidal Volume Set': 'Tidal Volume\n(mL)',
        'Resp Rate Set': 'Resp Rate\n(breaths/min)',
        'Pressure Control Set': 'Pressure Control\n(cmH₂O)',
        'PEEP Set': 'PEEP\n(cmH₂O)',
        'Pressure Support Set': 'Pressure Support\n(cmH₂O)',
        'Flow Rate Set': 'Flow Rate\n(L/min)',
        'Inspiratory Time Set': 'Insp. Time\n(seconds)'
    }

    # Filter to only columns that exist in the data (exclude metadata columns)
    available_cols = [col for col in settings_df.columns if col not in ['device_category', 'ventilator_setting', 'mode_category']]
    settings_display = {col: all_settings_display[col] for col in all_settings_display if col in available_cols}

    if len(settings_display) == 0:
        print("Warning: No ventilator settings columns found in the data!")
        return [], [], [], total_obs

    print(f"Found {len(settings_display)} ventilator settings columns in the data")

    # Prepare data for table
    table_data = []
    row_labels = []
    percentages_for_heatmap = []  # Store percentages for heatmap coloring

    # Get all unique device-mode combinations from the actual data
    # Sort by device category first, then by mode
    unique_combinations = settings_df[['device_category', mode_col]].drop_duplicates()
    unique_combinations = unique_combinations.sort_values(['device_category', mode_col])

    # Process all device-mode combinations from the data
    for _, combo in unique_combinations.iterrows():
        device = combo['device_category']
        mode = combo[mode_col]

        # Get data for this device-mode combination
        mask = (settings_df['device_category'] == device) & (settings_df[mode_col] == mode)
        if mask.any():
            settings_row = settings_df[mask].iloc[0]
            counts_row = counts_df[mask].iloc[0]

            row_data = []
            row_percentages = []  # Store percentages for this row

            for orig_col, _ in settings_display.items():
                median_iqr = settings_row[orig_col]
                count_col = orig_col.replace('Set', 'Set (N)')
                count = counts_row[count_col] if count_col in counts_row else 0
                pct_col = count_col.replace('(N)', '(%)')
                percentage = pct_df[mask][pct_col].iloc[0] if pct_col in pct_df.columns else 0

                cell_content = format_cell_content(median_iqr, percentage, count)
                row_data.append(cell_content)
                row_percentages.append(percentage if count > 0 else 0)

            table_data.append(row_data)
            percentages_for_heatmap.append(row_percentages)

            # Format row label
            device_label = device.upper() if device.lower() not in ['cpap', 'nippv'] else device.upper()
            mode_label = mode.replace('-', '-\n').title() if mode else 'Unknown'
            row_labels.append(f"{device_label}\n{mode_label}")

    return table_data, row_labels, list(settings_display.values()), total_obs, percentages_for_heatmap

def plot_ventilator_table(save_path=None, total_observations=None):
    """Create and save the ventilator settings table as an image.

    Parameters:
    -----------
    save_path : str
        Path to save the output image
    total_observations : int, optional
        Total number of respiratory support observations. If not provided, will be calculated from data.
    """

    table_data, row_labels, col_labels, total_obs, percentages = create_ventilator_table(total_observations)

    # Check if we have data to display
    if not table_data or not col_labels:
        print("⚠️ No data available to create ventilator settings table")
        return None

    # Create figure with dynamic sizing based on data
    num_cols = len(col_labels)
    num_rows = len(table_data)

    # Simple figure sizing
    fig_width = 20  # Fixed width that works
    fig_height = 14  # Fixed height that works

    fig = plt.figure(figsize=(fig_width, fig_height))

    # Simple subplot that just works
    ax = fig.add_subplot(111)
    ax.axis('tight')
    ax.axis('off')

    # Create the table with appropriate column widths
    col_width = 0.1  # Simple fixed width
    table = ax.table(cellText=table_data,
                     rowLabels=row_labels,
                     colLabels=col_labels,
                     cellLoc='center',
                     loc='center',
                     colWidths=[col_width] * len(col_labels))

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)  # Make rows taller to accommodate headers and 3 lines of text

    # Color coding for device categories
    device_colors = {
        'IMV': '#e8f4fd',      # Light blue
        'NIPPV': '#fff9e6',    # Light yellow
        'CPAP': '#f0f9ff'      # Very light blue
    }

    # Create color map function for heatmap (green gradient based on percentage)
    def get_heatmap_color(percentage):
        """Get color based on percentage - darker green for higher percentages."""
        if percentage == 0 or pd.isna(percentage):
            return 'white'
        # Map percentage (0-100) to color intensity
        # Use green color with varying intensity
        intensity = min(percentage / 50, 1.0)  # Cap at 50% for full color
        # RGB for green: darker green for higher values
        r = 1 - (intensity * 0.7)  # Reduce red
        g = 1 - (intensity * 0.3)  # Keep green high
        b = 1 - (intensity * 0.7)  # Reduce blue
        return (r, g, b)

    # Style cells - iterate through all cells in the table
    for key, cell in table.get_celld().items():
        row, col = key

        # Style header row
        if row == 0:
            if col >= 0:  # Column headers
                cell.set_facecolor('#2c3e50')
                cell.set_text_props(weight='bold', color='white')
                cell.set_height(0.05)
            else:  # Top-left corner cell (row label header)
                cell.set_facecolor('white')
                cell.get_text().set_text('')

        # Style data rows
        elif row > 0:
            if col == -1:  # Row label cells
                row_label = row_labels[row-1]
                device = row_label.split('\n')[0]
                if device in device_colors:
                    cell.set_facecolor(device_colors[device])
                cell.set_text_props(weight='bold', ha='left')
                cell.set_height(0.08)

            else:  # Data cells
                cell.set_height(0.08)

                # Apply heatmap coloring based on percentage
                if row-1 < len(percentages) and col < len(percentages[row-1]):
                    percentage = percentages[row-1][col]
                    cell.set_facecolor(get_heatmap_color(percentage))

                # Parse and style text
                text = cell.get_text().get_text()
                if text and text != '-':
                    lines = text.split('\n')
                    if len(lines) >= 3:
                        # Keep all three lines but make percentage smaller
                        cell.get_text().set_fontsize(8)

    # Add note about percentages and total observations at the bottom
    note_text = (f'Total respiratory support observations: {total_obs:,}\n'
                 f'Note: Percentages represent proportion of total respiratory support observations with each setting recorded.\n'
                 f'Not all settings are applicable to all ventilator modes. Dash (-) indicates no data available.')

    # Position note text at bottom
    # plt.figtext(0.5, 0.01, note_text,
    #             ha='center', va='bottom', fontsize=9, style='italic', wrap=True)

    # Determine save path using the helper
    if save_path is None:
        save_path = get_output_path('final', 'tableone', 'ventilator_settings_table.png')

    # Save the figure with tight bbox to remove extra whitespace
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✅ Ventilator settings table saved to: {save_path}")

    # Also save as PDF for better quality
    pdf_path = save_path.replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✅ PDF version saved to: {pdf_path}")

    plt.close(fig)  # Close figure to free memory

    return fig

if __name__ == "__main__":
    # Generate and display the table
    fig = plot_ventilator_table()