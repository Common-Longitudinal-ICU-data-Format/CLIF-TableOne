"""
MCIDE (Minimum Common Data Elements) Viewer Module
Displays MCIDE statistics for CLIF tables
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Optional
import glob


def check_mcide_results_available(output_dir: str = 'output') -> bool:
    """
    Check if MCIDE results are available.

    Parameters:
    -----------
    output_dir : str
        Base output directory

    Returns:
    --------
    bool
        True if MCIDE CSV files exist
    """
    mcide_dir = os.path.join(output_dir, 'final', 'tableone', 'mcide')

    if not os.path.exists(mcide_dir):
        return False

    # Check if there are any CSV files in the directory
    csv_files = glob.glob(os.path.join(mcide_dir, '*.csv'))
    return len(csv_files) > 0


def get_table_mcide_files(mcide_dir: str, table_name: str) -> List[str]:
    """
    Get all MCIDE CSV files for a specific table.

    Parameters:
    -----------
    mcide_dir : str
        Path to MCIDE directory
    table_name : str
        Name of the table

    Returns:
    --------
    List[str]
        List of CSV file paths for the table
    """
    # Try both with and without clif_ prefix
    patterns = [
        f"{table_name}_*_mcide.csv",
        f"clif_{table_name}_*_mcide.csv"
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(mcide_dir, pattern)))

    return sorted(list(set(files)))  # Remove duplicates and sort


def format_column_names(filename: str) -> str:
    """
    Extract and format column names from filename.

    Parameters:
    -----------
    filename : str
        CSV filename

    Returns:
    --------
    str
        Formatted column names for display
    """
    # Remove path and extension
    base_name = os.path.basename(filename).replace('_mcide.csv', '')

    # Remove clif_ prefix if present
    if base_name.startswith('clif_'):
        base_name = base_name[5:]

    # Remove table prefix
    parts = base_name.split('_')

    # Identify where table name ends and columns begin
    # Tables can have underscores in their names (e.g., medication_admin_continuous)
    table_prefixes = [
        'medication_admin_continuous',
        'medication_admin_intermittent',
        'microbiology_culture',
        'microbiology_nonculture',
        'microbiology_susceptibility',
        'patient_assessments',
        'patient_procedures',
        'respiratory_support',
        'code_status',
        'crrt_therapy',
        'ecmo_mcs',
        'hospital_diagnosis'
    ]

    for prefix in table_prefixes:
        if base_name.startswith(prefix + '_'):
            column_part = base_name[len(prefix) + 1:]
            break
    else:
        # Simple table name (single word)
        if '_' in base_name:
            column_part = '_'.join(parts[1:])
        else:
            column_part = base_name

    # Format column names
    columns = column_part.replace('_', ' ').title()
    return columns


def display_mcide_file(filepath: str, table_name: str):
    """
    Display a single MCIDE CSV file with formatting.

    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    table_name : str
        Name of the table for context
    """
    try:
        df = pd.read_csv(filepath)

        # Get column names for display
        column_desc = format_column_names(filepath)

        # Display header
        st.markdown(f"#### 📊 {column_desc}")

        # Add summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Unique Combinations", f"{len(df):,}")
        with col2:
            st.metric("Total Observations", f"{df['N'].sum():,}" if 'N' in df.columns else "N/A")
        with col3:
            # Calculate percentage of most common value
            if 'N' in df.columns and len(df) > 0:
                max_pct = (df['N'].max() / df['N'].sum() * 100)
                st.metric("Most Common %", f"{max_pct:.1f}%")
            else:
                st.metric("Most Common %", "N/A")

        # Add search/filter
        search_term = st.text_input(
            "🔍 Search/Filter",
            key=f"search_{filepath}",
            placeholder="Type to filter results..."
        )

        # Apply filter if search term exists
        if search_term:
            # Create a mask for all string columns
            mask = pd.Series([False] * len(df))
            for col in df.select_dtypes(include=['object']).columns:
                mask |= df[col].astype(str).str.contains(search_term, case=False, na=False)
            filtered_df = df[mask]
        else:
            filtered_df = df

        # Sort by count (N) descending if N column exists
        if 'N' in filtered_df.columns:
            filtered_df = filtered_df.sort_values('N', ascending=False)

        # Display options
        show_all = st.checkbox(
            "Show all rows",
            value=False,
            key=f"show_all_{filepath}"
        )

        if show_all:
            st.dataframe(
                filtered_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            # Show top 20 rows by default
            display_df = filtered_df.head(20)
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )
            if len(filtered_df) > 20:
                st.caption(f"Showing top 20 of {len(filtered_df)} rows. Check 'Show all rows' to see complete data.")

        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name=f"{table_name}_{column_desc.replace(' ', '_').lower()}_filtered.csv",
            mime="text/csv",
            key=f"download_{filepath}"
        )

    except Exception as e:
        st.error(f"Error loading file: {str(e)}")


def display_mcide_tab(output_dir: str = 'output'):
    """
    Display the MCIDE tab with all tables and their statistics.

    Parameters:
    -----------
    output_dir : str
        Base output directory
    """
    st.header("📊 MCIDE (Minimum Common Data Elements) Statistics")

    mcide_dir = os.path.join(output_dir, 'final', 'tableone', 'mcide')

    if not os.path.exists(mcide_dir):
        st.warning("⚠️ No MCIDE data found. Please run the table one generation first.")
        return

    # Get all available tables based on CSV files
    all_files = glob.glob(os.path.join(mcide_dir, '*_mcide.csv'))

    if not all_files:
        st.warning("⚠️ No MCIDE CSV files found in the directory.")
        st.info(f"Looking in: {mcide_dir}")
        return

    # Extract unique table names
    tables_with_data = set()
    table_mapping = {
        'patient': 'Patient',
        'hospitalization': 'Hospitalization',
        'adt': 'ADT',
        'code_status': 'Code Status',
        'crrt_therapy': 'CRRT Therapy',
        'ecmo_mcs': 'ECMO/MCS',
        'labs': 'Labs',
        'medication_admin_continuous': 'Medication Admin (Continuous)',
        'medication_admin_intermittent': 'Medication Admin (Intermittent)',
        'microbiology_culture': 'Microbiology Culture',
        'microbiology_nonculture': 'Microbiology Non-Culture',
        'microbiology_susceptibility': 'Microbiology Susceptibility',
        'patient_assessments': 'Patient Assessments',
        'patient_procedures': 'Patient Procedures',
        'position': 'Position',
        'respiratory_support': 'Respiratory Support',
        'vitals': 'Vitals',
        'hospital_diagnosis': 'Hospital Diagnosis'
    }

    # Identify which tables have MCIDE files
    for filepath in all_files:
        filename = os.path.basename(filepath)
        # Remove clif_ prefix if present
        if filename.startswith('clif_'):
            filename = filename[5:]

        # Try exact matches for compound table names first
        compound_tables = [
            'medication_admin_continuous',
            'medication_admin_intermittent',
            'microbiology_culture',
            'microbiology_nonculture',
            'microbiology_susceptibility',
            'patient_assessments',
            'patient_procedures',
            'respiratory_support',
            'code_status',
            'crrt_therapy',
            'ecmo_mcs',
            'hospital_diagnosis'
        ]

        matched = False
        for table_key in compound_tables:
            if filename.startswith(table_key + '_'):
                tables_with_data.add(table_key)
                matched = True
                break

        # If not matched yet, try simple table names
        if not matched:
            simple_tables = ['patient', 'hospitalization', 'adt', 'labs', 'vitals', 'position']
            for table_key in simple_tables:
                if filename.startswith(table_key + '_'):
                    tables_with_data.add(table_key)
                    break

    if not tables_with_data:
        st.warning("⚠️ Could not identify tables from MCIDE files.")
        return

    # Sort tables for consistent display
    sorted_tables = sorted(tables_with_data)

    # Create tabs for each table
    tab_names = [table_mapping.get(t, t.replace('_', ' ').title()) for t in sorted_tables]
    tabs = st.tabs(tab_names)

    # Display content for each table
    for idx, (tab, table_name) in enumerate(zip(tabs, sorted_tables)):
        with tab:
            # Get all MCIDE files for this table
            table_files = get_table_mcide_files(mcide_dir, table_name)

            if not table_files:
                st.info(f"No MCIDE data available for {table_mapping.get(table_name, table_name)}.")
                continue

            st.subheader(f"📋 {table_mapping.get(table_name, table_name)} MCIDE Statistics")

            # Show count of different MCIDE collections
            st.info(f"Found {len(table_files)} MCIDE collection(s) for this table")

            # Display each MCIDE file
            for filepath in table_files:
                display_mcide_file(filepath, table_name)
                st.divider()


def display_table_mcide(table_name: str, output_dir: str = 'output'):
    """
    Display MCIDE statistics for a specific table.

    Parameters:
    -----------
    table_name : str
        Name of the table to display MCIDE for
    output_dir : str
        Base output directory
    """
    import json

    mcide_dir = os.path.join(output_dir, 'final', 'tableone', 'mcide')
    stats_dir = os.path.join(output_dir, 'final', 'tableone', 'summary_stats')

    # Check if MCIDE data exists
    if not os.path.exists(mcide_dir):
        st.info("📊 No MCIDE data available yet. MCIDE statistics will be collected automatically when you run validation.")
        return

    # Get MCIDE files for this table
    table_files = get_table_mcide_files(mcide_dir, table_name)

    # Get summary statistics files if they exist
    stats_files = []
    if os.path.exists(stats_dir):
        # Look for stats files for this table
        patterns = [
            f"{table_name}_*.json",
            f"*{table_name}_*.json"
        ]
        for pattern in patterns:
            stats_files.extend(glob.glob(os.path.join(stats_dir, pattern)))

    if not table_files and not stats_files:
        st.info(f"📊 No MCIDE data available for {table_name} yet.")
        st.write("MCIDE statistics will be collected automatically when you run validation for this table.")
        return

    # Display MCIDE counts
    if table_files:
        st.subheader("📋 MCIDE Value Counts")
        st.caption(f"Found {len(table_files)} MCIDE collection(s) for this table")

        for filepath in table_files:
            display_mcide_file(filepath, table_name)
            if len(table_files) > 1:
                st.divider()

    # Display summary statistics if available
    if stats_files:
        st.divider()
        st.subheader("📊 Summary Statistics")
        st.caption(f"Found {len(stats_files)} summary statistics file(s) for this table")

        for stats_file in sorted(stats_files):
            try:
                # Read JSON file
                with open(stats_file, 'r') as f:
                    stats_data = json.load(f)

                # Extract filename for display
                stats_name = os.path.basename(stats_file).replace('.json', '').replace('_', ' ').title()
                st.markdown(f"#### {stats_name}")

                # Convert to DataFrame for better display
                if isinstance(stats_data, list) and stats_data:
                    df_stats = pd.DataFrame(stats_data)

                    # Format numeric columns
                    numeric_cols = ['total_obs', 'n', 'missing', 'min', 'max', 'mean', 'median', 'sd', 'q1', 'q3']
                    for col in numeric_cols:
                        if col in df_stats.columns:
                            # Format based on column type
                            if col in ['total_obs', 'n', 'missing']:
                                df_stats[col] = df_stats[col].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "")
                            else:
                                df_stats[col] = df_stats[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")

                    # Reorder columns if possible
                    preferred_order = ['category', 'variable', 'total_obs', 'n', 'missing', 'min', 'q1', 'median', 'q3', 'max', 'mean', 'sd']
                    existing_cols = [col for col in preferred_order if col in df_stats.columns]
                    other_cols = [col for col in df_stats.columns if col not in existing_cols]
                    df_stats = df_stats[existing_cols + other_cols]

                    # Display the dataframe
                    st.dataframe(
                        df_stats,
                        use_container_width=True,
                        hide_index=True
                    )

                    # Download button for stats
                    csv_stats = pd.DataFrame(stats_data).to_csv(index=False)
                    # Use full file path for unique key to avoid duplicates
                    unique_key = f"download_stats_{os.path.basename(stats_file)}_{id(stats_file)}"
                    st.download_button(
                        label="📥 Download Statistics",
                        data=csv_stats,
                        file_name=f"{stats_name.replace(' ', '_').lower()}.csv",
                        mime="text/csv",
                        key=unique_key
                    )
                else:
                    st.json(stats_data)

            except Exception as e:
                st.error(f"Error loading statistics file: {str(e)}")

            if len(stats_files) > 1:
                st.divider()


def show_mcide_results(output_dir: str = 'output'):
    """
    Main entry point for showing MCIDE results.

    Parameters:
    -----------
    output_dir : str
        Base output directory
    """
    display_mcide_tab(output_dir)