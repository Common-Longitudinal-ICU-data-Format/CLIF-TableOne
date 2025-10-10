"""
Categorical-Numeric Visualization Utility

This module provides reusable visualization functions for displaying numeric distributions
by categorical variables, useful for wide tables like CRRT therapy and respiratory support.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Optional


def show_categorical_numeric_distribution(
    df: pd.DataFrame,
    categorical_columns: List[str],
    numeric_columns: List[str],
    table_name: str,
    default_categorical: Optional[str] = None,
    default_numeric: Optional[str] = None,
    raw_df: Optional[pd.DataFrame] = None
):
    """
    Display interactive visualizations of numeric distributions by categorical variables.

    This function creates a dropdown for selecting one numeric variable, and creates
    a dropdown for each categorical variable to filter/view distributions.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the cleaned data (outliers removed)
    categorical_columns : List[str]
        List of categorical column names for filtering
    numeric_columns : List[str]
        List of numeric column names available for selection
    table_name : str
        Name of the table (for display purposes)
    default_categorical : str, optional
        Default categorical column to select (not used in new design)
    default_numeric : str, optional
        Default numeric column to select
    raw_df : pd.DataFrame, optional
        The raw dataframe before outlier removal (for dual plot comparison)
    """
    if df is None or df.empty:
        st.warning("No data available for visualization")
        return

    # Filter columns that exist in dataframe
    available_categorical = [col for col in categorical_columns if col in df.columns]
    available_numeric = [col for col in numeric_columns if col in df.columns]

    if not available_categorical or not available_numeric:
        st.warning("Not enough columns available for categorical-numeric visualization")
        return

    # Set default numeric
    if default_numeric is None or default_numeric not in available_numeric:
        default_numeric = available_numeric[0]

    # Create numeric variable selector
    selected_numeric = st.selectbox(
        "Select Numeric Variable",
        options=available_numeric,
        index=available_numeric.index(default_numeric),
        format_func=lambda x: x.replace('_', ' ').title(),
        key=f"{table_name}_numeric_select"
    )

    num_display = selected_numeric.replace('_', ' ').title()

    # Create dropdown selectors for each categorical variable
    for cat_col in available_categorical:
        if cat_col not in df.columns:
            continue

        cat_display = cat_col.replace('_', ' ').title()

        # Get unique categories (no 'All' option)
        unique_cats = sorted([str(x) for x in df[cat_col].dropna().unique()])

        # Create dropdown for this category variable
        selected_cat = st.selectbox(
            f"Select {cat_display}",
            options=unique_cats,
            index=0,  # Default to first category
            key=f"{table_name}_{cat_col}_filter"
        )

        # Filter cleaned data based on selection
        cat_valid_data = df[
            (df[cat_col].astype(str) == selected_cat) &
            (df[selected_numeric].notna())
        ].copy()

        # Filter raw data if available
        raw_cat_valid_data = None
        if raw_df is not None and not raw_df.empty:
            raw_cat_valid_data = raw_df[
                (raw_df[cat_col].astype(str) == selected_cat) &
                (raw_df[selected_numeric].notna())
            ].copy()

        if cat_valid_data.empty:
            st.warning(f"No valid data available for {cat_display}: {selected_cat}")
            continue

        st.markdown(f"### {num_display} by {cat_display}")

        # Create visualizations - dual plots if raw data available
        if raw_cat_valid_data is not None and not raw_cat_valid_data.empty:
            # Calculate number of observations replaced with NAs
            raw_count = len(raw_cat_valid_data)
            clean_count = len(cat_valid_data)
            replaced_count = raw_count - clean_count
            
            # Show banner about replaced observations
            if replaced_count > 0:
                st.info(f"ℹ️ **{replaced_count:,} observations** ({replaced_count/raw_count*100:.1f}%) replaced with NA due to outlier thresholds")
            
            # Dual violin plots: raw vs cleaned in 2 columns
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Raw Data**")
                fig_raw = px.violin(
                    raw_cat_valid_data,
                    y=selected_numeric,
                    box=True,
                    points='outliers',
                    title=f'{num_display} - Raw'
                )

                fig_raw.update_layout(
                    yaxis_title=num_display,
                    height=400,
                    showlegend=False
                )

                st.plotly_chart(fig_raw, use_container_width=True)
                
                # Summary statistics for raw data
                raw_category_data = raw_cat_valid_data[selected_numeric]
                raw_stats = [{
                    'Metric': 'Count',
                    'Value': len(raw_category_data)
                }, {
                    'Metric': 'Mean',
                    'Value': round(raw_category_data.mean(), 2)
                }, {
                    'Metric': 'Median',
                    'Value': round(raw_category_data.median(), 2)
                }, {
                    'Metric': 'Std',
                    'Value': round(raw_category_data.std(), 2)
                }, {
                    'Metric': 'Min',
                    'Value': round(raw_category_data.min(), 2)
                }, {
                    'Metric': 'Max',
                    'Value': round(raw_category_data.max(), 2)
                }]
                st.dataframe(pd.DataFrame(raw_stats), hide_index=True, width='stretch')

            with col2:
                st.markdown("**Outliers Removed**")
                fig_clean = px.violin(
                    cat_valid_data,
                    y=selected_numeric,
                    box=True,
                    points='outliers',
                    title=f'{num_display} - Cleaned'
                )

                fig_clean.update_layout(
                    yaxis_title=num_display,
                    height=400,
                    showlegend=False
                )

                st.plotly_chart(fig_clean, use_container_width=True)
                
                # Summary statistics for cleaned data
                category_data = cat_valid_data[selected_numeric]
                clean_stats = [{
                    'Metric': 'Count',
                    'Value': len(category_data)
                }, {
                    'Metric': 'Mean',
                    'Value': round(category_data.mean(), 2)
                }, {
                    'Metric': 'Median',
                    'Value': round(category_data.median(), 2)
                }, {
                    'Metric': 'Std',
                    'Value': round(category_data.std(), 2)
                }, {
                    'Metric': 'Min',
                    'Value': round(category_data.min(), 2)
                }, {
                    'Metric': 'Max',
                    'Value': round(category_data.max(), 2)
                }]
                st.dataframe(pd.DataFrame(clean_stats), hide_index=True, width='stretch')

        else:
            # Single plot mode (fallback)
            col1, col2 = st.columns([2, 1])

            with col1:
                # Violin plot for selected category only
                fig = px.violin(
                    cat_valid_data,
                    y=selected_numeric,
                    box=True,
                    points='outliers',
                    title=f'{num_display} Distribution - {cat_display}: {selected_cat}'
                )

                fig.update_layout(
                    yaxis_title=num_display,
                    height=500
                )

                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Summary statistics table
                st.write("**Summary Statistics:**")

                # Show stats for selected category only
                category_data = cat_valid_data[selected_numeric]
                summary_stats = [{
                    'Category': selected_cat,
                    'Count': len(category_data),
                    'Mean': round(category_data.mean(), 2),
                    'Median': round(category_data.median(), 2),
                    'Std': round(category_data.std(), 2),
                    'Min': round(category_data.min(), 2),
                    'Max': round(category_data.max(), 2)
                }]

                if summary_stats:
                    summary_df = pd.DataFrame(summary_stats)
                    st.dataframe(summary_df, hide_index=True, width='stretch')

        st.divider()  # Spacing  # Spacing  # Spacing
