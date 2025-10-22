"""
Table One Results Viewer Module

This module provides functions to display Table One generation results
in the Streamlit app, organized into multiple tabs for different analyses.
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path
import json


def check_tableone_results_available(output_dir='output'):
    """
    Check if Table One results are available.

    Parameters
    ----------
    output_dir : str
        Base output directory

    Returns
    -------
    bool
        True if Table One results exist
    """
    tableone_dir = Path(output_dir) / 'final' / 'tableone'

    # Check for key output files
    required_files = [
        'consort_flow_diagram.png',
        'table_one_overall.csv'
    ]

    if not tableone_dir.exists():
        return False

    for file in required_files:
        if not (tableone_dir / file).exists():
            return False

    return True


def display_cohort_tab(tableone_dir):
    """Display cohort diagrams and visualizations."""
    st.header("📊 Cohort Analysis")

    # CONSORT Flow Diagram
    consort_path = tableone_dir / 'consort_flow_diagram.png'
    if consort_path.exists():
        st.subheader("CONSORT Flow Diagram")
        st.image(str(consort_path), use_container_width=True)
    else:
        st.warning("⚠️ CONSORT flow diagram not found")

    st.divider()

    # UpSet Plot
    upset_path = tableone_dir / 'cohort_intersect_upset_plot.png'
    if upset_path.exists():
        st.subheader("Cohort Intersections (UpSet Plot)")
        st.caption("Displays overlaps between different cohort definitions")
        st.image(str(upset_path), use_container_width=True)
    else:
        st.info("ℹ️ UpSet plot not found")

    st.divider()

    # Venn Diagram
    venn_path = tableone_dir / 'venn_all_4_groups.png'
    if venn_path.exists():
        st.subheader("Cohort Venn Diagram")
        st.caption("4-way Venn diagram showing cohort overlaps")
        st.image(str(venn_path), use_container_width=True)
    else:
        st.info("ℹ️ Venn diagram not found")

    st.divider()

    # Code Status Visualization
    code_status_path = tableone_dir / 'code_status_stacked_bar_with_missingness_excl_missing_cat.png'
    if code_status_path.exists():
        st.subheader("Code Status Distribution")
        st.image(str(code_status_path), use_container_width=True)
    else:
        st.info("ℹ️ Code status visualization not found")

    st.divider()

    # Sankey Diagrams
    st.subheader("Patient Flow Sankey Diagrams")

    figures_dir = tableone_dir / 'figures'
    if figures_dir.exists():
        sankey_files = [
            ('sankey_matplotlib_icu.png', 'ICU Patients'),
            ('sankey_matplotlib_others.png', 'Other Patients'),
            ('sankey_matplotlib_high_o2_support.png', 'High O2 Support'),
            ('sankey_matplotlib_vaso_support.png', 'Vasopressor Support')
        ]

        cols = st.columns(2)
        for idx, (filename, title) in enumerate(sankey_files):
            sankey_path = figures_dir / filename
            if sankey_path.exists():
                with cols[idx % 2]:
                    st.caption(f"**{title}**")
                    st.image(str(sankey_path), use_container_width=True)
    else:
        st.info("ℹ️ Sankey diagrams not found")


def display_demographics_tab(tableone_dir):
    """Display demographics and Table One results."""
    st.header("👥 Demographics & Table One")

    # Extract and display key metrics at the top
    table_by_year_path = tableone_dir / 'table_one_by_year.csv'
    if table_by_year_path.exists():
        df_year = pd.read_csv(table_by_year_path)

        # Try to extract key metrics from the table
        try:
            # Create metric cards - Row 1
            st.subheader("📈 Key Cohort Metrics")

            cols = st.columns(5)

            # Metric 1: Total Patients
            with cols[0]:
                if 'Variable' in df_year.columns and 'Overall' in df_year.columns:
                    patient_rows = df_year[df_year['Variable'].str.contains('n|Number|Total|Encounters', case=False, na=False)]
                    if len(patient_rows) > 0:
                        total_val = patient_rows.iloc[0]['Overall']
                        if pd.notna(total_val):
                            total_str = str(total_val).replace(',', '').split()[0]
                            try:
                                total_patients = int(float(total_str))
                                st.metric("👥 Hospitalizations", f"{total_patients:,}")
                            except:
                                st.metric("👥 Hospitalizations", total_val)

            # Metric 2: Age
            with cols[1]:
                age_rows = df_year[df_year['Variable'].str.contains('Age|age', case=False, na=False)]
                if len(age_rows) > 0:
                    age_val = age_rows.iloc[0]['Overall']
                    if pd.notna(age_val):
                        st.metric("🎂 Median Age", age_val)

            # Metric 3: Gender (Female %)
            with cols[2]:
                female_rows = df_year[df_year['Variable'].str.contains('Female|Sex.*F|Gender.*F', case=False, na=False)]
                if len(female_rows) > 0:
                    female_val = female_rows.iloc[0]['Overall']
                    if pd.notna(female_val):
                        st.metric("👩 Female", female_val)

            # Metric 4: Mortality
            with cols[3]:
                mort_rows = df_year[df_year['Variable'].str.contains('Mortality|Died|Death|died', case=False, na=False)]
                if len(mort_rows) > 0:
                    mort_val = mort_rows.iloc[0]['Overall']
                    if pd.notna(mort_val):
                        st.metric("💔 Mortality", mort_val)

            # Metric 5: ICU Length of Stay
            with cols[4]:
                los_rows = df_year[df_year['Variable'].str.contains('ICU.*LOS|ICU.*length|ICU.*stay', case=False, na=False)]
                if len(los_rows) > 0:
                    los_val = los_rows.iloc[0]['Overall']
                    if pd.notna(los_val):
                        st.metric("⏱️ ICU LOS", los_val)

            # Row 2 - Additional metrics
            st.caption("")  # Small spacer
            cols2 = st.columns(5)

            # Metric 6: Mechanical Ventilation
            with cols2[0]:
                mech_vent_rows = df_year[df_year['Variable'].str.contains('Mechanical.*Vent|IMV|Invasive.*Vent', case=False, na=False)]
                if len(mech_vent_rows) > 0:
                    mv_val = mech_vent_rows.iloc[0]['Overall']
                    if pd.notna(mv_val):
                        st.metric("🫁 Mech Ventilation", mv_val)

            # Metric 7: Vasopressors
            with cols2[1]:
                vaso_rows = df_year[df_year['Variable'].str.contains('Vasopressor|Vaso|Pressor', case=False, na=False)]
                if len(vaso_rows) > 0:
                    vaso_val = vaso_rows.iloc[0]['Overall']
                    if pd.notna(vaso_val):
                        st.metric("💉 Vasopressors", vaso_val)

            # Metric 8: CRRT
            with cols2[2]:
                crrt_rows = df_year[df_year['Variable'].str.contains('CRRT|Dialysis|RRT', case=False, na=False)]
                if len(crrt_rows) > 0:
                    crrt_val = crrt_rows.iloc[0]['Overall']
                    if pd.notna(crrt_val):
                        st.metric("🩺 CRRT", crrt_val)

            # Metric 9: Hospital LOS
            with cols2[3]:
                hosp_los_rows = df_year[df_year['Variable'].str.contains('Hospital.*LOS|Hosp.*length|Hospital.*stay', case=False, na=False)]
                if len(hosp_los_rows) > 0:
                    hosp_los_val = hosp_los_rows.iloc[0]['Overall']
                    if pd.notna(hosp_los_val):
                        st.metric("🏥 Hospital LOS", hosp_los_val)

            # Metric 10: Discharge to hospice
            with cols2[4]:
                hospice_rows = df_year[df_year['Variable'].str.contains('Hospice|Palliative', case=False, na=False)]
                if len(hospice_rows) > 0:
                    hospice_val = hospice_rows.iloc[0]['Overall']
                    if pd.notna(hospice_val):
                        st.metric("🕊️ Hospice Discharge", hospice_val)

            st.divider()
        except Exception as e:
            # If metric extraction fails, just skip it
            pass

    # Table One by Year
    if table_by_year_path.exists():
        st.subheader("Table One - Stratified by Admission Year")
        df = pd.read_csv(table_by_year_path)
        st.dataframe(df, use_container_width=True, height=600)

        # Download button
        st.download_button(
            label="📥 Download Table One (by year)",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='table_one_by_year.csv',
            mime='text/csv'
        )
    else:
        st.warning("⚠️ Table One by year not found")

    st.divider()

    # Overall Table One
    table_overall_path = tableone_dir / 'table_one_overall.csv'
    if table_overall_path.exists():
        st.subheader("Table One - Overall Cohort")
        df = pd.read_csv(table_overall_path)
        st.dataframe(df, use_container_width=True, height=600)

        # Download button
        st.download_button(
            label="📥 Download Table One (overall)",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='table_one_overall.csv',
            mime='text/csv'
        )
    else:
        st.warning("⚠️ Overall Table One not found")

    st.divider()

    # Summary Statistics
    st.subheader("📋 Data Summary")

    col1, col2 = st.columns(2)

    with col1:
        # Show basic cohort stats if available
        upset_data_path = tableone_dir / 'upset_data.csv'
        if upset_data_path.exists():
            upset_df = pd.read_csv(upset_data_path)
            st.metric("Total Cohort Combinations", len(upset_df))
            st.caption("Number of unique cohort intersections")

    with col2:
        # Show final dataframe size if available
        final_df_path = tableone_dir / 'final_tableone_df.parquet'
        if final_df_path.exists():
            try:
                final_df = pd.read_parquet(final_df_path)
                st.metric("Total Hospitalizations", f"{len(final_df):,}")
                st.metric("Total Columns", len(final_df.columns))
            except Exception as e:
                st.caption(f"Could not read final dataframe: {e}")


def display_medications_tab(tableone_dir):
    """Display vasoactive, sedative, and paralytic medication visualizations."""
    st.header("💊 Medications Analysis")
    st.caption("Vasoactives, Sedatives, and Paralytics")

    # Vasoactive medications
    st.subheader("💉 Vasoactive Medications")

    vaso_area_path = tableone_dir / 'vasoactive_area_curve_7d.html'
    if vaso_area_path.exists():
        st.caption("**Area Under Curve (7 days)**")
        with open(vaso_area_path, 'r') as f:
            html_content = f.read()
        # Increased height to show full plot with legend
        st.components.v1.html(html_content, height=700, scrolling=False)
    else:
        st.info("ℹ️ Vasoactive area curve not found")

    vaso_dose_path = tableone_dir / 'vasoactive_median_dose_by_hour.html'
    if vaso_dose_path.exists():
        st.caption("**Median Dose by Hour**")
        with open(vaso_dose_path, 'r') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=700, scrolling=False)
    else:
        st.info("ℹ️ Vasoactive median dose plot not found")

    st.divider()

    # Sedative medications
    st.subheader("😴 Sedative Medications")

    sedative_area_path = tableone_dir / 'sedative_area_curve_7d.html'
    if sedative_area_path.exists():
        st.caption("**Area Under Curve (7 days)**")
        with open(sedative_area_path, 'r') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=700, scrolling=False)
    else:
        st.info("ℹ️ Sedative area curve not found")

    sedative_dose_path = tableone_dir / 'sedative_median_dose_by_hour.html'
    if sedative_dose_path.exists():
        st.caption("**Median Dose by Hour**")
        with open(sedative_dose_path, 'r') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=700, scrolling=False)
    else:
        st.info("ℹ️ Sedative median dose plot not found")

    st.divider()

    # Paralytic medications
    st.subheader("🦴 Paralytic Medications")

    paralytic_area_path = tableone_dir / 'paralytic_area_curve_7d.html'
    if paralytic_area_path.exists():
        st.caption("**Area Under Curve (7 days)**")
        with open(paralytic_area_path, 'r') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=700, scrolling=False)
    else:
        st.info("ℹ️ Paralytic area curve not found")

    paralytic_dose_path = tableone_dir / 'paralytic_median_dose_by_hour.html'
    if paralytic_dose_path.exists():
        st.caption("**Median Dose by Hour**")
        with open(paralytic_dose_path, 'r') as f:
            html_content = f.read()
        st.components.v1.html(html_content, height=700, scrolling=False)
    else:
        st.info("ℹ️ Paralytic median dose plot not found")

    st.divider()

    # Summary statistics
    meds_summary_path = tableone_dir / 'medications_summary_stats.csv'
    if meds_summary_path.exists():
        st.subheader("📊 Medication Summary Statistics")
        df = pd.read_csv(meds_summary_path)
        st.dataframe(df, use_container_width=True)

        st.download_button(
            label="📥 Download Medication Summary",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='medications_summary_stats.csv',
            mime='text/csv'
        )
    else:
        st.info("ℹ️ Medication summary statistics not found")


def display_imv_tab(tableone_dir):
    """Display invasive mechanical ventilation visualizations."""
    st.header("🫁 Invasive Mechanical Ventilation")

    # Tidal volume
    tidal_volume_path = tableone_dir / 'tidal_volume_volume_control_modes.png'
    if tidal_volume_path.exists():
        st.subheader("Tidal Volume - Volume Control Modes")
        st.image(str(tidal_volume_path), use_container_width=True)
    else:
        st.warning("⚠️ Tidal volume visualization not found")

    st.divider()

    # Pressure control
    pressure_control_path = tableone_dir / 'pressure_control_pressure_control_mode.png'
    if pressure_control_path.exists():
        st.subheader("Pressure Control - Pressure Control Mode")
        st.image(str(pressure_control_path), use_container_width=True)
    else:
        st.warning("⚠️ Pressure control visualization not found")

    st.divider()

    # Mode proportions
    mode_prop_path = tableone_dir / 'mode_proportions_first_24h_vertical.png'
    if mode_prop_path.exists():
        st.subheader("Ventilation Mode Proportions (First 24 hours)")
        st.image(str(mode_prop_path), use_container_width=True)
    else:
        st.warning("⚠️ Mode proportions visualization not found")

    st.divider()

    # Ventilator Settings Table
    st.subheader("📊 Ventilator Settings by Device Mode")

    # Display the combined ventilator settings table image
    vent_table_path = tableone_dir / 'ventilator_settings_table.png'
    if vent_table_path.exists():
        st.image(str(vent_table_path), use_container_width=True)

        # Provide download buttons for the underlying data
        col1, col2 = st.columns(2)

        with col1:
            # Download settings summary CSV
            vent_settings_path = tableone_dir / 'ventilator_settings_by_device_mode.csv'
            if vent_settings_path.exists():
                df = pd.read_csv(vent_settings_path)
                st.download_button(
                    label="📥 Download Settings Summary (CSV)",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name='ventilator_settings_by_device_mode.csv',
                    mime='text/csv',
                    key='vent_settings_summary'
                )

        with col2:
            # Download counts CSV
            vent_counts_path = tableone_dir / 'ventilator_settings_counts_by_device_mode.csv'
            if vent_counts_path.exists():
                df = pd.read_csv(vent_counts_path)
                st.download_button(
                    label="📥 Download Settings Counts (CSV)",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name='ventilator_settings_counts_by_device_mode.csv',
                    mime='text/csv',
                    key='vent_settings_counts'
                )
    else:
        # Fall back to displaying CSVs if image doesn't exist
        st.info("ℹ️ Ventilator settings table image not found. Displaying raw data instead.")

        col1, col2 = st.columns(2)

        with col1:
            # Ventilator settings summary
            vent_settings_path = tableone_dir / 'ventilator_settings_by_device_mode.csv'
            if vent_settings_path.exists():
                st.caption("**Ventilator Settings Summary**")
                df = pd.read_csv(vent_settings_path)
                st.dataframe(df, use_container_width=True, height=400)

                st.download_button(
                    label="📥 Download Settings Summary",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name='ventilator_settings_by_device_mode.csv',
                    mime='text/csv',
                    key='vent_settings_summary_fallback'
                )
            else:
                st.info("ℹ️ Ventilator settings summary not found")

        with col2:
            # Ventilator settings counts
            vent_counts_path = tableone_dir / 'ventilator_settings_counts_by_device_mode.csv'
            if vent_counts_path.exists():
                st.caption("**Ventilator Settings Counts**")
                df = pd.read_csv(vent_counts_path)
                st.dataframe(df, use_container_width=True, height=400)

                st.download_button(
                    label="📥 Download Settings Counts",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name='ventilator_settings_counts_by_device_mode.csv',
                    mime='text/csv',
                    key='vent_settings_counts_fallback'
                )
            else:
                st.info("ℹ️ Ventilator settings counts not found")


def display_comorbidities_tab(tableone_dir):
    """Display SOFA scores and comorbidity analysis."""
    st.header("📊 SOFA & CCI")

    # SOFA scores
    sofa_path = tableone_dir / 'sofa_mortality_histogram.png'
    if sofa_path.exists():
        st.subheader("SOFA Score & Mortality")
        st.image(str(sofa_path), use_container_width=True)
    else:
        st.info("ℹ️ SOFA mortality histogram not found")

    st.divider()

    # CCI analysis
    cci_path = tableone_dir / 'cci_mortality_hospice_comprehensive.png'
    if cci_path.exists():
        st.subheader("Charlson Comorbidity Index - Mortality & Hospice Analysis")
        st.image(str(cci_path), use_container_width=True)
    else:
        st.warning("⚠️ CCI comprehensive analysis not found")

    st.divider()

    # Comorbidities
    comorbid_path = tableone_dir / 'comorbidities_per_1000_barplot.png'
    if comorbid_path.exists():
        st.subheader("Comorbidity Prevalence")
        st.image(str(comorbid_path), use_container_width=True)

        # Show CSV if available
        comorbid_csv_path = tableone_dir / 'comorbidities_per_1000_hospitalizations.csv'
        if comorbid_csv_path.exists():
            with st.expander("📋 View Comorbidity Data"):
                df = pd.read_csv(comorbid_csv_path)
                st.dataframe(df, use_container_width=True)

                st.download_button(
                    label="📥 Download Comorbidity Data",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name='comorbidities_per_1000_hospitalizations.csv',
                    mime='text/csv'
                )
    else:
        st.info("ℹ️ Comorbidity prevalence plot not found")


def display_hospice_tab(tableone_dir):
    """Display hospice and mortality outcome visualizations."""
    st.header("🏥 Hospice & Mortality Outcomes")

    # Hospice mortality trends
    hospice_trends_path = tableone_dir / 'hospice_mortality_combined_trends.png'
    if hospice_trends_path.exists():
        st.subheader("Hospice & Mortality Combined Trends")
        st.image(str(hospice_trends_path), use_container_width=True)
    else:
        st.warning("⚠️ Hospice mortality trends not found")


def show_tableone_results(output_dir='output'):
    """
    Display Table One results in a tabbed interface.

    Parameters
    ----------
    output_dir : str
        Base output directory
    """
    tableone_dir = Path(output_dir) / 'final' / 'tableone'

    if not tableone_dir.exists():
        st.error("❌ Table One results directory not found")
        return

    # Header
    st.title("📊 Table One Results")
    st.caption("Comprehensive cohort analysis and visualization results")

    # Command line instructions
    st.info("💡 **To update Table One results, run the following command:**")
    st.code(
        "uv run run_tableone.py",
        language="bash"
    )
    st.caption("This will generate Table One statistics and visualizations based on your validated data.")

    # Execution report summary if available
    report_path = tableone_dir / 'execution_report.txt'
    if report_path.exists():
        with st.expander("ℹ️ Generation Summary", expanded=False):
            with open(report_path, 'r') as f:
                report_content = f.read()
            st.text(report_content)

    st.divider()

    # Create tabs
    tabs = st.tabs([
        "🏥 Cohort",
        "👥 Demographics",
        "💊 Medications",
        "🫁 IMV",
        "📊 SOFA & CCI",
        "🏥 Hospice & Outcomes"
    ])

    with tabs[0]:
        display_cohort_tab(tableone_dir)

    with tabs[1]:
        display_demographics_tab(tableone_dir)

    with tabs[2]:
        display_medications_tab(tableone_dir)

    with tabs[3]:
        display_imv_tab(tableone_dir)

    with tabs[4]:
        display_comorbidities_tab(tableone_dir)

    with tabs[5]:
        display_hospice_tab(tableone_dir)
