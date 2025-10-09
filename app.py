"""
CLIF 2.1 Validation & Summarization System
Streamlit Application

This application provides an interactive interface for validating and analyzing
CLIF 2.1 data tables using clifpy.
"""

import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
from modules.tables import PatientAnalyzer
from modules.utils import (
    get_validation_summary,
    get_missingness_summary,
    create_missingness_report
)
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="CLIF 2.1 Validation & Summarization",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        margin: 0;
        padding: 0;
    }
    .main-header h3 {
        margin: 0.5rem 0 0 0;
        padding: 0;
        opacity: 0.9;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .status-complete {
        color: #00c851;
        font-weight: bold;
    }
    .status-partial {
        color: #ffbb33;
        font-weight: bold;
    }
    .status-incomplete {
        color: #ff4444;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Available tables mapping
TABLE_ANALYZERS = {
    'patient': PatientAnalyzer
}

TABLE_DISPLAY_NAMES = {
    'patient': 'Patient',
    'hospitalization': 'Hospitalization',
    'adt': 'ADT',
    'vitals': 'Vitals',
    'labs': 'Labs',
    'medication_admin_continuous': 'Medication Admin (Continuous)',
    'medication_admin_intermittent': 'Medication Admin (Intermittent)',
    'respiratory_support': 'Respiratory Support',
    'patient_assessments': 'Patient Assessments',
    'patient_procedures': 'Patient Procedures',
    'crrt_therapy': 'CRRT Therapy',
    'ecmo_mcs': 'ECMO/MCS',
    'microbiology_culture': 'Microbiology Culture',
    'microbiology_nonculture': 'Microbiology Non-Culture',
    'microbiology_susceptibility': 'Microbiology Susceptibility',
    'hospital_diagnosis': 'Hospital Diagnosis',
    'position': 'Position',
    'code_status': 'Code Status'
}


def load_config(config_path: str) -> dict:
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"❌ Configuration file not found: {config_path}")
        st.info("Please ensure the config file exists and contains:")
        st.code("""{
    "site_name": "Your Hospital Name",
    "site_id": "YOUR_ID",
    "tables_path": "/path/to/clif/tables",
    "filetype": "parquet",
    "timezone": "UTC",
    "output_dir": "output"
}""", language='json')
        return None
    except json.JSONDecodeError as e:
        st.error(f"❌ Invalid JSON in configuration file: {e}")
        return None


def main():
    """Main application function."""
    # Sidebar - Configuration file loading
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Config file path input
        config_path = st.text_input(
            "Configuration File Path",
            value="config/config.json",
            help="Path to your site's configuration JSON file"
        )

        # Load configuration
        config = None
        if config_path:
            config = load_config(config_path)

        if not config:
            st.error("Please provide a valid configuration file")
            st.stop()

        # Display site information
        st.success(f"✅ Site: {config.get('site_name', 'Unknown')}")
        st.info(f"📁 Data Path: {config.get('tables_path', 'Not specified')}")

        # Check if data directory exists
        data_path = config.get('tables_path', '')
        if not os.path.exists(data_path):
            st.warning(f"⚠️ Data directory not found: {data_path}")

        st.divider()

        # Table selection
        st.subheader("📊 Table Selection")

        # Currently only Patient table is implemented
        available_tables = ['patient']
        other_tables = [t for t in TABLE_DISPLAY_NAMES.keys() if t != 'patient']

        selected_table = st.selectbox(
            "Select Table to Analyze",
            options=available_tables,
            format_func=lambda x: TABLE_DISPLAY_NAMES[x]
        )

        if other_tables:
            with st.expander("🚧 Coming Soon"):
                st.write("The following tables will be available soon:")
                for table in other_tables[:5]:
                    st.write(f"• {TABLE_DISPLAY_NAMES[table]}")
                if len(other_tables) > 5:
                    st.write(f"• ... and {len(other_tables) - 5} more")

        st.divider()

        # Analysis options
        st.subheader("🔍 Analysis Options")
        run_validation = st.checkbox("Run Validation", value=True)
        run_outlier_handling = st.checkbox(
            "Apply Outlier Handling",
            value=config.get('analysis_settings', {}).get('outlier_detection', False)
        )
        calculate_sofa = st.checkbox(
            "Calculate SOFA Scores",
            value=config.get('analysis_settings', {}).get('calculate_sofa', False),
            disabled=True,
            help="Requires additional clinical tables"
        )

        # Run analysis button
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            st.session_state.run_analysis = True
            st.session_state.config = config
            st.session_state.selected_table = selected_table
            st.session_state.run_validation = run_validation
            st.session_state.run_outlier_handling = run_outlier_handling
            st.session_state.calculate_sofa = calculate_sofa

    # Header with logo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Check if logo exists
        logo_path = Path("assets/clif_logo.png")
        if logo_path.exists():
            st.image(str(logo_path), width=200, use_column_width=False)
        else:
            # Placeholder for logo
            st.markdown("🏥", unsafe_allow_html=True)

        st.markdown(
            f'<div class="main-header">'
            f'<h1>CLIF 2.1 Validation & Summarization System</h1>'
            f'<h3>{config.get("site_name", "")}</h3>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Main content area
    if 'run_analysis' in st.session_state and st.session_state.run_analysis:
        analyze_table(
            st.session_state.selected_table,
            st.session_state.config,
            st.session_state.run_validation,
            st.session_state.run_outlier_handling
        )
    else:
        # Welcome message
        st.info("""
        👈 **Getting Started:**
        1. Ensure your configuration file is set correctly
        2. Select a table to analyze
        3. Choose analysis options
        4. Click "Run Analysis" to begin

        **Configuration file should contain:**
        - `site_name`: Your site identifier
        - `tables_path`: Path to your CLIF data tables
        - `filetype`: Format of your data files (parquet/csv)
        - `timezone`: Timezone for datetime processing
        """)

        # Show sample data structure
        with st.expander("📁 Expected Data Structure"):
            st.write("""
            Your data directory should contain CLIF 2.1 tables.
            Files can be named with or without the 'clif_' prefix:

            ```
            data/
            ├── patient.parquet         OR  clif_patient.parquet
            ├── hospitalization.parquet OR  clif_hospitalization.parquet
            ├── adt.parquet            OR  clif_adt.parquet
            ├── vitals.parquet         OR  clif_vitals.parquet
            ├── labs.parquet           OR  clif_labs.parquet
            └── ...
            ```

            The system will automatically detect both naming conventions.
            """)


def analyze_table(table_name, config, run_validation, run_outlier_handling):
    """
    Analyze a specific table with validation and summary tabs.

    Parameters:
    -----------
    table_name : str
        Name of the table to analyze
    config : dict
        Configuration dictionary
    run_validation : bool
        Whether to run validation
    run_outlier_handling : bool
        Whether to apply outlier handling
    """
    st.header(f"📋 {TABLE_DISPLAY_NAMES[table_name]} Analysis")

    # Initialize analyzer
    analyzer_class = TABLE_ANALYZERS.get(table_name)

    if not analyzer_class:
        st.error(f"❌ Analyzer not implemented for {table_name}")
        st.info("Currently only the Patient table analyzer is available. Other tables coming soon!")
        return

    # Extract parameters from config
    data_dir = config.get('tables_path', './data')
    filetype = config.get('filetype', 'parquet')
    timezone = config.get('timezone', 'UTC')
    output_dir = config.get('output_dir', 'output')

    try:
        with st.spinner(f"Loading {TABLE_DISPLAY_NAMES[table_name]} table..."):
            analyzer = analyzer_class(data_dir, filetype, timezone, output_dir)

        # Create tabs for validation and summary
        tab1, tab2 = st.tabs(["🔍 Validation Results", "📊 Summary Statistics"])

        with tab1:
            display_validation_results(analyzer, run_validation)

        with tab2:
            display_summary_statistics(analyzer)

    except Exception as e:
        st.error(f"❌ Error loading table: {str(e)}")
        st.info("Please check:")
        st.write("1. The data path is correct in your config file")
        st.write(f"2. The {table_name}.{filetype} file exists in the data directory")
        st.write("3. The file format matches the configured filetype")
        st.write("4. You have clifpy installed: `pip install clifpy`")


def display_validation_results(analyzer, run_validation):
    """
    Display validation results tab.

    Parameters:
    -----------
    analyzer : BaseTableAnalyzer
        The table analyzer instance
    run_validation : bool
        Whether to run validation
    """
    if not run_validation:
        st.info("ℹ️ Validation not run. Check the 'Run Validation' box in the sidebar to enable.")
        return

    with st.spinner("Running validation..."):
        validation_results = analyzer.validate()

    # Display status with color coding
    status = validation_results.get('status', 'unknown')
    status_colors = {
        'complete': 'green',
        'partial': 'orange',
        'incomplete': 'red'
    }
    status_class = f'status-{status}'

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Validation Status</h4>
            <p class="{status_class}">{status.upper()}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        data_info = validation_results.get('data_info', {})
        st.metric("Total Rows", f"{data_info.get('row_count', 0):,}")

    with col3:
        st.metric("Total Columns", data_info.get('column_count', 0))

    with col4:
        errors = validation_results.get('errors', {})
        error_count = sum([
            len(errors.get('schema_errors', [])),
            len(errors.get('data_quality_issues', [])),
            len(errors.get('other_errors', []))
        ])
        st.metric("Total Issues", error_count)

    # Display validation summary
    st.markdown("### Validation Summary")
    summary = get_validation_summary(validation_results)
    if summary:
        st.info(summary)

    # Display errors by category
    st.markdown("### Validation Issues")

    if error_count == 0:
        st.success("✅ No validation issues found!")
    else:
        # Schema errors
        schema_errors = errors.get('schema_errors', [])
        if schema_errors:
            with st.expander(f"⚠️ Schema Errors ({len(schema_errors)})", expanded=True):
                for error in schema_errors:
                    st.markdown(f"**{error['type']}**")
                    st.write(error['description'])
                    st.divider()

        # Data quality issues
        quality_issues = errors.get('data_quality_issues', [])
        if quality_issues:
            with st.expander(f"📊 Data Quality Issues ({len(quality_issues)})", expanded=True):
                for issue in quality_issues:
                    st.markdown(f"**{issue['type']}**")
                    st.write(issue['description'])
                    st.divider()

        # Other errors
        other_errors = errors.get('other_errors', [])
        if other_errors:
            with st.expander(f"ℹ️ Other Issues ({len(other_errors)})"):
                for error in other_errors:
                    st.markdown(f"**{error['type']}**")
                    st.write(error['description'])
                    st.divider()

    # Save validation results
    if st.button("💾 Save Validation Results"):
        try:
            analyzer.save_summary_data(validation_results, '_validation')
            st.success(f"Validation results saved to {analyzer.output_dir}/final/")
        except Exception as e:
            st.error(f"Error saving results: {e}")


def display_summary_statistics(analyzer):
    """
    Display summary statistics tab.

    Parameters:
    -----------
    analyzer : BaseTableAnalyzer
        The table analyzer instance
    """
    # Get summary statistics
    with st.spinner("Calculating summary statistics..."):
        summary_stats = analyzer.get_summary_statistics()

    # Data Info Section
    st.markdown("### 📊 Data Overview")
    data_info = summary_stats.get('data_info', {})

    if 'error' in data_info:
        st.error(data_info['error'])
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", f"{data_info.get('row_count', 0):,}")
    with col2:
        st.metric("Unique Patients", f"{data_info.get('unique_patients', 0):,}")
    with col3:
        st.metric("Total Columns", data_info.get('column_count', 0))
    with col4:
        st.metric("Death Records", f"{data_info.get('has_death_records', 0):,}")

    # Missingness Analysis Section
    st.markdown("### 🔍 Missingness Analysis")
    missingness = summary_stats.get('missingness', {})

    if 'error' not in missingness:
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Complete Columns",
                     f"{missingness.get('complete_columns_count', 0)}/{missingness.get('total_columns', 0)}")
        with col2:
            st.metric("Overall Missing %",
                     f"{missingness.get('overall_missing_percentage', 0):.2f}%")
        with col3:
            st.metric("Complete Rows %",
                     f"{missingness.get('complete_rows_percentage', 0):.2f}%")

        # Missingness summary
        miss_summary = get_missingness_summary(analyzer.table.df if hasattr(analyzer, 'table') and
                                              hasattr(analyzer.table, 'df') else pd.DataFrame())
        if miss_summary:
            st.info(miss_summary)

        # Show columns with missing data
        if missingness.get('columns_with_missing'):
            with st.expander("Columns with Missing Data", expanded=False):
                missing_df = pd.DataFrame(missingness['columns_with_missing'])
                st.dataframe(
                    missing_df,
                    use_container_width=True,
                    hide_index=True
                )

                # Create bar chart for missingness
                if not missing_df.empty:
                    fig = px.bar(
                        missing_df.head(10),
                        x='column',
                        y='missing_percent',
                        title='Top 10 Columns by Missing Percentage',
                        labels={'missing_percent': 'Missing %', 'column': 'Column'},
                        color='missing_percent',
                        color_continuous_scale='reds'
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # Distribution Analysis Section
    st.markdown("### 📈 Distribution Analysis")
    distributions = summary_stats.get('distributions', {})

    if distributions and 'error' not in distributions:
        for key, dist_data in distributions.items():
            if isinstance(dist_data, dict) and 'error' not in dist_data:
                # Format the display name
                display_name = key.replace('_', ' ').title()
                st.markdown(f"#### {display_name}")

                # For categorical distributions
                if 'categories' in dist_data:
                    col1, col2 = st.columns(2)

                    with col1:
                        # Create pie chart
                        categories_df = pd.DataFrame(dist_data['categories'])
                        if not categories_df.empty:
                            fig = px.pie(
                                categories_df,
                                values='count',
                                names='value',
                                title=f"{display_name} Distribution"
                            )
                            st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        # Display statistics
                        st.write(f"**Unique Values:** {dist_data.get('unique_values', 0)}")
                        st.write(f"**Missing:** {dist_data.get('missing_count', 0)} "
                                f"({dist_data.get('missing_percentage', 0)}%)")
                        if dist_data.get('mode'):
                            st.write(f"**Mode:** {dist_data.get('mode')}")

                        # Show value counts table
                        if categories_df.shape[0] > 0:
                            st.dataframe(
                                categories_df[['value', 'count', 'percentage']].head(5),
                                use_container_width=True,
                                hide_index=True
                            )

                # For mortality statistics
                elif 'mortality_rate' in dist_data:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Deaths", f"{dist_data.get('death_count', 0):,}")
                    with col2:
                        st.metric("Alive", f"{dist_data.get('alive_count', 0):,}")
                    with col3:
                        st.metric("Mortality Rate", f"{dist_data.get('mortality_rate', 0):.2f}%")

                st.divider()

    # Patient-specific summary table
    if hasattr(analyzer, 'generate_patient_summary'):
        st.markdown("### 📋 Patient Demographics Summary")
        summary_df = analyzer.generate_patient_summary()
        if not summary_df.empty:
            st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Data quality checks for Patient table
    if hasattr(analyzer, 'check_data_quality'):
        st.markdown("### ✅ Data Quality Checks")
        quality_checks = analyzer.check_data_quality()

        if 'error' not in quality_checks:
            for check_name, check_result in quality_checks.items():
                status_icon = "✅" if check_result['status'] == 'pass' else "⚠️" if check_result['status'] == 'warning' else "❌"
                check_display = check_name.replace('_', ' ').title()
                st.write(f"{status_icon} **{check_display}:** {check_result['count']} "
                        f"({check_result['percentage']}%)")

    # Save summary statistics
    if st.button("💾 Save Summary Statistics"):
        try:
            analyzer.save_summary_data(summary_stats, '_summary')
            st.success(f"Summary statistics saved to {analyzer.output_dir}/final/")
        except Exception as e:
            st.error(f"Error saving results: {e}")


if __name__ == "__main__":
    main()