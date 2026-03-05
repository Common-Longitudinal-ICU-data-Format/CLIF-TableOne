"""
CLIF 2.1 Validation & Summarization System
Streamlit Application

This application provides an interactive interface for validating and analyzing
CLIF 2.1 data tables using clifpy.
"""

import sys
import io

# Force UTF-8 encoding for Windows compatibility
# This ensures emojis and Unicode characters display correctly
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except (AttributeError, TypeError):
        # Fallback if running in an environment where stdout.buffer doesn't exist
        pass

import warnings
# Suppress Plotly and Streamlit deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='plotly')
warnings.filterwarnings('ignore', category=FutureWarning, module='streamlit')
warnings.filterwarnings('ignore', message='.*keyword arguments have been deprecated.*')

import streamlit as st
import pandas as pd
import json
import os
from pathlib import Path
from modules.tables import (
    PatientAnalyzer, HospitalizationAnalyzer, ADTAnalyzer, CodeStatusAnalyzer,
    CRRTTherapyAnalyzer, ECMOMCSAnalyzer, HospitalDiagnosisAnalyzer, LabsAnalyzer,
    MedicationAdminContinuousAnalyzer, MedicationAdminIntermittentAnalyzer,
    MicrobiologyCultureAnalyzer, MicrobiologyNoncultureAnalyzer,
    MicrobiologySusceptibilityAnalyzer, PatientAssessmentsAnalyzer,
    PatientProceduresAnalyzer, PositionAnalyzer, RespiratorySupportAnalyzer,
    VitalsAnalyzer
)
from modules.cli import ValidationPDFGenerator
from modules.cli.pdf_generator import _collect_dqa_issues, DQA_CATEGORIES
from modules.utils import (
    get_missingness_summary,
    create_missingness_report,
    create_error_id,
    create_feedback_structure,
    update_user_decision,
    get_feedback_summary,
    save_feedback,
    load_feedback,
    initialize_cache,
    cache_analysis,
    get_cached_analysis,
    is_table_cached,
    clear_all_cache,
    get_table_status,
    format_cache_timestamp,
    update_feedback_in_cache,
    get_completion_status,
    get_status_display,
    show_categorical_numeric_distribution
)
from modules.tableone.viewer import check_tableone_results_available, show_tableone_results
from modules.mcide.viewer import display_table_mcide
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="CLIF 2.1 Validation & Summarization",
    page_icon="images/clif_logo_red_no_text.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# Custom CSS for styling (theme colors from config.toml)
st.markdown("""
<style>
    /* Global font size increase */
    html {
        font-size: 18px;
    }

    .main-header {
        text-align: center;
        padding: 0.5rem;
        margin-bottom: 2rem;
    }
    .main-header h3 {
        margin: 0.5rem 0 0 0;
        padding: 0;
        font-weight: 400;
        font-size: 1.1rem;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.5rem;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] button {
        padding: 1.2rem 2.5rem;
        border-radius: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.8rem;
    }

    .metric-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }

    /* Left-align text in secondary buttons (status grid) */
    .stButton > button[kind="secondary"] {
        text-align: left !important;
        justify-content: flex-start !important;
    }

    .stButton > button[kind="secondary"] > div {
        justify-content: flex-start !important;
        text-align: left !important;
        width: 100%;
    }

    .stButton > button[kind="secondary"] > div > div {
        justify-content: flex-start !important;
        text-align: left !important;
        width: 100%;
    }

    .stButton > button[kind="secondary"] p {
        text-align: left !important;
        justify-content: flex-start !important;
        margin: 0 !important;
    }

    .stButton > button[kind="secondary"] div[data-testid="stMarkdownContainer"] {
        text-align: left !important;
        justify-content: flex-start !important;
        width: 100%;
    }

    /* Target all nested divs within secondary buttons */
    button[kind="secondary"] * {
        text-align: left !important;
    }
    .status-block-complete {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 2px solid #28a745;
        background-color: rgba(40, 167, 69, 0.1);
    }
    .status-block-partial {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 2px solid #ffc107;
        background-color: rgba(255, 193, 7, 0.1);
    }
    .status-block-incomplete {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 2px solid #dc3545;
        background-color: rgba(220, 53, 69, 0.1);
    }
    .status-block-complete h4 {
        margin: 0;
        font-weight: bold;
        color: #28a745;
    }
    .status-block-partial h4 {
        margin: 0;
        font-weight: bold;
        color: #ffc107;
    }
    .status-block-incomplete h4 {
        margin: 0;
        font-weight: bold;
        color: #dc3545;
    }
    .status-complete {
        font-weight: bold;
        color: #28a745;
    }
    .status-partial {
        font-weight: bold;
        color: #ffc107;
    }
    .status-incomplete {
        font-weight: bold;
        color: #dc3545;
    }

</style>
""", unsafe_allow_html=True)

# Available tables mapping
TABLE_ANALYZERS = {
    'patient': PatientAnalyzer,
    'hospitalization': HospitalizationAnalyzer,
    'adt': ADTAnalyzer,
    'code_status': CodeStatusAnalyzer,
    'crrt_therapy': CRRTTherapyAnalyzer,
    'ecmo_mcs': ECMOMCSAnalyzer,
    'hospital_diagnosis': HospitalDiagnosisAnalyzer,
    'labs': LabsAnalyzer,
    'medication_admin_continuous': MedicationAdminContinuousAnalyzer,
    'medication_admin_intermittent': MedicationAdminIntermittentAnalyzer,
    'microbiology_culture': MicrobiologyCultureAnalyzer,
    'microbiology_nonculture': MicrobiologyNoncultureAnalyzer,
    'microbiology_susceptibility': MicrobiologySusceptibilityAnalyzer,
    'patient_assessments': PatientAssessmentsAnalyzer,
    'patient_procedures': PatientProceduresAnalyzer,
    'position': PositionAnalyzer,
    'respiratory_support': RespiratorySupportAnalyzer,
    'vitals': VitalsAnalyzer
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


def show_instructions():
    """Display the Instructions page with workflow guide."""
    st.header("📖 How to Use This App")

    st.markdown("""
    ### What's This About?

    This app validates your CLIF 2.1 data against the official spec. It'll flag issues, but here's the thing—some "errors" might be totally fine for your site. The data at your site might not use a certain drug or a device, and that's totally fine. 

    ### Your Workflow:

    **1. Review Each Table**
    - Click through the tables in the grid on the Home page
    - Check out the validation results and summary stats
    - See what's flagged

    **2. Mark Site-Specific Stuff**
    - In the Validation tab, enable "📋 Review Status-Affecting Errors"
    - For each error, decide:
      - **Accept** → Legit issue, needs fixing
      - **Reject** → Not a problem for your site (add a reason why)
      - **Pending** → You'll deal with it later
    - Hit "💾 Save Feedback" when done

    **3. Regenerate the Combined Report**
    - Once you've reviewed all tables, head back to the Home page
    - Click "📄 Regenerate All Reports"
    - This creates a fresh combined report with your feedback applied

    **4. Need to Update Just One Table?**
    - Select that table from the dropdown
    - Check "🔄 Re-analyze table"
    - Click "🚀 Run Analysis"
    - Done. Just that table gets refreshed.

    **5. Check Out Table One**
    - Once generated, click "📊 Table One Results" in the sidebar
    - Explore cohorts, demographics, meds, ventilation, outcomes
    - All the good stuff in one place

    ---

    **Questions?** Hit me up on CLIF Slack- Kaveri Chhikara. You got this. 🚀
    """)


def show_home_page(config: dict, available_tables: list):
    """
    Display the home page with table status grid and combined report generation.

    Parameters:
    -----------
    config : dict
        Configuration dictionary
    available_tables : list
        List of available table names
    """
    st.header("📊 CLIF Validation Status Overview")
    st.markdown("### Table Validation Status")

    # Create status grid (3 columns x 6 rows for 18 tables)
    cols_per_row = 3
    rows = []
    for i in range(0, len(available_tables), cols_per_row):
        rows.append(available_tables[i:i + cols_per_row])

    # Display grid
    for row in rows:
        cols = st.columns(cols_per_row)
        for idx, table_name in enumerate(row):
            with cols[idx]:
                # Determine icon and caption based on cache status
                if is_table_cached(table_name):
                    completion = get_completion_status(table_name)
                    if completion['validation_complete']:
                        status = get_table_status(table_name)
                        status_icons = {
                            'complete': '🟢',
                            'partial': '🟡',
                            'incomplete': '🔴'
                        }
                        icon = status_icons.get(status, '📊')
                        cached = get_cached_analysis(table_name)
                        timestamp = format_cache_timestamp(cached['timestamp'])
                        caption_text = f"{status.upper()} • {timestamp}"
                    else:
                        icon = '📋'
                        caption_text = "Analyzed (no validation)"
                else:
                    icon = '⭕'
                    caption_text = "Not analyzed"

                # Clickable button for each table
                button_label = f"{icon} {TABLE_DISPLAY_NAMES[table_name]}"
                if st.button(
                    button_label,
                    key=f"home_table_{table_name}",
                    use_container_width=True,
                    type="secondary"
                ):
                    st.session_state.current_page = "📊 Table Analysis"
                    st.session_state.last_selected_table = table_name
                    st.rerun()

                st.caption(caption_text)

    st.divider()

    # Regenerate Reports Section (only show if combined report exists)
    output_dir = config.get('output_dir', 'output')
    combined_report_path = os.path.join(output_dir, 'final', 'reports', 'combined_validation_report.pdf')

    if os.path.exists(combined_report_path):
        st.markdown("### 🔄 Regenerate Reports")
        st.write("Regenerate all validation reports from existing validation results (accounts for any user feedback).")

        if st.button("📄 Regenerate All Reports", type="secondary", use_container_width=True):
            st.session_state.regenerate_reports = True
            st.rerun()

        st.divider()

    # Handle report regeneration
    if st.session_state.get('regenerate_reports', False):
        with st.spinner("Regenerating validation reports..."):
            try:
                from modules.cli import ValidationPDFGenerator
                from modules.reports.combined_report_generator import generate_combined_report

                reports_dir = os.path.join(output_dir, 'final', 'reports')
                results_dir = os.path.join(output_dir, 'final', 'results')
                os.makedirs(reports_dir, exist_ok=True)

                pdf_generator = ValidationPDFGenerator()
                regenerated_count = 0
                failed_tables = []

                # Regenerate individual table reports from validation JSON files
                for table_name in available_tables:
                    validation_json_path = os.path.join(results_dir, f'{table_name}_summary_validation.json')

                    if os.path.exists(validation_json_path):
                        try:
                            # Load validation results
                            with open(validation_json_path, 'r') as f:
                                validation_results = json.load(f)

                            # Check for user feedback
                            feedback = None
                            response_file = os.path.join(results_dir, f'{table_name}_validation_response.json')
                            if os.path.exists(response_file):
                                with open(response_file, 'r') as f:
                                    feedback = json.load(f)

                            # Generate PDF report
                            pdf_path = os.path.join(reports_dir, f"{table_name}_validation_report.pdf")

                            if pdf_generator.is_available():
                                pdf_generator.generate_validation_pdf(
                                    validation_results,
                                    table_name,
                                    pdf_path,
                                    config.get('site_name'),
                                    config.get('timezone', 'UTC'),
                                    feedback
                                )
                            else:
                                # Fall back to text report
                                txt_path = os.path.join(reports_dir, f"{table_name}_validation_report.txt")
                                pdf_generator.generate_text_report(
                                    validation_results,
                                    table_name,
                                    txt_path,
                                    config.get('site_name'),
                                    config.get('timezone', 'UTC'),
                                    feedback
                                )

                            regenerated_count += 1
                        except Exception as e:
                            failed_tables.append((table_name, str(e)))

                # Regenerate combined report
                try:
                    pdf_path = generate_combined_report(
                        output_dir,
                        available_tables,
                        config.get('site_name'),
                        config.get('timezone', 'UTC')
                    )

                    if pdf_path:
                        st.success(f"✅ Successfully regenerated {regenerated_count} individual reports")
                        st.success("✅ Successfully regenerated combined validation report")
                        st.success("✅ Successfully regenerated consolidated CSV")
                    else:
                        st.error("❌ Failed to regenerate combined report")
                except Exception as e:
                    st.error(f"❌ Error regenerating combined report: {e}")

                # Show any failures
                if failed_tables:
                    with st.expander(f"⚠️ Failed to regenerate {len(failed_tables)} table(s)", expanded=False):
                        for table_name, error in failed_tables:
                            st.error(f"**{TABLE_DISPLAY_NAMES.get(table_name, table_name)}**: {error}")

            except Exception as e:
                st.error(f"❌ Error during report regeneration: {e}")

        st.session_state.regenerate_reports = False

    # Analyze All Tables Section
    st.markdown("### Analyze All Tables")
    st.write("Run comprehensive analysis on all tables to generate validation reports, summary statistics, and aggregated outputs.")

    col1, col2 = st.columns([3, 1])
    with col1:
        # Run Validation checkbox (always checked, disabled)
        st.checkbox(
            "✓ Run Validation",
            value=True,
            disabled=True,
            key="run_validation_always",
            help="Validation is always performed - validates data against CLIF 2.1 schema using clifpy"
        )

        # Generate Summary Aggregates checkbox (optional)
        generate_aggregates = st.checkbox(
            "📊 Generate Summary Aggregates",
            value=False,
            key="generate_aggregates_check",
            help="Generate table-specific summary CSVs (demographics, hospitalizations, CRRT distributions, etc.)"
        )

        # Use 1k ICU Sample checkbox (RECOMMENDED - default checked)
        use_sample_bulk = st.checkbox(
            "⚡ Use 1k ICU Sample (RECOMMENDED)",
            value=True,
            help="Analyze using only 1k ICU hospitalizations (stratified by year). Much faster for large datasets. Sample will be created automatically from ADT table if needed.",
            key="bulk_sample_check"
        )

        # Time estimates based on sample setting
        if use_sample_bulk:
            st.success("✅ **Estimated time: 5-10 minutes** (with 1k sample)")
            st.caption("⚡ Sample mode: Only sample-eligible tables will use the 1k sample. Core tables (patient, hospitalization, ADT) will use full data.")
        else:
            st.warning("⚠️ **Estimated time: 30-60 minutes** (full dataset - may be longer for very large datasets)")
            st.caption("🐌 Full data mode: Analyzing complete dataset for all tables. Consider using 1k sample for faster results.")

        st.caption("📋 Always includes: Validation + Individual PDFs + Combined Report")
        if generate_aggregates:
            st.caption("📊 Will also generate: Summary Statistics (JSON) + Table-specific summary CSVs + CRRT distributions + Visualization data")
        else:
            st.caption("💡 Tip: Check 'Generate Summary Aggregates' to also create Summary Statistics JSONs and detailed CSV summaries")

    with col2:
        if st.button("🚀 Analyze All Tables", type="primary", width='stretch'):
            st.session_state.analyze_all_tables = True
            st.session_state.bulk_sample = use_sample_bulk
            st.session_state.generate_aggregates = generate_aggregates
            st.rerun()

    # Handle bulk analysis
    if st.session_state.get('analyze_all_tables', False):
        analyze_all_tables(
            config,
            available_tables,
            st.session_state.get('bulk_sample', False),
            st.session_state.get('generate_aggregates', False)
        )
        st.session_state.analyze_all_tables = False



def main():
    """Main application function."""
    # Initialize cache
    initialize_cache()

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

        # Store config in session state for cache manager
        st.session_state.config = config

        # Display site information
        st.success(f"✅ Site: {config.get('site_name', 'Unknown')}")
        st.info(f"📁 Data Path: {config.get('tables_path', 'Not specified')}")

        # Check if data directory exists
        data_path = config.get('tables_path', '')
        if not os.path.exists(data_path):
            st.warning(f"⚠️ Data directory not found: {data_path}")

        st.divider()

        # Navigation - Home button
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.current_page = "🏠 Home"
            st.rerun()

        # Instructions button
        if st.button("📖 Instructions", use_container_width=True):
            st.session_state.current_page = "📖 Instructions"
            st.rerun()

        # Table One button (only show if results are available)
        output_dir = config.get('output_dir', 'output')
        if check_tableone_results_available(output_dir):
            if st.button("📊 Table One Results", use_container_width=True):
                st.session_state.current_page = "📊 Table One Results"
                st.rerun()


        # All CLIF 2.1 tables are now implemented
        available_tables = ['patient', 'hospitalization', 'adt', 'code_status', 'crrt_therapy', 'ecmo_mcs',
                           'hospital_diagnosis', 'labs', 'medication_admin_continuous', 'medication_admin_intermittent',
                           'microbiology_culture', 'microbiology_nonculture', 'microbiology_susceptibility',
                           'patient_assessments', 'patient_procedures', 'position', 'respiratory_support', 'vitals']

        # Initialize navigation state if not exists (needed for sidebar logic)
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "🏠 Home"

        st.divider()

        # Table selection dropdown - always visible (except on Table One Results and MCIDE pages)
        # On Home page: show placeholder, on Table Analysis: show selected table
        if st.session_state.current_page == "🏠 Home":
            # Show dropdown with placeholder on Home page
            if 'last_selected_table' in st.session_state:
                # Get index of last selected table
                default_index = available_tables.index(st.session_state.last_selected_table)
            else:
                default_index = 0

            selected_table = st.selectbox(
                "Select Table to Analyze",
                options=available_tables,
                format_func=lambda x: TABLE_DISPLAY_NAMES[x],
                index=default_index,
                placeholder="Select a table...",
                label_visibility="visible"
            )

            # When user selects a table from dropdown on Home page, navigate to Table Analysis
            if 'last_selected_table' not in st.session_state:
                st.session_state.last_selected_table = selected_table

            if selected_table != st.session_state.last_selected_table:
                st.session_state.last_selected_table = selected_table
                st.session_state.current_page = "📊 Table Analysis"
                st.rerun()

        elif st.session_state.current_page == "📊 Table Analysis":
            # On Table Analysis page: show selected table
            selected_table = st.selectbox(
                "Select Table to Analyze",
                options=available_tables,
                format_func=lambda x: TABLE_DISPLAY_NAMES[x],
                index=available_tables.index(st.session_state.get('last_selected_table', available_tables[0]))
            )

            # When user selects a different table, update and stay on Table Analysis page
            if 'last_selected_table' not in st.session_state:
                st.session_state.last_selected_table = selected_table

            if selected_table != st.session_state.last_selected_table:
                st.session_state.last_selected_table = selected_table
                st.rerun()

            # Show re-analyze option if table is cached
            force_reanalyze = False
            if is_table_cached(selected_table):
                force_reanalyze = st.checkbox("🔄 Re-analyze table", value=False)

            # Sample option for eligible tables
            SAMPLE_ELIGIBLE_TABLES = [
                'labs', 'medication_admin_continuous', 'medication_admin_intermittent',
                'microbiology_nonculture', 'microbiology_susceptibility',
                'patient_assessments', 'patient_procedures', 'position',
                'respiratory_support', 'vitals'
            ]

            # Show sample option ONLY for eligible tables
            if selected_table in SAMPLE_ELIGIBLE_TABLES:
                from modules.utils.sampling import sample_exists

                # Check if sample file exists
                if sample_exists(config.get('output_dir', 'output')):
                    use_sample = st.checkbox(
                        "📊 Use 1k ICU Sample",
                        value=False,
                        help="Loads table with only 1k ICU hospitalizations (stratified proportional by year). Significantly faster for large tables."
                    )
                    st.session_state.use_sample = use_sample
                else:
                    st.info("ℹ️ Sample not yet generated. Analyze ADT table first to create sample.")
                    st.session_state.use_sample = False
            else:
                st.session_state.use_sample = False

            # Run analysis button - only on Table Analysis page
            if st.button("🚀 Run Analysis", type="primary", width='stretch'):
                st.session_state.run_analysis = True
                st.session_state.selected_table = selected_table
                st.session_state.run_validation = True
                st.session_state.run_outlier_handling = config.get('analysis_settings', {}).get('outlier_detection', False)
                st.session_state.calculate_sofa = False
                # Always force reanalyze when Run Analysis is clicked
                st.session_state.force_reanalyze = True
                st.rerun()
        else:
            # On Table One Results or MCIDE page: set default selected_table for page logic
            selected_table = st.session_state.get('last_selected_table', available_tables[0])

        st.divider()

        # Table selection with status indicators
        st.subheader("📊 Table Status")

        # Show cache status for available tables
        for table in available_tables:
            if is_table_cached(table):
                cached = get_cached_analysis(table)
                status_display = get_status_display(table)
                timestamp = format_cache_timestamp(cached['timestamp'])

                # Icon based on validation status if available
                completion = get_completion_status(table)
                if completion['validation_complete']:
                    status = get_table_status(table)
                    status_icons = {
                        'complete': '✅',
                        'partial': '⚠️',
                        'incomplete': '❌'
                    }
                    icon = status_icons.get(status, '📊')
                else:
                    icon = '📋'

                st.caption(f"{icon} {TABLE_DISPLAY_NAMES[table]} - {status_display} ({timestamp})")
            else:
                st.caption(f"⭕ {TABLE_DISPLAY_NAMES[table]} - Not analyzed")

        # Clear cache button
        st.divider()
        if st.button("🗑️ Clear All Cache", help="Clear all cached analyses"):
            clear_all_cache()
            st.success("Cache cleared!")
            st.rerun()

    # Header with logo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Check if logo exists in images folder
        logo_path = Path("images/clif_logo_v1_white.png")
        if not logo_path.exists():
            logo_path = Path("images/clif_logo_red_2.png")  # Fallback 1
        if not logo_path.exists():
            logo_path = Path("assets/clif_logo.png")  # Fallback 2

        if logo_path.exists():
            st.image(str(logo_path), width=450)

    # When run_analysis is clicked, switch to Table Analysis page
    if st.session_state.get('run_analysis', False):
        st.session_state.current_page = "📊 Table Analysis"

    st.divider()

    # Show appropriate page
    if st.session_state.current_page == "🏠 Home":
        show_home_page(config, available_tables)
        return
    elif st.session_state.current_page == "📖 Instructions":
        show_instructions()
        return
    elif st.session_state.current_page == "📊 Table One Results":
        show_tableone_results(config.get('output_dir', 'output'))
        return

    # Main content area (Table Analysis page)
    # Show analysis results if they exist (cached) or if user clicked Run Analysis
    if is_table_cached(selected_table):
        # Table has cached results - display them
        # If run_analysis was just clicked, use session state values and force_reanalyze flag
        if 'run_analysis' in st.session_state and st.session_state.run_analysis:
            analyze_table(
                st.session_state.get('selected_table', selected_table),
                config,
                st.session_state.get('run_validation', True),
                st.session_state.get('run_outlier_handling', config.get('analysis_settings', {}).get('outlier_detection', False)),
                st.session_state.get('force_reanalyze', False)
            )
        else:
            # Just viewing cached results
            analyze_table(
                selected_table,
                config,
                True,  # Always run validation
                config.get('analysis_settings', {}).get('outlier_detection', False),
                False  # Don't force reanalyze, use cache
            )
    elif 'run_analysis' in st.session_state and st.session_state.run_analysis:
        # No cache but user clicked Run Analysis - run fresh analysis
        analyze_table(
            st.session_state.get('selected_table', selected_table),
            config,
            st.session_state.get('run_validation', True),
            st.session_state.get('run_outlier_handling', config.get('analysis_settings', {}).get('outlier_detection', False)),
            st.session_state.get('force_reanalyze', False)
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


def _lazy_load_analyzer(table_name, config, cached_analyzer):
    """
    Lazy-load analyzer only when needed for specific features.

    Parameters:
    -----------
    table_name : str
        Name of the table
    config : dict
        Configuration dictionary
    cached_analyzer : object or None
        Cached analyzer if available

    Returns:
    --------
    object or None
        Analyzer instance
    """
    # If analyzer already exists in cache, return it
    if cached_analyzer is not None:
        return cached_analyzer

    # Otherwise create it
    analyzer_class = TABLE_ANALYZERS.get(table_name)
    if not analyzer_class:
        return None

    data_dir = config.get('tables_path', './data')
    filetype = config.get('filetype', 'parquet')
    timezone = config.get('timezone', 'UTC')
    output_dir = config.get('output_dir', 'output')

    try:
        analyzer = analyzer_class(data_dir, filetype, timezone, output_dir)
        # Update cache with the newly created analyzer
        cached = get_cached_analysis(table_name)
        if cached:
            cache_analysis(table_name, analyzer, cached['validation'], cached['summary'], cached.get('feedback'))
        return analyzer
    except Exception as e:
        st.warning(f"Could not load analyzer: {e}")
        return None


def analyze_table(table_name, config, run_validation, run_outlier_handling, force_reanalyze=False):
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
    force_reanalyze : bool
        Force re-analysis even if cached
    """
    st.header(f"📋 {TABLE_DISPLAY_NAMES[table_name]} Analysis")

    # Check cache first
    cached = get_cached_analysis(table_name) if not force_reanalyze else None

    if cached:
        st.info(f"📦 Using cached analysis from {format_cache_timestamp(cached['timestamp'])}")
        # Don't recreate analyzer when viewing cached results
        # The analyzer will be lazy-loaded only if needed for specific features
        analyzer = cached['analyzer']

        validation_results = cached['validation']
        summary_stats = cached['summary']
        existing_feedback = cached.get('feedback')
    else:
        # Initialize analyzer
        analyzer_class = TABLE_ANALYZERS.get(table_name)

        if not analyzer_class:
            st.error(f"❌ Analyzer not implemented for {table_name}")
            st.info("Currently only the Patient table analyzer is available. Other tables coming soon!")
            return

        # Extract parameters from config
        data_dir = config.get('tables_path', './data')
        # Support both 'filetype' and 'file_type' keys in config
        filetype = config.get('filetype') or config.get('file_type', 'parquet')
        timezone = config.get('timezone', 'UTC')
        output_dir = config.get('output_dir', 'output')

        try:
            # Check if sampling is enabled for eligible tables
            sample_filter = None
            SAMPLE_ELIGIBLE_TABLES = [
                'labs', 'medication_admin_continuous', 'medication_admin_intermittent',
                'microbiology_nonculture', 'microbiology_susceptibility',
                'patient_assessments', 'patient_procedures', 'position',
                'respiratory_support', 'vitals'
            ]

            if st.session_state.get('use_sample', False) and table_name in SAMPLE_ELIGIBLE_TABLES:
                from modules.utils.sampling import load_sample_list
                sample_filter = load_sample_list(output_dir)
                if sample_filter:
                    st.info(f"📊 Loading with 1k ICU sample ({len(sample_filter):,} hospitalizations)")

            with st.spinner(f"Loading {TABLE_DISPLAY_NAMES[table_name]} table..."):
                analyzer = analyzer_class(data_dir, filetype, timezone, output_dir)

                # Pass sample_filter to load_table for eligible tables
                if table_name in SAMPLE_ELIGIBLE_TABLES and sample_filter is not None:
                    analyzer.load_table(sample_filter=sample_filter)
                else:
                    analyzer.load_table()

            # Delete old validation response file if it exists (fresh analysis = no old feedback)
            results_dir = os.path.join(output_dir, 'final', 'results')
            os.makedirs(results_dir, exist_ok=True)
            response_file = os.path.join(results_dir, f'{table_name}_validation_response.json')
            if os.path.exists(response_file):
                try:
                    os.remove(response_file)
                    st.info("🔄 Cleared previous feedback - fresh analysis starting")
                except Exception as e:
                    st.warning(f"Could not remove old feedback file: {e}")

            # Run analysis
            hosp_years = st.session_state.get('hosp_years', None)
            with st.spinner("Running validation..."):
                validation_results = analyzer.validate(hosp_years=hosp_years) if run_validation else None

            # Inject per-column data profile stats (matching CLI runner)
            if validation_results and analyzer.table is not None and hasattr(analyzer.table, 'df') and analyzer.table.df is not None:
                from clifpy.utils.report_generator import compute_table_stats
                validation_results['total_rows'] = len(analyzer.table.df)
                validation_results['table_stats'] = compute_table_stats(
                    analyzer.table.df, analyzer.table.schema
                )

            # Extract cross-table cache and propagate hosp_years
            if validation_results and analyzer.table is not None:
                try:
                    cache = analyzer.extract_cross_table_cache()
                    if cache.get('hosp_years'):
                        st.session_state.hosp_years = cache['hosp_years']
                except Exception:
                    pass

            with st.spinner("Calculating summary statistics..."):
                summary_stats = analyzer.get_summary_statistics()

            # Automatically save validation and summary results to JSON files
            if validation_results:
                try:
                    analyzer.save_validation_results(validation_results)
                    st.success("✅ Validation results saved")
                except Exception as e:
                    st.warning(f"Could not save validation results: {e}")

                # Generate PDF report
                try:
                    reports_dir = os.path.join(output_dir, 'final', 'reports')
                    os.makedirs(reports_dir, exist_ok=True)

                    pdf_generator = ValidationPDFGenerator()
                    pdf_path = os.path.join(reports_dir, f"{table_name}_validation_report.pdf")

                    if pdf_generator.is_available():
                        pdf_generator.generate_validation_pdf(
                            validation_results,
                            table_name,
                            pdf_path,
                            config.get('site_name'),
                            config.get('timezone', 'UTC')
                        )
                        st.success(f"✅ Validation PDF report saved: {table_name}_validation_report.pdf")
                    else:
                        # Fall back to text report
                        txt_path = os.path.join(reports_dir, f"{table_name}_validation_report.txt")
                        pdf_generator.generate_text_report(
                            validation_results,
                            table_name,
                            txt_path,
                            config.get('site_name'),
                            config.get('timezone', 'UTC')
                        )
                        st.info("ℹ️ reportlab not available, generated text report instead")
                        st.success(f"✅ Validation text report saved: {table_name}_validation_report.txt")
                except Exception as e:
                    st.warning(f"Could not generate validation report: {e}")

            if summary_stats:
                try:
                    analyzer.save_summary_data(summary_stats, '_summary')
                    st.success("✅ Summary statistics saved")
                except Exception as e:
                    st.warning(f"Could not save summary statistics: {e}")

            # Save summary tables as CSV files
            try:
                results_dir = os.path.join(output_dir, 'final', 'results')
                os.makedirs(results_dir, exist_ok=True)

                # Save patient demographics summary
                if hasattr(analyzer, 'generate_patient_summary'):
                    patient_summary_df = analyzer.generate_patient_summary()
                    if not patient_summary_df.empty:
                        csv_filepath = os.path.join(results_dir, f"{table_name}_demographics_summary.csv")
                        patient_summary_df.to_csv(csv_filepath, index=False)
                        st.success(f"✅ Patient demographics summary CSV saved")

                # Save hospitalization summary
                if hasattr(analyzer, 'generate_hospitalization_summary'):
                    hosp_summary_df = analyzer.generate_hospitalization_summary()
                    if not hosp_summary_df.empty:
                        csv_filepath = os.path.join(results_dir, f"{table_name}_summary.csv")
                        hosp_summary_df.to_csv(csv_filepath, index=False)
                        st.success(f"✅ Hospitalization summary CSV saved")

                # Save ADT summary
                if hasattr(analyzer, 'generate_adt_summary'):
                    adt_summary_df = analyzer.generate_adt_summary()
                    if not adt_summary_df.empty:
                        csv_filepath = os.path.join(results_dir, f"{table_name}_summary.csv")
                        adt_summary_df.to_csv(csv_filepath, index=False)
                        st.success(f"✅ ADT summary CSV saved")

                # Save Hospital Diagnosis summary
                if hasattr(analyzer, 'generate_hospital_diagnosis_summary'):
                    hosp_diag_summary_df = analyzer.generate_hospital_diagnosis_summary()
                    if not hosp_diag_summary_df.empty:
                        csv_filepath = os.path.join(results_dir, f"{table_name}_summary.csv")
                        hosp_diag_summary_df.to_csv(csv_filepath, index=False)
                        st.success(f"✅ Hospital Diagnosis summary CSV saved")

                # Save CRRT numeric distributions
                if hasattr(analyzer, 'save_numeric_distributions'):
                    dist_filepath = analyzer.save_numeric_distributions()
                    if dist_filepath:
                        st.success(f"✅ CRRT numeric distributions saved")

                # Save CRRT visualization data (pre-computed with outlier handling)
                if hasattr(analyzer, 'save_visualization_data'):
                    viz_filepath = analyzer.save_visualization_data()
                    if viz_filepath:
                        st.success(f"✅ CRRT visualization data saved")
            except Exception as e:
                st.warning(f"Could not save summary CSV files: {e}")

            # Generate ICU sample after ADT analysis (only if sample doesn't exist)
            if table_name == 'adt' and analyzer.table is not None:
                from modules.utils.sampling import (
                    get_icu_hospitalizations_from_adt,
                    generate_stratified_sample,
                    save_sample_list,
                    sample_exists
                )

                # Only generate if sample doesn't already exist
                if not sample_exists(output_dir):
                    try:
                        with st.spinner("Generating 1k ICU sample for future analyses..."):
                            # Step 1: Get ICU hospitalizations from ADT
                            icu_hosp_ids = get_icu_hospitalizations_from_adt(analyzer.table.df)

                            if len(icu_hosp_ids) > 0:
                                # Step 2: Load hospitalization table to get years
                                hosp_analyzer = HospitalizationAnalyzer(data_dir, filetype, timezone, output_dir)
                                if hosp_analyzer.table is not None:
                                    # Step 3: Generate stratified sample
                                    sample_ids = generate_stratified_sample(
                                        hosp_analyzer.table.df,
                                        icu_hosp_ids,
                                        sample_size=1000
                                    )

                                    # Step 4: Save for future use
                                    save_sample_list(sample_ids, output_dir)
                                    st.success(f"✅ Generated 1k ICU sample (stratified by year) - {len(sample_ids):,} hospitalizations")
                                else:
                                    st.info("ℹ️ Could not load hospitalization table for sampling")
                            else:
                                st.warning("⚠️ No ICU hospitalizations found in ADT table")
                    except Exception as e:
                        st.warning(f"Could not generate sample: {e}")

            # No existing feedback since we just ran fresh analysis
            existing_feedback = None

            # Cache the results
            cache_analysis(table_name, analyzer, validation_results, summary_stats, existing_feedback)

            # Clear analysis flags
            st.session_state.force_reanalyze = False
            st.session_state.run_analysis = False  # Clear this so status updates and table switching works
            st.session_state.analysis_just_completed = True

        except Exception as e:
            st.error(f"❌ Error loading table: {str(e)}")
            st.info("Please check:")
            st.write("1. The data path is correct in your config file")
            st.write(f"2. The {table_name}.{filetype} file exists in the data directory")
            st.write("3. The file format matches the configured filetype")
            st.write("4. You have clifpy installed: `pip install clifpy`")
            return

    # Create tabs for validation, MCIDE, and summary
    tab1, tab2, tab3 = st.tabs(["🔍 Validation", "📋 MCIDE", "📊 Summary"])

    with tab1:
        display_validation_results(analyzer, validation_results, existing_feedback, table_name)

    with tab2:
        display_table_mcide(table_name, config.get('output_dir', 'output'))

    with tab3:
        display_summary_statistics(analyzer, summary_stats, table_name)

    # Clear analysis_just_completed flag without triggering rerun
    # The sidebar status will update on next user interaction
    if st.session_state.get('analysis_just_completed', False):
        st.session_state.analysis_just_completed = False


def analyze_all_tables(config, available_tables, use_sample=False, generate_aggregates=False):
    """
    Analyze all tables: validation, summary statistics, individual PDFs, and combined report.

    Parameters:
    -----------
    config : dict
        Configuration dictionary
    available_tables : list
        List of available table names
    use_sample : bool
        Whether to use 1k ICU sample for eligible tables
    generate_aggregates : bool
        Whether to generate table-specific summary CSV aggregates
    """
    # All available tables
    all_tables = [
        'patient', 'hospitalization', 'adt', 'code_status', 'crrt_therapy', 'ecmo_mcs',
        'hospital_diagnosis', 'labs', 'medication_admin_continuous', 'medication_admin_intermittent',
        'microbiology_culture', 'microbiology_nonculture', 'microbiology_susceptibility',
        'patient_assessments', 'patient_procedures', 'position', 'respiratory_support', 'vitals'
    ]

    # Sample-eligible tables (tables that can use hospitalization_id filter)
    # Note: code_status uses patient_id (not hospitalization_id), so it's excluded and uses full dataset
    SAMPLE_ELIGIBLE_TABLES = [
        'labs', 'medication_admin_continuous', 'medication_admin_intermittent',
        'microbiology_nonculture', 'microbiology_susceptibility', 'microbiology_culture',
        'vitals', 'patient_assessments', 'respiratory_support', 'position',
        'patient_procedures', 'adt', 'crrt_therapy', 'ecmo_mcs',
        'hospital_diagnosis'
    ]

    # Progress tracking
    st.markdown("### 🚀 Bulk Analysis Progress")
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_area = st.container()

    # Load sample filter if needed
    sample_filter = None
    if use_sample:
        from modules.utils.sampling import (
            get_icu_hospitalizations_from_adt,
            generate_stratified_sample,
            save_sample_list,
            load_sample_list,
            sample_exists
        )

        output_dir = config.get('output_dir', 'output')

        # Try to load existing sample
        if sample_exists(output_dir):
            sample_list = load_sample_list(output_dir)
            if sample_list:
                sample_filter = set(sample_list)
                st.info(f"✓ Using existing 1k ICU sample: {len(sample_filter)} hospitalizations")

        # If no sample exists, create it from ADT and hospitalization tables
        if not sample_filter:
            st.info("📊 Sample not found. Creating 1k ICU sample from ADT and hospitalization tables...")

            # Need to load ADT and hospitalization tables to create sample
            data_dir = config.get('tables_path', './data')
            filetype = config.get('filetype', 'parquet')
            timezone = config.get('timezone', 'UTC')

            try:
                # Load ADT table
                from modules.tables.adt_analysis import ADTAnalyzer
                adt_analyzer = ADTAnalyzer(data_dir, filetype, timezone, output_dir)

                if adt_analyzer.table is not None and hasattr(adt_analyzer.table, 'df'):
                    # Step 1: Get ICU hospitalizations from ADT
                    with st.spinner("Step 1/3: Identifying ICU hospitalizations from ADT table..."):
                        icu_hosp_ids = get_icu_hospitalizations_from_adt(adt_analyzer.table.df)
                        st.info(f"   Found {len(icu_hosp_ids):,} ICU hospitalizations")

                    # Step 2: Load hospitalization table
                    from modules.tables.hospitalization_analysis import HospitalizationAnalyzer
                    with st.spinner("Step 2/3: Loading hospitalization table..."):
                        hosp_analyzer = HospitalizationAnalyzer(data_dir, filetype, timezone, output_dir)

                    if hosp_analyzer.table is not None and hasattr(hosp_analyzer.table, 'df'):
                        # Step 3: Generate stratified sample
                        with st.spinner("Step 3/3: Generating stratified sample (proportional by admission year)..."):
                            sample_ids = generate_stratified_sample(
                                hosp_analyzer.table.df,
                                icu_hosp_ids,
                                sample_size=1000
                            )

                        # Step 4: Save for future use
                        if sample_ids:
                            save_sample_list(sample_ids, output_dir)
                            sample_filter = set(sample_ids)
                            st.success(f"✅ Generated 1k ICU sample (stratified by year) - {len(sample_ids):,} hospitalizations")
                        else:
                            st.warning("⚠️ Could not generate sample - no ICU hospitalizations found")
                    else:
                        st.warning("⚠️ Could not load hospitalization table for sampling. Proceeding without sample.")
                else:
                    st.warning("⚠️ Could not load ADT table for sampling. Proceeding without sample.")
            except Exception as e:
                st.error(f"❌ Error creating sample: {e}")
                st.warning("Proceeding without sample - will use full dataset for all tables.")

    # Results tracking
    results = {
        'success': [],
        'failed': [],
        'skipped': []
    }
    cross_table_caches = {}  # table_name -> lightweight cache dict for cross-table checks
    hosp_years = None  # cached hosp_years for temporal context

    # Analyze each table
    for idx, table_name in enumerate(all_tables):
        status_text.markdown(f"**Analyzing {TABLE_DISPLAY_NAMES.get(table_name, table_name)}...** ({idx + 1}/{len(all_tables)})")

        try:
            # Check if table should use sample
            use_sample_for_table = use_sample and table_name in SAMPLE_ELIGIBLE_TABLES

            # Get analyzer class
            analyzer_class = TABLE_ANALYZERS.get(table_name)
            if not analyzer_class:
                results['skipped'].append((table_name, 'No analyzer available'))
                continue

            # Initialize analyzer (table loads automatically in __init__)
            data_dir = config.get('tables_path', './data')
            filetype = config.get('filetype', 'parquet')
            timezone = config.get('timezone', 'UTC')
            output_dir = config.get('output_dir', 'output')

            # Pass sample_filter to __init__ to load table only once
            if use_sample_for_table and sample_filter:
                analyzer = analyzer_class(
                    data_dir=data_dir,
                    filetype=filetype,
                    timezone=timezone,
                    output_dir=output_dir,
                    sample_filter=sample_filter
                )
            else:
                analyzer = analyzer_class(
                    data_dir=data_dir,
                    filetype=filetype,
                    timezone=timezone,
                    output_dir=output_dir
                )

            # Check if table loaded (already loaded in __init__)
            if analyzer.table is None or not hasattr(analyzer.table, 'df') or analyzer.table.df is None:
                results['skipped'].append((table_name, 'Table not found or failed to load'))
                continue

            # Always run validation
            validation_results = analyzer.validate(hosp_years=hosp_years)

            # Inject per-column data profile stats (matching CLI runner)
            if validation_results and analyzer.table is not None and hasattr(analyzer.table, 'df') and analyzer.table.df is not None:
                from clifpy.utils.report_generator import compute_table_stats
                validation_results['total_rows'] = len(analyzer.table.df)
                validation_results['table_stats'] = compute_table_stats(
                    analyzer.table.df, analyzer.table.schema
                )

            # Extract cross-table cache and propagate hosp_years
            if validation_results and analyzer.table is not None:
                try:
                    cache = analyzer.extract_cross_table_cache()
                    cross_table_caches[table_name] = cache
                    if cache.get('hosp_years'):
                        hosp_years = cache['hosp_years']
                except Exception:
                    pass

            # Save validation results to disk for persistence
            if validation_results:
                try:
                    analyzer.save_validation_results(validation_results)
                except Exception as e:
                    # Log error but continue with other tables
                    if config.get('verbose', False):
                        st.warning(f"Could not save validation for {table_name}: {e}")

            # Generate summary statistics only if requested (requires validation)
            summary_stats = None
            if generate_aggregates:
                summary_stats = analyzer.get_summary_statistics()

                # Save summary stats to disk for persistence
                if summary_stats:
                    try:
                        analyzer.save_summary_data(summary_stats, '_summary')
                    except Exception as e:
                        # Log error but continue with other tables
                        if config.get('verbose', False):
                            st.warning(f"Could not save summary for {table_name}: {e}")

            # Generate individual PDF report for this table
            if validation_results:
                try:
                    from modules.cli import ValidationPDFGenerator

                    reports_dir = os.path.join(config.get('output_dir', 'output'), 'final', 'reports')
                    os.makedirs(reports_dir, exist_ok=True)

                    pdf_generator = ValidationPDFGenerator()
                    pdf_path = os.path.join(reports_dir, f"{table_name}_validation_report.pdf")

                    if pdf_generator.is_available():
                        pdf_generator.generate_validation_pdf(
                            validation_results,
                            table_name,
                            pdf_path,
                            config.get('site_name'),
                            config.get('timezone', 'UTC')
                        )
                        if config.get('verbose', False):
                            st.success(f"✅ Generated PDF report for {table_name}")
                    else:
                        # Fall back to text report
                        txt_path = os.path.join(reports_dir, f"{table_name}_validation_report.txt")
                        pdf_generator.generate_text_report(
                            validation_results,
                            table_name,
                            txt_path,
                            config.get('site_name'),
                            config.get('timezone', 'UTC')
                        )
                        if config.get('verbose', False):
                            st.info(f"ℹ️ Generated text report for {table_name} (PDF not available)")
                except Exception as e:
                    # Always show PDF generation errors so user knows why reports aren't created
                    st.warning(f"⚠️ Could not generate report for {table_name}: {e}")

            # Generate table-specific summary CSV aggregates if requested
            if generate_aggregates:
                try:
                    results_dir = os.path.join(config.get('output_dir', 'output'), 'final', 'results')
                    os.makedirs(results_dir, exist_ok=True)

                    # Save patient demographics summary
                    if hasattr(analyzer, 'generate_patient_summary'):
                        patient_summary_df = analyzer.generate_patient_summary()
                        if not patient_summary_df.empty:
                            csv_filepath = os.path.join(results_dir, f"{table_name}_demographics_summary.csv")
                            patient_summary_df.to_csv(csv_filepath, index=False)

                    # Save hospitalization summary
                    if hasattr(analyzer, 'generate_hospitalization_summary'):
                        hosp_summary_df = analyzer.generate_hospitalization_summary()
                        if not hosp_summary_df.empty:
                            csv_filepath = os.path.join(results_dir, f"{table_name}_summary.csv")
                            hosp_summary_df.to_csv(csv_filepath, index=False)

                    # Save ADT summary
                    if hasattr(analyzer, 'generate_adt_summary'):
                        adt_summary_df = analyzer.generate_adt_summary()
                        if not adt_summary_df.empty:
                            csv_filepath = os.path.join(results_dir, f"{table_name}_summary.csv")
                            adt_summary_df.to_csv(csv_filepath, index=False)

                    # Save Hospital Diagnosis summary
                    if hasattr(analyzer, 'generate_hospital_diagnosis_summary'):
                        hosp_diag_summary_df = analyzer.generate_hospital_diagnosis_summary()
                        if not hosp_diag_summary_df.empty:
                            csv_filepath = os.path.join(results_dir, f"{table_name}_summary.csv")
                            hosp_diag_summary_df.to_csv(csv_filepath, index=False)

                    # Save Vitals summary
                    if hasattr(analyzer, 'generate_vitals_summary'):
                        vitals_summary_df = analyzer.generate_vitals_summary()
                        if not vitals_summary_df.empty:
                            csv_filepath = os.path.join(results_dir, f"{table_name}_summary.csv")
                            vitals_summary_df.to_csv(csv_filepath, index=False)

                    # Save Labs summary
                    if hasattr(analyzer, 'generate_labs_summary'):
                        labs_summary_df = analyzer.generate_labs_summary()
                        if not labs_summary_df.empty:
                            csv_filepath = os.path.join(results_dir, f"{table_name}_summary.csv")
                            labs_summary_df.to_csv(csv_filepath, index=False)

                    # Save Respiratory Support summary
                    if hasattr(analyzer, 'generate_respiratory_summary'):
                        resp_summary_df = analyzer.generate_respiratory_summary()
                        if not resp_summary_df.empty:
                            csv_filepath = os.path.join(results_dir, f"{table_name}_summary.csv")
                            resp_summary_df.to_csv(csv_filepath, index=False)

                    # Save Position summary
                    if hasattr(analyzer, 'generate_position_summary'):
                        position_summary_df = analyzer.generate_position_summary()
                        if not position_summary_df.empty:
                            csv_filepath = os.path.join(results_dir, f"{table_name}_summary.csv")
                            position_summary_df.to_csv(csv_filepath, index=False)

                    # Save CRRT numeric distributions
                    if hasattr(analyzer, 'save_numeric_distributions'):
                        analyzer.save_numeric_distributions()

                    # Save CRRT visualization data
                    if hasattr(analyzer, 'save_visualization_data'):
                        analyzer.save_visualization_data()

                except Exception as e:
                    # Log error but continue with other tables
                    if config.get('verbose', False):
                        st.warning(f"Could not save summary CSVs for {table_name}: {e}")

            # Cache results in session state
            cache_analysis(table_name, analyzer, validation_results, summary_stats, None)

            results['success'].append(table_name)

        except Exception as e:
            error_msg = str(e)
            results['failed'].append((table_name, error_msg))
            if config.get('verbose', False):
                st.error(f"Error analyzing {table_name}: {error_msg}")

        # Update progress
        progress_bar.progress((idx + 1) / len(all_tables))

    # Run cache-based cross-table checks (relational + plausibility) matching CLI runner
    if len(cross_table_caches) > 1:
        try:
            from clifpy.utils.validator import (
                run_relational_integrity_checks_from_cache,
                run_cross_table_plausibility_checks_from_cache,
            )
            import json as _json

            output_dir = config.get('output_dir', 'output')
            pdf_generator = ValidationPDFGenerator()
            affected_tables = set()

            # --- Relational integrity (cache-based) ---
            rel_results = run_relational_integrity_checks_from_cache(cross_table_caches)
            for tname, rel_checks in rel_results.items():
                if not rel_checks:
                    continue
                serialized_rel = {k: v.to_dict() for k, v in rel_checks.items()}
                affected_tables.add(tname)

                # Update in-memory session state
                if tname in st.session_state.get('analyzed_tables', {}):
                    vr = st.session_state.analyzed_tables[tname].get('validation')
                    if vr:
                        vr.setdefault('completeness', {}).update(serialized_rel)

                # Update saved JSON file (merge into completeness, not relational)
                json_path = os.path.join(output_dir, 'final', 'clifpy', f'{tname}_dqa.json')
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        saved = _json.load(f)
                    saved.setdefault('completeness', {}).update(serialized_rel)
                    with open(json_path, 'w') as f:
                        _json.dump(saved, f, indent=2, default=str)

            # --- Cross-table plausibility (cache-based) ---
            plaus_results = run_cross_table_plausibility_checks_from_cache(cross_table_caches)
            for tname, plaus_checks in plaus_results.items():
                if not plaus_checks:
                    continue
                serialized_plaus = {k: v.to_dict() for k, v in plaus_checks.items()}
                affected_tables.add(tname)

                # Update in-memory session state
                if tname in st.session_state.get('analyzed_tables', {}):
                    vr = st.session_state.analyzed_tables[tname].get('validation')
                    if vr:
                        vr.setdefault('plausibility', {}).update(serialized_plaus)

                # Update saved JSON file (merge into plausibility)
                json_path = os.path.join(output_dir, 'final', 'clifpy', f'{tname}_dqa.json')
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        saved = _json.load(f)
                    saved.setdefault('plausibility', {}).update(serialized_plaus)
                    with open(json_path, 'w') as f:
                        _json.dump(saved, f, indent=2, default=str)

            # Regenerate PDFs for affected tables
            if affected_tables:
                reports_dir = os.path.join(output_dir, 'final', 'reports')
                for tname in affected_tables:
                    cached = st.session_state.get('analyzed_tables', {}).get(tname, {})
                    vr = cached.get('validation')
                    if vr and pdf_generator.is_available():
                        try:
                            pdf_path = os.path.join(reports_dir, f"{tname}_validation_report.pdf")
                            pdf_generator.generate_validation_pdf(
                                vr, tname, pdf_path,
                                config.get('site_name'),
                                config.get('timezone', 'UTC')
                            )
                        except Exception:
                            pass

            cross_table_caches.clear()
        except Exception:
            pass  # Cross-table checks are best-effort

    # Clear status and progress
    status_text.empty()
    progress_bar.empty()

    # Display results summary
    with results_area:
        st.markdown("### 📊 Analysis Complete")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "✅ Successful",
                len(results['success']),
                delta=f"{len(results['success'])}/{len(all_tables)} tables"
            )

        with col2:
            st.metric(
                "❌ Failed",
                len(results['failed']),
                delta=f"{len(results['failed'])}/{len(all_tables)} tables" if len(results['failed']) > 0 else None,
                delta_color="inverse"
            )

        with col3:
            st.metric(
                "⏭️ Skipped",
                len(results['skipped']),
                delta=f"{len(results['skipped'])}/{len(all_tables)} tables" if len(results['skipped']) > 0 else None,
                delta_color="off"
            )

        # Show successful tables
        if results['success']:
            with st.expander(f"✅ Successful Tables ({len(results['success'])})", expanded=True):
                success_names = [TABLE_DISPLAY_NAMES.get(t, t) for t in results['success']]
                st.write(", ".join(success_names))

        # Show failed tables
        if results['failed']:
            with st.expander(f"❌ Failed Tables ({len(results['failed'])})", expanded=True):
                for table_name, error in results['failed']:
                    st.error(f"**{TABLE_DISPLAY_NAMES.get(table_name, table_name)}**: {error}")

        # Show skipped tables
        if results['skipped']:
            with st.expander(f"⏭️ Skipped Tables ({len(results['skipped'])})", expanded=False):
                for table_name, reason in results['skipped']:
                    st.info(f"**{TABLE_DISPLAY_NAMES.get(table_name, table_name)}**: {reason}")

        # Generate combined report if we have successful validations
        if results['success']:
            st.divider()
            st.markdown("### 📄 Combined Validation Report")

            with st.spinner("Generating combined validation report..."):
                try:
                    from modules.reports.combined_report_generator import generate_combined_report

                    # Generate combined report
                    pdf_path = generate_combined_report(
                        config.get('output_dir', 'output'),
                        available_tables,
                        config.get('site_name'),
                        config.get('timezone', 'UTC'),
                        used_sampling=use_sample
                    )

                    if pdf_path:
                        st.success("✅ Combined validation report generated!")
                        st.success("✅ Consolidated CSV summary generated!")

                        # Provide download buttons for both PDF and CSV
                        col1, col2 = st.columns(2)

                        with col1:
                            # PDF download button
                            with open(pdf_path, 'rb') as f:
                                pdf_data = f.read()
                            st.download_button(
                                label="📥 Download PDF Report",
                                data=pdf_data,
                                file_name="combined_validation_report.pdf",
                                mime="application/pdf",
                                type="primary",
                                use_container_width=True
                            )

                        with col2:
                            # CSV download button
                            csv_path = os.path.join(config.get('output_dir', 'output'), 'final', 'results', 'consolidated_validation.csv')
                            if os.path.exists(csv_path):
                                with open(csv_path, 'r') as f:
                                    csv_data = f.read()
                                st.download_button(
                                    label="📥 Download CSV Summary",
                                    data=csv_data,
                                    file_name="consolidated_validation.csv",
                                    mime="text/csv",
                                    type="primary",
                                    use_container_width=True
                                )
                    else:
                        st.error("❌ Failed to generate combined report")

                except Exception as e:
                    st.error(f"❌ Error generating combined report: {e}")
                    if config.get('verbose', False):
                        import traceback
                        st.code(traceback.format_exc())

        # Generate MCIDE statistics after successful validation
        if results['success']:
            st.divider()
            st.markdown("### 📊 MCIDE Statistics Collection")

            with st.spinner("Collecting MCIDE statistics for validated tables..."):
                try:
                    # Import the MCIDE collector
                    import sys
                    from pathlib import Path

                    # Add code directory to path if needed
                    code_dir = Path(__file__).parent / 'code'
                    if str(code_dir) not in sys.path:
                        sys.path.insert(0, str(code_dir))

                    # Import and run the MCIDE collector
                    from modules.mcide import MCIDEStatsCollector

                    # Create collector instance
                    collector = MCIDEStatsCollector(config)

                    # Track collection progress
                    mcide_progress = st.progress(0)
                    mcide_status = st.empty()

                    # Define collection steps
                    collection_steps = [
                        ('patient', collector.collect_patient),
                        ('hospitalization', collector.collect_hospitalization),
                        ('adt', collector.collect_adt),
                        ('labs', collector.collect_labs_stats),
                        ('vitals', collector.collect_vitals_stats),
                        ('medication_admin_continuous', lambda: collector.collect_medication_stats('continuous')),
                        ('medication_admin_intermittent', lambda: collector.collect_medication_stats('intermittent')),
                        ('respiratory_support', collector.collect_respiratory_support),
                        ('microbiology_culture', collector.collect_microbiology_culture),
                        ('microbiology_nonculture', collector.collect_microbiology_nonculture),
                        ('microbiology_susceptibility', collector.collect_microbiology_susceptibility),
                        ('patient_assessments', collector.collect_patient_assessments),
                        ('patient_procedures', collector.collect_patient_procedures),
                        ('position', collector.collect_position),
                        ('crrt_therapy', collector.collect_crrt_stats),
                        ('ecmo_mcs', collector.collect_ecmo_stats),
                        ('hospital_diagnosis', collector.collect_hospital_diagnosis),
                        ('code_status', collector.collect_code_status)
                    ]

                    # Only collect for successfully validated tables
                    successful_collections = 0
                    total_steps = len([s for s in collection_steps if s[0] in results['success']])
                    current_step = 0

                    for table_name, collect_func in collection_steps:
                        if table_name in results['success']:
                            current_step += 1
                            mcide_status.text(f"Collecting MCIDE for {TABLE_DISPLAY_NAMES.get(table_name, table_name)}... ({current_step}/{total_steps})")

                            try:
                                collect_func()
                                successful_collections += 1
                            except Exception as e:
                                if config.get('verbose', False):
                                    st.warning(f"Could not collect MCIDE for {table_name}: {e}")

                            mcide_progress.progress(current_step / total_steps)

                    # Clear progress indicators
                    mcide_progress.empty()
                    mcide_status.empty()

                    if successful_collections > 0:
                        st.success(f"✅ MCIDE statistics collected for {successful_collections} tables")
                        st.info("Navigate to individual tables and click the 'MCIDE' tab to view the collected data")
                    else:
                        st.warning("⚠️ No MCIDE statistics were collected")

                except Exception as e:
                    st.error(f"❌ Error collecting MCIDE statistics: {e}")
                    if config.get('verbose', False):
                        import traceback
                        st.code(traceback.format_exc())

        # Next steps
        st.divider()
        st.markdown("### 🎯 Next Steps")
        st.write("✓ Navigate to individual tables to view detailed results")
        st.write("✓ Download the combined report above")
        st.write("✓ Check the MCIDE tab on individual tables for collected statistics")
        st.write("✓ Review and provide feedback on validation errors")


def _get_quality_check_definition(check_name: str) -> str:
    """Get human-readable definition for quality check types."""
    definitions = {
        'duplicate_patient_ids': 'Records where the patient_id appears more than once in the table',
        'duplicate_hospitalization_ids': 'Records where the hospitalization_id appears more than once in the table',
        'invalid_discharge_dates': 'Records where discharge_dttm is earlier than admission_dttm',
        'negative_ages': 'Records where age_at_admission is less than 0',
        'invalid_sex_categories': 'Records with sex_category values not in the standard set (Male, Female, Other, Unknown)',
        'future_death_dates': 'Records where death_dttm is in the future (after current date/time)',
        'invalid_location_dates': 'Records where out_dttm is earlier than in_dttm',
        'missing_location_category': 'Records with missing location_category values',
        'duplicate_adt_events': 'Records with duplicate ADT events (same hospitalization_id, in_dttm, and location_category)',
        'duplicate_patient_datetime': 'Records where the same patient has multiple code status entries at the same datetime',
        'future_code_status_dates': 'Records where start_dttm is in the future (after current date/time)',
        'patients_with_multiple_changes': 'Informational: Patients who have multiple code status changes over time',
        'invalid_code_status_categories': 'Records with code_status_category values not in the standard set (DNR, DNAR, UDNR, DNR/DNI, DNAR/DNI, AND, Full, Presume Full, Other)',
        'future_recorded_dates': 'Records where recorded_dttm is in the future (after current date/time)',
        'invalid_crrt_mode_categories': 'Records with crrt_mode_category values not in the standard set (scuf, cvvh, cvvhd, cvvhdf, avvh)',
        'negative_blood_flow_rate': 'Records where blood_flow_rate is negative (should be >= 0)',
        'negative_pre_filter_replacement_fluid_rate': 'Records where pre_filter_replacement_fluid_rate is negative (should be >= 0)',
        'negative_post_filter_replacement_fluid_rate': 'Records where post_filter_replacement_fluid_rate is negative (should be >= 0)',
        'negative_dialysate_flow_rate': 'Records where dialysate_flow_rate is negative (should be >= 0)',
        'negative_ultrafiltration_out': 'Records where ultrafiltration_out is negative (should be >= 0)',
        'invalid_device_categories': 'Records with device_category values not in the standard set (Impella, Centrimag, TandemHeart, HeartMate, ECMO, Other)',
        'invalid_mcs_groups': 'Records with mcs_group values not in the standard set of 20 device types',
        'negative_device_rate': 'Records where device_rate is negative (should be >= 0)',
        'negative_sweep': 'Records where sweep (gas flow rate) is negative (should be >= 0)',
        'negative_flow': 'Records where flow (blood flow) is negative (should be >= 0)',
        'invalid_fdO2': 'Records where fdO2 (fraction of delivered oxygen) is outside valid range (0.21-1.0)'
    }
    return definitions.get(check_name, 'No definition available')


def _details_to_kv(details: dict) -> list:
    """Extract key-value pairs from issue details dict, flattening simple values."""
    pairs = []
    for key, value in details.items():
        if isinstance(value, list):
            if not value:
                continue
            if isinstance(value[0], dict):
                # Skip tabular data — handled separately
                continue
            items = ", ".join(str(v) for v in value[:15])
            if len(value) > 15:
                items += f" ... ({len(value)} total)"
            pairs.append((key, items))
        elif isinstance(value, dict):
            items = ", ".join(f"{k}: {v}" for k, v in list(value.items())[:10])
            pairs.append((key, items))
        else:
            pairs.append((key, value))
    return pairs


def _render_issue_tables(details: dict):
    """Render any tabular (list-of-dicts) data from issue details via st.dataframe."""
    import pandas as pd
    for key, value in details.items():
        if isinstance(value, list) and value and isinstance(value[0], dict):
            st.caption(f"**{key}:**")
            st.dataframe(pd.DataFrame(value[:20]), use_container_width=True, hide_index=True)


def display_validation_results(analyzer, validation_results, existing_feedback, table_name):
    """
    Display DQA validation results tab using Plotly charts and native Streamlit widgets.

    Parameters:
    -----------
    analyzer : BaseTableAnalyzer
        The table analyzer instance
    validation_results : dict
        DQA results from run_full_dqa (keys: conformance, completeness,
        relational, plausibility)
    existing_feedback : dict or None
        Existing feedback structure if available (unused, kept for API compat)
    table_name : str
        Name of the table
    """
    if not validation_results:
        st.info("Validation not run. Check the 'Run Validation' box in the sidebar to enable.")
        return

    category_scores, all_issues = _collect_dqa_issues(validation_results)
    total_passed = sum(p for p, _ in category_scores.values())
    total_checks = sum(t for _, t in category_scores.values())
    error_count = sum(1 for i in all_issues if i['severity'] == 'error')
    warning_count = sum(1 for i in all_issues if i['severity'] == 'warning')
    overall_pct = round(total_passed / total_checks * 100, 1) if total_checks else 100

    # ── DQA Score hero ──
    st.markdown(
        f'<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:4px;">'
        f'<span style="font-size:2.2rem;font-weight:700;">{overall_pct}%</span>'
        f'<span style="font-size:0.85rem;color:#888;">{total_passed}/{total_checks} checks passed</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.progress(overall_pct / 100)

    # ── Metric cards: Errors, Warnings, per-category ──
    active_cats = [c for c in DQA_CATEGORIES if c in category_scores]
    card_cols = st.columns(2 + len(active_cats))

    with card_cols[0]:
        with st.container(border=True):
            st.metric("Errors", error_count, delta_color="inverse" if error_count > 0 else "off")
    with card_cols[1]:
        with st.container(border=True):
            st.metric("Warnings", warning_count, delta_color="inverse" if warning_count > 0 else "off")

    for idx, category in enumerate(active_cats):
        passed, total = category_scores[category]
        pct = round(passed / total * 100, 1) if total else 100
        # status = "Passing" if pct >= 85 else ("At Risk" if pct >= 60 else "Failing")
        delta_dir = "normal" if pct >= 85 else ("off" if pct >= 60 else "inverse")
        with card_cols[2 + idx]:
            with st.container(border=True):
                st.metric(category.title(), f"{passed}/{total}", delta_color=delta_dir)

    # ── Issues: unified list with inline review ──
    actionable = [i for i in all_issues if i['severity'] == 'error']

    # Load feedback for inline review controls
    output_dir = st.session_state.get('config', {}).get('output_dir', 'output')
    feedback = load_feedback(output_dir, table_name)
    if feedback is None and actionable:
        feedback = create_feedback_structure(validation_results, table_name)
    if actionable:
        # Style the review container via its key class
        st.markdown(
            '<style>'
            'div.st-key-review_resolve > div[data-testid="stVerticalBlockBorderWrapper"] {'
            '  border-color: rgba(255,152,0,0.4);'
            '}'
            'div.st-key-review_resolve > div[data-testid="stVerticalBlockBorderWrapper"] > div:first-child {'
            '  background: linear-gradient(180deg, rgba(230,81,0,0.12) 0%, rgba(191,54,12,0.06) 100%);'
            '}'
            '</style>',
            unsafe_allow_html=True,
        )
        with st.container(border=True, key="review_resolve"):
            st.markdown(
                f'<strong style="font-size:1.25rem;">'
                f'\U0001f50d Review & Resolve ({len(actionable)} error(s))'
                f'</strong>'
                f'<br><span style="font-size:0.85rem;opacity:0.5;">'
                f'Click the Feedback column to accept or reject each issue'
                f'</span>',
                unsafe_allow_html=True,
            )

            # Build issue index: eid + ensure feedback entries exist
            issue_eids = []  # parallel list: (eid, issue_dict) per row
            for issue in actionable:
                eid = create_error_id(issue)
                issue_eids.append((eid, issue))
                if feedback:
                    if eid not in feedback['user_decisions']:
                        feedback['user_decisions'][eid] = {
                            'error_type': issue.get('check_type', 'Unknown'),
                            'raw_type': issue.get('check_type', ''),
                            'category': issue.get('category', 'other'),
                            'severity': issue.get('severity', 'error'),
                            'description': issue.get('message', ''),
                            'decision': 'pending',
                            'reason': '',
                            'timestamp': None,
                        }
            # Recompute counts after ensuring all entries exist
            if feedback:
                feedback['total_errors'] = len(feedback['user_decisions'])
                feedback['accepted_count'] = sum(
                    1 for d in feedback['user_decisions'].values() if d['decision'] == 'accepted'
                )
                feedback['rejected_count'] = sum(
                    1 for d in feedback['user_decisions'].values() if d['decision'] == 'rejected'
                )
                feedback['pending_count'] = sum(
                    1 for d in feedback['user_decisions'].values() if d['decision'] == 'pending'
                )

            # Reserve placeholder for summary metrics (filled after processing edits)
            metrics_placeholder = st.empty()

            # Build display DataFrame with inline Feedback dropdown
            category_dots = {
                'conformance': '\U0001f534',
                'completeness': '\U0001f7e0',
                'plausibility': '\U0001f7e3',
            }
            decision_to_emoji = {
                'pending': '\u23f3 Pending',
                'accepted': '\u2705 Accepted',
                'rejected': '\u274c Rejected',
            }
            emoji_to_decision = {v: k for k, v in decision_to_emoji.items()}
            feedback_options = list(decision_to_emoji.values())

            rows = []
            for eid, issue in issue_eids:
                cat = issue['category'].lower()
                if feedback:
                    decision = feedback['user_decisions'].get(eid, {}).get('decision', 'pending')
                else:
                    decision = 'pending'
                rows.append({
                    'Feedback': decision_to_emoji.get(decision, decision_to_emoji['pending']),
                    '\u25cf': category_dots.get(cat, '\u26aa'),
                    'Check': issue.get('rule_description', issue['check_type']),
                    'Column': issue.get('column_field', 'N/A'),
                    'Message': issue['message'],
                })
            issues_df = pd.DataFrame(rows)

            # Category legend
            legend_parts = [f"{dot} {cat.title()}" for cat, dot in category_dots.items()]
            st.markdown("&emsp;".join(legend_parts))

            # Editable table with Feedback dropdown
            edited_df = st.data_editor(
                issues_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Feedback': st.column_config.SelectboxColumn(
                        'Feedback \u270e',
                        help='Click a cell to change the decision',
                        options=feedback_options,
                        required=True,
                    ),
                    '\u25cf': st.column_config.TextColumn('\u25cf', width='small'),
                },
                disabled=[c for c in issues_df.columns if c != 'Feedback'],
                key=f"issues_table_{table_name}",
            )

            # Process any decision changes from the edited table
            if feedback:
                for i, (eid, issue) in enumerate(issue_eids):
                    raw = edited_df.iloc[i]['Feedback']
                    new_decision = emoji_to_decision.get(raw, 'pending')
                    current = feedback['user_decisions'].get(eid, {}).get('decision', 'pending')
                    if new_decision and new_decision != current:
                        feedback = update_user_decision(feedback, eid, new_decision, '')

            # Rejection reason inputs for any rejected issues
            if feedback:
                rejected = [
                    (i, eid, issue)
                    for i, (eid, issue) in enumerate(issue_eids)
                    if emoji_to_decision.get(edited_df.iloc[i]['Feedback'], 'pending') == 'rejected'
                ]
                if rejected:
                    st.markdown("---")
                    st.markdown("**Rejection reasons**")
                    for _, eid, issue in rejected:
                        decision_info = feedback['user_decisions'].get(eid, {})
                        check = issue.get('rule_description', issue['check_type'])
                        col = issue.get('column_field', 'N/A')
                        st.markdown(
                            f"\u274c **{check}** on `{col}`  \n"
                            f"_{issue['message']}_"
                        )
                        reason = st.text_input(
                            "Reason for rejection",
                            value=decision_info.get('reason', ''),
                            placeholder="e.g. Known data limitation, will fix in next ETL run",
                            key=f"reason_{table_name}_{eid}",
                            label_visibility="collapsed",
                        )
                        if reason != decision_info.get('reason', ''):
                            feedback = update_user_decision(feedback, eid, 'rejected', reason)

            # Fill summary metrics now that decisions are up to date
            if feedback:
                with metrics_placeholder.container():
                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("Accepted", feedback['accepted_count'])
                    rc2.metric("Rejected", feedback['rejected_count'])
                    rc3.metric("Pending", feedback['pending_count'])

            # Save button
            if feedback:
                if st.button("Save Feedback", key=f"save_feedback_{table_name}"):
                    filepath = save_feedback(feedback, output_dir, table_name)
                    update_feedback_in_cache(table_name, feedback)
                    st.success(f"Feedback saved to {os.path.basename(filepath)}")

    else:
        st.success("No validation issues found!")

    # ── Warnings section (read-only) ──
    warnings_list = [i for i in all_issues if i['severity'] == 'warning']
    if warnings_list:
        with st.container(border=True):
            st.markdown(
                f'<strong style="font-size:1.15rem;">'
                f'\u26a0\ufe0f Warnings ({len(warnings_list)})'
                f'</strong>'
                f'<br><span style="font-size:0.85rem;opacity:0.5;">'
                f'Informational — no action required'
                f'</span>',
                unsafe_allow_html=True,
            )
            warn_rows = []
            for issue in warnings_list:
                warn_rows.append({
                    'Check': issue.get('rule_description', issue['check_type']),
                    'Column': issue.get('column_field', 'N/A'),
                    'Message': issue['message'],
                })
            st.dataframe(
                pd.DataFrame(warn_rows),
                use_container_width=True,
                hide_index=True,
            )


def _show_year_distribution(
    df: pd.DataFrame, 
    datetime_col: str, 
    label: str,
    count_by: str = 'hospitalization_id'
):
    """
    Show year distribution histogram using DuckDB for efficient computation.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe to analyze
    datetime_col : str
        Name of the datetime column to extract years from
    label : str
        Label for the y-axis (e.g., 'Hospitalizations', 'Patients', 'ADT Events')
    count_by : str, optional
        Column to use for unique counting. Options:
        - 'hospitalization_id': Count unique hospitalizations per year (default)
        - 'patient_id': Count unique patients per year
        - 'rows': Count total rows per year
    """
    if df is None or df.empty or datetime_col not in df.columns:
        st.warning("No data available for year distribution")
        return

    try:
        import duckdb

        # Determine aggregation method based on count_by parameter
        if count_by in ['hospitalization_id', 'patient_id'] and count_by in df.columns:
            # Count unique hospitalizations or patients per year
            result = duckdb.query(f"""
                SELECT
                    YEAR({datetime_col}) AS year,
                    COUNT(DISTINCT {count_by}) AS count
                FROM df
                WHERE {datetime_col} IS NOT NULL
                GROUP BY YEAR({datetime_col})
                ORDER BY year
            """).df()
        else:
            # Fall back to row counting
            result = duckdb.query(f"""
                SELECT
                    YEAR({datetime_col}) AS year,
                    COUNT(*) AS count
                FROM df
                WHERE {datetime_col} IS NOT NULL
                GROUP BY YEAR({datetime_col})
                ORDER BY year
            """).df()

        if result.empty:
            st.warning("No valid dates found")
            return

        # Create histogram using Plotly
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=result['year'],
            y=result['count'],
            marker=dict(
                color=result['count'],
                colorscale='Blues',
                showscale=True,
                colorbar=dict(title="Count")
            ),
            text=result['count'],
            texttemplate='%{text:,}',
            textposition='outside',
            hovertemplate='<b>Year %{x}</b><br>Count: %{y:,}<extra></extra>'
        ))

        fig.update_layout(
            title=f'Distribution of {label} by Year',
            xaxis_title='Year',
            yaxis_title=f'Number of {label}',
            xaxis=dict(
                tickmode='linear',
                dtick=1
            ),
            yaxis=dict(
                tickformat=','
            ),
            showlegend=False,
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Years", len(result))
        with col2:
            st.metric("Avg per Year", f"{result['count'].mean():,.0f}")
        with col3:
            st.metric("Max Year", f"{result['count'].max():,}")
        with col4:
            st.metric("Min Year", f"{result['count'].min():,}")

    except ImportError:
        st.warning("DuckDB not available. Install with: pip install duckdb")
    except Exception as e:
        st.error(f"Error computing year distribution: {e}")


def display_summary_statistics(analyzer, summary_stats, table_name):
    """
    Display summary statistics tab.

    Parameters:
    -----------
    analyzer : BaseTableAnalyzer or None
        The table analyzer instance (may be None for cached results)
    summary_stats : dict or None
        Summary statistics (may be None if not yet generated)
    table_name : str
        Name of the table being analyzed
    """

    # Check if validation was completed (requirement for summarization)
    completion = get_completion_status(table_name)

    if not completion['validation_complete']:
        st.warning("⚠️ Validation Required")
        st.info("Please run validation first before accessing summary statistics.")
        st.info("Go to the 'Validation Results' tab and click 'Run Analysis' with validation enabled.")
        return

    # Check if summary stats are available
    if not summary_stats:
        st.info("ℹ️ Summary statistics not yet generated. Click 'Run Analysis' in the sidebar to generate summary statistics.")
        return

    # Data Info Section
    st.markdown("### 📊 Data Overview")
    data_info = summary_stats.get('data_info', {})

    if 'error' in data_info:
        st.error(data_info['error'])
        return

    # Adjust columns based on table type
    if table_name == 'adt':
        # ADT table: show Total Rows, Unique Hospitalizations, Total Columns, Unique Locations
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{data_info.get('row_count', 0):,}")
        with col2:
            st.metric("Unique Hospitalizations", f"{data_info.get('unique_hospitalizations', 0):,}")
        with col3:
            st.metric("Total Columns", data_info.get('column_count', 0))
        with col4:
            st.metric("Unique Locations", data_info.get('unique_locations', 0))
    else:
        # Patient and Hospitalization tables
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{data_info.get('row_count', 0):,}")
        with col2:
            # Show unique hospitalizations for hospitalization table, unique patients otherwise
            if 'unique_hospitalizations' in data_info:
                st.metric("Unique Hospitalizations", f"{data_info.get('unique_hospitalizations', 0):,}")
            elif 'unique_patients' in data_info and data_info.get('unique_patients', 0) > 0:
                st.metric("Unique Patients", f"{data_info.get('unique_patients', 0):,}")
            elif 'unique_diagnosis_codes' in data_info:
                st.metric("Unique Diagnosis Codes", f"{data_info.get('unique_diagnosis_codes', 0):,}")
                # Show ICD format breakdown if available
                if data_info.get('icd10cm_percentage', 0) > 0 or data_info.get('icd9cm_percentage', 0) > 0:
                    icd10_pct = data_info.get('icd10cm_percentage', 0)
                    icd9_pct = data_info.get('icd9cm_percentage', 0)
                    help_text = f"ICD-10: {icd10_pct:.1f}%, ICD-9: {icd9_pct:.1f}%"
                    st.caption(help_text)
            else:
                # For tables without patient_id, show unique devices or modes
                if 'unique_devices' in data_info:
                    st.metric("Unique Devices", f"{data_info.get('unique_devices', 0):,}")
                elif 'unique_crrt_modes' in data_info:
                    st.metric("Unique CRRT Modes", f"{data_info.get('unique_crrt_modes', 0):,}")
        with col3:
            st.metric("Total Columns", data_info.get('column_count', 0))
        with col4:
            # Show death records for patient table only
            if 'has_death_records' in data_info:
                st.metric("Death Records", f"{int(data_info.get('has_death_records', 0)):,}")
            elif 'unique_diagnosis_codes' in data_info:
                st.metric("Avg Diagnoses/Hosp", f"{data_info.get('avg_diagnoses_per_hosp', 0):.1f}")
            elif 'unique_patients' in data_info and data_info.get('unique_patients', 0) > 0:
                st.metric("Unique Patients", f"{data_info.get('unique_patients', 0):,}")
            elif 'unique_med_categories' in data_info:
                st.metric("Unique Med Categories", f"{data_info.get('unique_med_categories', 0):,}")
            elif 'unique_crrt_modes' in data_info:
                st.metric("Unique CRRT Modes", f"{data_info.get('unique_crrt_modes', 0):,}")

    # Show dataset duration for hospitalization table
    if 'first_admission_year' in data_info and data_info.get('first_admission_year'):
        first_year = data_info.get('first_admission_year')
        last_year = data_info.get('last_admission_year')
        if first_year and last_year:
            st.info(f"📅 **Dataset Duration (admission_dttm):** {first_year} - {last_year} ({last_year - first_year + 1} years)")

            # Show year distribution histogram (lazy-load analyzer only when expander is opened)
            with st.expander("📊 View Year Distribution"):
                # Lazy load analyzer only when this feature is accessed
                if analyzer is None:
                    analyzer = _lazy_load_analyzer(table_name, st.session_state.config, analyzer)

                if analyzer and hasattr(analyzer, 'table') and hasattr(analyzer.table, 'df'):
                    _show_year_distribution(analyzer.table.df, 'admission_dttm', 'Hospitalizations', count_by='hospitalization_id')
                else:
                    st.warning("Data not available for year distribution")

    # Show dataset duration for ADT table
    if table_name == 'adt' and 'first_event_year' in data_info and data_info.get('first_event_year'):
        first_year = data_info.get('first_event_year')
        last_year = data_info.get('last_event_year')
        if first_year and last_year:
            st.info(f"📅 **Dataset Duration (in_dttm):** {first_year} - {last_year} ({last_year - first_year + 1} years)")

            # Show year distribution histogram (lazy-load analyzer only when expander is opened)
            with st.expander("📊 View Year Distribution"):
                # Lazy load analyzer only when this feature is accessed
                if analyzer is None:
                    analyzer = _lazy_load_analyzer(table_name, st.session_state.config, analyzer)

                if analyzer and hasattr(analyzer, 'table') and hasattr(analyzer.table, 'df'):
                    _show_year_distribution(analyzer.table.df, 'in_dttm', 'Hospitalizations', count_by='hospitalization_id')
                else:
                    st.warning("Data not available for year distribution")

    # Show dataset duration for CRRT table
    if table_name == 'crrt_therapy' and 'first_year' in data_info and data_info.get('first_year'):
        first_year = data_info.get('first_year')
        last_year = data_info.get('last_year')
        if first_year and last_year:
            st.info(f"📅 **Dataset Duration (recorded_dttm):** {first_year} - {last_year} ({last_year - first_year + 1} years)")

            # Show year distribution histogram (lazy-load analyzer only when expander is opened)
            with st.expander("📊 View Year Distribution"):
                # Lazy load analyzer only when this feature is accessed
                if analyzer is None:
                    analyzer = _lazy_load_analyzer(table_name, st.session_state.config, analyzer)

                if analyzer and hasattr(analyzer, 'table') and hasattr(analyzer.table, 'df'):
                    _show_year_distribution(analyzer.table.df, 'recorded_dttm', 'CRRT Hospitalizations', count_by='hospitalization_id')
                else:
                    st.warning("Data not available for year distribution")

    # Show dataset duration for Medication Admin Continuous table
    if table_name == 'medication_admin_continuous' and 'first_admin_year' in data_info and data_info.get('first_admin_year'):
        first_year = data_info.get('first_admin_year')
        last_year = data_info.get('last_admin_year')
        if first_year and last_year:
            st.info(f"📅 **Dataset Duration (admin_dttm):** {first_year} - {last_year} ({last_year - first_year + 1} years)")

            # Show year distribution histogram (lazy-load analyzer only when expander is opened)
            with st.expander("📊 View Year Distribution"):
                # Lazy load analyzer only when this feature is accessed
                if analyzer is None:
                    analyzer = _lazy_load_analyzer(table_name, st.session_state.config, analyzer)

                if analyzer and hasattr(analyzer, 'table') and hasattr(analyzer.table, 'df'):
                    _show_year_distribution(analyzer.table.df, 'admin_dttm', 'Medication Administrations', count_by='hospitalization_id')
                else:
                    st.warning("Data not available for year distribution")

    # Show dataset duration for Medication Admin Intermittent table
    if table_name == 'medication_admin_intermittent' and 'first_admin_year' in data_info and data_info.get('first_admin_year'):
        first_year = data_info.get('first_admin_year')
        last_year = data_info.get('last_admin_year')
        if first_year and last_year:
            st.info(f"📅 **Dataset Duration (admin_dttm):** {first_year} - {last_year} ({last_year - first_year + 1} years)")

            # Show year distribution histogram (lazy-load analyzer only when expander is opened)
            with st.expander("📊 View Year Distribution"):
                # Lazy load analyzer only when this feature is accessed
                if analyzer is None:
                    analyzer = _lazy_load_analyzer(table_name, st.session_state.config, analyzer)

                if analyzer and hasattr(analyzer, 'table') and hasattr(analyzer.table, 'df'):
                    _show_year_distribution(analyzer.table.df, 'admin_dttm', 'Medication Administrations', count_by='hospitalization_id')
                else:
                    st.warning("Data not available for year distribution")

    # Show hospitalization categories for ADT table
    if table_name == 'adt' and 'icu_hospitalizations' in data_info:
        st.markdown("#### 🏥 Hospitalization Categories")
        total_hospitalizations = data_info.get('unique_hospitalizations', 0)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            icu_count = data_info.get('icu_hospitalizations', 0)
            icu_pct = (icu_count / total_hospitalizations * 100) if total_hospitalizations > 0 else 0
            st.metric("ICU Hospitalizations", f"{icu_count:,} ({icu_pct:.1f}%)",
                     help="Hospitalizations that visited ICU at least once")
        with col2:
            icu_only_count = data_info.get('icu_only_hospitalizations', 0)
            icu_only_pct = (icu_only_count / total_hospitalizations * 100) if total_hospitalizations > 0 else 0
            st.metric("ICU-Only", f"{icu_only_count:,} ({icu_only_pct:.1f}%)",
                     help="Hospitalizations that only visited ICU")
        with col3:
            ed_count = data_info.get('ed_only_hospitalizations', 0)
            ed_pct = (ed_count / total_hospitalizations * 100) if total_hospitalizations > 0 else 0
            st.metric("ED-Only", f"{ed_count:,} ({ed_pct:.1f}%)",
                     help="Hospitalizations that only visited ED")
        with col4:
            ward_count = data_info.get('ward_only_hospitalizations', 0)
            ward_pct = (ward_count / total_hospitalizations * 100) if total_hospitalizations > 0 else 0
            st.metric("Ward-Only", f"{ward_count:,} ({ward_pct:.1f}%)",
                     help="Hospitalizations that only visited ward")

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
                     f"{missingness.get('overall_missing_percentage', 0):.2f}%",
                     help="Percentage of all cells (rows × columns) that contain missing values")
        with col3:
            st.metric("Complete Rows %",
                     f"{missingness.get('complete_rows_percentage', 0):.2f}%")

        # Show columns with missing data
        if missingness.get('columns_with_missing'):
            with st.expander("Columns with Missing Data", expanded=False):
                missing_df = pd.DataFrame(missingness['columns_with_missing'])
                st.dataframe(
                    missing_df,
                    width='stretch',
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
                # Skip mortality data (handled in missingness analysis)
                if key == 'mortality':
                    continue

                # Only display if it has the categorical distribution structure
                if 'values' not in dist_data or 'counts' not in dist_data:
                    continue

                # Format the display name
                display_name = key.replace('_', ' ').title()
                st.markdown(f"#### {display_name}")

                # For categorical distributions (check for 'values' key from get_categorical_distribution)
                if 'values' in dist_data and 'counts' in dist_data:
                    col1, col2 = st.columns(2)

                    # Create DataFrame from values, counts, percentages
                    categories_df = pd.DataFrame({
                        'value': dist_data['values'],
                        'count': dist_data['counts'],
                        'percentage': dist_data['percentages']
                    })

                    with col1:
                        # Create pie chart
                        if not categories_df.empty:
                            fig = px.pie(
                                categories_df,
                                values='count',
                                names='value',
                                title=f"{display_name} Distribution"
                            )

                            # Remove labels for Language Category to avoid clutter
                            if key == 'language_category':
                                fig.update_traces(textposition='none', textinfo='none')

                            st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        # Display statistics
                        st.write(f"**Unique Values:** {dist_data.get('unique_values', 0)}")
                        st.write(f"**Missing:** {dist_data.get('missing_count', 0)} "
                                f"({dist_data.get('missing_percentage', 0):.2f}%)")
                        st.write(f"**Total Rows:** {dist_data.get('total_rows', 0):,}")

                        # Show value counts table
                        if categories_df.shape[0] > 0:
                            st.dataframe(
                                categories_df[['value', 'count', 'percentage']].head(10),
                                width='stretch',
                                hide_index=True
                            )

                # Skip mortality statistics display (death_dttm missingness is shown in missingness analysis)

                st.divider()

    # Hospital Diagnosis-specific CCI distribution
    if table_name == 'hospital_diagnosis':
        # Lazy load analyzer if needed
        if analyzer is None:
            analyzer = _lazy_load_analyzer(table_name, st.session_state.config, analyzer)

        if analyzer and hasattr(analyzer, 'calculate_cci_distribution'):
            st.markdown("#### 🏥 Charlson Comorbidity Index (CCI) Distribution")
            st.caption("CCI scores are used to predict mortality risk based on comorbid conditions")

            cci_stats = analyzer.calculate_cci_distribution()

            if 'error' not in cci_stats:
                # Display CCI statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean CCI", f"{cci_stats['mean']:.2f}")
                with col2:
                    st.metric("Median CCI", f"{cci_stats['median']:.0f}")
                with col3:
                    st.metric("Max CCI", f"{cci_stats['max']}")
                with col4:
                    st.metric("Patients", f"{cci_stats['count']:,}")

                # Display risk categories
                if 'risk_categories' in cci_stats:
                    col1, col2 = st.columns(2)

                    with col1:
                        # Create pie chart for risk categories
                        risk_df = pd.DataFrame({
                            'Category': cci_stats['risk_categories']['labels'],
                            'Count': cci_stats['risk_categories']['counts'],
                            'Percentage': cci_stats['risk_categories']['percentages']
                        })

                        fig = px.pie(
                            risk_df,
                            values='Count',
                            names='Category',
                            title='CCI Risk Categories',
                            color_discrete_sequence=['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        st.write("**Risk Category Distribution:**")
                        st.dataframe(
                            risk_df[['Category', 'Count', 'Percentage']],
                            width='stretch',
                            hide_index=True
                        )

                st.divider()
            else:
                st.warning(cci_stats.get('error', 'Could not calculate CCI distribution'))

    # Hospital Diagnosis-specific Format Distribution
    if table_name == 'hospital_diagnosis':
        if analyzer is None:
            analyzer = _lazy_load_analyzer(table_name, st.session_state.config, analyzer)

        if analyzer and hasattr(analyzer, 'get_diagnosis_format_distribution'):
            st.markdown("#### 📊 Diagnosis Code Format Distribution")
            st.caption("Breakdown of diagnosis codes by ICD format (ICD-10CM vs ICD-9CM)")

            format_dist = analyzer.get_diagnosis_format_distribution()

            if 'error' not in format_dist and 'formats' in format_dist:
                col1, col2 = st.columns(2)

                with col1:
                    # Create pie chart for format distribution
                    format_df = pd.DataFrame(format_dist['formats'])
                    if not format_df.empty:
                        fig = px.pie(
                            format_df,
                            values='diagnosis_count',
                            names='format',
                            title='Diagnosis Codes by Format',
                            color_discrete_map={
                                'ICD10CM': '#3498db',
                                'ICD9CM': '#e74c3c'
                            }
                        )
                        fig.update_traces(
                            textposition='inside',
                            textinfo='percent+label',
                            hovertemplate='<b>%{label}</b><br>Count: %{value:,}<br>Percentage: %{percent}<extra></extra>'
                        )
                        st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.write("**Format Distribution Details:**")

                    # Create a summary table
                    for fmt_info in format_dist['formats']:
                        st.write(f"**{fmt_info['format']}:**")
                        st.write(f"  • Diagnoses: {fmt_info['diagnosis_count']:,} ({fmt_info['diagnosis_percentage']:.1f}%)")
                        st.write(f"  • Hospitalizations: {fmt_info['hospitalization_count']:,} ({fmt_info['hospitalization_percentage']:.1f}%)")

                    # Add CCI compatibility note
                    has_icd10 = any(fmt['format'] == 'ICD10CM' for fmt in format_dist['formats'])
                    if has_icd10:
                        st.success("✅ ICD-10CM codes available for CCI calculation")
                    else:
                        st.warning("⚠️ No ICD-10CM codes - CCI calculation not possible")

                st.divider()

    # Hospital Diagnosis-specific Top Diagnosis Codes
    if table_name == 'hospital_diagnosis':
        if analyzer is None:
            analyzer = _lazy_load_analyzer(table_name, st.session_state.config, analyzer)

        if analyzer and hasattr(analyzer, 'get_top_diagnosis_codes'):
            st.markdown("#### 🔝 Top Diagnosis Codes")
            st.caption("Most frequently occurring diagnosis codes across hospitalizations")

            top_codes = analyzer.get_top_diagnosis_codes(n=20)

            if 'error' not in top_codes:
                col1, col2 = st.columns(2)

                with col1:
                    # Create bar chart for top 10 codes
                    if len(top_codes['codes']) > 0:
                        top10_df = pd.DataFrame({
                            'Code': top_codes['codes'][:10],
                            'Hospitalizations': top_codes['counts'][:10],
                            'Percentage': top_codes['percentages'][:10]
                        })

                        fig = px.bar(
                            top10_df,
                            x='Hospitalizations',
                            y='Code',
                            orientation='h',
                            title='Top 10 Diagnosis Codes',
                            labels={'Hospitalizations': 'Number of Hospitalizations'},
                            text='Hospitalizations',
                            color='Hospitalizations',
                            color_continuous_scale='blues'
                        )
                        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
                        fig.update_layout(height=400, showlegend=False)
                        st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Display top 20 codes table
                    if len(top_codes['codes']) > 0:
                        st.write("**Top 20 Diagnosis Codes:**")
                        codes_df = pd.DataFrame({
                            'Code': top_codes['codes'][:20],
                            'Hospitalizations': [f"{c:,}" for c in top_codes['counts'][:20]],
                            'Percentage': [f"{p:.1f}%" for p in top_codes['percentages'][:20]]
                        })
                        st.dataframe(
                            codes_df,
                            width='stretch',
                            hide_index=True,
                            height=400
                        )

                st.divider()
            else:
                st.warning(top_codes.get('error', 'Could not retrieve top diagnosis codes'))

    # CRRT-specific numeric summary (before outlier handling)
    if table_name == 'crrt_therapy':
        # Lazy load analyzer if needed
        if analyzer is None:
            analyzer = _lazy_load_analyzer(table_name, st.session_state.config, analyzer)

        if analyzer and hasattr(analyzer, 'table') and hasattr(analyzer.table, 'df'):
            st.markdown("#### 📊 Numeric Variable Summary (Raw Data)")
            st.caption("Summary statistics before outlier handling")

            # Define numeric columns
            numeric_columns = [
                'blood_flow_rate',
                'pre_filter_replacement_fluid_rate',
                'post_filter_replacement_fluid_rate',
                'dialysate_flow_rate',
                'ultrafiltration_out'
            ]

            # Calculate summary statistics for raw data
            summary_rows = []
            df = analyzer.table.df

            for col in numeric_columns:
                if col in df.columns:
                    valid_data = df[col].dropna()
                    if len(valid_data) > 0:
                        summary_rows.append({
                            'Variable': col.replace('_', ' ').title(),
                            'Count': f"{len(valid_data):,}",
                            'Mean': f"{valid_data.mean():.2f}",
                            'Median': f"{valid_data.median():.2f}",
                            'Std': f"{valid_data.std():.2f}",
                            'Min': f"{valid_data.min():.2f}",
                            'Max': f"{valid_data.max():.2f}"
                        })

            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                st.dataframe(summary_df, hide_index=True, width='stretch')

            st.divider()

    # CRRT-specific categorical-numeric visualization
    if table_name == 'crrt_therapy':
        st.markdown("#### 📊 Numeric Distributions by Category")
        st.caption("Explore how numeric variables vary across different CRRT modes")

        # Try to load cached visualization data first
        import json
        output_dir = st.session_state.config.get('output_dir', 'output')
        results_dir = os.path.join(output_dir, 'final', 'results')
        os.makedirs(results_dir, exist_ok=True)
        viz_data_path = os.path.join(results_dir, f'{table_name}_visualization_data.json')

        use_cached_data = False
        viz_data = None

        if os.path.exists(viz_data_path):
            try:
                with open(viz_data_path, 'r') as f:
                    viz_data = json.load(f)
                use_cached_data = True
            except Exception as e:
                st.warning(f"Could not load cached visualization data: {e}")
                use_cached_data = False

        if not use_cached_data:
            # Fall back to loading and processing data on-the-fly
            # Lazy load analyzer if needed
            if analyzer is None:
                analyzer = _lazy_load_analyzer(table_name, st.session_state.config, analyzer)

            if analyzer and hasattr(analyzer, 'table') and hasattr(analyzer.table, 'df'):
                # Apply outlier handling to the data before visualization
                from modules.utils.outlier_handling import apply_outlier_ranges
                cleaned_df, outlier_stats = apply_outlier_ranges(
                    df=analyzer.table.df,
                    table_name=table_name
                )

                # Show outlier information
                total_outliers = sum(stats['outlier_count'] for stats in outlier_stats.values())
                if total_outliers > 0:
                    st.info(f"ℹ️ **{total_outliers:,} outliers replaced with NA** across numeric columns based on configured thresholds")

                # Define available columns
                categorical_columns = ['crrt_mode_category']
                numeric_columns = [
                    'blood_flow_rate',
                    'pre_filter_replacement_fluid_rate',
                    'post_filter_replacement_fluid_rate',
                    'dialysate_flow_rate',
                    'ultrafiltration_out'
                ]

                show_categorical_numeric_distribution(
                    df=cleaned_df,
                    categorical_columns=categorical_columns,
                    numeric_columns=numeric_columns,
                    table_name=table_name,
                    default_categorical='crrt_mode_category',
                    default_numeric='blood_flow_rate',
                    raw_df=analyzer.table.df  # Pass raw data for dual plots
                )
            else:
                st.warning("Data not available for visualization")
        else:
            # Use cached data for visualization
            if viz_data:
                # Show outlier information from cached data
                total_outliers = viz_data.get('total_outliers_replaced', 0)
                if total_outliers > 0:
                    st.info(f"ℹ️ **{total_outliers:,} outliers replaced with NA** across numeric columns based on configured thresholds")

                # Lazy load analyzer to get the dataframe
                if analyzer is None:
                    analyzer = _lazy_load_analyzer(table_name, st.session_state.config, analyzer)

                if analyzer and hasattr(analyzer, 'table') and hasattr(analyzer.table, 'df'):
                    # Apply outlier handling (same as cached)
                    from modules.utils.outlier_handling import apply_outlier_ranges
                    cleaned_df, _ = apply_outlier_ranges(
                        df=analyzer.table.df,
                        table_name=table_name
                    )

                    show_categorical_numeric_distribution(
                        df=cleaned_df,
                        categorical_columns=viz_data['categorical_columns'],
                        numeric_columns=viz_data['numeric_columns'],
                        table_name=table_name,
                        default_categorical='crrt_mode_category',
                        default_numeric='blood_flow_rate',
                        raw_df=analyzer.table.df  # Pass raw data for dual plots
                    )
                else:
                    st.warning("Data not available for visualization")

        st.divider()

    # Medication Admin Continuous - specific visualizations
    if table_name == 'medication_admin_continuous':
        # Lazy load analyzer if needed
        if analyzer is None:
            analyzer = _lazy_load_analyzer(table_name, st.session_state.config, analyzer)

        if analyzer and hasattr(analyzer, 'analyze_medication_by_group'):
            st.markdown("#### 💊 Medication Group Distribution")
            st.caption("Hospitalizations receiving each medication group")

            group_analysis = analyzer.analyze_medication_by_group()

            if 'error' not in group_analysis and 'groups' in group_analysis:
                # Display medication group metrics in columns
                groups_data = group_analysis['groups']
                total_hosps = group_analysis['total_hospitalizations']

                # Show top medication groups
                top_groups = sorted(groups_data.items(), key=lambda x: x[1]['hospitalization_count'], reverse=True)[:4]

                cols = st.columns(len(top_groups))
                for i, (group_name, group_data) in enumerate(top_groups):
                    with cols[i]:
                        # Custom display without arrow
                        st.markdown(f"""
                        <div style='padding: 10px 0;'>
                            <div style='font-size: 14px; margin-bottom: 4px; opacity: 0.7;'>{group_name.title()}</div>
                            <div style='font-size: 32px; font-weight: bold; line-height: 1.1;'>{group_data['hospitalization_count']:,}</div>
                            <div style='font-size: 14px; margin-top: 4px; opacity: 0.8;'>{group_data['percentage_of_hospitalizations']:.1f}% of hospitalizations</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.divider()

        # Medication dose statistics table
        if analyzer and hasattr(analyzer, 'get_dose_statistics_table'):
            st.markdown("#### 📊 Dose Distribution Statistics")
            st.caption("Statistical summary for each medication and dosing unit")

            # Get statistics table
            stats_df = analyzer.get_dose_statistics_table()

            if not stats_df.empty:
                # Display statistics table
                st.dataframe(
                    stats_df,
                    width='stretch',
                    hide_index=True,
                    height=400,
                    column_config={
                        "Medication": st.column_config.TextColumn("Medication", width="medium"),
                        "Unit": st.column_config.TextColumn("Unit", width="small"),
                        "Count": st.column_config.NumberColumn("Count", format="%d"),
                        "Min": st.column_config.NumberColumn("Min", format="%.2f"),
                        "Max": st.column_config.NumberColumn("Max", format="%.2f"),
                        "Mean": st.column_config.NumberColumn("Mean", format="%.2f"),
                        "Median": st.column_config.NumberColumn("Median", format="%.2f"),
                        "Q1": st.column_config.NumberColumn("Q1", format="%.2f"),
                        "Q3": st.column_config.NumberColumn("Q3", format="%.2f"),
                        "StdDev": st.column_config.NumberColumn("Std Dev", format="%.2f")
                    }
                )

                # Download option for statistics
                csv = stats_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Statistics CSV",
                    data=csv,
                    file_name="medication_dose_statistics.csv",
                    mime="text/csv"
                )
            else:
                st.info("No dose statistics available")

        # Grid of distribution plots
        if analyzer and hasattr(analyzer, 'generate_distribution_plots'):
            st.markdown("#### 📉 Dose Distribution Plots")
            st.caption("Visual distribution of doses for top medications")

            # Check if plot already exists
            intermediate_dir = os.path.join(st.session_state.config.get('output_dir', 'output'), 'intermediate')
            plot_file = os.path.join(intermediate_dir, 'medication_dose_distributions.png')

            if os.path.exists(plot_file):
                # Display existing plot
                from PIL import Image
                img = Image.open(plot_file)
                st.image(img, caption="Dose distributions for top medications (outliers removed)", width='stretch')

                # Regenerate button
                if st.button("🔄 Regenerate Plots"):
                    with st.spinner("Generating distribution plots..."):
                        plot_path = analyzer.generate_distribution_plots(max_meds=20)
                        if plot_path:
                            st.success("✅ Plots regenerated")
                            st.rerun()
            else:
                # Plots not yet generated - show info message
                st.info("📊 Distribution plots not yet generated. Run full analysis to generate plots.")

        st.divider()

    # Medication Admin Intermittent - specific visualizations
    if table_name == 'medication_admin_intermittent':
        # Lazy load analyzer if needed
        if analyzer is None:
            analyzer = _lazy_load_analyzer(table_name, st.session_state.config, analyzer)

        if analyzer and hasattr(analyzer, 'analyze_medication_by_group'):
            st.markdown("#### 💊 Medication Group Distribution")
            st.caption("Hospitalizations receiving each medication group")

            group_analysis = analyzer.analyze_medication_by_group()

            if 'error' not in group_analysis and 'groups' in group_analysis:
                # Display medication group metrics in columns
                groups_data = group_analysis['groups']
                total_hosps = group_analysis['total_hospitalizations']

                # Show top medication groups
                top_groups = sorted(groups_data.items(), key=lambda x: x[1]['hospitalization_count'], reverse=True)[:4]

                cols = st.columns(len(top_groups))
                for i, (group_name, group_data) in enumerate(top_groups):
                    with cols[i]:
                        # Custom display without arrow
                        st.markdown(f"""
                        <div style='padding: 10px 0;'>
                            <div style='font-size: 14px; margin-bottom: 4px; opacity: 0.7;'>{group_name.title()}</div>
                            <div style='font-size: 32px; font-weight: bold; line-height: 1.1;'>{group_data['hospitalization_count']:,}</div>
                            <div style='font-size: 14px; margin-top: 4px; opacity: 0.8;'>{group_data['percentage_of_hospitalizations']:.1f}% of hospitalizations</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.divider()

        # Medication dose statistics table
        if analyzer and hasattr(analyzer, 'get_dose_statistics_table'):
            st.markdown("#### 📊 Dose Distribution Statistics")
            st.caption("Statistical summary for each medication and dosing unit")

            # Get statistics table
            stats_df = analyzer.get_dose_statistics_table()

            if not stats_df.empty:
                # Display statistics table
                st.dataframe(
                    stats_df,
                    width='stretch',
                    hide_index=True,
                    height=400,
                    column_config={
                        "Medication": st.column_config.TextColumn("Medication", width="medium"),
                        "Unit": st.column_config.TextColumn("Unit", width="small"),
                        "Count": st.column_config.NumberColumn("Count", format="%d"),
                        "Min": st.column_config.NumberColumn("Min", format="%.2f"),
                        "Max": st.column_config.NumberColumn("Max", format="%.2f"),
                        "Mean": st.column_config.NumberColumn("Mean", format="%.2f"),
                        "Median": st.column_config.NumberColumn("Median", format="%.2f"),
                        "Q1": st.column_config.NumberColumn("Q1", format="%.2f"),
                        "Q3": st.column_config.NumberColumn("Q3", format="%.2f"),
                        "StdDev": st.column_config.NumberColumn("Std Dev", format="%.2f")
                    }
                )

                # Download option for statistics
                csv = stats_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Statistics CSV",
                    data=csv,
                    file_name="medication_intermittent_dose_statistics.csv",
                    mime="text/csv"
                )
            else:
                st.info("No dose statistics available")

        # Grid of distribution plots
        if analyzer and hasattr(analyzer, 'generate_distribution_plots'):
            st.markdown("#### 📉 Dose Distribution Plots")
            st.caption("Visual distribution of doses for top medications")

            # Check if plot already exists
            intermediate_dir = os.path.join(st.session_state.config.get('output_dir', 'output'), 'intermediate')
            plot_file = os.path.join(intermediate_dir, 'medication_intermittent_dose_distributions.png')

            if os.path.exists(plot_file):
                # Display existing plot
                from PIL import Image
                img = Image.open(plot_file)
                st.image(img, caption="Dose distributions for top medications (outliers removed)", width='stretch')

                # Regenerate button
                if st.button("🔄 Regenerate Plots", key="regen_intermittent_plots"):
                    with st.spinner("Generating distribution plots..."):
                        plot_path = analyzer.generate_distribution_plots(max_meds=20)
                        if plot_path:
                            st.success("✅ Plots regenerated")
                            st.rerun()
            else:
                # Plots not yet generated - show info message
                st.info("📊 Distribution plots not yet generated. Run full analysis to generate plots.")

        st.divider()

    # Show dataset duration for ECMO/MCS table
    if table_name == 'ecmo_mcs' and 'first_year' in data_info and data_info.get('first_year'):
        first_year = data_info.get('first_year')
        last_year = data_info.get('last_year')
        if first_year and last_year:
            st.info(f"📅 **Dataset Duration (recorded_dttm):** {first_year} - {last_year} ({last_year - first_year + 1} years)")

            # Show year distribution histogram (lazy-load analyzer only when expander is opened)
            with st.expander("📊 View Year Distribution"):
                # Lazy load analyzer only when this feature is accessed
                if analyzer is None:
                    analyzer = _lazy_load_analyzer(table_name, st.session_state.config, analyzer)

                if analyzer and hasattr(analyzer, 'table') and hasattr(analyzer.table, 'df'):
                    _show_year_distribution(analyzer.table.df, 'recorded_dttm', 'ECMO/MCS Hospitalizations', count_by='hospitalization_id')
                else:
                    st.warning("Data not available for year distribution")

    # Show dataset duration for Vitals table
    if table_name == 'vitals' and 'first_recording_year' in data_info and data_info.get('first_recording_year'):
        first_year = data_info.get('first_recording_year')
        last_year = data_info.get('last_recording_year')
        if first_year and last_year:
            st.info(f"📅 **Dataset Duration (recorded_dttm):** {first_year} - {last_year} ({last_year - first_year + 1} years)")

            # Show year distribution histogram (lazy-load analyzer only when expander is opened)
            with st.expander("📊 View Year Distribution"):
                # Lazy load analyzer only when this feature is accessed
                if analyzer is None:
                    analyzer = _lazy_load_analyzer(table_name, st.session_state.config, analyzer)

                if analyzer and hasattr(analyzer, 'table') and hasattr(analyzer.table, 'df'):
                    _show_year_distribution(analyzer.table.df, 'recorded_dttm', 'Vital Sign Records', count_by='hospitalization_id')
                else:
                    st.warning("Data not available for year distribution")

    # Show dataset duration for Respiratory Support table
    if table_name == 'respiratory_support' and 'first_recording_year' in data_info and data_info.get('first_recording_year'):
        first_year = data_info.get('first_recording_year')
        last_year = data_info.get('last_recording_year')
        if first_year and last_year:
            st.info(f"📅 **Dataset Duration (recorded_dttm):** {first_year} - {last_year} ({last_year - first_year + 1} years)")

            # Show year distribution histogram (lazy-load analyzer only when expander is opened)
            with st.expander("📊 View Year Distribution"):
                # Lazy load analyzer only when this feature is accessed
                if analyzer is None:
                    analyzer = _lazy_load_analyzer(table_name, st.session_state.config, analyzer)

                if analyzer and hasattr(analyzer, 'table') and hasattr(analyzer.table, 'df'):
                    _show_year_distribution(analyzer.table.df, 'recorded_dttm', 'Respiratory Support Events', count_by='hospitalization_id')
                else:
                    st.warning("Data not available for year distribution")

    # Show dataset duration for Labs table
    if table_name == 'labs' and 'first_lab_year' in data_info and data_info.get('first_lab_year'):
        first_year = data_info.get('first_lab_year')
        last_year = data_info.get('last_lab_year')
        if first_year and last_year:
            st.info(f"📅 **Dataset Duration (lab_order_dttm):** {first_year} - {last_year} ({last_year - first_year + 1} years)")

            # Show year distribution histogram (lazy-load analyzer only when expander is opened)
            with st.expander("📊 View Year Distribution"):
                # Lazy load analyzer only when this feature is accessed
                if analyzer is None:
                    analyzer = _lazy_load_analyzer(table_name, st.session_state.config, analyzer)

                if analyzer and hasattr(analyzer, 'table') and hasattr(analyzer.table, 'df'):
                    _show_year_distribution(analyzer.table.df, 'lab_order_dttm', 'Lab Orders', count_by='hospitalization_id')
                else:
                    st.warning("Data not available for year distribution")

    # Show dataset duration for Position table
    if table_name == 'position' and 'first_recording_year' in data_info and data_info.get('first_recording_year'):
        first_year = data_info.get('first_recording_year')
        last_year = data_info.get('last_recording_year')
        if first_year and last_year:
            st.info(f"📅 **Dataset Duration (recorded_dttm):** {first_year} - {last_year} ({last_year - first_year + 1} years)")

            # Show year distribution histogram (lazy-load analyzer only when expander is opened)
            with st.expander("📊 View Year Distribution"):
                # Lazy load analyzer only when this feature is accessed
                if analyzer is None:
                    analyzer = _lazy_load_analyzer(table_name, st.session_state.config, analyzer)

                if analyzer and hasattr(analyzer, 'table') and hasattr(analyzer.table, 'df'):
                    _show_year_distribution(analyzer.table.df, 'recorded_dttm', 'Position Records', count_by='hospitalization_id')
                else:
                    st.warning("Data not available for year distribution")

    # ECMO/MCS-specific numeric summary (before outlier handling)
    if table_name == 'ecmo_mcs':
        # Lazy load analyzer if needed
        if analyzer is None:
            analyzer = _lazy_load_analyzer(table_name, st.session_state.config, analyzer)

        if analyzer and hasattr(analyzer, 'table') and hasattr(analyzer.table, 'df'):
            st.markdown("#### 📊 Numeric Variable Summary (Raw Data)")
            st.caption("Summary statistics before outlier handling")

            # Define numeric columns
            numeric_columns = [
                'device_rate',
                'sweep',
                'flow',
                'fdO2'
            ]

            # Calculate summary statistics for raw data
            summary_rows = []
            df = analyzer.table.df

            for col in numeric_columns:
                if col in df.columns:
                    valid_data = df[col].dropna()
                    if len(valid_data) > 0:
                        summary_rows.append({
                            'Variable': col.replace('_', ' ').title(),
                            'Count': f"{len(valid_data):,}",
                            'Mean': f"{valid_data.mean():.2f}",
                            'Median': f"{valid_data.median():.2f}",
                            'Std': f"{valid_data.std():.2f}",
                            'Min': f"{valid_data.min():.2f}",
                            'Max': f"{valid_data.max():.2f}"
                        })

            if summary_rows:
                summary_df = pd.DataFrame(summary_rows)
                st.dataframe(summary_df, hide_index=True, width='stretch')

            st.divider()

    # ECMO/MCS-specific categorical-numeric visualization
    if table_name == 'ecmo_mcs':
        st.markdown("#### 📊 Numeric Distributions by Category")
        st.caption("Explore how numeric variables vary across different device categories and MCS groups")

        # Try to load cached visualization data first
        import json
        output_dir = st.session_state.config.get('output_dir', 'output')
        results_dir = os.path.join(output_dir, 'final', 'results')
        os.makedirs(results_dir, exist_ok=True)
        viz_data_path = os.path.join(results_dir, f'{table_name}_visualization_data.json')

        use_cached_data = False
        viz_data = None

        if os.path.exists(viz_data_path):
            try:
                with open(viz_data_path, 'r') as f:
                    viz_data = json.load(f)
                use_cached_data = True
            except Exception as e:
                st.warning(f"Could not load cached visualization data: {e}")
                use_cached_data = False

        if not use_cached_data:
            # Fall back to loading and processing data on-the-fly
            # Lazy load analyzer if needed
            if analyzer is None:
                analyzer = _lazy_load_analyzer(table_name, st.session_state.config, analyzer)

            if analyzer and hasattr(analyzer, 'table') and hasattr(analyzer.table, 'df'):
                # Apply outlier handling to the data before visualization
                from modules.utils.outlier_handling import apply_outlier_ranges
                cleaned_df, outlier_stats = apply_outlier_ranges(
                    df=analyzer.table.df,
                    table_name=table_name
                )

                # Show outlier information
                total_outliers = sum(stats['outlier_count'] for stats in outlier_stats.values())
                if total_outliers > 0:
                    st.info(f"ℹ️ **{total_outliers:,} outliers replaced with NA** across numeric columns based on configured thresholds")

                # Define available columns
                categorical_columns = ['device_category', 'mcs_group']
                numeric_columns = [
                    'device_rate',
                    'sweep',
                    'flow',
                    'fdO2'
                ]

                show_categorical_numeric_distribution(
                    df=cleaned_df,
                    categorical_columns=categorical_columns,
                    numeric_columns=numeric_columns,
                    table_name=table_name,
                    default_categorical='device_category',
                    default_numeric='flow',
                    raw_df=analyzer.table.df  # Pass raw data for dual plots
                )
            else:
                st.warning("Data not available for visualization")
        else:
            # Use cached data for visualization
            if viz_data:
                # Show outlier information from cached data
                total_outliers = viz_data.get('total_outliers_replaced', 0)
                if total_outliers > 0:
                    st.info(f"ℹ️ **{total_outliers:,} outliers replaced with NA** across numeric columns based on configured thresholds")

                # Lazy load analyzer to get the dataframe
                if analyzer is None:
                    analyzer = _lazy_load_analyzer(table_name, st.session_state.config, analyzer)

                if analyzer and hasattr(analyzer, 'table') and hasattr(analyzer.table, 'df'):
                    # Apply outlier handling (same as cached)
                    from modules.utils.outlier_handling import apply_outlier_ranges
                    cleaned_df, _ = apply_outlier_ranges(
                        df=analyzer.table.df,
                        table_name=table_name
                    )

                    show_categorical_numeric_distribution(
                        df=cleaned_df,
                        categorical_columns=viz_data['categorical_columns'],
                        numeric_columns=viz_data['numeric_columns'],
                        table_name=table_name,
                        default_categorical='device_category',
                        default_numeric='flow',
                        raw_df=analyzer.table.df  # Pass raw data for dual plots
                    )
                else:
                    st.warning("Data not available for visualization")

        st.divider()

    # Patient-specific summary table
    if analyzer and hasattr(analyzer, 'generate_patient_summary'):
        st.markdown("### 📋 Patient Demographics Summary")
        summary_df = analyzer.generate_patient_summary()
        if not summary_df.empty:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.dataframe(summary_df, width='stretch', hide_index=True)
            with col2:
                # Export to CSV button
                csv_data = summary_df.to_csv(index=False)
                st.download_button(
                    label="📥 Export CSV",
                    data=csv_data,
                    file_name=f"{table_name}_demographics_summary.csv",
                    mime="text/csv",
                    width='stretch'
                )

    # Hospitalization-specific summary table
    if analyzer and hasattr(analyzer, 'generate_hospitalization_summary'):
        st.markdown("### 📋 Hospitalization Summary")
        summary_df = analyzer.generate_hospitalization_summary()
        if not summary_df.empty:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.dataframe(summary_df, width='stretch', hide_index=True)
            with col2:
                # Export to CSV button
                csv_data = summary_df.to_csv(index=False)
                st.download_button(
                    label="📥 Export CSV",
                    data=csv_data,
                    file_name=f"{table_name}_summary.csv",
                    mime="text/csv",
                    width='stretch'
                )

    # ADT-specific summary table
    if analyzer and hasattr(analyzer, 'generate_adt_summary'):
        st.markdown("### 📋 ADT Summary")
        summary_df = analyzer.generate_adt_summary()
        if not summary_df.empty:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.dataframe(summary_df, width='stretch', hide_index=True)
            with col2:
                # Export to CSV button
                csv_data = summary_df.to_csv(index=False)
                st.download_button(
                    label="📥 Export CSV",
                    data=csv_data,
                    file_name=f"{table_name}_summary.csv",
                    mime="text/csv",
                    width='stretch'
                )

    # Hospital Diagnosis-specific summary table
    if analyzer and hasattr(analyzer, 'generate_hospital_diagnosis_summary'):
        st.markdown("### 📋 Hospital Diagnosis Summary")
        summary_df = analyzer.generate_hospital_diagnosis_summary()
        if not summary_df.empty:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.dataframe(summary_df, width='stretch', hide_index=True)
            with col2:
                # Export to CSV button
                csv_data = summary_df.to_csv(index=False)
                st.download_button(
                    label="📥 Export CSV",
                    data=csv_data,
                    file_name=f"{table_name}_diagnosis_summary.csv",
                    mime="text/csv",
                    width='stretch'
                )

    # Vitals-specific summary table
    if analyzer and hasattr(analyzer, 'generate_vitals_summary'):
        st.markdown("### 📋 Vitals Summary")
        summary_df = analyzer.generate_vitals_summary()
        if not summary_df.empty:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.dataframe(summary_df, width='stretch', hide_index=True)
            with col2:
                # Export to CSV button
                csv_data = summary_df.to_csv(index=False)
                st.download_button(
                    label="📥 Export CSV",
                    data=csv_data,
                    file_name=f"{table_name}_vitals_summary.csv",
                    mime="text/csv",
                    width='stretch'
                )

    # Respiratory Support-specific summary table
    if analyzer and hasattr(analyzer, 'generate_respiratory_summary'):
        st.markdown("### 📋 Respiratory Support Summary")
        summary_df = analyzer.generate_respiratory_summary()
        if not summary_df.empty:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.dataframe(summary_df, width='stretch', hide_index=True)
            with col2:
                # Export to CSV button
                csv_data = summary_df.to_csv(index=False)
                st.download_button(
                    label="📥 Export CSV",
                    data=csv_data,
                    file_name=f"{table_name}_respiratory_summary.csv",
                    mime="text/csv",
                    width='stretch'
                )

    # Labs-specific summary table
    if analyzer and hasattr(analyzer, 'generate_labs_summary'):
        st.markdown("### 📋 Labs Summary")
        summary_df = analyzer.generate_labs_summary()
        if not summary_df.empty:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.dataframe(summary_df, width='stretch', hide_index=True)
            with col2:
                # Export to CSV button
                csv_data = summary_df.to_csv(index=False)
                st.download_button(
                    label="📥 Export CSV",
                    data=csv_data,
                    file_name=f"{table_name}_labs_summary.csv",
                    mime="text/csv",
                    width='stretch'
                )

    # Position-specific summary table
    if analyzer and hasattr(analyzer, 'generate_position_summary'):
        st.markdown("### 📋 Position Summary")
        summary_df = analyzer.generate_position_summary()
        if not summary_df.empty:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.dataframe(summary_df, width='stretch', hide_index=True)
            with col2:
                # Export to CSV button
                csv_data = summary_df.to_csv(index=False)
                st.download_button(
                    label="📥 Export CSV",
                    data=csv_data,
                    file_name=f"{table_name}_position_summary.csv",
                    mime="text/csv",
                    width='stretch'
                )


if __name__ == "__main__":
    main()