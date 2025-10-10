"""
CLIF 2.1 Validation & Summarization System
Streamlit Application

This application provides an interactive interface for validating and analyzing
CLIF 2.1 data tables using clifpy.
"""

import warnings
# Suppress Plotly and Streamlit deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='plotly')
warnings.filterwarnings('ignore', category=FutureWarning, module='streamlit')
warnings.filterwarnings('ignore', message='.*keyword arguments have been deprecated.*')
warnings.filterwarnings('ignore', message='.*use_container_width.*')

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
from modules.utils import (
    get_validation_summary,
    get_missingness_summary,
    create_missingness_report,
    create_error_id,
    create_feedback_structure,
    update_user_decision,
    get_feedback_summary,
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
        padding: 0.5rem;
        margin-bottom: 2rem;
    }
    .main-header h3 {
        margin: 0.5rem 0 0 0;
        padding: 0;
        color: #999;
        font-weight: 400;
        font-size: 1.1rem;
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
    .status-block-complete {
        background: #d4edda;
        border: 2px solid #a8d5ba;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .status-block-partial {
        background: #fff3cd;
        border: 2px solid #f4d799;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .status-block-incomplete {
        background: #f8d7da;
        border: 2px solid #f1b8bc;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .status-block-complete h4 {
        color: #155724;
        margin: 0;
        font-weight: bold;
    }
    .status-block-partial h4 {
        color: #856404;
        margin: 0;
        font-weight: bold;
    }
    .status-block-incomplete h4 {
        color: #721c24;
        margin: 0;
        font-weight: bold;
    }
    .status-complete {
        color: #155724;
        font-weight: bold;
    }
    .status-partial {
        color: #856404;
        font-weight: bold;
    }
    .status-incomplete {
        color: #721c24;
        font-weight: bold;
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

        # All CLIF 2.1 tables are now implemented
        available_tables = ['patient', 'hospitalization', 'adt', 'code_status', 'crrt_therapy', 'ecmo_mcs',
                           'hospital_diagnosis', 'labs', 'medication_admin_continuous', 'medication_admin_intermittent',
                           'microbiology_culture', 'microbiology_nonculture', 'microbiology_susceptibility',
                           'patient_assessments', 'patient_procedures', 'position', 'respiratory_support', 'vitals']

        # Table selection dropdown
        selected_table = st.selectbox(
            "Select Table to Analyze",
            options=available_tables,
            format_func=lambda x: TABLE_DISPLAY_NAMES[x]
        )

        # Show re-analyze option if table is cached
        force_reanalyze = False
        if is_table_cached(selected_table):
            force_reanalyze = st.checkbox("🔄 Re-analyze table", value=False)

        # Run analysis button
        if st.button("🚀 Run Analysis", type="primary", width='stretch'):
            st.session_state.run_analysis = True
            st.session_state.selected_table = selected_table
            st.session_state.run_validation = True
            st.session_state.run_outlier_handling = config.get('analysis_settings', {}).get('outlier_detection', False)
            st.session_state.calculate_sofa = False
            # Always force reanalyze when Run Analysis is clicked
            st.session_state.force_reanalyze = True
            st.rerun()

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

        st.markdown(
            f'<div class="main-header">'
            f'<h3>CLIF 2.1 Validation & Summarization</h3>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Main content area
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
            with st.spinner(f"Loading {TABLE_DISPLAY_NAMES[table_name]} table..."):
                analyzer = analyzer_class(data_dir, filetype, timezone, output_dir)

            # Delete old validation response file if it exists (fresh analysis = no old feedback)
            response_file = os.path.join(output_dir, 'final', f'{table_name}_validation_response.json')
            if os.path.exists(response_file):
                try:
                    os.remove(response_file)
                    st.info("🔄 Cleared previous feedback - fresh analysis starting")
                except Exception as e:
                    st.warning(f"Could not remove old feedback file: {e}")

            # Run analysis
            with st.spinner("Running validation..."):
                validation_results = analyzer.validate() if run_validation else None

            with st.spinner("Calculating summary statistics..."):
                summary_stats = analyzer.get_summary_statistics()

            # Automatically save validation and summary results to JSON files
            if validation_results:
                try:
                    analyzer.save_summary_data(validation_results, '_validation')
                    st.success("✅ Validation results saved")
                except Exception as e:
                    st.warning(f"Could not save validation results: {e}")

                # Generate PDF report
                try:
                    final_dir = os.path.join(output_dir, 'final')
                    os.makedirs(final_dir, exist_ok=True)

                    pdf_generator = ValidationPDFGenerator()
                    pdf_path = os.path.join(final_dir, f"{table_name}_validation_report.pdf")

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
                        txt_path = os.path.join(final_dir, f"{table_name}_validation_report.txt")
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
                final_dir = os.path.join(output_dir, 'final')
                os.makedirs(final_dir, exist_ok=True)

                # Save patient demographics summary
                if hasattr(analyzer, 'generate_patient_summary'):
                    patient_summary_df = analyzer.generate_patient_summary()
                    if not patient_summary_df.empty:
                        csv_filepath = os.path.join(final_dir, f"{table_name}_demographics_summary.csv")
                        patient_summary_df.to_csv(csv_filepath, index=False)
                        st.success(f"✅ Patient demographics summary CSV saved")

                # Save hospitalization summary
                if hasattr(analyzer, 'generate_hospitalization_summary'):
                    hosp_summary_df = analyzer.generate_hospitalization_summary()
                    if not hosp_summary_df.empty:
                        csv_filepath = os.path.join(final_dir, f"{table_name}_summary.csv")
                        hosp_summary_df.to_csv(csv_filepath, index=False)
                        st.success(f"✅ Hospitalization summary CSV saved")

                # Save ADT summary
                if hasattr(analyzer, 'generate_adt_summary'):
                    adt_summary_df = analyzer.generate_adt_summary()
                    if not adt_summary_df.empty:
                        csv_filepath = os.path.join(final_dir, f"{table_name}_summary.csv")
                        adt_summary_df.to_csv(csv_filepath, index=False)
                        st.success(f"✅ ADT summary CSV saved")

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

    # Create tabs for validation and summary
    tab1, tab2 = st.tabs(["🔍 Validation", "📊 Summary"])

    with tab1:
        display_validation_results(analyzer, validation_results, existing_feedback, table_name)

    with tab2:
        display_summary_statistics(analyzer, summary_stats, table_name)

    # Clear analysis_just_completed flag without triggering rerun
    # The sidebar status will update on next user interaction
    if st.session_state.get('analysis_just_completed', False):
        st.session_state.analysis_just_completed = False


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


def display_validation_results(analyzer, validation_results, existing_feedback, table_name):
    """
    Display validation results tab with user feedback option.

    Parameters:
    -----------
    analyzer : BaseTableAnalyzer
        The table analyzer instance
    validation_results : dict
        Validation results
    existing_feedback : dict or None
        Existing feedback structure if available
    table_name : str
        Name of the table
    """
    if not validation_results:
        st.info("ℹ️ Validation not run. Check the 'Run Validation' box in the sidebar to enable.")
        return

    # Get current status (considering feedback if exists)
    current_status = get_table_status(table_name)
    if current_status == 'not_analyzed' or current_status == 'unknown':
        current_status = validation_results.get('status', 'unknown')

    status_block_class = f'status-block-{current_status}'

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="{status_block_class}">
            <h4>Validation Status</h4>
            <p class="status-{current_status}">{current_status.upper()}</p>
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

    # Classify errors by status impact
    from modules.utils.validation import classify_errors_by_status_impact

    # Get required columns from analyzer if available
    required_columns = []
    if analyzer and hasattr(analyzer, 'table') and hasattr(analyzer.table, 'schema'):
        required_columns = analyzer.table.schema.get('required_columns', [])

    # Get configured timezone
    config_timezone = st.session_state.config.get('timezone', 'UTC') if 'config' in st.session_state else 'UTC'

    classified_errors = classify_errors_by_status_impact(errors, required_columns, table_name, config_timezone)
    status_affecting = classified_errors['status_affecting']
    informational = classified_errors['informational']

    # Count total errors in each category
    status_affecting_count = sum([
        len(status_affecting.get('schema_errors', [])),
        len(status_affecting.get('data_quality_issues', [])),
        len(status_affecting.get('other_errors', []))
    ])

    informational_count = sum([
        len(informational.get('schema_errors', [])),
        len(informational.get('data_quality_issues', [])),
        len(informational.get('other_errors', []))
    ])

    # Display errors separated by status impact
    st.markdown("### Validation Issues")

    if error_count == 0:
        st.success("✅ No validation issues found!")
    else:
        # Status-Affecting Errors Section (require feedback)
        if status_affecting_count > 0:
            st.markdown(f"#### ⚠️ Status-Affecting Errors ({status_affecting_count})")
            st.caption("These errors affect the validation status and require your review.")

            # Schema errors
            schema_errors = status_affecting.get('schema_errors', [])
            if schema_errors:
                with st.expander(f"🔴 Schema Errors ({len(schema_errors)})", expanded=True):
                    for error in schema_errors:
                        st.markdown(f"**{error['type']}**")
                        st.write(error['description'])
                        st.divider()

            # Data quality issues
            quality_issues = status_affecting.get('data_quality_issues', [])
            if quality_issues:
                with st.expander(f"🟡 Data Quality Issues ({len(quality_issues)})", expanded=True):
                    for issue in quality_issues:
                        st.markdown(f"**{issue['type']}**")
                        st.write(issue['description'])

                        # Show additional details if available
                        if 'details' in issue and issue['details']:
                            details = issue['details']
                            # Display missing_values list if present
                            if 'missing_values' in details and details['missing_values']:
                                st.caption("**Missing values:**")
                                st.write(details['missing_values'])
                            # Display invalid_values list if present
                            elif 'invalid_values' in details and details['invalid_values']:
                                st.caption("**Invalid values:**")
                                st.write(details['invalid_values'])

                        st.divider()

            # Other errors (unlikely but handle)
            other_errors = status_affecting.get('other_errors', [])
            if other_errors:
                with st.expander(f"⚠️ Other Issues ({len(other_errors)})", expanded=True):
                    for error in other_errors:
                        st.markdown(f"**{error['type']}**")
                        st.write(error['description'])
                        st.divider()

        # Informational Issues Section (no feedback required)
        if informational_count > 0:
            st.markdown(f"#### ℹ️ Informational Issues ({informational_count})")
            st.caption("These issues are for your awareness but do not affect the validation status.")

            # Schema info
            schema_info = informational.get('schema_errors', [])
            if schema_info:
                with st.expander(f"📋 Schema Information ({len(schema_info)})"):
                    for error in schema_info:
                        st.markdown(f"**{error['type']}**")
                        st.write(error['description'])
                        st.divider()

            # Data quality observations
            quality_obs = informational.get('data_quality_issues', [])
            if quality_obs:
                with st.expander(f"📊 Data Quality Observations ({len(quality_obs)})"):
                    for issue in quality_obs:
                        st.markdown(f"**{issue['type']}**")
                        st.write(issue['description'])
                        st.divider()

            # Other observations
            other_obs = informational.get('other_errors', [])
            if other_obs:
                with st.expander(f"ℹ️ Other Observations ({len(other_obs)})"):
                    for error in other_obs:
                        st.markdown(f"**{error['type']}**")
                        st.write(error['description'])
                        st.divider()

    # Data quality checks section
    if analyzer and hasattr(analyzer, 'check_data_quality'):
        st.divider()
        st.markdown("### ✅ Data Quality Checks")
        quality_checks = analyzer.check_data_quality()

        if 'error' not in quality_checks:
            for check_name, check_result in quality_checks.items():
                status_icon = "✅" if check_result['status'] == 'pass' else "⚠️" if check_result['status'] == 'warning' else "❌"
                check_display = check_name.replace('_', ' ').title()

                # Show expandable details if there are issues
                if check_result['count'] > 0:
                    with st.expander(f"{status_icon} **{check_display}:** {check_result['count']} ({check_result['percentage']}%)", expanded=False):
                        st.write(f"**Definition:** {_get_quality_check_definition(check_name)}")

                        # Show sample of problematic rows if available
                        if 'examples' in check_result and check_result['examples'] is not None:
                            if not check_result['examples'].empty:
                                st.write("**Sample of problematic records:**")
                                st.dataframe(check_result['examples'], width='stretch', hide_index=True)
                else:
                    st.write(f"{status_icon} **{check_display}:** {check_result['count']} ({check_result['percentage']}%)")

    # User Feedback / Review Mode - ONLY for status-affecting errors
    if status_affecting_count > 0:
        st.divider()
        st.markdown("### 📝 Review & Feedback")
        st.caption("⚠️ Only status-affecting errors require feedback. Informational issues are acknowledged automatically.")

        # Initialize or load feedback
        if existing_feedback is None:
            # Create feedback structure with ONLY status-affecting errors
            # Build a modified validation_results dict with only status-affecting errors
            status_affecting_validation = {
                'status': validation_results.get('status', 'unknown'),
                'errors': status_affecting  # Use only status-affecting errors
            }
            existing_feedback = create_feedback_structure(status_affecting_validation, table_name)

        # Show feedback summary
        feedback_summary = get_feedback_summary(existing_feedback)
        st.info(feedback_summary)

        # Check if feedback status differs from original
        if existing_feedback['adjusted_status'] != existing_feedback['original_status']:
            st.success(f"🔄 Status adjusted from **{existing_feedback['original_status'].upper()}** "
                      f"to **{existing_feedback['adjusted_status'].upper()}** based on your feedback")

        # Toggle review mode
        review_mode = st.checkbox("📋 Review Status-Affecting Errors",
                                  help="Accept or reject status-affecting errors based on your site's data context")

        if review_mode:
            st.markdown("#### Review Each Status-Affecting Error")
            st.caption("Mark errors as 'Accepted' (valid issue) or 'Rejected' (site-specific, not an issue)")

            # Only show status-affecting errors for feedback
            all_errors = (status_affecting.get('schema_errors', []) +
                         status_affecting.get('data_quality_issues', []) +
                         status_affecting.get('other_errors', []))

            # Create decision tracking
            if 'error_decisions' not in st.session_state:
                st.session_state.error_decisions = {}

            for idx, error in enumerate(all_errors):
                error_id = create_error_id(error)

                # Get existing decision
                existing_decision_info = existing_feedback['user_decisions'].get(error_id, {})
                current_decision = existing_decision_info.get('decision', 'pending')
                current_reason = existing_decision_info.get('reason', '')

                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 2])

                    with col1:
                        st.markdown(f"**{error['type']}**")
                        st.caption(error['description'])

                    with col2:
                        decision = st.radio(
                            "Decision",
                            ["Pending", "Accepted", "Rejected"],
                            index=["pending", "accepted", "rejected"].index(current_decision.lower()),
                            key=f"decision_{error_id}",
                            horizontal=False
                        )

                    with col3:
                        if decision == "Rejected":
                            reason = st.text_input(
                                "Reason for rejection",
                                value=current_reason,
                                key=f"reason_{error_id}",
                                placeholder="e.g., Site-specific category"
                            )
                        else:
                            reason = ""
                            if decision == "Accepted" and current_reason:
                                st.caption(f"Previous reason: {current_reason}")

                    # Store decision
                    st.session_state.error_decisions[error_id] = {
                        'decision': decision.lower(),
                        'reason': reason
                    }

                    st.divider()

            # Save feedback button
            col1, col2 = st.columns(2)

            with col1:
                if st.button("💾 Save Feedback", type="primary", width='stretch'):
                    # Update all decisions
                    for error_id, decision_info in st.session_state.error_decisions.items():
                        existing_feedback = update_user_decision(
                            existing_feedback,
                            error_id,
                            decision_info['decision'],
                            decision_info['reason']
                        )

                    # Save to file
                    try:
                        # Save directly using the utility function (handles analyzer being None)
                        from modules.utils.feedback import save_feedback

                        # Get output directory from config
                        output_dir = st.session_state.config.get('output_dir', 'output')

                        filepath = save_feedback(existing_feedback, output_dir, table_name)

                        # Update cache
                        update_feedback_in_cache(table_name, existing_feedback)

                        # Regenerate PDF with updated feedback
                        try:
                            final_dir = os.path.join(output_dir, 'final')
                            os.makedirs(final_dir, exist_ok=True)

                            pdf_generator = ValidationPDFGenerator()

                            # Load the validation results to include in the PDF
                            validation_json_path = os.path.join(final_dir, f"{table_name}_summary_validation.json")
                            if os.path.exists(validation_json_path):
                                with open(validation_json_path, 'r') as f:
                                    validation_data = json.load(f)

                                # Update the validation data with the adjusted status from feedback
                                validation_data['status'] = existing_feedback['adjusted_status']
                                validation_data['is_valid'] = (existing_feedback['adjusted_status'] == 'complete')

                                pdf_path = os.path.join(final_dir, f"{table_name}_validation_report.pdf")

                                if pdf_generator.is_available():
                                    pdf_generator.generate_validation_pdf(
                                        validation_data,
                                        table_name,
                                        pdf_path,
                                        st.session_state.config.get('site_name'),
                                        st.session_state.config.get('timezone', 'UTC')
                                    )
                                else:
                                    # Fall back to text report
                                    txt_path = os.path.join(final_dir, f"{table_name}_validation_report.txt")
                                    pdf_generator.generate_text_report(
                                        validation_data,
                                        table_name,
                                        txt_path,
                                        st.session_state.config.get('site_name'),
                                        st.session_state.config.get('timezone', 'UTC')
                                    )
                        except Exception as e:
                            st.warning(f"Could not regenerate PDF report: {e}")

                        st.success(f"✅ Feedback saved successfully!")
                        st.info(f"📊 New status: **{existing_feedback['adjusted_status'].upper()}**")

                        import time
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error saving feedback: {e}")
                        import traceback
                        st.code(traceback.format_exc())

            with col2:
                if st.button("🔄 Reset to Clifpy Results", width='stretch'):
                    # Reset all to pending
                    for error_id in existing_feedback['user_decisions'].keys():
                        existing_feedback = update_user_decision(existing_feedback, error_id, 'pending', '')
                    st.session_state.error_decisions = {}
                    st.success("Reset complete")
                    st.rerun()


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
            elif 'unique_patients' in data_info and data_info.get('unique_patients', 0) > 0:
                st.metric("Unique Patients", f"{data_info.get('unique_patients', 0):,}")
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
        viz_data_path = os.path.join(output_dir, 'final', f'{table_name}_visualization_data.json')

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
        viz_data_path = os.path.join(output_dir, 'final', f'{table_name}_visualization_data.json')

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


if __name__ == "__main__":
    main()