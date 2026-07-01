"""Table-specific analysis modules for CLIF 3.0"""

from .base_table_analyzer import BaseTableAnalyzer
from .patient_analysis import PatientAnalyzer
from .hospitalization_analysis import HospitalizationAnalyzer
from .adt_analysis import ADTAnalyzer
from .code_status_analysis import CodeStatusAnalyzer
from .crrt_therapy_analysis import CRRTTherapyAnalyzer
from .mcs_analysis import MCSAnalyzer
from .hospital_diagnosis_analysis import HospitalDiagnosisAnalyzer
from .labs_analysis import LabsAnalyzer
from .medication_admin_continuous_analysis import MedicationAdminContinuousAnalyzer
from .medication_admin_intermittent_analysis import MedicationAdminIntermittentAnalyzer
from .microbiology_culture_analysis import MicrobiologyCultureAnalyzer
from .microbiology_nonculture_analysis import MicrobiologyNoncultureAnalyzer
from .microbiology_susceptibility_analysis import MicrobiologySusceptibilityAnalyzer
from .patient_assessments_analysis import PatientAssessmentsAnalyzer
from .patient_procedures_analysis import PatientProceduresAnalyzer
from .position_analysis import PositionAnalyzer
from .respiratory_support_analysis import RespiratorySupportAnalyzer
from .vitals_analysis import VitalsAnalyzer
from .generic_analysis import GenericTableAnalyzer, make_generic_analyzer

# New CLIF 3.0 tables that get schema-driven validation via GenericTableAnalyzer
# (they have a clifpy table class + 3.0 schema but no bespoke analytics here).
NEW_3_0_TABLES = [
    'airway', 'clinical_notes_facts', 'clinical_trial', 'drain', 'ed_encounter',
    'input', 'intermittent_dialysis', 'invasive_hemodynamics', 'key_icu_orders',
    'line', 'medication_orders', 'model_registry', 'output', 'patient_attributes',
    'patient_diagnosis', 'place_based_index', 'provider', 'radiology', 'scores',
    'therapy_details', 'transfusion', 'validated_diagnosis',
]

# name -> analyzer class for the generic 3.0 tables
GENERIC_TABLE_ANALYZERS = {name: make_generic_analyzer(name) for name in NEW_3_0_TABLES}

__all__ = [
    'BaseTableAnalyzer',
    'PatientAnalyzer',
    'HospitalizationAnalyzer',
    'ADTAnalyzer',
    'CodeStatusAnalyzer',
    'CRRTTherapyAnalyzer',
    'MCSAnalyzer',
    'HospitalDiagnosisAnalyzer',
    'LabsAnalyzer',
    'MedicationAdminContinuousAnalyzer',
    'MedicationAdminIntermittentAnalyzer',
    'MicrobiologyCultureAnalyzer',
    'MicrobiologyNoncultureAnalyzer',
    'MicrobiologySusceptibilityAnalyzer',
    'PatientAssessmentsAnalyzer',
    'PatientProceduresAnalyzer',
    'PositionAnalyzer',
    'RespiratorySupportAnalyzer',
    'VitalsAnalyzer',
    'GenericTableAnalyzer',
    'make_generic_analyzer',
    'NEW_3_0_TABLES',
    'GENERIC_TABLE_ANALYZERS',
]