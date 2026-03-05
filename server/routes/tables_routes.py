"""Table listing routes."""

from fastapi import APIRouter
from server.services import cache_service

router = APIRouter(prefix="/api", tags=["tables"])

TABLE_DISPLAY_NAMES = {
    'patient': 'Patient',
    'hospitalization': 'Hospitalization',
    'adt': 'ADT',
    'code_status': 'Code Status',
    'crrt_therapy': 'CRRT Therapy',
    'ecmo_mcs': 'ECMO/MCS',
    'hospital_diagnosis': 'Hospital Diagnosis',
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
}

ALL_TABLES = list(TABLE_DISPLAY_NAMES.keys())


@router.get("/tables")
async def get_tables():
    """Return all 18 tables with status/timestamp info."""
    cache_service.init()
    tables = {}
    for name in ALL_TABLES:
        cached = cache_service.get(name)
        tables[name] = {
            "display_name": TABLE_DISPLAY_NAMES[name],
            "status": cache_service.status(name) if cached else "not_analyzed",
            "timestamp": cached["timestamp"] if cached else None,
            "validation_complete": cached.get("validation_complete", False) if cached else False,
            "summary_complete": cached.get("summary_complete", False) if cached else False,
        }
    return {"tables": tables}
