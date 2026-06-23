"""Analysis execution routes (single table and bulk)."""

import os
import uuid
import threading

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from server import session
from server.services import cache_service
from modules.tables import (
    PatientAnalyzer, HospitalizationAnalyzer, ADTAnalyzer, CodeStatusAnalyzer,
    CRRTTherapyAnalyzer, HospitalDiagnosisAnalyzer, LabsAnalyzer,
    MedicationAdminContinuousAnalyzer, MedicationAdminIntermittentAnalyzer,
    MicrobiologyCultureAnalyzer, 
    MicrobiologySusceptibilityAnalyzer, PatientAssessmentsAnalyzer,
    PatientProceduresAnalyzer, PositionAnalyzer, RespiratorySupportAnalyzer,
    VitalsAnalyzer,
)

router = APIRouter(prefix="/api", tags=["analysis"])

TABLE_ANALYZERS = {
    'patient': PatientAnalyzer,
    'hospitalization': HospitalizationAnalyzer,
    'adt': ADTAnalyzer,
    'code_status': CodeStatusAnalyzer,
    'crrt_therapy': CRRTTherapyAnalyzer,
    'hospital_diagnosis': HospitalDiagnosisAnalyzer,
    'labs': LabsAnalyzer,
    'medication_admin_continuous': MedicationAdminContinuousAnalyzer,
    'medication_admin_intermittent': MedicationAdminIntermittentAnalyzer,
    'microbiology_culture': MicrobiologyCultureAnalyzer,
    'microbiology_susceptibility': MicrobiologySusceptibilityAnalyzer,
    'patient_assessments': PatientAssessmentsAnalyzer,
    'patient_procedures': PatientProceduresAnalyzer,
    'position': PositionAnalyzer,
    'respiratory_support': RespiratorySupportAnalyzer,
    'vitals': VitalsAnalyzer,
}

ALL_TABLES = list(TABLE_ANALYZERS.keys())


class AnalyzeRequest(BaseModel):
    generate_aggregates: bool = False


class AnalyzeAllRequest(BaseModel):
    generate_aggregates: bool = False
    tables: list[str] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_analyzer(table_name: str, config: dict):
    """Instantiate an analyzer."""
    analyzer_class = TABLE_ANALYZERS[table_name]
    data_dir = config.get('tables_path', './data')
    filetype = config.get('filetype') or config.get('file_type', 'parquet')
    timezone = config.get('timezone', 'UTC')
    output_dir = config.get('output_dir', 'output')
    clif_version = config.get('clif_version', '3.0')
    return analyzer_class(data_dir, filetype, timezone, output_dir,
                          clif_version=clif_version)


def _inject_table_stats(validation_results, analyzer):
    """Add total_rows and table_stats to validation results."""
    if (validation_results
            and analyzer.table is not None
            and hasattr(analyzer.table, 'df')
            and analyzer.table.df is not None):
        from modules.cli.pdf_generator import compute_table_stats
        validation_results['total_rows'] = len(analyzer.table.df)
        validation_results['table_stats'] = compute_table_stats(
            analyzer.table.df, analyzer.table.schema,
        )


def _extract_cross_table_cache(analyzer):
    """Extract cross-table cache and return (cache_dict, hosp_years_or_None)."""
    try:
        cache = analyzer.extract_cross_table_cache()
        return cache, cache.get('hosp_years')
    except Exception:
        return {}, None


def _save_and_generate_pdf(analyzer, table_name, validation_results, config):
    """Save validation JSON and generate per-table PDF."""
    if validation_results:
        try:
            analyzer.save_validation_results(validation_results)
        except Exception:
            pass
        try:
            from modules.cli import ValidationPDFGenerator
            from modules.utils.output_paths import PDF_REPORTS
            PDF_REPORTS.mkdir(parents=True, exist_ok=True)
            pdf_gen = ValidationPDFGenerator()
            pdf_path = str(PDF_REPORTS / f"{table_name}_validation_report.pdf")
            if pdf_gen.is_available():
                pdf_gen.generate_validation_pdf(
                    validation_results, table_name, pdf_path,
                    config.get('site_name'), config.get('timezone', 'UTC'),
                )
        except Exception:
            pass


def _update_task(task_id: str, **fields):
    """Convenience to update a task entry in session."""
    tasks = session.get("tasks") or {}
    task = tasks.get(task_id, {})
    task.update(fields)
    tasks[task_id] = task
    session.set("tasks", tasks)


# ---------------------------------------------------------------------------
# Background workers
# ---------------------------------------------------------------------------

def _run_single_analysis(task_id: str, table_name: str, config: dict,
                         generate_aggregates: bool):
    """Run analysis for a single table in a background thread."""
    try:
        _update_task(task_id, status="progress", table=table_name,
                     message=f"Loading {table_name}...", pct=0)

        analyzer = _build_analyzer(table_name, config)

        _update_task(task_id, message=f"Validating {table_name}...", pct=30)

        hosp_years = session.get("hosp_years")
        validation_results = analyzer.validate(hosp_years=hosp_years)
        _inject_table_stats(validation_results, analyzer)

        # Cross-table cache
        if validation_results and analyzer.table is not None:
            _, hy = _extract_cross_table_cache(analyzer)
            if hy:
                session.set("hosp_years", hy)

        _save_and_generate_pdf(analyzer, table_name, validation_results, config)

        _update_task(task_id, message=f"Summary statistics for {table_name}...", pct=70)

        summary_stats = analyzer.get_summary_statistics() if generate_aggregates else None
        if summary_stats:
            try:
                analyzer.save_summary_data(summary_stats, '_summary')
            except Exception:
                pass

        # Cache results
        cache_service.init()
        cache_service.cache(table_name, analyzer, validation_results, summary_stats)

        _update_task(task_id, status="complete", table=table_name,
                     message=f"{table_name} complete", pct=100)

    except Exception as e:
        _update_task(task_id, status="error", table=table_name, message=str(e))


def _run_bulk_analysis(task_id: str, config: dict,
                       generate_aggregates: bool,
                       tables: list[str] | None = None):
    """Run analysis for selected (or all) tables in a background thread."""
    tables_to_run = tables if tables else ALL_TABLES
    _update_task(task_id, status="progress", message="Starting bulk analysis...",
                 pct=0, results={"success": [], "failed": [], "skipped": []})

    cross_table_caches = {}
    hosp_years = session.get("hosp_years")

    for idx, table_name in enumerate(tables_to_run):
        pct = int((idx / len(tables_to_run)) * 100)
        _update_task(task_id, message=f"Analyzing {table_name}... ({idx + 1}/{len(tables_to_run)})", pct=pct)

        try:
            analyzer = _build_analyzer(table_name, config)

            if (analyzer.table is None
                    or not hasattr(analyzer.table, 'df')
                    or analyzer.table.df is None):
                tasks = session.get("tasks") or {}
                tasks[task_id]["results"]["skipped"].append(table_name)
                session.set("tasks", tasks)
                continue

            # Validate
            validation_results = analyzer.validate(hosp_years=hosp_years)
            _inject_table_stats(validation_results, analyzer)

            # Cross-table cache
            if validation_results and analyzer.table is not None:
                cache_data, hy = _extract_cross_table_cache(analyzer)
                cross_table_caches[table_name] = cache_data
                if hy:
                    hosp_years = hy
                    session.set("hosp_years", hosp_years)

            _save_and_generate_pdf(analyzer, table_name, validation_results, config)

            # Summary
            summary_stats = None
            if generate_aggregates:
                summary_stats = analyzer.get_summary_statistics()
                if summary_stats:
                    try:
                        analyzer.save_summary_data(summary_stats, '_summary')
                    except Exception:
                        pass

            cache_service.init()
            cache_service.cache(table_name, analyzer, validation_results, summary_stats)

            tasks = session.get("tasks") or {}
            tasks[task_id]["results"]["success"].append(table_name)
            session.set("tasks", tasks)

        except Exception as e:
            tasks = session.get("tasks") or {}
            tasks[task_id]["results"]["failed"].append({"table": table_name, "error": str(e)})
            session.set("tasks", tasks)

    # ---- Cross-table checks (best effort) ----
    if len(cross_table_caches) > 1:
        try:
            import json as _json
            from clifpy.utils.validator import (
                run_relational_integrity_checks_from_cache,
                run_cross_table_completeness_checks_from_cache,
                run_cross_table_plausibility_checks_from_cache,
            )
            store = session.get_store()

            rel_results = run_relational_integrity_checks_from_cache(cross_table_caches)
            for tname, rel_checks in rel_results.items():
                if not rel_checks:
                    continue
                serialized_rel = {k: v.to_dict() for k, v in rel_checks.items()}
                if tname in store.get('analyzed_tables', {}):
                    vr = store['analyzed_tables'][tname].get('validation')
                    if vr:
                        vr.setdefault('completeness', {}).update(serialized_rel)
                from modules.utils.output_paths import validation_json_reports_dir as _dpt
                json_path = str(_dpt() / f'{tname}_dqa.json')
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        saved = _json.load(f)
                    saved.setdefault('completeness', {}).update(serialized_rel)
                    with open(json_path, 'w', encoding='utf-8') as f:
                        _json.dump(saved, f, indent=2, default=str)

            cond_results = run_cross_table_completeness_checks_from_cache(cross_table_caches)
            for tname, cond_checks in cond_results.items():
                if not cond_checks:
                    continue
                serialized_cond = {k: v.to_dict() for k, v in cond_checks.items()}
                if tname in store.get('analyzed_tables', {}):
                    vr = store['analyzed_tables'][tname].get('validation')
                    if vr:
                        vr.setdefault('completeness', {}).update(serialized_cond)
                from modules.utils.output_paths import validation_json_reports_dir as _dpt
                json_path = str(_dpt() / f'{tname}_dqa.json')
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        saved = _json.load(f)
                    saved.setdefault('completeness', {}).update(serialized_cond)
                    with open(json_path, 'w', encoding='utf-8') as f:
                        _json.dump(saved, f, indent=2, default=str)

            plaus_results = run_cross_table_plausibility_checks_from_cache(cross_table_caches)
            for tname, plaus_checks in plaus_results.items():
                if not plaus_checks:
                    continue
                serialized_plaus = {k: v.to_dict() for k, v in plaus_checks.items()}
                if tname in store.get('analyzed_tables', {}):
                    vr = store['analyzed_tables'][tname].get('validation')
                    if vr:
                        vr.setdefault('plausibility', {}).update(serialized_plaus)
                from modules.utils.output_paths import validation_json_reports_dir as _dpt
                json_path = str(_dpt() / f'{tname}_dqa.json')
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        saved = _json.load(f)
                    saved.setdefault('plausibility', {}).update(serialized_plaus)
                    with open(json_path, 'w', encoding='utf-8') as f:
                        _json.dump(saved, f, indent=2, default=str)
        except Exception:
            pass

    # ---- Combined report ----
    try:
        from modules.reports.combined_report_generator import generate_combined_report
        output_dir = config.get('output_dir', 'output')
        generate_combined_report(
            output_dir, tables_to_run,
            config.get('site_name'), config.get('timezone', 'UTC'),
        )
    except Exception:
        pass

    # ---- Collect MCIDE ----
    tasks = session.get("tasks") or {}
    success_tables = tasks.get(task_id, {}).get("results", {}).get("success", [])
    try:
        from modules.mcide import MCIDEStatsCollector
        collector = MCIDEStatsCollector(config)
        method_map = {
            'patient': collector.collect_patient,
            'hospitalization': collector.collect_hospitalization,
            'adt': collector.collect_adt,
            'labs': collector.collect_labs_stats,
            'vitals': collector.collect_vitals_stats,
            'medication_admin_continuous': lambda: collector.collect_medication_stats('continuous'),
            'medication_admin_intermittent': lambda: collector.collect_medication_stats('intermittent'),
            'respiratory_support': collector.collect_respiratory_support,
            'microbiology_culture': collector.collect_microbiology_culture,
            'microbiology_nonculture': collector.collect_microbiology_nonculture,
            'microbiology_susceptibility': collector.collect_microbiology_susceptibility,
            'patient_assessments': collector.collect_patient_assessments,
            'patient_procedures': collector.collect_patient_procedures,
            'position': collector.collect_position,
            'crrt_therapy': collector.collect_crrt_stats,
            'ecmo_mcs': collector.collect_ecmo_stats,
            'hospital_diagnosis': collector.collect_hospital_diagnosis,
            'code_status': collector.collect_code_status,
        }
        for tbl in success_tables:
            try:
                if tbl in method_map:
                    method_map[tbl]()
            except Exception:
                pass
    except Exception:
        pass

    # Finalize
    tasks = session.get("tasks") or {}
    final_results = tasks.get(task_id, {}).get("results", {"success": [], "failed": [], "skipped": []})
    _update_task(task_id, status="complete", message="Bulk analysis complete",
                 pct=100, results=final_results)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/analyze/{name}")
async def analyze_table(name: str, body: AnalyzeRequest = AnalyzeRequest()):
    """Start analysis for a single table. Returns a task_id for SSE tracking."""
    if name not in TABLE_ANALYZERS:
        raise HTTPException(404, f"Unknown table: {name}")

    config = session.get("config")
    if not config:
        raise HTTPException(400, "No config loaded")

    task_id = str(uuid.uuid4())
    _update_task(task_id, status="starting", table=name)

    thread = threading.Thread(
        target=_run_single_analysis,
        args=(task_id, name, config, body.generate_aggregates),
        daemon=True,
    )
    thread.start()

    return {"task_id": task_id}


@router.post("/analyze-all")
async def analyze_all(body: AnalyzeAllRequest = AnalyzeAllRequest()):
    """Start bulk analysis for all tables. Returns a task_id for SSE tracking."""
    config = session.get("config")
    if not config:
        raise HTTPException(400, "No config loaded")

    if body.tables is not None:
        unknown = [t for t in body.tables if t not in TABLE_ANALYZERS]
        if unknown:
            raise HTTPException(400, f"Unknown tables: {unknown}")

    task_id = str(uuid.uuid4())
    _update_task(task_id, status="starting")

    thread = threading.Thread(
        target=_run_bulk_analysis,
        args=(task_id, config, body.generate_aggregates, body.tables),
        daemon=True,
    )
    thread.start()

    return {"task_id": task_id}
