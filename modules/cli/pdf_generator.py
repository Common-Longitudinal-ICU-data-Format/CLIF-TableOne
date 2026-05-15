"""PDF report generator — delegates to clifpy.utils.report_generator."""

import duckdb

from clifpy.utils.report_generator import (
    DQA_CATEGORIES,
    collect_dqa_issues,
    generate_validation_pdf,
    generate_text_report,
)
from clifpy.utils.report_generator import compute_table_stats as _clifpy_compute_table_stats

# Backward-compat alias (used by combined_report_generator and app.py)
_collect_dqa_issues = collect_dqa_issues


_DTYPE_NORMALIZATION = {
    'DOUBLE': 'FLOAT',
    'REAL': 'FLOAT',
    'DECIMAL': 'FLOAT',
    'BIGINT': 'INTEGER',
    'SMALLINT': 'INTEGER',
    'TINYINT': 'INTEGER',
    'INT': 'INTEGER',
    'TIMESTAMP': 'DATETIME',
    'TIMESTAMP WITH TIME ZONE': 'DATETIME',
    'TIMESTAMP_NS': 'DATETIME',
    'TIMESTAMP_MS': 'DATETIME',
    'TIMESTAMP_S': 'DATETIME',
    'TIMESTAMP_US': 'DATETIME',
    'TEXT': 'VARCHAR',
    'STRING': 'VARCHAR',
}


def _normalize_dtype(duckdb_type: str) -> str:
    return _DTYPE_NORMALIZATION.get(duckdb_type, duckdb_type)


def compute_table_stats(df, schema):
    """Wrap clifpy.compute_table_stats but report the *actual* source dtype.

    clifpy returns the schema-declared dtype, which hides type drift (e.g., a
    FLOAT column actually loaded as VARCHAR). We replace the dtype with the
    DuckDB-inferred type — normalized to CLIF spec names so flavor variants
    (DOUBLE/REAL → FLOAT, BIGINT/SMALLINT → INTEGER, TIMESTAMP → DATETIME) are
    displayed consistently.
    """
    stats = _clifpy_compute_table_stats(df, schema)
    if not stats or df is None:
        return stats

    try:
        con = duckdb.connect(':memory:')
        con.register('df', df)
        actual_types = {row[0]: row[1].upper() for row in con.execute("DESCRIBE df").fetchall()}
    except Exception:
        return stats

    for s in stats:
        actual = actual_types.get(s['column'])
        if actual:
            s['dtype'] = _normalize_dtype(actual)
    return stats


class ValidationPDFGenerator:
    """Thin wrapper around clifpy report functions for class-based callers."""

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def generate_validation_pdf(validation_data, table_name, output_path,
                                site_name=None, timezone=None, feedback=None):
        return generate_validation_pdf(validation_data, table_name, output_path, site_name, feedback)

    @staticmethod
    def generate_text_report(validation_data, table_name, output_path,
                             site_name=None, timezone=None, feedback=None):
        return generate_text_report(validation_data, table_name, output_path, site_name)
