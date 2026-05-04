"""Legacy code/ directory retained for archived notebooks and reports.

The active pipeline now lives under ``modules/`` (e.g. SOFA computation
moved from ``code/sofa_polars.py`` to ``modules/sofa/calculator.py``).
This package no longer re-exports anything; the previous import line
``from .sofa_polars import compute_sofa_polars`` referenced a file that
was moved to ``code/archives/sofa_polars.py`` and broke any code that
imported the ``code`` package (including pytest, which imports stdlib
``code`` via ``pdb`` — Python's import precedence picks up this directory
first when it's on ``sys.path``).
"""

__all__: list[str] = []
