"""
Table One Generation Module for CLIF 2.1 Data.

`main` is exposed lazily via PEP-562 __getattr__ so importing lighter siblings
of this package (e.g. ``modules.tableone.suppression``,
``modules.tableone.cohort_filter``) doesn't pull in the full generator + clifpy
dependency tree. This matters for unit tests that need to exercise the small
modules without a full CLIF env. ``from modules.tableone import main`` still
works — it just doesn't happen at package-init time.
"""

__all__ = ['main']


def __getattr__(name):
    if name == 'main':
        from .generator import main
        return main
    raise AttributeError(f"module 'modules.tableone' has no attribute {name!r}")