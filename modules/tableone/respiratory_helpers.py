"""Respiratory-support waterfall helpers.

Extracted from `generator.py` as a pure refactor.
"""

from __future__ import annotations


__all__ = ["_waterfall_chunk"]


def _waterfall_chunk(args):
    """Process a chunk of encounters through the waterfall (top-level for pickling).

    The function lives at module scope (not inside another function) so it
    can be pickled when used with multiprocessing pools.
    """
    chunk_df, id_col = args
    from clifpy.utils.waterfall import process_resp_support_waterfall
    return process_resp_support_waterfall(chunk_df, id_col=id_col, verbose=False)
