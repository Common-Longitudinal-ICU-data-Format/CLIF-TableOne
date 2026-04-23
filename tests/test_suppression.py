"""Unit tests for modules.tableone.suppression.

Run with: .venv/bin/python -m pytest tests/test_suppression.py -v
Or standalone: .venv/bin/python tests/test_suppression.py
"""
import sys
from pathlib import Path

# Allow `python tests/test_suppression.py` without PYTHONPATH gymnastics.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import pandas as pd

from modules.tableone.suppression import (
    MergeRules, SuppressionConfig,
    apply_cell_suppression, apply_merges, suppress_dataframe,
    parse_cell, format_cell, split_variable,
    scan_small_cells, apply_suppression_to_tree,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _race_df() -> pd.DataFrame:
    """Approximate the race block from Emory's table_one_overall.csv."""
    return pd.DataFrame([
        {'Variable': 'N: Encounter blocks',                                      'Overall': '314,828'},
        {'Variable': 'Race: Black or African American',                          'Overall': '93,050 (46.8%)'},
        {'Variable': 'Race: White',                                              'Overall': '78,313 (39.4%)'},
        {'Variable': 'Race: Unknown',                                            'Overall': '10,213 (5.1%)'},
        {'Variable': 'Race: Asian',                                              'Overall': '9,442 (4.8%)'},
        {'Variable': 'Race: Other',                                              'Overall': '6,595 (3.3%)'},
        {'Variable': 'Race: American Indian or Alaska Native',                   'Overall': '438 (0.2%)'},
        {'Variable': 'Race: Missing',                                            'Overall': '366 (0.2%)'},
        {'Variable': 'Race: Native Hawaiian or Other Pacific Islander',          'Overall': '219 (0.1%)'},
        {'Variable': 'Age at admission, median [Q1, Q3]',                        'Overall': '60 [41, 72]'},
    ])


def _rules() -> MergeRules:
    r = MergeRules()
    r.variables = {
        'Race': {
            'Other / Unknown': [
                'American Indian or Alaska Native',
                'Native Hawaiian or Other Pacific Islander',
                'Other',
                'Missing',
            ],
        },
    }
    return r


# ---------------------------------------------------------------------------
# parse_cell / format_cell / split_variable
# ---------------------------------------------------------------------------

def test_parse_cell_with_pct():
    c = parse_cell('93,050 (46.8%)')
    assert c.is_count
    assert c.n == 93050
    assert c.pct == 46.8


def test_parse_cell_bare_count():
    c = parse_cell('314,828')
    assert c.is_count
    assert c.n == 314828
    assert c.pct is None


def test_parse_cell_continuous_passes_through():
    c = parse_cell('60 [41, 72]')
    assert not c.is_count
    assert c.n is None


def test_parse_cell_empty():
    c = parse_cell('')
    assert not c.is_count


def test_format_cell():
    assert format_cell(93050, 46.8) == '93,050 (46.8%)'
    assert format_cell(438) == '438'


def test_split_variable():
    assert split_variable('Race: White') == ('Race', 'White')
    assert split_variable('Admission type: osh') == ('Admission type', 'osh')
    assert split_variable('Age at admission, median [Q1, Q3]') == ('', 'Age at admission, median [Q1, Q3]')


# ---------------------------------------------------------------------------
# apply_merges
# ---------------------------------------------------------------------------

def test_merge_collapses_listed_rows():
    df = _race_df()
    out = apply_merges(df, _rules())
    labels = out['Variable'].tolist()
    # Four source rows collapse into one merged row; Black/White/Unknown/Asian
    # survive untouched; the continuous row and N passthrough
    assert 'Race: Other / Unknown' in labels
    assert 'Race: Other' not in labels
    assert 'Race: Missing' not in labels
    assert 'Race: American Indian or Alaska Native' not in labels
    assert 'Race: Native Hawaiian or Other Pacific Islander' not in labels


def test_merge_sums_counts_and_recomputes_pct():
    df = _race_df()
    out = apply_merges(df, _rules())
    merged_row = out[out['Variable'] == 'Race: Other / Unknown'].iloc[0]
    c = parse_cell(merged_row['Overall'])
    # 6595 + 438 + 366 + 219 = 7618
    assert c.n == 7618
    # Original "Race: Other" reports 3.3% on N=6595; implied denom = 6595/0.033 = 199848.
    # 7618 / 199848 = 3.81% — tolerate float drift
    assert c.pct is not None
    assert 3.5 < c.pct < 4.2


def test_merge_leaves_unrelated_rows_alone():
    df = _race_df()
    out = apply_merges(df, _rules())
    # N row and continuous row should still be present with their original values
    n_row = out[out['Variable'] == 'N: Encounter blocks'].iloc[0]
    assert n_row['Overall'] == '314,828'
    age_row = out[out['Variable'] == 'Age at admission, median [Q1, Q3]'].iloc[0]
    assert age_row['Overall'] == '60 [41, 72]'


# ---------------------------------------------------------------------------
# apply_cell_suppression
# ---------------------------------------------------------------------------

def test_suppression_passes_through_when_no_small_cells():
    df = _race_df()  # lowest count is 219, above threshold 10
    out, log = apply_cell_suppression(df, SuppressionConfig())
    assert log == []
    assert out['Overall'].tolist() == df['Overall'].tolist()


def test_suppression_single_small_cell_triggers_complementary():
    df = pd.DataFrame([
        {'Variable': 'Race: White', 'Overall': '200'},
        {'Variable': 'Race: Black', 'Overall': '180'},
        {'Variable': 'Race: Asian', 'Overall': '80'},   # next-smallest
        {'Variable': 'Race: Other', 'Overall': '5'},    # the small cell
    ])
    out, log = apply_cell_suppression(df, SuppressionConfig())
    # Both 'Race: Other' (5) and the next-smallest 'Race: Asian' (80) should be suppressed
    other_row = out[out['Variable'] == 'Race: Other'].iloc[0]
    asian_row = out[out['Variable'] == 'Race: Asian'].iloc[0]
    assert other_row['Overall'] == '<10'
    assert asian_row['Overall'] == '<10'
    # The large rows stay as-is
    assert out[out['Variable'] == 'Race: White'].iloc[0]['Overall'] == '200'


def test_suppression_two_or_more_small_cells_no_complementary():
    df = pd.DataFrame([
        {'Variable': 'Race: White', 'Overall': '100'},
        {'Variable': 'Race: Black', 'Overall': '9'},
        {'Variable': 'Race: Asian', 'Overall': '3'},
    ])
    out, _log = apply_cell_suppression(df, SuppressionConfig())
    assert out[out['Variable'] == 'Race: White'].iloc[0]['Overall'] == '100'
    assert out[out['Variable'] == 'Race: Black'].iloc[0]['Overall'] == '<10'
    assert out[out['Variable'] == 'Race: Asian'].iloc[0]['Overall'] == '<10'


def test_suppression_group_total_below_threshold():
    df = pd.DataFrame([
        {'Variable': 'Race: White', 'Overall': '5'},
        {'Variable': 'Race: Other', 'Overall': '3'},
    ])
    out, _log = apply_cell_suppression(df, SuppressionConfig())
    assert out[out['Variable'] == 'Race: White'].iloc[0]['Overall'] == '<10'
    assert out[out['Variable'] == 'Race: Other'].iloc[0]['Overall'] == '<10'


def test_suppression_ignores_continuous_and_text_cells():
    df = pd.DataFrame([
        {'Variable': 'Age at admission, median [Q1, Q3]', 'Overall': '60 [41, 72]'},
        {'Variable': 'N: Encounter blocks',               'Overall': '314,828'},
    ])
    out, _log = apply_cell_suppression(df, SuppressionConfig())
    assert out['Overall'].tolist() == df['Overall'].tolist()


def test_suppression_threshold_boundary():
    """A cell exactly at the threshold (N = threshold) isn't itself small,
    but complementary suppression can still pull it in if it's the smallest
    non-suppressed sibling of a genuinely small cell — without that, the
    small cell's count could be back-calculated from the group total."""
    df = pd.DataFrame([
        {'Variable': 'Race: White', 'Overall': '100'},
        {'Variable': 'Race: Asian', 'Overall': '10'},   # at threshold — not small on its own
        {'Variable': 'Race: Other', 'Overall': '9'},    # below threshold — small
    ])
    out, _log = apply_cell_suppression(df, SuppressionConfig())
    # Asian gets complementary-suppressed because it's the smallest remaining
    # sibling once Other is hidden. White stays visible.
    assert out[out['Variable'] == 'Race: White'].iloc[0]['Overall'] == '100'
    assert out[out['Variable'] == 'Race: Asian'].iloc[0]['Overall'] == '<10'
    assert out[out['Variable'] == 'Race: Other'].iloc[0]['Overall'] == '<10'


# ---------------------------------------------------------------------------
# suppress_dataframe end-to-end
# ---------------------------------------------------------------------------

def test_suppress_dataframe_resolves_small_via_merge():
    """A small Race row that's listed in a merge rule shouldn't appear as a
    visible small cell — it gets folded into the merged group."""
    df = pd.DataFrame([
        {'Variable': 'Race: White',                                     'Overall': '900'},
        {'Variable': 'Race: Black',                                     'Overall': '500'},
        {'Variable': 'Race: Other',                                     'Overall': '5'},
        {'Variable': 'Race: American Indian or Alaska Native',          'Overall': '3'},
        {'Variable': 'Race: Native Hawaiian or Other Pacific Islander', 'Overall': '2'},
        {'Variable': 'Race: Missing',                                   'Overall': '1'},
    ])
    out, log = suppress_dataframe(df, _rules())
    # The merged row has N = 5 + 3 + 2 + 1 = 11, above threshold
    merged = out[out['Variable'] == 'Race: Other / Unknown'].iloc[0]
    c = parse_cell(merged['Overall'])
    assert c.n == 11
    # No cells suppressed (the small sources got merged away)
    assert not any(v == '<10' for v in out['Overall'])


def test_idempotence():
    """Running suppression on its own output is a no-op (counts stay the
    same; already-suppressed cells stay suppressed)."""
    df = _race_df()
    once, _ = suppress_dataframe(df, _rules())
    twice, _ = suppress_dataframe(once, _rules())
    assert once.equals(twice)


# ---------------------------------------------------------------------------
# scan_small_cells and apply_suppression_to_tree
# ---------------------------------------------------------------------------

def test_apply_suppression_to_tree(tmp_path):
    intermediate = tmp_path / 'intermediate'
    final = tmp_path / 'final'
    (intermediate / 'overall').mkdir(parents=True)
    df = pd.DataFrame([
        {'Variable': 'Race: White',       'Overall': '1000'},
        {'Variable': 'Race: Asian',       'Overall': '500'},
        # 'Other' is in the merge rules — it'll collapse into 'Other / Unknown'
        # with N=5, which is still small → triggers complementary suppression
        # of the next-smallest sibling (Asian).
        {'Variable': 'Race: Other',       'Overall': '5'},
    ])
    df.to_csv(intermediate / 'overall' / 'table_one_overall.csv', index=False)
    written = apply_suppression_to_tree(intermediate, final, _rules())
    out_path = final / 'overall' / 'table_one_overall.csv'
    assert out_path in written
    assert out_path.exists()
    got = pd.read_csv(out_path, dtype=str, keep_default_na=False)
    labels = got['Variable'].tolist()
    # 'Race: Other' is gone — folded into the merged group
    assert 'Race: Other' not in labels
    assert 'Race: Other / Unknown' in labels
    merged_row = got[got['Variable'] == 'Race: Other / Unknown'].iloc[0]
    assert merged_row['Overall'] == '<10'
    # Complementary suppression on the smallest remaining sibling (Asian)
    assert got[got['Variable'] == 'Race: Asian'].iloc[0]['Overall'] == '<10'
    assert got[got['Variable'] == 'Race: White'].iloc[0]['Overall'] == '1000'


def test_scan_small_cells_reports_after_merge(tmp_path):
    intermediate = tmp_path / 'intermediate'
    (intermediate / 'overall').mkdir(parents=True)
    df = pd.DataFrame([
        {'Variable': 'Race: White',                                     'Overall': '1000'},
        {'Variable': 'Race: Black',                                     'Overall': '500'},
        # Residual small cell not covered by the Race merge rule:
        {'Variable': 'Race: Asian',                                     'Overall': '5'},
    ])
    df.to_csv(intermediate / 'overall' / 'table_one_overall.csv', index=False)
    cells = scan_small_cells(intermediate, _rules(), cohort='ci')
    # 'Race: Asian' isn't in the rule's source list, so it remains visible
    rows = [c for c in cells if c.row == 'Asian']
    assert len(rows) >= 1
    assert rows[0].raw_n == 5


if __name__ == '__main__':
    import subprocess
    sys.exit(subprocess.call([sys.executable, '-m', 'pytest', __file__, '-v']))
