"""Regression test for the DST spring-forward fix at base_table_analyzer.py:146."""
from datetime import datetime
import polars as pl


def test_replace_time_zone_handles_spring_forward():
    """Without non_existent='null', this would raise."""
    df = pl.DataFrame({
        "ts": [
            datetime(2018, 3, 11, 1, 30),
            datetime(2018, 3, 11, 2, 0),   # the gap
            datetime(2018, 3, 11, 3, 0),
        ]
    })

    # Should not raise; gap row should become NULL
    out = df.with_columns(
        pl.col("ts").dt.replace_time_zone(
            "US/Central",
            non_existent="null",
            ambiguous="earliest",
        )
    )

    # The gap row is null; the others survive
    assert out["ts"][1] is None
    assert out["ts"][0] is not None
    assert out["ts"][2] is not None
