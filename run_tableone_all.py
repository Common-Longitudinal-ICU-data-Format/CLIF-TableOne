#!/usr/bin/env python3
"""
Run both critical-illness and ward Table One pipelines sequentially.

Each pipeline runs in an isolated subprocess so peak memory equals the larger
of the two cohorts, not the sum. This is important for 16GB systems where
running both cohorts in a single Python process would exceed available RAM.

Usage:
    uv run run_tableone_all.py
"""

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent
PIPELINES = [
    ('Critical illness', 'run_tableone.py'),
    ('Ward',             'run_tableone_ward.py'),
]


def main():
    """Run each pipeline in turn as an isolated subprocess."""
    failures = []
    for label, script in PIPELINES:
        print(f"\n{'='*80}")
        print(f"  Running: {label}")
        print(f"{'='*80}")
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / script)],
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            failures.append(label)
            print(f"\n❌ {label} pipeline FAILED (exit code {result.returncode})")
        else:
            print(f"\n✅ {label} pipeline complete")

    print(f"\n{'='*80}")
    if failures:
        print(f"❌ Failed pipelines: {', '.join(failures)}")
        print(f"{'='*80}\n")
        sys.exit(1)
    print("✅ All Table One pipelines complete")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
