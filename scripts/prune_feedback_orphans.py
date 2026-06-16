"""One-shot cleanup: remove orphaned decisions from saved feedback files.

Background
----------
Each saved feedback decision is keyed by ``error_id``, a hash of the issue's
*message* text (see ``modules/utils/feedback.create_error_id``). When validation
re-runs and a message changes, the previously-saved decision no longer matches
any current issue. It is never rendered in the review UI (which only draws
current issues) yet is still tallied by ``_recompute_counts`` — surfacing as a
phantom, unclearable "pending" atom (e.g. a permanent "+1 pending" on a table
you have fully reviewed).

The server now prunes these automatically whenever feedback is loaded through
``_resolve_feedback`` (review page + save). But the tables-list cards read the
feedback file straight off disk, so they keep showing the stale count until the
file itself is rewritten. This script rewrites every affected file once.

Safety
------
* Dry-run by default: prints what it *would* change and writes nothing.
  Pass ``--apply`` to actually rewrite files.
* A table whose DQA report cannot be loaded is SKIPPED untouched — we can't
  tell orphans from a missing validation result, so we never guess.
* Files with no orphans are left byte-for-byte untouched.
* The original ``timestamp`` is preserved (the card's "Reviewed …" date does
  not move); only ``user_decisions``, the counts, and ``adjusted_status`` change.
* When ``--apply``, a ``<file>.bak`` backup is written before each rewrite.

Usage
-----
    uv run python scripts/prune_feedback_orphans.py            # dry-run
    uv run python scripts/prune_feedback_orphans.py --apply    # rewrite files
"""

import argparse
import glob
import json
import os
import shutil
import sys

# Make the project root importable when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.utils.feedback import (  # noqa: E402
    prune_orphan_decisions,
    recalculate_status,
)
from modules.utils.output_paths import (  # noqa: E402
    validation_feedback_dir,
    validation_json_reports_dir,
)


def _load_error_issues(table_name: str):
    """Current 'error' issues for a table, from its DQA JSON report.

    Returns None when the report is missing/unreadable (caller must skip,
    not prune) and [] only when the report genuinely has no error issues.
    """
    from modules.cli.pdf_generator import _collect_dqa_issues

    dqa_path = validation_json_reports_dir() / f"{table_name}_dqa.json"
    if not dqa_path.exists():
        return None
    try:
        with open(dqa_path, "r", encoding="utf-8") as f:
            validation = json.load(f)
    except (OSError, ValueError):
        return None
    _, all_issues = _collect_dqa_issues(validation)
    return [i for i in all_issues if i.get("severity") == "error"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="Rewrite files. Without this flag the script only reports (dry-run).",
    )
    args = parser.parse_args()

    fb_dir = validation_feedback_dir()
    paths = sorted(glob.glob(str(fb_dir / "*_validation_response.json")))
    if not paths:
        print(f"No feedback files found under {fb_dir}")
        return 0

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] scanning {len(paths)} feedback file(s) in {fb_dir}\n")

    total_pruned = changed = skipped = 0
    for path in paths:
        name = os.path.basename(path).replace("_validation_response.json", "")
        try:
            with open(path, "r", encoding="utf-8") as f:
                feedback = json.load(f)
        except (OSError, ValueError) as e:
            print(f"  SKIP   {name}: unreadable feedback file ({e})")
            skipped += 1
            continue

        issues = _load_error_issues(name)
        if issues is None:
            print(f"  SKIP   {name}: no DQA report to validate against")
            skipped += 1
            continue

        before = dict(feedback.get("user_decisions", {}))
        before_pending = feedback.get("pending_count")
        prune_orphan_decisions(feedback, issues)
        removed = [k for k in before if k not in feedback.get("user_decisions", {})]

        if not removed:
            print(f"  ok     {name}: no orphans")
            continue

        total_pruned += len(removed)
        changed += 1
        print(
            f"  PRUNE  {name}: removed {len(removed)} orphan(s) "
            f"(pending {before_pending} -> {feedback['pending_count']})"
        )
        for k in removed:
            print(f"             - {k}")

        if args.apply:
            # Preserve the original review timestamp; only the decision set and
            # derived fields change. Recompute adjusted_status since dropping a
            # pending atom can flip a table's status.
            feedback["adjusted_status"] = recalculate_status(
                feedback.get("original_status", feedback.get("adjusted_status")),
                feedback,
            )
            # Back up the ORIGINAL file (still untouched on disk) before writing.
            shutil.copy2(path, path + ".bak")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(feedback, f, indent=2, default=str)

    print(
        f"\n[{mode}] {changed} file(s) with orphans, "
        f"{total_pruned} orphan(s) total, {skipped} skipped."
    )
    if not args.apply and changed:
        print("Re-run with --apply to rewrite these files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
