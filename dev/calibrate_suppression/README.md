# Cross-site small-cell calibration (dev-only)

This tool helps the consortium maintainer pick the canonical merge rules in
`config/tableone_merge_rules.yaml`. The YAML is fixed in code — sites don't
edit it — so the choice of which rows to collapse needs to be informed by
what's actually small across multiple sites, not just one.

## Workflow

1. Each participating site runs the Table One pipeline on their data and
   shares their `output/intermediate/tableone/` directory (e.g. zipped and
   emailed / uploaded).
2. Drop each site's copy into
   `dev/calibrate_suppression/sites/<site_name>/`. The expected layout is
   the standard Table One intermediate tree:

   ```
   dev/calibrate_suppression/sites/
     emory/
       overall/table_one_overall.csv
       overall_ward/table_one_overall.csv
       strata/icu/...
     rush/
       overall/...
     ...
   ```

3. From the repo root run:

   ```
   .venv/bin/python dev/calibrate_suppression/calibrate.py
   ```

4. Read `dev/calibrate_suppression/out/cross_site_review.csv` — one row per
   `(csv_file, variable, row, data_column)` with a column per site showing
   the count (or `<10`). Sort by "# sites small" to find the rows most
   chronically below threshold.
5. The same run prints `candidate_merges.md` — grouped by variable — with
   clinical groupings suggested for the canonical YAML.
6. Edit `config/tableone_merge_rules.yaml`, commit, release.

## What "small" means here

The calibration tool applies the same threshold as production suppression
(default 10, read from the committed YAML). A cell with N in `[1, 9]` is
"small". Cells of exactly `0` are considered non-existent categories at that
site and are excluded from the count of "# sites small" — they wouldn't
benefit from being folded into a merge group.

## Notes

- **Clinical judgement still required.** The tool suggests candidates, not
  final rules. A row being small at every site doesn't automatically mean it
  should be merged — maybe it's a rare-but-important category that deserves
  its own visibility. The maintainer decides.
- **Add sites incrementally.** Re-running the script with more sites
  refines the picture. Earlier rules don't need to change; new candidates
  surface as sites are added.
- **Nothing in this folder is shipped.** The `sites/` directory is
  `.gitignore`d — patient-derived counts stay local.
