# DQA Session Checklist — 2026-04-21

Every issue raised during this session, with expected post-fix behavior so you can verify on each re-run. Items are grouped by DQA check. Each row links to the commit that resolved it.

## How to use this doc

After `uv run python run_project.py --validate-only`, walk the PDF + webapp for a few tables (hospitalization, adt, labs, microbiology_culture, patient_assessments, respiratory_support, crrt_therapy, microbiology_susceptibility) and check each item. If any drift, git log the affected commit for context.

---

## Cross-cutting / display

| # | Expected behavior | Verify | Fix commit |
|---|---|---|---|
| 1 | Every DQA check reports `atomic_total` / `atomic_passed`. Missing → `ValueError`. | No `"0/0"` or `"N/1"` placeholders on real checks. | `67f1026` feat: atomic-level error/warning counts + K.4 atomic=1 per FK |
| 2 | "Not applicable" DQA paths score `0/0`, not `1/1`. | Patient: 82/92 (not 84/94). P.6 "No category columns found" contributes 0. | `d56bf12` |
| 3 | DQA Summary **Errors** / **Warnings** columns sum `atomic_count` (not row count). | K.3 error "Missing 9 mCIDE values" adds 9 to the Errors column. | `67f1026` + `a0f49c2` (webapp) |
| 4 | PDF + webapp render `atomic_count=0` as `—`, not `0`. | K.4 reverse-direction row shows `—` in Checks column. | `47609f1` + `034a705` |
| 5 | INFO row findings are rule-specific, not "N checks passed". | K.1 INFO says "Required columns below null thresholds (err >X%, warn >Y%)". C.2 says "All required columns present". | `abf2d25` + `c729dee` |
| 6 | Collapsed INFO rows dedupe `column_field` and cap at 200 chars. | Micro-culture P.6 INFO does NOT show "organism_category, organism_category, ..." repeating. | `6dfdf41` |
| 7 | Micro-culture PDF exists (previously failed to render). | `output/final/validation/pdf_reports/microbiology_culture_validation_report.pdf` exists. | `6dfdf41` |
| 8 | P.6 footnote points to `output/final/validation/monthly_trends/` (not `clifpy/monthly_trends/`). | P.6 section in any PDF with temporal findings. | `82aa929` |
| 9 | Synth INFO row lists column names when partial-pass (from err/warn rows). | K.3 partial-pass synth row shows the relevant columns, not `NA`. | `03140cc` |
| 10 | Synth INFO finding says "Remaining X" (not "All X") when there are errors in same check. | K.3 partial-pass INFO reads "Remaining mCIDE values represented". | `4a14e92` |

---

## Per-check expectations

### C.1 table_presence

| # | Expected | Verify |
|---|---|---|
| C1.1 | Absent table scored as `0/N` conformance with **N errors**, one row saying "Table not present in dataset — N conformance atoms could not be evaluated". | Click micro_susc card → Errors=11, Conformance=0/11, Completeness=N/A, Plausibility=N/A. |
| C1.2 | Absent-table per-table PDF exists even if site didn't submit (via `build_absent_table_dqa_result`). | `dev/dqa_session_checklist` Cross-site comparability preserved. |

### C.2 required_columns · C.3 column_dtypes

| # | Expected | Verify |
|---|---|---|
| C23.1 | Collapsed INFO row reads "All required columns present" / "All dtypes match schema" with the Checks column holding the total column count. | Any table's Conformance section. |

### C.4 datetime_format

| # | Expected | Verify |
|---|---|---|
| C4.1 | Collapsed INFO says "All datetime columns timezone-aware". | Any table with datetime columns. |

### C.5 categorical_values

| # | Expected | Verify |
|---|---|---|
| C5.1 | INFO finding "All categorical values conform to mCIDE". | Any silent-pass categorical check. |

### C.6 category_group_mapping

| # | Expected | Verify |
|---|---|---|
| C6.1 | Mismatch finding shows the category name, NOT `?:`. | patient_assessments + mac. Format: "braden_activity: found 'X', expected 'Y'". | `c0b72e0` |
| C6.2 | After normalization (case/whitespace), true matches are NOT flagged as mismatches. | patient_assessments: braden_*, cam_*, sat_*, sbt_* with matching groups should NOT appear as warnings. Only actual diffs (e.g., trailing-space variants) surface. |

### C.7 lab_reference_units

| # | Expected | Verify |
|---|---|---|
| C7.1 | labs shows 52 C.7 atoms (1 per entry in `lab_reference_units`). | labs PDF. |

---

### K.1 missingness

| # | Expected | Verify |
|---|---|---|
| K1.1 | Lenient scoring: only errors reduce `atomic_passed`. A column with warning-level missingness is still a passing atom. | Table with K.1 warnings shows Non-Error column unchanged. | `1398f93` |
| K1.2 | INFO finding surfaces thresholds: "Required columns below null thresholds (err >X%, warn >Y%)". | Any table's K.1 INFO row. | `c729dee` |

### K.2 conditional_requirements

| # | Expected | Verify |
|---|---|---|
| K2.1 | Atomic granularity = 1 per `(rule × then_required column)`, not 1 per rule. | respiratory_support K.2 atomic_total = 34 (not 15). crrt_therapy K.2 = 3. | `9d3a36c` |
| K2.2 | Lenient scoring — warning-level violations don't fail atoms. | Respiratory_support Completeness = 66/68 (only K.1 error + K.3 error subtract; K.2 warnings don't). |

### K.3 mcide_value_coverage

| # | Expected | Verify |
|---|---|---|
| K3.1 | Per-column INFO row: "Column X: {found}/{expected} mCIDE values present" with `atomic_count = found`. | Micro-culture K.3 shows per-column INFO rows with counts 43, 3, 519 (not a single row with checks=565). | `de7ad9e` |
| K3.2 | Collapsed INFO `atomic_count` sums per-row counts (not `len(grp)`). | Any K.3 with multiple column INFOs sums correctly. | `de7ad9e` |

### K.4 relational_integrity

| # | Expected | Verify |
|---|---|---|
| K4.1 | Atomic_total = 1 per FK (forward direction only). Reverse direction has `atomic_count=0`, renders as `—`. | patient_procedures K.4: row 1 cardinality `checks=—`, row 2 FK integrity `checks=1`. | `fb037ad` + `44767dd` |
| K4.2 | Forward orphan severity thresholds: **>10% orphans → ERROR**, **>1% → WARNING**, else silent pass. | patient_procedures K.4 (6.4% coverage → 93.6% orphans) surfaces as ERROR, fails the atom. | `44767dd` |
| K4.3 | Reverse direction always WARNING (never error), atomic_count=0. | "hospitalization_id values in hospitalization not found in patient_procedures" stays WARNING regardless of coverage. | `44767dd` |

### K.5 cross_table_conditional_completeness

| # | Expected | Verify |
|---|---|---|
| K5.1 | Result attaches to `target_table` (not source). | patient has K.5=1 (from hospitalization.discharge_category=Expired → death_dttm rule). |

---

### P.1 chronological_order

| # | Expected | Verify |
|---|---|---|
| P1.1 | Check renamed from `temporal_ordering` to `chronological_order`. | validation_rules.yaml top-level key, API symbols. |

### P.2 numeric_range_plausibility

| # | Expected | Verify |
|---|---|---|
| P2.1 | Lenient scoring: warnings don't reduce `atomic_passed`. | crrt_therapy Plausibility 8/8 (not 5/8) when only warnings present. | `1398f93` |
| P2.2 | Silent-pass columns each emit their own INFO row. | respiratory_support P.2: 16 warnings + 1 per-column INFO for the silent-pass column (not synth-inherited). | `c0b72e0` |
| P2.3 | `int(None)` on all-null columns no longer crashes the check. | crrt_therapy P.2 runs all 5 atoms (not 0/1 via exception fallback). | `6dfdf41` |

### P.3 field_plausibility

| # | Expected | Verify |
|---|---|---|
| P3.1 | Lenient scoring. | Warning-only check still passes atom. |

### P.4 medication_dose_unit_consistency

| # | Expected | Verify |
|---|---|---|
| P4.1 | Only medication tables (mac, mai) have P.4 = 1. | Others show 0. |

### P.5 overlapping_periods

| # | Expected | Verify |
|---|---|---|
| P5.1 | Only adt has P.5 = 1. | Per the doc. |

### P.6 category_temporal_consistency

| # | Expected | Verify |
|---|---|---|
| P6.1 | Atomic granularity = 1 per `(category column × value present in data)`, not per column. | hospitalization P.6 atomic_total = 23 (if all permissible values present) or fewer (if some missing from data). | `e4072f2` |
| P6.2 | Lenient scoring — warnings don't reduce atomic_passed. | Hospitalization P.6 passes all atoms under lenient logic. | `e4072f2` |
| P6.3 | Every value gets its own row (WARNING absent-in-some-years / INFO present-in-all). Per-value sparklines. | Micro-culture's organism_category rows visible per value. |
| P6.4 | P.6 denominator may be **less than** the doc's stated count when data is missing permissible values. Doc reflects "perfect site"; real site sees fewer atoms for values not in data (K.3 flags those separately). | hospitalization P.6: doc says 23 (6+17), data-present 18 means some values absent. Section §8 of `dqa-check-count.md` explicitly flags this. |

### P.7 duplicate_composite_keys

| # | Expected | Verify |
|---|---|---|
| P7.1 | 1 per table with composite_keys in validation_rules.yaml. | code_status, crrt_therapy have P.7 = 0 (no composite key defined). |

### P.8 cross_table_temporal

| # | Expected | Verify |
|---|---|---|
| P8.1 | 1 per `(table × time column)` for tables carrying hospitalization_id, excluding hospitalization itself. | adt=2, labs=3, micro_culture=3, crrt_therapy=1. |

---

## Webapp behaviors

| # | Expected | Verify |
|---|---|---|
| W.1 | Clicking an absent-table card shows 0/N Conformance + N/A Completeness + N/A Plausibility + Errors=N. | micro_susc card. | `e17b647` + `2210ca0` |
| W.2 | Clear-all-feedback regenerates per-table PDFs (not just combined). | Clear feedback → per-table PDF no longer has feedback banner or adjusted counts. | `99bc0f0` |
| W.3 | `ERRORS`/`WARNINGS` cards show atomic sums (not row counts). | Consistent with PDF DQA Summary. | `a0f49c2` |
| W.4 | Reverse-direction K.4 warning row shows `—` in Checks column. | Any table with K.4 reverse-orphans. | `034a705` |
| W.5 | No LLM/AI summary section on home page. | Legacy `/api/llm/*` removed. | `a0f49c2` |

---

## Docs

| # | Expected | Verify |
|---|---|---|
| D.1 | `dev/dqa-check-count.md` pillar totals + per-table totals match the schema-derived reproduce script (§9). | Run §9 script → `GRAND TOTAL 3363`. |
| D.2 | All 4 doc locations per table agree (section header, derivation-table total, §5 matrix row, §5 pillar totals). | Verified 2026-04-21; script in conversation history. |
| D.3 | §8 caveat explains real-site P.6 denominator < "perfect site" count. | `dqa-check-count.md` §8. |

---

## Where settings come from

| setting | file |
|---|---|
| Numeric range (P.2) | `clifpy/schemas/outlier_config.yaml` |
| Conditional rules (K.2) | `clifpy/schemas/validation_rules.yaml → conditional_requirements` |
| Composite keys (P.7) | `clifpy/schemas/validation_rules.yaml → composite_keys` |
| Chronological rules (P.1) | `clifpy/schemas/validation_rules.yaml → chronological_order` |
| Cross-table temporal (P.8) | `_CROSS_TABLE_TIME_COLUMNS` in `validator.py` |
| Cross-table conditional (K.5) | `clifpy/schemas/validation_rules.yaml → cross_table_conditional_requirements` |
| mCIDE permissible values (C.5, K.3, P.6) | Per-table `{table}_schema.yaml → columns[].permissible_values` |
| Category-group mappings (C.6) | Per-table `{table}_schema.yaml → {name}_category_to_group_mapping` |
| Lab reference units (C.7) | `labs_schema.yaml → lab_reference_units` |
| FK relationships (K.4) | `validation_rules.yaml → relational_integrity` |
| Field plausibility rules (P.3) | `validation_rules.yaml → field_plausibility_rules` |
| Overlapping periods (P.5) | `validation_rules.yaml → overlapping_periods` |

---

## If a regression reappears

1. Find the item in this checklist.
2. Check the linked commit — does the current clifpy `main` contain it?
   ```
   cd /Users/dema/WD/clifpy && git log --oneline | grep <commit>
   ```
3. If present but the behavior is wrong, the run likely used stale bytecode or an older uvicorn process. Kill everything and re-run:
   ```
   pkill -f 'uvicorn\|run_project'
   find /Users/dema/WD/{clifpy,CLIF-TableOne} -type d -name __pycache__ -exec rm -rf {} +
   uv run python run_project.py --validate-only
   ```
4. If still wrong, the check may have a new edge case — open a follow-up issue.
